"""ComfyUI node for local, mask-aware repair-prompt generation.

The node supports a local Florence caption path and a direct edit instruction
path with optional local Russian translation. Both load their models lazily.
Text results are cached without retaining models or image tensors.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import gc
import hashlib
import importlib
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from .auto_prompt_core import (
    DEFAULT_CONTEXT_PADDING,
    DEFAULT_REPAIR_PROMPT,
    compose_edit_instruction_prompt,
    compose_repair_prompt,
    normalize_edit_instruction,
    prepare_caption_inputs,
    sanitize_caption,
)


MODEL_NAME = "Florence-2-base-PromptGen-v2.0"
TRANSLATION_MODEL_NAME = "opus-mt-ru-en"
_CYRILLIC = re.compile(r"[\u0400-\u052f]")
_QUOTED_LITERAL = re.compile(r'(«[^»]+»|“[^”]+”|"[^"]+")')
_RUSSIAN_BOW_TIE = re.compile(r"\bгалстук[-‑– ]бабочк\w*\b", re.IGNORECASE)
_RUSSIAN_TIE_REPLACEMENT = re.compile(
    r"\bгалстук\s+на\s+(?P<modifiers>(?:[\w-]+\s+){0,4})бабочк\w*\b", re.IGNORECASE
)
_RUSSIAN_INSECT = re.compile(r"\b(?:жив\w*|лета\w*|настоящ\w*)\b", re.IGNORECASE)
_TRANSLATED_TIE_REPLACEMENT = re.compile(
    r"(\b(?:neck\s*)?tie\s+(?:to|with|into)\s+(?:[\w-]+\s+){0,5})butterfly\b", re.IGNORECASE
)
MODEL_PRECISION = "fp16"
MODEL_ATTENTION = "sdpa"
# PromptGen v2's descriptive task is preferable to its stylised tag/mixed
# modes for masked repair: the result is factual context, not a replacement
# generation prompt.
CAPTION_TASK = "caption"
CAPTION_MAX_NEW_TOKENS = 512
CAPTION_NUM_BEAMS = 3
CAPTION_SEED = 1
MAX_CACHE_ENTRIES = 8


class CaptionBackend(Protocol):
    """Small seam used by the node and by CPU-only tests."""

    def caption(self, clips: Iterable[np.ndarray]) -> list[str]:
        ...

    def close(self) -> None:
        ...


_caption_cache: "OrderedDict[tuple[Any, ...], tuple[tuple[str, ...], str]]" = OrderedDict()
_caption_cache_lock = threading.RLock()
# Tests and downstream local integrations may replace this with a constructor
# returning a CaptionBackend.  The default remains the real Florence backend.
_caption_backend_factory: Callable[[], CaptionBackend] | None = None


def clear_caption_cache() -> None:
    """Drop cached caption strings without touching other Comfy models."""

    with _caption_cache_lock:
        _caption_cache.clear()


def _as_numpy(value: Any, *, label: str) -> np.ndarray:
    """Convert a Comfy tensor/array without importing torch at module import."""

    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu").numpy()
    array = np.asarray(value)
    if label == "images":
        if array.ndim != 4:
            raise ValueError("images must have shape T x H x W x C")
        array = np.nan_to_num(array.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] > 3:
            array = array[..., :3]
        return np.clip(array, 0.0, 1.0)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim not in (2, 3):
        raise ValueError("mask must have shape T x H x W or H x W")
    return np.nan_to_num(array.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)


def _sample_digest(value: np.ndarray) -> str:
    """Bound the cache key cost while retaining source/mask/session identity."""

    array = np.asarray(value)
    if array.ndim < 2:
        sample = array
    else:
        temporal = np.linspace(0, max(0, array.shape[0] - 1), min(4, array.shape[0]), dtype=np.int64) if array.ndim >= 3 else np.array([0])
        sample = array[temporal] if array.ndim >= 3 else array
        if sample.ndim == 4:  # T x H x W x C
            y_step = max(1, sample.shape[1] // 32)
            x_step = max(1, sample.shape[2] // 32)
            sample = sample[:, ::y_step, ::x_step, :3]
        elif sample.ndim == 3:  # T x H x W
            y_step = max(1, sample.shape[1] // 32)
            x_step = max(1, sample.shape[2] // 32)
            sample = sample[:, ::y_step, ::x_step]
    if np.issubdtype(np.asarray(sample).dtype, np.floating):
        sample = np.asarray(np.clip(sample, 0.0, 1.0) * 255.0, dtype=np.uint8)
    else:
        sample = np.asarray(sample)
    digest = hashlib.sha1()
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.ascontiguousarray(sample).tobytes())
    return digest.hexdigest()[:20]


def _cache_key(images: np.ndarray, mask: np.ndarray, max_frames: int, context_padding: float, context=None) -> tuple[Any, ...]:
    if context is not None:
        # Hash the actual bounded crops, including frames missed by a sparse
        # full-clip sample. Mask bits distinguish selections in the same crop.
        visual_key = tuple((clip.shape, hashlib.sha1(np.ascontiguousarray(clip).tobytes()).hexdigest())
                           for clip in context["clips"])
        mask_frames = mask[None] if mask.ndim == 2 else mask
        mask_key = tuple((int(meta["frame"]), hashlib.sha1(np.packbits(
            mask_frames[min(int(meta["frame"]), len(mask_frames) - 1)] > 0.01).tobytes()).hexdigest())
                         for meta in context["cropmeta"])
    else:
        visual_key, mask_key = _sample_digest(images), _sample_digest(mask)
    return (
        MODEL_NAME,
        MODEL_PRECISION,
        MODEL_ATTENTION,
        CAPTION_TASK,
        int(max_frames),
        round(float(context_padding), 4),
        visual_key,
        mask_key,
    )


def _cache_get(key: tuple[Any, ...]) -> tuple[list[str], str] | None:
    with _caption_cache_lock:
        value = _caption_cache.get(key)
        if value is None:
            return None
        _caption_cache.move_to_end(key)
        captions, diagnostic = value
        return list(captions), str(diagnostic)


def _cache_put(key: tuple[Any, ...], captions: Iterable[str], diagnostic: str) -> None:
    clean = tuple(str(item) for item in captions if str(item).strip())
    with _caption_cache_lock:
        _caption_cache[key] = (clean, str(diagnostic or ""))
        _caption_cache.move_to_end(key)
        while len(_caption_cache) > MAX_CACHE_ENTRIES:
            _caption_cache.popitem(last=False)


def _model_roots() -> list[Path]:
    """Resolve configured LLM folders without creating or downloading one.

    ComfyUI's ``folder_paths`` registry is authoritative inside Comfy. A
    standalone caller may opt in with ``THS_UNIVERSAL_INPAINT_LLM_PATH`` or
    ``COMFYUI_LLM_PATH``; no workstation-specific fallback is shipped.
    """

    roots: list[Path] = []
    for env_name in ("THS_UNIVERSAL_INPAINT_LLM_PATH", "COMFYUI_LLM_PATH"):
        raw = os.environ.get(env_name)
        if raw:
            roots.append(Path(raw).expanduser())
    try:
        folder_paths = importlib.import_module("folder_paths")
        get_paths = getattr(folder_paths, "get_folder_paths", None)
        if callable(get_paths):
            try:
                roots.extend(Path(path) for path in get_paths("LLM") if path)
            except KeyError:
                pass  # A fresh ComfyUI may not register LLM model folders.
        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            roots.append(Path(models_dir) / "LLM")
    except Exception:
        pass
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = os.path.normcase(os.path.abspath(str(root)))
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _local_model_path() -> Path | None:
    required = ("config.json", "model.safetensors", "tokenizer.json")
    for root in _model_roots():
        candidate = root / MODEL_NAME
        if candidate.is_dir() and all((candidate / name).is_file() for name in required):
            return candidate
    return None


def _local_translation_path() -> Path:
    required = ("config.json", "pytorch_model.bin", "source.spm", "target.spm", "vocab.json")
    for root in _model_roots():
        candidate = root / TRANSLATION_MODEL_NAME
        if all((candidate / name).is_file() for name in required):
            return candidate
    raise RuntimeError(
        "Для перевода с русского установите локальную модель opus-mt-ru-en с профилем instruction. "
        "См. docs/INSTALL_RU.md."
    )


@lru_cache(maxsize=32)
def _translate_locally(path: Path, text: str) -> str:
    """Translate unquoted fragments on CPU and keep literal signs intact."""
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    parts = _QUOTED_LITERAL.split(text)
    indices = [index for index in range(0, len(parts), 2) if _CYRILLIC.search(parts[index])]
    if not indices:
        return text
    tokenizer = MarianTokenizer.from_pretrained(path, local_files_only=True)
    model = MarianMTModel.from_pretrained(path, local_files_only=True, weights_only=True).eval()
    inputs = tokenizer([parts[index].strip() for index in indices], padding=True, return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(**inputs, num_beams=4, max_new_tokens=192)
    for index, translation in zip(indices, tokenizer.batch_decode(output, skip_special_tokens=True), strict=True):
        translation = translation.strip()
        if not translation or _CYRILLIC.search(translation):
            raise RuntimeError("OPUS-MT не смог перевести задачу. Уточните формулировку.")
        original = parts[index]
        if index + 1 < len(parts) and not original.rstrip().endswith((".", "!", "?")):
            translation = translation.rstrip(". ")
        # Text-to-video instructions describe the masked subject. Marian may
        # insert a second-person possessive after an imperative in Russian.
        translation = re.sub(r"(?i)\b(make|recolor|dye|paint)(\s+)(?:your|my)\b", r"\1\2the", translation)
        if _RUSSIAN_BOW_TIE.search(original):
            translation = re.sub(r"(?i)\bbutterfly[\s-]+tie\b", "bow tie", translation)
        replacement = _RUSSIAN_TIE_REPLACEMENT.search(original)
        if replacement and not _RUSSIAN_INSECT.search(replacement["modifiers"]):
            translation = _TRANSLATED_TIE_REPLACEMENT.sub(r"\1bow tie", translation, count=1)
        parts[index] = original[: len(original) - len(original.lstrip())] + translation + original[len(original.rstrip()):]
    return "".join(parts)


def _instruction_prompt(request: Any) -> str:
    text = normalize_edit_instruction(request)
    if any(_CYRILLIC.search(part) for part in _QUOTED_LITERAL.split(text)[::2]):
        path = _local_translation_path()
        try:
            text = _translate_locally(path, text)
        except (ImportError, OSError, ValueError) as exc:
            raise RuntimeError("Не удалось загрузить локальный переводчик. Проверьте профиль instruction и совместимость PyTorch.") from exc
    return compose_edit_instruction_prompt(text)


def _florence_nodes_module():
    """Import the installed extension by its exact Comfy package name."""

    candidates = (
        "custom_nodes.comfyui-florence2.nodes",
        "comfyui-florence2.nodes",
    )
    errors: list[str] = []
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - only target runtime paths
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("installed comfyui-florence2 extension is unavailable (" + "; ".join(errors) + ")")


class _FlorenceCaptionBackend:
    """Lazy adapter around native Florence2ModelLoader/Florence2Run nodes."""

    def __init__(self) -> None:
        model_path = _local_model_path()
        if model_path is None:
            raise FileNotFoundError(
                f"local {MODEL_NAME} checkpoint is missing; refusing to download a model"
            )
        module = _florence_nodes_module()
        loader = getattr(module, "Florence2ModelLoader", None)
        runner = getattr(module, "Florence2Run", None)
        if loader is None or runner is None:
            raise RuntimeError("comfyui-florence2 does not expose Florence2ModelLoader/Florence2Run")
        self._module = module
        self._model: Any = None
        self._runner: Any = None
        self._model_path = model_path
        try:
            describe_inputs = getattr(loader, "INPUT_TYPES", None)
            if callable(describe_inputs):
                describe_inputs()  # The native loader populates model_paths here.
            loaded = loader().loadmodel(
                MODEL_NAME,
                MODEL_PRECISION,
                MODEL_ATTENTION,
                lora=None,
                convert_to_safetensors=False,
            )
            values = getattr(loaded, "result", loaded)
            self._model = values[0] if isinstance(values, (tuple, list)) else values
            if not isinstance(self._model, dict) or "model" not in self._model:
                raise RuntimeError("Florence2ModelLoader returned no model payload")
            self._runner = runner()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _caption_result(result: Any) -> str:
        # Florence2Run returns (IMAGE, MASK, STRING, JSON).  Caption tasks
        # intentionally leave image/mask placeholders empty; output[2] is the
        # native generated text and may be a string or a one-item list.
        if isinstance(result, (tuple, list)) and len(result) >= 3:
            value = result[2]
        else:
            value = result
        if isinstance(value, (tuple, list)):
            value = value[0] if value else ""
        return sanitize_caption(value)

    def caption(self, clips: Iterable[np.ndarray]) -> list[str]:
        if self._model is None or self._runner is None:
            raise RuntimeError("Florence2 backend is not loaded")
        captions: list[str] = []
        try:
            import torch

            for clip in clips:
                array = np.asarray(clip, dtype=np.float32)
                if array.ndim != 3 or array.shape[-1] != 3:
                    continue
                tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
                result = self._runner.encode(
                    image=tensor,
                    text_input="",
                    florence2_model=self._model,
                    task=CAPTION_TASK,
                    fill_mask=False,
                    keep_model_loaded=False,
                    max_new_tokens=CAPTION_MAX_NEW_TOKENS,
                    num_beams=CAPTION_NUM_BEAMS,
                    do_sample=False,
                    output_mask_select="",
                    seed=CAPTION_SEED,
                )
                caption = self._caption_result(result)
                if caption:
                    captions.append(caption)
            return captions
        finally:
            # Florence2Run already offloads when keep_model_loaded=False.  The
            # explicit close in the outer node handles failures and guarantees
            # no VLM object survives into the H3 graph.
            pass

    def close(self) -> None:
        model_payload = self._model
        self._model = None
        self._runner = None
        if isinstance(model_payload, dict):
            model = model_payload.get("model")
            if model is not None:
                try:
                    module = self._module
                    offload_device = getattr(module, "offload_device", None)
                    if offload_device is not None and hasattr(model, "to"):
                        model.to(offload_device)
                    elif hasattr(model, "cpu"):
                        model.cpu()
                except Exception:
                    pass
        del model_payload
        gc.collect()
        try:
            module = self._module
            mm = getattr(module, "mm", None)
            if mm is not None and hasattr(mm, "soft_empty_cache"):
                mm.soft_empty_cache()
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _make_backend() -> CaptionBackend:
    factory = _caption_backend_factory
    return factory() if factory is not None else _FlorenceCaptionBackend()


def _preview_tensor(images: np.ndarray, mask: np.ndarray):
    """Return a full-frame green mask overlay as a native Comfy IMAGE."""

    image = np.asarray(images, dtype=np.float32)[..., :3].copy()
    alpha = np.clip(np.asarray(mask, dtype=np.float32)[..., None], 0.0, 1.0) * 0.38
    green = np.zeros_like(image)
    green[..., 1] = 1.0
    preview = np.clip(image * (1.0 - alpha) + green * alpha, 0.0, 1.0)
    try:
        import torch

        return torch.from_numpy(np.ascontiguousarray(preview, dtype=np.float32))
    except Exception:  # CPU-only test harnesses may intentionally omit torch.
        return np.ascontiguousarray(preview, dtype=np.float32)


def _small_preview_tensor(images: np.ndarray):
    """Return a tiny valid IMAGE without allocating a full-frame overlay."""

    source = np.asarray(images, dtype=np.float32)
    if source.ndim != 4 or source.shape[0] < 1 or source.shape[1] < 1 or source.shape[2] < 1:
        preview = np.zeros((1, 1, 1, 3), dtype=np.float32)
    else:
        pixel = source[:1, :1, :1, :3]
        if pixel.shape[-1] == 1:
            pixel = np.repeat(pixel, 3, axis=-1)
        elif pixel.shape[-1] < 3:
            pixel = np.zeros((1, 1, 1, 3), dtype=np.float32)
        preview = np.ascontiguousarray(pixel, dtype=np.float32)
    try:
        import torch

        return torch.from_numpy(preview)
    except Exception:  # CPU-only test harnesses may intentionally omit torch.
        return preview


def _status_join(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if str(part).strip())


class THSUniversalAutoPrompt:
    """Generate a safe masked-repair prompt from local Florence captions."""

    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Universal Inpaint/Repair"
    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("prompt", "caption", "preview")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "mode": (["instruction", "auto", "manual"], {"default": "instruction"}),
                "user_hint": ("STRING", {"default": "", "multiline": True}),
                "manual_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "max_frames": ("INT", {"default": 3, "min": 1, "max": 3, "step": 1}),
                "context_padding": ("FLOAT", {"default": DEFAULT_CONTEXT_PADDING, "min": 0.0, "max": 4.0, "step": 0.05}),
                "preview_enabled": ("BOOLEAN", {"default": True}),
                "repair_mode": ("COMBO", {"forceInput": True}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, mode=None, **kwargs):
        if mode in (None, "", "instruction", "auto", "manual"):
            return True
        return "mode must be instruction, auto or manual"

    def run(
        self,
        images,
        mask,
        mode="auto",
        user_hint="",
        manual_prompt="",
        max_frames=3,
        context_padding=DEFAULT_CONTEXT_PADDING,
        preview_enabled=True,
        repair_mode=None,
    ):
        # AutoPrompt is also an output node. A full queue can reach it even
        # when H3 is using an image guide, where no text task is required.
        if repair_mode in ("with_reference", "corrected_frame"):
            preview = _small_preview_tensor(_as_numpy(images[:1, :1, :1], label="images"))
            return ("", "", preview)

        mode_value = str(mode or "auto").strip().lower()
        if isinstance(preview_enabled, str):
            preview_on = preview_enabled.strip().lower() not in {"", "0", "false", "off", "no"}
        else:
            preview_on = True if preview_enabled is None else bool(preview_enabled)
        if mode_value == "instruction":
            prompt = _instruction_prompt(user_hint)
            if preview_on:
                preview = _preview_tensor(_as_numpy(images, label="images"), _as_numpy(mask, label="mask"))
            else:
                preview = _small_preview_tensor(_as_numpy(images[:1, :1, :1], label="images"))
            return {
                "ui": {"auto_prompt_text": [prompt], "auto_prompt_caption": [""],
                       "auto_prompt_instruction": [str(user_hint)], "auto_prompt_status": ["Промпт сформирован из вашей задачи."]},
                "result": (prompt, "", preview),
            }
        source = _as_numpy(images, label="images")
        source_mask = _as_numpy(mask, label="mask")
        max_frames = max(1, min(3, int(max_frames)))
        context_padding = float(context_padding)
        if mode_value not in {"auto", "manual"}:
            mode_value = "auto"

        # Manual mode has no semantic-caption dependency.  Avoid preparing
        # Telea crops (and therefore avoid any model-facing image work) when a
        # frozen prompt is already supplied.
        if mode_value == "manual":
            context = {"clips": [], "diagnostic": ""}
        else:
            context = prepare_caption_inputs(
                source,
                source_mask,
                max_frames=max_frames,
                context_padding=context_padding,
            )
        diagnostic = str(context.get("diagnostic", "") or "")
        captions: list[str] = []
        caption_diagnostic = ""

        if mode_value == "auto" and context.get("clips"):
            key = _cache_key(source, source_mask, max_frames, context_padding, context=context)
            cached = _cache_get(key)
            if cached is not None:
                captions, caption_diagnostic = cached
            else:
                backend: CaptionBackend | None = None
                try:
                    backend = _make_backend()
                    captions = [sanitize_caption(item) for item in backend.caption(context["clips"])]
                    captions = [item for index, item in enumerate(captions) if item and item not in captions[:index]]
                    if not captions:
                        caption_diagnostic = "Florence-2 returned an empty or unusable caption."
                except Exception as exc:
                    captions = []
                    caption_diagnostic = f"Florence-2 captioning unavailable ({type(exc).__name__})."
                finally:
                    if backend is not None:
                        try:
                            backend.close()
                        except Exception:
                            pass
                if captions:
                    _cache_put(key, captions, caption_diagnostic)
        elif mode_value == "manual":
            caption_diagnostic = "Manual mode skipped local VLM captioning."
        else:
            caption_diagnostic = "No usable masked context crop was available."

        prompt = compose_repair_prompt(
            captions,
            user_hint,
            mode=mode_value,
            manual_prompt=manual_prompt,
            diagnostic=_status_join(diagnostic, caption_diagnostic),
        )
        caption_text = " ".join(item for item in captions if item)
        if mode_value == "manual":
            # Compare the composed result rather than the raw field: control
            # text may be sanitised to an empty body and must not be reported
            # as an accepted manual instruction.
            if "Observed source appearance:" in prompt:
                status = "manual: local VLM skipped; supplied repair text used."
            else:
                status = "manual: no repair text supplied; generic masked-repair fallback used."
        elif captions:
            status = f"auto: Florence-2 PromptGen captioned {len(captions)} context frame(s)."
        else:
            status = "auto: generic masked-repair fallback used; no semantic caption was accepted."
        status = _status_join(status, diagnostic, caption_diagnostic)
        preview = _preview_tensor(source, source_mask) if preview_on else _small_preview_tensor(source)
        ui = {
            "auto_prompt_text": [prompt],
            "auto_prompt_caption": [caption_text],
            "auto_prompt_status": [status],
        }
        return {"ui": ui, "result": (prompt, caption_text, preview)}


NODE_CLASS_MAPPINGS = {"THSUniversalAutoPrompt": THSUniversalAutoPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"THSUniversalAutoPrompt": "Universal Auto Prompt (Florence2)"}


__all__ = [
    "CAPTION_TASK",
    "MODEL_NAME",
    "THSUniversalAutoPrompt",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "clear_caption_cache",
]
