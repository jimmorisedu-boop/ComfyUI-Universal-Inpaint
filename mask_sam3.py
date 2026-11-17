"""Lazy SAM3 box-segmentation adapter for the universal mask editor.

This module intentionally keeps all SAM3/Comfy imports behind the first
``sam3`` rectangle request.  The mask editor remains importable in a normal
CPU test process, while a Comfy runtime can reuse its installed native SAM3
model patcher.  No checkpoint download is attempted here: a missing local
checkpoint is reported as a typed :class:`MaskRefinementError`.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.util
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .mask_core import MaskRefinementError


# Keep this cache deliberately small.  SAM3 masks are one byte/pixel after
# normalization, but a few 4K source frames can still consume a meaningful
# amount of RAM.  The key includes a complete source digest, rectangle and
# checkpoint identity so source edits/checkpoint swaps invalidate entries.
MAX_SAM3_CACHE_ENTRIES = 8
MAX_SAM3_CACHE_BYTES = 128 * 1024 * 1024
_SAM3_CACHE: "OrderedDict[tuple[Any, ...], tuple[np.ndarray, int]]" = OrderedDict()
_SAM3_CACHE_BYTES = 0
_SAM3_CACHE_HITS = 0
_SAM3_CACHE_MISSES = 0
_SAM3_CACHE_LOCK = threading.RLock()

# The native interactive predictor is not re-entrant and several Comfy
# prompts may execute in parallel.  Serialize only the short model call; the
# surrounding parser/rasterizer stays fully concurrent.
_SAM3_INFERENCE_LOCK = threading.RLock()
_SAM3_NATIVE_PACKAGE = "_ths_native_sam3_nodes"
_SAM3_NATIVE_MODULE: Any | None = None
_SAM3_NATIVE_CONFIG: dict[str, Any] | None = None
_SAM3_NATIVE_ERROR: str | None = None


class SAM3UnavailableError(MaskRefinementError):
    """SAM3 cannot be used in this runtime or no local weights are present."""


def clear_sam3_cache() -> None:
    """Drop process-local rectangle masks and diagnostic counters."""

    global _SAM3_CACHE_BYTES, _SAM3_CACHE_HITS, _SAM3_CACHE_MISSES
    with _SAM3_CACHE_LOCK:
        _SAM3_CACHE.clear()
        _SAM3_CACHE_BYTES = 0
        _SAM3_CACHE_HITS = 0
        _SAM3_CACHE_MISSES = 0


def sam3_cache_info() -> dict[str, Any]:
    """Return bounded cache diagnostics suitable for node metadata/tests."""

    with _SAM3_CACHE_LOCK:
        return {
            "entries": len(_SAM3_CACHE),
            "bytes": int(_SAM3_CACHE_BYTES),
            "hits": int(_SAM3_CACHE_HITS),
            "misses": int(_SAM3_CACHE_MISSES),
            "max_entries": int(MAX_SAM3_CACHE_ENTRIES),
            "max_bytes": int(MAX_SAM3_CACHE_BYTES),
        }


def _runtime_roots() -> list[Path]:
    """Return explicitly configured/registered Comfy roots.

    The public addon intentionally has no workstation-specific fallback. A
    normal ComfyUI process supplies ``folder_paths``; standalone probes can
    set ``COMFYUI_PATH``, ``COMFY_BASE_PATH`` or ``THS_COMFYUI_PATH``.
    """

    roots: list[Path] = []
    for env_name in ("COMFYUI_PATH", "COMFY_BASE_PATH", "THS_COMFYUI_PATH"):
        raw = os.environ.get(env_name)
        if raw:
            roots.append(Path(raw).expanduser())
    try:
        import folder_paths  # type: ignore

        for attr in ("base_path", "models_dir"):
            raw = getattr(folder_paths, attr, None)
            if raw:
                value = Path(str(raw)).expanduser()
                roots.append(value if attr == "base_path" else value.parent)
        getter = getattr(folder_paths, "get_folder_paths", None)
        if callable(getter):
            for path in getter("checkpoints") or []:
                roots.append(Path(str(path)).expanduser().parent)
    except Exception:
        pass

    result: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            resolved = root
        key = str(resolved).lower()
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def _checkpoint_candidates() -> list[Path]:
    candidates: list[Path] = []
    for root in _runtime_roots():
        candidates.extend([
            root / "models" / "sam3" / "sam3.safetensors",
            root / "models" / "sam3" / "sam3.pt",
            root / "models" / "checkpoints" / "SAM3" / "sam3.safetensors",
            root / "models" / "checkpoints" / "SAM3" / "sam3.pt",
        ])
    result: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        key = str(resolved).lower()
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def _native_nodes_root() -> Path | None:
    """Locate the installed ComfyUI-SAM3 (native ModelPatcher) package."""

    for root in _runtime_roots():
        for candidate in (
            root / "custom_nodes" / "comfyui-sam3" / "nodes",
            root / "custom_nodes" / "ComfyUI-SAM3" / "nodes",
        ):
            if (candidate / "_model_cache.py").is_file() and (candidate / "sam3_model_patcher.py").is_file():
                return candidate.resolve()
    return None


def _native_model_cache_module() -> Any:
    """Load the native adapter as a synthetic package with relative imports."""

    global _SAM3_NATIVE_MODULE, _SAM3_NATIVE_ERROR
    if _SAM3_NATIVE_MODULE is not None:
        return _SAM3_NATIVE_MODULE
    if _SAM3_NATIVE_ERROR:
        raise SAM3UnavailableError(_SAM3_NATIVE_ERROR)
    nodes_root = _native_nodes_root()
    if nodes_root is None:
        _SAM3_NATIVE_ERROR = (
            "SAM3 native Comfy adapter is not installed; expected "
            "custom_nodes/comfyui-sam3/nodes/_model_cache.py"
        )
        raise SAM3UnavailableError(_SAM3_NATIVE_ERROR)
    try:
        # A synthetic package avoids importing the adapter's complete node
        # registry (which can pull optional UI modules) while preserving the
        # relative imports used by _model_cache.py and sam3_model_patcher.py.
        package = sys.modules.get(_SAM3_NATIVE_PACKAGE)
        if package is None:
            spec = importlib.util.spec_from_loader(_SAM3_NATIVE_PACKAGE, loader=None, is_package=True)
            if spec is None:
                raise ImportError("could not create SAM3 adapter package spec")
            package = importlib.util.module_from_spec(spec)
            package.__path__ = [str(nodes_root)]
            package.__package__ = _SAM3_NATIVE_PACKAGE
            sys.modules[_SAM3_NATIVE_PACKAGE] = package
        _SAM3_NATIVE_MODULE = importlib.import_module(f"{_SAM3_NATIVE_PACKAGE}._model_cache")
        return _SAM3_NATIVE_MODULE
    except SAM3UnavailableError:
        raise
    except Exception as exc:
        _SAM3_NATIVE_ERROR = (
            "SAM3 native Comfy adapter import failed: "
            f"{type(exc).__name__}: {exc}"
        )
        raise SAM3UnavailableError(_SAM3_NATIVE_ERROR) from exc


def _select_checkpoint() -> Path | None:
    for path in _checkpoint_candidates():
        try:
            if path.is_file() and path.stat().st_size > 0:
                return path
        except OSError:
            continue
    return None


def _native_config() -> dict[str, Any]:
    """Build a local-only config accepted by ``get_or_build_model``."""

    global _SAM3_NATIVE_CONFIG, _SAM3_NATIVE_ERROR
    if _SAM3_NATIVE_CONFIG is not None:
        return dict(_SAM3_NATIVE_CONFIG)
    checkpoint = _select_checkpoint()
    if checkpoint is None:
        _SAM3_NATIVE_ERROR = (
            "SAM3 local weights were not found; expected models/sam3/sam3.safetensors "
            "or models/checkpoints/SAM3/sam3.safetensors (no download is attempted)"
        )
        raise SAM3UnavailableError(_SAM3_NATIVE_ERROR)
    if checkpoint.suffix.lower() != ".safetensors":
        # The installed ModelPatcher adapter consumes the safetensors remapped
        # checkpoint.  Do not silently route a .pt file through a different
        # implementation: that could change mask semantics or VRAM handling.
        raise SAM3UnavailableError(
            f"SAM3 checkpoint {checkpoint} is not a safetensors file; "
            "install a local SAM3 safetensors checkpoint for the native adapter"
        )
    nodes_root = _native_nodes_root()
    if nodes_root is None:
        raise SAM3UnavailableError(
            "SAM3 native Comfy adapter is unavailable; cannot load local weights safely"
        )
    bpe = nodes_root / "sam3" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe.is_file():
        raise SAM3UnavailableError(
            f"SAM3 tokenizer asset is missing: {bpe}; reinstall the native adapter"
        )

    dtype = "fp32"
    try:
        import torch  # type: ignore
        import comfy.model_management as model_management  # type: ignore

        device = model_management.get_torch_device()
        if torch.cuda.is_available():
            if bool(getattr(model_management, "should_use_bf16", lambda _device: False)(device)):
                dtype = "bf16"
            elif bool(getattr(model_management, "should_use_fp16", lambda _device: True)(device)):
                dtype = "fp16"
    except Exception:
        # The adapter itself will provide an actionable dependency error when
        # model construction is attempted; capability checks remain cheap.
        pass
    _SAM3_NATIVE_CONFIG = {
        "checkpoint_path": str(checkpoint),
        "bpe_path": str(bpe),
        "precision": dtype,
        "dtype": dtype,
        "compile": False,
    }
    return dict(_SAM3_NATIVE_CONFIG)


def sam3_availability() -> dict[str, Any]:
    """Return non-loading SAM3 capability metadata for the frontend."""

    try:
        checkpoint = _select_checkpoint()
        if checkpoint is None:
            return {
                "available": False,
                "error": (
                    "SAM3: local weights were not found; expected "
                    "models/sam3/sam3.safetensors or models/checkpoints/SAM3/sam3.safetensors"
                ),
                "adapter": "comfyui-sam3",
            }
        if checkpoint.suffix.lower() != ".safetensors":
            return {
                "available": False,
                "error": f"SAM3: checkpoint is not safetensors: {checkpoint}",
                "adapter": "comfyui-sam3",
                "checkpoint": str(checkpoint),
            }
        nodes_root = _native_nodes_root()
        if nodes_root is None:
            return {
                "available": False,
                "error": "SAM3: native Comfy adapter is not installed",
                "adapter": "comfyui-sam3",
                "checkpoint": str(checkpoint),
            }
        bpe = nodes_root / "sam3" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe.is_file():
            return {
                "available": False,
                "error": f"SAM3: tokenizer asset is missing: {bpe}",
                "adapter": "comfyui-sam3",
                "checkpoint": str(checkpoint),
            }
        try:
            import torch  # type: ignore  # noqa: F401
            import comfy.model_management  # type: ignore  # noqa: F401
        except Exception as exc:
            return {
                "available": False,
                "error": (
                    "SAM3: native runtime dependencies are unavailable; run inside "
                    f"the ComfyUI Python environment ({type(exc).__name__}: {exc})"
                ),
                "adapter": "comfyui-sam3",
                "checkpoint": str(checkpoint),
            }
        return {
            "available": True,
            "error": None,
            "adapter": "comfyui-sam3",
            "checkpoint": str(checkpoint),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"SAM3: capability check failed: {type(exc).__name__}: {exc}",
            "adapter": "comfyui-sam3",
        }


def _source_digest(image: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(image))
    digest = hashlib.sha256()
    digest.update(repr(tuple(int(value) for value in array.shape)).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _rectangle_bounds(event: Mapping[str, Any], height: int, width: int) -> tuple[int, int, int, int]:
    x0 = int(np.floor(float(event["x"]) * (width - 1)))
    y0 = int(np.floor(float(event["y"]) * (height - 1)))
    x1 = int(np.ceil((float(event["x"]) + float(event["w"])) * (width - 1)))
    y1 = int(np.ceil((float(event["y"]) + float(event["h"])) * (height - 1)))
    x0 = min(max(0, x0), width - 1)
    y0 = min(max(0, y0), height - 1)
    x1 = min(max(x0, x1), width - 1)
    y1 = min(max(y0, y1), height - 1)
    return x0, y0, x1, y1


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if tuple(mask.shape) == tuple(shape):
        return np.asarray(mask)
    try:
        import cv2  # type: ignore

        return cv2.resize(np.asarray(mask, dtype=np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    except Exception:
        try:
            from PIL import Image

            image = Image.fromarray(np.asarray(mask, dtype=np.float32), mode="F")
            return np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.NEAREST))
        except Exception as exc:
            raise SAM3UnavailableError(
                f"SAM3 returned a mask with shape {tuple(mask.shape)} instead of {shape}, "
                f"and no resize backend is available: {exc}"
            ) from exc


def _cache_get(key: tuple[Any, ...]) -> np.ndarray | None:
    global _SAM3_CACHE_HITS, _SAM3_CACHE_MISSES, _SAM3_CACHE_BYTES
    with _SAM3_CACHE_LOCK:
        entry = _SAM3_CACHE.get(key)
        if entry is None:
            _SAM3_CACHE_MISSES += 1
            return None
        _SAM3_CACHE.move_to_end(key)
        _SAM3_CACHE_HITS += 1
        return np.asarray(entry[0], dtype=bool).copy()


def _cache_put(key: tuple[Any, ...], mask: np.ndarray) -> None:
    global _SAM3_CACHE_BYTES
    array = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    item_bytes = int(array.nbytes)
    if item_bytes <= 0 or item_bytes > int(MAX_SAM3_CACHE_BYTES):
        return
    with _SAM3_CACHE_LOCK:
        previous = _SAM3_CACHE.pop(key, None)
        if previous is not None:
            _SAM3_CACHE_BYTES = max(0, _SAM3_CACHE_BYTES - int(previous[1]))
        while _SAM3_CACHE and (
            len(_SAM3_CACHE) >= int(MAX_SAM3_CACHE_ENTRIES)
            or _SAM3_CACHE_BYTES + item_bytes > int(MAX_SAM3_CACHE_BYTES)
        ):
            _old_key, old = _SAM3_CACHE.popitem(last=False)
            _SAM3_CACHE_BYTES = max(0, _SAM3_CACHE_BYTES - int(old[1]))
        if _SAM3_CACHE_BYTES + item_bytes > int(MAX_SAM3_CACHE_BYTES):
            return
        _SAM3_CACHE[key] = (array.copy(), item_bytes)
        _SAM3_CACHE_BYTES += item_bytes


def _offload_model(model: Any) -> None:
    """Return a native ModelPatcher to its offload device after inference."""

    removed = False
    try:
        import comfy.model_management as model_management  # type: ignore

        # ``load_models_gpu`` registers a LoadedModel entry.  Ask that entry
        # to unload itself so Comfy's finalizer/bookkeeping stay in sync, then
        # remove only this SAM3 model from the active list.  Calling
        # ``unpatch_model`` directly would leave a stale LoadedModel pointing
        # at zero loaded bytes and can confuse the next H3 eviction pass.
        loaded_models = getattr(model_management, "current_loaded_models", None)
        if isinstance(loaded_models, list):
            for index in range(len(loaded_models) - 1, -1, -1):
                entry = loaded_models[index]
                if getattr(entry, "model", None) is model:
                    unload = getattr(entry, "model_unload", None)
                    if callable(unload):
                        unload()
                    loaded_models.pop(index)
                    removed = True
                    break
    except Exception:
        removed = False
    if not removed:
        # Test doubles and older Comfy builds may not expose the registry;
        # retain a best-effort offload fallback in those runtimes.
        unpatch = getattr(model, "unpatch_model", None)
        if callable(unpatch):
            try:
                unpatch(getattr(model, "_offload_device", None))
            except TypeError:
                try:
                    unpatch()
                except Exception:
                    pass
            except Exception:
                pass
    try:
        import comfy.model_management as model_management  # type: ignore

        model_management.soft_empty_cache(force=True)
    except Exception:
        pass


def _predict_native(image: np.ndarray, event: Mapping[str, Any]) -> np.ndarray:
    """Run one box or point prompt through the native interactive predictor."""

    config = _native_config()
    cache_module = _native_model_cache_module()
    getter = getattr(cache_module, "get_or_build_model", None)
    if not callable(getter):
        raise SAM3UnavailableError("SAM3 native adapter has no get_or_build_model()")

    try:
        import torch  # type: ignore  # noqa: F401
        import comfy.model_management as model_management  # type: ignore
        from PIL import Image
    except Exception as exc:
        raise SAM3UnavailableError(
            "SAM3 requires the installed ComfyUI runtime, torch and Pillow: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise SAM3UnavailableError("SAM3 rectangle refinement requires an HxWx3 source frame")
    height, width = map(int, array.shape[:2])
    point_mode = isinstance(event.get("points"), (list, tuple))
    if point_mode:
        x0 = y0 = 0
        x1, y1 = width - 1, height - 1
    else:
        x0, y0, x1, y1 = _rectangle_bounds(event, height, width)
    selection = np.zeros((height, width), dtype=bool)
    selection[y0:y1 + 1, x0:x1 + 1] = True
    selected_pixels = int(selection.sum())
    if selected_pixels < 1:
        raise SAM3UnavailableError("SAM3 prompt received an empty selection")

    # Native Comfy SAM3 accepts a PIL image and pixel xyxy box with
    # normalize_coords=True (the predictor divides x/y by original width/
    # height before mapping to its 1008px prompt space).
    rgb = np.asarray(np.round(np.clip(np.nan_to_num(array[..., :3]), 0.0, 1.0) * 255.0), dtype=np.uint8)
    pil_image = Image.fromarray(rgb, mode="RGB")
    model = None
    try:
        model = getter(config)
        load_models_gpu = getattr(model_management, "load_models_gpu", None)
        if not callable(load_models_gpu):
            raise SAM3UnavailableError("Comfy model_management.load_models_gpu() is unavailable")
        load_models_gpu([model])
        processor = getattr(model, "processor", None)
        if processor is None:
            raise SAM3UnavailableError("SAM3 native model has no processor")
        sync = getattr(processor, "sync_device_with_model", None)
        if callable(sync):
            sync()
        predictor = getattr(processor, "model", None)
        if predictor is None or not callable(getattr(predictor, "predict_inst", None)):
            raise SAM3UnavailableError(
                "SAM3 interactive predictor is unavailable; load a model with "
                "enable_inst_interactivity=True"
            )
        state = processor.set_image(pil_image)
        raw_box = None
        point_coords = None
        point_labels = None
        if point_mode:
            raw_points = event.get("points", [])
            point_coords = np.asarray(
                [[float(point[0]) * max(1, width - 1), float(point[1]) * max(1, height - 1)] for point in raw_points],
                dtype=np.float32,
            )
            point_labels = np.asarray(event.get("labels", [1] * len(point_coords)), dtype=np.int32)
            if point_coords.ndim != 2 or point_coords.shape[1] != 2 or len(point_coords) != len(point_labels):
                raise SAM3UnavailableError("SAM3 point prompt must contain matching points and labels")
        else:
            raw_box = np.asarray([x0, y0, x1, y1], dtype=np.float32)
        masks, scores, _low_res = predictor.predict_inst(
            state,
            point_coords=point_coords,
            point_labels=point_labels,
            box=raw_box,
            mask_input=None,
            multimask_output=True,
            normalize_coords=True,
        )
        masks_array = np.asarray(masks)
        scores_array = np.asarray(scores).reshape(-1) if scores is not None else np.arange(len(masks_array), dtype=np.float32)
        if masks_array.ndim == 2:
            masks_array = masks_array[None, ...]
        if masks_array.ndim != 3 or masks_array.shape[0] < 1:
            raise SAM3UnavailableError("SAM3 returned no mask candidates for the prompt")
        valid: list[tuple[float, int, np.ndarray]] = []
        for index, candidate in enumerate(masks_array):
            candidate = _resize_mask(np.squeeze(candidate), (height, width))
            candidate = np.asarray(candidate > 0.0, dtype=bool) & selection
            foreground_pixels = int(candidate.sum())
            if foreground_pixels < 1 or foreground_pixels >= selected_pixels:
                continue
            try:
                score = float(scores_array[index])
            except (TypeError, ValueError, IndexError):
                score = float("-inf")
            if not np.isfinite(score):
                score = float("-inf")
            valid.append((score, index, candidate))
        if not valid:
            raise SAM3UnavailableError(
                "SAM3 found no valid foreground object for the prompt "
                "(all candidates were empty or filled the selection)"
            )
        _score, _index, selected = max(valid, key=lambda item: (item[0], -item[1]))
        return np.asarray(selected, dtype=bool)
    except SAM3UnavailableError:
        raise
    except Exception as exc:
        raise SAM3UnavailableError(
            f"SAM3 rectangle segmentation failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        # Release per-image feature state before returning the ModelPatcher to
        # its offload device.  The model cache may retain CPU weights, but it
        # must not permanently occupy VRAM between H3 requests.
        try:
            if "state" in locals():
                del state
        except Exception:
            pass
        _offload_model(model)
        gc.collect()


def refine_rectangle_sam3(image: np.ndarray, event: Mapping[str, Any]) -> np.ndarray:
    """Segment one rectangle with SAM3 and return a source-sized bool mask.

    Errors are always typed and explicit; callers must block refinement rather
    than silently falling back to a full rectangle or GrabCut.
    """

    if image is None:
        raise SAM3UnavailableError("SAM3 rectangle refinement requires a source frame")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[0] < 1 or array.shape[1] < 1:
        raise SAM3UnavailableError("SAM3 rectangle refinement requires a valid source frame")
    try:
        config = _native_config()
        checkpoint = config.get("checkpoint_path")
        checkpoint_identity = None
        if checkpoint:
            path = Path(str(checkpoint))
            try:
                stat = path.stat()
                checkpoint_identity = (str(path), int(stat.st_mtime_ns), int(stat.st_size))
            except OSError:
                checkpoint_identity = (str(path),)
        key = (
            "ths-sam3-rectangle-v1",
            _source_digest(array),
            tuple(float(event[name]) for name in ("x", "y", "w", "h")),
            checkpoint_identity,
        )
        cached = _cache_get(key)
        if cached is not None:
            return cached
        with _SAM3_INFERENCE_LOCK:
            # A concurrent request may have populated the same key while this
            # call waited for the model lock.
            cached = _cache_get(key)
            if cached is not None:
                return cached
            result = _predict_native(array, event)
            result = np.asarray(result, dtype=bool)
            expected = tuple(int(value) for value in array.shape[:2])
            if result.shape != expected:
                raise SAM3UnavailableError(
                    f"SAM3 returned mask shape {tuple(result.shape)}; expected {expected}"
                )
            x0, y0, x1, y1 = _rectangle_bounds(event, expected[0], expected[1])
            selection = np.zeros(expected, dtype=bool)
            selection[y0:y1 + 1, x0:x1 + 1] = True
            result &= selection
            selected_pixels = int(selection.sum())
            foreground_pixels = int(result.sum())
            if foreground_pixels < 1 or foreground_pixels >= selected_pixels:
                raise SAM3UnavailableError(
                    "SAM3: no valid foreground object remained inside the rectangle"
                )
            _cache_put(key, result)
            return result.copy()
    except SAM3UnavailableError:
        raise
    except Exception as exc:
        raise SAM3UnavailableError(
            f"SAM3 rectangle refinement failed: {type(exc).__name__}: {exc}"
        ) from exc


__all__ = [
    "MAX_SAM3_CACHE_ENTRIES",
    "MAX_SAM3_CACHE_BYTES",
    "SAM3UnavailableError",
    "clear_sam3_cache",
    "sam3_cache_info",
    "sam3_availability",
    "refine_rectangle_sam3",
]
