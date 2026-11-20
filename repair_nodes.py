"""Small, standalone nodes used by the universal masked H3 repair workflow.

The module deliberately has no import-time dependency on ComfyUI or torch.  This
keeps the CPU/unit-test path useful on a clean machine while the same classes use
ComfyUI's normal ``IMAGE``/``MASK``/``LATENT`` contracts when loaded as a custom
node.  H3 conditioning is a thin wrapper around the installed
``MiniMaxH3ImageToVideo.execute`` implementation; it does not recreate any H3
internals.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:  # Optional for the lightweight CPU tests.
    import numpy as np
except Exception:  # pragma: no cover - numpy is present in the test image.
    np = None  # type: ignore[assignment]

try:  # ComfyUI supplies torch; tests intentionally do not require it.
    import torch
except Exception:  # pragma: no cover - exercised only on a no-torch machine.
    torch = None  # type: ignore[assignment]

try:  # OpenCV is already part of the target ComfyUI image; keep CPU imports optional.
    import cv2
except Exception:  # pragma: no cover - lightweight unit environments may omit it.
    cv2 = None  # type: ignore[assignment]


CATEGORY = "Universal Inpaint"

# The reference graph deliberately keeps its upload widget and reference
# conditioning separate from the historical reference-free graph.  The
# sentinel is a real combo value rather than an empty required ``LoadImage``
# input, so a reference-free run can validate and execute without touching the
# ComfyUI input directory or decoding an image.
REFERENCE_NONE = "__none__"
REFERENCE_MODES = ("without_reference", "with_reference", "corrected_frame")
REFERENCE_IMAGE_SIZES = ("match", "max")
DEFAULT_REFERENCE_PROMPT = (
    "Use <Picture 1> inside the user's repair mask. Apply the reference's "
    "visual identity and appearance to the masked replacement; the replacement "
    "may change source colors and materials to match it. Preserve the source "
    "exactly outside the repair mask."
)
DEFAULT_REFERENCE_NEGATIVE = (
    "visible seams, flicker, temporal inconsistency, distorted geometry, "
    "texture discontinuity, unintended changes, duplicated details"
)
DEFAULT_CORRECTED_FRAME_PROMPT = (
    "Use the supplied corrected frame as a correction guide for the user's "
    "masked region. Restore the defect with natural texture and geometry while "
    "preserving subject identity, source timing, camera, lighting, materials "
    "and every pixel outside the repair mask. Do not copy the guide outside "
    "the masked region, add new objects, or change the unmasked background."
)


def _normalise_reference_mode(mode: Any) -> str:
    value = str(mode or "without_reference").strip().lower()
    if value not in REFERENCE_MODES:
        raise ValueError(
            "mode must be without_reference, with_reference, or corrected_frame; "
            f"received {mode!r}"
        )
    return value


def _reference_image_missing(value: Any) -> bool:
    """Treat unresolved/empty optional image containers as no image."""

    if value is None:
        return True
    return isinstance(value, (list, tuple, dict, set)) and len(value) == 0


def _reference_file_choices() -> list[str]:
    """Return bounded Comfy input filenames for the upload-aware combo.

    Importing ``folder_paths`` is intentionally lazy.  Unit tests can import
    the custom node package without a ComfyUI checkout, and the reference-free
    execution path never needs to enumerate or open an image.
    """

    choices = [REFERENCE_NONE]
    try:
        import folder_paths  # type: ignore

        input_dir = Path(str(folder_paths.get_input_directory()))
        files = [
            name
            for name in input_dir.iterdir()
            if name.is_file()
        ] if hasattr(input_dir, "iterdir") else []
        # ``folder_paths`` exposes the same content-type filter as the native
        # LoadImage node.  If an older Comfy build does not provide it, keep the
        # safe sentinel and let uploaded values validate at execution time.
        filter_files = getattr(folder_paths, "filter_files_content_types", None)
        if filter_files is not None:
            files = filter_files([name.name for name in files], ["image"])
        choices.extend(sorted(str(name) for name in files if str(name).strip()))
    except Exception:
        # Discovery is a UI convenience only; absence of a local input folder
        # must not make the optional reference feature unloadable.
        pass
    # Preserve deterministic UI order and avoid duplicate sentinels/files.
    return list(dict.fromkeys(choices))


def _reference_prompt(prompt: str) -> str:
    """Make the native ``<Picture 1>`` label and mask scope explicit.

    The instruction is intentionally stronger than a source-image caption:
    reference appearance is applied only to user-masked pixels, while colors
    and materials in the requested replacement region may change to match the
    reference.  Unmasked source pixels remain the responsibility of the final
    exact composite node.
    """

    user_prompt = str(prompt or "").strip()
    instruction = (
        "Use <Picture 1> only as the visual identity/appearance reference for "
        "the pixels inside the user's repair mask. Apply the reference there "
        "so the masked replacement may change the source colors, materials, "
        "clothing or other appearance requested for replacement; do not preserve the original "
        "masked appearance. Preserve the source exactly outside the repair "
        "mask."
    )
    if "<picture 1>" in user_prompt.lower():
        return (
            f"{instruction}\n{user_prompt}\n"
            "Reference scope rule (highest priority): when the request text "
            "conflicts with the reference appearance, do not preserve the "
            "original masked colors or materials."
            if user_prompt
            else instruction
        )
    return (
        f"{instruction}\n{user_prompt}\n"
        "Reference scope rule (highest priority): when the request text "
        "conflicts with the reference appearance, do not preserve the "
        "original masked colors or materials."
        if user_prompt
        else instruction
    )


def _is_torch(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _ndim(value: Any) -> int:
    return int(value.ndim)


def _batch_length(value: Any) -> int:
    try:
        return int(value.shape[0])
    except Exception as exc:  # pragma: no cover - defensive error text.
        raise ValueError("expected a batched IMAGE/MASK tensor") from exc


def _normalise_mask(mask: Any) -> Any:
    """Return masks as [T,H,W], accepting Comfy's [H,W] shorthand."""

    if not hasattr(mask, "shape"):
        raise ValueError("mask must be a tensor/array")
    if _ndim(mask) == 2:
        if _is_torch(mask):
            return mask.unsqueeze(0)
        return np.expand_dims(mask, 0)
    if _ndim(mask) == 3:
        return mask
    # A few mask nodes expose [T,1,H,W]; accepting it is harmless and makes the
    # composite/pad nodes tolerant of both Comfy conventions.
    if _ndim(mask) == 4 and int(mask.shape[1]) == 1:
        return mask[:, 0]
    raise ValueError(f"expected mask [T,H,W] or [H,W], got {tuple(mask.shape)}")


def _concat(a: Any, b: Any, dim: int = 0) -> Any:
    if _is_torch(a):
        return torch.cat((a, b), dim=dim)
    return np.concatenate((a, b), axis=dim)


def _repeat_last(value: Any, count: int) -> Any:
    if count <= 0:
        return value[:0]
    tail = value[-1:]
    if _is_torch(value):
        return tail.expand(count, *tail.shape[1:]).clone()
    return np.repeat(tail, count, axis=0).copy()


def _zeros_like_shape(value: Any, shape: tuple[int, ...], *, dtype: Any = None) -> Any:
    if _is_torch(value):
        return torch.zeros(shape, dtype=dtype or value.dtype, device=value.device)
    return np.zeros(shape, dtype=dtype or value.dtype)


def _h3_canvas_size(width: int, height: int) -> tuple[int, int]:
    """H3 token grid: 16 pixel VAE compression and 2x2 latent patches."""
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError("H3 source width and height must be positive")
    return (width + 31) // 32 * 32, (height + 31) // 32 * 32


def _pad_h3_images(images: Any) -> Any:
    """Extend the right and bottom edges without moving source pixels."""
    count, height, width, channels = images.shape
    canvas_width, canvas_height = _h3_canvas_size(width, height)
    if (canvas_width, canvas_height) == (width, height):
        return images
    shape = (count, canvas_height, canvas_width, channels)
    padded = images.new_empty(shape) if _is_torch(images) else np.empty(shape, dtype=images.dtype)
    padded[:, :height, :width] = images
    padded[:, :height, width:] = images[:, :, -1:]
    padded[:, height:] = padded[:, height - 1:height]
    return padded


def _next_h3_frame_count(frame_count: int, modulo: int = 17, remainder: int = 5) -> int:
    """Smallest H3 timeline (17*n+5) containing ``frame_count`` frames."""

    frame_count = int(frame_count)
    modulo = int(modulo)
    remainder = int(remainder)
    if frame_count < 1:
        raise ValueError("cannot pad an empty timeline")
    if modulo < 1:
        raise ValueError("modulo must be positive")
    if remainder < 0 or remainder >= modulo:
        raise ValueError("remainder must be in [0, modulo)")
    if frame_count <= remainder:
        return max(frame_count, remainder)
    return remainder + modulo * math.ceil((frame_count - remainder) / modulo)


def telea_prefill_masked(images: Any, mask: Any, *, radius: float = 3.0) -> Any:
    """Fill masked RGB pixels with a local OpenCV Telea prefill.

    This is a model-input aid only: the returned tensor keeps the original
    values bit-for-bit outside ``mask > 0.01``.  The final output graph still
    composites generated pixels over the untouched source, so prefill cannot
    leak into unmasked delivery pixels.  Torch tensors retain their original
    dtype/device; NumPy arrays retain their dtype.
    """

    if np is None:
        return images.clone() if _is_torch(images) else images
    if not hasattr(images, "shape") or _ndim(images) != 4:
        raise ValueError("images must have shape T x H x W x C")
    source_mask = _normalise_mask(mask)
    if _batch_length(source_mask) != _batch_length(images):
        raise ValueError("image and mask frame counts must match")
    if cv2 is None:
        return images.clone() if _is_torch(images) else np.array(images, copy=True)

    mask_np = source_mask.detach().to(device="cpu", dtype=torch.float32).numpy() if _is_torch(source_mask) else np.asarray(source_mask)
    threshold = np.asarray(mask_np > 0.01, dtype=bool)
    if _is_torch(images):
        result = images.clone()
        for index in range(_batch_length(images)):
            frame_mask = threshold[index]
            if not np.any(frame_mask):
                continue
            frame = images[index].detach().to(device="cpu", dtype=torch.float32).numpy()
            try:
                channels = int(frame.shape[-1])
                rgb = np.clip(frame[..., : min(3, channels)], 0.0, 1.0)
                encoded = np.asarray(np.round(rgb * 255.0), dtype=np.uint8)
                inpainted = cv2.inpaint(encoded, np.asarray(frame_mask, dtype=np.uint8) * 255, float(radius), cv2.INPAINT_TELEA)
                filled = np.asarray(inpainted, dtype=np.float32) / 255.0
                replacement = torch.from_numpy(np.asarray(filled)).to(device=images.device, dtype=images.dtype)
                result[index, ..., : replacement.shape[-1]][frame_mask] = replacement[frame_mask]
            except Exception:
                # A malformed/degenerate frame should retain its source rather
                # than make a repair queue fail before the H3 graph runs.
                continue
        return result

    result_np = np.array(images, copy=True)
    for index in range(_batch_length(result_np)):
        frame_mask = threshold[index]
        if not np.any(frame_mask):
            continue
        frame = np.asarray(result_np[index], dtype=np.float32)
        try:
            channels = int(frame.shape[-1])
            rgb = np.clip(frame[..., : min(3, channels)], 0.0, 1.0)
            encoded = np.asarray(np.round(rgb * 255.0), dtype=np.uint8)
            inpainted = cv2.inpaint(encoded, np.asarray(frame_mask, dtype=np.uint8) * 255, float(radius), cv2.INPAINT_TELEA)
            filled = np.asarray(inpainted, dtype=np.float32) / 255.0
            result_np[index, ..., : filled.shape[-1]][frame_mask] = filled[frame_mask].astype(result_np.dtype, copy=False)
        except Exception:
            continue
    return result_np


def pad_source_inputs(
    images: Any,
    mask: Any,
    audio: Mapping[str, Any] | None = None,
    model_fps: float = 24.0,
    modulo: int = 17,
    remainder: int = 5,
    silence_sample_rate: int = 48_000,
) -> tuple[Any, Any, dict[str, Any], int, int]:
    """Pad source RGB/mask/audio to H3's grid without changing real frames.

    ``model_fps`` is H3's internal timeline rate.  It is intentionally separate
    from the source playback FPS, which is carried unchanged by the final
    ``CreateVideo`` node in the workflow.
    """

    if not hasattr(images, "shape") or _ndim(images) < 1 or _batch_length(images) < 1:
        raise ValueError("images must contain at least one frame")
    source_mask = _normalise_mask(mask)
    original_count = _batch_length(images)
    if _batch_length(source_mask) != original_count:
        raise ValueError("image and mask frame counts must match")
    model_fps = float(model_fps)
    if not math.isfinite(model_fps) or model_fps <= 0:
        raise ValueError("model_fps must be positive")
    sample_rate = int(silence_sample_rate)
    if sample_rate <= 0:
        raise ValueError("silence_sample_rate must be positive")

    padded_count = _next_h3_frame_count(original_count, modulo, remainder)
    added = padded_count - original_count
    if added:
        padded_images = _concat(images, _repeat_last(images, added))
        padded_mask = _concat(source_mask, _repeat_last(source_mask, added))
    else:
        padded_images = images
        padded_mask = source_mask

    padded_audio = _pad_audio_for_count(
        images,
        padded_count,
        audio,
        model_fps=model_fps,
        silence_sample_rate=sample_rate,
    )
    return padded_images, padded_mask, padded_audio, original_count, padded_count


def _pad_audio_for_count(
    template: Any,
    padded_count: int,
    audio: Mapping[str, Any] | None = None,
    *,
    model_fps: float = 24.0,
    silence_sample_rate: int = 48_000,
) -> dict[str, Any]:
    """Apply the existing H3 waveform truncation/zero-padding contract.

    ``THSUniversalH3PadAudio`` uses this helper without touching masks or RGB
    frames.  The arithmetic and waveform shape normalization intentionally
    match :func:`pad_source_inputs` exactly, including Python's ``round``
    behaviour at half-sample boundaries.
    """

    model_fps = float(model_fps)
    if not math.isfinite(model_fps) or model_fps <= 0:
        raise ValueError("model_fps must be positive")
    sample_rate = int(silence_sample_rate)
    if sample_rate <= 0:
        raise ValueError("silence_sample_rate must be positive")

    waveform: Any
    if audio is None:
        waveform = _zeros_like_shape(
            template,
            (1, 2, 0),
            dtype=(torch.float32 if _is_torch(template) else np.float32),
        )
    else:
        if not isinstance(audio, Mapping):
            raise ValueError("audio must be a mapping with waveform and sample_rate")
        waveform = audio.get("waveform")
        if waveform is None or not hasattr(waveform, "shape"):
            raise ValueError("audio must contain a waveform tensor/array")
        sample_rate = int(audio.get("sample_rate", 0))
        if sample_rate <= 0:
            raise ValueError("audio sample_rate must be positive")
        if _ndim(waveform) == 1:
            waveform = waveform[None, None, :]
        elif _ndim(waveform) == 2:
            waveform = waveform[None, ...]
        elif _ndim(waveform) < 1:
            raise ValueError("audio waveform must have a sample dimension")

    target_samples = round(int(padded_count) / model_fps * sample_rate)
    kept = waveform[..., :target_samples]
    current = int(kept.shape[-1])
    if current < target_samples:
        pad_shape = tuple(kept.shape[:-1]) + (target_samples - current,)
        zeros = _zeros_like_shape(kept, pad_shape, dtype=kept.dtype)
        kept = _concat(kept, zeros, dim=-1)
    # Copy the waveform so downstream nodes never mutate a caller's audio.
    if hasattr(kept, "clone"):
        kept = kept.clone()
    elif hasattr(kept, "copy"):
        kept = kept.copy()
    return {"waveform": kept, "sample_rate": sample_rate}


def _node_result(result: Any) -> tuple[Any, ...]:
    values = getattr(result, "result", result)
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        return tuple(values)
    return (values,)


def _encode_negative(clip: Any, prompt: str) -> Any:
    """Use the same installed CLIP scheduling API as the H3 node itself."""

    try:
        tokens = clip.tokenize(prompt, images=[])
    except TypeError:
        # Older compatible CLIP wrappers omit the optional images keyword.
        tokens = clip.tokenize(prompt)
    return clip.encode_from_tokens_scheduled(tokens)


class THSUniversalH3RepairConditioning:
    """Generic H3 t2va/fl2va conditioning for masked source-latent repair.

    With ``anchor_source_edges=False`` (the workflow default) this is genuinely
    reference-free: the only image fed to H3 is the RGB source latent in the
    sampler's ``latent_image`` input.  Enabling the optional edge anchors passes
    the first/last source frames to the installed H3 image-to-video node; this is
    a conservative fallback when a shot benefits from endpoint guidance, not an
    identity/reference-image path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": "Restore only the user-masked pixels with natural texture and geometry inferred from adjacent unmasked frames. Preserve subject identity, source timing, camera, lighting, materials and all unmasked pixels. Do not add text, logos, symbols, faces, emojis or new objects."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "changed background, changed clothing, extra people, camera motion, unmasked edits, text, logos, symbols, faces, emojis, cartoon artifacts"}),
                "width": ("INT", {"default": 768, "min": 32, "max": 16_384, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 16_384, "step": 32}),
                "length": ("INT", {"default": 22, "min": 5, "max": 3_600, "step": 17}),
                "anchor_source_edges": ("BOOLEAN", {"default": False, "tooltip": "Optional first/last source-frame anchors; leave off for reference-free masked repair."}),
            },
            "optional": {"source_images": ("IMAGE",)},
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "h3_latent")
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/H3"

    def run(
        self,
        clip,
        vae,
        prompt,
        negative_prompt,
        width,
        height,
        length,
        anchor_source_edges=False,
        source_images=None,
    ):
        canvas_width, canvas_height = _h3_canvas_size(width, height)
        if anchor_source_edges and source_images is None:
            raise ValueError("source_images is required when anchor_source_edges is enabled")
        first_frame = last_frame = None
        if anchor_source_edges:
            if _batch_length(source_images) < 1:
                raise ValueError("source_images must contain at least one frame")
            first_frame = source_images[:1]
            last_frame = source_images[-1:]
            if tuple(source_images.shape[1:3]) == (int(height), int(width)):
                first_frame = _pad_h3_images(first_frame)
                last_frame = _pad_h3_images(last_frame)

        # This is the installed ComfyUI H3 call, kept deliberately explicit so
        # upgrades fail loudly rather than silently inventing conditioning keys.
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo

        result = MiniMaxH3ImageToVideo.execute(
            clip=clip,
            vae=vae,
            prompt=str(prompt),
            width=canvas_width,
            height=canvas_height,
            length=int(length),
            first_frame=first_frame,
            last_frame=last_frame,
        )
        values = _node_result(result)
        if len(values) < 2:
            raise ValueError("MiniMaxH3ImageToVideo returned no latent")
        positive, h3_latent = values[:2]
        negative = _encode_negative(clip, str(negative_prompt))
        return (positive, negative, h3_latent)


class THSUniversalReferenceImage:
    """Optional, upload-aware reference image loader for the reference graph.

    This node intentionally owns both the mode selector and the filename
    combo.  In ``without_reference`` mode it returns ``None`` immediately and
    does not import ``folder_paths``/``nodes`` or perform any validation or
    image I/O.  In ``with_reference`` mode it reuses ComfyUI's native
    ``LoadImage`` implementation after validating its bounded input path.
    """

    RETURN_TYPES = ("IMAGE", "COMBO")
    RETURN_NAMES = ("reference_image", "mode")
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/Reference"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    list(REFERENCE_MODES),
                    {
                        "default": "without_reference",
                        "tooltip": "without_reference runs with no selected/uploaded image; with_reference uses a style reference; corrected_frame uses the upload as a masked correction guide.",
                    },
                ),
                "reference_image": (
                    _reference_file_choices(),
                    {
                        "default": REFERENCE_NONE,
                        "image_upload": True,
                        "tooltip": "Upload/select a single reference image. Leave __none__ while mode is without_reference.",
                    },
                ),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, mode=None, reference_image=None, **kwargs):
        try:
            mode_value = _normalise_reference_mode(mode)
        except ValueError as exc:
            return str(exc)
        # The critical off path: no folder-path lookup, existence check or
        # decoder call when no reference is requested.
        if mode_value == "without_reference":
            return True
        value = str(reference_image or "").strip()
        if not value or value == REFERENCE_NONE:
            return "reference_image is required when mode is with_reference or corrected_frame"
        try:
            import folder_paths  # type: ignore

            exists = folder_paths.exists_annotated_filepath(value)
        except Exception as exc:
            return f"cannot validate reference image {value!r}: {exc}"
        if not exists:
            return f"Invalid reference image file: {value}"
        return True

    @classmethod
    def IS_CHANGED(cls, mode="without_reference", reference_image=REFERENCE_NONE):
        """Invalidate Comfy's cache when the selected upload changes.

        OFF is a stable sentinel and deliberately performs no path lookup.  ON
        delegates to the native loader's file hash when available, with a
        bounded annotated-path hash fallback for older Comfy builds.
        """

        mode_value = _normalise_reference_mode(mode)
        if mode_value == "without_reference":
            return "without_reference"
        value = str(reference_image or "").strip()
        if not value or value == REFERENCE_NONE:
            return "with_reference:missing"
        try:
            from nodes import LoadImage  # type: ignore

            changed = getattr(LoadImage, "IS_CHANGED", None)
            if changed is not None:
                return changed(value)
        except Exception:
            pass
        try:
            import folder_paths  # type: ignore

            path = folder_paths.get_annotated_filepath(value)
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        except Exception:
            return f"with_reference:{value}"

    def run(self, mode="without_reference", reference_image=REFERENCE_NONE):
        mode_value = _normalise_reference_mode(mode)
        if mode_value == "without_reference":
            # Do not even import the Comfy image loader on the OFF path.
            return (None, mode_value)
        validation = self.VALIDATE_INPUTS(mode_value, reference_image)
        if validation is not True:
            raise ValueError(str(validation))
        try:
            # Calling the installed native node gives us its bounded annotated
            # filepath handling, animated-image support and device/dtype
            # placement rather than duplicating those semantics here.
            from nodes import LoadImage  # type: ignore
        except Exception as exc:  # pragma: no cover - only a broken Comfy install.
            raise RuntimeError("ComfyUI LoadImage is unavailable") from exc
        result = LoadImage().load_image(str(reference_image))
        values = _node_result(result)
        if not values or values[0] is None:
            raise ValueError("reference image loader returned no image")
        return (values[0], mode_value)


class THSUniversalH3ReferenceConditioning:
    """Native MiniMax H3 ref2va conditioning with a safe OFF mode.

    ``without_reference`` calls the historical ImageToVideo path verbatim and
    never passes a reference image.  ``with_reference`` calls the installed
    ``MiniMaxH3ReferenceToVideo`` node with one labelled ``<Picture 1>`` image;
    the reference prompt explicitly scopes appearance changes to the user mask.
    ``corrected_frame`` intentionally stays on the native ImageToVideo path:
    its uploaded frame is consumed by :class:`THSUniversalCorrectedFrameGuide`
    as an arbitrary-frame AddGuide image, while this conditioner supplies only
    a generic correction-guide prompt (never ref2va or a source caption).
    The output tuple remains ``positive, negative, h3_latent`` so the source
    latent/noise-mask/composite branch is unchanged.
    """

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "h3_latent")
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/H3"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Restore only the user-masked pixels with natural texture and geometry inferred from adjacent unmasked frames.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_REFERENCE_NEGATIVE,
                    },
                ),
                "reference_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_REFERENCE_PROMPT,
                    },
                ),
                "width": ("INT", {"default": 768, "min": 32, "max": 16_384, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 16_384, "step": 32}),
                "length": ("INT", {"default": 22, "min": 5, "max": 3_600, "step": 17}),
                "mode": (
                    "COMBO",
                    {
                        "options": list(REFERENCE_MODES),
                        "default": "without_reference",
                         "tooltip": "without_reference preserves the original H3 conditioning; with_reference uses <Picture 1> only inside the user mask; corrected_frame uses a separate masked correction guide.",
                    },
                ),
                "ref_image_size": (
                    list(REFERENCE_IMAGE_SIZES),
                    {
                        "default": "match",
                        "tooltip": "match scales the reference to the generation area; max retains more identity detail but costs more tokens.",
                    },
                ),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                # Retain the source-images socket for graph compatibility and
                # future endpoint-anchor experiments; ref2va itself does not
                # use it for reference conditioning.
                "source_images": ("IMAGE",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, mode=None, ref_image_size=None, reference_image=None, **kwargs):
        try:
            mode_value = _normalise_reference_mode(mode)
        except ValueError as exc:
            return str(exc)
        size = str(ref_image_size or "match").strip().lower()
        if size not in REFERENCE_IMAGE_SIZES:
            return "ref_image_size must be match or max"
        # Linked IMAGE inputs are intentionally unresolved during Comfy's
        # preflight validation (``get_input_data`` presents them as ``None``).
        # The upload node validates the filename before it executes, while the
        # run method below gives a clear error for direct/malformed calls.
        return True

    def run(
        self,
        clip,
        vae,
        prompt,
        negative_prompt,
        width,
        height,
        length,
        mode="without_reference",
        ref_image_size="match",
        reference_prompt=None,
        reference_image=None,
        source_images=None,
    ):
        canvas_width, canvas_height = _h3_canvas_size(width, height)
        mode_value = _normalise_reference_mode(mode)
        size_value = str(ref_image_size or "match").strip().lower()
        if mode_value == "with_reference" and size_value not in REFERENCE_IMAGE_SIZES:
            raise ValueError("ref_image_size must be match or max")

        if mode_value in {"without_reference", "corrected_frame"}:
            # Keep this call byte-for-byte equivalent to the historical wrapper
            # (including explicit None endpoint anchors).  The corrected-frame
            # path deliberately ignores the selector image here: it is passed
            # only to THSUniversalCorrectedFrameGuide, which validates and
            # merges the guide at the requested original frame.
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo

            native_prompt = (
                str(prompt)
                if mode_value == "without_reference"
                else DEFAULT_CORRECTED_FRAME_PROMPT
            )
            result = MiniMaxH3ImageToVideo.execute(
                clip=clip,
                vae=vae,
                prompt=native_prompt,
                width=canvas_width,
                height=canvas_height,
                length=int(length),
                first_frame=None,
                last_frame=None,
            )
        else:
            if _reference_image_missing(reference_image):
                raise ValueError("reference_image is required when mode is with_reference")
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

            result = MiniMaxH3ReferenceToVideo.execute(
                clip=clip,
                # The reference path deliberately ignores the source/auto
                # caption.  A caption inferred from the source frame can
                # contradict the requested replacement, so only the explicit
                # reference prompt is sent to native ref2va.
                prompt=_reference_prompt(
                    str(reference_prompt or DEFAULT_REFERENCE_PROMPT)
                ),
                width=canvas_width,
                height=canvas_height,
                length=int(length),
                ref_image_size=size_value,
                vae=vae,
                ref_images={"ref_image_1": reference_image},
            )

        values = _node_result(result)
        if len(values) < 2:
            raise ValueError("MiniMax H3 conditioning returned no latent")
        positive, h3_latent = values[:2]
        negative = _encode_negative(clip, str(negative_prompt))
        return (positive, negative, h3_latent)


class THSUniversalH3PadSource:
    """Pad RGB/mask to the H3 token grid and RGB/mask/audio to the H3 timeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "model_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01, "tooltip": "MiniMax H3 internal timeline; output playback FPS remains the source video's FPS."}),
                "modulo": ("INT", {"default": 17, "min": 1, "max": 256}),
                "remainder": ("INT", {"default": 5, "min": 0, "max": 255}),
                "silence_sample_rate": ("INT", {"default": 48_000, "min": 8_000, "max": 192_000}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "prefill": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Заполнение из соседних пикселей (Telea). Отключите для замены человека или крупного объекта.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "INT", "INT")
    RETURN_NAMES = ("padded_images", "padded_mask", "padded_audio", "original_count", "padded_count")
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/Frames"

    def run(self, images, mask, model_fps=24.0, modulo=17, remainder=5, silence_sample_rate=48_000, audio=None, prefill=True):
        mask = _normalise_mask(mask)
        if tuple(mask.shape) != tuple(images.shape[:3]):
            raise ValueError("source images and mask must have the same frame count and canvas")
        # Prefill only the model-facing source.  The pure helper remains a
        # geometry/audio pad contract, while the final composite still receives
        # the untouched source IMAGE from GetVideoComponents.
        model_source = telea_prefill_masked(images, mask) if prefill else images
        prepared_images = _pad_h3_images(model_source)
        if tuple(prepared_images.shape[:3]) != tuple(mask.shape):
            _, height, width = mask.shape
            padded_mask = _zeros_like_shape(mask, tuple(prepared_images.shape[:3]))
            padded_mask[:, :height, :width] = mask
            mask = padded_mask
        return pad_source_inputs(
            prepared_images, mask, audio, model_fps, modulo, remainder, silence_sample_rate
        )


class THSUniversalH3PadAudio:
    """Pad only the waveform to H3's ``17*n+5`` timeline.

    The RGB/mask path remains owned by :class:`THSUniversalH3PadSource`; this
    node exists for repeat runs where source images and masks are already
    prepared and generating another padded image tensor would be wasteful.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "modulo": ("INT", {"default": 17, "min": 1, "max": 256}),
                "remainder": ("INT", {"default": 5, "min": 0, "max": 255}),
                "silence_sample_rate": ("INT", {"default": 48_000, "min": 8_000, "max": 192_000}),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("AUDIO", "INT", "INT")
    RETURN_NAMES = ("padded_audio", "original_count", "padded_count")
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/Frames"

    def run(
        self,
        images,
        model_fps=24.0,
        modulo=17,
        remainder=5,
        silence_sample_rate=48_000,
        audio=None,
    ):
        if not hasattr(images, "shape") or _ndim(images) < 1:
            raise ValueError("images must contain at least one frame")
        original_count = _batch_length(images)
        if original_count < 1:
            raise ValueError("images must contain at least one frame")
        padded_count = _next_h3_frame_count(original_count, modulo, remainder)
        padded_audio = _pad_audio_for_count(
            images,
            padded_count,
            audio,
            model_fps=model_fps,
            silence_sample_rate=silence_sample_rate,
        )
        return padded_audio, original_count, padded_count


class THSUniversalPromptText:
    """Frozen multiline prompt passthrough with no model/image dependencies."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/Repair"

    def run(self, text=""):
        # Do not strip or normalize: the UI worker copies the approved prompt
        # verbatim and the conditioning node should receive exact newlines.
        return (str(text),)


class THSUniversalRepairCompare:
    """Output-node preview of synchronized original/result frames.

    Full-resolution tensors are never placed in the websocket payload.  The
    node writes bounded JPEGs into the existing owned temp cache and returns
    ``/view`` descriptors with exact zero-based frame indexes for a UI wipe.
    """

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/Review"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "result": ("IMAGE",),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
            },
        }

    def run(self, source, result, fps=24.0):
        if not hasattr(source, "shape") or not hasattr(result, "shape"):
            raise ValueError("source and result must be IMAGE tensors/arrays")
        if tuple(source.shape[:3]) != tuple(result.shape[:3]):
            raise ValueError("source and result must have matching frame and canvas geometry")
        fps_value = float(fps or 24.0)
        if not math.isfinite(fps_value) or fps_value <= 0:
            raise ValueError("fps must be positive")
        source_np = source.detach().to(device="cpu").numpy() if _is_torch(source) else np.asarray(source)
        result_np = result.detach().to(device="cpu").numpy() if _is_torch(result) else np.asarray(result)
        # Import lazily so lightweight repair tests do not need Pillow or
        # ComfyUI's folder_paths module just to import this node package.
        from .mask_core import cache_compare_preview_assets

        descriptors, cache_info = cache_compare_preview_assets(
            source_np,
            result_np,
            source_session="repair-compare",
        )
        status = (
            f"Compare preview ready for {len(descriptors)} synchronized frame(s) at {fps_value:g} FPS."
            if cache_info.get("available")
            else f"Compare preview unavailable: {cache_info.get('error', 'unknown error')}"
        )
        meta = {
            "version": 1,
            "fps": fps_value,
            "frame_count": int(source_np.shape[0]),
            "width": int(source_np.shape[2]),
            "height": int(source_np.shape[1]),
            "cache_available": bool(cache_info.get("available")),
            "cache_session": cache_info.get("session"),
            "cache_error": cache_info.get("error"),
        }
        ui = {
            "universal_repair_compare_frames": descriptors,
            "universal_repair_compare_meta": [meta],
            "universal_repair_compare_status": [status],
        }
        return {"ui": ui, "result": ()}


class THSUniversalTrimFrames:
    """Remove only the synthetic H3 tail; real frame order/count is untouched."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "original_count": ("INT", {"default": 1, "min": 1})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/Frames"

    def run(self, images, original_count):
        count = int(original_count)
        if count < 1 or _batch_length(images) < count:
            raise ValueError(f"generated {_batch_length(images)} frames, expected at least {count}")
        return (images[:count],)


def composite_exact(source: Any, generated: Any, repair_mask: Any) -> Any:
    """Blend generated pixels only where ``repair_mask`` is non-zero.

    ``torch.where``/``numpy.where`` explicitly selects the source for zero-alpha
    pixels, so untouched RGB values remain bit-identical rather than merely
    mathematically equal after a multiply/add round trip.
    """

    if not hasattr(source, "shape") or not hasattr(generated, "shape"):
        raise ValueError("source and generated must be IMAGE tensors/arrays")
    if tuple(source.shape) != tuple(generated.shape):
        raise ValueError("source and generated images must have identical shape")
    mask = _normalise_mask(repair_mask)
    if _batch_length(mask) == 1 and _batch_length(source) != 1:
        if _is_torch(mask):
            mask = mask.expand(_batch_length(source), *mask.shape[1:])
        else:
            mask = np.repeat(mask, _batch_length(source), axis=0)
    if tuple(mask.shape) != tuple(source.shape[:3]):
        raise ValueError("repair mask must match [frames,height,width] of the images")

    if _is_torch(source):
        if not _is_torch(generated) or not _is_torch(mask):
            raise ValueError("source, generated and mask must use the same tensor backend")
        alpha = mask.to(device=source.device, dtype=source.dtype).clamp(0, 1)[..., None]
        blended = source * (1.0 - alpha) + generated * alpha
        return torch.where(alpha == 0, source, blended).to(dtype=source.dtype)
    source_np = np.asarray(source)
    generated_np = np.asarray(generated)
    alpha = np.asarray(mask, dtype=source_np.dtype).clip(0, 1)[..., None]
    blended = source_np * (1.0 - alpha) + generated_np * alpha
    return np.where(alpha == 0, source_np, blended).astype(source_np.dtype, copy=False)


class THSUniversalMaskedComposite:
    """Final pixel composite: outside the user repair mask, source wins exactly."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source": ("IMAGE",), "generated": ("IMAGE",), "repair_mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/Output"

    def run(self, source, generated, repair_mask):
        if _ndim(source) == _ndim(generated) == 4 and tuple(source.shape) != tuple(generated.shape):
            _, height, width, _ = source.shape
            canvas_width, canvas_height = _h3_canvas_size(width, height)
            if tuple(generated.shape[1:3]) == (canvas_height, canvas_width):
                generated = generated[:, :height, :width]
        return (composite_exact(source, generated, repair_mask),)


NODE_CLASS_MAPPINGS = {
    "THSUniversalH3RepairConditioning": THSUniversalH3RepairConditioning,
    "THSUniversalReferenceImage": THSUniversalReferenceImage,
    "THSUniversalH3ReferenceConditioning": THSUniversalH3ReferenceConditioning,
    "THSUniversalH3PadSource": THSUniversalH3PadSource,
    "THSUniversalH3PadAudio": THSUniversalH3PadAudio,
    "THSUniversalPromptText": THSUniversalPromptText,
    "THSUniversalRepairCompare": THSUniversalRepairCompare,
    "THSUniversalTrimFrames": THSUniversalTrimFrames,
    "THSUniversalMaskedComposite": THSUniversalMaskedComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "THSUniversalH3RepairConditioning": "THS Universal H3 Repair Conditioning",
    "THSUniversalReferenceImage": "THS Universal Reference Image (optional upload)",
    "THSUniversalH3ReferenceConditioning": "THS Universal H3 Reference Conditioning",
    "THSUniversalH3PadSource": "THS Universal H3 Pad Source (17n+5)",
    "THSUniversalH3PadAudio": "THS Universal H3 Pad Audio (17n+5)",
    "THSUniversalPromptText": "THS Universal Frozen Prompt Text",
    "THSUniversalRepairCompare": "THS Universal Repair Compare",
    "THSUniversalTrimFrames": "THS Universal Trim Synthetic Tail",
    "THSUniversalMaskedComposite": "THS Universal Exact Outside-Mask Composite",
}


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "THSUniversalH3RepairConditioning",
    "THSUniversalReferenceImage",
    "THSUniversalH3ReferenceConditioning",
    "THSUniversalH3PadSource",
    "THSUniversalH3PadAudio",
    "THSUniversalPromptText",
    "THSUniversalRepairCompare",
    "THSUniversalTrimFrames",
    "THSUniversalMaskedComposite",
    "composite_exact",
    "pad_source_inputs",
    "telea_prefill_masked",
]
