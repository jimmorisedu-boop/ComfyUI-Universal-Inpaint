"""Arbitrary-frame corrected-image guidance for the universal H3 workflow.

The node in this module is intentionally independent from ComfyUI imports at
module load time.  CPU/unit-test environments can exercise all validation and
mask-compositing branches without importing the native H3 implementation; the
native ``MiniMaxH3AddGuide`` class is imported only after a corrected-frame
request has passed validation.
"""

from __future__ import annotations

from typing import Any

try:  # Optional in the lightweight CPU harness.
    import numpy as np
except Exception:  # pragma: no cover - numpy is present in the test image.
    np = None  # type: ignore[assignment]

try:  # ComfyUI supplies torch; tests may use NumPy only.
    import torch
except Exception:  # pragma: no cover - exercised only on a no-torch machine.
    torch = None  # type: ignore[assignment]

from .repair_nodes import (
    CATEGORY,
    REFERENCE_MODES,
    _normalise_mask,
    _normalise_reference_mode,
    _pad_h3_images,
    composite_exact,
)


CORRECTED_FRAME_MODE = "corrected_frame"


def _is_torch(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _shape(value: Any) -> tuple[int, ...]:
    if not hasattr(value, "shape"):
        raise ValueError("expected an IMAGE/MASK tensor or array")
    return tuple(int(dimension) for dimension in value.shape)


def _finite(value: Any, name: str) -> None:
    """Reject NaN/Inf values before they reach VAE encoding."""

    try:
        if _is_torch(value):
            finite = torch.isfinite(value).all().item()
        else:
            if np is None:
                raise ValueError("NumPy is required for array validation")
            finite = np.isfinite(np.asarray(value)).all()
    except Exception as exc:
        raise ValueError(f"{name} must contain finite numeric values") from exc
    if not bool(finite):
        raise ValueError(f"{name} must contain finite numeric values")


def _has_nonzero(value: Any) -> bool:
    if _is_torch(value):
        return bool(torch.any(value != 0).item())
    if np is None:
        return bool(any(item != 0 for item in value))
    return bool(np.any(np.asarray(value) != 0))


def _node_result(result: Any) -> tuple[Any, ...]:
    values = getattr(result, "result", result)
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        return tuple(values)
    return (values,)


def _preview_first_frame(source_images: Any) -> Any:
    """Return a safe one-frame preview for bypassed modes.

    Bypass must not inspect the optional corrected upload.  The source image is
    required by the graph and is already available, so showing its first frame
    keeps the preview socket useful without validating or touching any guide
    data.
    """

    try:
        return source_images[:1]
    except Exception:
        return None


def _validate_corrected_frame(
    source_images: Any,
    mask: Any,
    corrected_image: Any,
    frame_number: Any,
) -> tuple[Any, Any, int]:
    """Validate and return the selected source frame/mask plus zero-based index."""

    source_shape = _shape(source_images)
    if len(source_shape) != 4:
        raise ValueError("source_images must have shape [frames,height,width,channels]")
    source_frames, source_height, source_width, source_channels = source_shape
    if source_frames < 1:
        raise ValueError("source_images must contain at least one original frame")

    mask_value = _normalise_mask(mask)
    mask_shape = _shape(mask_value)
    if len(mask_shape) != 3:
        raise ValueError("mask must have shape [frames,height,width]")
    if mask_shape != (source_frames, source_height, source_width):
        raise ValueError(
            "source_images and mask must have equal frame counts and matching canvas shapes"
        )

    corrected_shape = _shape(corrected_image)
    if corrected_shape != (1, source_height, source_width, source_channels):
        raise ValueError(
            "corrected_image must be exactly one frame with the same canvas and channels "
            "as source_images"
        )

    # Keep the original-frame contract explicit: frame_number is 1-based and
    # must never index the padded H3 timeline.
    try:
        numeric_frame = int(frame_number)
    except (TypeError, ValueError) as exc:
        raise ValueError("frame_number must be a 1-based integer") from exc
    try:
        if float(frame_number) != float(numeric_frame):
            raise ValueError("frame_number must be a 1-based integer")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc) == "frame_number must be a 1-based integer":
            raise
        raise ValueError("frame_number must be a 1-based integer") from exc
    if numeric_frame < 1 or numeric_frame > source_frames:
        raise ValueError(
            f"frame_number {numeric_frame} is outside original source frames 1..{source_frames}"
        )

    _finite(source_images, "source_images")
    _finite(mask_value, "mask")
    _finite(corrected_image, "corrected_image")
    frame_index = numeric_frame - 1
    selected_mask = mask_value[frame_index : frame_index + 1]
    if not _has_nonzero(selected_mask):
        raise ValueError(f"mask is empty for original frame {numeric_frame}")
    return source_images[frame_index : frame_index + 1], selected_mask, frame_index


class THSUniversalCorrectedFrameGuide:
    """Apply an uploaded corrected frame as an arbitrary H3 AddGuide anchor.

    ``source_images`` and ``mask`` are deliberately the original unpadded
    inputs.  The node validates the 1-based original frame number, composites
    only the selected frame inside its soft mask, and forwards that merged
    single-frame image to native ``MiniMaxH3AddGuide``.  Other selector modes
    return the incoming conditioning untouched without looking at the optional
    corrected image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "source_images": ("IMAGE",),
                "mask": ("MASK",),
                "mode": (
                    "COMBO",
                    {
                        "options": list(REFERENCE_MODES),
                        "default": "without_reference",
                    },
                ),
                "frame_number": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
            "optional": {"corrected_image": ("IMAGE",)},
        }

    RETURN_TYPES = ("CONDITIONING", "IMAGE", "STRING")
    RETURN_NAMES = ("positive", "guide_preview", "status")
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/Reference"

    def run(
        self,
        positive,
        latent,
        vae,
        source_images,
        mask,
        mode="without_reference",
        frame_number=1,
        corrected_image=None,
    ):
        mode_value = _normalise_reference_mode(mode)
        if mode_value != CORRECTED_FRAME_MODE:
            # Do not inspect corrected_image here.  In particular, an OFF run
            # with a missing/stale upload remains a valid reference-free path.
            return (
                positive,
                _preview_first_frame(source_images),
                f"corrected-frame guide bypassed (mode={mode_value})",
            )

        if corrected_image is None:
            raise ValueError("corrected_image is required when mode is corrected_frame")
        selected_source, selected_mask, frame_index = _validate_corrected_frame(
            source_images,
            mask,
            corrected_image,
            frame_number,
        )
        merged = composite_exact(selected_source, corrected_image, selected_mask)

        try:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide
        except Exception as exc:  # pragma: no cover - only a broken Comfy install.
            raise RuntimeError("ComfyUI MiniMaxH3AddGuide is unavailable") from exc

        result = MiniMaxH3AddGuide.execute(
            positive=positive,
            latent=latent,
            frame_idx=frame_index,
            vae=vae,
            image=_pad_h3_images(merged),
        )
        values = _node_result(result)
        if not values or values[0] is None:
            raise ValueError("MiniMaxH3AddGuide returned no positive conditioning")
        status = (
            f"corrected-frame guide applied to original frame {frame_index + 1} "
            f"(native frame_idx={frame_index})"
        )
        return values[0], merged, status


NODE_CLASS_MAPPINGS = {
    "THSUniversalCorrectedFrameGuide": THSUniversalCorrectedFrameGuide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "THSUniversalCorrectedFrameGuide": "THS Universal Corrected Frame Guide",
}


__all__ = [
    "CORRECTED_FRAME_MODE",
    "THSUniversalCorrectedFrameGuide",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
