"""ComfyUI adapter for the self-contained universal video mask editor.

The node deliberately does not import torch or Comfy internals at module load
time.  That keeps the parser/test harness usable on CPU-only Python while the
actual Comfy execution still returns native IMAGE/MASK tensors.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from .mask_core import (
    MAX_FRAMES,
    MAX_PREVIEW_SIZE,
    MaskDataError,
    MaskRefinementError,
    MaskSourceMismatch,
    _data_url,
    _json_value,
    _mask_png_payload,
    build_mask_sequence,
    cache_preview_assets,
    canonical_mask_data,
    encode_preview_assets,
    frame_fingerprint,
    optical_flow_available,
    parse_mask_data,
    postprocess_mask,
    rasterize_events,
    raw_tracking_cache_info,
)
from .mask_sam3_video import SAM3VideoError, _overlay_events, _prompt_events, calculate_frame


SAM3_FRAME_PREVIEW = "sam3_frame_v1"


def _image_array(images: Any) -> np.ndarray:
    if hasattr(images, "detach"):
        images = images.detach().to(device="cpu", dtype=getattr(images, "dtype", None)).numpy()
    images = np.asarray(images)
    if images.ndim != 4:
        raise MaskDataError("images must have shape T x H x W x C")
    return images


def _normalize_images(images: np.ndarray) -> np.ndarray:
    return np.nan_to_num(images.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)


def _torch_tensor(array: np.ndarray):
    import torch

    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())


def _preview_tensor(images: np.ndarray, mask: np.ndarray):
    """Make a lightweight green-overlay IMAGE preview without touching source."""

    image = np.asarray(images, dtype=np.float32).copy()
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] < 3:
        raise MaskDataError("images must have one or at least three channels")
    image = image[..., :3]
    alpha = np.clip(mask[..., None], 0.0, 1.0) * 0.38
    green = np.zeros_like(image)
    green[..., 1] = 1.0
    return _torch_tensor(image * (1.0 - alpha) + green * alpha)


def _execution_blocker(message: str | None):
    # Comfy moved this helper once; keep the add-on importable in test Python
    # and block downstream H3 work whenever the installed runtime supports it.
    for module_name in ("comfy_execution.graph_utils", "comfy_execution.graph"):
        try:
            module = __import__(module_name, fromlist=["ExecutionBlocker"])
            blocker = getattr(module, "ExecutionBlocker", None)
            if blocker is not None:
                return blocker(message)
        except Exception:
            continue
    return None


def _blocked_source_response(
    source: np.ndarray,
    *,
    session: str,
    fingerprint: str,
    fps: float,
    tracking_mode: str,
    start_frame: int,
    end_frame: int,
    diagnostic: MaskSourceMismatch | MaskRefinementError | SAM3VideoError,
):
    """Return a source-only preview while stopping every downstream output.

    This path is intentionally shared by stale-source documents and failed
    rectangle refinement.  It never canonicalizes or rasterizes the submitted
    document, never writes edit-base descriptors, and only succeeds when the
    installed Comfy runtime provides ``ExecutionBlocker``.
    """

    mask_blocker = _execution_blocker(None)
    preview_blocker = _execution_blocker(None)
    if mask_blocker is None or preview_blocker is None:
        raise diagnostic

    count, height, width = map(int, source.shape[:3])
    # A read-only broadcasted scalar keeps mismatch/refinement recovery bounded
    # even for long 1080p/4K clips; preview encoders only read the mask.
    empty_mask = np.broadcast_to(
        np.asarray(0.0, dtype=np.float32),
        (count, height, width),
    )
    previews = encode_preview_assets(source, empty_mask)
    lazy_frames, cache_info = cache_preview_assets(
        source,
        empty_mask,
        source_session=session,
    )
    meta = {
        "version": 1,
        "session": session,
        "source": {
            "frames": count,
            "width": width,
            "height": height,
            "fingerprint": fingerprint,
        },
        "fps": float(fps or 24.0),
        "tracking_mode": str(tracking_mode),
        "start_frame": max(0, int(start_frame)),
        "end_frame": count - 1 if int(end_frame) < 0 else min(count - 1, int(end_frame)),
        "anchors": [],
        "has_mask": False,
        "preview_frames": len(previews),
        "all_frame_cache_available": bool(cache_info.get("available")),
        "all_frames_available": bool(cache_info.get("available")),
        "all_frame_cache_frames": len(lazy_frames),
        "all_frame_cache_session": cache_info.get("session"),
        "all_frame_cache_error": cache_info.get("error"),
        "tracking_available": bool(optical_flow_available()),
        "raw_tracking_cache": raw_tracking_cache_info(),
        "refinement_methods": _refinement_capabilities(),
    }
    if isinstance(diagnostic, MaskSourceMismatch):
        meta.update({
            "source_mismatch": True,
            "source_mismatch_reason": diagnostic.reason,
            "previous_source": dict(diagnostic.previous_source),
        })
        status = (
            "Mask source changed; review the source-only preview and rebind or "
            "discard the existing edits before queueing again."
        )
    elif isinstance(diagnostic, SAM3VideoError):
        sam3_meta = dict(getattr(diagnostic, "sam3_meta", {}) or {})
        if not sam3_meta:
            sam3_meta = {
                "frame_ready": False,
                "track_ready": False,
                "frame": max(0, int(start_frame)),
                "seed_token": "",
                "direction": "forward",
                "computed_frames": [],
            }
        sam3_meta["error"] = str(diagnostic)
        meta["sam3"] = sam3_meta
        meta["sam3_error"] = str(diagnostic)
        status = f"SAM3 video is blocked: {diagnostic}"
    else:
        meta["refinement_error"] = str(diagnostic)
        status = (
            "Rectangle refinement failed; review the source-only preview and "
            "adjust the rectangle before queueing again."
        )
    if not cache_info.get("available"):
        status += f" All-frame lazy preview unavailable: {cache_info.get('error', 'unknown error')}"
    return {
        "ui": {
            "universal_mask_preview": previews,
            "universal_mask_frames": lazy_frames,
            "universal_mask_meta": [meta],
            "universal_mask_status": [status],
        },
        "result": (mask_blocker, preview_blocker),
    }


def _source_signature(images: np.ndarray, fingerprint: str) -> str:
    count, height, width = map(int, images.shape[:3])
    # Dimensions/count intentionally define the session binding.  The digest
    # is stable, cheap, and does not copy or upload any pixel data.
    return hashlib.sha1(f"{count}:{height}:{width}:{fingerprint}".encode("ascii")).hexdigest()[:16]


def _sam3_frame_preview(source, document, *, fingerprint, fps, grow, feather, start_frame, end_frame):
    """Return the changed mask and edit base over the browser's cached source."""
    count, height, width = map(int, source.shape[:3])
    if count > MAX_FRAMES:
        raise MaskDataError("clips longer than 4096 frames are not supported by this editor")
    parsed = parse_mask_data(document, count, width=width, height=height, fingerprint=fingerprint)
    sam3 = parsed.get("sam3", {})
    frame = sam3.get("frame")
    if sam3.get("action") != "frame" or frame is None or not _prompt_events(parsed, frame):
        return None
    blocker = _execution_blocker(None)
    if blocker is None:
        return None
    start = max(0, int(start_frame))
    end = count - 1 if int(end_frame) < 0 else min(count - 1, int(end_frame))
    if not start <= frame <= end:
        return None  # Use the ordinary range validation and recovery path.

    t0 = time.perf_counter()
    base, seed, sam3_meta = calculate_frame(source, parsed, frame=frame, interactive=True)
    prediction_ms = round((time.perf_counter() - t0) * 1000)
    base = base.astype(np.float32)
    events = _overlay_events(parsed, frame)
    drawn = rasterize_events(events, height, width, _normalize_images(source[frame]), base) if events else base
    mask = postprocess_mask(drawn[None], grow=int(grow), feather=int(feather))[0]
    scale = min(1.0, MAX_PREVIEW_SIZE / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    patch = {
        "frame": frame,
        "mask": _data_url(_mask_png_payload(mask, size), "image/png"),
        "edit_base": _data_url(_mask_png_payload(base, size), "image/png"),
    }
    bound_source = {"frames": count, "width": width, "height": height, "fingerprint": fingerprint}
    parsed["source"] = bound_source
    sam3["seed_token"] = seed.token
    meta = {
        "version": 1,
        "source": bound_source,
        "fps": float(fps or 24.0),
        "tracking_mode": "sam3",
        "start_frame": start,
        "end_frame": end,
        "anchors": [frame],
        "has_mask": bool(np.any(mask > 1e-5)),
        "sam3": sam3_meta,
        "frame_preview_ms": prediction_ms,
    }
    return {
        "ui": {
            "universal_mask_frame_patch": [patch],
            "universal_mask_meta": [meta],
            "universal_mask_data": [canonical_mask_data(parsed, count, width=width, height=height, fingerprint=fingerprint)],
        },
        "result": (blocker, blocker),
    }


def _opencv_refinement_available() -> bool:
    try:
        import ths_universal_inpaint.mask_core as core

        return core.cv2 is not None
    except Exception:
        return False


def _refinement_capabilities() -> dict[str, Any]:
    """Return frontend-facing refinement capability metadata without loading SAM3."""

    grabcut_error = None if _opencv_refinement_available() else "OpenCV is unavailable"
    try:
        from .mask_sam3 import sam3_availability

        sam3 = dict(sam3_availability())
    except Exception as exc:
        sam3 = {
            "available": False,
            "error": f"SAM3: capability check failed: {type(exc).__name__}: {exc}",
            "adapter": "comfyui-sam3",
        }
    return {
        "grabcut": {"available": grabcut_error is None, "error": grabcut_error},
        "sam3": sam3,
    }


class THSUniversalMaskEditor:
    """Paint/shape any inpaint area and track it through a video clip.

    Empty ``mask_data`` is a preview-only pass: the node returns bounded frame
    previews in ``ui`` and blocks downstream graph work when Comfy's
    ``ExecutionBlocker`` is available.  Queue again after drawing and the
    generated mask is released to the rest of the H3 workflow.
    """

    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Universal Inpaint/Masking"
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("edit_mask", "preview")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask_data": ("STRING", {"default": "{}", "multiline": True}),
                "tracking_mode": (["optical_flow", "static", "keyframes", "sam3"], {"default": "sam3"}),
                "grow": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1}),
                "feather": ("INT", {"default": 4, "min": 0, "max": 256, "step": 1}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": 4096, "step": 1}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, tracking_mode=None, **kwargs):
        if tracking_mode in (None, "", "optical_flow", "static", "keyframes", "sam3"):
            return True
        return "tracking_mode must be optical_flow, static, keyframes, or sam3"

    def run(self, images, mask_data="{}", tracking_mode="optical_flow", grow=8, feather=4,
            start_frame=0, end_frame=-1, fps=24.0, unique_id=None):
        del unique_id  # token isolation uses the per-node output session signature
        source = _image_array(images)
        count, height, width = map(int, source.shape[:3])
        fingerprint = frame_fingerprint(source)
        session = _source_signature(source, fingerprint)
        editor_frame_preview = False
        try:
            if str(tracking_mode) == "sam3":
                mask_data = _json_value(mask_data)
                editor_frame_preview = mask_data.get("editor_request") == SAM3_FRAME_PREVIEW
                if editor_frame_preview:
                    response = _sam3_frame_preview(
                        source, mask_data, fingerprint=fingerprint, fps=fps, grow=grow, feather=feather,
                        start_frame=start_frame, end_frame=end_frame,
                    )
                    if response is not None:
                        return response
            source = _normalize_images(source)
            mask, parsed, anchors, edit_bases, sam3_meta = build_mask_sequence(
                source,
                mask_data,
                tracking_mode=tracking_mode,
                grow=int(grow),
                feather=int(feather),
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                fingerprint=fingerprint,
                return_edit_bases=True,
                return_sam3_meta=True,
            )
        except (MaskSourceMismatch, MaskRefinementError, SAM3VideoError) as exc:
            if editor_frame_preview and not isinstance(exc, MaskSourceMismatch):
                # The editor already has the source. Let its scoped execution
                # error handler clear readiness without rebuilding the clip.
                raise
            if editor_frame_preview:
                source = _normalize_images(source)
            return _blocked_source_response(
                source,
                session=session,
                fingerprint=fingerprint,
                fps=float(fps or 24.0),
                tracking_mode=str(tracking_mode),
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                diagnostic=exc,
            )
        bound_document = dict(parsed)
        bound_document["source"] = {
            "frames": count,
            "width": width,
            "height": height,
            "fingerprint": fingerprint,
        }
        if str(tracking_mode) == "sam3":
            bound_document["sam3"] = dict(bound_document.get("sam3", {}))
            # A frame calculation returns the generated token through meta;
            # include it in the canonical document so a queue/reload can carry
            # the exact source-bound prerequisite into the explicit Track step.
            if sam3_meta.get("seed_token"):
                bound_document["sam3"]["seed_token"] = str(sam3_meta["seed_token"])
        canonical = canonical_mask_data(bound_document, count, width=width, height=height, fingerprint=fingerprint)
        has_mask = bool(np.any(mask > 1e-5))
        previews = encode_preview_assets(source, mask)
        lazy_frames, cache_info = cache_preview_assets(
            source,
            mask,
            edit_bases=edit_bases,
            source_session=session,
        )
        raw_cache_info = raw_tracking_cache_info()
        preview = _preview_tensor(source, mask)
        meta = {
            "version": 1,
            "session": session,
            "source": {"frames": count, "width": width, "height": height, "fingerprint": fingerprint},
            "fps": float(fps or 24.0),
            "tracking_mode": str(tracking_mode),
            "start_frame": max(0, int(start_frame)),
            "end_frame": count - 1 if int(end_frame) < 0 else min(count - 1, int(end_frame)),
            "anchors": sorted(int(frame) for frame in anchors),
            "has_mask": has_mask,
            "preview_frames": len(previews),
            "all_frame_cache_available": bool(cache_info.get("available")),
            # Keep a short alias for clients that describe the capability as
            # ``all_frames_available``; both fields are intentionally emitted
            # during the transition to lazy /view descriptors.
            "all_frames_available": bool(cache_info.get("available")),
            "all_frame_cache_frames": len(lazy_frames),
            "all_frame_cache_session": cache_info.get("session"),
            "all_frame_cache_error": cache_info.get("error"),
            "tracking_available": bool(optical_flow_available()),
            "raw_tracking_cache": raw_cache_info,
            "refinement_methods": _refinement_capabilities(),
        }
        if str(tracking_mode) == "sam3":
            meta["sam3"] = dict(sam3_meta)
        if str(tracking_mode) == "sam3":
            if not bool(sam3_meta.get("frame_ready")):
                status = "SAM3 seed frame is not ready. Select a marked object box and Calculate frame."
            elif not bool(sam3_meta.get("track_ready")):
                status = "SAM3 frame ready — press Track video before running downstream H3."
            elif has_mask:
                status = "SAM3 video mask ready — queue again to run the downstream H3 graph."
            else:
                status = "SAM3 tracking produced an empty selected-object mask; review the range and prompt."
        elif str(tracking_mode) == "optical_flow" and not optical_flow_available():
            status = "OpenCV optical flow is unavailable; unanchored frames hold the nearest anchor. Verify preview before H3."
        elif has_mask:
            status = "Mask ready — queue again to run the downstream H3 graph."
        else:
            status = "Preview prepared. Paint a region, then Queue again; empty masks stay blocked."
        if not cache_info.get("available"):
            status += f" All-frame lazy preview unavailable: {cache_info.get('error', 'unknown error')}"
        ui = {
            "universal_mask_preview": previews,
            "universal_mask_frames": lazy_frames,
            "universal_mask_meta": [meta],
            "universal_mask_data": [canonical],
            "universal_mask_status": [status],
        }
        mask_output: Any = _torch_tensor(mask)
        if not has_mask or (str(tracking_mode) == "sam3" and not bool(sam3_meta.get("track_ready"))):
            blocker = _execution_blocker(
                "THSUniversalMaskEditor is waiting for a completed SAM3 video track or a non-empty mask."
                if str(tracking_mode) == "sam3"
                else "THSUniversalMaskEditor is waiting for a non-empty mask. Paint/shape the target and queue again."
            )
            if blocker is not None:
                mask_output = blocker
        return {"ui": ui, "result": (mask_output, preview)}


NODE_CLASS_MAPPINGS = {"THSUniversalMaskEditor": THSUniversalMaskEditor}
NODE_DISPLAY_NAME_MAPPINGS = {"THSUniversalMaskEditor": "Universal Video Mask Editor"}


__all__ = ["THSUniversalMaskEditor", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
