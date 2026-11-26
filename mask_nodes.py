"""ComfyUI adapter for the self-contained universal video mask editor.

The node deliberately does not import torch or Comfy internals at module load
time.  That keeps the parser/test harness usable on CPU-only Python while the
actual Comfy execution still returns native IMAGE/MASK tensors.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np

from .mask_core import (
    MAX_FRAMES,
    MAX_PREVIEW_SIZE,
    MaskDataError,
    MaskRefinementError,
    MaskSourceMismatch,
    _apply_layer_blend,
    _compute_layer_raw,
    _data_url,
    _json_value,
    _mask_png_payload,
    build_mask_sequence,
    cache_preview_assets,
    canonical_mask_data,
    encode_preview_assets,
    frame_fingerprint,
    mask_extent,
    optical_flow_available,
    parse_mask_data,
    plan_crop_window,
    postprocess_mask,
    rasterize_events,
    raw_tracking_cache_info,
    restore_crop_window,
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


def _mask_array(mask: Any) -> np.ndarray:
    if hasattr(mask, "detach"):
        mask = mask.detach().to(device="cpu", dtype=getattr(mask, "dtype", None)).numpy()
    arr = np.asarray(mask)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim != 3:
        raise MaskDataError("mask must have shape T x H x W or H x W")
    return arr


def _normalize_images(images: np.ndarray) -> np.ndarray:
    source = np.asarray(images, dtype=np.float32)
    if source.ndim == 3:
        return source if np.isfinite(source).all() else np.nan_to_num(source, nan=0.0, posinf=1.0, neginf=0.0)
    invalid = [index for index, frame in enumerate(source) if not np.isfinite(frame).all()]
    if invalid:
        source = source.copy()
        for index in invalid:
            np.nan_to_num(source[index], copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    return source


def _torch_tensor(array: np.ndarray):
    import torch

    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def _preview_tensor(images: np.ndarray, mask: np.ndarray):
    """Make a lightweight green-overlay IMAGE preview without touching source."""

    if images.shape[-1] == 2 or images.shape[-1] < 1:
        raise MaskDataError("images must have one or at least three channels")
    output = np.empty((*mask.shape, 3), dtype=np.float32)
    for frame, frame_mask, destination in zip(images, mask, output):
        alpha = np.clip(frame_mask, 0.0, 1.0) * 0.38
        np.multiply(frame[..., :3], (1.0 - alpha)[..., None], out=destination)
        destination[..., 1] += alpha
    return _torch_tensor(output)


def _execution_blocker(message: str | None):
    # Comfy moved this helper once; keep the add-on importable in test Python
    # and block downstream graph work whenever the installed runtime supports it.
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

    blockers = tuple(_execution_blocker(None) for _ in range(6))
    if any(item is None for item in blockers):
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
        "result": blockers,
    }


def _source_signature(images: np.ndarray, fingerprint: str) -> str:
    count, height, width = map(int, images.shape[:3])
    # Dimensions/count intentionally define the session binding.  The digest
    # is stable, cheap, and does not copy or upload any pixel data.
    return hashlib.sha1(f"{count}:{height}:{width}:{fingerprint}".encode("ascii")).hexdigest()[:16]


def _active_layer_entry(parsed: dict[str, Any]) -> dict[str, Any] | None:
    layers = parsed.get("layers")
    if not isinstance(layers, list) or not layers:
        return None
    active_id = str(parsed.get("active_layer_id") or "")
    for layer in layers:
        if isinstance(layer, dict) and str(layer.get("id") or "") == active_id:
            return layer
    return layers[0] if isinstance(layers[0], dict) else None


def _sam3_frame_preview(source, document, *, fingerprint, fps, grow, feather, start_frame, end_frame):
    """Return the changed mask and edit base over the browser's cached source."""
    count, height, width = map(int, source.shape[:3])
    if count > MAX_FRAMES:
        raise MaskDataError("clips longer than 4096 frames are not supported by this editor")
    parsed = parse_mask_data(document, count, width=width, height=height, fingerprint=fingerprint)
    active_layer = _active_layer_entry(parsed)
    if active_layer is not None:
        if active_layer.get("type") != "sam3":
            return None
        sam3 = active_layer.get("sam3") if isinstance(active_layer.get("sam3"), dict) else {}
        layer_doc: dict[str, Any] = {
            "version": 1,
            "frames": active_layer.get("frames", {}),
            "tracking_mode": "sam3",
            "sam3": sam3,
        }
        if active_layer.get("anchors"):
            layer_doc["anchors"] = active_layer["anchors"]
        range_start = int(active_layer.get("start_frame", 0))
        range_end = int(active_layer.get("end_frame", -1))
    else:
        layer_doc = parsed
        sam3 = parsed.get("sam3", {})
        range_start = int(start_frame)
        range_end = int(end_frame)

    frame = sam3.get("frame")
    if sam3.get("action") != "frame" or frame is None or not _prompt_events(layer_doc, frame):
        return None
    blocker = _execution_blocker(None)
    if blocker is None:
        return None
    start = max(0, range_start)
    end = count - 1 if range_end < 0 else min(count - 1, range_end)
    if not start <= frame <= end:
        return None  # Use the ordinary range validation and recovery path.

    t0 = time.perf_counter()
    base, seed, sam3_meta = calculate_frame(source, layer_doc, frame=frame, interactive=True)
    prediction_ms = round((time.perf_counter() - t0) * 1000)
    base = base.astype(np.float32)
    sam3["seed_token"] = seed.token
    if isinstance(parsed.get("sam3"), dict):
        parsed["sam3"]["seed_token"] = seed.token

    norm_source = _normalize_images(source)
    events = _overlay_events(layer_doc, frame)
    drawn_active = rasterize_events(events, height, width, norm_source[frame], base) if events else base

    layers_preview_meta: list[dict[str, Any]] = []
    if isinstance(parsed.get("layers"), list):
        drawn = np.zeros((height, width), dtype=np.float32)
        composite_base = np.zeros((height, width), dtype=np.float32)
        active_id = str(active_layer.get("id") or "") if active_layer else ""
        for layer in parsed["layers"]:
            if not layer.get("enabled", True):
                layers_preview_meta.append({
                    "id": layer["id"],
                    "name": layer["name"],
                    "type": layer["type"],
                    "blend": layer["blend"],
                    "enabled": False,
                    "start_frame": int(layer.get("start_frame", 0)),
                    "end_frame": int(layer.get("end_frame", -1)),
                    "has_mask": False,
                    "anchors": [],
                })
                continue
            if str(layer["id"]) == active_id:
                l_raw_frame = drawn_active
                l_base_frame = base
                l_sam3 = sam3_meta
                l_has = bool(np.any(drawn_active > 1e-5))
                l_anchors = [int(frame)]
            else:
                o_start = max(0, int(layer.get("start_frame", 0)))
                o_end_raw = int(layer.get("end_frame", -1))
                o_end = count - 1 if o_end_raw < 0 else min(count - 1, o_end_raw)
                o_doc: dict[str, Any] = {
                    "version": 1,
                    "frames": layer.get("frames", {}),
                    "tracking_mode": layer["type"],
                }
                if layer.get("anchors"):
                    o_doc["anchors"] = layer["anchors"]
                if layer["type"] == "sam3" and isinstance(layer.get("sam3"), dict):
                    o_doc["sam3"] = layer["sam3"]
                # Skip untracked companion SAM3 layers so frame preview never triggers an extra native track.
                if layer["type"] == "sam3" and not (
                    isinstance(o_doc.get("sam3"), dict)
                    and o_doc["sam3"].get("action") == "track"
                    and o_doc["sam3"].get("seed_token")
                ):
                    l_raw_frame = np.zeros((height, width), dtype=np.float32)
                    l_sam3 = {}
                    l_anchors = []
                else:
                    try:
                        l_seq, _, l_anchors_set, l_sam3 = _compute_layer_raw(
                            norm_source, o_doc, tracking_mode=layer["type"], start=o_start, end=o_end
                        )
                        l_raw_frame = l_seq[frame]
                        l_anchors = sorted(int(f) for f in l_anchors_set)
                    except SAM3VideoError:
                        l_raw_frame = np.zeros((height, width), dtype=np.float32)
                        l_sam3 = {}
                        l_anchors = []
                l_base_frame = l_raw_frame
                l_has = bool(np.any(l_raw_frame > 1e-5))
            _apply_layer_blend(drawn, l_raw_frame, layer["blend"])
            _apply_layer_blend(composite_base, l_base_frame, layer["blend"])
            layers_preview_meta.append({
                "id": layer["id"],
                "name": layer["name"],
                "type": layer["type"],
                "blend": layer["blend"],
                "enabled": True,
                "start_frame": int(layer.get("start_frame", 0)),
                "end_frame": int(layer.get("end_frame", -1)),
                "has_mask": l_has,
                "anchors": l_anchors,
                **({"sam3": dict(l_sam3)} if layer["type"] == "sam3" and l_sam3 else {}),
            })
    else:
        drawn = drawn_active
        composite_base = base

    mask = postprocess_mask(drawn[None], grow=int(grow), feather=int(feather))[0]
    scale = min(1.0, MAX_PREVIEW_SIZE / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    patch = {
        "frame": frame,
        "mask": _data_url(_mask_png_payload(mask, size), "image/png"),
        "raw_mask": _data_url(_mask_png_payload(drawn, size), "image/png"),
        "edit_base": _data_url(_mask_png_payload(composite_base, size), "image/png"),
    }
    if seed.objects or getattr(seed, "refinement_masks", ()):
        from PIL import Image

        # Each preview pixel stores up to eight membership bits. This keeps
        # overlapping objects independently clickable with one small image.
        membership = np.zeros((size[1], size[0]), dtype=np.uint8)
        for index, (_, object_mask) in enumerate(seed.objects):
            obj_frame = seed.object_frames[index] if index < len(seed.object_frames) else seed.frame
            if int(obj_frame) != int(frame):
                continue
            thumbnail = Image.fromarray(object_mask).resize(size, Image.Resampling.NEAREST)
            membership |= np.asarray(thumbnail, dtype=np.uint8) << index
        for obj_id, ref_frame, ref_mask in getattr(seed, "refinement_masks", ()):
            if int(ref_frame) != int(frame):
                continue
            bit_index = max(0, int(obj_id) - 1)
            if bit_index >= 8:
                continue
            thumbnail = Image.fromarray(np.asarray(ref_mask, dtype=bool)).resize(size, Image.Resampling.NEAREST)
            membership |= np.asarray(thumbnail, dtype=np.uint8) << bit_index
        patch["object_map"] = _data_url(_mask_png_payload(membership / 255.0, size), "image/png")
    if sam3_meta.get("keep_existing_mask"):
        patch["keep_existing_mask"] = True
    bound_source = {"frames": count, "width": width, "height": height, "fingerprint": fingerprint}
    parsed["source"] = bound_source
    extent = mask_extent(mask[None])
    if extent is not None:
        extent = {"box": extent["box"], "frames": [int(frame)]}
    raw_extent = mask_extent(drawn[None])
    if raw_extent is not None:
        raw_extent = {"box": raw_extent["box"], "frames": [int(frame)]}
    applied_postprocess = {
        "grow": int(max(0, min(256, int(grow)))),
        "feather": int(max(0, min(256, int(feather)))),
    }
    crop_plan = plan_crop_window((count, height, width), extent, parsed.get("crop"))
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
        "crop_extent": extent,
        "raw_crop_extent": raw_extent,
        "mask_postprocess": applied_postprocess,
        "crop_plan": crop_plan,
    }
    if layers_preview_meta:
        meta["layers"] = layers_preview_meta
        meta["active_layer_id"] = str(parsed.get("active_layer_id") or layers_preview_meta[0]["id"])
    return {
        "ui": {
            "universal_mask_frame_patch": [patch],
            "universal_mask_meta": [meta],
            "universal_mask_data": [canonical_mask_data(parsed, count, width=width, height=height, fingerprint=fingerprint)],
        },
        "result": (blocker, blocker, blocker, blocker, blocker, blocker),
    }


def _opencv_refinement_available() -> bool:
    try:
        from . import mask_core as core

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
    """Paint or select any object/region and track its mask through a video clip.

    Empty ``mask_data`` is a preview-only pass: the node returns bounded frame
    previews in ``ui`` and blocks downstream graph work when Comfy's
    ``ExecutionBlocker`` is available.  Queue again after creating the mask
    and the generated mask is released to downstream workflow nodes.
    """

    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "DAI/Universal Mask Editor"
    SEARCH_ALIASES = [
        "dai",
        "dai mask",
        "dai video mask",
        "dai mask editor",
        "video mask editor",
        "universal mask editor",
    ]
    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE", "MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("edit_mask", "preview", "cropped_images", "cropped_mask", "crop_info", "original_images")

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
                "preview_output": ("BOOLEAN", {"default": True, "tooltip": "Формировать видео с зелёной маской для выхода preview. Предпросмотр внутри редактора работает и без этого выхода."}),
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
            start_frame=0, end_frame=-1, fps=24.0, unique_id=None, preview_output=True):
        del unique_id  # token isolation uses the per-node output session signature
        source = _image_array(images)
        count, height, width = map(int, source.shape[:3])
        fingerprint = frame_fingerprint(source)
        session = _source_signature(source, fingerprint)
        editor_frame_preview = False
        try:
            wants_sam3_preview = (
                str(tracking_mode) == "sam3"
                or (isinstance(mask_data, str) and SAM3_FRAME_PREVIEW in mask_data)
                or (isinstance(mask_data, dict) and mask_data.get("editor_request") == SAM3_FRAME_PREVIEW)
            )
            if wants_sam3_preview:
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
            mask, parsed, anchors, edit_bases, sam3_meta, raw_mask, layers_meta = build_mask_sequence(
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
                return_raw_mask=True,
                return_layers_meta=True,
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
        if str(tracking_mode) == "sam3" or ("layers" in bound_document and sam3_meta):
            if "sam3" in bound_document or str(tracking_mode) == "sam3":
                bound_document["sam3"] = dict(bound_document.get("sam3", {}))
                # A frame calculation returns the generated token through meta;
                # include it in the canonical document so a queue/reload can carry
                # the exact source-bound prerequisite into the explicit Track step.
                if sam3_meta.get("seed_token"):
                    bound_document["sam3"]["seed_token"] = str(sam3_meta["seed_token"])
        canonical = canonical_mask_data(bound_document, count, width=width, height=height, fingerprint=fingerprint)
        has_mask = any(float(frame.max()) > 1e-5 for frame in mask)
        lazy_frames, cache_info = cache_preview_assets(
            source,
            mask,
            edit_bases=edit_bases,
            raw_masks=raw_mask,
            source_session=session,
        )
        preview_kwargs: dict[str, Any] = {}
        if not cache_info.get("available"):
            preview_kwargs = {"edit_bases": edit_bases, "raw_masks": raw_mask}
        previews = encode_preview_assets(source, mask, **preview_kwargs)
        raw_cache_info = raw_tracking_cache_info()
        preview = _preview_tensor(source, mask) if preview_output else _execution_blocker(None)
        extent = mask_extent(mask)
        raw_extent = mask_extent(raw_mask)
        applied_postprocess = {
            "grow": int(max(0, min(256, int(grow)))),
            "feather": int(max(0, min(256, int(feather)))),
        }
        crop_plan = plan_crop_window(mask.shape, extent, parsed.get("crop"))
        crop_x, crop_y, crop_r, crop_b = crop_plan["box"]
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
            "crop_extent": extent,
            "raw_crop_extent": raw_extent,
            "mask_postprocess": applied_postprocess,
            "crop_plan": crop_plan,
        }
        if layers_meta:
            meta["layers"] = layers_meta
            meta["active_layer_id"] = str(parsed.get("active_layer_id") or layers_meta[0]["id"])
        untracked_sam3 = False
        if layers_meta and isinstance(parsed.get("layers"), list):
            layer_meta_map = {str(item.get("id")): item for item in layers_meta}
            for layer in parsed["layers"]:
                if not layer.get("enabled", True) or layer.get("type") != "sam3":
                    continue
                l_sam3_cfg = layer.get("sam3") if isinstance(layer.get("sam3"), dict) else {}
                l_doc = {
                    "version": 1,
                    "frames": layer.get("frames", {}),
                    **({"sam3": l_sam3_cfg} if l_sam3_cfg else {}),
                }
                l_frame = int(l_sam3_cfg.get("frame", max(0, int(layer.get("start_frame", 0)))))
                has_prompt = bool(_prompt_events(l_doc, l_frame))
                l_track_ready = bool(
                    layer_meta_map.get(str(layer.get("id")), {}).get("sam3", {}).get("track_ready")
                )
                if has_prompt and not l_track_ready:
                    untracked_sam3 = True
                    break
        else:
            untracked_sam3 = str(tracking_mode) == "sam3" and not bool(sam3_meta.get("track_ready"))

        if str(tracking_mode) == "sam3" or sam3_meta:
            meta["sam3"] = dict(sam3_meta)
        if str(tracking_mode) == "sam3" and not (layers_meta and has_mask and not untracked_sam3):
            if not bool(sam3_meta.get("frame_ready")):
                status = "SAM3 seed frame is not ready. Select a marked object box and Calculate frame."
            elif not bool(sam3_meta.get("track_ready")):
                status = "SAM3 frame ready — press Track video before running downstream nodes."
            elif has_mask:
                status = "SAM3 video mask ready — queue again to run downstream nodes."
            else:
                status = "SAM3 tracking produced an empty selected-object mask; review the range and prompt."
        elif untracked_sam3:
            status = "SAM3 layer is waiting for Track video before running downstream nodes."
        elif str(tracking_mode) == "optical_flow" and not optical_flow_available():
            status = "OpenCV optical flow is unavailable; unanchored frames hold the nearest anchor. Verify preview before running the graph."
        elif has_mask:
            status = "Mask ready — queue again to run downstream nodes."
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
        cropped_images_out: Any = _torch_tensor(source[:, crop_y:crop_b, crop_x:crop_r])
        cropped_mask_out: Any = _torch_tensor(mask[:, crop_y:crop_b, crop_x:crop_r])
        crop_info_out: Any = json.dumps(crop_plan, ensure_ascii=False)
        original_images_out: Any = images if hasattr(images, "detach") else _torch_tensor(source)
        if not has_mask or untracked_sam3:
            blocker = _execution_blocker(
                "THSUniversalMaskEditor is waiting for a completed SAM3 video track or a non-empty mask."
                if (str(tracking_mode) == "sam3" or untracked_sam3)
                else "THSUniversalMaskEditor is waiting for a non-empty mask. Paint/shape the target and queue again."
            )
            if blocker is not None:
                mask_output = blocker
                cropped_images_out = blocker
                cropped_mask_out = blocker
                crop_info_out = blocker
                original_images_out = blocker
        return {"ui": ui, "result": (mask_output, preview, cropped_images_out, cropped_mask_out, crop_info_out, original_images_out)}


class THSUniversalMaskRestoreCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "cropped_images": ("IMAGE",),
                "crop_info": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "mask": ("MASK",),
                "edge_feather": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "resize_method": (["lanczos", "bicubic", "bilinear", "area", "nearest"], {"default": "lanczos"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "color_match": (["none", "mean_std"], {"default": "none"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "run"
    CATEGORY = "DAI/Universal Mask Editor"
    SEARCH_ALIASES = [
        "dai",
        "dai restore",
        "dai crop",
        "dai restore crop",
        "video mask restore crop",
        "restore crop",
        "stitch crop",
    ]
    OUTPUT_NODE = False

    def run(
        self,
        original_images: Any,
        cropped_images: Any,
        crop_info: str,
        mask: Any = None,
        edge_feather: int = 0,
        resize_method: str = "lanczos",
        invert_mask: bool = False,
        color_match: str = "none",
    ):
        orig = _normalize_images(_image_array(original_images))
        crop = _normalize_images(_image_array(cropped_images))
        mask_arr = _normalize_images(_mask_array(mask)) if mask is not None else None
        feather = 0 if edge_feather is None else int(edge_feather)
        restored, applied_mask = restore_crop_window(
            orig,
            crop,
            crop_info,
            mask=mask_arr,
            edge_feather=feather,
            resize_method=resize_method,
            invert_mask=bool(invert_mask),
            color_match=color_match,
        )
        return (_torch_tensor(restored), _torch_tensor(applied_mask))


NODE_CLASS_MAPPINGS = {
    "THSUniversalMaskEditor": THSUniversalMaskEditor,
    "THSUniversalMaskRestoreCrop": THSUniversalMaskRestoreCrop,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "THSUniversalMaskEditor": "DAI Video Mask Editor",
    "THSUniversalMaskRestoreCrop": "DAI Video Mask Restore Crop",
}


__all__ = [
    "THSUniversalMaskEditor",
    "THSUniversalMaskRestoreCrop",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
