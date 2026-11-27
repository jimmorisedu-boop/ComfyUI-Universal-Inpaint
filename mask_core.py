"""Pure-numpy mask editing and short-video tracking helpers.

The ComfyUI node is intentionally a thin adapter around this module.  Keeping
the edit log parser/rasterizer independent of torch makes it possible to run
the important safety and tracking tests on a normal Python installation and
also keeps the browser contract easy to reason about.

Coordinates in the edit log are always normalized ``[0, 1]`` values.  The
canonical document shape is::

    {
      "version": 1,
      "source": {"frames": 12, "width": 1920, "height": 1080},
      "frames": {
        "0": {
          "strokes": [{"mode":"add", "radius":0.025,
                       "points":[[.2,.3],[.25,.34]]}],
          "rects": [{"mode":"add", "x":.2, "y":.2, "w":.3, "h":.4,
                       "refine": false, "refine_method": "grabcut"}],
          "polygons": [{"mode":"erase", "points":[[...], ...]}]
        }
      },
      "anchors": [0],
      "tracking_mode": "optical_flow"
    }

Only the edit log is serialized by the widget.  Raw frames and generated
masks never travel through ``mask_data``.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import re
import shutil
import threading
import uuid
import zlib
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # OpenCV is present in the Comfy environment, but keep import optional.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised on minimal CPU installs
    cv2 = None


MAX_FRAMES = 4096
MAX_EDIT_FRAMES = 4096
MAX_EVENTS = 4096
MAX_POINTS = 65536
MAX_POINTS_PER_EVENT = 4096
MAX_SAM3_OBJECTS = 8
MAX_SAM3_OBJECT_POINTS = 64
MAX_SAM3_REFINEMENT_FRAMES = 32
MAX_MASK_LAYERS = 16
_LAYER_TYPES = ("sam3", "optical_flow", "keyframes", "static")
_LAYER_BLENDS = ("add", "subtract", "intersect")
_LAYER_TYPE_NAMES = {
    "sam3": "SAM3",
    "optical_flow": "Optical Flow",
    "keyframes": "Ключи",
    "static": "Статика",
}
MAX_PREVIEW_FRAMES = 48
MAX_PREVIEW_SIZE = 640
MAX_FLOW_DIM = 960
MAX_GRABCUT_DIM = 960
GRABCUT_ITERATIONS = 5
# Raw tracking results can be much larger than the browser's small preview
# payloads (a 120-frame 768x768 mask pair is hundreds of MiB decoded).  Keep
# immutable, lossless compressed entries in a byte-bounded LRU and bypass an
# item when its compressed representation is still too large.  The decoded
# arrays are never retained by the cache.
MAX_RAW_TRACKING_CACHE_ENTRIES = 8
MAX_RAW_TRACKING_CACHE_BYTES = 256 * 1024 * 1024
MAX_RAW_TRACKING_CACHE_ITEM_BYTES = 128 * 1024 * 1024
MAX_RAW_TRACKING_CACHE_SOURCE_BYTES = 2 * 1024 * 1024 * 1024
# Lazy previews are written below Comfy's temporary directory.  The cache is
# deliberately bounded and cleanup only ever considers directories created by
# this module (see ``cache_preview_assets``).
MAX_TEMP_CACHE_BYTES = 256 * 1024 * 1024
MAX_TEMP_CACHE_SESSIONS = 8
MAX_TEMP_CACHE_SESSION_BYTES = 128 * 1024 * 1024
TEMP_CACHE_ROOT = "ths_universal_mask"
TEMP_CACHE_PREFIX = "session-"


_raw_tracking_cache: "OrderedDict[tuple[Any, ...], tuple[bytes, tuple[int, ...], str, bytes, tuple[int, ...], str, tuple[int, ...], int]]" = OrderedDict()
_raw_tracking_cache_lock = threading.RLock()
_raw_tracking_cache_bytes = 0
_raw_tracking_cache_hits = 0
_raw_tracking_cache_misses = 0
_raw_tracking_cache_bypasses = 0
_raw_tracking_cache_stores = 0


def clear_raw_tracking_cache() -> None:
    """Clear the bounded raw tracking cache and its diagnostic counters."""

    global _raw_tracking_cache_bytes
    global _raw_tracking_cache_hits, _raw_tracking_cache_misses
    global _raw_tracking_cache_bypasses, _raw_tracking_cache_stores
    with _raw_tracking_cache_lock:
        _raw_tracking_cache.clear()
        _raw_tracking_cache_bytes = 0
        _raw_tracking_cache_hits = 0
        _raw_tracking_cache_misses = 0
        _raw_tracking_cache_bypasses = 0
        _raw_tracking_cache_stores = 0


def raw_tracking_cache_info() -> dict[str, Any]:
    """Return bounded cache counters suitable for node diagnostics/tests."""

    with _raw_tracking_cache_lock:
        return {
            "entries": len(_raw_tracking_cache),
            "bytes": int(_raw_tracking_cache_bytes),
            "hits": int(_raw_tracking_cache_hits),
            "misses": int(_raw_tracking_cache_misses),
            "bypasses": int(_raw_tracking_cache_bypasses),
            "stores": int(_raw_tracking_cache_stores),
            "max_entries": int(MAX_RAW_TRACKING_CACHE_ENTRIES),
            "max_bytes": int(MAX_RAW_TRACKING_CACHE_BYTES),
            "max_item_bytes": int(MAX_RAW_TRACKING_CACHE_ITEM_BYTES),
        }


class MaskDataError(ValueError):
    """Raised when a browser edit document is malformed or stale."""


class MaskSourceMismatch(MaskDataError):
    """A valid edit document was authored for a different source clip.

    Source mismatches are deliberately distinct from malformed mask data.  The
    node can use this typed error to return a source-only preview and let the
    browser ask the user whether to discard or rebind the old edits without
    ever rasterizing them against the new clip.
    """

    _REASONS = frozenset({"frames", "width", "height", "fingerprint"})
    _INT_FIELDS = frozenset({"frames", "width", "height"})

    def __init__(
        self,
        reason: str,
        previous_source: Mapping[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        if reason not in self._REASONS:
            raise ValueError(f"unsupported mask source mismatch reason: {reason!r}")
        self.reason = str(reason)
        self.previous_source = self._sanitize_source(previous_source)
        super().__init__(message or f"mask_data belongs to a different source {reason}")

    @classmethod
    def _sanitize_source(cls, value: Mapping[str, Any] | None) -> dict[str, Any]:
        """Keep only bounded, JSON-safe source identity fields."""

        if not isinstance(value, Mapping):
            return {}
        sanitized: dict[str, Any] = {}
        for key in ("frames", "width", "height"):
            raw = value.get(key)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            try:
                number = int(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if float(raw) != number or number < 1:
                continue
            sanitized[key] = number
        raw_fingerprint = value.get("fingerprint")
        if raw_fingerprint is not None:
            if isinstance(raw_fingerprint, (str, int, float, bool)):
                fingerprint = str(raw_fingerprint)
                # Source fingerprints are generated as short digests.  Bound
                # the diagnostic field if a hand-authored document is hostile.
                if fingerprint:
                    sanitized["fingerprint"] = fingerprint[:256]
        return sanitized


class MaskRefinementError(MaskDataError):
    """Rectangle refinement could not produce a meaningful foreground mask."""


def _json_value(value: Any) -> Any:
    if value is None or value == "":
        return {}
    if isinstance(value, (str, bytes, bytearray)):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            message = getattr(exc, "msg", str(exc))
            raise MaskDataError(f"mask_data must be valid JSON: {message}") from exc
    if not isinstance(value, Mapping):
        raise MaskDataError("mask_data must be a JSON object")
    return value


def _finite_unit(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaskDataError(f"{label} must be a finite number in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise MaskDataError(f"{label} must be a finite number in [0, 1]")
    return 0.0 if number == 0.0 else number


def _positive_unit(value: Any, label: str, *, maximum: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaskDataError(f"{label} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0 or number > maximum:
        raise MaskDataError(f"{label} must be in (0, {maximum}]")
    return number


def _frame_number(value: Any, frame_count: int | None, label: str = "frame") -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaskDataError(f"{label} must be an integer")
    number = int(value)
    if float(value) != number:
        raise MaskDataError(f"{label} must be an integer")
    if frame_count is not None and not 0 <= number < frame_count:
        raise MaskDataError(
            f"{label} {number} is outside this clip (0..{max(0, frame_count - 1)})"
        )
    return number


def _points(value: Any, label: str, *, minimum: int = 1) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) < minimum:
        raise MaskDataError(f"{label} must contain at least {minimum} points")
    if len(value) > MAX_POINTS_PER_EVENT:
        raise MaskDataError(f"{label} has too many points ({len(value)})")
    result: list[list[float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise MaskDataError(f"{label}[{index}] must be [x, y]")
        result.append([
            _finite_unit(point[0], f"{label}[{index}].x"),
            _finite_unit(point[1], f"{label}[{index}].y"),
        ])
    return result


def _event_mode(value: Any, label: str) -> str:
    mode = "add" if value is None else value
    if mode not in {"add", "erase"}:
        raise MaskDataError(f"{label} mode must be 'add' or 'erase'")
    return str(mode)


def _normalise_event(event: Any, label: str, frame_count: int | None = None) -> tuple[str, dict[str, Any]]:
    """Normalize one event from either the canonical or compact UI spelling."""

    if not isinstance(event, Mapping):
        raise MaskDataError(f"{label} must be an object")
    kind = str(event.get("type", event.get("kind", "stroke"))).lower()
    if kind in {"brush", "paint", "stroke"}:
        points = _points(event.get("points"), f"{label}.points")
        radius = _positive_unit(event.get("radius", 0.02), f"{label}.radius", maximum=0.5)
        return "strokes", {
            "mode": _event_mode(event.get("mode"), label),
            "radius": radius,
            "points": points,
        }
    if kind in {"rect", "rectangle", "box"}:
        # Accept x0/y0/x1/y1 as well as x/y/w/h so imported edit logs are easy
        # to use.  Rectangles are required to stay inside the normalized image;
        # silently clipping a typo is unsafe for an inpaint mask.
        if "x" in event:
            x = _finite_unit(event.get("x"), f"{label}.x")
            y = _finite_unit(event.get("y"), f"{label}.y")
            width = _positive_unit(event.get("w", event.get("width")), f"{label}.w")
            height = _positive_unit(event.get("h", event.get("height")), f"{label}.h")
        else:
            x = _finite_unit(event.get("x0"), f"{label}.x0")
            y = _finite_unit(event.get("y0"), f"{label}.y0")
            x1 = _finite_unit(event.get("x1"), f"{label}.x1")
            y1 = _finite_unit(event.get("y1"), f"{label}.y1")
            width, height = x1 - x, y1 - y
            if width <= 0 or height <= 0:
                raise MaskDataError(f"{label} must have positive width and height")
        if x + width > 1.0 + 1e-9 or y + height > 1.0 + 1e-9:
            raise MaskDataError(f"{label} extends outside [0, 1]")
        normalized: dict[str, Any] = {
            "mode": _event_mode(event.get("mode"), label),
            "x": x,
            "y": y,
            "w": width,
            "h": height,
            "refine": bool(event.get("refine", event.get("grabcut", False))),
        }
        # ``refine:true`` predates method selection and remains GrabCut for
        # backwards compatibility.  Keep an explicitly supplied method in
        # canonical data so the browser can round-trip the selected backend;
        # omit the inferred legacy default to preserve old documents byte-for-
        # byte where possible.
        if "refine_method" in event:
            method = event.get("refine_method")
            if not isinstance(method, str) or method not in {"sam3", "grabcut"}:
                raise MaskDataError(
                    f"{label}.refine_method must be 'sam3' or 'grabcut'"
                )
            normalized["refine_method"] = str(method)
        return "rects", normalized
    if kind in {"polygon", "poly"}:
        return "polygons", {
            "mode": _event_mode(event.get("mode"), label),
            "points": _points(event.get("points"), f"{label}.points", minimum=3),
        }
    raise MaskDataError(f"{label}.type must be stroke, rect, or polygon")


def _empty_frame() -> dict[str, Any]:
    # ``inherit_present`` is internal bookkeeping only.  It lets the parser
    # preserve an explicitly supplied ``inherit:false`` without adding a new
    # field to legacy documents that never supplied the flag.
    return {
        "strokes": [], "rects": [], "polygons": [], "events": [],
        "inherit": False, "inherit_present": False,
    }


def _add_event(frames: dict[int, dict[str, Any]], frame: int, event: Any,
               label: str, frame_count: int | None) -> None:
    kind, normalized = _normalise_event(event, label, frame_count)
    bucket = frames.setdefault(frame, _empty_frame())
    normalized["type"] = "stroke" if kind == "strokes" else ("rect" if kind == "rects" else "polygon")
    normalized["order"] = len(bucket["events"])
    bucket["events"].append(normalized)
    bucket[kind].append(normalized)


def frame_fingerprint(images: np.ndarray) -> str:
    """Return a bounded content fingerprint for source-session isolation."""

    array = np.asarray(images)
    if array.ndim != 4 or array.shape[0] < 1:
        raise MaskDataError("images must have shape T x H x W x C")
    # Four temporal samples and at most a 32x32 spatial grid keep this cheap
    # even for 4K clips while distinguishing same-sized clips in practice.
    indices = np.linspace(0, array.shape[0] - 1, min(4, array.shape[0]), dtype=np.int64)
    y_step = max(1, array.shape[1] // 32)
    x_step = max(1, array.shape[2] // 32)
    sample = array[indices, ::y_step, ::x_step, :3].astype(np.float32, copy=False)
    sample = np.nan_to_num(sample, nan=0.0, posinf=1.0, neginf=0.0)
    quantized = np.asarray(np.clip(sample, 0.0, 1.0) * 255.0, dtype=np.uint8)
    import hashlib
    return hashlib.sha1(quantized.tobytes()).hexdigest()[:20]


def optical_flow_available() -> bool:
    """Whether OpenCV tracking is available in this runtime."""

    return cv2 is not None


def _crop_int(value: Any, label: str, *, minimum: int = 0, maximum: int = 1 << 20) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise MaskDataError(f"{label} must be an integer")
    integer = int(round(float(value)))
    if not minimum <= integer <= maximum:
        raise MaskDataError(f"{label} must be between {minimum} and {maximum}")
    return integer


def normalize_crop_settings(raw: Any) -> dict[str, Any] | None:
    """Validate optional mask-aware autocrop settings from mask_data['crop']."""

    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise MaskDataError("mask_data.crop must be an object")
    mode = str(raw.get("mode", "full"))
    if mode not in {"full", "auto", "manual"}:
        raise MaskDataError("mask_data.crop.mode must be full, auto, or manual")
    margin = _crop_int(raw.get("margin", 64), "mask_data.crop.margin", minimum=0, maximum=4096)
    multiple = _crop_int(raw.get("multiple", 1), "mask_data.crop.multiple", minimum=1, maximum=512)
    result: dict[str, Any] = {"mode": mode, "margin": margin, "multiple": multiple}
    if raw.get("box") is not None:
        raw_box = raw.get("box")
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            raise MaskDataError("mask_data.crop.box must be a 4-element list")
        box = [_crop_int(item, f"mask_data.crop.box[{i}]", minimum=-4096, maximum=65536) for i, item in enumerate(raw_box)]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise MaskDataError("mask_data.crop.box must have positive width and height")
        result["box"] = box
    if raw.get("size") is not None:
        raw_size = raw.get("size")
        if not isinstance(raw_size, (list, tuple)) or len(raw_size) != 2:
            raise MaskDataError("mask_data.crop.size must be [width, height]")
        size = [_crop_int(item, f"mask_data.crop.size[{i}]", minimum=1, maximum=65536) for i, item in enumerate(raw_size)]
        result["size"] = size
    return result


def mask_extent(mask: np.ndarray) -> dict[str, Any] | None:
    """Return half-open pixel bounding box and non-empty frame indices across a mask sequence."""

    if hasattr(mask, "detach"):
        mask = mask.detach().to(device="cpu").numpy()
    array = np.asarray(mask, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise MaskDataError("mask must have shape T x H x W")
    count, height, width = map(int, array.shape)
    active: list[int] = []
    left, top = width, height
    right = bottom = 0
    for index, frame in enumerate(array):
        pixels = frame > 1e-5
        rows = np.flatnonzero(pixels.any(axis=1))
        if not rows.size:
            continue
        cols = np.flatnonzero(pixels.any(axis=0))
        active.append(index)
        left = min(left, int(cols[0]))
        top = min(top, int(rows[0]))
        right = max(right, int(cols[-1]) + 1)
        bottom = max(bottom, int(rows[-1]) + 1)
    return {"box": [left, top, right, bottom], "frames": active} if active else None


def _fit_crop_axis(lo: int, hi: int, limit: int, multiple: int = 1) -> tuple[int, int]:
    min_size = min(limit, max(1, min(128, limit // 3)))
    raw_size = min(limit, max(min_size, hi - lo))
    if multiple > 1:
        max_mult = (limit // multiple) * multiple
        if max_mult >= multiple:
            rounded = ((raw_size + multiple - 1) // multiple) * multiple
            size = min(max_mult, max(multiple, rounded))
        else:
            size = limit
    else:
        size = raw_size
    start = max(0, min(limit - size, (lo + hi - size) // 2))
    return start, start + size


def plan_crop_window(
    shape: tuple[int, int, int],
    extent: Mapping[str, Any] | None,
    crop_opts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a canvas-safe spatial crop window from the mask extent and crop settings."""

    count, height, width = map(int, shape)
    opts = normalize_crop_settings(crop_opts) or {"mode": "full", "margin": 64, "multiple": 1}
    mode = opts["mode"]
    margin = int(opts["margin"])
    multiple = int(opts.get("multiple", 1))
    if mode == "full" or not extent or not extent.get("frames"):
        left, top, right, bottom = 0, 0, width, height
    else:
        x, y, r, b = map(int, extent["box"])
        left, top, right, bottom = x - margin, y - margin, r + margin, b + margin
        if mode == "manual" and opts.get("box") and opts.get("size") == [width, height]:
            u, v, s, t = map(int, opts["box"])
            left, top, right, bottom = min(x, u), min(y, v), max(r, s), max(b, t)
        left, right = _fit_crop_axis(max(0, left), min(width, right), width, multiple)
        top, bottom = _fit_crop_axis(max(0, top), min(height, bottom), height, multiple)
        if mode == "auto" and (right - left) * (bottom - top) >= 0.90 * width * height:
            if multiple > 1:
                max_w = (width // multiple) * multiple or width
                max_h = (height // multiple) * multiple or height
                left = max(0, min(width - max_w, (x + r - max_w) // 2))
                right = left + max_w
                top = max(0, min(height - max_h, (y + b - max_h) // 2))
                bottom = top + max_h
            else:
                left, top, right, bottom = 0, 0, width, height
    frames = list(extent["frames"]) if extent and extent.get("frames") else []
    active_range = [int(frames[0]), int(frames[-1]) + 1] if frames else [0, count]
    return {
        "mode": mode,
        "margin": margin,
        "multiple": multiple,
        "box": [int(left), int(top), int(right), int(bottom)],
        "width": int(right - left),
        "height": int(bottom - top),
        "source_width": width,
        "source_height": height,
        "active_frames": active_range,
        "total_frames": count,
    }


_RESIZE_METHODS = ("lanczos", "bicubic", "bilinear", "area", "nearest")
_COLOR_MATCH_MODES = ("none", "mean_std")


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise MaskDataError(f"{name} must be an integer")
    return int(value)


def parse_crop_info(
    crop_info: Any,
    orig_h: int | None = None,
    orig_w: int | None = None,
) -> dict[str, Any]:
    if isinstance(crop_info, str):
        try:
            payload = json.loads(crop_info)
        except (TypeError, ValueError) as exc:
            raise MaskDataError("crop_info must be valid JSON") from exc
    elif isinstance(crop_info, dict):
        payload = crop_info
    else:
        raise MaskDataError("crop_info must be a JSON string or mapping")

    if not isinstance(payload, dict):
        raise MaskDataError("crop_info must be a JSON object")

    raw_box = payload.get("box")
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        raise MaskDataError("crop_info.box must be a 4-element [left, top, right, bottom] list")

    left = _require_int(raw_box[0], "crop_info.box[0]")
    top = _require_int(raw_box[1], "crop_info.box[1]")
    right = _require_int(raw_box[2], "crop_info.box[2]")
    bottom = _require_int(raw_box[3], "crop_info.box[3]")

    if "source_width" in payload:
        source_width = _require_int(payload["source_width"], "crop_info.source_width")
    elif orig_w is not None:
        source_width = _require_int(orig_w, "orig_w")
    else:
        raise MaskDataError("crop_info missing source_width")

    if "source_height" in payload:
        source_height = _require_int(payload["source_height"], "crop_info.source_height")
    elif orig_h is not None:
        source_height = _require_int(orig_h, "orig_h")
    else:
        raise MaskDataError("crop_info missing source_height")

    if orig_w is not None and source_width != int(orig_w):
        raise MaskDataError(
            f"crop_info.source_width ({source_width}) does not match original_images width ({orig_w})"
        )
    if orig_h is not None and source_height != int(orig_h):
        raise MaskDataError(
            f"crop_info.source_height ({source_height}) does not match original_images height ({orig_h})"
        )

    if source_width <= 0 or source_height <= 0:
        raise MaskDataError("crop_info source dimensions must be positive")
    if left < 0 or top < 0 or right > source_width or bottom > source_height or left >= right or top >= bottom:
        raise MaskDataError(
            f"crop_info.box [{left}, {top}, {right}, {bottom}] is invalid for {source_width}x{source_height}"
        )

    active_raw = payload.get("active_frames")
    active_frames: list[int] | None = None
    if active_raw is not None:
        if not isinstance(active_raw, (list, tuple)) or len(active_raw) != 2:
            raise MaskDataError("crop_info.active_frames must be a [start, end] pair")
        start_frame = _require_int(active_raw[0], "crop_info.active_frames[0]")
        end_frame = _require_int(active_raw[1], "crop_info.active_frames[1]")
        if start_frame < 0 or end_frame < start_frame:
            raise MaskDataError("crop_info.active_frames must satisfy 0 <= start <= end")
        active_frames = [start_frame, end_frame]

    return {
        "mode": str(payload.get("mode") or "auto"),
        "margin": int(payload.get("margin", 0)),
        "multiple": int(payload.get("multiple", 1)),
        "box": [left, top, right, bottom],
        "width": right - left,
        "height": bottom - top,
        "source_width": source_width,
        "source_height": source_height,
        "active_frames": active_frames,
        "total_frames": int(payload["total_frames"]) if payload.get("total_frames") is not None else None,
    }


def _resize_hw(array: np.ndarray, target_h: int, target_w: int, method: str = "lanczos") -> np.ndarray:
    method_key = str(method or "lanczos").strip().lower()
    if method_key not in _RESIZE_METHODS:
        raise MaskDataError(f"Unsupported resize_method: {method!r}")

    src = np.asarray(array, dtype=np.float32)
    if src.shape[0] == target_h and src.shape[1] == target_w:
        return np.clip(src, 0.0, 1.0).astype(np.float32, copy=False)

    if cv2 is not None:
        interp_map = {
            "lanczos": cv2.INTER_LANCZOS4,
            "bicubic": cv2.INTER_CUBIC,
            "bilinear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "nearest": cv2.INTER_NEAREST,
        }
        interp = interp_map[method_key]
        if src.ndim == 3 and src.shape[2] == 1:
            resized = cv2.resize(src[:, :, 0], (target_w, target_h), interpolation=interp)[:, :, None]
        else:
            resized = cv2.resize(src, (target_w, target_h), interpolation=interp)
        return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)

    from PIL import Image

    resample_map = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
        "area": Image.Resampling.BOX,
        "nearest": Image.Resampling.NEAREST,
    }
    resample = resample_map[method_key]
    if src.ndim == 2:
        pil_img = Image.fromarray(src, mode="F")
        out = np.asarray(pil_img.resize((target_w, target_h), resample=resample), dtype=np.float32)
        return np.clip(out, 0.0, 1.0)

    channels = [
        np.asarray(
            Image.fromarray(src[:, :, ch], mode="F").resize((target_w, target_h), resample=resample),
            dtype=np.float32,
        )
        for ch in range(src.shape[2])
    ]
    return np.clip(np.stack(channels, axis=-1), 0.0, 1.0)


def _window_edge_mask(
    height: int,
    width: int,
    edge_feather: int,
    touch_left: bool,
    touch_top: bool,
    touch_right: bool,
    touch_bottom: bool,
) -> np.ndarray:
    if edge_feather <= 0:
        return np.ones((height, width), dtype=np.float32)

    mask_x = np.ones(width, dtype=np.float32)
    if width > 1 and (not touch_left or not touch_right):
        rx = min(int(edge_feather), width // 2)
        if rx > 0:
            ramp_x = (0.5 - 0.5 * np.cos(np.pi * (np.arange(rx, dtype=np.float32) + 0.5) / rx)).astype(np.float32)
            if not touch_left:
                mask_x[:rx] = np.minimum(mask_x[:rx], ramp_x)
            if not touch_right:
                mask_x[-rx:] = np.minimum(mask_x[-rx:], ramp_x[::-1])

    mask_y = np.ones(height, dtype=np.float32)
    if height > 1 and (not touch_top or not touch_bottom):
        ry = min(int(edge_feather), height // 2)
        if ry > 0:
            ramp_y = (0.5 - 0.5 * np.cos(np.pi * (np.arange(ry, dtype=np.float32) + 0.5) / ry)).astype(np.float32)
            if not touch_top:
                mask_y[:ry] = np.minimum(mask_y[:ry], ramp_y)
            if not touch_bottom:
                mask_y[-ry:] = np.minimum(mask_y[-ry:], ramp_y[::-1])

    return (mask_y[:, None] * mask_x[None, :]).astype(np.float32)


def _finite_float_array(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr


def restore_crop_window(
    original_images: np.ndarray,
    cropped_images: np.ndarray,
    crop_info: Any,
    mask: np.ndarray | None = None,
    edge_feather: int = 0,
    resize_method: str = "lanczos",
    invert_mask: bool = False,
    color_match: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    orig = _finite_float_array(original_images)
    crop = _finite_float_array(cropped_images)
    if orig.ndim != 4:
        raise MaskDataError("original_images must have shape T x H x W x C")
    if crop.ndim != 4:
        raise MaskDataError("cropped_images must have shape T x H x W x C")
    if orig.shape[0] < 1 or crop.shape[0] < 1:
        raise MaskDataError("original_images and cropped_images must be non-empty")

    feather_px = int(edge_feather)
    if feather_px < 0:
        raise MaskDataError("edge_feather must be >= 0")

    color_mode = str(color_match or "none").strip().lower()
    if color_mode not in _COLOR_MATCH_MODES:
        raise MaskDataError(f"Unsupported color_match: {color_match!r}")

    orig_t, orig_h, orig_w, orig_c = orig.shape
    crop_t, _, _, crop_c = crop.shape
    info = parse_crop_info(crop_info, orig_h=orig_h, orig_w=orig_w)
    left, top, right, bottom = info["box"]
    target_w = right - left
    target_h = bottom - top

    mask_arr: np.ndarray | None = None
    mask_is_full = False
    if mask is not None:
        mask_arr = _finite_float_array(mask)
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[None, :, :]
        if mask_arr.ndim != 3:
            raise MaskDataError("mask must have shape T x H x W or H x W")
        if mask_arr.shape[0] < 1:
            raise MaskDataError("mask must be non-empty")
        mask_arr = np.clip(mask_arr, 0.0, 1.0)
        mask_is_full = mask_arr.shape[1] == orig_h and mask_arr.shape[2] == orig_w

    output = np.clip(orig, 0.0, 1.0).copy()
    applied_mask = np.zeros((orig_t, orig_h, orig_w), dtype=np.float32)

    active = info.get("active_frames")
    if active is not None:
        start_frame = max(0, min(active[0], orig_t))
        end_frame = max(start_frame, min(active[1], orig_t))
    else:
        start_frame, end_frame = 0, orig_t

    span = end_frame - start_frame
    if span <= 0:
        return output, applied_mask

    write_count = span if crop_t == 1 else min(crop_t, span)
    window_edge = _window_edge_mask(
        target_h,
        target_w,
        feather_px,
        touch_left=(left == 0),
        touch_top=(top == 0),
        touch_right=(right == orig_w),
        touch_bottom=(bottom == orig_h),
    )
    blend_c = min(orig_c, crop_c)
    crop_full_timeline = crop_t == orig_t and orig_t > 1

    for step in range(write_count):
        orig_idx = start_frame + step
        if crop_t == 1:
            crop_idx = 0
        elif crop_full_timeline:
            crop_idx = orig_idx
        else:
            crop_idx = step

        crop_patch = _resize_hw(crop[crop_idx], target_h, target_w, method=resize_method)
        orig_patch = output[orig_idx, top:bottom, left:right, :blend_c]

        if mask_arr is None:
            local_alpha = window_edge
        else:
            t_mask = mask_arr.shape[0]
            if t_mask == 1:
                m_idx = 0
            elif t_mask == orig_t:
                m_idx = orig_idx
            else:
                m_idx = min(step, t_mask - 1)

            raw_m = mask_arr[m_idx]
            if mask_is_full:
                local_m = raw_m[top:bottom, left:right]
            else:
                local_m = _resize_hw(raw_m, target_h, target_w, method="bilinear")

            if invert_mask:
                local_m = 1.0 - local_m
            local_alpha = np.clip(local_m, 0.0, 1.0)
            if feather_px > 0:
                local_alpha = local_alpha * window_edge

            if color_mode == "mean_std":
                border = local_alpha < 0.05
                if int(np.count_nonzero(border)) >= 16:
                    crop_rgb = crop_patch[:, :, :blend_c]
                    orig_b = orig_patch[border]
                    crop_b = crop_rgb[border]
                    orig_mean = orig_b.mean(axis=0)
                    crop_mean = crop_b.mean(axis=0)
                    orig_std = orig_b.std(axis=0)
                    crop_std = crop_b.std(axis=0)
                    scale = np.where(
                        crop_std > 1e-4,
                        np.clip(orig_std / np.maximum(crop_std, 1e-4), 0.7, 1.4),
                        1.0,
                    )
                    adjusted = np.clip((crop_rgb - crop_mean) * scale + orig_mean, 0.0, 1.0)
                    crop_patch = crop_patch.copy()
                    crop_patch[:, :, :blend_c] = adjusted

        alpha_3d = local_alpha[:, :, None]
        output[orig_idx, top:bottom, left:right, :blend_c] = (
            crop_patch[:, :, :blend_c] * alpha_3d + orig_patch * (1.0 - alpha_3d)
        )
        applied_mask[orig_idx, top:bottom, left:right] = local_alpha

    return np.clip(output, 0.0, 1.0), np.clip(applied_mask, 0.0, 1.0)


def _sam3_prompt(prompt: Any, frame: int, frame_count: int | None, label: str) -> dict[str, Any]:
    if not isinstance(prompt, Mapping):
        raise MaskDataError(f"{label} must be an object")
    prompt_frame = _frame_number(prompt.get("frame", frame), frame_count, f"{label}.frame")
    if prompt_frame < 0:
        raise MaskDataError(f"{label}.frame must be >= 0")
    if all(name in prompt for name in ("x", "y", "w", "h")):
        box = {name: _finite_unit(prompt[name], f"{label}.{name}") for name in ("x", "y", "w", "h")}
        if box["w"] <= 0 or box["h"] <= 0:
            raise MaskDataError(f"{label} width and height must be positive")
        if box["x"] + box["w"] > 1.0 or box["y"] + box["h"] > 1.0:
            raise MaskDataError(f"{label} extends outside [0, 1]")
        parsed_prompt: dict[str, Any] = {**box, "frame": prompt_frame}
        total_points = 0
    else:
        points = _points(prompt.get("points"), f"{label}.points")
        labels = prompt.get("labels", [1] * len(points))
        if not isinstance(labels, (list, tuple)) or len(labels) != len(points):
            raise MaskDataError(f"{label}.labels must match points")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or item not in (0, 1) for item in labels):
            raise MaskDataError(f"{label}.labels must be 0 or 1")
        parsed_prompt = {"points": points, "labels": [int(item) for item in labels], "frame": prompt_frame}
        total_points = len(points)

    refinements_raw = prompt.get("refinements")
    if refinements_raw is not None:
        if not isinstance(refinements_raw, (list, tuple)):
            raise MaskDataError(f"{label}.refinements must be a list")
        if len(refinements_raw) > MAX_SAM3_REFINEMENT_FRAMES:
            raise MaskDataError(f"{label}.refinements exceeds maximum ({MAX_SAM3_REFINEMENT_FRAMES})")
        seen_frames: set[int] = set()
        refinements: list[dict[str, Any]] = []
        for index, entry in enumerate(refinements_raw):
            r_label = f"{label}.refinements[{index}]"
            if not isinstance(entry, Mapping):
                raise MaskDataError(f"{r_label} must be an object")
            if "frame" not in entry:
                raise MaskDataError(f"{r_label}.frame is required")
            r_frame = _frame_number(entry.get("frame"), frame_count, f"{r_label}.frame")
            if r_frame < 0:
                raise MaskDataError(f"{r_label}.frame must be >= 0")
            if r_frame == prompt_frame:
                raise MaskDataError(f"{r_label}.frame must differ from primary object frame {prompt_frame}")
            if r_frame in seen_frames:
                raise MaskDataError(f"{r_label}.frame duplicates frame {r_frame}")
            seen_frames.add(r_frame)
            r_points = _points(entry.get("points"), f"{r_label}.points")
            r_labels = entry.get("labels", [1] * len(r_points))
            if not isinstance(r_labels, (list, tuple)) or len(r_labels) != len(r_points):
                raise MaskDataError(f"{r_label}.labels must match points")
            if any(isinstance(val, bool) or not isinstance(val, (int, float)) or val not in (0, 1) for val in r_labels):
                raise MaskDataError(f"{r_label}.labels must be 0 or 1")
            total_points += len(r_points)
            refinements.append({
                "frame": r_frame,
                "points": r_points,
                "labels": [int(val) for val in r_labels],
            })
        if total_points > MAX_SAM3_OBJECT_POINTS:
            raise MaskDataError(f"SAM3 object exceeds maximum {MAX_SAM3_OBJECT_POINTS} points")
        if refinements:
            refinements.sort(key=lambda item: item["frame"])
            parsed_prompt["refinements"] = refinements

    return parsed_prompt


def _parse_declared_source(
    raw: Mapping[str, Any],
    frame_count: int | None,
    *,
    width: int | None = None,
    height: int | None = None,
    fingerprint: str | None = None,
) -> dict[str, Any]:
    source = raw.get("source") if isinstance(raw.get("source"), Mapping) else {}
    declared_frames = source.get("frames", source.get("frame_count", source.get("count")))
    declared_width = source.get("width")
    declared_height = source.get("height")
    declared_fingerprint = source.get("fingerprint", source.get("session"))
    if declared_frames is not None:
        declared_frames = _frame_number(declared_frames, None, "source.frames")
        if declared_frames < 1 or declared_frames > MAX_FRAMES:
            raise MaskDataError("source.frames must be between 1 and 4096")
    previous_source: dict[str, Any] = {}
    if declared_frames is not None:
        previous_source["frames"] = int(declared_frames)
    for number, label in ((declared_width, "source.width"), (declared_height, "source.height")):
        if number is not None:
            try:
                valid_integer = (
                    not isinstance(number, bool)
                    and isinstance(number, (int, float))
                    and int(number) == number
                    and int(number) >= 1
                )
            except (TypeError, ValueError, OverflowError):
                valid_integer = False
            if not valid_integer:
                raise MaskDataError(f"{label} must be a positive integer")
    if declared_width is not None:
        previous_source["width"] = int(declared_width)
    if declared_height is not None:
        previous_source["height"] = int(declared_height)
    if declared_fingerprint is not None:
        previous_source["fingerprint"] = declared_fingerprint
    if frame_count is not None and declared_frames is not None and declared_frames != frame_count:
        raise MaskSourceMismatch(
            "frames",
            previous_source,
            "mask_data belongs to a different frame count",
        )
    if width is not None and declared_width is not None and int(declared_width) != int(width):
        raise MaskSourceMismatch(
            "width",
            previous_source,
            "mask_data belongs to a different source width",
        )
    if height is not None and declared_height is not None and int(declared_height) != int(height):
        raise MaskSourceMismatch(
            "height",
            previous_source,
            "mask_data belongs to a different source height",
        )
    if fingerprint is not None and declared_fingerprint is not None and str(declared_fingerprint) != str(fingerprint):
        raise MaskSourceMismatch(
            "fingerprint",
            previous_source,
            "mask_data belongs to a different source clip",
        )
    canonical_source: dict[str, Any] = {}
    if declared_frames is not None:
        canonical_source["frames"] = int(declared_frames)
    if declared_width is not None:
        canonical_source["width"] = int(declared_width)
    if declared_height is not None:
        canonical_source["height"] = int(declared_height)
    if declared_fingerprint is not None:
        canonical_source["fingerprint"] = str(declared_fingerprint)
    return canonical_source

def _parse_frame_payload(
    raw: Mapping[str, Any],
    frame_count: int | None,
    *,
    label_prefix: str = "",
) -> tuple[dict[str, dict[str, Any]], list[int], int, int]:
    frames: dict[int, dict[str, Any]] = {}
    total_events = 0
    total_points = 0

    def ingest(frame_value: Any, payload: Any, label: str) -> None:
        nonlocal total_events, total_points
        frame = _frame_number(frame_value, frame_count, f"{label}.frame")
        if isinstance(payload, Mapping):
            bucket = frames.setdefault(frame, _empty_frame())
            if "inherit" in payload:
                inherit = payload.get("inherit")
                if not isinstance(inherit, bool):
                    raise MaskDataError(f"{label}.inherit must be a boolean")
                bucket["inherit"] = inherit
                bucket["inherit_present"] = True
            # The editor emits this list when a frame mixes tools.  It is the
            # only representation that preserves rectangle->erase->polygon
            # order; grouped legacy fields are intentionally skipped when it
            # is present to avoid applying events twice.
            if "events" in payload:
                entries = payload.get("events", [])
                if not isinstance(entries, (list, tuple)):
                    raise MaskDataError(f"{label}.events must be a list")
                for index, event in enumerate(entries):
                    _add_event(frames, frame, event, f"{label}.events[{index}]", frame_count)
                    total_events += 1
                    total_points += len(event.get("points", [])) if isinstance(event, Mapping) else 0
                return
            # Canonical grouped shape.
            for kind, key in (("strokes", "strokes"), ("rects", "rects"), ("rects", "rectangles"), ("polygons", "polygons")):
                entries = payload.get(key, [])
                if entries is None:
                    continue
                if not isinstance(entries, (list, tuple)):
                    raise MaskDataError(f"{label}.{key} must be a list")
                for index, event in enumerate(entries):
                    event = dict(event) if isinstance(event, Mapping) else event
                    if isinstance(event, Mapping) and "type" not in event:
                        event["type"] = "stroke" if kind == "strokes" else ("rect" if kind == "rects" else "polygon")
                    _add_event(frames, frame, event, f"{label}.{key}[{index}]", frame_count)
                    total_events += 1
                    total_points += len(event.get("points", [])) if isinstance(event, Mapping) else 0
            # A frame may also be represented as a flat ``edits`` list.
            entries = payload.get("edits", [])
            if entries is not None:
                if not isinstance(entries, (list, tuple)):
                    raise MaskDataError(f"{label}.edits must be a list")
                for index, event in enumerate(entries):
                    _add_event(frames, frame, event, f"{label}.edits[{index}]", frame_count)
                    total_events += 1
                    total_points += len(event.get("points", [])) if isinstance(event, Mapping) else 0
        elif isinstance(payload, (list, tuple)):
            for index, event in enumerate(payload):
                _add_event(frames, frame, event, f"{label}[{index}]", frame_count)
                total_events += 1
                total_points += len(event.get("points", [])) if isinstance(event, Mapping) else 0
        else:
            raise MaskDataError(f"{label} must be an object or list")

    grouped = raw.get("frames", {})
    if grouped is None:
        grouped = {}
    if not isinstance(grouped, Mapping):
        raise MaskDataError(f"mask_data.{label_prefix}frames must be an object keyed by frame number")
    for key, payload in grouped.items():
        frame_key: Any = key
        if isinstance(key, str) and key.strip():
            try:
                frame_key = int(key)
            except ValueError as exc:
                raise MaskDataError(f"{label_prefix}frames key {key!r} must be an integer") from exc
        ingest(frame_key, payload, f"{label_prefix}frames[{key!r}]")

    # Imported compact records: each record has a frame field and type.
    edits = raw.get("edits", [])
    if edits is not None:
        if not isinstance(edits, (list, tuple)):
            raise MaskDataError(f"mask_data.{label_prefix}edits must be a list")
        for index, event in enumerate(edits):
            if not isinstance(event, Mapping) or "frame" not in event:
                raise MaskDataError(f"{label_prefix}edits[{index}] must include frame")
            ingest(event["frame"], [event], f"{label_prefix}edits[{index}]")

    # ``keys``/``keyframes`` are accepted as aliases for grouped frame data;
    # their explicit frame list is retained below to keep anchor intent clear.
    explicit_anchors: list[int] = []
    anchors = raw.get("anchors", raw.get("keyframes", raw.get("keys", [])))
    if anchors is not None:
        if not isinstance(anchors, (list, tuple)):
            raise MaskDataError(f"mask_data.{label_prefix}anchors must be a list")
        for index, item in enumerate(anchors):
            if isinstance(item, Mapping):
                if "frame" not in item:
                    raise MaskDataError(f"{label_prefix}anchors[{index}] is missing frame")
                frame = _frame_number(item["frame"], frame_count, f"{label_prefix}anchors[{index}].frame")
                explicit_anchors.append(frame)
                if any(key in item for key in ("inherit", "events", "strokes", "rects", "rectangles", "polygons", "edits")):
                    ingest(frame, item, f"{label_prefix}anchors[{index}]")
            else:
                explicit_anchors.append(_frame_number(item, frame_count, f"{label_prefix}anchors[{index}]"))

    if total_events > MAX_EVENTS:
        raise MaskDataError(f"mask_data has too many edit events ({total_events})")
    if total_points > MAX_POINTS:
        raise MaskDataError(f"mask_data has too many points ({total_points})")
    if len(frames) > MAX_EDIT_FRAMES:
        raise MaskDataError("mask_data has too many edited frames")

    canonical_frames: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for frame in sorted(frames):
        bucket = frames[frame]
        if bucket["events"] or bucket.get("inherit_present"):
            canonical_bucket: dict[str, Any] = {"events": [
                {key: value for key, value in event.items() if key != "order"}
                for event in bucket["events"]
            ]}
            if bucket.get("inherit_present"):
                canonical_bucket["inherit"] = bool(bucket.get("inherit"))
            canonical_frames[str(frame)] = canonical_bucket
    return canonical_frames, sorted(set(explicit_anchors)), total_events, total_points


def _parse_sam3_config(
    sam3_raw: Any,
    frame_count: int | None,
    *,
    label_prefix: str = "",
) -> dict[str, Any]:
    if not isinstance(sam3_raw, Mapping):
        raise MaskDataError(f"mask_data.{label_prefix}sam3 must be an object")
    action = str(sam3_raw.get("action", "frame"))
    if action not in {"frame", "track"}:
        raise MaskDataError(f"mask_data.{label_prefix}sam3.action must be 'frame' or 'track'")
    frame = _frame_number(sam3_raw.get("frame", 0), frame_count, f"{label_prefix}sam3.frame")
    direction = str(sam3_raw.get("direction", "forward"))
    if direction not in {"forward", "backward", "both"}:
        raise MaskDataError(f"mask_data.{label_prefix}sam3.direction must be forward, backward, or both")
    token = sam3_raw.get("seed_token", "")
    if token is None:
        token = ""
    if not isinstance(token, (str, int, float)) or isinstance(token, bool):
        raise MaskDataError(f"mask_data.{label_prefix}sam3.seed_token must be a string")
    token = str(token)
    if len(token) > 256:
        raise MaskDataError(f"mask_data.{label_prefix}sam3.seed_token is too long")
    sam3: dict[str, Any] = {
        "action": action,
        "frame": frame,
        "direction": direction,
        "seed_token": token,
    }
    def _total_sam3_points(entry: Mapping[str, Any]) -> int:
        total = len(entry.get("points", [])) if isinstance(entry.get("points"), list) else 0
        for ref in entry.get("refinements", []):
            total += len(ref.get("points", []))
        return total

    prompt = sam3_raw.get("prompt")
    objects = sam3_raw.get("objects")
    if prompt is not None:
        sam3["prompt"] = _sam3_prompt(prompt, frame, frame_count, f"{label_prefix}sam3.prompt")
        if _total_sam3_points(sam3["prompt"]) > MAX_SAM3_OBJECT_POINTS:
            raise MaskDataError(
                f"SAM3 object requires a positive point and at most {MAX_SAM3_OBJECT_POINTS} points"
            )
    if objects is not None:
        if prompt is not None or not isinstance(objects, list) or not 1 <= len(objects) <= MAX_SAM3_OBJECTS:
            raise MaskDataError(
                f"{label_prefix}sam3.objects must contain 1–{MAX_SAM3_OBJECTS} objects and replaces sam3.prompt"
            )
        sam3["objects"] = [
            _sam3_prompt(item, frame, frame_count, f"{label_prefix}sam3.objects[{index}]")
            for index, item in enumerate(objects)
        ]
        for item in sam3["objects"]:
            if ("points" in item and 1 not in item["labels"]) or _total_sam3_points(item) > MAX_SAM3_OBJECT_POINTS:
                raise MaskDataError(
                    f"SAM3 object requires a positive point and at most {MAX_SAM3_OBJECT_POINTS} points"
                )
    return sam3


def _layer_bound_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise MaskDataError(f"{name} must be an integer")
    int_val = int(value)
    if int_val < minimum or int_val > maximum:
        raise MaskDataError(f"{name} must be between {minimum} and {maximum}")
    return int_val


def _parse_layers(
    layers_raw: Any,
    raw: Mapping[str, Any],
    frame_count: int | None,
) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(layers_raw, list) or not 1 <= len(layers_raw) <= MAX_MASK_LAYERS:
        raise MaskDataError(f"mask_data.layers must be a list of 1–{MAX_MASK_LAYERS} layer objects")

    parsed_layers: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    cum_events = 0
    cum_points = 0

    for idx, item in enumerate(layers_raw):
        if not isinstance(item, Mapping):
            raise MaskDataError(f"mask_data.layers[{idx}] must be an object")

        raw_id = item.get("id", f"layer-{idx + 1}")
        if not isinstance(raw_id, str) or not 1 <= len(raw_id.strip()) <= 64:
            raise MaskDataError(f"mask_data.layers[{idx}].id must be a 1–64 character string")
        layer_id = raw_id.strip()
        if layer_id in seen_ids:
            raise MaskDataError(f"Duplicate mask layer id: {layer_id!r}")
        seen_ids.add(layer_id)

        layer_type = str(item.get("type", ""))
        if layer_type not in _LAYER_TYPES:
            raise MaskDataError(f"mask_data.layers[{idx}].type must be one of {_LAYER_TYPES}")

        blend = str(item.get("blend", "add"))
        if blend not in _LAYER_BLENDS:
            raise MaskDataError(f"mask_data.layers[{idx}].blend must be one of {_LAYER_BLENDS}")

        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            raise MaskDataError(f"mask_data.layers[{idx}].enabled must be a boolean")

        raw_name = item.get("name", _LAYER_TYPE_NAMES[layer_type])
        if not isinstance(raw_name, str) or len(raw_name) > 64:
            raise MaskDataError(f"mask_data.layers[{idx}].name must be a string up to 64 characters")
        name = raw_name.strip() or _LAYER_TYPE_NAMES[layer_type]

        start_val = _layer_bound_int(item.get("start_frame", 0), f"layers[{idx}].start_frame", 0, MAX_FRAMES)
        end_val = _layer_bound_int(item.get("end_frame", -1), f"layers[{idx}].end_frame", -1, MAX_FRAMES)

        l_frames, l_anchors, l_events, l_points = _parse_frame_payload(
            item, frame_count, label_prefix=f"layers[{idx}]."
        )
        cum_events += l_events
        cum_points += l_points
        if cum_events > MAX_EVENTS:
            raise MaskDataError(f"mask_data has too many edit events ({cum_events})")
        if cum_points > MAX_POINTS:
            raise MaskDataError(f"mask_data has too many points ({cum_points})")

        layer_entry: dict[str, Any] = {
            "id": layer_id,
            "name": name,
            "type": layer_type,
            "blend": blend,
            "enabled": enabled,
            "start_frame": start_val,
            "end_frame": end_val,
            "frames": l_frames,
        }
        if l_anchors:
            layer_entry["anchors"] = l_anchors
        if layer_type == "sam3" and item.get("sam3") is not None:
            layer_entry["sam3"] = _parse_sam3_config(item["sam3"], frame_count, label_prefix=f"layers[{idx}].")
        parsed_layers.append(layer_entry)

    active_raw = raw.get("active_layer_id")
    if active_raw is not None:
        if not isinstance(active_raw, str) or active_raw not in seen_ids:
            raise MaskDataError("mask_data.active_layer_id must reference one of mask_data.layers[].id")
        active_id = active_raw
    else:
        active_id = parsed_layers[0]["id"]
    return parsed_layers, active_id


def parse_mask_data(value: Any, frame_count: int | None = None, *, width: int | None = None,
                    height: int | None = None, fingerprint: str | None = None) -> dict[str, Any]:
    """Validate and canonicalize the browser's normalized edit document.

    The parser accepts a compact top-level ``edits`` list for imported masks,
    while returning only the stable frame-grouped representation used by the
    node and browser.  ``source`` is optional for hand-authored data, but when
    present its frame count and dimensions must match the current clip.
    Multi-layer documents (``layers``) are validated independently per layer and
    mirrored onto top-level ``frames`` / ``tracking_mode`` / ``sam3``.
    """

    raw = _json_value(value)
    canonical_source = _parse_declared_source(
        raw, frame_count, width=width, height=height, fingerprint=fingerprint
    )
    layers_raw = raw.get("layers")
    result: dict[str, Any] = {"version": 1, "frames": {}}
    if canonical_source:
        result["source"] = canonical_source

    if layers_raw is not None:
        parsed_layers, active_id = _parse_layers(layers_raw, raw, frame_count)
        active_layer = next(item for item in parsed_layers if item["id"] == active_id)
        result["frames"] = copy.deepcopy(active_layer["frames"])
        if active_layer.get("anchors"):
            result["anchors"] = list(active_layer["anchors"])
        if raw.get("tracking_mode") is not None:
            mode = str(raw["tracking_mode"])
            if mode not in _LAYER_TYPES:
                raise MaskDataError("tracking_mode must be optical_flow, static, keyframes, or sam3")
            result["tracking_mode"] = mode
        else:
            result["tracking_mode"] = active_layer["type"]
        if "sam3" in active_layer:
            result["sam3"] = copy.deepcopy(active_layer["sam3"])
        result["layers"] = parsed_layers
        result["active_layer_id"] = active_id
    else:
        canonical_frames, explicit_anchors, _, _ = _parse_frame_payload(raw, frame_count)
        result["frames"] = canonical_frames
        if explicit_anchors:
            result["anchors"] = explicit_anchors
        if raw.get("tracking_mode") is not None:
            mode = str(raw["tracking_mode"])
            if mode not in _LAYER_TYPES:
                raise MaskDataError("tracking_mode must be optical_flow, static, keyframes, or sam3")
            result["tracking_mode"] = mode
        sam3_raw = raw.get("sam3")
        if sam3_raw is not None:
            result["sam3"] = _parse_sam3_config(sam3_raw, frame_count)

    crop = normalize_crop_settings(raw.get("crop"))
    if crop is not None:
        result["crop"] = crop
    return result


def canonical_mask_data(value: Any, frame_count: int | None = None, *, width: int | None = None,
                        height: int | None = None, fingerprint: str | None = None) -> str:
    parsed = parse_mask_data(value, frame_count, width=width, height=height, fingerprint=fingerprint)
    return json.dumps(parsed, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


canonicalize_mask_data = canonical_mask_data


def _event_list(bucket: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(bucket.get("events"), list):
        return [copy.deepcopy(event) for event in bucket["events"]]
    result: list[dict[str, Any]] = []
    for key in ("strokes", "rects", "polygons"):
        for event in bucket.get(key, []) or []:
            copy_event = copy.deepcopy(event)
            copy_event["type"] = "stroke" if key == "strokes" else ("rect" if key == "rects" else "polygon")
            result.append(copy_event)
    return result


def _stroke_brush(height: int, width: int, radius: float, points: Sequence[Sequence[float]]) -> np.ndarray:
    px = np.asarray([[float(x) * max(0, width - 1), float(y) * max(0, height - 1)] for x, y in points], dtype=np.float64)
    radius_px = max(float(radius) * min(height, width), math.sqrt(0.5))
    threshold_sq = radius_px * radius_px
    brush = np.zeros((height, width), dtype=bool)

    def segment(start: np.ndarray, end: np.ndarray) -> None:
        x_min = max(0, int(math.floor(min(start[0], end[0]) - radius_px)))
        x_max = min(width - 1, int(math.ceil(max(start[0], end[0]) + radius_px)))
        y_min = max(0, int(math.floor(min(start[1], end[1]) - radius_px)))
        y_max = min(height - 1, int(math.ceil(max(start[1], end[1]) + radius_px)))
        if x_min > x_max or y_min > y_max:
            return
        yy, xx = np.ogrid[y_min:y_max + 1, x_min:x_max + 1]
        vx, vy = end[0] - start[0], end[1] - start[1]
        length_sq = vx * vx + vy * vy
        if length_sq <= 1e-18:
            nearest_x, nearest_y = start[0], start[1]
        else:
            t = ((xx - start[0]) * vx + (yy - start[1]) * vy) / length_sq
            t = np.clip(t, 0.0, 1.0)
            nearest_x = start[0] + t * vx
            nearest_y = start[1] + t * vy
        distance_sq = (xx - nearest_x) ** 2 + (yy - nearest_y) ** 2
        brush[y_min:y_max + 1, x_min:x_max + 1] |= distance_sq <= threshold_sq + 1e-12

    for start, end in zip(px[:-1], px[1:]):
        segment(start, end)
    if len(px) == 1:
        segment(px[0], px[0])
    return brush


def _polygon_mask(height: int, width: int, points: Sequence[Sequence[float]]) -> np.ndarray:
    coords = np.asarray([[round(float(x) * (width - 1)), round(float(y) * (height - 1))] for x, y in points], dtype=np.int32)
    if cv2 is not None:
        result = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(result, [coords.reshape(-1, 1, 2)], 1)
        return result.astype(bool)
    # Small dependency-free fallback for test/minimal environments.
    yy, xx = np.mgrid[0:height, 0:width]
    inside = np.zeros((height, width), dtype=bool)
    x0, y0 = coords[-1]
    for x1, y1 in coords:
        crossing = ((y1 > yy) != (y0 > yy)) & (xx < (x0 - x1) * (yy - y1) / (y0 - y1 + 1e-12) + x1)
        inside ^= crossing
        x0, y0 = x1, y1
    return inside


def refine_rectangle(image: np.ndarray, event: Mapping[str, Any]) -> np.ndarray | None:
    """Refine a rectangle with bounded GrabCut; return ``None`` on failure.

    GrabCut runs on a context crop around the selected rectangle.  Pixels
    outside the selection are definite background and the rectangle is passed
    as the initial probable-foreground region through ``GC_INIT_WITH_RECT``.
    The result is clipped to the selected rectangle before it is expanded back
    to source dimensions, so refinement can never paint outside the user's
    box.  Large source frames are downscaled for bounded CPU cost.
    """

    if cv2 is None or image is None:
        return None
    try:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[0] < 1 or image.shape[1] < 1 or image.shape[-1] < 3:
            return None
        height, width = map(int, image.shape[:2])
        x0 = int(math.floor(float(event["x"]) * (width - 1)))
        y0 = int(math.floor(float(event["y"]) * (height - 1)))
        x1 = int(math.ceil((float(event["x"]) + float(event["w"])) * (width - 1)))
        y1 = int(math.ceil((float(event["y"]) + float(event["h"])) * (height - 1)))
        x0 = min(max(0, x0), width - 1)
        y0 = min(max(0, y0), height - 1)
        x1 = min(max(x0, x1), width - 1)
        y1 = min(max(y0, y1), height - 1)
        selected_width = x1 - x0 + 1
        selected_height = y1 - y0 + 1
        if selected_width < 3 or selected_height < 3:
            return None

        # A context crop supplies definite background samples.  If the box
        # covers the entire source there is no safe segmentation signal and
        # the caller must report refinement failure instead of painting a
        # fabricated full-frame object.
        margin = max(2, int(round(min(selected_width, selected_height) * 0.1)))
        crop_x0 = max(0, x0 - margin)
        crop_y0 = max(0, y0 - margin)
        crop_x1 = min(width - 1, x1 + margin)
        crop_y1 = min(height - 1, y1 + margin)
        crop_width = crop_x1 - crop_x0 + 1
        crop_height = crop_y1 - crop_y0 + 1
        if crop_width <= selected_width and crop_height <= selected_height:
            return None

        crop = np.asarray(image[crop_y0:crop_y1 + 1, crop_x0:crop_x1 + 1, :3])
        rgb = np.asarray(np.round(np.clip(np.nan_to_num(crop), 0.0, 1.0) * 255.0), dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        scale = min(1.0, float(MAX_GRABCUT_DIM) / max(crop_width, crop_height))
        if scale < 1.0:
            work_width = max(2, int(round(crop_width * scale)))
            work_height = max(2, int(round(crop_height * scale)))
            bgr = cv2.resize(bgr, (work_width, work_height), interpolation=cv2.INTER_AREA)
        else:
            work_width, work_height = crop_width, crop_height

        rect_x = int(round((x0 - crop_x0) * scale))
        rect_y = int(round((y0 - crop_y0) * scale))
        rect_width = max(1, int(round(selected_width * scale)))
        rect_height = max(1, int(round(selected_height * scale)))
        rect_x = min(max(0, rect_x), work_width - 1)
        rect_y = min(max(0, rect_y), work_height - 1)
        rect_width = min(rect_width, work_width - rect_x)
        rect_height = min(rect_height, work_height - rect_y)
        if rect_width < 2 or rect_height < 2:
            return None
        background = np.zeros((1, 65), dtype=np.float64)
        foreground = np.zeros((1, 65), dtype=np.float64)
        labels, _, _ = cv2.grabCut(
            bgr,
            None,
            (rect_x, rect_y, rect_width, rect_height),
            background,
            foreground,
            GRABCUT_ITERATIONS,
            cv2.GC_INIT_WITH_RECT,
        )
        work_mask = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
        if (work_width, work_height) != (crop_width, crop_height):
            work_mask = cv2.resize(work_mask, (crop_width, crop_height), interpolation=cv2.INTER_NEAREST)
        selected = np.zeros((crop_height, crop_width), dtype=bool)
        local_x0, local_y0 = x0 - crop_x0, y0 - crop_y0
        selected[local_y0:local_y0 + selected_height, local_x0:local_x0 + selected_width] = True
        result = np.zeros((height, width), dtype=bool)
        result[crop_y0:crop_y1 + 1, crop_x0:crop_x1 + 1] = np.asarray(work_mask, dtype=bool) & selected
        return result
    except Exception:
        return None


def rasterize_events(events: Iterable[Mapping[str, Any]], height: int, width: int,
                     image: np.ndarray | None = None,
                     initial_mask: np.ndarray | None = None) -> np.ndarray:
    """Rasterize ordered add/erase events into a float mask.

    ``initial_mask`` is intentionally opt-in.  Legacy documents continue to
    rasterize onto zero; the ``inherit:true`` frame flag is the only caller
    that supplies an inherited baseline.
    """

    if height < 1 or width < 1:
        raise MaskDataError("mask dimensions must be positive")
    if initial_mask is None:
        result = np.zeros((height, width), dtype=np.float32)
    else:
        result = np.asarray(initial_mask, dtype=np.float32)
        if result.shape != (height, width):
            raise MaskDataError("initial mask dimensions do not match events")
        result = np.clip(result, 0.0, 1.0).copy()
    for index, event in enumerate(events):
        kind = str(event.get("type", "stroke"))
        if kind == "stroke":
            covered = _stroke_brush(height, width, float(event["radius"]), event["points"])
        elif kind == "rect":
            covered = None
            if bool(event.get("refine")) and str(event.get("mode", "add")) == "add":
                method = str(event.get("refine_method", "grabcut"))
                if method not in {"sam3", "grabcut"}:
                    raise MaskDataError(
                        f"{index}: rectangle refine_method must be 'sam3' or 'grabcut'"
                    )
                if method == "sam3":
                    # Keep SAM3 lazy: importing its native adapter at module
                    # load time would make CPU-only parser tests depend on
                    # ComfyUI/torch.  Failures are typed and explicit so the
                    # node returns its source-only blocked preview.
                    from .mask_sam3 import refine_rectangle_sam3

                    if image is None:
                        raise MaskRefinementError(
                            f"SAM3: {index}: rectangle refinement requires a source frame"
                        )
                    covered = refine_rectangle_sam3(image, event)
                else:
                    covered = refine_rectangle(image, event) if image is not None else None
                    if covered is None:
                        raise MaskRefinementError(
                            f"{index}: rectangle refinement is unavailable for this source"
                        )
                if covered is None:
                    raise MaskRefinementError(
                        f"{index}: rectangle refinement is unavailable for this source"
                    )
                covered = np.asarray(covered, dtype=bool)
                if covered.shape != (height, width):
                    prefix = "SAM3: " if method == "sam3" else ""
                    raise MaskRefinementError(
                        f"{prefix}{index}: rectangle refinement returned an invalid mask"
                    )
                x0 = int(math.floor(float(event["x"]) * (width - 1)))
                y0 = int(math.floor(float(event["y"]) * (height - 1)))
                x1 = int(math.ceil((float(event["x"]) + float(event["w"])) * (width - 1)))
                y1 = int(math.ceil((float(event["y"]) + float(event["h"])) * (height - 1)))
                x0 = min(max(0, x0), width - 1)
                y0 = min(max(0, y0), height - 1)
                x1 = min(max(x0, x1), width - 1)
                y1 = min(max(y0, y1), height - 1)
                selection = np.zeros((height, width), dtype=bool)
                selection[y0:y1 + 1, x0:x1 + 1] = True
                covered &= selection
                selected_pixels = int(selection.sum())
                foreground_pixels = int(covered.sum())
                if foreground_pixels < 1 or foreground_pixels >= selected_pixels:
                    prefix = "SAM3: " if method == "sam3" else ""
                    raise MaskRefinementError(
                        f"{prefix}{index}: rectangle refinement found no separable foreground"
                    )
            if covered is None:
                x0 = int(math.floor(float(event["x"]) * (width - 1)))
                y0 = int(math.floor(float(event["y"]) * (height - 1)))
                x1 = int(math.ceil((float(event["x"]) + float(event["w"])) * (width - 1)))
                y1 = int(math.ceil((float(event["y"]) + float(event["h"])) * (height - 1)))
                covered = np.zeros((height, width), dtype=bool)
                covered[max(0, y0):min(height, y1 + 1), max(0, x0):min(width, x1 + 1)] = True
        elif kind == "polygon":
            covered = _polygon_mask(height, width, event["points"])
        else:
            raise MaskDataError(f"event {index} has unsupported type {kind!r}")
        result[covered] = 0.0 if event.get("mode", "add") == "erase" else 1.0
    return result


def _gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        return image
    if image.shape[-1] == 1:
        return image[..., 0]
    if cv2 is not None:
        return cv2.cvtColor(np.clip(image[..., :3], 0.0, 1.0), cv2.COLOR_RGB2GRAY)
    return np.mean(image[..., :3], axis=-1)


def _flow_between(previous: np.ndarray, following: np.ndarray) -> np.ndarray | None:
    if cv2 is None:
        return None
    try:
        # ComfyUI IMAGE tensors are normalized float32 RGB values in [0, 1].
        # Farneback's polynomial expansion is numerically stable at the
        # conventional 8-bit intensity scale; at [0, 1] the derivatives
        # collapse toward zero and the phase-correlation constant-field
        # fallback masks the loss of local/non-rigid motion.
        previous_gray = np.asarray(_gray(previous), dtype=np.float32) * 255.0
        following_gray = np.asarray(_gray(following), dtype=np.float32) * 255.0
        height, width = previous_gray.shape[:2]
        scale = min(1.0, float(MAX_FLOW_DIM) / max(height, width))
        if scale < 1.0:
            size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            previous_gray = cv2.resize(previous_gray, size, interpolation=cv2.INTER_AREA)
            following_gray = cv2.resize(following_gray, size, interpolation=cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray, following_gray, None,
            pyr_scale=0.5, levels=3, winsize=21, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        # Flat graphic patches (and synthetic QA clips) can produce a nearly
        # zero Farneback field despite an obvious translation.  Phase
        # correlation supplies a conservative constant field in that case;
        # it is still OpenCV optical tracking, not a semantic segmentation
        # claim.
        if float(np.max(np.abs(flow))) < 0.25:
            try:
                shift, response = cv2.phaseCorrelate(previous_gray, following_gray)
                if float(response) > 0.05 and max(abs(float(shift[0])), abs(float(shift[1]))) < max(previous_gray.shape) * 0.5:
                    flow[..., 0] = float(shift[0])
                    flow[..., 1] = float(shift[1])
            except Exception:
                pass
        return flow, (width, height), (previous_gray.shape[1], previous_gray.shape[0])
    except Exception:
        return None


def _warp_mask(mask: np.ndarray, flow: Any, *, backward: bool = False) -> np.ndarray:
    if flow is None or cv2 is None:
        return mask.copy()
    if isinstance(flow, tuple):
        flow, original_size, flow_size = flow
    else:
        original_size = (mask.shape[1], mask.shape[0])
        flow_size = (flow.shape[1], flow.shape[0])
    target_width, target_height = map(int, flow_size)
    work = mask.astype(np.float32)
    if (work.shape[1], work.shape[0]) != (target_width, target_height):
        work = cv2.resize(work, (target_width, target_height), interpolation=cv2.INTER_AREA)
    height, width = work.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    # Farneback's flow is a source->destination displacement.  Sampling the
    # source at destination minus displacement is accurate for the small
    # inter-frame motions expected in the editor and avoids a costly splat.
    direction = -1.0 if not backward else 1.0
    map_x = xx + direction * flow[..., 0]
    map_y = yy + direction * flow[..., 1]
    warped = cv2.remap(work, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    if (warped.shape[1], warped.shape[0]) != tuple(map(int, original_size)):
        warped = cv2.resize(warped, tuple(map(int, original_size)), interpolation=cv2.INTER_LINEAR)
    return warped


def _propagate_to(images: np.ndarray, source_mask: np.ndarray, source: int, target: int,
                  flow_cache: dict[tuple[int, int], Any]) -> np.ndarray:
    current = source_mask.astype(np.float32, copy=True)
    if target == source:
        return current
    step = 1 if target > source else -1
    for frame in range(source, target, step):
        a, b = (frame, frame + 1) if step > 0 else (frame, frame - 1)
        key = (min(a, b), max(a, b))
        if key not in flow_cache:
            flow_cache[key] = _flow_between(images[key[0]], images[key[1]])
        current = _warp_mask(current, flow_cache[key], backward=step < 0)
    return current


def propagate_masks(images: np.ndarray, anchors: Mapping[int, np.ndarray], start_frame: int,
                    end_frame: int, *, phase: str = "optical_flow") -> np.ndarray:
    """Propagate from nearest explicit anchor while preserving anchors exactly."""

    count, height, width = images.shape[:3]
    result = np.zeros((count, height, width), dtype=np.float32)
    if not anchors:
        return result
    from .mask_progress import frame_progress

    ordered = sorted(int(frame) for frame in anchors)
    completed = {frame for frame in ordered if start_frame <= frame <= end_frame}
    report = frame_progress(phase, end_frame - start_frame + 1)
    report(len(completed))
    def sweep(source: int, assign_from: int, stop_before: int, step: int) -> None:
        current = np.asarray(anchors[source], dtype=np.float32).copy()
        flow_cache: dict[tuple[int, int], Any] = {}
        frame = source
        while (step > 0 and frame < stop_before) or (step < 0 and frame > stop_before):
            next_frame = frame + step
            key = (min(frame, next_frame), max(frame, next_frame))
            if key not in flow_cache:
                flow_cache[key] = _flow_between(images[key[0]], images[key[1]])
            current = _warp_mask(current, flow_cache[key], backward=step < 0)
            frame = next_frame
            if (step > 0 and frame >= assign_from) or (step < 0 and frame >= assign_from):
                result[frame] = current
                if start_frame <= frame <= end_frame and frame not in completed:
                    completed.add(frame)
                    report(len(completed))

    # Before/after the outer anchors are one-way sweeps.  Between anchors,
    # split at the nearest-anchor midpoint so each adjacent flow is computed
    # once and no O(N²) per-target re-tracking is needed.
    first, last = ordered[0], ordered[-1]
    if start_frame < first:
        sweep(first, start_frame, start_frame, -1)
    if end_frame > last:
        sweep(last, last + 1, end_frame, 1)
    for left, right in zip(ordered[:-1], ordered[1:]):
        midpoint = (left + right) // 2
        if midpoint > left:
            sweep(left, left + 1, midpoint + 1, 1)
        if right > midpoint + 1:
            sweep(right, midpoint + 1, midpoint, -1)
    for frame in ordered:
        if start_frame <= frame <= end_frame:
            result[frame] = np.asarray(anchors[frame], dtype=np.float32)
    for frame, mask in anchors.items():
        if start_frame <= frame <= end_frame:
            result[int(frame)] = np.asarray(mask, dtype=np.float32)
    return result


def _hold_masks(images: np.ndarray, anchors: Mapping[int, np.ndarray], start_frame: int,
                end_frame: int) -> np.ndarray:
    """Fill a keyframe sequence by holding the nearest explicit anchor.

    ``keyframes`` is intentionally deterministic and does not invoke optical
    flow.  The browser's explicit anchors remain exact; unanchored frames use
    the temporally nearest anchor (ties prefer the earlier frame).
    """

    count, height, width = map(int, images.shape[:3])
    result = np.zeros((count, height, width), dtype=np.float32)
    if not anchors or start_frame > end_frame:
        return result
    ordered = sorted(int(frame) for frame in anchors)
    for frame in range(max(0, int(start_frame)), min(count - 1, int(end_frame)) + 1):
        nearest = min(ordered, key=lambda item: (abs(item - frame), item))
        result[frame] = np.asarray(anchors[nearest], dtype=np.float32)
    return result


def postprocess_mask(mask: np.ndarray, grow: int = 0, feather: int = 0) -> np.ndarray:
    """Grow then feather a ``TxHxW`` mask with bounded integer radii."""

    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 3:
        raise MaskDataError("mask must have shape T x H x W")
    grow = int(max(0, min(256, grow)))
    feather = int(max(0, min(256, feather)))
    result = np.empty_like(mask, order="C")
    np.clip(mask, 0.0, 1.0, out=result)
    if cv2 is None:
        return result
    kernel = None
    if grow:
        size = grow * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    size = feather * 2 + 1
    for frame in result:
        if kernel is not None:
            cv2.dilate(frame, kernel, dst=frame)
        if feather:
            cv2.GaussianBlur(frame, (size, size), sigmaX=max(0.1, feather / 2.0), dst=frame)
    return np.clip(result, 0.0, 1.0, out=result)


def _frame_inherits(bucket: Mapping[str, Any]) -> bool:
    """Return the canonical per-frame inherit flag.

    ``parse_mask_data`` validates the type before this helper is reached.  A
    defensive check keeps callers that pass hand-built canonical mappings from
    silently treating a malformed truthy value as a valid flag.
    """

    value = bucket.get("inherit", False)
    if not isinstance(value, bool):
        raise MaskDataError("frame.inherit must be a boolean")
    return value


def _tracking_anchors_and_bases(
    images: np.ndarray,
    frame_buckets: Mapping[str, Mapping[str, Any]],
    explicit_anchors: set[int],
    start: int,
    end: int,
    tracking_mode: str = "optical_flow",
) -> tuple[dict[int, np.ndarray], np.ndarray, set[int], np.ndarray]:
    """Build final tracking anchors and raw edit baselines.

    Non-inherited buckets are the independent anchors and retain the original
    zero-based rasterization semantics.  Inherited buckets first receive a
    baseline propagated from only those independent anchors; their ordered
    events are then applied to that baseline.  Finally every resulting anchor
    (including inherited corrections) is propagated, so a correction remains
    stable for neighbouring frames.
    """

    count, height, width = map(int, images.shape[:3])
    zero = np.zeros((count, height, width), dtype=np.float32)
    if start > end:
        return {}, zero, set(), zero

    independent: dict[int, np.ndarray] = {}
    independent_frames: set[int] = set()
    for key, bucket in frame_buckets.items():
        frame = int(key)
        if not start <= frame <= end or _frame_inherits(bucket):
            continue
        independent[frame] = rasterize_events(
            _event_list(bucket), height, width, images[frame]
        )
        independent_frames.add(frame)

    # Anchor lists are metadata, not independent paint. In particular an
    # inherited correction can also be listed there after a workflow reload.
    # An intentionally empty key is represented by its own inherit:false
    # bucket; an anchor number without a bucket keeps legacy behaviour.

    if independent:
        inherited_baseline = (
            _hold_masks(images, independent, start, end)
            if str(tracking_mode) == "keyframes"
            else propagate_masks(images, independent, start, end)
        )
    else:
        inherited_baseline = zero.copy()
    # ``edit_base`` is what the browser should use before applying local
    # events.  Independent anchors intentionally expose a zero baseline so a
    # user can opt into a genuinely new anchor; every other frame gets the
    # propagated baseline from the independent anchors.
    edit_bases = inherited_baseline.copy()
    for frame in independent_frames:
        edit_bases[frame] = 0.0

    final_anchors = dict(independent)
    for key, bucket in frame_buckets.items():
        frame = int(key)
        if not start <= frame <= end or not _frame_inherits(bucket):
            continue
        baseline = inherited_baseline[frame] if independent else np.zeros((height, width), dtype=np.float32)
        final_anchors[frame] = rasterize_events(
            _event_list(bucket), height, width, images[frame], baseline
        )
    return final_anchors, edit_bases, set(final_anchors), inherited_baseline


def derive_edit_base_sequence(
    images: np.ndarray,
    parsed: Mapping[str, Any],
    *,
    tracking_mode: str = "optical_flow",
    start_frame: int = 0,
    end_frame: int = -1,
) -> np.ndarray:
    """Return raw per-frame baselines used by the browser's edit canvas.

    ``parsed`` should be the canonical document returned by
    :func:`parse_mask_data`.  This helper intentionally omits grow/feather
    post-processing: the browser needs the exact baseline onto which it will
    paint its normalized events.
    """

    images = np.asarray(images)
    if images.ndim != 4 or images.shape[0] < 1 or images.shape[1] < 1 or images.shape[2] < 1:
        raise MaskDataError("images must have shape T x H x W x C and contain at least one frame")
    count, height, width = map(int, images.shape[:3])
    mode = str(tracking_mode or parsed.get("tracking_mode", "optical_flow"))
    if mode not in {"optical_flow", "static", "keyframes", "sam3"}:
        raise MaskDataError("tracking_mode must be optical_flow, static, keyframes, or sam3")
    start = max(0, int(start_frame))
    end = count - 1 if int(end_frame) < 0 else min(count - 1, int(end_frame))
    result = np.zeros((count, height, width), dtype=np.float32)
    if start > end or mode in {"static", "sam3"}:
        return result
    frame_buckets = parsed.get("frames", {})
    if not isinstance(frame_buckets, Mapping):
        raise MaskDataError("parsed mask frames must be an object")
    explicit = {int(frame) for frame in parsed.get("anchors", [])}
    _final, result, _anchors, _baseline = _tracking_anchors_and_bases(
        images, frame_buckets, explicit, start, end, tracking_mode=mode
    )
    return result


def _source_identity(images: np.ndarray) -> str:
    """Hash the complete source content for collision-safe cache isolation."""

    array = np.asarray(images)
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(repr(tuple(int(value) for value in array.shape)).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _raw_tracking_cache_key(
    images: np.ndarray,
    parsed: Mapping[str, Any],
    tracking_mode: str,
    start: int,
    end: int,
) -> tuple[Any, ...]:
    """Build a key that excludes only grow/feather post-processing controls."""

    # Include function identities so a test/runtime hot-patch of the tracking
    # implementation cannot reuse a result produced by a different engine.
    document_key = json.dumps(parsed, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return (
        "ths-raw-tracking-v1",
        _source_identity(images),
        document_key,
        str(tracking_mode),
        int(start),
        int(end),
        id(rasterize_events),
        id(_flow_between),
        id(_warp_mask),
        id(propagate_masks),
    )


def _cache_unpack_array(payload: bytes, shape: tuple[int, ...], dtype_name: str) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    expected = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    if expected < 0 or expected > MAX_RAW_TRACKING_CACHE_ITEM_BYTES * 16:
        raise ValueError("raw tracking cache array exceeds the safe decode bound")
    raw = zlib.decompress(payload)
    if len(raw) != expected:
        raise ValueError("raw tracking cache payload size mismatch")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _raw_tracking_cache_get(
    key: tuple[Any, ...],
) -> tuple[np.ndarray, np.ndarray, set[int]] | None:
    global _raw_tracking_cache_bytes, _raw_tracking_cache_hits, _raw_tracking_cache_misses
    with _raw_tracking_cache_lock:
        entry = _raw_tracking_cache.get(key)
        if entry is None:
            _raw_tracking_cache_misses += 1
            return None
        try:
            mask_payload, mask_shape, mask_dtype, edit_payload, edit_shape, edit_dtype, anchors, _item_bytes = entry
            mask = _cache_unpack_array(mask_payload, mask_shape, mask_dtype)
            edit_bases = _cache_unpack_array(edit_payload, edit_shape, edit_dtype)
        except Exception:
            # Cache entries are process-local and should never be trusted after
            # an unexpected decode failure.  Remove only this owned entry.
            _raw_tracking_cache.pop(key, None)
            _raw_tracking_cache_bytes = max(0, _raw_tracking_cache_bytes - int(entry[-1]))
            _raw_tracking_cache_misses += 1
            return None
        _raw_tracking_cache.move_to_end(key)
        _raw_tracking_cache_hits += 1
        return mask, edit_bases, set(int(frame) for frame in anchors)


def _raw_tracking_cache_put(
    key: tuple[Any, ...],
    mask: np.ndarray,
    edit_bases: np.ndarray,
    anchors: set[int],
) -> bool:
    """Store an immutable lossless entry if it fits the configured bounds."""

    global _raw_tracking_cache_bytes, _raw_tracking_cache_bypasses, _raw_tracking_cache_stores
    try:
        mask_array = np.ascontiguousarray(np.asarray(mask))
        edit_array = np.ascontiguousarray(np.asarray(edit_bases))
        mask_payload = zlib.compress(memoryview(mask_array).cast("B"), level=1)
        edit_payload = zlib.compress(memoryview(edit_array).cast("B"), level=1)
    except Exception:
        with _raw_tracking_cache_lock:
            _raw_tracking_cache_bypasses += 1
        return False

    item_bytes = int(len(mask_payload) + len(edit_payload))
    if (
        MAX_RAW_TRACKING_CACHE_ENTRIES < 1
        or MAX_RAW_TRACKING_CACHE_BYTES < 1
        or MAX_RAW_TRACKING_CACHE_ITEM_BYTES < 1
        or item_bytes > int(MAX_RAW_TRACKING_CACHE_ITEM_BYTES)
    ):
        with _raw_tracking_cache_lock:
            _raw_tracking_cache_bypasses += 1
        return False

    entry = (
        bytes(mask_payload),
        tuple(int(value) for value in mask_array.shape),
        mask_array.dtype.str,
        bytes(edit_payload),
        tuple(int(value) for value in edit_array.shape),
        edit_array.dtype.str,
        tuple(sorted(int(frame) for frame in anchors)),
        item_bytes,
    )
    with _raw_tracking_cache_lock:
        previous = _raw_tracking_cache.pop(key, None)
        if previous is not None:
            _raw_tracking_cache_bytes = max(0, _raw_tracking_cache_bytes - int(previous[-1]))
        while _raw_tracking_cache and (
            len(_raw_tracking_cache) >= int(MAX_RAW_TRACKING_CACHE_ENTRIES)
            or _raw_tracking_cache_bytes + item_bytes > int(MAX_RAW_TRACKING_CACHE_BYTES)
        ):
            _old_key, old_entry = _raw_tracking_cache.popitem(last=False)
            _raw_tracking_cache_bytes = max(0, _raw_tracking_cache_bytes - int(old_entry[-1]))
        if _raw_tracking_cache_bytes + item_bytes > int(MAX_RAW_TRACKING_CACHE_BYTES):
            _raw_tracking_cache_bypasses += 1
            return False
        _raw_tracking_cache[key] = entry
        _raw_tracking_cache_bytes += item_bytes
        _raw_tracking_cache_stores += 1
    return True


def _build_raw_tracking_sequence(
    images: np.ndarray,
    parsed: Mapping[str, Any],
    *,
    tracking_mode: str,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, set[int], dict[str, Any]]:
    """Build the pre-grow/feather tracking result and edit baselines."""

    count, height, width = map(int, images.shape[:3])
    mask = np.zeros((count, height, width), dtype=np.float32)
    if start > end:
        return mask, mask.copy(), set(), {}

    if tracking_mode == "sam3":
        from .mask_sam3_video import SAM3VideoError, _prompt_events, frame_result, track_video

        sam3 = parsed.get("sam3") if isinstance(parsed.get("sam3"), Mapping) else {}
        frame = int(sam3.get("frame", start)) if isinstance(sam3, Mapping) else int(start)
        action = str(sam3.get("action", "frame")) if isinstance(sam3, Mapping) else "frame"
        direction = str(sam3.get("direction", "forward")) if isinstance(sam3, Mapping) else "forward"
        if action == "frame":
            # Loading frames or an empty SAM3 draft is a source-preview pass,
            # not a failed refinement.  Do not touch the native model until a
            # marked object box exists on the selected seed frame.
            if not _prompt_events(parsed, frame):
                result = type("_SAM3EmptyResult", (), {
                    "masks": mask,
                    "edit_bases": mask.copy(),
                    "anchors": frozenset(),
                    "meta": {
                        "frame_ready": False,
                        "track_ready": False,
                        "frame": frame,
                        "seed_token": str(sam3.get("seed_token", "")) if isinstance(sam3, Mapping) else "",
                        "direction": direction,
                        "computed_frames": [],
                    },
                })()
            else:
                result = frame_result(images, parsed, frame=frame)
        elif action == "track":
            token = str(sam3.get("seed_token", "")) if isinstance(sam3, Mapping) else ""
            result = track_video(
                images,
                parsed,
                frame=frame,
                token=token,
                start=start,
                end=end,
                direction=direction,
            )
        else:  # parse_mask_data normally prevents this; retain a typed guard for hand-built mappings.
            raise SAM3VideoError("SAM3 action must be frame or track")
        return result.masks, result.edit_bases, set(result.anchors), dict(result.meta)

    frame_buckets = parsed.get("frames", {})
    if tracking_mode == "static":
        events: list[Mapping[str, Any]] = []
        for key in sorted(frame_buckets, key=lambda item: int(item)):
            events.extend(_event_list(frame_buckets[key]))
        static_mask = (
            rasterize_events(events, height, width, images[start])
            if events else np.zeros((height, width), dtype=np.float32)
        )
        for frame in range(start, end + 1):
            mask[frame] = static_mask
        return mask, np.zeros_like(mask), ({start} if events else set()), {}

    explicit = set(int(frame) for frame in parsed.get("anchors", []))
    anchor_masks, edit_bases, anchors, independent_sequence = _tracking_anchors_and_bases(
        images, frame_buckets, explicit, start, end, tracking_mode=tracking_mode
    )
    if anchor_masks:
        mask = (
            _hold_masks(images, anchor_masks, start, end)
            if tracking_mode == "keyframes"
            else propagate_masks(images, anchor_masks, start, end, phase="optical_flow_corrections")
            if any(_frame_inherits(bucket) for bucket in frame_buckets.values())
            else independent_sequence
        )
    return mask, edit_bases, anchors, {}


def _apply_layer_blend(target: np.ndarray, src: np.ndarray, blend: str) -> None:
    if blend == "subtract":
        np.subtract(target, src, out=target)
        np.maximum(target, 0.0, out=target)
    elif blend == "intersect":
        np.minimum(target, src, out=target)
    else:
        np.maximum(target, src, out=target)


def _compute_layer_raw(
    images: np.ndarray,
    doc: Mapping[str, Any],
    *,
    tracking_mode: str,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, set[int], dict[str, Any]]:
    global _raw_tracking_cache_bypasses
    count, height, width = map(int, images.shape[:3])
    cache_allowed = (
        tracking_mode != "sam3"
        and int(MAX_RAW_TRACKING_CACHE_ENTRIES) > 0
        and int(MAX_RAW_TRACKING_CACHE_BYTES) > 0
        and int(MAX_RAW_TRACKING_CACHE_ITEM_BYTES) > 0
        and int(images.nbytes) <= int(MAX_RAW_TRACKING_CACHE_SOURCE_BYTES)
    )
    cached: tuple[np.ndarray, np.ndarray, set[int]] | None = None
    cache_key: tuple[Any, ...] | None = None
    if cache_allowed:
        cache_key = _raw_tracking_cache_key(images, doc, tracking_mode, start, end)
        cached = _raw_tracking_cache_get(cache_key)
    else:
        with _raw_tracking_cache_lock:
            _raw_tracking_cache_bypasses += 1

    sam3_meta: dict[str, Any] = {}
    if cached is None:
        built = _build_raw_tracking_sequence(
            images, doc, tracking_mode=tracking_mode, start=start, end=end
        )
        raw_mask, edit_bases, anchors, sam3_meta = built
        if cache_allowed and cache_key is not None:
            _raw_tracking_cache_put(cache_key, raw_mask, edit_bases, anchors)
    else:
        raw_mask, edit_bases, anchors = cached

    if tracking_mode == "sam3" and sam3_meta.get("seed_token"):
        parsed_sam3 = doc.get("sam3") if isinstance(doc, Mapping) else None
        if isinstance(parsed_sam3, dict):
            parsed_sam3["seed_token"] = str(sam3_meta["seed_token"])

    if tracking_mode == "sam3":
        # Apply only frame-local brush/erase/ordinary rectangle events.  Native
        # SAM3 prompts are excluded by ``_overlay_events`` and remain the sole
        # source of tracker anchors/object identity.
        from .mask_sam3_video import _overlay_events

        # ``mask_sam3_video`` may intentionally alias the immutable raw mask as
        # its edit baseline to avoid retaining a second multi-GB array.  The
        # browser baseline must stay pre-brush, so detach only for this pass.
        if np.shares_memory(raw_mask, edit_bases):
            edit_bases = np.asarray(edit_bases, dtype=np.float32).copy()
        computed_frames = {
            int(frame)
            for frame in sam3_meta.get("computed_frames", [])
            if 0 <= int(frame) < count
        }
        for frame in range(start, end + 1):
            if computed_frames and frame not in computed_frames:
                raw_mask[frame] = 0.0
                continue
            if not computed_frames:
                raw_mask[frame] = 0.0
                continue
            events = _overlay_events(doc, frame)
            if events:
                raw_mask[frame] = rasterize_events(
                    events, height, width, images[frame], raw_mask[frame]
                )

    return raw_mask, edit_bases, anchors, sam3_meta


def build_mask_sequence(images: np.ndarray, mask_data: Any = "{}", *, tracking_mode: str = "optical_flow",
                        grow: int = 8, feather: int = 4, start_frame: int = 0,
                        end_frame: int = -1, fingerprint: str | None = None,
                        return_edit_bases: bool = False,
                        return_sam3_meta: bool = False,
                        return_raw_mask: bool = False,
                        return_layers_meta: bool = False,
                        return_layer_masks: bool = False) -> tuple:
    """Build a safe ``TxHxW`` mask and return ``(mask, canonical_doc, anchors)``.

    Empty or malformed edits never become a full-frame mask: malformed input
    raises ``MaskDataError`` and an empty document returns all zeros.
    ``return_edit_bases`` appends the raw editing baseline without running
    optical flow again when the node also needs browser preview assets.
    """

    images = np.asarray(images)
    if images.ndim != 4 or images.shape[0] < 1 or images.shape[1] < 1 or images.shape[2] < 1:
        raise MaskDataError("images must have shape T x H x W x C and contain at least one frame")
    count, height, width = map(int, images.shape[:3])
    if count > MAX_FRAMES:
        raise MaskDataError("clips longer than 4096 frames are not supported by this editor")
    parsed = parse_mask_data(mask_data, count, width=width, height=height, fingerprint=fingerprint)
    layers_meta: list[dict[str, Any]] = []
    layer_masks: dict[str, np.ndarray] = {}

    if "layers" in parsed:
        active_id = str(parsed.get("active_layer_id") or parsed["layers"][0]["id"])
        raw_mask = np.zeros((count, height, width), dtype=np.float32)
        edit_bases = np.zeros((count, height, width), dtype=np.float32)
        anchors: set[int] = set()
        sam3_meta: dict[str, Any] = {}
        emit_layer_masks = bool(return_layer_masks and len(parsed["layers"]) > 1)

        for layer in parsed["layers"]:
            if not layer.get("enabled", True):
                layers_meta.append({
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

            l_start = max(0, int(layer.get("start_frame", 0)))
            l_end_raw = int(layer.get("end_frame", -1))
            l_end = count - 1 if l_end_raw < 0 else min(count - 1, l_end_raw)
            layer_doc: dict[str, Any] = {
                "version": 1,
                "frames": copy.deepcopy(layer.get("frames", {})),
                "tracking_mode": layer["type"],
            }
            if layer.get("anchors"):
                layer_doc["anchors"] = list(layer["anchors"])
            if layer["type"] == "sam3" and isinstance(layer.get("sam3"), Mapping):
                layer_doc["sam3"] = copy.deepcopy(layer["sam3"])

            l_raw, l_edit_bases, l_anchors, l_sam3 = _compute_layer_raw(
                images, layer_doc, tracking_mode=layer["type"], start=l_start, end=l_end
            )
            if layer["type"] == "sam3" and isinstance(layer_doc.get("sam3"), Mapping):
                if layer_doc["sam3"].get("seed_token"):
                    layer.setdefault("sam3", {})["seed_token"] = str(layer_doc["sam3"]["seed_token"])
                    if layer["id"] == active_id and isinstance(parsed.get("sam3"), dict):
                        parsed["sam3"]["seed_token"] = str(layer_doc["sam3"]["seed_token"])

            blend = layer["blend"]
            _apply_layer_blend(raw_mask, l_raw, blend)
            base_contrib = l_edit_bases if layer["id"] == active_id else l_raw
            _apply_layer_blend(edit_bases, base_contrib, blend)
            if emit_layer_masks:
                layer_masks[str(layer["id"])] = l_raw
            anchors |= l_anchors
            if layer["id"] == active_id and l_sam3:
                sam3_meta = dict(l_sam3)

            layers_meta.append({
                "id": layer["id"],
                "name": layer["name"],
                "type": layer["type"],
                "blend": blend,
                "enabled": True,
                "start_frame": int(layer.get("start_frame", 0)),
                "end_frame": int(layer.get("end_frame", -1)),
                "has_mask": bool(np.any(l_raw > 1e-5)),
                "anchors": sorted(int(f) for f in l_anchors),
                **({"sam3": dict(l_sam3)} if layer["type"] == "sam3" and l_sam3 else {}),
            })
    else:
        tracking_mode = str(tracking_mode or parsed.get("tracking_mode", "optical_flow"))
        if tracking_mode not in {"optical_flow", "static", "keyframes", "sam3"}:
            raise MaskDataError("tracking_mode must be optical_flow, static, keyframes, or sam3")
        start = max(0, int(start_frame))
        end = count - 1 if int(end_frame) < 0 else min(count - 1, int(end_frame))
        raw_mask, edit_bases, anchors, sam3_meta = _compute_layer_raw(
            images, parsed, tracking_mode=tracking_mode, start=start, end=end
        )

    mask = postprocess_mask(raw_mask, grow=grow, feather=feather)
    # Tracking already preserves the painted anchors. Apply edge controls to
    # every frame, including anchors, so keyframes cannot produce hard-edge pops.
    outputs: list[Any] = [mask, parsed, anchors]
    if return_edit_bases:
        outputs.append(edit_bases)
    if return_sam3_meta:
        outputs.append(sam3_meta)
    if return_raw_mask:
        outputs.append(raw_mask)
    if return_layers_meta:
        outputs.append(layers_meta)
    if return_layer_masks:
        outputs.append(layer_masks)
    return tuple(outputs)


def _image_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise MaskDataError("preview images must be H x W x C")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] < 3:
        raise MaskDataError("preview images need at least one channel")
    return np.asarray(np.round(np.clip(np.nan_to_num(image[..., :3]), 0.0, 1.0) * 255.0), dtype=np.uint8)


def _data_url(payload: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


def _preview_buffers(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    max_size: int = MAX_PREVIEW_SIZE,
) -> tuple[bytes, bytes, tuple[int, int]]:
    """Encode one source/mask pair without constructing a websocket payload."""

    from PIL import Image

    bounded_size = max(1, min(int(max_size), MAX_PREVIEW_SIZE))
    source = Image.fromarray(_image_uint8(image), mode="RGB")
    source.thumbnail((bounded_size, bounded_size), Image.Resampling.LANCZOS)
    mask_payload = _mask_png_payload(mask, source.size)
    source_buffer = BytesIO()
    source.save(source_buffer, format="JPEG", quality=86, optimize=True)
    return source_buffer.getvalue(), mask_payload, (int(source.size[0]), int(source.size[1]))


def _preview_rgb_buffer(image: np.ndarray, *, max_size: int = MAX_PREVIEW_SIZE) -> tuple[bytes, tuple[int, int]]:
    """Encode one bounded RGB JPEG for compare previews."""

    from PIL import Image

    bounded_size = max(1, min(int(max_size), MAX_PREVIEW_SIZE))
    preview = Image.fromarray(_image_uint8(image), mode="RGB")
    preview.thumbnail((bounded_size, bounded_size), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=86, optimize=True)
    return buffer.getvalue(), (int(preview.size[0]), int(preview.size[1]))


def _mask_png_payload(mask: np.ndarray, size: tuple[int, int]) -> bytes:
    """Encode a grayscale mask at the already-selected preview size."""

    from PIL import Image

    mask_array = np.asarray(np.round(np.clip(np.nan_to_num(mask), 0.0, 1.0) * 255.0), dtype=np.uint8)
    if mask_array.ndim != 2:
        raise MaskDataError("preview masks must contain H x W frames")
    mask_image = Image.fromarray(mask_array, mode="L")
    mask_image = mask_image.resize(size, Image.Resampling.NEAREST)
    mask_buffer = BytesIO()
    mask_image.save(mask_buffer, format="PNG", optimize=True)
    return mask_buffer.getvalue()


def _owned_cache_directories(temp_root: Path) -> list[Path]:
    """Return only direct child cache directories owned by this module."""

    root = temp_root / TEMP_CACHE_ROOT
    if not root.is_dir():
        return []
    try:
        resolved_root = root.resolve()
        if resolved_root.parent != temp_root.resolve():
            return []
        candidates: list[Path] = []
        for child in root.iterdir():
            if (not child.is_dir() or child.is_symlink()
                    or not re.fullmatch(r"session-[A-Za-z0-9_.-]{1,48}-[0-9a-f]{32}", child.name)
                    or not (child / ".ths-preview-owned").is_file()):
                continue
            # Refuse symlinked/out-of-root paths even when a hostile temp
            # directory is supplied by a test or another local process.
            if child.resolve().parent != resolved_root:
                continue
            candidates.append(child)
        return candidates
    except OSError:
        return []


def _cache_directory_size(path: Path) -> int:
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += int(item.stat().st_size)
                except OSError:
                    continue
    except OSError:
        return total
    return total


def _cleanup_owned_preview_cache(temp_root: Path, *, protected: Path | None = None) -> None:
    """Bound this module's temp cache without touching unrelated temp files."""

    candidates = _owned_cache_directories(temp_root)
    if not candidates:
        return
    protected_resolved = protected.resolve() if protected is not None else None

    def size() -> int:
        return sum(_cache_directory_size(path) for path in candidates if path.exists())

    while len(candidates) > MAX_TEMP_CACHE_SESSIONS or size() > MAX_TEMP_CACHE_BYTES:
        removable = [
            path for path in candidates
            if protected_resolved is None or path.resolve() != protected_resolved
        ]
        if not removable:
            break
        removable.sort(key=lambda path: path.stat().st_mtime_ns if path.exists() else 0)
        victim = removable[0]
        try:
            shutil.rmtree(victim)
        except OSError:
            # A concurrent Comfy request may still be serving a file.  Leave
            # it alone and stop rather than widening cleanup scope.
            break
        candidates = [path for path in candidates if path != victim]


def cache_preview_assets(
    images: np.ndarray,
    masks: np.ndarray,
    *,
    edit_bases: np.ndarray | None = None,
    raw_masks: np.ndarray | None = None,
    layer_masks: Mapping[str, np.ndarray] | None = None,
    max_size: int = MAX_PREVIEW_SIZE,
    source_session: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Write immutable all-frame previews below Comfy's temp directory.

    The returned descriptors are consumed by Comfy's existing ``/view`` route
    (``type=temp``); raw pixels are never put into the websocket payload.  A
    missing ``folder_paths`` module or an unwritable temp directory is a
    normal, diagnosable fallback rather than a node failure.
    """

    images = np.asarray(images)
    masks = np.asarray(masks)
    if images.ndim != 4 or masks.ndim != 3 or images.shape[0] != masks.shape[0]:
        raise MaskDataError("preview images/masks must have matching T dimensions")
    if edit_bases is not None:
        edit_bases = np.asarray(edit_bases)
        if edit_bases.shape != masks.shape:
            raise MaskDataError("preview edit bases must match mask dimensions")
    if raw_masks is not None:
        raw_masks = np.asarray(raw_masks)
        if raw_masks.shape != masks.shape:
            raise MaskDataError("preview raw masks must match mask dimensions")
    validated_layer_masks: dict[str, np.ndarray] = {}
    if layer_masks:
        for layer_id, layer_arr in layer_masks.items():
            arr = np.asarray(layer_arr)
            if arr.shape != masks.shape:
                raise MaskDataError("preview layer masks must match mask dimensions")
            validated_layer_masks[str(layer_id)] = arr
    try:
        from PIL import Image  # noqa: F401  # imported for an actionable fallback below
        import folder_paths  # type: ignore
    except Exception as exc:
        return [], {"available": False, "error": f"temp preview cache unavailable: {exc}"}

    try:
        getter = getattr(folder_paths, "get_temp_directory", None)
        raw_temp = getter() if callable(getter) else getattr(folder_paths, "temp_directory", None)
        if not raw_temp:
            raise RuntimeError("folder_paths has no temp directory")
        temp_root = Path(str(raw_temp)).expanduser().resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        cache_root = temp_root / TEMP_CACHE_ROOT
        if cache_root.resolve().parent != temp_root:
            raise RuntimeError("preview cache directory resolves outside Comfy temp")
        cache_root.mkdir(parents=True, exist_ok=True)
        _cleanup_owned_preview_cache(temp_root)

        source_tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(source_session or "source"))[:48] or "source"
        session_name = f"{TEMP_CACHE_PREFIX}{source_tag}-{uuid.uuid4().hex}"
        cache_dir = cache_root / session_name
        cache_dir.mkdir(parents=False, exist_ok=False)
        (cache_dir / ".ths-preview-owned").write_text("ths-mask-preview-v1\n", encoding="ascii")
        subfolder = str(cache_dir.relative_to(temp_root)).replace("\\", "/")
        descriptors: list[dict[str, Any]] = []
        written = 0
        try:
            for frame in range(int(images.shape[0])):
                source_bytes, mask_bytes, size = _preview_buffers(
                    images[frame], masks[frame], max_size=max_size
                )
                edit_bytes = None
                if edit_bases is not None:
                    edit_bytes = _mask_png_payload(edit_bases[frame], size)
                raw_bytes = None
                if raw_masks is not None:
                    raw_bytes = _mask_png_payload(raw_masks[frame], size)
                written += (
                    len(source_bytes)
                    + len(mask_bytes)
                    + (len(edit_bytes) if edit_bytes else 0)
                    + (len(raw_bytes) if raw_bytes else 0)
                )
                if written > MAX_TEMP_CACHE_SESSION_BYTES:
                    raise RuntimeError(
                        f"all-frame preview cache exceeds {MAX_TEMP_CACHE_SESSION_BYTES} bytes"
                    )
                source_name = f"frame_{frame:06d}.jpg"
                mask_name = f"mask_{frame:06d}.png"
                (cache_dir / source_name).write_bytes(source_bytes)
                (cache_dir / mask_name).write_bytes(mask_bytes)
                descriptor: dict[str, Any] = {
                    "frame": frame,
                    "filename": source_name,
                    "subfolder": subfolder,
                    "type": "temp",
                    "width": size[0],
                    "height": size[1],
                    "mask": {
                        "filename": mask_name,
                        "subfolder": subfolder,
                        "type": "temp",
                    },
                }
                if edit_bytes is not None:
                    edit_name = f"edit_base_{frame:06d}.png"
                    (cache_dir / edit_name).write_bytes(edit_bytes)
                    descriptor["edit_base"] = {
                        "filename": edit_name,
                        "subfolder": subfolder,
                        "type": "temp",
                    }
                if raw_bytes is not None:
                    raw_name = f"raw_mask_{frame:06d}.png"
                    (cache_dir / raw_name).write_bytes(raw_bytes)
                    descriptor["raw_mask"] = {
                        "filename": raw_name,
                        "subfolder": subfolder,
                        "type": "temp",
                    }
                if validated_layer_masks:
                    layer_desc: dict[str, dict[str, str]] = {}
                    for layer_idx, (layer_id, layer_arr) in enumerate(validated_layer_masks.items()):
                        if not bool(np.any(layer_arr[frame] > 1e-5)):
                            continue
                        layer_bytes = _mask_png_payload(layer_arr[frame], size)
                        written += len(layer_bytes)
                        if written > MAX_TEMP_CACHE_SESSION_BYTES:
                            raise RuntimeError(
                                f"all-frame preview cache exceeds {MAX_TEMP_CACHE_SESSION_BYTES} bytes"
                            )
                        layer_name = f"layer_{layer_idx}_{frame:06d}.png"
                        (cache_dir / layer_name).write_bytes(layer_bytes)
                        layer_desc[layer_id] = {
                            "filename": layer_name,
                            "subfolder": subfolder,
                            "type": "temp",
                        }
                    descriptor["has_layer_masks"] = True
                    if layer_desc:
                        descriptor["layer_masks"] = layer_desc
                descriptors.append(descriptor)
        except Exception:
            shutil.rmtree(cache_dir, ignore_errors=True)
            raise

        # The cache directory is immutable after this point.  Protect the
        # current session while evicting older owned sessions.
        _cleanup_owned_preview_cache(temp_root, protected=cache_dir)
        return descriptors, {
            "available": True,
            "error": None,
            "session": session_name,
            "subfolder": subfolder,
            "frames": len(descriptors),
            "bytes": written,
        }
    except Exception as exc:
        return [], {"available": False, "error": f"temp preview cache unavailable: {exc}"}


def cache_compare_preview_assets(
    source: np.ndarray,
    result: np.ndarray,
    *,
    max_size: int = MAX_PREVIEW_SIZE,
    source_session: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Write bounded original/result JPEG descriptors for synchronized review."""

    source = np.asarray(source)
    result = np.asarray(result)
    if source.ndim != 4 or result.ndim != 4:
        raise MaskDataError("compare source/result must have shape T x H x W x C")
    if tuple(source.shape[:3]) != tuple(result.shape[:3]):
        raise MaskDataError("compare source/result must have matching frame and canvas geometry")
    try:
        from PIL import Image  # noqa: F401  # imported for an actionable fallback below
        import folder_paths  # type: ignore
    except Exception as exc:
        return [], {"available": False, "error": f"temp preview cache unavailable: {exc}"}

    try:
        getter = getattr(folder_paths, "get_temp_directory", None)
        raw_temp = getter() if callable(getter) else getattr(folder_paths, "temp_directory", None)
        if not raw_temp:
            raise RuntimeError("folder_paths has no temp directory")
        temp_root = Path(str(raw_temp)).expanduser().resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        cache_root = temp_root / TEMP_CACHE_ROOT
        if cache_root.resolve().parent != temp_root:
            raise RuntimeError("preview cache directory resolves outside Comfy temp")
        cache_root.mkdir(parents=True, exist_ok=True)
        _cleanup_owned_preview_cache(temp_root)

        source_tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(source_session or "compare"))[:48] or "compare"
        session_name = f"{TEMP_CACHE_PREFIX}{source_tag}-{uuid.uuid4().hex}"
        cache_dir = cache_root / session_name
        cache_dir.mkdir(parents=False, exist_ok=False)
        (cache_dir / ".ths-preview-owned").write_text("ths-repair-compare-v1\n", encoding="ascii")
        subfolder = str(cache_dir.relative_to(temp_root)).replace("\\", "/")
        descriptors: list[dict[str, Any]] = []
        written = 0
        try:
            for frame in range(int(source.shape[0])):
                original_bytes, original_size = _preview_rgb_buffer(source[frame], max_size=max_size)
                result_bytes, result_size = _preview_rgb_buffer(result[frame], max_size=max_size)
                if original_size != result_size:
                    raise RuntimeError("compare preview dimensions diverged")
                written += len(original_bytes) + len(result_bytes)
                if written > MAX_TEMP_CACHE_SESSION_BYTES:
                    raise RuntimeError(
                        f"compare preview cache exceeds {MAX_TEMP_CACHE_SESSION_BYTES} bytes"
                    )
                original_name = f"original_{frame:06d}.jpg"
                result_name = f"result_{frame:06d}.jpg"
                (cache_dir / original_name).write_bytes(original_bytes)
                (cache_dir / result_name).write_bytes(result_bytes)
                descriptors.append({
                    "frame": int(frame),
                    "width": original_size[0],
                    "height": original_size[1],
                    "original": {
                        "filename": original_name,
                        "subfolder": subfolder,
                        "type": "temp",
                    },
                    "result": {
                        "filename": result_name,
                        "subfolder": subfolder,
                        "type": "temp",
                    },
                })
        except Exception:
            shutil.rmtree(cache_dir, ignore_errors=True)
            raise

        _cleanup_owned_preview_cache(temp_root, protected=cache_dir)
        return descriptors, {
            "available": True,
            "error": None,
            "session": session_name,
            "subfolder": subfolder,
            "frames": len(descriptors),
            "bytes": written,
        }
    except Exception as exc:
        return [], {"available": False, "error": f"temp preview cache unavailable: {exc}"}


def encode_preview_assets(images: np.ndarray, masks: np.ndarray, *, max_frames: int = MAX_PREVIEW_FRAMES,
                          max_size: int = MAX_PREVIEW_SIZE,
                          edit_bases: np.ndarray | None = None,
                          raw_masks: np.ndarray | None = None,
                          layer_masks: Mapping[str, np.ndarray] | None = None) -> list[dict[str, Any]]:
    """Encode bounded source/mask previews for the Comfy ``ui`` channel."""

    if images.ndim != 4 or masks.ndim != 3 or images.shape[0] != masks.shape[0]:
        raise MaskDataError("preview images/masks must have matching T dimensions")
    if edit_bases is not None:
        edit_bases = np.asarray(edit_bases)
        if edit_bases.shape != masks.shape:
            raise MaskDataError("preview edit bases must match mask dimensions")
    if raw_masks is not None:
        raw_masks = np.asarray(raw_masks)
        if raw_masks.shape != masks.shape:
            raise MaskDataError("preview raw masks must match mask dimensions")
    validated_layer_masks: dict[str, np.ndarray] = {}
    if layer_masks:
        for layer_id, layer_arr in layer_masks.items():
            arr = np.asarray(layer_arr)
            if arr.shape != masks.shape:
                raise MaskDataError("preview layer masks must match mask dimensions")
            validated_layer_masks[str(layer_id)] = arr
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        return []
    count = int(images.shape[0])
    limit = max(1, min(int(max_frames), MAX_PREVIEW_FRAMES))
    if count <= limit:
        indices = list(range(count))
    else:
        indices = sorted(set(np.linspace(0, count - 1, limit, dtype=np.int64).tolist()))
    result: list[dict[str, Any]] = []
    for frame in indices:
        source_payload, mask_payload, size = _preview_buffers(
            images[frame], masks[frame], max_size=max_size
        )
        entry: dict[str, Any] = {
            "frame": int(frame),
            "width": size[0],
            "height": size[1],
            "data_url": _data_url(source_payload, "image/jpeg"),
            "mask_url": _data_url(mask_payload, "image/png"),
        }
        if edit_bases is not None:
            entry["edit_base"] = _data_url(_mask_png_payload(edit_bases[frame], size), "image/png")
        if raw_masks is not None:
            entry["raw_mask"] = _data_url(_mask_png_payload(raw_masks[frame], size), "image/png")
        if validated_layer_masks:
            layer_urls = {
                layer_id: _data_url(_mask_png_payload(layer_arr[frame], size), "image/png")
                for layer_id, layer_arr in validated_layer_masks.items()
                if bool(np.any(layer_arr[frame] > 1e-5))
            }
            entry["has_layer_masks"] = True
            if layer_urls:
                entry["layer_masks"] = layer_urls
        result.append(entry)
    return result


__all__ = [
    "MAX_FRAMES", "MAX_EVENTS", "MAX_POINTS", "MAX_PREVIEW_FRAMES", "MAX_PREVIEW_SIZE", "MAX_MASK_LAYERS",
    "MAX_TEMP_CACHE_BYTES", "MAX_TEMP_CACHE_SESSIONS", "MAX_TEMP_CACHE_SESSION_BYTES",
    "MAX_RAW_TRACKING_CACHE_ENTRIES", "MAX_RAW_TRACKING_CACHE_BYTES",
    "MAX_RAW_TRACKING_CACHE_ITEM_BYTES", "MAX_RAW_TRACKING_CACHE_SOURCE_BYTES",
    "MaskDataError", "MaskSourceMismatch", "MaskRefinementError",
    "frame_fingerprint", "optical_flow_available", "normalize_crop_settings", "mask_extent", "plan_crop_window",
    "parse_crop_info", "restore_crop_window",
    "parse_mask_data", "canonical_mask_data", "canonicalize_mask_data",
    "rasterize_events", "refine_rectangle", "propagate_masks", "postprocess_mask",
    "derive_edit_base_sequence", "build_mask_sequence", "encode_preview_assets", "cache_preview_assets",
    "cache_compare_preview_assets", "clear_raw_tracking_cache", "raw_tracking_cache_info",
]
