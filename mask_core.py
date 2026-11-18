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


def parse_mask_data(value: Any, frame_count: int | None = None, *, width: int | None = None,
                    height: int | None = None, fingerprint: str | None = None) -> dict[str, Any]:
    """Validate and canonicalize the browser's normalized edit document.

    The parser accepts a compact top-level ``edits`` list for imported masks,
    while returning only the stable frame-grouped representation used by the
    node and browser.  ``source`` is optional for hand-authored data, but when
    present its frame count and dimensions must match the current clip.
    """

    raw = _json_value(value)
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
            # only representation that preserves rectangle→erase→polygon
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
        raise MaskDataError("mask_data.frames must be an object keyed by frame number")
    for key, payload in grouped.items():
        frame_key: Any = key
        if isinstance(key, str) and key.strip():
            try:
                frame_key = int(key)
            except ValueError as exc:
                raise MaskDataError(f"frames key {key!r} must be an integer") from exc
        ingest(frame_key, payload, f"frames[{key!r}]")

    # Imported compact records: each record has a frame field and type.
    edits = raw.get("edits", [])
    if edits is not None:
        if not isinstance(edits, (list, tuple)):
            raise MaskDataError("mask_data.edits must be a list")
        for index, event in enumerate(edits):
            if not isinstance(event, Mapping) or "frame" not in event:
                raise MaskDataError(f"edits[{index}] must include frame")
            ingest(event["frame"], [event], f"edits[{index}]")

    # ``keys``/``keyframes`` are accepted as aliases for grouped frame data;
    # their explicit frame list is retained below to keep anchor intent clear.
    explicit_anchors: list[int] = []
    anchors = raw.get("anchors", raw.get("keyframes", raw.get("keys", [])))
    if anchors is not None:
        if not isinstance(anchors, (list, tuple)):
            raise MaskDataError("mask_data.anchors must be a list")
        for index, item in enumerate(anchors):
            if isinstance(item, Mapping):
                if "frame" not in item:
                    raise MaskDataError(f"anchors[{index}] is missing frame")
                frame = _frame_number(item["frame"], frame_count, f"anchors[{index}].frame")
                explicit_anchors.append(frame)
                if any(key in item for key in ("inherit", "events", "strokes", "rects", "rectangles", "polygons", "edits")):
                    ingest(frame, item, f"anchors[{index}]")
            else:
                explicit_anchors.append(_frame_number(item, frame_count, f"anchors[{index}]"))

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
    canonical_source: dict[str, Any] = {}
    if declared_frames is not None:
        canonical_source["frames"] = int(declared_frames)
    if declared_width is not None:
        canonical_source["width"] = int(declared_width)
    if declared_height is not None:
        canonical_source["height"] = int(declared_height)
    if declared_fingerprint is not None:
        canonical_source["fingerprint"] = str(declared_fingerprint)
    result: dict[str, Any] = {"version": 1, "frames": canonical_frames}
    if canonical_source:
        result["source"] = canonical_source
    if explicit_anchors:
        result["anchors"] = sorted(set(explicit_anchors))
    if raw.get("tracking_mode") is not None:
        mode = str(raw["tracking_mode"])
        if mode not in {"optical_flow", "static", "keyframes", "sam3"}:
            raise MaskDataError("tracking_mode must be optical_flow, static, keyframes, or sam3")
        result["tracking_mode"] = mode
    # Stateful SAM3 video control is deliberately explicit and source-bound.
    # Keep the small action/frame/direction/token contract in canonical data;
    # unknown fields are not allowed to influence cache identity or tracking.
    sam3_raw = raw.get("sam3")
    if sam3_raw is not None:
        if not isinstance(sam3_raw, Mapping):
            raise MaskDataError("mask_data.sam3 must be an object")
        action = str(sam3_raw.get("action", "frame"))
        if action not in {"frame", "track"}:
            raise MaskDataError("mask_data.sam3.action must be 'frame' or 'track'")
        frame = _frame_number(sam3_raw.get("frame", 0), frame_count, "sam3.frame")
        direction = str(sam3_raw.get("direction", "forward"))
        if direction not in {"forward", "backward", "both"}:
            raise MaskDataError("mask_data.sam3.direction must be forward, backward, or both")
        token = sam3_raw.get("seed_token", "")
        if token is None:
            token = ""
        if not isinstance(token, (str, int, float)) or isinstance(token, bool):
            raise MaskDataError("mask_data.sam3.seed_token must be a string")
        token = str(token)
        if len(token) > 256:
            raise MaskDataError("mask_data.sam3.seed_token is too long")
        sam3 = {
            "action": action,
            "frame": frame,
            "direction": direction,
            "seed_token": token,
        }
        # An optional explicit prompt is accepted for clients that keep the
        # object box outside the per-frame event log.  It is normalized into a
        # bounded rectangle and does not change the legacy edit schema.
        prompt = sam3_raw.get("prompt")
        if prompt is not None:
            if not isinstance(prompt, Mapping):
                raise MaskDataError("mask_data.sam3.prompt must be an object")
            prompt_frame = _frame_number(prompt.get("frame", frame), frame_count, "sam3.prompt.frame")
            if all(name in prompt for name in ("x", "y", "w", "h")):
                prompt_event = {"type": "rect", "mode": "add", "sam3_prompt": True}
                for name in ("x", "y", "w", "h"):
                    prompt_event[name] = _finite_unit(prompt.get(name), f"sam3.prompt.{name}")
                if prompt_event["w"] <= 0 or prompt_event["h"] <= 0:
                    raise MaskDataError("sam3.prompt width and height must be positive")
                if prompt_event["x"] + prompt_event["w"] > 1.0 or prompt_event["y"] + prompt_event["h"] > 1.0:
                    raise MaskDataError("sam3.prompt extends outside [0, 1]")
                sam3["prompt"] = {"x": prompt_event["x"], "y": prompt_event["y"],
                                   "w": prompt_event["w"], "h": prompt_event["h"],
                                   "frame": prompt_frame}
            elif isinstance(prompt.get("points"), (list, tuple)):
                points = _points(prompt.get("points"), "sam3.prompt.points")
                labels = prompt.get("labels", [1] * len(points))
                if not isinstance(labels, (list, tuple)) or len(labels) != len(points):
                    raise MaskDataError("sam3.prompt.labels must match points")
                normalized_labels: list[int] = []
                for index, label in enumerate(labels):
                    if isinstance(label, bool) or not isinstance(label, (int, float)) or int(label) != label or int(label) not in {0, 1}:
                        raise MaskDataError(f"sam3.prompt.labels[{index}] must be 0 or 1")
                    normalized_labels.append(int(label))
                sam3["prompt"] = {"points": points, "labels": normalized_labels, "frame": prompt_frame}
            else:
                raise MaskDataError("sam3.prompt must contain a box or points")
        result["sam3"] = sam3
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
                    end_frame: int) -> np.ndarray:
    """Propagate from nearest explicit anchor while preserving anchors exactly."""

    count, height, width = images.shape[:3]
    result = np.zeros((count, height, width), dtype=np.float32)
    if not anchors:
        return result
    ordered = sorted(int(frame) for frame in anchors)
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
    result = np.clip(mask, 0.0, 1.0).copy()
    if cv2 is None:
        return result
    if grow:
        size = grow * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        result = np.stack([cv2.dilate(frame, kernel) for frame in result], axis=0)
    if feather:
        size = feather * 2 + 1
        result = np.stack([
            cv2.GaussianBlur(frame, (size, size), sigmaX=max(0.1, feather / 2.0))
            for frame in result
        ], axis=0)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


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
            else propagate_masks(images, anchor_masks, start, end)
            if any(_frame_inherits(bucket) for bucket in frame_buckets.values())
            else independent_sequence
        )
    return mask, edit_bases, anchors, {}


def build_mask_sequence(images: np.ndarray, mask_data: Any = "{}", *, tracking_mode: str = "optical_flow",
                        grow: int = 8, feather: int = 4, start_frame: int = 0,
                        end_frame: int = -1, fingerprint: str | None = None,
                        return_edit_bases: bool = False,
                        return_sam3_meta: bool = False) -> tuple:
    """Build a safe ``TxHxW`` mask and return ``(mask, canonical_doc, anchors)``.

    Empty or malformed edits never become a full-frame mask: malformed input
    raises ``MaskDataError`` and an empty document returns all zeros.
    ``return_edit_bases`` appends the raw editing baseline without running
    optical flow again when the node also needs browser preview assets.
    """

    global _raw_tracking_cache_bypasses
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[0] < 1 or images.shape[1] < 1 or images.shape[2] < 1:
        raise MaskDataError("images must have shape T x H x W x C and contain at least one frame")
    count, height, width = map(int, images.shape[:3])
    if count > MAX_FRAMES:
        raise MaskDataError("clips longer than 4096 frames are not supported by this editor")
    parsed = parse_mask_data(mask_data, count, width=width, height=height, fingerprint=fingerprint)
    tracking_mode = str(tracking_mode or parsed.get("tracking_mode", "optical_flow"))
    if tracking_mode not in {"optical_flow", "static", "keyframes", "sam3"}:
        raise MaskDataError("tracking_mode must be optical_flow, static, keyframes, or sam3")
    start = max(0, int(start_frame))
    end = count - 1 if int(end_frame) < 0 else min(count - 1, int(end_frame))
    # Native SAM3 video results have their own source/token-aware cache.  Do
    # not put local brush edits into the generic whole-document cache: a brush
    # change must recompose from the immutable native raw sequence without a
    # second model run.
    cache_allowed = (
        tracking_mode != "sam3"
        and
        int(MAX_RAW_TRACKING_CACHE_ENTRIES) > 0
        and int(MAX_RAW_TRACKING_CACHE_BYTES) > 0
        and int(MAX_RAW_TRACKING_CACHE_ITEM_BYTES) > 0
        and int(images.nbytes) <= int(MAX_RAW_TRACKING_CACHE_SOURCE_BYTES)
    )
    cached: tuple[np.ndarray, np.ndarray, set[int]] | None = None
    cache_key: tuple[Any, ...] | None = None
    if cache_allowed:
        cache_key = _raw_tracking_cache_key(images, parsed, tracking_mode, start, end)
        cached = _raw_tracking_cache_get(cache_key)
    else:
        with _raw_tracking_cache_lock:
            _raw_tracking_cache_bypasses += 1

    sam3_meta: dict[str, Any] = {}
    if cached is None:
        built = _build_raw_tracking_sequence(
            images, parsed, tracking_mode=tracking_mode, start=start, end=end
        )
        raw_mask, edit_bases, anchors, sam3_meta = built
        if cache_allowed and cache_key is not None:
            _raw_tracking_cache_put(cache_key, raw_mask, edit_bases, anchors)
    else:
        raw_mask, edit_bases, anchors = cached

    if tracking_mode == "sam3" and sam3_meta.get("seed_token"):
        parsed_sam3 = parsed.get("sam3")
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
            events = _overlay_events(parsed, frame)
            if events:
                raw_mask[frame] = rasterize_events(
                    events, height, width, images[frame], raw_mask[frame]
                )

    mask = postprocess_mask(raw_mask, grow=grow, feather=feather)
    # Tracking already preserves the painted anchors. Apply edge controls to
    # every frame, including anchors, so keyframes cannot produce hard-edge pops.
    if return_edit_bases and return_sam3_meta:
        return mask, parsed, anchors, edit_bases, sam3_meta
    if return_edit_bases:
        return mask, parsed, anchors, edit_bases
    if return_sam3_meta:
        return mask, parsed, anchors, sam3_meta
    return mask, parsed, anchors


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
                written += len(source_bytes) + len(mask_bytes) + (len(edit_bytes) if edit_bytes else 0)
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
                          max_size: int = MAX_PREVIEW_SIZE) -> list[dict[str, Any]]:
    """Encode bounded source/mask previews for the Comfy ``ui`` channel."""

    if images.ndim != 4 or masks.ndim != 3 or images.shape[0] != masks.shape[0]:
        raise MaskDataError("preview images/masks must have matching T dimensions")
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
        result.append({
            "frame": int(frame),
            "width": size[0],
            "height": size[1],
            "data_url": _data_url(source_payload, "image/jpeg"),
            "mask_url": _data_url(mask_payload, "image/png"),
        })
    return result


__all__ = [
    "MAX_FRAMES", "MAX_EVENTS", "MAX_POINTS", "MAX_PREVIEW_FRAMES", "MAX_PREVIEW_SIZE",
    "MAX_TEMP_CACHE_BYTES", "MAX_TEMP_CACHE_SESSIONS", "MAX_TEMP_CACHE_SESSION_BYTES",
    "MAX_RAW_TRACKING_CACHE_ENTRIES", "MAX_RAW_TRACKING_CACHE_BYTES",
    "MAX_RAW_TRACKING_CACHE_ITEM_BYTES", "MAX_RAW_TRACKING_CACHE_SOURCE_BYTES",
    "MaskDataError", "MaskSourceMismatch", "MaskRefinementError",
    "frame_fingerprint", "optical_flow_available", "parse_mask_data", "canonical_mask_data", "canonicalize_mask_data",
    "rasterize_events", "refine_rectangle", "propagate_masks", "postprocess_mask",
    "derive_edit_base_sequence", "build_mask_sequence", "encode_preview_assets", "cache_preview_assets",
    "cache_compare_preview_assets", "clear_raw_tracking_cache", "raw_tracking_cache_info",
]
