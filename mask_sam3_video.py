"""Stateful native SAM3 video adapter for the universal mask editor.

The browser/editor deliberately keeps the video contract small: a frame
calculation creates a source-bound seed token and a later track action reuses
that token.  This module owns the process-local seed/raw tracking caches and
keeps all native ComfyUI imports lazy so CPU tests can exercise the contract
without importing torch or starting a model.

No optical-flow fallback is present here.  A native failure is surfaced as a
typed ``SAM3VideoError`` and callers must block downstream execution.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import inspect
import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .mask_progress import frame_progress
from .mask_sam3 import (
    SAM3UnavailableError,
    _native_config,
    _native_model_cache_module,
    _native_nodes_root,
    _offload_model,
    _predict_native,
    _rectangle_bounds,
    _resize_mask,
)

log = logging.getLogger(__name__)


class SAM3VideoError(SAM3UnavailableError):
    """A native video seed/propagation prerequisite failed."""

    def __init__(self, message: str, *, meta: Mapping[str, Any] | None = None) -> None:
        self.sam3_meta = dict(meta or {})
        super().__init__(message)


@dataclass(frozen=True)
class SAM3Seed:
    token: str
    source_digest: str
    source_shape: tuple[int, int, int, int]
    frame: int
    prompt_digest: str
    box: tuple[float, float, float, float]
    mask: np.ndarray
    object_id: int = 1
    objects: tuple[tuple[str, np.ndarray], ...] = ()
    object_frames: tuple[int, ...] = ()
    refinement_masks: tuple[tuple[int, int, np.ndarray], ...] = ()


@dataclass(frozen=True)
class SAM3VideoResult:
    masks: np.ndarray
    edit_bases: np.ndarray
    anchors: frozenset[int]
    meta: dict[str, Any]


MAX_SEEDS = 32
MAX_RAW_RESULTS = 8
MAX_RAW_CACHE_BYTES = 256 * 1024 * 1024
MAX_RAW_CACHE_ITEM_BYTES = 128 * 1024 * 1024
MAX_OBJECT_TRACK_CACHE_BYTES = 128 * 1024 * 1024
_SEED_CACHE: "OrderedDict[str, SAM3Seed]" = OrderedDict()
_RAW_CACHE: "OrderedDict[tuple[Any, ...], tuple[bytes, tuple[int, ...], frozenset[int], dict[str, Any], int]]" = OrderedDict()
_OBJECT_TRACK_CACHE: "OrderedDict[tuple[str, int], tuple[bytes, tuple[int, ...], int]]" = OrderedDict()
_CACHE_LOCK = threading.RLock()
_VIDEO_INFERENCE_LOCK = threading.RLock()
_VIDEO_MODULE: Any | None = None
_VIDEO_ERROR: str | None = None
_LAST_NATIVE_SEED_STRATEGY = "unknown"
_RAW_CACHE_BYTES = 0
_RAW_CACHE_HITS = 0
_RAW_CACHE_MISSES = 0
_OBJECT_TRACK_CACHE_BYTES = 0


def clear_sam3_video_cache() -> None:
    """Drop process-local SAM3 video seeds and raw tracking results."""

    global _RAW_CACHE_BYTES, _RAW_CACHE_HITS, _RAW_CACHE_MISSES, _OBJECT_TRACK_CACHE_BYTES
    with _CACHE_LOCK:
        _SEED_CACHE.clear()
        _RAW_CACHE.clear()
        _OBJECT_TRACK_CACHE.clear()
        _REF_MASK_CACHE.clear()
        _RAW_CACHE_BYTES = 0
        _RAW_CACHE_HITS = 0
        _RAW_CACHE_MISSES = 0
        _OBJECT_TRACK_CACHE_BYTES = 0


def sam3_video_cache_info() -> dict[str, int]:
    with _CACHE_LOCK:
        return {
            "seeds": len(_SEED_CACHE),
            "raw_results": len(_RAW_CACHE),
            "raw_cache_bytes": int(_RAW_CACHE_BYTES),
            "raw_cache_hits": int(_RAW_CACHE_HITS),
            "raw_cache_misses": int(_RAW_CACHE_MISSES),
            "raw_cache_max_bytes": int(MAX_RAW_CACHE_BYTES),
        }


def _native_video_modules() -> tuple[Any, Any, Any, Any]:
    """Load the native video-state helpers as a synthetic package.

    ``mask_sam3`` already locates the installed ``nodes`` directory for the
    image predictor.  Reusing that package root keeps relative imports intact
    while avoiding the native node registry/UI import at addon startup.
    """

    global _VIDEO_MODULE, _VIDEO_ERROR
    if _VIDEO_MODULE is not None:
        return _VIDEO_MODULE
    if _VIDEO_ERROR:
        raise SAM3VideoError(_VIDEO_ERROR)
    nodes_root = _native_nodes_root()
    if nodes_root is None:
        _VIDEO_ERROR = "SAM3 native video adapter is not installed"
        raise SAM3VideoError(_VIDEO_ERROR)
    package_name = "_ths_native_sam3_nodes"
    try:
        package = importlib.import_module(package_name)
    except Exception:
        package = None
    if package is None:
        import sys
        import importlib.util

        spec = importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
        if spec is None:
            _VIDEO_ERROR = "could not create native SAM3 package spec"
            raise SAM3VideoError(_VIDEO_ERROR)
        package = importlib.util.module_from_spec(spec)
        package.__path__ = [str(nodes_root)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    try:
        video_state = importlib.import_module(f"{package_name}.video_state")
        reconstructor = importlib.import_module(f"{package_name}.inference_reconstructor")
        return_value = (
            video_state,
            reconstructor,
            getattr(video_state, "VideoPrompt"),
            getattr(video_state, "VideoConfig"),
        )
        _VIDEO_MODULE = return_value
        return return_value
    except Exception as exc:
        _VIDEO_ERROR = f"SAM3 native video adapter import failed: {type(exc).__name__}: {exc}"
        raise SAM3VideoError(_VIDEO_ERROR) from exc


def _source_digest(images: np.ndarray, *, normalize: bool = False) -> str:
    array = np.asarray(images)
    digest = hashlib.sha256()
    digest.update(repr(tuple(int(value) for value in array.shape)).encode("ascii"))
    if normalize:
        # Match the queued track's canonical float32 bytes, with at most one
        # frame's scratch space even for noncontiguous or nonfinite input.
        digest.update(b"float32")
        for frame in array:
            frame = np.ascontiguousarray(frame, dtype=np.float32)
            if not np.isfinite(frame).all():
                frame = np.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=0.0)
            digest.update(memoryview(frame).cast("B"))
    else:
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _prompt_digest(prompts: Iterable[Mapping[str, Any]], frame: int) -> str:
    payload = []
    for item in prompts:
        if isinstance(item.get("points"), (list, tuple)):
            entry: dict[str, Any] = {
                "frame": int(item.get("frame", frame)),
                "points": [[float(point[0]), float(point[1])] for point in item.get("points", [])],
                "labels": [int(label) for label in item.get("labels", [])],
            }
        else:
            entry = {
                "frame": int(item.get("frame", frame)),
                "box": [float(item[name]) for name in ("x", "y", "w", "h")],
            }
        if "object_id" in item:
            entry["object_id"] = int(item["object_id"])
        refs = item.get("refinements")
        if isinstance(refs, (list, tuple)) and refs:
            entry["refinements"] = [
                {
                    "frame": int(ref.get("frame", frame)),
                    "points": [[float(pt[0]), float(pt[1])] for pt in ref.get("points", [])],
                    "labels": [int(lbl) for lbl in ref.get("labels", [])],
                }
                for ref in refs
            ]
        payload.append(entry)
    encoded = repr((int(frame), payload)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


def _seed_token(source_digest: str, frame: int, prompt_digest: str) -> str:
    # Include a nonce so an explicitly stale token can never accidentally
    # become valid after a cache eviction/recalculation.
    nonce = uuid.uuid4().hex
    payload = f"sam3v1|{source_digest}|{frame}|{prompt_digest}|{nonce}".encode("ascii")
    return "sam3v1-" + hashlib.sha256(payload).hexdigest()[:48]


def _normal_box(event: Mapping[str, Any]) -> tuple[float, float, float, float]:
    values = tuple(float(event[name]) for name in ("x", "y", "w", "h"))
    x, y, w, h = values
    if not all(np.isfinite(values)) or w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > 1 or y + h > 1:
        raise SAM3VideoError("SAM3 box prompt must stay inside normalized [0, 1] coordinates")
    return values


def _prompt_events(parsed: Mapping[str, Any], frame: int) -> list[dict[str, Any]]:
    """Extract SAM3 box prompts while leaving brush/erase events as overlays."""

    sam3 = parsed.get("sam3") if isinstance(parsed.get("sam3"), Mapping) else {}
    if sam3.get("objects"):
        return [{**item, "object_id": index + 1} for index, item in enumerate(sam3["objects"])]
    prompt = sam3.get("prompt") if isinstance(sam3, Mapping) else None
    events: list[dict[str, Any]] = []
    explicit_frames: set[int] = set()
    if isinstance(prompt, Mapping):
        refs = prompt.get("refinements")
        clean_refs = [dict(ref) for ref in refs] if isinstance(refs, (list, tuple)) and refs else None
        if all(name in prompt for name in ("x", "y", "w", "h")):
            item = dict(prompt)
            item["frame"] = int(prompt.get("frame", frame))
            events.append(item)
            explicit_frames.add(int(item["frame"]))
        elif isinstance(prompt.get("points"), (list, tuple)):
            points = prompt.get("points", [])
            labels = prompt.get("labels", [1] * len(points))
            if isinstance(labels, (list, tuple)) and len(points) == len(labels) and points:
                entry_pt: dict[str, Any] = {
                    "frame": int(prompt.get("frame", frame)),
                    "points": [[float(point[0]), float(point[1])] for point in points],
                    "labels": [int(label) for label in labels],
                    "prompt_type": "point",
                }
                if clean_refs:
                    entry_pt["refinements"] = clean_refs
                events.append(entry_pt)
                explicit_frames.add(int(prompt.get("frame", frame)))
    frames = parsed.get("frames", {})
    if isinstance(frames, Mapping):
        for key, bucket in frames.items():
            try:
                item_frame = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(bucket, Mapping):
                continue
            if item_frame in explicit_frames:
                continue
            raw_events = bucket.get("events", [])
            if not isinstance(raw_events, (list, tuple)):
                continue
            for event in raw_events:
                if not isinstance(event, Mapping) or str(event.get("type", "")) != "rect":
                    continue
                if str(event.get("mode", "add")) != "add":
                    continue
                # A SAM3 tracking document marks its object prompt explicitly.
                # Legacy GrabCut rectangles remain local overlays and are never
                # silently promoted to a native tracking anchor.
                if str(event.get("refine_method", "")) != "sam3" and not bool(event.get("sam3_prompt", False)):
                    continue
                item = dict(event)
                item["frame"] = item_frame
                events.append(item)
    if not events:
        return []
    # Keep deterministic ordering and ensure the seed frame has a prompt.
    events.sort(key=lambda item: (int(item.get("frame", frame)) != int(frame), int(item.get("frame", frame))))
    return events


def _overlay_events(parsed: Mapping[str, Any], frame: int) -> list[dict[str, Any]]:
    """Return frame-local edits that are not native SAM3 object prompts."""

    frames = parsed.get("frames", {})
    bucket = frames.get(str(frame), {}) if isinstance(frames, Mapping) else {}
    if not isinstance(bucket, Mapping):
        return []
    events = bucket.get("events", [])
    if not isinstance(events, (list, tuple)):
        return []
    result: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if str(event.get("type", "")) == "rect" and str(event.get("mode", "add")) == "add":
            if str(event.get("refine_method", "")) == "sam3" or bool(event.get("sam3_prompt", False)):
                continue
        result.append(dict(event))
    return result


def _require_range(images: np.ndarray, seed: SAM3Seed, start: int, end: int) -> tuple[int, int]:
    count = int(np.asarray(images).shape[0])
    start = max(0, int(start))
    end = count - 1 if int(end) < 0 else min(count - 1, int(end))
    if start > end:
        raise SAM3VideoError("SAM3 tracking range is empty")
    if not start <= seed.frame <= end:
        raise SAM3VideoError(
            f"SAM3 tracking range {start}..{end} does not include seed frame {seed.frame}"
        )
    return start, end


def _has_prompt_on_frame(item: Mapping[str, Any], target_frame: int) -> bool:
    if int(item.get("frame", target_frame)) == int(target_frame):
        return True
    return any(int(ref.get("frame", -1)) == int(target_frame) for ref in item.get("refinements", []))


_REF_MASK_CACHE: "OrderedDict[tuple[str, int, int, str], np.ndarray]" = OrderedDict()


def _store_object_track_mask(source_digest: str, object_id: int, raw_mask: np.ndarray) -> None:
    global _OBJECT_TRACK_CACHE_BYTES
    payload, shape = _pack_binary_mask(raw_mask)
    item_bytes = len(payload)
    if item_bytes <= 0 or item_bytes > int(MAX_OBJECT_TRACK_CACHE_BYTES):
        return
    key = (str(source_digest), int(object_id))
    with _CACHE_LOCK:
        previous = _OBJECT_TRACK_CACHE.pop(key, None)
        if previous is not None:
            _OBJECT_TRACK_CACHE_BYTES = max(0, _OBJECT_TRACK_CACHE_BYTES - int(previous[-1]))
        while _OBJECT_TRACK_CACHE and _OBJECT_TRACK_CACHE_BYTES + item_bytes > int(MAX_OBJECT_TRACK_CACHE_BYTES):
            _old_key, old_entry = _OBJECT_TRACK_CACHE.popitem(last=False)
            _OBJECT_TRACK_CACHE_BYTES = max(0, _OBJECT_TRACK_CACHE_BYTES - int(old_entry[-1]))
        _OBJECT_TRACK_CACHE[key] = (payload, shape, item_bytes)
        _OBJECT_TRACK_CACHE_BYTES += item_bytes
        _OBJECT_TRACK_CACHE.move_to_end(key)


def _lookup_prior_frame_mask(
    source_digest: str,
    object_id: int,
    frame: int,
    expected: tuple[int, int],
    *,
    allow_combined_fallback: bool = True,
) -> np.ndarray | None:
    payload = None
    shape: tuple[int, ...] = ()
    with _CACHE_LOCK:
        entry = _OBJECT_TRACK_CACHE.get((str(source_digest), int(object_id)))
        if entry is None and allow_combined_fallback:
            entry = _OBJECT_TRACK_CACHE.get((str(source_digest), 0))
        if entry is not None:
            _OBJECT_TRACK_CACHE.move_to_end((str(source_digest), int(object_id) if (str(source_digest), int(object_id)) in _OBJECT_TRACK_CACHE else 0))
            payload, shape, _ = entry
        elif allow_combined_fallback:
            for key in reversed(list(_RAW_CACHE.keys())):
                if len(key) >= 1 and key[0] == source_digest:
                    raw_payload, raw_shape, _anchors, _meta, _bytes = _RAW_CACHE[key]
                    payload, shape = raw_payload, raw_shape
                    break
    if payload is None or len(shape) != 3 or not (0 <= int(frame) < int(shape[0])):
        return None
    unpacked = _unpack_binary_mask(payload, shape)
    frame_mask = np.asarray(unpacked[int(frame)] > 0.5, dtype=bool)
    if frame_mask.shape != expected:
        frame_mask = np.asarray(_resize_mask(frame_mask.astype(np.float32), expected) > 0.5, dtype=bool)
    return frame_mask.copy() if bool(frame_mask.any()) else None


def _connected_components(mask: np.ndarray) -> tuple[int, np.ndarray]:
    binary = np.asarray(mask, dtype=bool)
    if not bool(binary.any()):
        return 1, np.zeros(binary.shape, dtype=np.int32)
    try:
        import cv2  # type: ignore

        num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8))
        return int(num_labels), np.asarray(labels, dtype=np.int32)
    except Exception:
        h, w = binary.shape
        labels = np.zeros((h, w), dtype=np.int32)
        current = 0
        for sy, sx in zip(*np.nonzero(binary)):
            sy_i, sx_i = int(sy), int(sx)
            if labels[sy_i, sx_i] != 0:
                continue
            current += 1
            stack = [(sy_i, sx_i)]
            labels[sy_i, sx_i] = current
            while stack:
                cy, cx = stack.pop()
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if binary[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
        return current + 1, labels


def _safe_interior_point(safe_fg: np.ndarray) -> tuple[int, int] | None:
    fg = np.asarray(safe_fg, dtype=bool)
    if not bool(fg.any()):
        return None
    try:
        import cv2  # type: ignore

        dist = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5)
        dist = np.where(fg, dist, -1.0)
        best_y, best_x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        return int(best_x), int(best_y)
    except Exception:
        ys, xs = np.nonzero(fg)
        mid = len(xs) // 2
        return int(xs[mid]), int(ys[mid])


def _prepare_refinement_prompt(
    prior_mask: np.ndarray,
    ref: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, list[list[float]]]:
    clean_prior = np.asarray(prior_mask, dtype=bool).copy()
    h, w = clean_prior.shape[:2]
    pts = [list(pt) for pt in ref.get("points", [])]
    lbls = [int(lbl) for lbl in ref.get("labels", [1] * len(pts))]
    user_pos: list[tuple[int, int]] = []
    user_neg: list[tuple[int, int]] = []
    for pt, lbl in zip(pts, lbls):
        px = min(max(0, int(round(float(pt[0]) * max(1, w - 1)))), w - 1)
        py = min(max(0, int(round(float(pt[1]) * max(1, h - 1)))), h - 1)
        if lbl == 1:
            user_pos.append((px, py))
        elif lbl == 0:
            user_neg.append((px, py))
    if user_neg and bool(clean_prior.any()):
        _num_labels, comp_labels = _connected_components(clean_prior)
        keep_comps = {int(comp_labels[py, px]) for px, py in user_pos if int(comp_labels[py, px]) > 0}
        neg_comps = {int(comp_labels[ny, nx]) for nx, ny in user_neg if int(comp_labels[ny, nx]) > 0}
        drop_comps = neg_comps - keep_comps
        if drop_comps:
            candidate_prior = clean_prior & ~np.isin(comp_labels, list(drop_comps))
            if bool(candidate_prior.any()):
                clean_prior = candidate_prior
    safe_fg = clean_prior.copy()
    if user_neg and bool(safe_fg.any()):
        radius = max(12, int(round(0.04 * min(h, w)))) if min(h, w) >= 120 else max(1, int(round(0.08 * min(h, w))))
        yy, xx = np.ogrid[:h, :w]
        for nx, ny in user_neg:
            safe_fg[((yy - ny) ** 2 + (xx - nx) ** 2) <= radius * radius] = False
    auto_pos_points: list[list[float]] = []
    aug_points = list(pts)
    aug_labels = list(lbls)
    need_auto_pos = bool(safe_fg.any()) and not any(bool(safe_fg[py, px]) for px, py in user_pos)
    if need_auto_pos:
        interior = _safe_interior_point(safe_fg)
        if interior is not None:
            ax, ay = interior
            norm_pt = [round(ax / max(1, w - 1), 6), round(ay / max(1, h - 1), 6)]
            aug_points.append(norm_pt)
            aug_labels.append(1)
            auto_pos_points.append(norm_pt)
    aug_event = {**dict(ref), "points": aug_points, "labels": aug_labels}
    return aug_event, clean_prior, auto_pos_points


def _drop_negated_disconnected_islands(
    mask: np.ndarray,
    ref: Mapping[str, Any],
    auto_pos_points: Iterable[Iterable[float]],
) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if not bool(result.any()):
        return result
    h, w = result.shape[:2]
    pts = list(ref.get("points", []))
    lbls = list(ref.get("labels", [1] * len(pts)))
    neg_pixels = [
        (min(max(0, int(round(float(pt[0]) * max(1, w - 1)))), w - 1), min(max(0, int(round(float(pt[1]) * max(1, h - 1)))), h - 1))
        for pt, lbl in zip(pts, lbls)
        if int(lbl) == 0
    ]
    if not neg_pixels:
        return result
    pos_pixels = [
        (min(max(0, int(round(float(pt[0]) * max(1, w - 1)))), w - 1), min(max(0, int(round(float(pt[1]) * max(1, h - 1)))), h - 1))
        for pt, lbl in zip(pts, lbls)
        if int(lbl) == 1
    ] + [
        (min(max(0, int(round(float(pt[0]) * max(1, w - 1)))), w - 1), min(max(0, int(round(float(pt[1]) * max(1, h - 1)))), h - 1))
        for pt in auto_pos_points
    ]
    num_labels, comp_labels = _connected_components(result)
    if num_labels <= 2:
        return result
    keep_comps = {int(comp_labels[py, px]) for px, py in pos_pixels if int(comp_labels[py, px]) > 0}
    neg_comps = {int(comp_labels[ny, nx]) for nx, ny in neg_pixels if int(comp_labels[ny, nx]) > 0}
    drop_comps = neg_comps - keep_comps
    if drop_comps:
        filtered = result & ~np.isin(comp_labels, list(drop_comps))
        if bool(filtered.any()):
            return filtered
    return result


def _call_predict_native(
    image: np.ndarray,
    event: Mapping[str, Any],
    *,
    mask_input: np.ndarray | None = None,
    **options: Any,
) -> np.ndarray:
    if mask_input is not None:
        try:
            params = inspect.signature(_predict_native).parameters
        except (TypeError, ValueError):
            params = {}
        if "mask_input" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return np.asarray(_predict_native(image, event, mask_input=mask_input, **options), dtype=bool)
    return np.asarray(_predict_native(image, event, **options), dtype=bool)


def calculate_frame(
    images: np.ndarray,
    parsed: Mapping[str, Any],
    *,
    frame: int,
    interactive: bool = False,
) -> tuple[np.ndarray, SAM3Seed, dict[str, Any]]:
    """Calculate one native SAM3 seed frame, with no propagation."""

    images = np.asarray(images) if interactive else np.asarray(images, dtype=np.float32)
    if images.ndim != 4 or images.shape[0] < 1:
        raise SAM3VideoError("SAM3 frame calculation requires T x H x W x C images")
    frame = int(frame)
    if not 0 <= frame < images.shape[0]:
        raise SAM3VideoError(f"SAM3 seed frame {frame} is outside this clip")
    prompts = _prompt_events(parsed, frame)
    if not prompts or not any(_has_prompt_on_frame(item, frame) for item in prompts):
        meta = {"frame_ready": False, "track_ready": False, "frame": frame,
                "seed_token": "", "direction": "forward", "computed_frames": [],
                "error": "SAM3 frame calculation requires a marked box prompt on the selected frame"}
        raise SAM3VideoError(meta["error"], meta=meta)
    selected_prompts = prompts if parsed.get("sam3", {}).get("objects") else prompts[:1]
    matching = [item for item in selected_prompts if int(item.get("frame", frame)) == frame]
    primary_event = matching[0] if matching else selected_prompts[0]
    box = _normal_box(primary_event) if not isinstance(primary_event.get("points"), (list, tuple)) else (0.0, 0.0, 1.0, 1.0)
    object_frames_expected = [int(item.get("frame", frame)) for item in selected_prompts]
    for obj_frame in object_frames_expected:
        if not 0 <= obj_frame < images.shape[0]:
            raise SAM3VideoError(f"SAM3 object frame {obj_frame} is outside this clip")
    refinements_on_frame = [
        ref for item in selected_prompts
        for ref in item.get("refinements", [])
        if int(ref.get("frame", -1)) == frame
    ]
    source_digest = _source_digest(images, normalize=interactive)
    prompt_digest = _prompt_digest(prompts, frame)
    cache_identity = (source_digest, int(frame), prompt_digest, box)
    has_only_unresolved_shape = bool(
        refinements_on_frame
        and not any(f == frame for f in object_frames_expected)
        and not any(1 in ref.get("labels", []) for ref in refinements_on_frame)
    )
    with _CACHE_LOCK:
        for cached in _SEED_CACHE.values():
            if (cached.source_digest, cached.frame, cached.prompt_digest, cached.box) == cache_identity:
                prior_available = has_only_unresolved_shape and any(
                    _lookup_prior_frame_mask(
                        source_digest,
                        int(item.get("object_id", idx + 1)),
                        frame,
                        tuple(int(value) for value in images[frame].shape[:2]),
                        allow_combined_fallback=(len(selected_prompts) == 1),
                    )
                    is not None
                    for idx, item in enumerate(selected_prompts)
                    for ref in item.get("refinements", [])
                    if int(ref.get("frame", -1)) == frame
                )
                if prior_available and not bool(cached.mask.any()):
                    break
                _SEED_CACHE.move_to_end(cached.token)
                mask = np.asarray(cached.mask, dtype=bool).copy()
                cached_obj_frames = list(cached.object_frames) if cached.object_frames else [frame]
                computed_frames = sorted(set([frame] + cached_obj_frames))
                meta: dict[str, Any] = {
                    "frame_ready": True,
                    "track_ready": False,
                    "frame": frame,
                    "seed_token": cached.token,
                    "direction": "forward",
                    "computed_frames": computed_frames,
                }
                if has_only_unresolved_shape and not bool(mask.any()):
                    meta["keep_existing_mask"] = True
                return mask, cached, meta
    expected = tuple(int(value) for value in images[frame].shape[:2])
    with _CACHE_LOCK:
        recent_masks = {}
        for cached in _SEED_CACHE.values():
            if cached.source_digest != source_digest:
                continue
            for idx, (key, object_mask) in enumerate(cached.objects):
                c_frame = cached.object_frames[idx] if idx < len(cached.object_frames) else cached.frame
                recent_masks[(int(c_frame), key)] = object_mask
        recent_ref_masks = {
            (r_oid, r_frm, r_key): r_mask.copy()
            for (r_src, r_oid, r_frm, r_key), r_mask in _REF_MASK_CACHE.items()
            if r_src == source_digest
        }
    refinement_masks: list[tuple[int, int, np.ndarray]] = []
    unresolved_neg_only = False
    refined_on_frame_objects: set[int] = set()
    try:
        options = {"keep_loaded": True} if interactive else {}
        object_masks = []
        object_frames: list[int] = []
        mask = np.zeros(expected, dtype=bool)
        selected_frame_img = None
        for index, item in enumerate(selected_prompts, start=1):
            _throw_interrupted()
            obj_frame = int(item.get("frame", frame))
            primary_item = {k: v for k, v in item.items() if k not in ("object_id", "refinements")}
            key = _prompt_digest((primary_item,), obj_frame)
            object_mask = recent_masks.get((obj_frame, key))
            if object_mask is None:
                obj_image = images[obj_frame]
                if interactive:
                    obj_image = np.nan_to_num(obj_image.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
                object_mask = _call_predict_native(obj_image, primary_item, **options)
                if object_mask.shape != expected or not bool(object_mask.any()):
                    raise SAM3VideoError(f"SAM3 object {index} returned an empty or invalid seed mask")
                object_mask = object_mask.copy()
                recent_masks[(obj_frame, key)] = object_mask
            object_masks.append((key, object_mask))
            object_frames.append(obj_frame)
            if obj_frame == frame:
                np.logical_or(mask, object_mask, out=mask)
        for index, item in enumerate(selected_prompts, start=1):
            object_id = int(item.get("object_id", index))
            for ref in item.get("refinements", []):
                r_frame = int(ref.get("frame", frame))
                if not 0 <= r_frame < images.shape[0]:
                    continue
                ref_key = _prompt_digest((ref,), r_frame)
                if r_frame != frame:
                    cached_ref = recent_ref_masks.get((object_id, r_frame, ref_key))
                    if cached_ref is not None and cached_ref.shape == expected and bool(cached_ref.any()):
                        refinement_masks.append((object_id, r_frame, cached_ref.copy()))
                    continue
                if selected_frame_img is None:
                    selected_frame_img = images[frame]
                    if interactive:
                        selected_frame_img = np.nan_to_num(selected_frame_img.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
                prior_mask = _lookup_prior_frame_mask(
                    source_digest,
                    object_id,
                    frame,
                    expected,
                    allow_combined_fallback=(len(selected_prompts) == 1),
                )
                if prior_mask is not None:
                    _throw_interrupted()
                    aug_event, clean_prior, auto_pos = _prepare_refinement_prompt(prior_mask, ref)
                    ref_mask = _call_predict_native(selected_frame_img, aug_event, mask_input=clean_prior, **options)
                    ref_mask = _drop_negated_disconnected_islands(ref_mask, ref, auto_pos)
                    if ref_mask.shape == expected and bool(ref_mask.any()):
                        ref_mask = ref_mask.copy()
                        refinement_masks.append((object_id, frame, ref_mask))
                        refined_on_frame_objects.add(object_id)
                        np.logical_or(mask, ref_mask, out=mask)
                        with _CACHE_LOCK:
                            _REF_MASK_CACHE[(source_digest, object_id, frame, ref_key)] = ref_mask
                            _REF_MASK_CACHE.move_to_end((source_digest, object_id, frame, ref_key))
                            while len(_REF_MASK_CACHE) > int(MAX_SEEDS) * 4:
                                _REF_MASK_CACHE.popitem(last=False)
                elif 1 in ref.get("labels", []):
                    _throw_interrupted()
                    ref_mask = _call_predict_native(selected_frame_img, ref, **options)
                    if ref_mask.shape == expected and bool(ref_mask.any()):
                        ref_mask = ref_mask.copy()
                        refinement_masks.append((object_id, frame, ref_mask))
                        refined_on_frame_objects.add(object_id)
                        np.logical_or(mask, ref_mask, out=mask)
                        with _CACHE_LOCK:
                            _REF_MASK_CACHE[(source_digest, object_id, frame, ref_key)] = ref_mask
                            _REF_MASK_CACHE.move_to_end((source_digest, object_id, frame, ref_key))
                            while len(_REF_MASK_CACHE) > int(MAX_SEEDS) * 4:
                                _REF_MASK_CACHE.popitem(last=False)
                else:
                    unresolved_neg_only = True
        if refined_on_frame_objects and len(selected_prompts) > 1:
            for index, item in enumerate(selected_prompts, start=1):
                other_id = int(item.get("object_id", index))
                if other_id in refined_on_frame_objects or int(item.get("frame", frame)) == frame:
                    continue
                other_mask = _lookup_prior_frame_mask(source_digest, other_id, frame, expected, allow_combined_fallback=False)
                if other_mask is not None and other_mask.shape == expected and bool(other_mask.any()):
                    np.logical_or(mask, other_mask, out=mask)
        _throw_interrupted()
    except SAM3UnavailableError as exc:
        meta = {"frame_ready": False, "track_ready": False, "frame": frame,
                "seed_token": "", "direction": "forward", "computed_frames": [],
                "error": str(exc)}
        raise SAM3VideoError(str(exc), meta=meta) from exc
    keep_existing_mask = bool(unresolved_neg_only and not bool(mask.any()))
    token = _seed_token(source_digest, frame, prompt_digest)
    seed = SAM3Seed(
        token,
        source_digest,
        tuple(int(v) for v in images.shape),
        frame,
        prompt_digest,
        box,
        mask,
        objects=tuple(object_masks),
        object_frames=tuple(object_frames),
        refinement_masks=tuple(refinement_masks),
    )
    with _CACHE_LOCK:
        _SEED_CACHE[token] = seed
        _SEED_CACHE.move_to_end(token)
        while len(_SEED_CACHE) > int(MAX_SEEDS):
            _SEED_CACHE.popitem(last=False)
    computed_frames = sorted(set([frame] + object_frames))
    meta = {"frame_ready": True, "track_ready": False, "frame": frame,
            "seed_token": token, "direction": "forward", "computed_frames": computed_frames}
    meta["seed_strategy"] = "native_image_prompt"
    if keep_existing_mask:
        meta["keep_existing_mask"] = True
    return np.asarray(mask, dtype=bool).copy(), seed, meta


def _lookup_seed(images: np.ndarray, parsed: Mapping[str, Any], frame: int, token: str) -> SAM3Seed:
    token = str(token or "")
    base_meta = {"frame_ready": False, "track_ready": False, "frame": int(frame),
                 "seed_token": token, "direction": "forward", "computed_frames": []}
    if not token:
        raise SAM3VideoError("SAM3 track requires a non-empty frame seed_token", meta={**base_meta, "error": "missing seed_token"})
    with _CACHE_LOCK:
        seed = _SEED_CACHE.get(token)
        if seed is not None:
            _SEED_CACHE.move_to_end(token)
    if seed is None:
        raise SAM3VideoError("SAM3 seed_token is stale or unknown; Calculate frame again", meta={**base_meta, "error": "stale seed_token"})
    source_digest = _source_digest(images)
    if seed.source_digest != source_digest or seed.frame != int(frame):
        raise SAM3VideoError("SAM3 seed_token does not belong to this source or seed frame", meta={**base_meta, "error": "seed source/frame mismatch"})
    prompts = _prompt_events(parsed, int(frame))
    if not prompts or _prompt_digest(prompts, int(frame)) != seed.prompt_digest:
        raise SAM3VideoError("SAM3 seed_token does not match the current object prompt; Calculate frame again", meta={**base_meta, "error": "seed prompt mismatch"})
    return seed


def _throw_interrupted() -> None:
    try:
        import comfy.model_management as model_management  # type: ignore

        callback = getattr(model_management, "throw_exception_if_processing_interrupted", None)
        if callable(callback):
            callback()
    except ImportError:
        return


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pack_binary_mask(mask: np.ndarray) -> tuple[bytes, tuple[int, ...]]:
    array = np.asarray(mask) > 0.5
    shape = tuple(int(value) for value in array.shape)
    packed = np.packbits(array.reshape(-1), bitorder="little")
    return bytes(memoryview(np.ascontiguousarray(packed)).cast("B")), shape


def _unpack_binary_mask(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape, dtype=np.int64))
    raw = np.frombuffer(payload, dtype=np.uint8)
    values = np.unpackbits(raw, bitorder="little", count=count)
    return values.reshape(shape).astype(np.float32)


def _select_object_mask(outputs: Mapping[str, Any], object_id: int, shape: tuple[int, int]) -> np.ndarray | None:
    raw = None
    for key in ("out_binary_masks", "video_res_masks", "masks"):
        if key in outputs and outputs[key] is not None:
            raw = outputs[key]
            break
    if raw is None:
        return None
    masks = _as_numpy(raw)
    ids_metadata = outputs.get("obj_ids", outputs.get("out_obj_ids"))
    if ids_metadata is not None:
        ids_array = _as_numpy(ids_metadata).reshape(-1).astype(np.int64, copy=False)
        if not np.any(ids_array == int(object_id)):
            return None
    while masks.ndim > 3 and masks.shape[0] == 1:
        masks = masks[0]
    if masks.ndim == 2:
        selected = masks
    elif masks.ndim == 3:
        ids = ids_metadata
        selected_index = 0
        if ids is not None:
            ids_array = _as_numpy(ids).reshape(-1)
            matches = np.where(ids_array.astype(np.int64, copy=False) == int(object_id))[0]
            if len(matches):
                selected_index = int(matches[0])
            else:
                # Object identity is explicit in native outputs.  Never fall
                # back positionally to another detected object when the
                # selected object disappeared or was remapped.
                return None
        elif int(object_id) > 0 and int(object_id) < masks.shape[0]:
            selected_index = int(object_id)
        if selected_index >= masks.shape[0]:
            return None
        selected = masks[selected_index]
    else:
        return None
    selected = _resize_mask(np.squeeze(selected), shape)
    return np.asarray(selected > 0.0, dtype=bool)


def _tracked_response_mask(outputs, object_ids, shape):
    native_ids = outputs.get("obj_ids", outputs.get("out_obj_ids"))
    if len(object_ids) > 1 and native_ids is None:
        raise SAM3VideoError("SAM3 multi-object propagation did not report object IDs")
    combined = None
    for object_id in object_ids:
        selected = _select_object_mask(outputs, object_id, shape)
        if selected is not None:
            combined = selected if combined is None else combined | selected
    if combined is not None:
        return combined

    # Native SAM3 removes empty masks and their IDs together on occluded frames.
    # A valid response for such a frame is different from a missing response.
    payload = outputs.get("out_binary_masks")
    if native_ids is not None and payload is not None:
        ids = _as_numpy(native_ids).reshape(-1)
        masks = _as_numpy(payload)
        if masks.ndim == 3 and len(masks) == len(ids) and not np.isin(ids, object_ids).any():
            return np.zeros(shape, dtype=bool)
    return None


def _native_track(
    images: np.ndarray,
    prompts: list[dict[str, Any]],
    *,
    seed: SAM3Seed,
    start: int,
    end: int,
    direction: str,
    object_masks_out: dict[int, dict[int, np.ndarray]] | None = None,
) -> dict[int, np.ndarray]:
    """Propagate selected objects independently and union them per frame."""

    global _LAST_NATIVE_SEED_STRATEGY
    _LAST_NATIVE_SEED_STRATEGY = "unknown"
    if direction not in {"forward", "backward", "both"}:
        raise SAM3VideoError("SAM3 direction must be forward, backward, or both")
    try:
        import torch  # type: ignore
        import comfy.model_management as model_management  # type: ignore

        video_state_mod, reconstructor_mod, video_prompt_cls, video_config_cls = _native_video_modules()
        cache_module = _native_model_cache_module()
        getter = getattr(cache_module, "get_or_build_model", None)
        if not callable(getter):
            raise SAM3VideoError("SAM3 native adapter has no get_or_build_model()")
        video_state = video_state_mod.create_video_state(
            video_frames=torch.from_numpy(np.asarray(images, dtype=np.float32)),
            config=video_config_cls(),
        )
        # Box prompts use normalized xyxy corners.  The native reconstructor
        # converts these to the special [2, 3] corner labels expected by the
        # stateful tracker. Legacy corrections share ID 1; explicit objects
        # have individual IDs and masks so their motion remains independent.
        for item in prompts:
            item_frame = int(item.get("frame", seed.frame))
            if not 0 <= item_frame < video_state.num_frames:
                continue
            object_id = int(item.get("object_id", seed.object_id))
            if isinstance(item.get("points"), (list, tuple)):
                video_state = video_state.with_prompt(
                    video_prompt_cls.create_point(
                        item_frame,
                        object_id,
                        item.get("points", []),
                        item.get("labels", [1] * len(item.get("points", []))),
                    )
                )
            else:
                x, y, w, h = _normal_box(item)
                video_state = video_state.with_prompt(
                    video_prompt_cls.create_box(item_frame, object_id, [x, y, x + w, y + h], is_positive=True)
                )
        if not video_state.prompts:
            raise SAM3VideoError("SAM3 native video state has no object prompt")
        model = getter(_native_config())
        load_models_gpu = getattr(model_management, "load_models_gpu", None)
        if not callable(load_models_gpu):
            raise SAM3VideoError("Comfy model_management.load_models_gpu() is unavailable")
        load_models_gpu([model])
        masks: dict[int, np.ndarray] = {}
        try:
            # Reconstruct one immutable session and issue exact range requests.
            inference_state = reconstructor_mod.get_inference_state(model, video_state)
            # The native reconstructor returns the session handle from
            # ``start_session`` (``{"session_id": ...}``) rather than the
            # mutable predictor state itself.  Resolve that handle through the
            # predictor's owned session table before touching tracker internals.
            if not isinstance(inference_state, Mapping) or "tracker_inference_states" not in inference_state:
                sessions = getattr(model, "_ALL_INFERENCE_STATES", {})
                session_entry = sessions.get(video_state.session_uuid) if isinstance(sessions, Mapping) else None
                if isinstance(session_entry, Mapping) and isinstance(session_entry.get("state"), Mapping):
                    inference_state = session_entry["state"]
            if not isinstance(inference_state, Mapping) or "tracker_inference_states" not in inference_state:
                raise SAM3VideoError("SAM3 native inference session did not expose tracker state")
            # Feed the exact Calculate-frame mask into the *tracker* state.
            # ``SAM3UnifiedModel.model`` is the full video inference module and
            # does not expose ``add_new_mask``; the method belongs to its
            # ``tracker`` and expects the per-object tracker inference state.
            native_video_model = getattr(model, "model", None)
            tracker = getattr(native_video_model, "tracker", None)
            state_lookup = getattr(native_video_model, "_get_tracker_inference_states_by_obj_ids", None)
            add_new_mask = getattr(tracker, "add_new_mask", None)
            if tracker is None:
                raise SAM3VideoError("SAM3 native video tracker is unavailable; cannot bind the calculated seed mask")
            if not callable(state_lookup) or not callable(add_new_mask):
                raise SAM3VideoError(
                    "SAM3 native tracker lacks add_new_mask(state, frame_idx, obj_id); "
                    "refusing a box-only seed that could change object identity"
                )
            seeds = [
                (
                    index,
                    int(seed.object_frames[index - 1]) if index - 1 < len(seed.object_frames) else int(seed.frame),
                    object_mask,
                )
                for index, (_, object_mask) in enumerate(seed.objects, start=1)
            ] or [(seed.object_id, int(seed.frame), seed.mask)]
            updated_states = {}
            for object_id, obj_frame, object_mask in seeds:
                tracker_states = state_lookup(inference_state, [object_id])
                if not tracker_states:
                    raise SAM3VideoError(f"SAM3 native tracker did not create selected object {object_id}")
                tracker_state = tracker_states[0]
                add_new_mask(
                    inference_state=tracker_state,
                    frame_idx=int(obj_frame),
                    obj_id=object_id,
                    mask=torch.from_numpy(np.asarray(object_mask, dtype=np.float32)),
                )
                updated_states[id(tracker_state)] = tracker_state
            ref_mask_map = {
                (int(oid), int(rf)): m
                for oid, rf, m in getattr(seed, "refinement_masks", ())
            }
            prepare_feats = getattr(native_video_model, "_prepare_backbone_feats", None)
            mask_refs: list[tuple[int, int, np.ndarray]] = []
            point_refs: list[tuple[int, int, Mapping[str, Any]]] = []
            for index, item in enumerate(prompts, start=1):
                object_id = int(item.get("object_id", index))
                for ref in item.get("refinements", []):
                    r_frame = int(ref.get("frame", seed.frame))
                    if not int(start) <= r_frame <= int(end):
                        continue
                    r_mask = ref_mask_map.get((object_id, r_frame))
                    if r_mask is not None:
                        mask_refs.append((object_id, r_frame, r_mask))
                    else:
                        point_refs.append((object_id, r_frame, ref))
            for object_id, r_frame, r_mask in mask_refs:
                if callable(prepare_feats):
                    prepare_feats(inference_state, r_frame, False)
                tracker_states = state_lookup(inference_state, [object_id])
                if not tracker_states:
                    continue
                tracker_state = tracker_states[0]
                add_new_mask(
                    inference_state=tracker_state,
                    frame_idx=int(r_frame),
                    obj_id=object_id,
                    mask=torch.from_numpy(np.asarray(r_mask, dtype=np.float32)),
                )
                updated_states[id(tracker_state)] = tracker_state
            preflight = getattr(tracker, "propagate_in_video_preflight", None)
            if callable(preflight):
                for tracker_state in updated_states.values():
                    preflight(tracker_state, run_mem_encoder=True)

            _LAST_NATIVE_SEED_STRATEGY = "native_tracker_add_new_mask"
            prompt_frames = sorted({
                f for f in (
                    [int(item.get("frame", seed.frame)) for item in prompts]
                    + [int(ref["frame"]) for item in prompts for ref in item.get("refinements", [])]
                )
                if int(start) <= f <= int(end)
            }) or [int(seed.frame)]
            min_f, max_f = prompt_frames[0], prompt_frames[-1]
            spans: list[tuple[str, int, int]] = []
            if direction in {"forward", "both"}:
                spans.append(("forward", min_f, max(0, int(end) - min_f)))
            if direction in {"backward", "both"}:
                spans.append(("backward", max_f, max(0, max_f - int(start))))

            if point_refs:
                for request_direction, start_idx, span in spans:
                    _throw_interrupted()
                    warmup_request = {
                        "type": "propagate_in_video",
                        "session_id": video_state.session_uuid,
                        "propagation_direction": request_direction,
                        "start_frame_index": int(start_idx),
                        "max_frame_num_to_track": int(span),
                    }
                    for _response in model.handle_stream_request(warmup_request):
                        _throw_interrupted()
                add_new_points = getattr(tracker, "add_new_points", None)
                refined_states = {}
                for object_id, r_frame, ref in point_refs:
                    if callable(prepare_feats):
                        prepare_feats(inference_state, r_frame, False)
                    tracker_states = state_lookup(inference_state, [object_id])
                    if not tracker_states or not callable(add_new_points):
                        continue
                    tracker_state = tracker_states[0]
                    add_new_points(
                        inference_state=tracker_state,
                        frame_idx=r_frame,
                        obj_id=object_id,
                        points=ref["points"],
                        labels=ref["labels"],
                        clear_old_points=True,
                        rel_coordinates=True,
                        use_prev_mem_frame=True,
                    )
                    refined_states[id(tracker_state)] = tracker_state
                if callable(preflight):
                    for tracker_state in refined_states.values():
                        preflight(tracker_state, run_mem_encoder=True)

            progress_start = min_f if direction == "forward" else int(start)
            progress_end = max_f if direction == "backward" else int(end)
            report = frame_progress("sam3_track", progress_end - progress_start + 1)
            completed = set(prompt_frames)
            report(len(completed))
            object_ids = [object_id for object_id, _, _ in seeds]
            shape = tuple(int(v) for v in images.shape[1:3])
            for request_direction, start_idx, span in spans:
                _throw_interrupted()
                request = {
                    "type": "propagate_in_video",
                    "session_id": video_state.session_uuid,
                    "propagation_direction": request_direction,
                    "start_frame_index": int(start_idx),
                    "max_frame_num_to_track": int(span),
                }
                for response in model.handle_stream_request(request):
                    _throw_interrupted()
                    frame_idx = response.get("frame_index", response.get("frame_idx"))
                    if frame_idx is None or not int(start) <= int(frame_idx) <= int(end):
                        continue
                    outputs = response.get("outputs", response)
                    if not isinstance(outputs, Mapping):
                        continue
                    combined = _tracked_response_mask(outputs, object_ids, shape)
                    if combined is not None:
                        index = int(frame_idx)
                        masks[index] = combined
                        if object_masks_out is not None:
                            for oid in object_ids:
                                oid_mask = _select_object_mask(outputs, oid, shape)
                                if oid_mask is not None:
                                    object_masks_out.setdefault(int(oid), {})[index] = oid_mask
                        if progress_start <= index <= progress_end and index not in completed:
                            completed.add(index)
                            report(len(completed))
        finally:
            try:
                close_session = getattr(model, "close_session", None)
                if callable(close_session):
                    close_session(video_state.session_uuid)
            except Exception:
                pass
            try:
                cleanup = getattr(video_state_mod, "cleanup_temp_dir", None)
                if callable(cleanup):
                    cleanup(video_state.temp_dir)
            except Exception:
                pass
            _offload_model(model)
            gc.collect()
        return masks
    except SAM3VideoError:
        raise
    except Exception as exc:
        try:
            model_value = locals().get("model")
            if model_value is not None:
                _offload_model(model_value)
        except Exception:
            pass
        raise SAM3VideoError(f"SAM3 native video propagation failed: {type(exc).__name__}: {exc}") from exc


def track_video(
    images: np.ndarray,
    parsed: Mapping[str, Any],
    *,
    frame: int,
    token: str,
    start: int,
    end: int,
    direction: str,
) -> SAM3VideoResult:
    """Validate a seed token and run native propagation over the exact range."""

    images = np.asarray(images, dtype=np.float32)
    seed = _lookup_seed(images, parsed, int(frame), token)
    start, end = _require_range(images, seed, start, end)
    direction = str(direction or "forward")
    prompts = _prompt_events(parsed, seed.frame)
    source_digest = seed.source_digest
    prompt_digest = seed.prompt_digest
    cache_key = (source_digest, prompt_digest, int(seed.frame), int(start), int(end), direction, int(seed.object_id))
    global _RAW_CACHE_BYTES, _RAW_CACHE_HITS, _RAW_CACHE_MISSES
    with _CACHE_LOCK:
        cached = _RAW_CACHE.get(cache_key)
        if cached is not None:
            _RAW_CACHE_HITS += 1
            _RAW_CACHE.move_to_end(cache_key)
            payload, shape, anchors, cached_meta, _item_bytes = cached
            restored = _unpack_binary_mask(payload, shape)
            meta = dict(cached_meta)
            meta["cache_hit"] = True
            meta.update(sam3_video_cache_info())
            return SAM3VideoResult(restored.copy(), restored, anchors, meta)
        _RAW_CACHE_MISSES += 1
    with _VIDEO_INFERENCE_LOCK:
        with _CACHE_LOCK:
            cached = _RAW_CACHE.get(cache_key)
            if cached is not None:
                _RAW_CACHE_HITS += 1
                _RAW_CACHE.move_to_end(cache_key)
                payload, shape, anchors, cached_meta, _item_bytes = cached
                restored = _unpack_binary_mask(payload, shape)
                meta = dict(cached_meta)
                meta["cache_hit"] = True
                meta.update(sam3_video_cache_info())
                return SAM3VideoResult(restored.copy(), restored, anchors, meta)
        prompt_frames = sorted({
            f for f in (
                [int(item.get("frame", seed.frame)) for item in prompts]
                + [int(ref["frame"]) for item in prompts for ref in item.get("refinements", [])]
            )
            if int(start) <= f <= int(end)
        }) or [int(seed.frame)]
        min_f, max_f = prompt_frames[0], prompt_frames[-1]
        raw = np.zeros(images.shape[:3], dtype=np.float32)
        raw[seed.frame] = np.asarray(seed.mask, dtype=np.float32)
        object_masks_out: dict[int, dict[int, np.ndarray]] = {}
        try:
            try:
                track_params = inspect.signature(_native_track).parameters
            except (TypeError, ValueError):
                track_params = {}
            if "object_masks_out" in track_params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in track_params.values()
            ):
                native = _native_track(
                    images,
                    prompts,
                    seed=seed,
                    start=start,
                    end=end,
                    direction=direction,
                    object_masks_out=object_masks_out,
                )
            else:
                native = _native_track(images, prompts, seed=seed, start=start, end=end, direction=direction)
        except SAM3VideoError:
            raise
        except Exception as exc:
            raise SAM3VideoError(f"SAM3 native video propagation failed: {type(exc).__name__}: {exc}") from exc
        for index, mask in native.items():
            index = int(index)
            if start <= index <= end:
                value = np.asarray(mask, dtype=np.float32)
                if value.shape != raw.shape[1:]:
                    value = _resize_mask(value, raw.shape[1:])
                raw[index] = np.clip(value, 0.0, 1.0)
        # Keep the visible seed pixel-identical even when the native stream
        # omits the prompted frame or returns a postprocessed variant.
        if len(set(seed.object_frames or (seed.frame,))) <= 1 and len(prompt_frames) <= 1:
            raw[seed.frame] = np.asarray(seed.mask, dtype=np.float32)
        else:
            seed_entries = [
                (
                    int(seed.object_frames[idx]) if idx < len(seed.object_frames) else int(seed.frame),
                    obj_mask,
                )
                for idx, (_, obj_mask) in enumerate(seed.objects)
            ] or [(int(seed.frame), seed.mask)]
            for obj_frame, obj_mask in seed_entries:
                if start <= obj_frame <= end:
                    raw[obj_frame] = np.maximum(raw[obj_frame], np.asarray(obj_mask, dtype=np.float32))
        # Native tracking is the only anchor source.  Browser brush/erase
        # events are applied by mask_core after this immutable raw result is
        # cached, so a local edit never becomes a hidden tracker prompt.
        edit_bases = raw
        anchors = frozenset(prompt_frames)
        if direction == "forward":
            requested = set(range(min_f, end + 1))
        elif direction == "backward":
            requested = set(range(start, max_f + 1))
        else:
            requested = set(range(start, end + 1))
        covered = set(int(index) for index in native if start <= int(index) <= end)
        covered.update(prompt_frames)
        empty_frames = sorted(
            int(index)
            for index, value in native.items()
            if start <= int(index) <= end and not bool(np.asarray(value).any())
        )
        missing = sorted(int(index) for index in requested - covered)
        if missing:
            meta = {
                "frame_ready": True,
                "track_ready": False,
                "frame": seed.frame,
                "seed_token": seed.token,
                "direction": direction,
                "computed_frames": sorted(covered),
                "empty_frames": empty_frames,
                "missing_frames": missing,
                "error": "SAM3 native propagation returned an incomplete selected-object range",
                "seed_strategy": _LAST_NATIVE_SEED_STRATEGY,
            }
            raise SAM3VideoError(meta["error"], meta=meta)
        computed = sorted(covered)
        meta = {"frame_ready": True, "track_ready": True, "frame": seed.frame,
                "seed_token": seed.token, "direction": direction,
                "computed_frames": computed,
                "empty_frames": empty_frames,
                "seed_strategy": _LAST_NATIVE_SEED_STRATEGY,
                "cache_hit": False,
                "cache_stored": False}
        meta.update(sam3_video_cache_info())
        with _CACHE_LOCK:
            payload, shape = _pack_binary_mask(raw)
            item_bytes = len(payload)
            if item_bytes <= int(MAX_RAW_CACHE_ITEM_BYTES) and item_bytes <= int(MAX_RAW_CACHE_BYTES):
                previous = _RAW_CACHE.pop(cache_key, None)
                if previous is not None:
                    _RAW_CACHE_BYTES = max(0, _RAW_CACHE_BYTES - int(previous[-1]))
                while _RAW_CACHE and (
                    len(_RAW_CACHE) >= int(MAX_RAW_RESULTS)
                    or _RAW_CACHE_BYTES + item_bytes > int(MAX_RAW_CACHE_BYTES)
                ):
                    _old_key, old_entry = _RAW_CACHE.popitem(last=False)
                    _RAW_CACHE_BYTES = max(0, _RAW_CACHE_BYTES - int(old_entry[-1]))
                if _RAW_CACHE_BYTES + item_bytes <= int(MAX_RAW_CACHE_BYTES):
                    _RAW_CACHE[cache_key] = (payload, shape, anchors, dict(meta), item_bytes)
                    _RAW_CACHE_BYTES += item_bytes
                    _RAW_CACHE.move_to_end(cache_key)
                    meta["cache_stored"] = True
                    meta.update(sam3_video_cache_info())
                    _RAW_CACHE[cache_key] = (payload, shape, anchors, dict(meta), item_bytes)
        _store_object_track_mask(source_digest, 0, raw)
        if object_masks_out:
            for oid, frame_dict in object_masks_out.items():
                obj_arr = np.zeros(images.shape[:3], dtype=np.float32)
                for idx, val in frame_dict.items():
                    i = int(idx)
                    if start <= i <= end:
                        arr_v = np.asarray(val, dtype=np.float32)
                        if arr_v.shape != obj_arr.shape[1:]:
                            arr_v = _resize_mask(arr_v, obj_arr.shape[1:])
                        obj_arr[i] = np.clip(arr_v, 0.0, 1.0)
                idx_0 = int(oid) - 1
                if 0 <= idx_0 < len(seed.objects):
                    o_frame = int(seed.object_frames[idx_0]) if idx_0 < len(seed.object_frames) else int(seed.frame)
                    if start <= o_frame <= end and o_frame not in frame_dict:
                        obj_arr[o_frame] = np.maximum(obj_arr[o_frame], np.asarray(seed.objects[idx_0][1], dtype=np.float32))
                for r_oid, r_frame, r_mask in getattr(seed, "refinement_masks", ()):
                    if int(r_oid) == int(oid) and start <= int(r_frame) <= end and int(r_frame) not in frame_dict:
                        obj_arr[int(r_frame)] = np.maximum(obj_arr[int(r_frame)], np.asarray(r_mask, dtype=np.float32))
                _store_object_track_mask(source_digest, int(oid), obj_arr)
        elif len(prompts) <= 1:
            _store_object_track_mask(source_digest, 1, raw)
        return SAM3VideoResult(raw, edit_bases, anchors, meta)


def frame_result(images: np.ndarray, parsed: Mapping[str, Any], *, frame: int) -> SAM3VideoResult:
    mask, seed, meta = calculate_frame(images, parsed, frame=int(frame))
    raw = np.zeros(np.asarray(images).shape[:3], dtype=np.float32)
    raw[int(frame)] = np.asarray(mask, dtype=np.float32)
    meta.update(sam3_video_cache_info())
    return SAM3VideoResult(raw, raw, frozenset({int(frame)}), meta)


__all__ = [
    "SAM3VideoError",
    "SAM3Seed",
    "SAM3VideoResult",
    "calculate_frame",
    "frame_result",
    "track_video",
    "clear_sam3_video_cache",
    "sam3_video_cache_info",
    "_native_track",
    "_overlay_events",
]
