"""Context crops and safe prompt composition for local masked-video repair.

This module intentionally contains no ComfyUI, torch, or model imports.  The
auto-prompt node can therefore use it on a CPU-only preview path and the core
behaviour is easy to test with ordinary NumPy arrays.  The model-facing node
is responsible for captions (Florence/VLM/etc.); this file only prepares
bounded visual context and turns *observed* descriptive text into a compact
repair instruction.

``build_context_crops`` returns a stable dictionary contract::

    {
        "clips": [np.ndarray(H_i, W_i, 3), ...],
        "cropmeta": [{"frame": 12, "actual_frame": 12,
                       "bbox": [x0, y0, x1, y1],
                       "maskfraction": 0.03, ...}, ...],
        "diagnostic": "",
        "status": "ok" | "fallback",
    }

Coordinates use the usual half-open NumPy convention ``[x0, y0, x1, y1]``.
Pixels covered by the repair mask are neutralised with OpenCV Telea before
cropping, while pixels outside the mask remain bit-identical.  A full-frame
mask has no trustworthy surrounding context; it is reported as a fallback and
never causes an invented object description.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # OpenCV is part of the target Comfy image, but tests may be minimal.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised only without OpenCV.
    cv2 = None  # type: ignore[assignment]


MAX_CONTEXT_FRAMES = 3
MAX_CONTEXT_LONG_EDGE = 512
MIN_CONTEXT_PADDING = 32
DEFAULT_CONTEXT_PADDING = 0.6
PROMPT_MAX_CHARS = 1000
DEFAULT_REPAIR_PROMPT = (
    "Restore only the pixels inside the user mask using the adjacent unmasked "
    "source context. Preserve subject identity, geometry, natural texture, "
    "colors, lighting, focus, camera, motion, timing, and frame-to-frame "
    "continuity; every unmasked pixel remains unchanged."
)

_CONTROL_TOKEN_RE = re.compile(r"(?i)(?:@?\b(?:image|video|audio)\s*\d+\b|<\s*(?:image|video|audio)\s*\d+\s*>)")
_CONTROL_BRACKET_RE = re.compile(r"\[[^\]]{0,160}\]")
_INSTRUCTION_RE = re.compile(
    r"(?i)(?:\bnew\s+scene\b|\bnew\s+shot\b|\breference\s+(?:image|video)\b|"
    r"\bsystem\s+prompt\b|\bdeveloper\s+message\b|\bassistant\s*:|\bjailbreak\b|"
    r"\binstruction\s*:)"
)
_LEADING_IMPERATIVE_RE = re.compile(
    r"(?i)^\s*(?:add|change|replace|remove|insert|create|generate|make|use|"
    r"keep|preserve|restore|render|show|set|ensure|do|don't|do not)\b"
)
_MALICIOUS_RE = re.compile(
    r"(?i)\b(?:ignore|disregard|bypass|override)\b.{0,32}\b(?:previous|above|system|developer|instruction|prompt)\b"
)
_WHITESPACE_RE = re.compile(r"\s+")


def _as_images(images: Any) -> np.ndarray:
    array = np.asarray(images)
    if array.ndim != 4:
        raise ValueError("images must have shape T x H x W x C")
    if array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError("images must contain at least one non-empty frame")
    if array.shape[3] < 1:
        raise ValueError("images must contain at least one channel")
    # Caption previews are RGB.  Keep source values in the model's usual
    # [0,1] range without changing the caller's array.
    result = np.nan_to_num(array.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    if result.shape[-1] == 1:
        result = np.repeat(result, 3, axis=-1)
    elif result.shape[-1] > 3:
        result = result[..., :3]
    return np.clip(result, 0.0, 1.0)


def _as_mask(mask: Any, frame_count: int, height: int, width: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError("mask must have shape T x H x W or H x W")
    if tuple(array.shape) != (frame_count, height, width):
        raise ValueError(
            "image and mask geometry must match "
            f"(received images {(frame_count, height, width)}, mask {tuple(array.shape)})"
        )
    return np.nan_to_num(array.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)


def select_representative_indices(mask: Any, max_frames: int = MAX_CONTEXT_FRAMES) -> list[int]:
    """Choose first/middle/last distinct frames whose masks are non-empty.

    Empty frames are never selected.  For fewer than three eligible frames all
    eligible frames are returned; for a longer clip the middle index is chosen
    from the eligible list, which keeps the result deterministic even when the
    source timeline has sparse masks.
    """

    limit = max(1, min(MAX_CONTEXT_FRAMES, int(max_frames)))
    array = np.asarray(mask)
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError("mask must have shape T x H x W or H x W")
    occupancy = np.any(np.nan_to_num(array, nan=0.0) > 1e-5, axis=(1, 2))
    eligible = [int(index) for index in np.flatnonzero(occupancy)]
    if len(eligible) <= limit:
        return eligible
    if limit == 1:
        return [eligible[0]]
    if limit == 2:
        return [eligible[0], eligible[-1]]
    # Use the upper middle for an even number of eligible frames.  It is the
    # sample nearest the temporal midpoint between first and last and avoids
    # biasing the caption set toward the opening frame.
    middle = eligible[len(eligible) // 2]
    return [eligible[0], middle, eligible[-1]]


def _telea_prefill(frame: np.ndarray, frame_mask: np.ndarray) -> tuple[np.ndarray, str | None]:
    """Neutralise masked pixels while preserving outside-mask values exactly."""

    active = np.asarray(frame_mask > 1e-5, dtype=bool)
    if not np.any(active):
        return np.array(frame, copy=True), None
    source = np.asarray(frame, dtype=np.float32)
    if cv2 is None:
        # The crop still contains real context; expose the limitation instead
        # of pretending the damaged pixels were repaired.
        return np.array(source, copy=True), "OpenCV Telea is unavailable; masked pixels were left unchanged."
    try:
        encoded = np.asarray(np.round(np.clip(source, 0.0, 1.0) * 255.0), dtype=np.uint8)
        painted = cv2.inpaint(encoded, np.asarray(active, dtype=np.uint8) * 255, 3.0, cv2.INPAINT_TELEA)
        filled = np.asarray(painted, dtype=np.float32) / 255.0
        result = np.array(source, copy=True)
        # Explicit boolean selection is important: multiplying by an alpha can
        # alter untouched floating-point pixels by a rounding step.
        result[active] = filled[active]
        return result, None
    except Exception as exc:  # pragma: no cover - defensive for degenerate cv2 builds.
        return np.array(source, copy=True), f"Telea prefill failed ({type(exc).__name__}); masked pixels were left unchanged."


def _crop_bounds(
    frame_mask: np.ndarray,
    *,
    context_padding: float,
    min_padding: int = MIN_CONTEXT_PADDING,
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
    active = np.asarray(frame_mask > 1e-5, dtype=bool)
    ys, xs = np.where(active)
    if not len(xs):
        return None, None
    height, width = active.shape
    original = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    if original == (0, 0, width, height):
        # No unmasked context exists.  This is intentionally a hard fallback,
        # rather than a full-frame crop that would invite caption hallucination.
        return original, None
    try:
        factor = float(context_padding)
    except (TypeError, ValueError):
        factor = DEFAULT_CONTEXT_PADDING
    if not math.isfinite(factor):
        factor = DEFAULT_CONTEXT_PADDING
    factor = max(0.0, factor)
    try:
        floor = max(32, int(min_padding))
    except (TypeError, ValueError):
        floor = MIN_CONTEXT_PADDING
    bbox_width = original[2] - original[0]
    bbox_height = original[3] - original[1]
    padding = max(floor, int(round(max(bbox_width, bbox_height) * factor)))
    x0 = max(0, original[0] - padding)
    y0 = max(0, original[1] - padding)
    x1 = min(width, original[2] + padding)
    y1 = min(height, original[3] + padding)
    return original, (x0, y0, x1, y1)


def _padding_value(context_padding: Any, bbox: tuple[int, int, int, int]) -> int:
    try:
        factor = float(context_padding)
    except (TypeError, ValueError):
        factor = DEFAULT_CONTEXT_PADDING
    if not math.isfinite(factor):
        factor = DEFAULT_CONTEXT_PADDING
    factor = max(0.0, factor)
    return max(
        MIN_CONTEXT_PADDING,
        int(round(factor * max(bbox[2] - bbox[0], bbox[3] - bbox[1]))),
    )


def _resize_bound(image: np.ndarray, max_long_edge: int = MAX_CONTEXT_LONG_EDGE) -> tuple[np.ndarray, float]:
    height, width = map(int, image.shape[:2])
    bound = max(1, int(max_long_edge))
    longest = max(height, width)
    if longest <= bound:
        return np.array(image, copy=True), 1.0
    scale = float(bound) / float(longest)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    if cv2 is not None:
        resized = cv2.resize(np.asarray(image, dtype=np.float32), target, interpolation=cv2.INTER_AREA)
    else:  # deterministic nearest fallback for minimal CPU environments.
        y = np.minimum((np.arange(target[1]) / scale).astype(int), height - 1)
        x = np.minimum((np.arange(target[0]) / scale).astype(int), width - 1)
        resized = np.asarray(image)[np.ix_(y, x)]
    return np.asarray(resized, dtype=np.float32), scale


def build_context_crops(
    images: Any,
    mask: Any,
    *,
    max_frames: int = MAX_CONTEXT_FRAMES,
    context_padding: float = DEFAULT_CONTEXT_PADDING,
    max_long_edge: int = MAX_CONTEXT_LONG_EDGE,
) -> dict[str, Any]:
    """Build ≤3 neutralised RGB context crops and metadata.

    ``maskfraction`` is the active mask area divided by the full source-frame
    area; ``crop_mask_fraction`` is provided alongside it for callers that
    need the crop-local ratio.  The returned ``clips`` list is ordered exactly
    like ``cropmeta`` and contains one ``H x W x 3`` float32 array per frame.
    """

    source = _as_images(images)
    frames, height, width = map(int, source.shape[:3])
    source_mask = _as_mask(mask, frames, height, width)
    selected = select_representative_indices(source_mask, max_frames)
    clips: list[np.ndarray] = []
    cropmeta: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    for frame_index in selected:
        frame_mask = source_mask[frame_index]
        original_bbox, crop_bbox = _crop_bounds(
            frame_mask,
            context_padding=context_padding,
            min_padding=MIN_CONTEXT_PADDING,
        )
        if original_bbox is None:
            continue
        if crop_bbox is None:
            diagnostics.append(
                f"Frame {frame_index}: mask covers the full frame; surrounding context cannot be inferred safely."
            )
            continue
        prefilled, prefill_diagnostic = _telea_prefill(source[frame_index], frame_mask)
        if prefill_diagnostic:
            diagnostics.append(f"Frame {frame_index}: {prefill_diagnostic}")
        x0, y0, x1, y1 = crop_bbox
        crop = prefilled[y0:y1, x0:x1, :3]
        resized, scale = _resize_bound(crop, max_long_edge)
        crop_area = max(1, (y1 - y0) * (x1 - x0))
        mask_area = int(np.count_nonzero(frame_mask > 1e-5))
        crop_mask_area = int(np.count_nonzero(frame_mask[y0:y1, x0:x1] > 1e-5))
        metadata = {
            "frame": int(frame_index),
            "actual_frame": int(frame_index),
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "mask_bbox": [int(v) for v in original_bbox],
            "maskfraction": float(mask_area / float(height * width)),
            "mask_fraction": float(mask_area / float(height * width)),
            "crop_mask_fraction": float(crop_mask_area / float(crop_area)),
            "context_padding": _padding_value(context_padding, original_bbox),
            "scale": float(scale),
            "source_shape": [frames, height, width, int(source.shape[-1])],
            "crop_shape": [int(resized.shape[0]), int(resized.shape[1]), int(resized.shape[2])],
        }
        clips.append(np.asarray(resized, dtype=np.float32))
        cropmeta.append(metadata)
    if not selected:
        diagnostics.append("No non-empty mask frames are available for context captioning.")
    elif not clips and not diagnostics:
        diagnostics.append("No usable context crop was produced from the selected mask frames.")
    diagnostic = " ".join(dict.fromkeys(diagnostics))
    return {
        "clips": clips,
        "cropmeta": cropmeta,
        "crop_meta": cropmeta,
        "diagnostic": diagnostic,
        "diagnostics": diagnostics,
        "status": "ok" if clips else "fallback",
        "selected_frames": [int(index) for index in selected],
    }


# Short aliases are useful to the node and keep future integrations from
# depending on the longer implementation name.
prepare_context = build_context_crops
select_context_frames = select_representative_indices


def _caption_text(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("caption", "text", "description", "content"):
            if value.get(key) is not None:
                return str(value.get(key))
        return ""
    if value is None:
        return ""
    return str(value)


def sanitize_caption(value: Any, *, max_chars: int = 360) -> str:
    """Keep descriptive caption text while dropping control/instruction prose."""

    text = _caption_text(value).replace("\x00", " ")
    text = _CONTROL_TOKEN_RE.sub(" ", text)
    # Bracketed model directives are often copied verbatim from a VLM prompt;
    # they are not visual evidence and may contain hidden control text.
    text = _CONTROL_BRACKET_RE.sub(" ", text)
    sentences = re.split(r"(?<=[.!?;])\s+|[\r\n]+", text)
    kept: list[str] = []
    for sentence in sentences:
        candidate = _WHITESPACE_RE.sub(" ", sentence).strip(" \t.,;:/")
        if not candidate:
            continue
        lower = candidate.lower()
        if _MALICIOUS_RE.search(candidate) or any(token in lower for token in ("system prompt", "jailbreak", "developer message", "assistant:")):
            continue
        if _LEADING_IMPERATIVE_RE.search(candidate) or _INSTRUCTION_RE.search(candidate):
            # A descriptive sentence can include the word "camera" as a
            # scene fact, but captions containing explicit prompt controls are
            # safer to omit than to pass through.  The actual source frames
            # remain available to the H3 conditioning path.
            continue
        kept.append(candidate)
    text = _WHITESPACE_RE.sub(" ", " ".join(kept)).strip()
    if len(text) > max(1, int(max_chars)):
        text = text[: int(max_chars)].rsplit(" ", 1)[0].rstrip(" ,;:")
    return text


def _clean_hint(value: Any, *, max_chars: int = 360) -> str:
    """Sanitise optional user text without treating it as caption evidence."""

    text = _caption_text(value).replace("\x00", " ")
    text = _CONTROL_TOKEN_RE.sub(" ", text)
    text = _CONTROL_BRACKET_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if _MALICIOUS_RE.search(text) or any(token in text.lower() for token in ("system prompt", "jailbreak", "developer message")):
        return ""
    if len(text) > max(1, int(max_chars)):
        text = text[: int(max_chars)].rsplit(" ", 1)[0].rstrip(" ,;:")
    return text


def _compact_observation(value: str) -> str:
    """Keep concrete source appearance, omitting speculative fill geometry."""
    text = re.split(r"\s+(?=(?:The|This|It)\s+(?:image|fabric|object|surface|background|material)\b)", value, maxsplit=1)[0]
    text = re.sub(r"(?i)^(?:the|this)\s+image\s+(?:is|shows)\s+", "", text)
    text = re.sub(r"(?i)^(?:a\s+)?close[-\s]*up(?:\s+(?:shot|photograph|photo|view))?\s+of\s+", "", text)
    # Neutralised holes can look polygonal to a caption model. Those shapes
    # are not evidence of the missing object and must not become edit targets.
    text = re.split(r"(?i)\b(?:with|featuring|having)\s+(?=[^.;]{0,160}\b(?:hexagon\w*|triang\w*|polygon\w*|geometric)\b)", text, maxsplit=1)[0]
    text = re.sub(r"(?i)\b(?:a\s+)?piece\s+of\s+", "", text)
    return text.strip(" .,;:")[:180].rsplit(" ", 1)[0] if len(text) > 180 else text.strip(" .,;:")


INSTRUCTION_MAX_CHARS = 500


def normalize_edit_instruction(value: Any) -> str:
    """Keep an edit request separate from captions and restoration defaults."""
    text = _WHITESPACE_RE.sub(" ", str(value or "").replace("\x00", " ")).strip()
    if len(text) > INSTRUCTION_MAX_CHARS:
        raise ValueError(f"Опишите правку в пределах {INSTRUCTION_MAX_CHARS} символов.")
    return text


def compose_edit_instruction_prompt(instruction: str) -> str:
    """Describe a local H3 edit conditioned by the source latent and mask."""
    edit = _WHITESPACE_RE.sub(" ", instruction.replace("\x00", " ")).strip()
    if not edit:
        raise ValueError("Введите задачу в поле «Что изменить?».")
    detail = ""
    bow_tie = re.fullmatch(
        r"(?:change|replace) (?:the )?(?:neck\s*)?tie (?:to|into|with) (?:an? |the )?"
        r"((?:[\w-]+ ){0,4}bow tie)[.!]?", edit, re.IGNORECASE,
    )
    if bow_tie:
        appearance = bow_tie[1]
        article = "an" if appearance[0].lower() in "aeiou" else "a"
        edit = f"Replace the necktie with {article} {appearance} at the shirt collar"
        detail = "Keep the bow tie attached to the collar and moving with the wearer. "
    elif re.fullmatch(r"(?:change|replace) (?:the )?door (?:to|into|with) (?:a |the )?wall[.!]?", edit, re.IGNORECASE):
        edit = "Replace the door with a continuous section of wall"
        detail = "Extend the adjacent wall's surface and perspective through the former doorway. "
    if not edit.endswith((".", "!", "?")):
        edit += "."
    return (
        f"{edit} {detail}"
        "Match the source lighting and perspective. Keep the edited area consistent from frame to frame. "
        "Preserve the source camera movement, framing, timing and all areas outside the mask."
    )


def compose_repair_prompt(
    captions: Iterable[Any] | Any = (),
    user_hint: Any = "",
    *,
    mode: str = "auto",
    manual_prompt: Any = "",
    diagnostic: str = "",
    max_chars: int = PROMPT_MAX_CHARS,
) -> str:
    """Compose a concise positive H3 masked-repair prompt.

    Captions are treated as descriptive observations only.  The function never
    claims that a CV heuristic is an AI caption and never adds H3's external
    ``Image1``/``Video1`` reference syntax: this local workflow receives source
    frames directly.  ``mode='manual'`` bypasses captions and uses
    ``manual_prompt`` when supplied, otherwise the same safe generic fallback.
    """

    mode_value = str(mode or "auto").strip().lower()
    if mode_value not in {"auto", "manual"}:
        mode_value = "auto"
    if mode_value == "manual":
        body = _clean_hint(manual_prompt, max_chars=480)
        if not body:
            body = _clean_hint(user_hint, max_chars=480)
    else:
        if isinstance(captions, (str, bytes, Mapping)):
            values: Iterable[Any] = [captions]
        else:
            try:
                iter(captions)
            except TypeError:
                values = [captions]
            else:
                values = captions
        cleaned: list[str] = []
        for value in values:
            item = _compact_observation(sanitize_caption(value))
            if item and item not in cleaned:
                cleaned.append(item)
        body = "; ".join(cleaned[:3])
        hint = _clean_hint(user_hint)
        if hint:
            body = f"{body} {hint}".strip()
    if body:
        prompt = f"{DEFAULT_REPAIR_PROMPT} Observed source appearance: {body}."
    else:
        prompt = DEFAULT_REPAIR_PROMPT
    # Keep a diagnostic factual and short; it is useful in a UI status string,
    # but should not turn an unavailable caption into an invented description.
    if diagnostic and not body:
        prompt += " Caption context is unavailable; infer the repair from adjacent unmasked pixels only."
    prompt = _WHITESPACE_RE.sub(" ", prompt).strip()
    limit = max(128, int(max_chars))
    if len(prompt) > limit:
        prompt = prompt[:limit].rsplit(" ", 1)[0].rstrip(" ,;:") + "."
    return prompt


compose_prompt = compose_repair_prompt


def prepare_caption_inputs(
    images: Any,
    mask: Any,
    max_frames: int = MAX_CONTEXT_FRAMES,
    context_padding: float = DEFAULT_CONTEXT_PADDING,
) -> dict[str, Any]:
    """Compatibility wrapper used by the Comfy auto-prompt node.

    The return value is the same plain dictionary as :func:`build_context_crops`.
    Keeping it dictionary-shaped avoids a dataclass dependency in custom-node
    import paths and makes the fields directly serialisable to a UI status.
    """

    return build_context_crops(
        images,
        mask,
        max_frames=max_frames,
        context_padding=context_padding,
    )


def build_prompt_context(
    images: Any,
    mask: Any,
    captions: Iterable[Any] | Any = (),
    user_hint: Any = "",
    *,
    mode: str = "auto",
    manual_prompt: Any = "",
    max_frames: int = MAX_CONTEXT_FRAMES,
    context_padding: float = DEFAULT_CONTEXT_PADDING,
) -> dict[str, Any]:
    """Prepare crops and compose a prompt in one deterministic helper.

    ``captions`` is supplied by the node's selected caption backend; no model
    is invoked here.  The resulting dictionary is intentionally small and
    stable for the UI contract: ``prompt``, ``caption``, ``status``,
    ``diagnostic``, ``clips``, and ``cropmeta``.
    """

    context = build_context_crops(
        images,
        mask,
        max_frames=max_frames,
        context_padding=context_padding,
    )
    if isinstance(captions, (str, bytes, Mapping)):
        caption_values = [captions]
    else:
        try:
            caption_values = list(captions)
        except TypeError:
            caption_values = [captions]
    caption_items = [sanitize_caption(value) for value in caption_values]
    caption_items = [item for index, item in enumerate(caption_items) if item and item not in caption_items[:index]]
    prompt = compose_repair_prompt(
        caption_items,
        user_hint,
        mode=mode,
        manual_prompt=manual_prompt,
        diagnostic=context.get("diagnostic", ""),
    )
    status = "ok" if context.get("clips") else "fallback"
    return {
        "prompt": prompt,
        "caption": " ".join(caption_items),
        "status": status,
        "diagnostic": context.get("diagnostic", ""),
        "clips": context.get("clips", []),
        "cropmeta": context.get("cropmeta", []),
        "crop_meta": context.get("cropmeta", []),
        "selected_frames": context.get("selected_frames", []),
    }


__all__ = [
    "DEFAULT_CONTEXT_PADDING",
    "DEFAULT_REPAIR_PROMPT",
    "MAX_CONTEXT_FRAMES",
    "MAX_CONTEXT_LONG_EDGE",
    "MIN_CONTEXT_PADDING",
    "PROMPT_MAX_CHARS",
    "build_context_crops",
    "build_prompt_context",
    "compose_prompt",
    "compose_repair_prompt",
    "prepare_caption_inputs",
    "prepare_context",
    "sanitize_caption",
    "select_context_frames",
    "select_representative_indices",
]
