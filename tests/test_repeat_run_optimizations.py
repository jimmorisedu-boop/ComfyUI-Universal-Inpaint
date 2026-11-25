from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ths_universal_inpaint import mask_core  # noqa: E402


def _clip(frames: int = 5, height: int = 32, width: int = 40) -> np.ndarray:
    images = np.zeros((frames, height, width, 3), dtype=np.float32)
    for frame in range(frames):
        images[frame, 10:18, 4 + frame : 12 + frame, :2] = 1.0
    return images


def _document(x: float = 0.1):
    return {
        "frames": {
            "0": {
                "events": [
                    {"type": "rect", "mode": "add", "x": x, "y": 0.2, "w": 0.2, "h": 0.2}
                ]
            }
        }
    }


def test_raw_tracking_cache_reuses_preprocessed_result_but_reapplies_edges(monkeypatch):
    images = _clip()
    mask_core.clear_raw_tracking_cache()
    calls = []
    original = mask_core._flow_between

    def counted(previous, following):
        calls.append(1)
        return original(previous, following)

    monkeypatch.setattr(mask_core, "_flow_between", counted)
    first, _parsed, _anchors, first_bases = mask_core.build_mask_sequence(
        images, _document(), tracking_mode="optical_flow", grow=0, feather=0, return_edit_bases=True
    )
    second, _parsed, _anchors, second_bases = mask_core.build_mask_sequence(
        images, _document(), tracking_mode="optical_flow", grow=2, feather=1, return_edit_bases=True
    )
    assert calls, "the first tracking pass should compute optical flow"
    info = mask_core.raw_tracking_cache_info()
    assert info["hits"] >= 1

    # The edge controls are the only changed inputs: compare the hit result to
    # an uncached execution with the same controls for bit-identical output.
    mask_core.clear_raw_tracking_cache()
    expected, _parsed, _anchors, expected_bases = mask_core.build_mask_sequence(
        images, _document(), tracking_mode="optical_flow", grow=2, feather=1, return_edit_bases=True
    )
    np.testing.assert_array_equal(second, expected)
    np.testing.assert_array_equal(second_bases, expected_bases)
    assert not np.shares_memory(first_bases, second_bases)


def test_raw_tracking_cache_invalidates_source_document_mode_and_range(monkeypatch):
    images = _clip()
    mask_core.clear_raw_tracking_cache()
    calls = []
    original = mask_core._flow_between
    monkeypatch.setattr(mask_core, "_flow_between", lambda a, b: (calls.append(1) or original(a, b)))

    def run(source, document, mode="optical_flow", start=0, end=-1):
        return mask_core.build_mask_sequence(
            source,
            document,
            tracking_mode=mode,
            start_frame=start,
            end_frame=end,
            grow=0,
            feather=0,
        )

    run(images, _document())
    first_count = len(calls)
    run(images, _document())
    assert len(calls) == first_count
    run(images, _document(0.2))
    assert len(calls) > first_count
    after_document = len(calls)
    run(images, _document(), mode="keyframes")
    # Keyframes intentionally hold the nearest explicit anchor and do not
    # invoke optical flow; this keeps the mode deterministic and avoids
    # silently selecting a different tracker backend.
    assert len(calls) == after_document
    misses_before_range = mask_core.raw_tracking_cache_info()["misses"]
    run(images, _document(), start=1, end=3)
    assert mask_core.raw_tracking_cache_info()["misses"] > misses_before_range
    changed = images.copy()
    changed[0, 0, 0, 0] = 0.75  # full-content identity catches unsampled edits too
    after_range = len(calls)
    run(changed, _document())
    assert len(calls) > after_range


def test_raw_tracking_cache_is_bounded_and_oversized_items_bypass(monkeypatch):
    images = _clip(frames=2, height=8, width=8)
    mask_core.clear_raw_tracking_cache()
    monkeypatch.setattr(mask_core, "MAX_RAW_TRACKING_CACHE_ITEM_BYTES", 1)
    mask_core.build_mask_sequence(images, _document(), grow=0, feather=0)
    mask_core.build_mask_sequence(images, _document(), grow=1, feather=1)
    info = mask_core.raw_tracking_cache_info()
    assert info["entries"] == 0
    assert info["bypasses"] >= 1
