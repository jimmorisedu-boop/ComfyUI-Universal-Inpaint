from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ths_universal_inpaint import auto_prompt_nodes as apn  # noqa: E402
from ths_universal_inpaint import mask_core  # noqa: E402
from ths_universal_inpaint.repair_nodes import (  # noqa: E402
    THSUniversalH3PadAudio,
    THSUniversalPromptText,
    THSUniversalRepairCompare,
    pad_source_inputs,
)


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
    after_mode = len(calls)
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


def test_pad_audio_matches_existing_pad_source_waveform_bit_for_bit():
    images = np.arange(6 * 4 * 4 * 3, dtype=np.float32).reshape(6, 4, 4, 3)
    mask = np.zeros((6, 4, 4), dtype=np.float32)
    waveform = np.arange(2 * 31, dtype=np.float32).reshape(2, 31)
    audio = {"waveform": waveform, "sample_rate": 24}
    expected = pad_source_inputs(images, mask, audio, model_fps=24, modulo=17, remainder=5)
    actual = THSUniversalH3PadAudio().run(
        images, audio=audio, model_fps=24, modulo=17, remainder=5, silence_sample_rate=48_000
    )
    np.testing.assert_array_equal(actual[0]["waveform"], expected[2]["waveform"])
    assert actual[0]["sample_rate"] == expected[2]["sample_rate"]
    assert actual[1:] == expected[3:]

    silent_expected = pad_source_inputs(images, mask, None, model_fps=24, modulo=17, remainder=5)[2]
    silent_actual = THSUniversalH3PadAudio().run(images, model_fps=24, modulo=17, remainder=5)[0]
    np.testing.assert_array_equal(silent_actual["waveform"], silent_expected["waveform"])


def test_manual_autoprompt_skips_context_prepare_and_disabled_preview_is_tiny(monkeypatch):
    images = np.zeros((3, 32, 32, 3), dtype=np.float32)
    mask = np.zeros((3, 32, 32), dtype=np.float32)
    mask[:, 8:16, 8:16] = 1.0
    monkeypatch.setattr(apn, "prepare_caption_inputs", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("manual mode must skip context preparation")))
    monkeypatch.setattr(apn, "_preview_tensor", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("disabled preview must not allocate overlay")))
    result = apn.THSUniversalAutoPrompt().run(
        images, mask, mode="manual", manual_prompt="keep this exact text", preview_enabled=False
    )
    assert result["result"][2].shape[0:3] == (1, 1, 1)
    assert "keep this exact text" in result["result"][0]


def test_prompt_text_roundtrips_multiline_exactly():
    text = "line one\n  line two\n\nline four"
    assert THSUniversalPromptText().run(text) == (text,)
    assert THSUniversalPromptText.RETURN_TYPES == ("STRING",)


def test_repair_compare_uses_bounded_temp_descriptors(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(get_temp_directory=lambda: str(tmp_path)))
    source = np.zeros((3, 8, 10, 3), dtype=np.float32)
    result = np.ones_like(source)
    payload = THSUniversalRepairCompare().run(source, result, fps=24)
    frames = payload["ui"]["universal_repair_compare_frames"]
    assert [item["frame"] for item in frames] == [0, 1, 2]
    assert all("data_url" not in item and "original" in item and "result" in item for item in frames)
    assert all((tmp_path / item["original"]["subfolder"] / item["original"]["filename"]).is_file() for item in frames)
    assert payload["result"] == ()
