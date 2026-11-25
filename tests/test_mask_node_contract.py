import copy
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ths_universal_inpaint.mask_nodes import THSUniversalMaskEditor  # noqa: E402
import ths_universal_inpaint.mask_nodes as mask_nodes  # noqa: E402
from ths_universal_inpaint.mask_core import MaskDataError, frame_fingerprint  # noqa: E402


def test_node_contract_matches_workflow_schema_without_importing_torch():
    inputs = THSUniversalMaskEditor.INPUT_TYPES()
    required = inputs["required"]
    assert required["images"][0] == "IMAGE"
    assert required["mask_data"][0] == "STRING"
    assert required["mask_data"][1]["default"] == "{}"
    assert required["tracking_mode"][1]["default"] == "sam3"
    assert required["grow"][1]["default"] == 8
    assert required["feather"][1]["default"] == 4
    assert required["start_frame"][1]["default"] == 0
    assert required["end_frame"][1]["default"] == -1
    assert inputs["optional"]["fps"][0] == "FLOAT"
    assert inputs["hidden"]["unique_id"] == "UNIQUE_ID"
    assert THSUniversalMaskEditor.RETURN_TYPES == ("MASK", "IMAGE")
    assert THSUniversalMaskEditor.RETURN_NAMES == ("edit_mask", "preview")
    assert THSUniversalMaskEditor.OUTPUT_NODE is True


class _ExecutionBlocker:
    calls = []

    def __new__(cls, message):
        cls.calls.append(message)
        instance = super().__new__(cls)
        instance.message = message
        return instance


def _install_execution_blocker(monkeypatch):
    _ExecutionBlocker.calls = []
    package = types.ModuleType("comfy_execution")
    graph_utils = types.ModuleType("comfy_execution.graph_utils")
    graph_utils.ExecutionBlocker = _ExecutionBlocker
    package.graph_utils = graph_utils
    monkeypatch.setitem(sys.modules, "comfy_execution", package)
    monkeypatch.setitem(sys.modules, "comfy_execution.graph_utils", graph_utils)
    monkeypatch.setattr(mask_nodes, "encode_preview_assets", lambda images, masks: [
        {"frame": 0, "mask_zero": not bool(np.any(masks))}
    ])
    monkeypatch.setattr(mask_nodes, "cache_preview_assets", lambda images, masks, **kwargs: (
        [{"frame": 0, "mask_zero": not bool(np.any(masks))}],
        {"available": True, "error": None, "session": "test", "frames": 1},
    ))


def test_source_mismatch_returns_source_only_preview_and_blocks_both_outputs(monkeypatch):
    _install_execution_blocker(monkeypatch)
    source = np.zeros((2, 12, 16, 3), dtype=np.float32)
    fingerprint = frame_fingerprint(source)
    document = {
        "source": {"frames": 3, "width": 16, "height": 12, "fingerprint": fingerprint},
        "frames": {"0": {"events": [{
            "type": "rect", "mode": "add", "x": 0.1, "y": 0.1, "w": 0.4, "h": 0.4,
        }]}},
    }
    original = copy.deepcopy(document)
    response = THSUniversalMaskEditor().run(
        source, document, tracking_mode="static", grow=0, feather=0, fps=24.0
    )
    assert document == original
    assert _ExecutionBlocker.calls == [None, None]
    assert all(getattr(item, "message", "not-blocked") is None for item in response["result"])
    assert "universal_mask_data" not in response["ui"]
    assert response["ui"]["universal_mask_preview"][0]["mask_zero"] is True
    assert response["ui"]["universal_mask_frames"][0]["mask_zero"] is True
    meta = response["ui"]["universal_mask_meta"][0]
    assert meta["source_mismatch"] is True
    assert meta["source_mismatch_reason"] == "frames"
    assert meta["previous_source"] == {
        "frames": 3, "width": 16, "height": 12, "fingerprint": fingerprint,
    }
    assert meta["source"] == {
        "frames": 2, "width": 16, "height": 12, "fingerprint": fingerprint,
    }
    assert meta["has_mask"] is False and meta["anchors"] == []


def test_refinement_failure_returns_source_only_preview_and_blocks_both_outputs(monkeypatch):
    _install_execution_blocker(monkeypatch)
    source = np.zeros((2, 12, 16, 3), dtype=np.float32)
    document = {"frames": {"0": {"events": [{
        "type": "rect", "mode": "add", "refine": True,
        "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0,
    }]}}}
    response = THSUniversalMaskEditor().run(
        source, document, tracking_mode="static", grow=0, feather=0
    )
    assert _ExecutionBlocker.calls == [None, None]
    assert all(getattr(item, "message", "not-blocked") is None for item in response["result"])
    assert "universal_mask_data" not in response["ui"]
    meta = response["ui"]["universal_mask_meta"][0]
    assert "refinement_error" in meta and meta["refinement_error"]
    assert "source_mismatch" not in meta
    assert meta["has_mask"] is False and meta["anchors"] == []


def test_invalid_json_still_raises_instead_of_becoming_recovery_preview(monkeypatch):
    _install_execution_blocker(monkeypatch)
    source = np.zeros((1, 8, 8, 3), dtype=np.float32)
    with pytest.raises(MaskDataError) as exc_info:
        THSUniversalMaskEditor().run(source, "{not-json", tracking_mode="static")
    assert type(exc_info.value) is MaskDataError
    assert _ExecutionBlocker.calls == []


def test_disabling_preview_output_still_prepares_editor_frames(monkeypatch):
    _install_execution_blocker(monkeypatch)
    monkeypatch.setattr(mask_nodes, "_torch_tensor", lambda x: x)
    monkeypatch.setattr(mask_nodes, "_preview_tensor", lambda *_: pytest.fail("must not allocate RGB preview"))
    result = THSUniversalMaskEditor().run(
        np.zeros((2, 12, 16, 3), dtype=np.float32),
        {"frames": {"0": {"events": [{"type": "rect", "x": .1, "y": .1, "w": .4, "h": .4}]}}},
        tracking_mode="static", preview_output=False,
    )
    assert result["ui"]["universal_mask_frames"]
    assert result["result"][0].any()
    assert isinstance(result["result"][1], _ExecutionBlocker)


def test_scoped_sam3_frame_returns_only_changed_masks_and_a_trackable_seed(monkeypatch):
    import base64
    import io
    import json

    from PIL import Image
    from ths_universal_inpaint import mask_sam3_video as video

    _install_execution_blocker(monkeypatch)
    source = np.zeros((3, 24, 64, 3), dtype=np.float16)[:, :, ::2, :]
    source[1] = 0.25
    source[2, 0, 0] = [np.nan, np.inf, -np.inf]
    native_mask = np.zeros(source.shape[1:3], dtype=bool)
    native_mask[4:18, 6:20] = True
    calls = []

    def predict(image, prompt, *, keep_loaded):
        calls.append(keep_loaded)
        np.testing.assert_array_equal(image, source[1])
        if prompt["labels"] == [1, 0]:
            return native_mask
        other = np.zeros_like(native_mask)
        other[4:18, 14:25] = True
        return other

    def reject_full_preview(*args, **kwargs):
        pytest.fail("one click must not build or encode a full-clip preview")

    video.clear_sam3_video_cache()
    monkeypatch.setattr(video, "_predict_native", predict)
    for name in ("build_mask_sequence", "encode_preview_assets", "cache_preview_assets", "_preview_tensor"):
        monkeypatch.setattr(mask_nodes, name, reject_full_preview)
    document = {
        "editor_request": mask_nodes.SAM3_FRAME_PREVIEW,
        "tracking_mode": "sam3",
        "source": {"frames": 3, "width": 32, "height": 24, "fingerprint": frame_fingerprint(source)},
        "sam3": {"action": "frame", "frame": 1, "prompt": {"frame": 1, "points": [[.3, .3], [.8, .8]], "labels": [1, 0]}},
        "frames": {"1": {"inherit": True, "events": [{"type": "rect", "mode": "add", "x": .8, "y": .8, "w": .1, "h": .1}]}},
    }
    result = THSUniversalMaskEditor().run(source, json.dumps(document), tracking_mode="sam3", grow=0, feather=0)
    ui = result["ui"]
    assert calls == [True]
    assert all(isinstance(output, _ExecutionBlocker) for output in result["result"])
    assert "universal_mask_frames" not in ui and "universal_mask_preview" not in ui
    patch, = ui["universal_mask_frame_patch"]
    assert patch["frame"] == 1

    def pixels(name):
        payload = base64.b64decode(patch[name].split(",", 1)[1])
        return np.asarray(Image.open(io.BytesIO(payload))) > 0

    np.testing.assert_array_equal(pixels("edit_base"), native_mask)
    assert pixels("mask").sum() > native_mask.sum()
    saved = json.loads(ui["universal_mask_data"][0])
    assert "editor_request" not in saved
    token = saved["sam3"]["seed_token"]
    normalized = np.nan_to_num(source.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    assert video._lookup_seed(normalized, saved, 1, token).frame == 1
    normalized[2, 1, 1, 0] = 0.5
    with pytest.raises(video.SAM3VideoError, match="does not belong"):
        video._lookup_seed(normalized, saved, 1, token)
    assert ui["universal_mask_meta"][0]["sam3"]["track_ready"] is False

    sam3 = document["sam3"]
    sam3["objects"] = [sam3.pop("prompt"), {"frame": 1, "points": [[.6, .3]], "labels": [1]}]
    grouped = THSUniversalMaskEditor().run(source, json.dumps(document), tracking_mode="sam3", grow=0, feather=0)
    data = grouped["ui"]["universal_mask_frame_patch"][0]["object_map"].split(",", 1)[1]
    membership = np.asarray(Image.open(io.BytesIO(base64.b64decode(data))))
    assert list(membership[8, [2, 8, 16, 24]]) == [0, 1, 3, 2]
