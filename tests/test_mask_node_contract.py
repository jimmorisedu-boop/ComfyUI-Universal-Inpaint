import copy
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ths_universal_inpaint.mask_nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    THSUniversalMaskEditor,
    THSUniversalMaskRestoreCrop,
)
import ths_universal_inpaint.mask_nodes as mask_nodes  # noqa: E402
import ths_universal_inpaint.mask_core as mask_core  # noqa: E402
from ths_universal_inpaint.mask_core import (  # noqa: E402
    MaskDataError,
    frame_fingerprint,
    mask_extent,
    parse_crop_info,
    plan_crop_window,
    restore_crop_window,
)


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
    assert THSUniversalMaskEditor.RETURN_TYPES == ("MASK", "IMAGE", "IMAGE", "MASK", "STRING")
    assert THSUniversalMaskEditor.RETURN_NAMES == ("edit_mask", "preview", "cropped_images", "cropped_mask", "crop_info")
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
    assert _ExecutionBlocker.calls == [None, None, None, None, None]
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
    assert _ExecutionBlocker.calls == [None, None, None, None, None]
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
    result = THSUniversalMaskEditor().run(source, json.dumps(document), tracking_mode="sam3", grow=2, feather=1)
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
    assert pixels("raw_mask").sum() > native_mask.sum()
    assert pixels("mask").sum() > pixels("raw_mask").sum()
    patch_meta = ui["universal_mask_meta"][0]
    assert patch_meta["mask_postprocess"] == {"grow": 2, "feather": 1}
    assert patch_meta["raw_crop_extent"]["frames"] == [1]
    assert patch_meta["crop_extent"]["box"] != patch_meta["raw_crop_extent"]["box"]
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


def test_autocrop_modes_crop_images_and_masks_accurately(monkeypatch):
    import json
    from ths_universal_inpaint.mask_core import mask_extent, plan_crop_window

    monkeypatch.setattr(mask_nodes, "encode_preview_assets", lambda images, masks: [])
    monkeypatch.setattr(mask_nodes, "cache_preview_assets", lambda images, masks, **kwargs: (
        [],
        {"available": True, "error": None, "session": "crop-test", "frames": len(images)},
    ))
    monkeypatch.setattr(mask_nodes, "_torch_tensor", lambda x: np.asarray(x, dtype=np.float32))
    monkeypatch.setattr(mask_nodes, "_preview_tensor", lambda x, m: np.asarray(x, dtype=np.float32))

    source = np.zeros((3, 300, 400, 3), dtype=np.float32)
    source[:, 60:140, 80:160, 1] = 1.0
    auto_doc = {
        "crop": {"mode": "auto", "margin": 20},
        "frames": {
            "0": {"events": [{"type": "rect", "mode": "add", "x": 80 / 400, "y": 60 / 300, "w": 80 / 400, "h": 80 / 300}]}
        },
    }
    res_auto = THSUniversalMaskEditor().run(source, auto_doc, tracking_mode="static", grow=4, feather=2)
    mask_full, _, cropped_img, cropped_mask, crop_info_raw = res_auto["result"]
    crop_info = json.loads(crop_info_raw)
    assert crop_info["mode"] == "auto"
    assert tuple(cropped_img.shape) == (3, crop_info["height"], crop_info["width"], 3)
    assert tuple(cropped_mask.shape) == (3, crop_info["height"], crop_info["width"])
    assert 120 <= crop_info["width"] < 400 and 100 <= crop_info["height"] < 300
    assert float(cropped_mask.max()) == pytest.approx(1.0)
    run_meta = res_auto["ui"]["universal_mask_meta"][0]
    assert run_meta["crop_extent"]["frames"] == [0, 1, 2]
    assert run_meta["mask_postprocess"] == {"grow": 4, "feather": 2}
    assert run_meta["raw_crop_extent"]["frames"] == [0, 1, 2]
    rx0, ry0, rx1, ry1 = run_meta["raw_crop_extent"]["box"]
    gx0, gy0, gx1, gy1 = run_meta["crop_extent"]["box"]
    assert 78 <= rx0 <= 80 and 58 <= ry0 <= 60 and 160 <= rx1 <= 162 and 140 <= ry1 <= 142
    assert gx0 < rx0 and gy0 < ry0 and gx1 > rx1 and gy1 > ry1

    # Manual crop expands outward cleanly while protecting the masked bounding box
    manual_plan = plan_crop_window(
        (3, 300, 400),
        mask_extent(mask_full),
        {"mode": "manual", "margin": 20, "box": [40, 20, 240, 220], "size": [400, 300]},
    )
    assert manual_plan["box"] == [40, 20, 240, 220]
    assert manual_plan["multiple"] == 1

    # Side-multiple crop rounds width & height to exact multiples in run() and plan_crop_window()
    mult_doc = {
        "crop": {"mode": "auto", "margin": 20, "multiple": 64},
        "frames": auto_doc["frames"],
    }
    res_mult = THSUniversalMaskEditor().run(source, mult_doc, tracking_mode="static", grow=4, feather=2)
    _, _, mult_img, mult_mask, mult_info_raw = res_mult["result"]
    mult_info = json.loads(mult_info_raw)
    assert mult_info["multiple"] == 64
    assert mult_info["width"] % 64 == 0 and mult_info["height"] % 64 == 0
    assert tuple(mult_img.shape) == (3, mult_info["height"], mult_info["width"], 3)
    assert tuple(mult_mask.shape) == (3, mult_info["height"], mult_info["width"])

    # Non-divisible 1920x1080 canvas with >=90% fallback preserves multiple=64
    hd_plan = plan_crop_window(
        (2, 1080, 1920),
        {"box": [20, 20, 1900, 1060], "frames": [0, 1]},
        {"mode": "auto", "margin": 64, "multiple": 64},
    )
    assert hd_plan["width"] % 64 == 0 and hd_plan["height"] == 1024

    # Invalid crop mode or multiple must raise MaskDataError
    with pytest.raises(MaskDataError, match="crop.mode"):
        THSUniversalMaskEditor().run(source, {"crop": {"mode": "invalid"}, "frames": {}}, tracking_mode="static")
    for bad_mult in (0, -8, 999):
        with pytest.raises(MaskDataError, match="crop.multiple"):
            plan_crop_window((1, 300, 400), mask_extent(mask_full), {"mode": "auto", "margin": 10, "multiple": bad_mult})


def test_restore_crop_node_contract_and_mappings():
    assert NODE_CLASS_MAPPINGS["THSUniversalMaskEditor"] is THSUniversalMaskEditor
    assert NODE_CLASS_MAPPINGS["THSUniversalMaskRestoreCrop"] is THSUniversalMaskRestoreCrop
    assert NODE_DISPLAY_NAME_MAPPINGS["THSUniversalMaskEditor"] == "DAI Video Mask Editor"
    assert NODE_DISPLAY_NAME_MAPPINGS["THSUniversalMaskRestoreCrop"] == "DAI Video Mask Restore Crop"
    assert THSUniversalMaskEditor.CATEGORY == "DAI/Universal Mask Editor"
    assert "dai" in [alias.lower() for alias in THSUniversalMaskEditor.SEARCH_ALIASES]
    assert "dai" in [alias.lower() for alias in THSUniversalMaskRestoreCrop.SEARCH_ALIASES]

    inputs = THSUniversalMaskRestoreCrop.INPUT_TYPES()
    required = inputs["required"]
    optional = inputs["optional"]
    assert required["original_images"][0] == "IMAGE"
    assert required["cropped_images"][0] == "IMAGE"
    assert required["crop_info"][0] == "STRING"
    assert required["crop_info"][1]["forceInput"] is True

    assert optional["mask"][0] == "MASK"
    assert optional["edge_feather"][1]["default"] == 0
    assert optional["resize_method"][1]["default"] == "lanczos"
    assert optional["invert_mask"][1]["default"] is False
    assert optional["color_match"][1]["default"] == "none"

    assert THSUniversalMaskRestoreCrop.RETURN_TYPES == ("IMAGE", "MASK")
    assert THSUniversalMaskRestoreCrop.RETURN_NAMES == ("image", "mask")
    assert THSUniversalMaskRestoreCrop.CATEGORY == "DAI/Universal Mask Editor"
    assert THSUniversalMaskRestoreCrop.OUTPUT_NODE is False


def test_restore_crop_roundtrip_with_cropped_and_full_masks(monkeypatch):
    monkeypatch.setattr(mask_nodes, "_torch_tensor", lambda array: np.asarray(array, dtype=np.float32))

    rng = np.random.default_rng(42)
    source = rng.uniform(0.1, 0.7, size=(3, 128, 160, 3)).astype(np.float32)
    mask_data = json.dumps(
        {
            "crop": {"mode": "auto", "margin": 8, "multiple": 16},
            "frames": {
                "0": {
                    "events": [
                        {"type": "rect", "mode": "add", "x": 0.30, "y": 0.30, "w": 0.25, "h": 0.25},
                    ]
                }
            },
        }
    )

    editor_out = THSUniversalMaskEditor().run(
        source,
        mask_data,
        tracking_mode="static",
        grow=0,
        feather=0,
    )["result"]
    edit_mask, _preview, cropped_images, cropped_mask, crop_info = editor_out
    info = parse_crop_info(crop_info, orig_h=128, orig_w=160)
    left, top, right, bottom = info["box"]
    assert info["multiple"] == 16
    assert (right - left) % 16 == 0
    assert (bottom - top) % 16 == 0

    edited_crop = np.clip(cropped_images + 0.25, 0.0, 1.0)
    restored_from_crop, applied_from_crop = THSUniversalMaskRestoreCrop().run(
        source,
        edited_crop,
        crop_info,
        mask=cropped_mask,
    )
    restored_from_full, applied_from_full = THSUniversalMaskRestoreCrop().run(
        source,
        edited_crop,
        crop_info,
        mask=edit_mask,
    )

    np.testing.assert_allclose(restored_from_crop, restored_from_full, atol=1e-6)
    np.testing.assert_allclose(applied_from_crop, applied_from_full, atol=1e-6)

    outside = edit_mask < 0.01
    inside = edit_mask > 0.99
    assert np.any(outside) and np.any(inside)
    np.testing.assert_allclose(restored_from_crop[outside], source[outside], atol=1e-6)
    expected_inside = np.zeros_like(source)
    expected_inside[:, top:bottom, left:right, :] = edited_crop
    np.testing.assert_allclose(restored_from_crop[inside], expected_inside[inside], atol=1e-6)

    restored_inv, applied_inv = THSUniversalMaskRestoreCrop().run(
        source,
        edited_crop,
        crop_info,
        mask=cropped_mask,
        invert_mask=True,
    )
    outside_box = np.ones((3, 128, 160), dtype=bool)
    outside_box[:, top:bottom, left:right] = False
    np.testing.assert_allclose(restored_inv[outside_box], source[outside_box], atol=1e-6)
    assert np.all(applied_inv[outside_box] == 0.0)
    # Inverted matte inside the crop box takes unmasked border pixels instead of center.
    assert np.allclose(applied_inv[:, top, left], 1.0, atol=1e-6)


def test_restore_crop_upscaled_diffusion_and_color_match(monkeypatch):
    monkeypatch.setattr(mask_nodes, "_torch_tensor", lambda array: np.asarray(array, dtype=np.float32))

    ys = np.linspace(0.2, 0.5, 128, dtype=np.float32)[:, None]
    xs = np.linspace(0.3, 0.6, 160, dtype=np.float32)[None, :]
    r_ch, g_ch, b_ch = np.broadcast_arrays(ys + xs * 0.2, 0.25 + ys * 0.4, 0.35 + xs * 0.3)
    base_frame = np.stack([r_ch, g_ch, b_ch], axis=-1)
    source = np.repeat(base_frame[None, ...], 2, axis=0).astype(np.float32)

    full_mask = np.zeros((2, 128, 160), dtype=np.float32)
    full_mask[:, 40:72, 48:80] = 1.0
    crop_info = json.dumps(plan_crop_window(full_mask.shape, mask_extent(full_mask), {"mode": "auto", "margin": 16}))
    info = parse_crop_info(crop_info, orig_h=128, orig_w=160)
    left, top, right, bottom = info["box"]
    crop_h, crop_w = bottom - top, right - left

    orig_patch = source[:, top:bottom, left:right, :]
    # Simulate 2x upscaled diffusion patch with a uniform brightness shift.
    up_h, up_w = crop_h * 2, crop_w * 2
    upscaled_crop = np.repeat(np.repeat(np.clip(orig_patch + 0.18, 0.0, 1.0), 2, axis=1), 2, axis=2)
    upscaled_mask = np.zeros((2, up_h, up_w), dtype=np.float32)
    upscaled_mask[:, 32:-32, 32:-32] = 1.0

    restored_no_cm, _ = THSUniversalMaskRestoreCrop().run(
        source,
        upscaled_crop,
        crop_info,
        mask=upscaled_mask,
        resize_method="bilinear",
        color_match="none",
    )
    restored_cm, _ = THSUniversalMaskRestoreCrop().run(
        source,
        upscaled_crop,
        crop_info,
        mask=upscaled_mask,
        resize_method="bilinear",
        color_match="mean_std",
    )

    assert restored_cm.shape == source.shape
    err_no_cm = float(np.mean(np.abs(restored_no_cm[:, 40:72, 48:80, :] - source[:, 40:72, 48:80, :])))
    err_cm = float(np.mean(np.abs(restored_cm[:, 40:72, 48:80, :] - source[:, 40:72, 48:80, :])))
    assert err_cm < err_no_cm * 0.25


def test_restore_crop_no_mask_edge_feather_border_aware():
    source = np.zeros((1, 120, 160, 3), dtype=np.float32)
    crop = np.ones((1, 60, 80, 3), dtype=np.float32)
    crop_info = json.dumps(
        {
            "mode": "auto",
            "box": [0, 0, 80, 60],
            "source_width": 160,
            "source_height": 120,
            "active_frames": [0, 1],
            "total_frames": 1,
        }
    )

    restored, applied = restore_crop_window(source, crop, crop_info, mask=None, edge_feather=8)
    # Left and top touch the frame border -> no feather decay on outer image edges.
    assert np.allclose(applied[0, :40, 0], 1.0, atol=1e-6)
    assert np.allclose(applied[0, 0, :60], 1.0, atol=1e-6)
    # Interior crop seams taper smoothly close to zero.
    assert float(applied[0, 20, 79]) < 0.05
    assert float(applied[0, 59, 30]) < 0.05
    # Center interior stays solid 1.0 and outside stays untouched 0.0.
    assert np.allclose(restored[0, 15:45, 15:65, :], 1.0, atol=1e-6)
    assert np.allclose(restored[0, 60:, :, :], 0.0, atol=1e-6)
    assert np.allclose(restored[0, :, 80:, :], 0.0, atol=1e-6)

    # Full-frame passthrough touches all four edges -> edge_feather keeps everything at 1.0.
    full_info = json.dumps(plan_crop_window((1, 120, 160), None, {"mode": "full"}))
    full_crop = np.full((1, 120, 160, 3), 0.75, dtype=np.float32)
    full_restored, full_applied = restore_crop_window(source, full_crop, full_info, mask=None, edge_feather=12)
    assert np.allclose(full_applied, 1.0, atol=1e-6)
    assert np.allclose(full_restored, 0.75, atol=1e-6)


def test_restore_crop_frame_broadcast_active_frames_and_partial_clip():
    source = np.zeros((5, 64, 64, 3), dtype=np.float32)
    info = {
        "mode": "auto",
        "box": [16, 16, 48, 48],
        "source_width": 64,
        "source_height": 64,
        "active_frames": [1, 4],
        "total_frames": 5,
    }

    # Single cropped frame broadcasts across active_frames [1, 4).
    single_crop = np.full((1, 32, 32, 3), 0.8, dtype=np.float32)
    out_single, mask_single = restore_crop_window(source, single_crop, info)
    assert np.allclose(out_single[0], 0.0)
    assert np.allclose(out_single[4], 0.0)
    assert np.allclose(out_single[1:4, 16:48, 16:48, :], 0.8, atol=1e-6)
    assert np.allclose(mask_single[0], 0.0) and np.allclose(mask_single[4], 0.0)

    # Shorter clip T_crop=2 updates only frames [1, 3), keeping [0, 3, 4] untouched.
    short_crop = np.full((2, 32, 32, 3), 0.5, dtype=np.float32)
    out_short, mask_short = restore_crop_window(source, short_crop, info)
    assert np.allclose(out_short[1:3, 16:48, 16:48, :], 0.5, atol=1e-6)
    assert np.allclose(out_short[3], 0.0)
    assert np.allclose(mask_short[3], 0.0)

    # Full-frame edit_mask (T=5) aligns by original frame index (1, 2, 3).
    full_edit_mask = np.zeros((5, 64, 64), dtype=np.float32)
    full_edit_mask[1, 20:28, 20:28] = 1.0
    full_edit_mask[2, 28:36, 28:36] = 1.0
    full_edit_mask[3, 36:44, 36:44] = 1.0
    crop_three = np.ones((3, 32, 32, 3), dtype=np.float32)
    out_aligned, mask_aligned = restore_crop_window(source, crop_three, info, mask=full_edit_mask)
    np.testing.assert_allclose(mask_aligned[1], full_edit_mask[1], atol=1e-6)
    np.testing.assert_allclose(mask_aligned[2], full_edit_mask[2], atol=1e-6)
    np.testing.assert_allclose(mask_aligned[3], full_edit_mask[3], atol=1e-6)
    np.testing.assert_allclose(out_aligned[1, :, :, 0], full_edit_mask[1], atol=1e-6)


def test_restore_crop_pil_fallback_and_validation(monkeypatch):
    monkeypatch.setattr(mask_core, "cv2", None)
    source = np.zeros((1, 80, 100, 3), dtype=np.float32)
    crop = np.full((1, 64, 64, 3), 0.6, dtype=np.float32)
    info = {
        "mode": "auto",
        "box": [18, 8, 50, 40],
        "source_width": 100,
        "source_height": 80,
        "active_frames": [0, 1],
        "total_frames": 1,
    }
    restored, applied = restore_crop_window(source, crop, info, resize_method="lanczos")
    assert restored.shape == source.shape
    assert np.allclose(restored[0, 8:40, 18:50, :], 0.6, atol=1e-3)
    assert np.allclose(applied[0, 8:40, 18:50], 1.0, atol=1e-6)

    for bad_info in ("{not-json", "[1, 2, 3]", '{"box": [0, 0, 10, 10]}'):
        with pytest.raises(MaskDataError):
            parse_crop_info(bad_info)

    with pytest.raises(MaskDataError, match="source_width"):
        restore_crop_window(
            source,
            crop,
            {**info, "source_width": 120},
        )
    with pytest.raises(MaskDataError, match="invalid"):
        parse_crop_info({**info, "box": [10, 10, 120, 40]}, orig_h=80, orig_w=100)
    with pytest.raises(MaskDataError, match="edge_feather"):
        restore_crop_window(source, crop, info, edge_feather=-1)
    with pytest.raises(MaskDataError, match="resize_method"):
        restore_crop_window(source, crop, info, resize_method="spline99")
    with pytest.raises(MaskDataError, match="color_match"):
        restore_crop_window(source, crop, info, color_match="histogram")
