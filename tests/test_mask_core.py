import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ths_universal_inpaint.mask_core import (  # noqa: E402
    MaskDataError,
    MaskRefinementError,
    MaskSourceMismatch,
    _flow_between,
    build_mask_sequence,
    cache_preview_assets,
    canonical_mask_data,
    frame_fingerprint,
    parse_mask_data,
    rasterize_events,
    refine_rectangle,
)


def _clip(frames=4, height=48, width=64):
    return np.zeros((frames, height, width, 3), dtype=np.float32)


def test_static_rectangle_and_ordered_erase_are_not_full_frame():
    images = _clip(frames=2)
    data = {
        "frames": {
            "0": {
                "events": [
                    {"type": "rect", "mode": "add", "x": 0.20, "y": 0.20, "w": 0.35, "h": 0.35},
                    {"type": "stroke", "mode": "erase", "radius": 0.055, "points": [[0.36, 0.36]]},
                ]
            }
        }
    }
    mask, _, anchors = build_mask_sequence(images, data, tracking_mode="static", grow=0, feather=0)
    assert anchors == {0}
    assert 0 < int(mask[0].sum()) < images.shape[1] * images.shape[2]
    # The erase follows the rectangle and therefore removes its centre.
    assert mask[0, 17, 23] == 0
    assert mask[0, 12, 14] == 1
    # Static mode intentionally repeats the finite area, never a full-frame fill.
    assert np.array_equal(mask[0], mask[1])


def test_polygon_and_rectangle_edits_preserve_cross_tool_order():
    images = _clip(frames=1, height=32, width=32)
    data = {
        "frames": {"0": {"events": [
            {"type": "rect", "mode": "add", "x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8},
            {"type": "stroke", "mode": "erase", "radius": 0.12, "points": [[0.5, 0.5]]},
            {"type": "polygon", "mode": "add", "points": [[0.45, 0.45], [0.55, 0.45], [0.5, 0.58]]},
        ]}}
    }
    mask, _, _ = build_mask_sequence(images, data, tracking_mode="static", grow=0, feather=0)
    # Polygon re-adds a small triangle inside the erased centre.
    assert mask[0, 16, 16] == 1
    assert mask[0, 13, 16] == 0


def test_translated_patch_is_tracked_forward_with_optical_flow():
    frames, height, width = 5, 64, 80
    images = _clip(frames, height, width)
    for frame in range(frames):
        images[frame, 22:38, 12 + frame * 4:28 + frame * 4, 0] = 1.0
        images[frame, 22:38, 12 + frame * 4:28 + frame * 4, 1] = 0.8
    data = {"frames": {"0": {"events": [{
        "type": "rect", "mode": "add", "x": 12 / width, "y": 22 / height,
        "w": 16 / width, "h": 16 / height,
    }]}}}
    mask, _, _ = build_mask_sequence(images, data, tracking_mode="optical_flow", grow=0, feather=0)
    centers = [np.argwhere(mask[frame] > 0.5).mean(axis=0) for frame in range(frames)]
    assert centers[0][1] < centers[-1][1]
    assert centers[-1][1] - centers[0][1] > 8
    assert abs(centers[-1][0] - centers[0][0]) < 4


def test_optical_flow_preserves_opposing_local_motion_instead_of_global_fallback():
    """A constant phase-correlation shift must not replace local Farneback flow."""
    height, width = 96, 128
    rng = np.random.default_rng(4)
    yy, xx = np.indices((height, width), dtype=np.float32)
    background = np.empty((height, width, 3), dtype=np.float32)
    gradient = (xx + yy) / 256.0
    background[..., 0] = 0.20 + 0.10 * gradient
    background[..., 1] = 0.30 + 0.10 * gradient
    background[..., 2] = 0.25 + 0.05 * gradient
    texture = rng.random((24, 24, 3), dtype=np.float32)
    images = np.stack([background.copy(), background.copy()])
    # The left patch moves right while the right patch moves left.  Their
    # opposite motions make a single global phase shift observably wrong.
    images[0, 30:54, 20:44] = texture
    images[0, 30:54, 84:108] = 1.0 - texture
    images[1, 30:54, 28:52] = texture
    images[1, 30:54, 76:100] = 1.0 - texture

    tracked = _flow_between(images[0], images[1])
    assert tracked is not None
    flow = tracked[0] if isinstance(tracked, tuple) else tracked
    left = flow[33:51, 23:41]
    right = flow[33:51, 79:97]
    left_median = np.median(left.reshape(-1, 2), axis=0)
    right_median = np.median(right.reshape(-1, 2), axis=0)
    assert left_median[0] > 4.0
    assert right_median[0] < -4.0
    assert abs(float(left_median[1])) < 1.0
    assert abs(float(right_median[1])) < 1.0


def test_optical_flow_sweeps_backward_from_late_anchor_without_gaps():
    frames, height, width = 5, 64, 80
    images = _clip(frames, height, width)
    for frame in range(frames):
        images[frame, 22:38, 12 + frame * 4:28 + frame * 4, :2] = 1.0
    data = {"frames": {"4": {"events": [{
        "type": "rect", "mode": "add", "x": 28 / width, "y": 22 / height,
        "w": 16 / width, "h": 16 / height,
    }]}}}
    mask, _, _ = build_mask_sequence(images, data, tracking_mode="optical_flow", grow=0, feather=0)
    assert np.all(mask.sum(axis=(1, 2)) > 0)
    centers = [np.argwhere(mask[frame] > 0.5).mean(axis=0) for frame in range(frames)]
    assert centers[0][1] < centers[-1][1]


def test_optical_flow_two_anchors_cover_every_interval_frame():
    frames, height, width = 7, 64, 80
    images = _clip(frames, height, width)
    for frame in range(frames):
        images[frame, 22:38, 12 + frame * 3:28 + frame * 3, :2] = 1.0
    data = {"frames": {
        "1": {"events": [{"type": "rect", "mode": "add", "x": 15 / width, "y": 22 / height, "w": 16 / width, "h": 16 / height}]},
        "5": {"events": [{"type": "rect", "mode": "add", "x": 27 / width, "y": 22 / height, "w": 16 / width, "h": 16 / height}]},
    }}
    mask, _, _ = build_mask_sequence(images, data, tracking_mode="keyframes", grow=0, feather=0,
                                     start_frame=1, end_frame=5)
    assert np.all(mask[1:6].sum(axis=(1, 2)) > 0)


def test_keyframe_anchor_is_exact_and_range_is_zero_outside():
    images = _clip(frames=6, height=40, width=50)
    data = {"frames": {
        "1": {"events": [{"type": "rect", "mode": "add", "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}]},
        "4": {"events": [{"type": "rect", "mode": "add", "x": 0.6, "y": 0.6, "w": 0.2, "h": 0.2}]},
    }}
    mask, _, anchors = build_mask_sequence(images, data, tracking_mode="keyframes", grow=0, feather=0,
                                            start_frame=1, end_frame=4)
    assert anchors == {1, 4}
    assert mask[0].sum() == 0 and mask[5].sum() == 0
    # A correction frame is kept pixel-for-pixel, not replaced by a propagated mask.
    expected_first = rasterize_events(data["frames"]["1"]["events"], 40, 50)
    expected_last = rasterize_events(data["frames"]["4"]["events"], 40, 50)
    assert np.array_equal(mask[1], expected_first)
    assert np.array_equal(mask[4], expected_last)


def test_invalid_coordinates_and_source_fingerprint_are_rejected():
    with pytest.raises(MaskDataError, match="outside"):
        parse_mask_data({"frames": {"0": {"events": [{"type": "rect", "x": 0.9, "y": 0, "w": 0.2, "h": 0.2}]}}}, 1)
    images = _clip(frames=2)
    document = {"source": {"frames": 2, "width": 64, "height": 48, "fingerprint": frame_fingerprint(images)}, "frames": {}}
    parse_mask_data(document, 2, width=64, height=48, fingerprint=frame_fingerprint(images))
    changed = images.copy(); changed[0, 0, 0, 0] = 1.0
    with pytest.raises(MaskDataError, match="different source clip"):
        parse_mask_data(document, 2, width=64, height=48, fingerprint=frame_fingerprint(changed))


@pytest.mark.parametrize(
    ("reason", "current"),
    [
        ("frames", {"frame_count": 3, "width": 64, "height": 48, "fingerprint": "source-a"}),
        ("width", {"frame_count": 2, "width": 65, "height": 48, "fingerprint": "source-a"}),
        ("height", {"frame_count": 2, "width": 64, "height": 49, "fingerprint": "source-a"}),
        ("fingerprint", {"frame_count": 2, "width": 64, "height": 48, "fingerprint": "source-b"}),
    ],
)
def test_source_mismatch_is_typed_and_preserves_sanitized_previous_source(reason, current):
    document = {"source": current, "frames": {}}
    with pytest.raises(MaskSourceMismatch) as exc_info:
        parse_mask_data(document, 2, width=64, height=48, fingerprint="source-a")
    error = exc_info.value
    assert error.reason == reason
    assert error.previous_source == {
        "frames": 3 if reason == "frames" else 2,
        "width": current["width"],
        "height": current["height"],
        "fingerprint": current["fingerprint"],
    }


def test_same_source_remains_valid_and_invalid_json_is_not_swallowed_as_mismatch():
    document = {
        "source": {"frames": 2, "width": 64, "height": 48, "fingerprint": "source-a"},
        "frames": {},
    }
    parsed = parse_mask_data(document, 2, width=64, height=48, fingerprint="source-a")
    assert parsed["source"]["frames"] == 2
    with pytest.raises(MaskDataError) as exc_info:
        parse_mask_data("{not-json", 2, width=64, height=48, fingerprint="source-a")
    assert type(exc_info.value) is MaskDataError


def _refinement_fixture():
    import cv2

    height, width = 128, 160
    image = np.zeros((height, width, 3), dtype=np.float32)
    cv2.ellipse(image, (80, 64), (30, 40), 0, 0, 360, (0.8, 0.3, 0.1), -1)
    event = {
        "type": "rect", "mode": "add", "refine": True,
        "x": 0.20, "y": 0.125, "w": 0.60, "h": 0.75,
    }
    return image, event


def test_refine_rectangle_segments_object_and_never_paints_outside_selection():
    pytest.importorskip("cv2")
    image, event = _refinement_fixture()
    refined = refine_rectangle(image, event)
    assert refined is not None
    assert refined.shape == image.shape[:2]
    x0, y0 = int(np.floor(event["x"] * (image.shape[1] - 1))), int(np.floor(event["y"] * (image.shape[0] - 1)))
    x1 = int(np.ceil((event["x"] + event["w"]) * (image.shape[1] - 1)))
    y1 = int(np.ceil((event["y"] + event["h"]) * (image.shape[0] - 1)))
    outside = refined.copy()
    outside[y0:y1 + 1, x0:x1 + 1] = False
    assert not np.any(outside)
    selected_area = (x1 - x0 + 1) * (y1 - y0 + 1)
    assert 0 < int(refined.sum()) < selected_area
    rasterized = rasterize_events([event], image.shape[0], image.shape[1], image=image)
    assert np.array_equal(rasterized.astype(bool), refined)


def test_refinement_failure_is_explicit_for_full_frame_or_missing_opencv(monkeypatch):
    image, event = _refinement_fixture()
    full_frame = dict(event, x=0.0, y=0.0, w=1.0, h=1.0)
    with pytest.raises(MaskRefinementError, match="unavailable"):
        rasterize_events([full_frame], image.shape[0], image.shape[1], image=image)
    import ths_universal_inpaint.mask_core as core
    monkeypatch.setattr(core, "cv2", None)
    assert refine_rectangle(image, event) is None
    with pytest.raises(MaskRefinementError, match="unavailable"):
        rasterize_events([event], image.shape[0], image.shape[1], image=image)


def test_canonical_serialization_is_ordered_and_bounded():
    raw = {"frames": {"0": {"events": [
        {"type": "rect", "mode": "add", "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
        {"type": "stroke", "mode": "erase", "radius": 0.02, "points": [[0.2, 0.2]]},
    ]}}}
    encoded = canonical_mask_data(raw, 1, width=10, height=10)
    assert '"events"' in encoded and '"rect"' in encoded and '"stroke"' in encoded
    assert encoded.index('"rect"') < encoded.index('"stroke"')


def test_static_mask_edge_controls_apply_identically_on_the_anchor_frame():
    images = _clip(frames=3, height=40, width=50)
    data = {"frames": {"0": {"events": [
        {"type": "rect", "mode": "add", "x": 0.3, "y": 0.3, "w": 0.2, "h": 0.2}
    ]}}}
    mask, _, anchors = build_mask_sequence(
        images, data, tracking_mode="static", grow=3, feather=2
    )
    assert anchors == {0}
    assert np.array_equal(mask[0], mask[1])
    assert np.array_equal(mask[1], mask[2])
    assert np.any((mask[0] > 0) & (mask[0] < 1))
    assert mask[0, 16, 12] > 0  # Outside the painted rectangle, inside its grown edge.


def _inherit_document():
    return {
        "frames": {
            "0": {"events": [
                {"type": "rect", "mode": "add", "x": 0.15, "y": 0.15, "w": 0.55, "h": 0.55},
            ]},
            "2": {"inherit": True, "events": [
                {"type": "rect", "mode": "erase", "x": 0.15, "y": 0.15, "w": 0.15, "h": 0.15},
                {"type": "rect", "mode": "add", "x": 0.70, "y": 0.70, "w": 0.12, "h": 0.12},
            ]},
        }
    }


def test_inherit_is_canonical_and_must_be_boolean():
    parsed = parse_mask_data(_inherit_document(), 4)
    assert parsed["frames"]["2"]["inherit"] is True
    assert '"inherit":true' in canonical_mask_data(_inherit_document(), 4)

    for invalid in ("true", 1, 0, None, [], {}):
        document = {"frames": {"0": {"inherit": invalid, "events": []}}}
        with pytest.raises(MaskDataError, match="inherit must be a boolean"):
            parse_mask_data(document, 1)


def test_inherited_erase_keeps_the_propagated_mask_and_becomes_anchor():
    images = _clip()
    mask, _parsed, anchors = build_mask_sequence(
        images, _inherit_document(), tracking_mode="keyframes", grow=0, feather=0
    )
    assert anchors == {0, 2}
    # The independently painted anchor is present outside the inherited erase.
    assert mask[2, 6, 6] == 0
    assert mask[2, 18, 18] == 1
    # A frame after the correction inherits the corrected anchor rather than
    # reverting to the pre-correction rectangle.
    assert mask[3, 6, 6] == 0
    assert mask[3, 18, 18] == 1
    assert mask[3, 25, 25] == 1


def test_reloaded_anchor_list_does_not_erase_inherited_baseline():
    images = _clip()
    document = _inherit_document()
    expected, _, _ = build_mask_sequence(images, document, grow=0, feather=0)
    document["anchors"] = [0, 2]
    actual, parsed, _ = build_mask_sequence(images, document, grow=0, feather=0)
    assert np.array_equal(actual, expected)
    from ths_universal_inpaint.mask_core import derive_edit_base_sequence
    bases = derive_edit_base_sequence(images, parsed)
    assert bases[2, 18, 18] == 1
    assert bases[0].sum() == 0


def test_empty_independent_key_survives_roundtrip_and_stops_mask():
    document = _inherit_document()
    document["frames"]["2"] = {"events": [], "inherit": False}
    canonical = canonical_mask_data(document, 4)
    mask, parsed, anchors = build_mask_sequence(_clip(), canonical, grow=0, feather=0)
    assert parsed["frames"]["2"] == {"events": [], "inherit": False}
    assert anchors == {0, 2}
    assert mask[0].sum() > 0
    assert mask[2:].sum() == 0


def test_edit_preview_reuses_tracking_pass(monkeypatch):
    import ths_universal_inpaint.mask_core as core
    calls = []
    monkeypatch.setattr(core, "_flow_between", lambda a, b: calls.append(1))
    document = _inherit_document()
    del document["frames"]["2"]
    mask, _, _, bases = build_mask_sequence(_clip(), document, grow=0, feather=0, return_edit_bases=True)
    assert len(calls) == 3  # one adjacent-frame pass, also reused by edit preview
    assert bases[0].sum() == 0
    assert np.array_equal(bases[2], mask[2])


def test_cache_eviction_preserves_unowned_paths(tmp_path, monkeypatch):
    import ths_universal_inpaint.mask_core as core
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(get_temp_directory=lambda: str(tmp_path)))
    monkeypatch.setattr(core, "MAX_TEMP_CACHE_SESSIONS", 1)
    unrelated = tmp_path / core.TEMP_CACHE_ROOT / "session-user-backup"
    unrelated.mkdir(parents=True)
    (unrelated / "keep.txt").write_text("user data", encoding="utf-8")
    images = _clip(frames=1, height=8, width=8)
    masks = np.zeros((1, 8, 8), dtype=np.float32)
    old, _ = cache_preview_assets(images, masks)
    new, _ = cache_preview_assets(images, masks)
    assert not (tmp_path / old[0]["subfolder"]).exists()
    assert (tmp_path / new[0]["subfolder"]).is_dir()
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_static_mode_keeps_legacy_zero_based_event_order_with_inherit_flag():
    images = _clip(frames=2)
    document = {
        "frames": {
            "0": {"events": [{"type": "rect", "mode": "add", "x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6}]},
            "1": {"inherit": True, "events": [{"type": "rect", "mode": "erase", "x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}]},
        }
    }
    mask, _parsed, _anchors = build_mask_sequence(
        images, document, tracking_mode="static", grow=0, feather=0
    )
    expected = rasterize_events(
        [*document["frames"]["0"]["events"], *document["frames"]["1"]["events"]],
        images.shape[1],
        images.shape[2],
    )
    assert np.array_equal(mask[0], expected)
    assert np.array_equal(mask[1], expected)


def test_build_mask_sequence_can_return_pre_postprocess_raw_mask():
    images = _clip(frames=2, height=32, width=32)
    document = {
        "frames": {
            "0": {"events": [{"type": "rect", "mode": "add", "x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}]}
        }
    }
    mask, _parsed, _anchors, raw_mask = build_mask_sequence(
        images,
        document,
        tracking_mode="static",
        grow=4,
        feather=3,
        return_raw_mask=True,
    )
    assert raw_mask.shape == mask.shape
    assert float(raw_mask[0, 6, 16]) == 0.0
    assert float(mask[0, 6, 16]) > 0.0


def test_all_frame_cache_uses_temp_descriptors_and_isolates_sessions(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(
        get_temp_directory=lambda: str(tmp_path),
    ))
    images = _clip(frames=3, height=40, width=60)
    masks = np.zeros((3, 40, 60), dtype=np.float32)
    edit_bases = np.zeros_like(masks)
    raw_masks = np.zeros_like(masks)
    descriptors, info = cache_preview_assets(
        images, masks, edit_bases=edit_bases, raw_masks=raw_masks, source_session="source-A"
    )
    assert info["available"] is True
    assert len(descriptors) == 3
    first = descriptors[0]
    assert first["type"] == "temp"
    assert first["mask"]["type"] == "temp"
    assert first["raw_mask"]["type"] == "temp"
    assert first["edit_base"]["type"] == "temp"
    assert "data_url" not in first
    assert first["width"] <= 640 and first["height"] <= 640
    for item in descriptors:
        for descriptor in (item, item["mask"], item["raw_mask"], item["edit_base"]):
            assert (tmp_path / descriptor["subfolder"] / descriptor["filename"]).is_file()

    second, second_info = cache_preview_assets(
        images, masks, edit_bases=edit_bases, source_session="source-A"
    )
    assert second_info["available"] is True
    assert second_info["session"] != info["session"]
    assert second[0]["subfolder"] != first["subfolder"]


def test_parse_mask_data_layers_validation_and_v1_compat():
    from ths_universal_inpaint.mask_core import MAX_MASK_LAYERS

    legacy = parse_mask_data({"frames": {"0": {"events": [{"type": "rect", "mode": "add", "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}]}}})
    assert "layers" not in legacy and "active_layer_id" not in legacy

    multi = parse_mask_data(
        {
            "layers": [
                {
                    "id": "layer-1",
                    "name": "SAM3",
                    "type": "sam3",
                    "blend": "add",
                    "enabled": True,
                    "start_frame": 0,
                    "end_frame": 3,
                    "sam3": {"action": "frame", "frame": 1, "direction": "both", "prompt": {"frame": 1, "points": [[0.2, 0.2]], "labels": [1]}},
                    "frames": {},
                },
                {
                    "id": "layer-2",
                    "name": "Статика",
                    "type": "static",
                    "blend": "subtract",
                    "enabled": True,
                    "start_frame": 1,
                    "end_frame": -1,
                    "frames": {"0": {"events": [{"type": "rect", "mode": "add", "x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2}]}},
                },
            ],
            "active_layer_id": "layer-2",
        },
        frame_count=4,
    )
    assert len(multi["layers"]) == 2
    assert multi["active_layer_id"] == "layer-2"
    assert multi["tracking_mode"] == "static"
    assert multi["layers"][0]["type"] == "sam3"
    assert multi["layers"][1]["blend"] == "subtract"

    # Defaults active_layer_id to first layer when omitted.
    default_active = parse_mask_data({"layers": [{"id": "a", "type": "static", "frames": {}}]})
    assert default_active["active_layer_id"] == "a"

    # Validation failures.
    for bad in (
        {"layers": []},
        {"layers": [{"id": f"l-{i}", "type": "static", "frames": {}} for i in range(MAX_MASK_LAYERS + 1)]},
        {"layers": [{"id": "dup", "type": "static", "frames": {}}, {"id": "dup", "type": "static", "frames": {}}]},
        {"layers": [{"id": "a", "type": "unknown", "frames": {}}]},
        {"layers": [{"id": "a", "type": "static", "blend": "xor", "frames": {}}]},
        {"layers": [{"id": "a", "type": "static", "frames": {}}], "active_layer_id": "missing"},
    ):
        with pytest.raises(MaskDataError):
            parse_mask_data(bad)


def test_build_mask_sequence_multilayer_compositing_and_independent_cache(monkeypatch):
    import ths_universal_inpaint.mask_core as core
    import ths_universal_inpaint.mask_sam3_video as video

    core.clear_raw_tracking_cache()
    video.clear_sam3_video_cache()
    images = _clip(frames=4, height=40, width=60)
    track_calls = []

    def fake_predict(image, event):
        out = np.zeros(images.shape[1:3], dtype=bool)
        out[5:20, 5:25] = True
        return out

    def fake_track(imgs, prompts, *, seed, start, end, direction):
        track_calls.append((start, end, direction))
        out = np.zeros(imgs.shape[1:3], dtype=bool)
        out[5:20, 5:25] = True
        return {idx: out.copy() for idx in range(start, end + 1)}

    monkeypatch.setattr(video, "_predict_native", fake_predict)
    monkeypatch.setattr(video, "_native_track", fake_track)

    sam3_seed_doc = {
        "tracking_mode": "sam3",
        "sam3": {"action": "frame", "frame": 0, "direction": "both", "prompt": {"frame": 0, "points": [[0.2, 0.2]], "labels": [1]}},
        "frames": {},
    }
    _, _, _, _, seed_meta = core.build_mask_sequence(
        images, sam3_seed_doc, tracking_mode="sam3", grow=0, feather=0, return_edit_bases=True, return_sam3_meta=True
    )
    token = seed_meta["seed_token"]

    # Pass 1: SAM3 (+) union Static (+), across distinct ranges.
    doc_union = {
        "layers": [
            {
                "id": "l-sam3",
                "type": "sam3",
                "blend": "add",
                "enabled": True,
                "start_frame": 0,
                "end_frame": 3,
                "sam3": {"action": "track", "frame": 0, "direction": "both", "seed_token": token, "prompt": {"frame": 0, "points": [[0.2, 0.2]], "labels": [1]}},
                "frames": {},
            },
            {
                "id": "l-static",
                "type": "static",
                "blend": "add",
                "enabled": True,
                "start_frame": 2,
                "end_frame": 3,
                "frames": {"0": {"events": [{"type": "rect", "mode": "add", "x": 0.6, "y": 0.5, "w": 0.25, "h": 0.3}]}},
            },
        ],
        "active_layer_id": "l-static",
    }
    union_mask, _, _, _, _, _, layers_meta = core.build_mask_sequence(
        images, doc_union, tracking_mode="static", grow=0, feather=0, return_edit_bases=True, return_sam3_meta=True, return_raw_mask=True, return_layers_meta=True
    )
    assert len(track_calls) == 1
    assert len(layers_meta) == 2 and all(item["has_mask"] for item in layers_meta)
    assert float(union_mask[0, 10, 10]) == 1.0  # SAM3 active on frame 0
    assert float(union_mask[0, 25, 42]) == 0.0  # Static inactive on frame 0 (start_frame=2)
    assert float(union_mask[2, 25, 42]) == 1.0  # Static active on frame 2

    # Pass 2: Add Static subtract layer carving into SAM3 region without re-running native SAM3 track.
    before_hits = core.raw_tracking_cache_info()["hits"]
    doc_sub = {
        "layers": [
            doc_union["layers"][0],
            doc_union["layers"][1],
            {
                "id": "l-cut",
                "type": "static",
                "blend": "subtract",
                "enabled": True,
                "start_frame": 0,
                "end_frame": 3,
                "frames": {"0": {"events": [{"type": "rect", "mode": "add", "x": 10 / 59, "y": 8 / 39, "w": 8 / 59, "h": 8 / 39}]}},
            },
        ],
        "active_layer_id": "l-cut",
    }
    sub_mask, _, _ = core.build_mask_sequence(images, doc_sub, tracking_mode="static", grow=0, feather=0)
    assert len(track_calls) == 1, "SAM3 video cache reused without a second native track run"
    assert core.raw_tracking_cache_info()["hits"] > before_hits, "l-static layer hits per-layer tracking cache"
    assert float(sub_mask[1, 12, 14]) == 0.0, "subtract layer cuts hole inside SAM3 region"
    assert float(sub_mask[1, 6, 6]) == 1.0, "uncut SAM3 pixel stays masked"

    # Intersect + disabled layer + single postprocess pass after composite.
    doc_intersect = {
        "layers": [
            {
                "id": "l-base",
                "type": "static",
                "blend": "add",
                "enabled": True,
                "frames": {"0": {"events": [{"type": "rect", "mode": "add", "x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6}]}},
            },
            {
                "id": "l-gate",
                "type": "keyframes",
                "blend": "intersect",
                "enabled": True,
                "frames": {"0": {"events": [{"type": "rect", "mode": "add", "x": 0.3, "y": 0.3, "w": 0.5, "h": 0.5}]}},
            },
            {
                "id": "l-off",
                "type": "static",
                "blend": "add",
                "enabled": False,
                "frames": {"0": {"events": [{"type": "rect", "mode": "add", "x": 0.85, "y": 0.05, "w": 0.1, "h": 0.1}]}},
            },
        ],
        "active_layer_id": "l-gate",
    }
    inter_mask, _, _ = core.build_mask_sequence(images, doc_intersect, tracking_mode="keyframes", grow=0, feather=0)
    assert float(inter_mask[0, 6, 8]) == 0.0, "outside intersection is zero"
    assert float(inter_mask[0, 20, 25]) == 1.0, "inside intersection is one"
    assert float(inter_mask[0, 3, 55]) == 0.0, "disabled layer does not contribute"

    grown_mask, _, _, grown_raw = core.build_mask_sequence(
        images, doc_intersect, tracking_mode="keyframes", grow=3, feather=2, return_raw_mask=True
    )
    assert float(grown_raw[0, 10, 25]) == 0.0
    assert float(grown_mask[0, 10, 25]) > 0.0, "postprocess grow/feather applies to combined composite"


def test_encode_preview_assets_optional_edit_base_and_raw_mask():
    from ths_universal_inpaint.mask_core import encode_preview_assets

    images = np.zeros((3, 16, 16, 3), dtype=np.float32)
    masks = np.zeros((3, 16, 16), dtype=np.float32)
    edit_bases = np.zeros((3, 16, 16), dtype=np.float32)
    raw_masks = np.zeros((3, 16, 16), dtype=np.float32)

    compact = encode_preview_assets(images, masks)
    assert len(compact) == 3
    assert "edit_base" not in compact[0] and "raw_mask" not in compact[0]

    fallback = encode_preview_assets(images, masks, edit_bases=edit_bases, raw_masks=raw_masks)
    assert len(fallback) == 3
    assert fallback[0]["edit_base"].startswith("data:image/png;base64,")
    assert fallback[0]["raw_mask"].startswith("data:image/png;base64,")

