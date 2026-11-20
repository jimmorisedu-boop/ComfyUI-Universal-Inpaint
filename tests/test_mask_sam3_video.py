import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ths_universal_inpaint.mask_core as core  # noqa: E402
import ths_universal_inpaint.mask_sam3_video as video  # noqa: E402


def _clip(frames=4, height=20, width=30):
    return np.zeros((frames, height, width, 3), dtype=np.float32)


def _doc(action="frame", token="", frame=1, direction="forward", local=False):
    events = [{
        "type": "rect", "mode": "add", "refine": True,
        "refine_method": "sam3", "x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3,
    }]
    if local:
        events.append({
            "type": "stroke", "mode": "erase", "radius": 0.08,
            "points": [[0.25, 0.25]],
        })
    return {
        "tracking_mode": "sam3",
        "sam3": {"action": action, "frame": frame, "direction": direction, "seed_token": token},
        "frames": {str(frame): {"events": events}},
    }


def test_frame_action_is_separate_and_never_calls_flow(monkeypatch):
    images = _clip()
    calls = []
    monkeypatch.setattr(core, "_flow_between", lambda *args: calls.append(True))
    monkeypatch.setattr(video, "_predict_native", lambda image, event: np.pad(
        np.ones((4, 5), dtype=bool), ((4, 12), (5, 20)), constant_values=False
    ))
    video.clear_sam3_video_cache()
    mask, parsed, anchors, edit_bases, meta = core.build_mask_sequence(
        images, _doc(), tracking_mode="sam3", grow=0, feather=0,
        return_edit_bases=True, return_sam3_meta=True,
    )
    assert meta["frame_ready"] is True and meta["track_ready"] is False
    assert meta["computed_frames"] == [1]
    assert anchors == {1}
    assert np.any(mask[1]) and not np.any(mask[0]) and not np.any(mask[2])
    assert not calls
    assert np.array_equal(mask, edit_bases)
    assert parsed["sam3"]["action"] == "frame"


def test_point_preview_preserves_prompt_and_tracks_only_on_explicit_action(monkeypatch):
    import copy
    images = _clip()
    selected = np.zeros(images.shape[1:3], dtype=bool)
    selected[4:12, 6:15] = True
    calls = []
    monkeypatch.setattr(video, "_predict_native", lambda image, event: (calls.append(event), selected.copy())[1])
    monkeypatch.setattr(core, "_flow_between", lambda *args: (_ for _ in ()).throw(AssertionError("Optical Flow must not run")))
    track_calls = []
    def track(images, prompts, *, seed, start, end, direction):
        track_calls.append(prompts)
        return {i: selected.copy() for i in range(start, end + 1)}
    monkeypatch.setattr(video, "_native_track", track)
    doc = {"tracking_mode": "sam3", "frames": {}, "sam3": {
        "action": "frame", "frame": 1, "direction": "both",
        "prompt": {"frame": 1, "points": [[.3, .3], [.8, .8]], "labels": [1, 0]},
    }}
    video.clear_sam3_video_cache()
    mask, parsed, anchors, bases, meta = core.build_mask_sequence(
        images, doc, tracking_mode="sam3", grow=0, feather=0,
        return_edit_bases=True, return_sam3_meta=True)
    assert parsed["sam3"]["prompt"] == doc["sam3"]["prompt"]
    assert calls[0]["points"] == [[.3, .3], [.8, .8]] and calls[0]["labels"] == [1, 0]
    assert not track_calls and meta["computed_frames"] == [1]
    assert meta["frame_ready"] and not meta["track_ready"]
    tracked = copy.deepcopy(doc)
    tracked["sam3"].update(action="track", seed_token=meta["seed_token"])
    tracked["frames"]["2"] = {"inherit": True, "events": [{"type": "stroke", "mode": "erase", "radius": .12, "points": [[.3,.3]]}]}
    result, _, _, _, tracked_meta = core.build_mask_sequence(
        images, tracked, tracking_mode="sam3", grow=0, feather=0,
        return_edit_bases=True, return_sam3_meta=True)
    assert len(track_calls) == 1 and tracked_meta["track_ready"]
    assert np.array_equal(result[1] > 0, selected)
    assert result[2].sum() < selected.sum(), "manual eraser must survive point-based tracking"


def test_object_previews_are_unioned_and_unchanged_objects_are_reused(monkeypatch):
    images = _clip()
    left = np.zeros(images.shape[1:3], dtype=bool)
    right = left.copy()
    left[3:14, 3:13] = True
    right[3:14, 18:26] = True
    calls = []

    def predict(_image, event):
        calls.append(event["labels"])
        if event["points"][0][0] > .5:
            return right
        result = left.copy()
        if 0 in event["labels"]:
            result[:, :7] = False
        return result

    video.clear_sam3_video_cache()
    monkeypatch.setattr(video, "_predict_native", predict)
    a = {"frame": 1, "points": [[.25, .3]], "labels": [1]}
    b = {"frame": 1, "points": [[.75, .3]], "labels": [1]}
    doc = {"sam3": {"frame": 1, "prompt": a}}
    first, _, _ = video.calculate_frame(images, core.parse_mask_data(doc, len(images)), frame=1)
    np.testing.assert_array_equal(first, left)
    doc["sam3"].pop("prompt")
    doc["sam3"]["objects"] = [a, b]
    group, seed, _ = video.calculate_frame(images, core.parse_mask_data(doc, len(images)), frame=1)
    np.testing.assert_array_equal(group, left | right)
    assert len(seed.objects) == 2 and calls == [[1], [1]]

    negative = {**a, "points": a["points"] + [[.15, .3]], "labels": [1, 0]}
    doc["sam3"]["objects"] = [negative, b]
    parsed = core.parse_mask_data(doc, len(images))
    refined, refined_seed, _ = video.calculate_frame(images, parsed, frame=1)
    assert calls == [[1], [1], [1, 0]]
    assert not refined[5, 4] and refined[5, 20]
    with pytest.raises(video.SAM3VideoError, match="current object prompt"):
        video._lookup_seed(images, parsed, 1, seed.token)
    assert video._lookup_seed(images, parsed, 1, refined_seed.token) is refined_seed
    assert [p["object_id"] for p in video._prompt_events(parsed, 1)] == [1, 2]


def test_track_requires_fresh_seed_and_uses_native_only(monkeypatch):
    images = _clip()
    monkeypatch.setattr(video, "_predict_native", lambda image, event: np.pad(
        np.ones((4, 5), dtype=bool), ((4, 12), (5, 20)), constant_values=False
    ))
    video.clear_sam3_video_cache()
    _frame_mask, _parsed, _anchors, _bases, frame_meta = core.build_mask_sequence(
        images, _doc(), tracking_mode="sam3", grow=0, feather=0,
        return_edit_bases=True, return_sam3_meta=True,
    )
    token = frame_meta["seed_token"]
    native_calls = []
    monkeypatch.setattr(core, "_flow_between", lambda *args: (_ for _ in ()).throw(AssertionError("flow")))

    def fake_track(images, prompts, *, seed, start, end, direction):
        native_calls.append((start, end, direction, seed.object_id))
        base = np.zeros(images.shape[1:3], dtype=bool)
        base[4:12, 6:15] = True
        return {frame: base.copy() for frame in range(start, end + 1)}

    monkeypatch.setattr(video, "_native_track", fake_track)
    tracked_doc = _doc(action="track", token=token, direction="both")
    mask, _parsed, _anchors, _bases, meta = core.build_mask_sequence(
        images, tracked_doc, tracking_mode="sam3", grow=0, feather=0,
        return_edit_bases=True, return_sam3_meta=True,
    )
    assert meta["track_ready"] is True
    assert meta["computed_frames"] == [0, 1, 2, 3]
    assert np.all(mask.sum(axis=(1, 2)) > 0)
    assert native_calls == [(0, 3, "both", 1)]

    stale = _doc(action="track", token="sam3v1-stale", direction="forward")
    with pytest.raises(video.SAM3VideoError, match="stale|unknown"):
        core.build_mask_sequence(images, stale, tracking_mode="sam3", grow=0, feather=0)


def test_raw_track_cache_survives_grow_feather_and_local_brush_edits(monkeypatch):
    images = _clip()
    monkeypatch.setattr(video, "_predict_native", lambda image, event: np.pad(
        np.ones((4, 5), dtype=bool), ((4, 12), (5, 20)), constant_values=False
    ))
    video.clear_sam3_video_cache()
    _mask, _parsed, _anchors, _bases, frame_meta = core.build_mask_sequence(
        images, _doc(), tracking_mode="sam3", grow=0, feather=0,
        return_edit_bases=True, return_sam3_meta=True,
    )
    calls = []

    def fake_track(images, prompts, *, seed, start, end, direction):
        calls.append(1)
        base = np.zeros(images.shape[1:3], dtype=bool)
        base[4:12, 6:15] = True
        return {frame: base.copy() for frame in range(start, end + 1)}

    monkeypatch.setattr(video, "_native_track", fake_track)
    first_doc = _doc(action="track", token=frame_meta["seed_token"])
    first, _p, _a, _b, _m = core.build_mask_sequence(
        images, first_doc, tracking_mode="sam3", grow=0, feather=0,
        return_edit_bases=True, return_sam3_meta=True,
    )
    second_doc = _doc(action="track", token=frame_meta["seed_token"], local=True)
    second, _p, _a, _b, _m = core.build_mask_sequence(
        images, second_doc, tracking_mode="sam3", grow=3, feather=2,
        return_edit_bases=True, return_sam3_meta=True,
    )
    assert len(calls) == 1
    assert np.any(first != second)


def test_track_rejects_lost_object_identity_and_missing_requested_frames(monkeypatch):
    images = _clip(frames=3)
    monkeypatch.setattr(video, "_predict_native", lambda image, event: np.pad(
        np.ones((4, 5), dtype=bool), ((4, 12), (5, 20)), constant_values=False
    ))
    video.clear_sam3_video_cache()
    _mask, _parsed, _anchors, _bases, frame_meta = core.build_mask_sequence(
        images, _doc(frame=0), tracking_mode="sam3", grow=0, feather=0,
        return_edit_bases=True, return_sam3_meta=True,
    )
    token = frame_meta["seed_token"]
    # An explicit native obj_ids array without id 1 must not select another
    # detected object by position.
    def wrong_object(images, prompts, *, seed, start, end, direction):
        return {index: np.ones(images.shape[1:3], dtype=bool) for index in range(start, end + 1)}

    monkeypatch.setattr(video, "_native_track", wrong_object)
    # The low-level selector is the identity gate used by native responses.
    assert video._select_object_mask(
        {"out_binary_masks": np.ones((2, 20, 30), dtype=bool), "out_obj_ids": np.array([7, 8])},
        1,
        (20, 30),
    ) is None

    monkeypatch.setattr(video, "_native_track", lambda *args, **kwargs: {
        kwargs["start"]: np.ones(images.shape[1:3], dtype=bool),
    })
    with pytest.raises(video.SAM3VideoError, match="incomplete"):
        core.build_mask_sequence(
            images,
            _doc(action="track", token=token, frame=0),
            tracking_mode="sam3", grow=0, feather=0,
        )
