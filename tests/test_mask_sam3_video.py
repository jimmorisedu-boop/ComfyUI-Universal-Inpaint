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


def test_native_occlusion_is_a_computed_empty_frame():
    shape = (20, 30)
    empty = {"obj_ids": np.array([], dtype=int), "out_binary_masks": np.empty((0, *shape), bool)}
    other = {"obj_ids": np.array([7]), "out_binary_masks": np.ones((1, *shape), bool)}
    for response in (empty, other):
        mask = video._tracked_response_mask(response, [1], shape)
        assert mask.shape == shape and not mask.any()
    assert video._tracked_response_mask(other, [1, 7], shape).all()
    assert video._tracked_response_mask({"obj_ids": np.array([])}, [1], shape) is None
    assert video._tracked_response_mask({**empty, "obj_ids": np.array([1])}, [1], shape) is None


def test_multiframe_objects_and_cross_frame_refinements(monkeypatch):
    images = _clip(frames=5)
    m_obj1 = np.zeros(images.shape[1:3], dtype=bool)
    m_obj2 = m_obj1.copy()
    m_obj1[2:8, 2:10] = True
    m_obj2[10:16, 15:24] = True
    predict_frames = []

    def predict(_image, event):
        f = int(event["frame"])
        predict_frames.append((f, list(event.get("labels", []))))
        return m_obj2.copy() if f == 2 else m_obj1.copy()

    video.clear_sam3_video_cache()
    monkeypatch.setattr(video, "_predict_native", predict)

    # 1. Object 1 anchored on frame 0
    obj1 = {"frame": 0, "points": [[0.2, 0.2]], "labels": [1]}
    parsed1 = core.parse_mask_data({"sam3": {"frame": 0, "prompt": obj1}}, len(images))
    _, seed1, meta1 = video.calculate_frame(images, parsed1, frame=0)
    assert meta1["computed_frames"] == [0]
    assert predict_frames == [(0, [1])]

    # 2. Object 2 added on frame 2 reuses cached Object 1 without re-predicting frame 0
    obj2 = {"frame": 2, "points": [[0.65, 0.65]], "labels": [1]}
    doc2 = {"sam3": {"frame": 2, "objects": [obj1, obj2]}}
    parsed2 = core.parse_mask_data(doc2, len(images))
    mask2, seed2, meta2 = video.calculate_frame(images, parsed2, frame=2)
    assert predict_frames == [(0, [1]), (2, [1])]
    assert seed2.object_frames == (0, 2)
    assert meta2["computed_frames"] == [0, 2]
    assert np.array_equal(mask2, m_obj2)

    # 3. Cross-frame negative-only refinement on frame 3 keeps existing mask & reuses caches
    obj1_refined = {
        **obj1,
        "refinements": [{"frame": 3, "points": [[0.25, 0.25]], "labels": [0]}],
    }
    doc3 = {"sam3": {"frame": 3, "objects": [obj1_refined, obj2]}}
    parsed3 = core.parse_mask_data(doc3, len(images))
    mask3, seed3, meta3 = video.calculate_frame(images, parsed3, frame=3)
    assert predict_frames == [(0, [1]), (2, [1])]
    assert meta3.get("keep_existing_mask") is True
    assert seed3.token != seed2.token

    # 4. Track video spans across multi-frame anchors {0, 2, 3}
    track_calls = []

    def fake_track(imgs, prompts, *, seed, start, end, direction):
        track_calls.append((seed.object_frames, [p.get("refinements") for p in prompts]))
        return {idx: (m_obj1 | m_obj2).copy() for idx in range(start, end + 1)}

    monkeypatch.setattr(video, "_native_track", fake_track)
    res = video.track_video(
        images, parsed3, frame=3, token=seed3.token, start=0, end=4, direction="both"
    )
    assert res.anchors == frozenset({0, 2, 3})
    assert track_calls and track_calls[0][0] == (0, 2)
    assert np.any(res.masks[0, 2:8, 2:10]) and np.any(res.masks[2, 10:16, 15:24])


def test_cross_frame_negative_refinement_with_prior_mask_subtracts_hand(monkeypatch):
    import types
    import ths_universal_inpaint.mask_sam3 as sam3_mod

    images = _clip(frames=4, height=20, width=30)
    face = np.zeros((20, 30), dtype=bool)
    face[2:10, 12:24] = True
    hand = np.zeros((20, 30), dtype=bool)
    hand[12:19, 2:9] = True
    face_and_hand = face | hand

    predict_calls = []

    def fake_predict(_image, event, *, keep_loaded=False, mask_input=None):
        predict_calls.append((dict(event), None if mask_input is None else mask_input.copy()))
        if mask_input is not None:
            # Even if SAM3 leaks back a disconnected island at the negative point,
            # _drop_negated_disconnected_islands must ensure it is excluded.
            return face_and_hand.copy()
        return face.copy()

    video.clear_sam3_video_cache()
    monkeypatch.setattr(video, "_predict_native", fake_predict)

    # 1. Seed Object 1 on frame 0 and run track_video where frame 2 erroneously includes hand.
    obj1 = {"frame": 0, "points": [[0.6, 0.3]], "labels": [1]}
    parsed0 = core.parse_mask_data({"sam3": {"frame": 0, "prompt": obj1}}, len(images))
    _m0, seed0, _meta0 = video.calculate_frame(images, parsed0, frame=0)

    def fake_track_pass1(imgs, prompts, *, seed, start, end, direction):
        return {
            0: face.copy(),
            1: face.copy(),
            2: face_and_hand.copy(),
            3: face_and_hand.copy(),
        }

    monkeypatch.setattr(video, "_native_track", fake_track_pass1)
    video.track_video(images, parsed0, frame=0, token=seed0.token, start=0, end=3, direction="both")

    # 2. Now click '-' on the hand on frame 2 (cross-frame negative-only refinement).
    obj1_neg = {
        **obj1,
        "refinements": [{"frame": 2, "points": [[5 / 29.0, 15 / 19.0]], "labels": [0]}],
    }
    parsed2 = core.parse_mask_data({"sam3": {"frame": 2, "prompt": obj1_neg}}, len(images))
    mask2, seed2, meta2 = video.calculate_frame(images, parsed2, frame=2)

    assert not meta2.get("keep_existing_mask"), "prior-conditioned negative refinement must update frame immediately"
    assert len(predict_calls) == 2
    ref_event, passed_prior = predict_calls[1]
    assert passed_prior is not None
    assert bool(passed_prior[5, 18]) and not bool(passed_prior[15, 5]), "disconnected negated island cleared from mask_input"
    assert 0 in ref_event["labels"] and 1 in ref_event["labels"], "auto-positive anchor appended on safe foreground"
    assert bool(mask2[5, 18]) and not bool(mask2[15, 5]), "hand subtracted while face preserved"
    assert len(seed2.refinement_masks) == 1
    r_oid, r_frm, r_mask = seed2.refinement_masks[0]
    assert (r_oid, r_frm) == (1, 2)
    assert bool(r_mask[5, 18]) and not bool(r_mask[15, 5])

    # 3. Verify _native_track feeds seed2.refinement_masks via add_new_mask on frame 2 (not unconditioned add_new_points).
    monkeypatch.undo()
    monkeypatch.setattr(video, "_predict_native", fake_predict)
    events_log = []

    class FakeTracker:
        def add_new_mask(self, *, inference_state, frame_idx, obj_id, mask):
            events_log.append(("add_new_mask", int(obj_id), int(frame_idx), bool(mask[5, 18]), bool(mask[15, 5])))

        def add_new_points(self, **kwargs):
            events_log.append(("add_new_points", kwargs.get("frame_idx")))

        def propagate_in_video_preflight(self, state, run_mem_encoder=True):
            events_log.append(("preflight", bool(run_mem_encoder)))

    class FakeModel:
        def __init__(self):
            self.model = types.SimpleNamespace(
                tracker=FakeTracker(),
                _get_tracker_inference_states_by_obj_ids=lambda st, ids: [{"obj_id": ids[0]}],
                _prepare_backbone_feats=lambda st, f, rev: events_log.append(("prepare_feats", int(f))),
            )

        def handle_stream_request(self, request):
            events_log.append(("propagate", request["propagation_direction"], request["start_frame_index"]))
            for idx in range(4):
                yield {"frame_index": idx, "outputs": {"obj_ids": np.array([1]), "out_binary_masks": face[None, ...]}}

    fake_video_state = types.SimpleNamespace(
        num_frames=4,
        prompts=[1],
        session_uuid="sess-1",
        temp_dir=None,
        with_prompt=lambda p: fake_video_state,
    )
    fake_vmod = types.SimpleNamespace(
        create_video_state=lambda video_frames, config: fake_video_state,
        cleanup_temp_dir=lambda d: None,
    )
    fake_reconstructor = types.SimpleNamespace(
        get_inference_state=lambda m, vs: {"tracker_inference_states": [{}]},
    )
    fake_prompt_cls = types.SimpleNamespace(
        create_point=lambda f, oid, pts, lbls: (f, oid),
        create_box=lambda f, oid, box, is_positive=True: (f, oid),
    )
    fake_cache = types.SimpleNamespace(get_or_build_model=lambda cfg: FakeModel())
    monkeypatch.setattr(video, "_native_video_modules", lambda: (fake_vmod, fake_reconstructor, fake_prompt_cls, lambda: {}))
    monkeypatch.setattr(video, "_native_model_cache_module", lambda: fake_cache)
    monkeypatch.setattr(video, "_native_config", lambda: {})
    monkeypatch.setattr(video, "_offload_model", lambda m: None)
    monkeypatch.setitem(sys.modules, "comfy", types.SimpleNamespace(model_management=types.SimpleNamespace(load_models_gpu=lambda models: None)))
    monkeypatch.setitem(sys.modules, "comfy.model_management", sys.modules["comfy"].model_management)

    per_obj = {}
    out = video._native_track(
        images,
        video._prompt_events(parsed2, 2),
        seed=seed2,
        start=0,
        end=3,
        direction="both",
        object_masks_out=per_obj,
    )
    assert ("add_new_mask", 1, 0, True, False) in events_log
    assert ("add_new_mask", 1, 2, True, False) in events_log
    assert not any(ev[0] == "add_new_points" for ev in events_log)
    assert 1 in per_obj and 2 in per_obj[1]
    assert set(out.keys()) == {0, 1, 2, 3}
    assert sam3_mod is not None


def test_cold_cache_cross_frame_negative_refinement_warms_up_before_add_new_points(monkeypatch):
    import types

    images = _clip(frames=4, height=20, width=30)
    face = np.zeros((20, 30), dtype=bool)
    face[2:10, 12:24] = True
    video.clear_sam3_video_cache()
    monkeypatch.setattr(video, "_predict_native", lambda img, ev, **kw: face.copy())

    obj1_neg = {
        "frame": 0,
        "points": [[0.6, 0.3]],
        "labels": [1],
        "refinements": [{"frame": 2, "points": [[0.15, 0.75]], "labels": [0]}],
    }
    parsed = core.parse_mask_data({"sam3": {"frame": 2, "prompt": obj1_neg}}, len(images))
    _mask, seed, meta = video.calculate_frame(images, parsed, frame=2)
    assert meta.get("keep_existing_mask") is True
    assert seed.refinement_masks == ()

    events_log = []

    class FakeTracker:
        def add_new_mask(self, *, inference_state, frame_idx, obj_id, mask):
            events_log.append(("add_new_mask", int(frame_idx)))

        def add_new_points(self, **kwargs):
            events_log.append(("add_new_points", int(kwargs["frame_idx"]), bool(kwargs.get("use_prev_mem_frame"))))

        def propagate_in_video_preflight(self, state, run_mem_encoder=True):
            events_log.append(("preflight", bool(run_mem_encoder)))

    class FakeModel:
        def __init__(self):
            self.model = types.SimpleNamespace(
                tracker=FakeTracker(),
                _get_tracker_inference_states_by_obj_ids=lambda st, ids: [{"obj_id": ids[0]}],
                _prepare_backbone_feats=lambda st, f, rev: events_log.append(("prepare_feats", int(f))),
            )

        def handle_stream_request(self, request):
            events_log.append(("propagate", request["propagation_direction"]))
            for idx in range(4):
                yield {"frame_index": idx, "outputs": {"obj_ids": np.array([1]), "out_binary_masks": face[None, ...]}}

    fake_video_state = types.SimpleNamespace(
        num_frames=4,
        prompts=[1],
        session_uuid="sess-cold",
        temp_dir=None,
        with_prompt=lambda p: fake_video_state,
    )
    fake_vmod = types.SimpleNamespace(
        create_video_state=lambda video_frames, config: fake_video_state,
        cleanup_temp_dir=lambda d: None,
    )
    fake_reconstructor = types.SimpleNamespace(
        get_inference_state=lambda m, vs: {"tracker_inference_states": [{}]},
    )
    fake_prompt_cls = types.SimpleNamespace(
        create_point=lambda f, oid, pts, lbls: (f, oid),
        create_box=lambda f, oid, box, is_positive=True: (f, oid),
    )
    fake_cache = types.SimpleNamespace(get_or_build_model=lambda cfg: FakeModel())
    monkeypatch.setattr(video, "_native_video_modules", lambda: (fake_vmod, fake_reconstructor, fake_prompt_cls, lambda: {}))
    monkeypatch.setattr(video, "_native_model_cache_module", lambda: fake_cache)
    monkeypatch.setattr(video, "_native_config", lambda: {})
    monkeypatch.setattr(video, "_offload_model", lambda m: None)
    monkeypatch.setitem(sys.modules, "comfy", types.SimpleNamespace(model_management=types.SimpleNamespace(load_models_gpu=lambda models: None)))
    monkeypatch.setitem(sys.modules, "comfy.model_management", sys.modules["comfy"].model_management)

    video._native_track(images, video._prompt_events(parsed, 2), seed=seed, start=0, end=3, direction="forward")
    first_prop = next(i for i, e in enumerate(events_log) if e[0] == "propagate")
    add_pts_idx = next(i for i, e in enumerate(events_log) if e[0] == "add_new_points")
    last_prop = max(i for i, e in enumerate(events_log) if e[0] == "propagate")
    assert first_prop < add_pts_idx < last_prop, f"expected warmup propagate -> add_new_points -> final propagate, got {events_log}"
    assert ("add_new_points", 2, True) in events_log


def test_predict_native_ranks_candidate_excluding_negative_points_above_higher_score(monkeypatch):
    import types
    import torch
    import ths_universal_inpaint.mask_sam3 as sam3_mod

    image = np.zeros((20, 30, 3), dtype=np.float32)
    face = np.zeros((20, 30), dtype=np.float32)
    face[2:10, 12:24] = 1.0
    hand = np.zeros((20, 30), dtype=np.float32)
    hand[12:19, 2:9] = 1.0
    both = np.maximum(face, hand)
    seen = {}

    class FakePredictor:
        def __init__(self):
            w = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
            self.inst_interactive_predictor = types.SimpleNamespace(
                model=types.SimpleNamespace(
                    sam_prompt_encoder=types.SimpleNamespace(
                        mask_input_size=(16, 16),
                        mask_downscaling=[types.SimpleNamespace(weight=w)],
                    )
                )
            )

        def predict_inst(self, state, *, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True, normalize_coords=True):
            seen["mask_input_shape"] = None if mask_input is None else tuple(mask_input.shape)
            # Candidate 0 has higher score (0.95) but STILL covers the negative hand point;
            # Candidate 1 has lower score (0.81) and excludes the negative hand point.
            return np.stack([both, face], axis=0), np.array([0.95, 0.81], dtype=np.float32), None

    fake_model = types.SimpleNamespace(
        processor=types.SimpleNamespace(
            set_image=lambda pil: {"img": pil},
            model=FakePredictor(),
        )
    )
    monkeypatch.setattr(sam3_mod, "_native_config", lambda: {})
    monkeypatch.setattr(sam3_mod, "_native_model_cache_module", lambda: types.SimpleNamespace(get_or_build_model=lambda cfg: fake_model))
    monkeypatch.setattr(sam3_mod, "_offload_model", lambda m: None)
    monkeypatch.setitem(sys.modules, "comfy", types.SimpleNamespace(model_management=types.SimpleNamespace(load_models_gpu=lambda models: None)))
    monkeypatch.setitem(sys.modules, "comfy.model_management", sys.modules["comfy"].model_management)

    event = {"points": [[18 / 29.0, 5 / 19.0], [5 / 29.0, 15 / 19.0]], "labels": [1, 0]}
    selected = sam3_mod._predict_native(image, event, mask_input=(face > 0.5))
    assert seen["mask_input_shape"] == (1, 16, 16)
    assert bool(selected[5, 18]) and not bool(selected[15, 5])


