import json

import numpy as np
import pytest

from ths_universal_inpaint.h3_window import THSUniversalH3Window, mask_extent, plan_window
from ths_universal_inpaint.repair_nodes import THSUniversalH3PadSource, THSUniversalMaskedComposite
from ths_universal_inpaint.corrected_frame_nodes import THSUniversalCorrectedFrameGuide


def clip():
    source = np.zeros((36, 278, 373, 3), np.float32)
    mask = np.zeros(source.shape[:3], np.float32)
    mask[7:29, 60:100, 140:180] = .5
    mask[28, 90, 258] = 1e-6
    return source, mask


@pytest.mark.parametrize("probe", [-1, 24])
def test_roundtrip_original_time_coordinates_feather_and_audio(probe):
    source, mask = clip()
    audio = {"waveform": np.arange(4000)[None,None], "sample_rate": 2500}
    crop, alpha, sound, w = THSUniversalH3Window().run(source, mask, audio=audio, fps=25, probe_frame=probe)
    assert (w.first,w.stop,w.edit_first,w.edit_stop) == ((3,33,7,29) if probe < 0 else (13,33,17,29))
    assert 0 < w.box[0] <= 140 < 259 <= w.box[2] < source.shape[2]
    assert np.array_equal(sound["waveform"], audio["waveform"][...,w.first*100:w.stop*100])
    padded, _, _, count, _ = THSUniversalH3PadSource().run(crop, alpha, prefill=False)
    actual, = THSUniversalMaskedComposite().run(source, np.ones_like(padded[:count]), mask, w)
    expected = mask if probe < 0 else mask[w.edit_first:w.edit_stop]
    np.testing.assert_array_equal(actual, np.broadcast_to(expected[...,None],actual.shape))
    assert not source.any()


def test_wide_motion_reverts_to_full_and_manual_never_clips_mask():
    source, mask = clip()
    mask[7,0,0] = mask[28,-1,-1] = 1
    for opts in ({}, {"mode":"manual", "box":[30,30,270,260], "size":[373,278]}):
        assert plan_window(mask.shape,mask_extent(mask),json.dumps(opts)).box == (0,0,373,278)
    with pytest.raises(ValueError,match="непустая"):
        plan_window(mask.shape,None)
    with pytest.raises(ValueError,match="фрагменте"):
        plan_window(mask.shape,{"box":[0,0,1,1],"frames":[0,35]},probe_frame=17)


def test_corrected_anchor_uses_original_preview_and_local_model_index(monkeypatch):
    from test_corrected_frame_nodes import _install_fake_add_guide
    calls = _install_fake_add_guide(monkeypatch)
    source, mask = clip()
    _,_,_,win = THSUniversalH3Window().run(source,mask,probe_frame=24)
    _,preview,_ = THSUniversalCorrectedFrameGuide().run("p",{},None,source,mask,"corrected_frame",25,np.ones_like(source[:1]),win)
    assert calls[0]["frame_idx"] == 24 - win.first
    assert preview.shape == source[:1].shape
    h,w = win.crop(preview).shape[1:3]
    np.testing.assert_array_equal(calls[0]["image"][:,:h,:w],win.crop(preview))
