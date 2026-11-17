import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ths_universal_inpaint.mask_sam3 as sam3  # noqa: E402
from ths_universal_inpaint.mask_core import (  # noqa: E402
    MaskDataError,
    MaskRefinementError,
    parse_mask_data,
    rasterize_events,
)


def _event(**overrides):
    value = {
        "type": "rect",
        "mode": "add",
        "refine": True,
        "refine_method": "sam3",
        "x": 0.2,
        "y": 0.2,
        "w": 0.4,
        "h": 0.4,
    }
    value.update(overrides)
    return value


def test_parser_validates_and_round_trips_refine_method():
    parsed = parse_mask_data({"frames": {"0": {"events": [_event()]}}}, 1)
    assert parsed["frames"]["0"]["events"][0]["refine_method"] == "sam3"
    grabcut = parse_mask_data(
        {"frames": {"0": {"events": [{k: v for k, v in _event().items() if k != "refine_method"}]}}},
        1,
    )
    assert "refine_method" not in grabcut["frames"]["0"]["events"][0]
    with pytest.raises(MaskDataError, match="refine_method"):
        parse_mask_data({"frames": {"0": {"events": [_event(refine_method="bogus")]}}}, 1)
    with pytest.raises(MaskDataError, match="refine_method"):
        parse_mask_data({"frames": {"0": {"events": [_event(refine_method={})]}}}, 1)


def test_sam3_adapter_maps_pixel_box_clips_result_and_reuses_cache(monkeypatch):
    image = np.zeros((20, 30, 3), dtype=np.float32)
    event = _event(x=0.25, y=0.20, w=0.4, h=0.5)
    calls = []
    sam3.clear_sam3_cache()
    monkeypatch.setattr(
        sam3,
        "_native_config",
        lambda: {"checkpoint_path": "C:/models/sam3.safetensors"},
    )

    def fake_predict(source, received):
        calls.append(dict(received))
        mask = np.zeros(source.shape[:2], dtype=bool)
        # Deliberately include an outside pixel; the adapter must clip it.
        mask[5:10, 8:15] = True
        mask[0, 0] = True
        return mask

    monkeypatch.setattr(sam3, "_predict_native", fake_predict)
    first = sam3.refine_rectangle_sam3(image, event)
    second = sam3.refine_rectangle_sam3(image, event)
    assert first.shape == image.shape[:2]
    assert first.dtype == np.bool_
    assert np.array_equal(first, second)
    assert len(calls) == 1
    x0 = int(np.floor(event["x"] * (image.shape[1] - 1)))
    y0 = int(np.floor(event["y"] * (image.shape[0] - 1)))
    x1 = int(np.ceil((event["x"] + event["w"]) * (image.shape[1] - 1)))
    y1 = int(np.ceil((event["y"] + event["h"]) * (image.shape[0] - 1)))
    outside = first.copy()
    outside[y0:y1 + 1, x0:x1 + 1] = False
    assert not outside.any()
    assert sam3.sam3_cache_info()["hits"] == 1
    assert calls[0]["refine_method"] == "sam3"


def test_sam3_failure_is_typed_and_never_falls_back(monkeypatch):
    image = np.zeros((12, 16, 3), dtype=np.float32)
    event = _event()
    monkeypatch.setattr(
        sam3,
        "_native_config",
        lambda: {"checkpoint_path": "C:/models/sam3.safetensors"},
    )
    monkeypatch.setattr(
        sam3,
        "_predict_native",
        lambda source, received: np.zeros(source.shape[:2], dtype=bool),
    )
    sam3.clear_sam3_cache()
    with pytest.raises(MaskRefinementError, match="SAM3"):
        rasterize_events([event], image.shape[0], image.shape[1], image=image)
