from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from ths_universal_inpaint.corrected_frame_nodes import (  # noqa: E402
    THSUniversalCorrectedFrameGuide,
)


def _install_fake_add_guide(monkeypatch):
    calls = []
    package = types.ModuleType("comfy_extras")
    package.__path__ = []
    module = types.ModuleType("comfy_extras.nodes_minimax_h3")

    class FakeAddGuide:
        @classmethod
        def execute(cls, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(result=("guided-positive",))

    module.MiniMaxH3AddGuide = FakeAddGuide
    monkeypatch.setitem(sys.modules, "comfy_extras", package)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_minimax_h3", module)
    return calls


def _fixtures(frame_count=6, height=4, width=5, channels=3):
    source = np.zeros((frame_count, height, width, channels), dtype=np.float32)
    source[2] = 0.25
    mask = np.zeros((frame_count, height, width), dtype=np.float32)
    mask[2, 1:3, 2:4] = 0.5
    corrected = np.ones((1, height, width, channels), dtype=np.float32)
    return source, mask, corrected


def test_schema_contract_has_original_inputs_and_three_mode_selector():
    schema = THSUniversalCorrectedFrameGuide.INPUT_TYPES()
    assert list(schema["required"]) == [
        "positive",
        "latent",
        "vae",
        "source_images",
        "mask",
        "mode",
        "frame_number",
    ]
    assert schema["required"]["mode"][1]["options"] == [
        "without_reference",
        "with_reference",
        "corrected_frame",
    ]
    assert schema["required"]["frame_number"][1]["default"] == 1
    assert list(schema["optional"]) == ["corrected_image"]
    assert THSUniversalCorrectedFrameGuide.RETURN_NAMES == (
        "positive",
        "guide_preview",
        "status",
    )


def test_bypass_does_not_inspect_missing_optional_upload():
    source, mask, _corrected = _fixtures()
    result = THSUniversalCorrectedFrameGuide().run(
        positive="positive",
        latent={"samples": "latent"},
        vae="vae",
        source_images=source,
        mask=mask,
        mode="without_reference",
        frame_number=7,
        corrected_image=object(),
    )
    assert result[0] == "positive"
    np.testing.assert_array_equal(result[1], source[:1])
    assert "without_reference" in result[2]


def test_corrected_frame_uses_original_frame_index_and_exact_masked_merge(monkeypatch):
    calls = _install_fake_add_guide(monkeypatch)
    source, mask, corrected = _fixtures()
    result = THSUniversalCorrectedFrameGuide().run(
        positive="positive",
        latent={"samples": "latent"},
        vae="vae",
        source_images=source,
        mask=mask,
        mode="corrected_frame",
        frame_number=3,
        corrected_image=corrected,
    )
    assert result[0] == "guided-positive"
    assert result[1].shape == (1, 4, 5, 3)
    assert calls[0]["frame_idx"] == 2
    assert calls[0]["vae"] == "vae"
    assert calls[0]["image"].shape == (1, 32, 32, 3)
    np.testing.assert_array_equal(calls[0]["image"][:, :4, :5], result[1])
    merged = calls[0]["image"][0]
    np.testing.assert_array_equal(merged[0, 0], source[2, 0, 0])
    np.testing.assert_array_equal(merged[1, 2], np.full((3,), 0.625, dtype=np.float32))
    assert "original frame 3" in result[2]


@pytest.mark.parametrize(
    "frame_number,match",
    [
        (7, "original source frames"),
        (0, "original source frames"),
    ],
)
def test_frame_number_is_checked_against_unpadded_source(monkeypatch, frame_number, match):
    _install_fake_add_guide(monkeypatch)
    source, mask, corrected = _fixtures(frame_count=6)
    with pytest.raises(ValueError, match=match):
        THSUniversalCorrectedFrameGuide().run(
            positive="positive",
            latent={"samples": "latent"},
            vae="vae",
            source_images=source,
            mask=mask,
            mode="corrected_frame",
            frame_number=frame_number,
            corrected_image=corrected,
        )


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda source, mask, corrected: (source, mask[:-1], corrected), "equal frame counts"),
        (lambda source, mask, corrected: (source, mask[:, :-1], corrected), "matching canvas"),
        (lambda source, mask, corrected: (source, mask, corrected[:, :-1]), "exactly one frame"),
        (lambda source, mask, corrected: (source, np.zeros_like(mask), corrected), "mask is empty"),
    ],
)
def test_corrected_frame_rejects_shape_or_empty_mask(monkeypatch, mutator, match):
    _install_fake_add_guide(monkeypatch)
    source, mask, corrected = mutator(*_fixtures())
    with pytest.raises(ValueError, match=match):
        THSUniversalCorrectedFrameGuide().run(
            positive="positive",
            latent={"samples": "latent"},
            vae="vae",
            source_images=source,
            mask=mask,
            mode="corrected_frame",
            frame_number=3,
            corrected_image=corrected,
        )


def test_corrected_frame_rejects_nonfinite_source_and_mask(monkeypatch):
    _install_fake_add_guide(monkeypatch)
    source, mask, corrected = _fixtures()
    source[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="source_images must contain finite"):
        THSUniversalCorrectedFrameGuide().run(
            positive="positive",
            latent={"samples": "latent"},
            vae="vae",
            source_images=source,
            mask=mask,
            mode="corrected_frame",
            frame_number=3,
            corrected_image=corrected,
        )

    source, mask, corrected = _fixtures()
    mask[2, 1, 1] = np.inf
    with pytest.raises(ValueError, match="mask must contain finite"):
        THSUniversalCorrectedFrameGuide().run(
            positive="positive",
            latent={"samples": "latent"},
            vae="vae",
            source_images=source,
            mask=mask,
            mode="corrected_frame",
            frame_number=3,
            corrected_image=corrected,
        )
