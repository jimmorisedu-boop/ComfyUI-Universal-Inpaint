from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from ths_universal_inpaint.repair_nodes import (  # noqa: E402
    THSUniversalH3PadSource,
    THSUniversalMaskedComposite,
    THSUniversalTrimFrames,
    _next_h3_frame_count,
    composite_exact,
    pad_source_inputs,
    telea_prefill_masked,
)


def test_h3_grid_boundaries_are_smallest_valid_timeline():
    assert _next_h3_frame_count(1) == 5
    assert _next_h3_frame_count(5) == 5
    assert _next_h3_frame_count(6) == 22
    assert _next_h3_frame_count(22) == 22
    assert _next_h3_frame_count(23) == 39


def test_pad_source_inputs_preserves_real_frames_mask_and_audio():
    images = np.arange(6 * 4 * 4 * 3, dtype=np.float32).reshape(6, 4, 4, 3)
    mask = np.zeros((6, 4, 4), dtype=np.float32)
    mask[:, 1:3, 1:3] = 1.0
    waveform = np.arange(2 * 5, dtype=np.float32).reshape(2, 5)

    padded_images, padded_mask, padded_audio, original_count, padded_count = pad_source_inputs(
        images,
        mask,
        {"waveform": waveform, "sample_rate": 24},
        model_fps=24,
    )

    assert (original_count, padded_count) == (6, 22)
    assert padded_images.shape == (22, 4, 4, 3)
    assert padded_mask.shape == (22, 4, 4)
    np.testing.assert_array_equal(padded_images[:6], images)
    np.testing.assert_array_equal(padded_images[6:], np.repeat(images[-1:], 16, axis=0))
    np.testing.assert_array_equal(padded_mask[:6], mask)
    np.testing.assert_array_equal(padded_mask[6:], np.repeat(mask[-1:], 16, axis=0))
    assert padded_audio["sample_rate"] == 24
    assert padded_audio["waveform"].shape == (1, 2, 22)
    np.testing.assert_array_equal(padded_audio["waveform"][0, :, :5], waveform)


def test_telea_prefill_changes_only_masked_pixels():
    pytest.importorskip("cv2")
    height = width = 24
    images = np.full((1, height, width, 3), 0.35, dtype=np.float32)
    images[0, :, :, 1] = np.linspace(0.1, 0.8, width, dtype=np.float32)[None, :]
    # Obvious synthetic checker inside a finite repair box.
    images[0, 8:16, 8:16] = np.where(
        (np.indices((8, 8)).sum(axis=0) % 2)[..., None],
        np.array([1.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    mask = np.zeros((1, height, width), dtype=np.float32)
    mask[:, 7:17, 7:17] = 1.0
    filled = telea_prefill_masked(images, mask)

    outside = ~mask[0].astype(bool)
    np.testing.assert_array_equal(filled[0][outside], images[0][outside])
    assert np.any(np.abs(filled[0][mask[0].astype(bool)] - images[0][mask[0].astype(bool)]) > 1e-3)


def test_pad_node_prefills_model_source_but_preserves_pad_helper_contract():
    pytest.importorskip("cv2")
    images = np.full((5, 16, 16, 3), 0.4, dtype=np.float32)
    images[:, 6:10, 6:10] = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    mask = np.zeros((5, 16, 16), dtype=np.float32)
    mask[:, 5:11, 5:11] = 1.0
    padded, _, _, original_count, padded_count = THSUniversalH3PadSource().run(images, mask, model_fps=24)

    assert (original_count, padded_count) == (5, 5)
    outside = ~mask[0].astype(bool)
    np.testing.assert_array_equal(padded[0, :16, :16][outside], images[0][outside])
    assert np.any(np.abs(padded[0, :16, :16][mask[0].astype(bool)] - images[0][mask[0].astype(bool)]) > 1e-3)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_h3_padding_and_output_crop_keep_source_coordinates(backend, monkeypatch):
    # Isolate geometry from the existing Telea prefill and exercise both axes.
    monkeypatch.setattr("ths_universal_inpaint.repair_nodes.telea_prefill_masked", lambda images, mask: images)
    images_np = np.arange(2 * 37 * 33 * 3, dtype=np.float32).reshape(2, 37, 33, 3) / 10_000
    mask_np = np.zeros(images_np.shape[:3], dtype=np.float32)
    mask_np[:, -2:, -2:] = 1
    as_tensor = pytest.importorskip("torch").from_numpy if backend == "torch" else np.asarray
    images, mask = as_tensor(images_np), as_tensor(mask_np)

    padded_images, padded_mask, _, count, padded_count = THSUniversalH3PadSource().run(images, mask)

    assert (count, padded_count) == (2, 5)
    assert tuple(padded_images.shape) == (5, 64, 64, 3)
    assert tuple(padded_mask.shape) == (5, 64, 64)
    np.testing.assert_array_equal(padded_images[:count, :37, :33], images_np)
    np.testing.assert_array_equal(padded_images[:count, -1, -1], images_np[:, -1, -1])
    np.testing.assert_array_equal(padded_mask[:count, :37, :33], mask_np)
    assert float(padded_mask[:, 37:].sum() + padded_mask[:, :, 33:].sum()) == 0

    decoded = as_tensor(np.full(padded_images.shape, 0.75, dtype=np.float32))
    trimmed, = THSUniversalTrimFrames().run(decoded, count)
    output, = THSUniversalMaskedComposite().run(images, trimmed, mask)
    np.testing.assert_array_equal(output, np.where(mask_np[..., None] > 0, 0.75, images_np))


@pytest.mark.parametrize("tensor", [False, True])
def test_recolor_keeps_mask_geometry_source_shading_and_input(tensor):
    images = np.zeros((2, 32, 32, 3), dtype=np.float32)
    images[:, 8:16] = 0.15
    images[:, 16:24] = 0.65
    mask = np.zeros(images.shape[:3], dtype=np.float32)
    mask[:, 8:24, 8:24] = 1
    original = images.copy()
    convert = pytest.importorskip("torch").from_numpy if tensor else np.asarray
    padded = THSUniversalH3PadSource().run(convert(images), convert(mask), source_edit="#ff0000")[0]
    result = np.asarray(padded[:2])
    np.testing.assert_array_equal(result[mask == 0], original[mask == 0])
    np.testing.assert_array_equal(images, original)
    assert result[0, 12, 12, 0] > result[0, 12, 12, 1]
    assert result[0, 20, 12, 0] > result[0, 12, 12, 0]
    np.testing.assert_array_equal(result[0], result[1])
    composite = THSUniversalMaskedComposite()
    assert composite.check_lazy_status(images, None, mask, source_edit="#ff0000") == []
    assert composite.check_lazy_status(images, None, mask, source_edit="reference") == ["generated"]
    direct, = composite.run(convert(images), None, convert(mask), source_edit="#ff0000")
    np.testing.assert_array_equal(np.asarray(direct), result)


@pytest.mark.parametrize("tensor", [False, True])
def test_neutral_fill_is_constant_and_preserves_surrounding_source(tensor):
    convert = pytest.importorskip("torch").from_numpy if tensor else np.asarray
    images = np.random.default_rng(4).random((2, 32, 32, 3), dtype=np.float32)
    mask = np.zeros(images.shape[:3], dtype=np.float32)
    mask[:, 8:16, 8:16] = 1
    actual = THSUniversalH3PadSource().run(convert(images), convert(mask), source_edit="clear",
                                           fill_strategy="neutral_gray")[0][:2]
    np.testing.assert_array_equal(np.asarray(actual)[mask == 0], images[mask == 0])
    np.testing.assert_allclose(np.asarray(actual)[mask == 1], 128 / 255)


@pytest.mark.parametrize("fill_strategy", ["context", "neutral_gray"])
def test_photo_subject_moves_with_mask_and_corrected_frame_retains_source(fill_strategy):
    pytest.importorskip("cv2")
    images = np.full((2, 64, 64, 3), 0.25, dtype=np.float32)
    mask = np.zeros(images.shape[:3], dtype=np.float32)
    mask[0, 12:44, 4:36] = 1
    mask[1, 12:44, 20:52] = 1
    photo = np.zeros((1, 40, 40, 3), dtype=np.float32)
    photo[0, 6:34, 8:32, 0] = 1
    pad = THSUniversalH3PadSource()
    prepared = pad.run(images, mask, source_edit="keep", replacement_image=photo,
                       repair_mode="with_reference", fill_strategy=fill_strategy)[0][:2]
    assert prepared[0, 28, 20, 0] == prepared[1, 28, 36, 0] == 1
    np.testing.assert_array_equal(prepared[mask == 0], images[mask == 0])
    corrected = pad.run(images, mask, source_edit="clear", replacement_image=photo, repair_mode="corrected_frame")[0]
    np.testing.assert_array_equal(corrected[:2], images)


def test_pad_without_audio_creates_silent_audio_contract():
    images = np.zeros((5, 2, 2, 3), dtype=np.float32)
    mask = np.ones((5, 2, 2), dtype=np.float32)
    _, _, audio, original_count, padded_count = pad_source_inputs(images, mask, None, model_fps=24)
    assert original_count == padded_count == 5
    assert audio["waveform"].shape == (1, 2, 10_000)
    assert float(audio["waveform"].sum()) == 0.0


@pytest.mark.parametrize("settings", [{"prefill": False}, {"prefill": True, "source_edit": "keep"}])
def test_pad_node_can_keep_original_pixels_as_the_h3_source(monkeypatch, settings):
    def forbidden(*args):
        pytest.fail("replacement mode should keep the original pixel values")

    monkeypatch.setattr("ths_universal_inpaint.repair_nodes.telea_prefill_masked", forbidden)
    images = np.random.default_rng(42).random((6, 35, 33, 3), dtype=np.float32)
    mask = np.ones(images.shape[:3], dtype=np.float32)
    padded, padded_mask, _, original_count, padded_count = THSUniversalH3PadSource().run(images, mask, **settings)
    assert (original_count, padded_count) == (6, 22)
    np.testing.assert_array_equal(padded[:original_count, :35, :33], images)
    assert np.count_nonzero(padded_mask[:, 35:]) == 0


def test_composite_exactly_preserves_zero_mask_pixels():
    source = np.array(
        [[[[0.1234567, 0.2222222, 0.3333333], [0.8765432, 0.1111111, 0.9999999]]]],
        dtype=np.float32,
    )
    generated = np.ones_like(source)
    mask = np.array([[[0.0, 1.0]]], dtype=np.float32)

    output = composite_exact(source, generated, mask)

    np.testing.assert_array_equal(output[0, 0, 0], source[0, 0, 0])
    np.testing.assert_array_equal(output[0, 0, 1], generated[0, 0, 1])


def test_composite_rejects_geometry_mismatch():
    source = np.zeros((2, 4, 4, 3), dtype=np.float32)
    generated = np.zeros((2, 5, 4, 3), dtype=np.float32)
    mask = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="identical shape"):
        composite_exact(source, generated, mask)
