from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from ths_universal_inpaint.repair_nodes import (  # noqa: E402
    THSUniversalH3PadSource,
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
    np.testing.assert_array_equal(padded[0][outside], images[0][outside])
    assert np.any(np.abs(padded[0][mask[0].astype(bool)] - images[0][mask[0].astype(bool)]) > 1e-3)


def test_pad_without_audio_creates_silent_audio_contract():
    images = np.zeros((5, 2, 2, 3), dtype=np.float32)
    mask = np.ones((5, 2, 2), dtype=np.float32)
    _, _, audio, original_count, padded_count = pad_source_inputs(images, mask, None, model_fps=24)
    assert original_count == padded_count == 5
    assert audio["waveform"].shape == (1, 2, 10_000)
    assert float(audio["waveform"].sum()) == 0.0


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
