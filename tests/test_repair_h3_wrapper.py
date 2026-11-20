from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ths_universal_inpaint.repair_nodes import THSUniversalH3RepairConditioning  # noqa: E402


class FakeClip:
    def __init__(self):
        self.token_calls = []

    def tokenize(self, prompt, images=None):
        self.token_calls.append((prompt, images))
        return {"prompt": prompt, "images": images}

    def encode_from_tokens_scheduled(self, tokens):
        return ("encoded", tokens)


def install_fake_h3(monkeypatch, calls):
    package = types.ModuleType("comfy_extras")
    package.__path__ = []
    module = types.ModuleType("comfy_extras.nodes_minimax_h3")

    class FakeH3:
        @classmethod
        def execute(cls, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(result=(("positive", kwargs), {"samples": "h3"}))

    module.MiniMaxH3ImageToVideo = FakeH3
    monkeypatch.setitem(sys.modules, "comfy_extras", package)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_minimax_h3", module)


def test_wrapper_calls_installed_h3_api_without_reference_images(monkeypatch):
    calls = []
    install_fake_h3(monkeypatch, calls)
    clip = FakeClip()

    positive, negative, h3_latent = THSUniversalH3RepairConditioning().run(
        clip=clip,
        vae=object(),
        prompt="repair masked pixels",
        negative_prompt="do not alter context",
        width=32,
        height=32,
        length=5,
        anchor_source_edges=False,
    )

    assert calls and calls[0]["first_frame"] is None and calls[0]["last_frame"] is None
    assert "ref_images" not in calls[0]
    assert calls[0]["length"] == 5
    assert h3_latent == {"samples": "h3"}
    assert clip.token_calls[-1] == ("do not alter context", [])
    assert positive[0] == "positive"
    assert negative[0] == "encoded"


def test_wrapper_edge_anchor_is_explicit_and_uses_first_last_frames(monkeypatch):
    calls = []
    install_fake_h3(monkeypatch, calls)
    clip = FakeClip()
    source = np.zeros((5, 32, 32, 3), dtype=np.float32)
    source[0] = 0.25
    source[-1] = 0.75

    THSUniversalH3RepairConditioning().run(
        clip=clip,
        vae=object(),
        prompt="repair",
        negative_prompt="",
        width=32,
        height=32,
        length=5,
        anchor_source_edges=True,
        source_images=source,
    )

    np.testing.assert_array_equal(calls[0]["first_frame"], source[:1])
    np.testing.assert_array_equal(calls[0]["last_frame"], source[-1:])


def test_wrapper_aligns_1080_canvas_before_h3_call(monkeypatch):
    calls = []
    install_fake_h3(monkeypatch, calls)
    THSUniversalH3RepairConditioning().run(
        clip=FakeClip(),
        vae=object(),
        prompt="repair",
        negative_prompt="",
        width=1080,
        height=1920,
        length=119,
    )
    assert (calls[0]["width"], calls[0]["height"], calls[0]["length"]) == (1088, 1920, 119)


def test_wrapper_rejects_empty_canvas_before_h3_call(monkeypatch):
    calls = []
    install_fake_h3(monkeypatch, calls)
    with pytest.raises(ValueError, match="must be positive"):
        THSUniversalH3RepairConditioning().run(
            clip=FakeClip(),
            vae=object(),
            prompt="repair",
            negative_prompt="",
            width=0,
            height=32,
            length=5,
        )
    assert not calls
