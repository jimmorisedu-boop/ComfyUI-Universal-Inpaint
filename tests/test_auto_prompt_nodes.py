from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ths_universal_inpaint import auto_prompt_nodes as apn  # noqa: E402
from ths_universal_inpaint.auto_prompt_core import DEFAULT_REPAIR_PROMPT, compose_edit_instruction_prompt  # noqa: E402


def _fixture():
    images = np.zeros((3, 32, 32, 3), dtype=np.float32)
    images[..., 0] = 0.2
    images[..., 1] = np.linspace(0.1, 0.8, 32, dtype=np.float32)[None, :, None]
    mask = np.zeros((3, 32, 32), dtype=np.float32)
    mask[:, 12:20, 12:20] = 1.0
    return images, mask


class _FakeBackend:
    def __init__(self, captions=("a person in a red coat",)):
        self.captions = list(captions)
        self.calls = []
        self.closed = False

    def caption(self, clips):
        self.calls.append(list(clips))
        return list(self.captions)

    def close(self):
        self.closed = True


def test_node_contract_has_mask_aware_auto_and_manual_modes():
    inputs = apn.THSUniversalAutoPrompt.INPUT_TYPES()
    required = inputs["required"]
    assert required["images"][0] == "IMAGE"
    assert required["mask"][0] == "MASK"
    assert required["mode"][0] == ["instruction", "auto", "manual"]
    assert required["mode"][1]["default"] == "instruction"
    assert required["user_hint"][1]["default"] == ""
    assert required["manual_prompt"][1]["default"] == ""
    assert inputs["optional"]["max_frames"][1]["default"] == 3
    assert inputs["optional"]["context_padding"][1]["default"] == 0.6
    assert apn.THSUniversalAutoPrompt.RETURN_TYPES == ("STRING", "STRING", "IMAGE")
    assert apn.THSUniversalAutoPrompt.RETURN_NAMES == ("prompt", "caption", "preview")
    assert apn.THSUniversalAutoPrompt.OUTPUT_NODE is True


def test_instruction_uses_translation_as_the_intended_edit_without_captioning(monkeypatch, tmp_path):
    def forbidden(*args):
        raise AssertionError("an explicit edit must not trigger Florence")

    monkeypatch.setattr(apn, "_make_backend", forbidden)
    monkeypatch.setattr(apn, "_local_translation_path", lambda: tmp_path)
    monkeypatch.setattr(apn, "_translate_locally", lambda path, text: "Make the hair gray")
    node = apn.THSUniversalAutoPrompt()
    images = np.zeros((2, 4, 4, 3), dtype=np.float32)
    mask = np.ones((2, 4, 4), dtype=np.float32)
    task = "Сделай волосы седыми"
    response = node.run(images, mask, mode="instruction", user_hint=task, manual_prompt="old text", preview_enabled=False)
    prompt, caption, preview = response["result"]
    assert prompt.startswith("Make the hair gray.")
    assert "outside the mask" in prompt and "old text" not in prompt
    assert caption == "" and tuple(preview.shape) == (1, 1, 1, 3)
    assert response["ui"]["auto_prompt_instruction"] == [task]
    monkeypatch.setattr(apn, "_local_translation_path", forbidden)
    english = node.run(images, mask, mode="instruction", user_hint='Write "[2026]" on the sign', preview_enabled=False)
    assert 'Write "[2026]" on the sign' in english["result"][0]
    with pytest.raises(ValueError, match="Что изменить"):
        node.run(images, mask, mode="instruction", user_hint="  ")


@pytest.mark.parametrize("repair_mode", ["with_reference", "corrected_frame"])
def test_image_repair_mode_runs_with_no_instruction_or_caption_model(repair_mode, monkeypatch):
    def forbidden(*args):
        pytest.fail("image-guided repair tried to generate an AutoPrompt")

    monkeypatch.setattr(apn, "_instruction_prompt", forbidden)
    monkeypatch.setattr(apn, "_make_backend", forbidden)
    images, mask = _fixture()
    for prompt_mode in ("instruction", "auto", "manual"):
        prompt, caption, preview = apn.THSUniversalAutoPrompt().run(
            images, mask, mode=prompt_mode, user_hint="", repair_mode=repair_mode,
        )
        assert prompt == caption == ""
        assert tuple(preview.shape) == (1, 1, 1, 3)


@pytest.mark.parametrize("edit, start, detail", [
    ("Change tie to black bow tie", "Replace the necktie with a black bow tie at the shirt collar.", "moving with the wearer"),
    ("Change the door to the wall.", "Replace the door with a continuous section of wall.", "adjacent wall's surface"),
    ("Add a butterfly that flies past the tie.", "Add a butterfly that flies past the tie.", "Match the source lighting"),
])
def test_h3_edit_prompt_describes_visible_change_and_source_continuity(edit, start, detail):
    prompt = compose_edit_instruction_prompt(edit)
    assert prompt.startswith(start)
    assert detail in prompt
    assert "from frame to frame" in prompt and "outside the mask" in prompt
    assert "<Video 1>" not in prompt


def test_auto_mode_calls_injected_backend_and_caches_only_strings(monkeypatch):
    images, mask = _fixture()
    backend = _FakeBackend()
    apn.clear_caption_cache()
    monkeypatch.setattr(apn, "_caption_backend_factory", lambda: backend)

    first = apn.THSUniversalAutoPrompt().run(images, mask, mode="auto")
    assert first["result"][1] == "a person in a red coat"
    assert "Observed source appearance: a person in a red coat." in first["result"][0]
    assert "auto: Florence-2 PromptGen captioned 1 context frame(s)." in first["ui"]["auto_prompt_status"][0]
    assert first["ui"]["auto_prompt_text"] == [first["result"][0]]
    assert first["ui"]["auto_prompt_caption"] == [first["result"][1]]
    assert first["result"][2].shape == images.shape
    assert len(backend.calls) == 1
    assert len(backend.calls[0]) == 3
    assert backend.closed is True

    # Same image/mask/model parameters hit the bounded string cache; the
    # backend is not loaded again and no model object is retained there.
    second = apn.THSUniversalAutoPrompt().run(images, mask, mode="auto")
    assert second["result"][0] == first["result"][0]
    assert len(backend.calls) == 1
    assert len(apn._caption_cache) == 1
    assert all(isinstance(key, tuple) and isinstance(value[0], tuple) for key, value in apn._caption_cache.items())
    assert all(all(isinstance(text, str) for text in value[0]) for value in apn._caption_cache.values())


def test_manual_mode_skips_vlm_and_uses_generic_fallback_when_blank(monkeypatch):
    images, mask = _fixture()

    def fail_backend():
        raise AssertionError("manual mode must not construct a VLM")

    monkeypatch.setattr(apn, "_caption_backend_factory", fail_backend)
    result = apn.THSUniversalAutoPrompt().run(
        images,
        mask,
        mode="manual",
        user_hint="keep the existing blue fabric",
        manual_prompt="",
    )
    assert result["result"][1] == ""
    assert "existing blue fabric" in result["result"][0]
    assert "manual: local VLM skipped" in result["ui"]["auto_prompt_status"][0]

    blank = apn.THSUniversalAutoPrompt().run(images, mask, mode="manual")
    assert blank["result"][0].startswith(DEFAULT_REPAIR_PROMPT)
    assert "generic masked-repair fallback" in blank["ui"]["auto_prompt_status"][0]


def test_auto_vlm_error_is_explicit_generic_fallback(monkeypatch):
    images, mask = _fixture()
    apn.clear_caption_cache()

    def broken_backend():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(apn, "_caption_backend_factory", broken_backend)
    result = apn.THSUniversalAutoPrompt().run(images, mask, mode="auto")
    assert result["result"][1] == ""
    assert result["result"][0].startswith(DEFAULT_REPAIR_PROMPT)
    status = result["ui"]["auto_prompt_status"][0]
    assert "generic masked-repair fallback" in status
    assert "no semantic caption was accepted" in status
    assert "Florence-2 captioning unavailable" in status


def test_native_florence_adapter_uses_registered_loader_and_run_methods(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    model_path = tmp_path / apn.MODEL_NAME
    model_path.mkdir()
    for filename in ("config.json", "model.safetensors", "tokenizer.json"):
        (model_path / filename).write_bytes(b"local")

    calls = {}

    class FakeModel:
        def __init__(self):
            self.to_calls = []

        def to(self, device):
            self.to_calls.append(device)
            return self

        def cpu(self):
            self.to_calls.append("cpu")
            return self

    fake_model = FakeModel()

    class Loader:
        def loadmodel(self, *args, **kwargs):
            calls["loader"] = (args, kwargs)
            return ({"model": fake_model},)

    class Runner:
        def encode(self, **kwargs):
            calls["encode"] = kwargs
            return (None, None, "a textured garment", None)

    class MM:
        def __init__(self):
            self.empty = 0

        def soft_empty_cache(self):
            self.empty += 1

    mm = MM()
    fake_module = type("FlorenceModule", (), {
        "Florence2ModelLoader": Loader,
        "Florence2Run": Runner,
        "offload_device": "cpu",
        "mm": mm,
    })
    monkeypatch.setattr(apn, "_local_model_path", lambda: model_path)
    monkeypatch.setattr(apn, "_florence_nodes_module", lambda: fake_module)

    backend = apn._FlorenceCaptionBackend()
    try:
        assert backend.caption([np.zeros((12, 12, 3), dtype=np.float32)]) == ["a textured garment"]
        assert calls["loader"][0][:3] == (apn.MODEL_NAME, apn.MODEL_PRECISION, apn.MODEL_ATTENTION)
        assert calls["loader"][1] == {"lora": None, "convert_to_safetensors": False}
        assert calls["encode"]["task"] == apn.CAPTION_TASK
        assert calls["encode"]["keep_model_loaded"] is False
        assert calls["encode"]["fill_mask"] is False
    finally:
        backend.close()
    assert mm.empty >= 1
    assert fake_model.to_calls


def test_cache_is_bounded(monkeypatch):
    apn.clear_caption_cache()
    monkeypatch.setattr(apn, "_caption_backend_factory", lambda: _FakeBackend(("cloth",)))
    for index in range(apn.MAX_CACHE_ENTRIES + 3):
        images, mask = _fixture()
        images[0, 0, 0, 0] = float(index) / 10.0
        apn.THSUniversalAutoPrompt().run(images, mask, mode="auto")
    assert len(apn._caption_cache) <= apn.MAX_CACHE_ENTRIES
