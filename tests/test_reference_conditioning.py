from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from ths_universal_inpaint.repair_nodes import (  # noqa: E402
    DEFAULT_CORRECTED_FRAME_PROMPT,
    REFERENCE_NONE,
    THSUniversalH3PadSource,
    THSUniversalH3ReferenceConditioning,
    THSUniversalReferenceImage,
    composite_exact,
)


def test_registered_nodes_expose_comfy_server_and_executor_contract():
    from ths_universal_inpaint import NODE_CLASS_MAPPINGS

    # Comfy server.node_info reads these attributes before any run method is
    # called. Direct wrapper tests alone cannot catch registration failures.
    for name, node_class in NODE_CLASS_MAPPINGS.items():
        assert isinstance(node_class.RETURN_TYPES, tuple), name
        assert callable(getattr(node_class, node_class.FUNCTION)), name
        assert callable(node_class.INPUT_TYPES), name
        assert isinstance(node_class.CATEGORY, str), name
        if hasattr(node_class, "RETURN_NAMES"):
            assert len(node_class.RETURN_NAMES) == len(node_class.RETURN_TYPES), name


class FakeClip:
    def __init__(self):
        self.token_calls = []

    def tokenize(self, prompt, images=None):
        self.token_calls.append((prompt, images))
        return {"prompt": prompt, "images": images}

    def encode_from_tokens_scheduled(self, tokens):
        return ("encoded", tokens)


def install_fake_h3(monkeypatch):
    image_calls = []
    reference_calls = []
    package = types.ModuleType("comfy_extras")
    package.__path__ = []
    module = types.ModuleType("comfy_extras.nodes_minimax_h3")

    class FakeImageToVideo:
        @classmethod
        def execute(cls, **kwargs):
            image_calls.append(kwargs)
            return types.SimpleNamespace(result=("positive-image", {"samples": "image"}))

    class FakeReferenceToVideo:
        @classmethod
        def execute(cls, **kwargs):
            reference_calls.append(kwargs)
            return types.SimpleNamespace(result=("positive-reference", {"samples": "reference"}))

    module.MiniMaxH3ImageToVideo = FakeImageToVideo
    module.MiniMaxH3ReferenceToVideo = FakeReferenceToVideo
    monkeypatch.setitem(sys.modules, "comfy_extras", package)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_minimax_h3", module)
    return image_calls, reference_calls


def _run_conditioner(conditioner, *, mode, reference_image=None, prompt="repair this masked region"):
    return conditioner.run(
        clip=FakeClip(),
        vae=object(),
        prompt=prompt,
        negative_prompt="do not change unmasked pixels",
        width=32,
        height=32,
        length=5,
        mode=mode,
        ref_image_size="match",
        reference_image=reference_image,
    )


def test_off_calls_original_h3_without_ref_images_or_image_decode(monkeypatch):
    image_calls, reference_calls = install_fake_h3(monkeypatch)
    image = np.zeros((1, 32, 32, 3), dtype=np.float32)

    positive, negative, latent = _run_conditioner(
        THSUniversalH3ReferenceConditioning(),
        mode="without_reference",
        reference_image=image,
        prompt="keep this exact prompt",
    )

    assert positive == "positive-image"
    assert negative[0] == "encoded"
    assert latent == {"samples": "image"}
    assert not reference_calls
    assert len(image_calls) == 1
    assert image_calls[0]["prompt"] == "keep this exact prompt"
    assert image_calls[0]["first_frame"] is None
    assert image_calls[0]["last_frame"] is None
    assert "ref_images" not in image_calls[0]

    # The upload-aware node must not touch either Comfy loader or a missing path
    # when OFF.  A loader that raises makes accidental imports/decode obvious.
    class RaisingLoadImage:
        def __init__(self):
            raise AssertionError("OFF path attempted image decode")

    monkeypatch.setitem(sys.modules, "nodes", types.SimpleNamespace(LoadImage=RaisingLoadImage))
    assert THSUniversalReferenceImage().run("without_reference", "missing.png") == (None, "without_reference", "context")


def test_on_calls_native_reference_with_tensor_vae_and_picture_label(monkeypatch):
    image_calls, reference_calls = install_fake_h3(monkeypatch)
    reference = np.full((1, 24, 20, 3), 0.8, dtype=np.float32)
    vae = object()

    class CaptureClip(FakeClip):
        pass

    clip = CaptureClip()
    result = THSUniversalH3ReferenceConditioning().run(
        clip=clip,
        vae=vae,
        prompt="replace the masked coat with the reference appearance",
        negative_prompt="unmasked edits",
        width=32,
        height=32,
        length=5,
        mode="with_reference",
        ref_image_size="max",
        reference_image=reference,
    )

    assert not image_calls
    assert len(reference_calls) == 1
    call = reference_calls[0]
    assert call["vae"] is vae
    assert call["ref_image_size"] == "max"
    assert call["ref_images"] == {"ref_image_1": reference}
    assert "<Picture 1>" in call["prompt"]
    assert "inside the user's repair mask" in call["prompt"]
    assert "may change the source colors" in call["prompt"]
    assert result[0] == "positive-reference"
    assert result[2] == {"samples": "reference"}


def test_corrected_frame_uses_native_image_to_video_and_generic_guide_prompt(monkeypatch):
    image_calls, reference_calls = install_fake_h3(monkeypatch)
    reference = np.full((1, 24, 20, 3), 0.8, dtype=np.float32)

    result = THSUniversalH3ReferenceConditioning().run(
        clip=FakeClip(),
        vae=object(),
        prompt="a source caption that must not become the guide prompt",
        negative_prompt="unmasked edits",
        width=32,
        height=32,
        length=5,
        mode="corrected_frame",
        ref_image_size="max",
        reference_prompt="Use <Picture 1>; this must not be used for corrected_frame",
        reference_image=reference,
    )

    assert not reference_calls
    assert len(image_calls) == 1
    call = image_calls[0]
    assert call["prompt"] == DEFAULT_CORRECTED_FRAME_PROMPT
    assert "<Picture 1>" not in call["prompt"]
    assert "source caption" not in call["prompt"]
    assert call["first_frame"] is None
    assert call["last_frame"] is None
    assert result[0] == "positive-image"


def test_on_missing_reference_is_clear_and_off_validator_accepts_sentinel(monkeypatch):
    install_fake_h3(monkeypatch)
    assert THSUniversalReferenceImage.VALIDATE_INPUTS("without_reference", REFERENCE_NONE) is True
    assert "required" in str(THSUniversalReferenceImage.VALIDATE_INPUTS("with_reference", REFERENCE_NONE))
    assert THSUniversalH3ReferenceConditioning.VALIDATE_INPUTS(
        "with_reference", "match", reference_image=None
    ) is True
    with pytest.raises(ValueError, match="reference_image is required"):
        _run_conditioner(THSUniversalH3ReferenceConditioning(), mode="with_reference")
    with pytest.raises(ValueError, match="reference_image is required"):
        _run_conditioner(THSUniversalH3ReferenceConditioning(), mode="with_reference", reference_image=[])


def test_on_loader_reuses_bounded_native_load_image(monkeypatch):
    image = np.zeros((1, 8, 8, 3), dtype=np.float32)
    paths = []
    folder_paths = types.SimpleNamespace(
        exists_annotated_filepath=lambda value: value == "upload/ref.png",
        get_input_directory=lambda: ".",
    )

    class FakeLoadImage:
        def load_image(self, value):
            paths.append(value)
            return image, "alpha-mask"

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setitem(sys.modules, "nodes", types.SimpleNamespace(LoadImage=FakeLoadImage))
    selected = THSUniversalReferenceImage().run("with_reference", "upload/ref.png", "Без заливки")
    assert selected[0] is image
    assert selected[1:] == ("with_reference", "none")
    assert paths == ["upload/ref.png"]


def test_reference_loader_schema_exposes_upload_and_sentinel_without_io(monkeypatch):
    # Keep schema discovery independent from a real Comfy input directory.
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(get_input_directory=lambda: "does-not-exist"))
    schema = THSUniversalReferenceImage.INPUT_TYPES()["required"]
    choices, options = schema["reference_image"]
    assert choices[0] == REFERENCE_NONE
    assert options["image_upload"] is True
    assert options["default"] == REFERENCE_NONE
    assert schema["mode"][0] == ["without_reference", "with_reference", "corrected_frame"]


def test_reference_loader_cache_key_is_stable_off_and_delegates_on(monkeypatch):
    assert THSUniversalReferenceImage.IS_CHANGED("without_reference", "missing.png") == "without_reference"
    seen = []

    class FakeLoadImage:
        @classmethod
        def IS_CHANGED(cls, value):
            seen.append(value)
            return "file-hash"

    monkeypatch.setitem(sys.modules, "nodes", types.SimpleNamespace(LoadImage=FakeLoadImage))
    assert THSUniversalReferenceImage.IS_CHANGED("with_reference", "input/ref.png") == "file-hash"
    assert seen == ["input/ref.png"]


def test_reference_fill_output_matches_pad_combo_and_keeps_legacy_default():
    fill_input = THSUniversalH3PadSource.INPUT_TYPES()["optional"]["fill_strategy"]
    assert THSUniversalReferenceImage.RETURN_TYPES[2] == fill_input[0]
    assert fill_input[1]["default"] == "context"


def test_reference_graph_keeps_exact_outside_mask_composite_contract():
    source = np.array([[[[0.1234567, 0.2222222, 0.3333333], [0.8765432, 0.1111111, 0.9999999]]]], dtype=np.float32)
    generated = np.ones_like(source)
    mask = np.array([[[0.0, 1.0]]], dtype=np.float32)
    output = composite_exact(source, generated, mask)
    np.testing.assert_array_equal(output[0, 0, 0], source[0, 0, 0])
    np.testing.assert_array_equal(output[0, 0, 1], generated[0, 0, 1])
