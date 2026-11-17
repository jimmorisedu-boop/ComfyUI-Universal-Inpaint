from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import setup as setup_tool  # noqa: E402


def _comfy(tmp_path: Path) -> Path:
    root = tmp_path / "ComfyUI"
    (root / "models").mkdir(parents=True)
    (root / "main.py").write_text("# fixture\n", encoding="utf-8")
    return root


def _manifest() -> dict:
    return {
        "version": setup_tool.VERSION,
        "models": [
            {"id": "public", "profiles": ["editor"], "repo": "org/repo", "revision": "abc", "filename": "weights.bin", "destination": "models/public/weights.bin", "size_bytes": 3, "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "gated": False, "source": "https://huggingface.co/org/repo"},
            {"id": "gated", "profiles": ["sam3"], "repo": "org/gated", "revision": "def", "filename": "sam3.safetensors", "destination": "models/sam3/sam3.safetensors", "size_bytes": None, "sha256": None, "gated": True, "source": "https://huggingface.co/org/gated"},
        ],
    }


def test_model_plan_reports_disk_and_existing_state(tmp_path: Path) -> None:
    root = _comfy(tmp_path)
    plan = setup_tool.build_download_plan(comfy_root=root, model_ids=["public"], manifest=_manifest())
    assert plan["models"][0]["status"] == "missing"
    assert plan["known_bytes_to_download"] == 3
    assert plan["free_bytes"] > 0

    destination = root / "models" / "public" / "weights.bin"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"abc")
    again = setup_tool.build_download_plan(comfy_root=root, model_ids=["public"], manifest=_manifest())
    assert again["models"][0]["status"] == "already_present"


def test_gated_download_requires_explicit_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _comfy(tmp_path)
    plan = setup_tool.build_download_plan(comfy_root=root, model_ids=["gated"], manifest=_manifest())
    with pytest.raises(setup_tool.SetupError, match="gated"):
        setup_tool.apply_download_plan(plan, manifest=_manifest(), allow_gated=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(setup_tool, "_hf_token", lambda: None)
    with pytest.raises(setup_tool.SetupError, match="HF_TOKEN"):
        setup_tool.apply_download_plan(plan, manifest=_manifest(), allow_gated=True)


def test_download_checksum_and_resume_cache_are_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _comfy(tmp_path)
    cache = tmp_path / "cache"
    model = _manifest()["models"][0]
    monkeypatch.setattr(setup_tool, "_hf_token", lambda: None)
    plan = setup_tool.build_download_plan(comfy_root=root, model_ids=["public"], cache_dir=cache, manifest=_manifest())

    class Response:
        status = 200
        def read(self, _size: int = -1) -> bytes:
            if getattr(self, "used", False):
                return b""
            self.used = True
            return b"abc"
        def __enter__(self):
            self.used = False
            return self
        def __exit__(self, *_args):
            return False
        def close(self):
            pass
        def getcode(self):
            return 200

    monkeypatch.setattr(setup_tool.urllib.request, "urlopen", lambda request, timeout=0: Response())
    applied = setup_tool.apply_download_plan(plan, manifest=_manifest(), allow_gated=False)
    assert applied["applied"][0]["status"] == "downloaded"
    assert (root / "models/public/weights.bin").read_bytes() == b"abc"

    # A conflicting existing file is never overwritten, even if a download is requested again.
    (root / "models/public/weights.bin").write_bytes(b"xyz")
    with pytest.raises(setup_tool.SetupError, match="overwrite"):
        setup_tool._download_one(root, cache, model, allow_gated=False)


def test_resume_requires_matching_content_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _comfy(tmp_path)
    cache = tmp_path / "cache"
    model = _manifest()["models"][0]
    partial = setup_tool._cache_path(cache, model)
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"a")
    monkeypatch.setattr(setup_tool, "_hf_token", lambda: None)

    class Response:
        status = 206
        headers = {"Content-Range": "bytes 0-2/3"}
        def read(self, _size: int = -1) -> bytes:
            return b"bc"
        def close(self):
            pass
    monkeypatch.setattr(setup_tool.urllib.request, "urlopen", lambda request, timeout=0: Response())
    with pytest.raises(setup_tool.SetupError, match="starts at"):
        setup_tool._download_one(root, cache, model, allow_gated=False)
    assert partial.read_bytes() == b"a"


def test_model_redirect_never_forwards_token_to_cdn():
    request = setup_tool.urllib.request.Request(
        "https://huggingface.co/org/repo/resolve/revision/model",
        headers={"Authorization": "Bearer fixture-token"},
    )
    handler = setup_tool._HFRedirectHandler()
    redirected = handler.redirect_request(request, None, 302, "Found", {}, "https://cdn.example.invalid/model")
    assert redirected.get_header("Authorization") is None
    with pytest.raises(setup_tool.SetupError, match="HTTPS"):
        handler.redirect_request(request, None, 302, "Found", {}, "http://huggingface.co/model")


def test_publish_preserves_existing_destination(tmp_path: Path):
    source, destination = tmp_path / "temporary", tmp_path / "model"
    source.write_bytes(b"new")
    destination.write_bytes(b"existing")
    with pytest.raises(setup_tool.SetupError, match="overwrite"):
        setup_tool._publish_file_no_clobber(source, destination)
    assert destination.read_bytes() == b"existing"
