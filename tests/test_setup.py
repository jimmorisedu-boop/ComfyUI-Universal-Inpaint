from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import setup as setup_tool  # noqa: E402


def _comfy(tmp_path: Path) -> Path:
    root = tmp_path / "ComfyUI"
    (root / "custom_nodes").mkdir(parents=True)
    (root / "user" / "default" / "workflows").mkdir(parents=True)
    (root / "comfy_extras").mkdir()
    (root / "main.py").write_text("# fixture\n", encoding="utf-8")
    return root


def test_relative_path_rejects_traversal_and_absolute() -> None:
    with pytest.raises(setup_tool.SetupError):
        setup_tool.validate_relative_path("../outside/file")
    with pytest.raises(setup_tool.SetupError):
        setup_tool.validate_relative_path("C:/outside/file")
    with pytest.raises(setup_tool.SetupError):
        setup_tool.validate_relative_path("/absolute/file")
    for unsafe in ("C:relative/file", "..\\outside\\file", "\\\\server\\share\\file"):
        with pytest.raises(setup_tool.SetupError):
            setup_tool.validate_relative_path(unsafe)
    assert setup_tool.validate_relative_path("models\\sam3\\weights") == Path("models/sam3/weights")


def test_detect_root_and_explicit_python(tmp_path: Path) -> None:
    root = _comfy(tmp_path)
    assert setup_tool.detect_comfy_root(root) == root.resolve()
    assert setup_tool.resolve_python(root, sys.executable).resolve() == Path(sys.executable).resolve()
    with pytest.raises(setup_tool.SetupError):
        setup_tool.detect_comfy_root(tmp_path / "not-comfy")
    with pytest.raises(setup_tool.SetupError, match="explicit Python"):
        setup_tool.resolve_python(root, tmp_path / "missing-python")


def test_install_plan_is_read_only_and_idempotent(tmp_path: Path) -> None:
    root = _comfy(tmp_path)
    source = tmp_path / "repo"
    (source / "custom_nodes" / "ths_universal_inpaint").mkdir(parents=True)
    (source / "custom_nodes" / "ths_universal_inpaint" / "__init__.py").write_text("x=1\n", encoding="utf-8")
    (source / "requirements-editor.txt").write_text("numpy>=1.0\n", encoding="utf-8")
    (source / "workflows").mkdir()
    (source / "workflows" / "V70_Universal_Repair_Workbench.json").write_text("{}\n", encoding="utf-8")
    deps = {
        "version": setup_tool.VERSION,
        "profiles": {"editor": {"requirements": ["requirements-editor.txt"], "custom_nodes": [], "workflows": []}},
        "custom_nodes": {},
    }
    plan = setup_tool.build_install_plan(comfy_root=root, profile_names=["editor"], repo_root=source, dependencies=deps)
    assert plan["would_mutate"] is True
    assert plan["add_on"]["status"] == "copy"
    assert not (root / "custom_nodes" / "ComfyUI-Universal-Inpaint").exists()

    # A conflicting destination is reported before any mutation.
    destination = root / "custom_nodes" / "ComfyUI-Universal-Inpaint"
    destination.mkdir()
    (destination / "__init__.py").write_text("different\n", encoding="utf-8")
    second = setup_tool.build_install_plan(comfy_root=root, profile_names=["editor"], repo_root=source, dependencies=deps)
    assert "conflict_no_overwrite" in second["errors"][0]


def test_existing_custom_node_wrong_remote_is_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _comfy(tmp_path)
    source = tmp_path / "repo"
    (source / "custom_nodes" / "ths_universal_inpaint").mkdir(parents=True)
    (source / "requirements-editor.txt").write_text("numpy\n", encoding="utf-8")
    node = root / "custom_nodes" / "fake-node"
    node.mkdir()
    deps = {
        "version": setup_tool.VERSION,
        "profiles": {"editor": {"requirements": ["requirements-editor.txt"], "custom_nodes": ["fake"], "workflows": []}},
        "custom_nodes": {"fake": {"destination": "custom_nodes/fake-node", "repo": "https://example.invalid/right.git", "expected_remote": "https://example.invalid/right.git", "revision": "abc"}},
    }
    def fake_git(_destination: Path, *args: str) -> str | None:
        if args[:3] == ("remote", "get-url", "origin"):
            return "https://example.invalid/wrong.git"
        return "abc"
    monkeypatch.setattr(setup_tool, "_git_output", fake_git)
    plan = setup_tool.build_install_plan(comfy_root=root, profile_names=["editor"], repo_root=source, dependencies=deps)
    assert any("wrong_remote" in error for error in plan["errors"])


def test_legacy_addon_duplicate_is_refused_without_deletion(tmp_path: Path) -> None:
    root = _comfy(tmp_path)
    legacy = root / "custom_nodes" / "ths_universal_inpaint"
    legacy.mkdir()
    (legacy / "old.py").write_text("legacy\n", encoding="utf-8")
    source = tmp_path / "repo"
    (source / "custom_nodes" / "ths_universal_inpaint").mkdir(parents=True)
    (source / "custom_nodes" / "ths_universal_inpaint" / "__init__.py").write_text("x=1\n", encoding="utf-8")
    (source / "requirements-editor.txt").write_text("numpy\n", encoding="utf-8")
    deps = {"version": setup_tool.VERSION, "profiles": {"editor": {"requirements": ["requirements-editor.txt"], "custom_nodes": [], "workflows": []}}, "custom_nodes": {}}
    plan = setup_tool.build_install_plan(comfy_root=root, profile_names=["editor"], repo_root=source, dependencies=deps)
    assert any("legacy add-on" in error for error in plan["errors"])
    assert (legacy / "old.py").read_text(encoding="utf-8") == "legacy\n"


def test_install_never_passes_torch_or_upgrade_to_pip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _comfy(tmp_path)
    source = tmp_path / "repo"
    addon = source / "custom_nodes" / "ths_universal_inpaint"
    addon.mkdir(parents=True)
    (addon / "__init__.py").write_text("x=1\n", encoding="utf-8")
    (source / "requirements-editor.txt").write_text("numpy>=1.0\n", encoding="utf-8")
    deps = {"version": setup_tool.VERSION, "profiles": {"editor": {"requirements": ["requirements-editor.txt"], "custom_nodes": [], "workflows": []}}, "custom_nodes": {}}
    plan = setup_tool.build_install_plan(comfy_root=root, profile_names=["editor"], repo_root=source, dependencies=deps)
    calls: list[list[str]] = []
    def fake_run(argv, **kwargs):
        values = [str(x) for x in argv]
        calls.append(values)
        if "--report" in values:
            report = Path(values[values.index("--report") + 1])
            report.write_text('{"install": []}', encoding="utf-8")
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(setup_tool, "_run", fake_run)
    result = setup_tool.apply_install_plan(plan, python=Path(sys.executable))
    assert result["mode"] == "apply"
    pip_calls = [call for call in calls if "pip" in call]
    assert pip_calls
    assert any("--dry-run" in call and "--report" in call for call in pip_calls)
    assert all("--upgrade" not in call and "--no-deps" not in call for call in pip_calls)
    assert all(not any(token.lower().startswith("torch") for token in call) for call in pip_calls)


def test_pip_report_rejects_gpu_stack_change_before_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _comfy(tmp_path)
    source = tmp_path / "repo"
    addon = source / "custom_nodes" / "ths_universal_inpaint"
    addon.mkdir(parents=True)
    (addon / "__init__.py").write_text("x=1\n", encoding="utf-8")
    (source / "requirements-editor.txt").write_text("numpy\n", encoding="utf-8")
    deps = {"version": setup_tool.VERSION, "profiles": {"editor": {"requirements": ["requirements-editor.txt"], "custom_nodes": [], "workflows": []}}, "custom_nodes": {}}
    plan = setup_tool.build_install_plan(comfy_root=root, profile_names=["editor"], repo_root=source, dependencies=deps)
    def fake_run(argv, **kwargs):
        values = [str(x) for x in argv]
        if "--report" in values:
            report = Path(values[values.index("--report") + 1])
            report.write_text('{"install": [{"metadata": {"name": "torch", "version": "99.0"}}]}', encoding="utf-8")
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(setup_tool, "_run", fake_run)
    with pytest.raises(setup_tool.SetupError, match="protected GPU"):
        setup_tool.apply_install_plan(plan, python=Path(sys.executable))
    assert not (root / "custom_nodes" / "ComfyUI-Universal-Inpaint").exists()


def test_check_rejects_unsupported_python_and_missing_h3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _comfy(tmp_path)
    def fake_run(argv, **kwargs):
        values = [str(x) for x in argv]
        if "-c" in values and "version_info" not in values:
            return type("R", (), {"returncode": 0, "stdout": "3.9.18\n3.9.18", "stderr": ""})()
        return type("R", (), {"returncode": 1, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(setup_tool, "_run", fake_run)
    result = setup_tool.check_runtime(comfy_root=root, python=Path(sys.executable), profiles=["h3"])
    assert result["python_supported"] is False
    assert result["status"] == "missing_capabilities"


def test_pip_installs_complete_hashed_transaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requirement = tmp_path / "requirements.txt"
    requirement.write_text("parent-package\n", encoding="utf-8")
    installed = []
    monkeypatch.setattr(setup_tool, "_python_has_module", lambda *_args: False)

    def fake_run(argv, **kwargs):
        argv = list(map(str, argv))
        if "--report" in argv:
            artifacts = [{"metadata": {"name": name}, "download_info": {
                "url": f"https://example.invalid/{name}-1.0.whl",
                "archive_info": {"hashes": {"sha256": "a" * 64}}
            }} for name in ("parent-package", "transitive-dependency")]
            Path(argv[argv.index("--report") + 1]).write_text(json.dumps({"install": artifacts}), encoding="utf-8")
        else:
            assert "--no-deps" in argv and "--require-hashes" in argv
            installed.extend(Path(argv[argv.index("-r") + 1]).read_text(encoding="utf-8").splitlines())
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(setup_tool, "_run", fake_run)
    result = setup_tool._pip_plan_and_install(Path(sys.executable), requirement)
    assert result["status"] == "installed"
    assert len(installed) == 2
    assert any(line.startswith("transitive-dependency @ ") for line in installed)
    assert all("--hash=sha256:" in line for line in installed)


def test_copy_ignores_local_metadata_and_remains_idempotent(tmp_path: Path) -> None:
    source, destination = tmp_path / "repo", tmp_path / "installed"
    source.mkdir()
    (source / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    (source / ".env").write_text("fixture-secret", encoding="utf-8")
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("fixture", encoding="utf-8")
    assert setup_tool._copy_tree_if_absent(source, destination) == "copied"
    assert not (destination / ".env").exists()
    assert not (destination / ".git").exists()
    (destination / "__pycache__").mkdir()
    (destination / "__pycache__" / "module.pyc").write_bytes(b"cache")
    assert setup_tool._copy_tree_if_absent(source, destination) == "already_identical"


def test_check_cli_reports_failure(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(setup_tool, "detect_comfy_root", lambda *_args: None)
    assert setup_tool.main(["check", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "unknown"
