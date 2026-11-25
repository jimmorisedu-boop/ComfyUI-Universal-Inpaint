#!/usr/bin/env python3
"""Plan and safely install Universal Inpaint into an existing ComfyUI tree.

The command is deliberately conservative:

* ``plan`` and ``check`` are read-only (the default when no command is given).
* ``install`` and ``download`` mutate only with an explicit ``--apply``.
* an existing custom-node checkout must have the exact expected remote and
  pinned revision; an existing workflow/model is never overwritten.
* dependency installation uses a pip ``--dry-run --report`` guard and rejects
  any proposed torch/CUDA/vision package change before the real resolver runs.
  The selected ComfyUI runtime remains authoritative.
* model files are downloaded from a pinned Hugging Face revision into a
  resumable cache, checked for size/hash, then copied only to a new destination.

This module uses only the Python standard library so it can run with system
Python, a ComfyUI portable ``python_embeded`` interpreter, or a virtualenv.
It never imports ComfyUI, torch, or custom nodes during planning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request


VERSION = "0.2.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES_PATH = REPO_ROOT / "config" / "dependencies.json"
MODELS_PATH = REPO_ROOT / "config" / "models.json"
HF_BASE_URL = "https://huggingface.co"
# Packages whose binary/CUDA builds belong to the user's ComfyUI runtime.  A
# pip resolver report that proposes any of these (including an nvidia-* wheel)
# is rejected before the real install starts.
PROTECTED_GPU_PREFIXES = ("torch", "torchvision", "torchaudio", "xformers", "triton", "nvidia-")
COPY_IGNORED = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "build", "dist", "acceptance", "logs", "models", "input", "output", ".secrets", "credentials"}


class SetupError(RuntimeError):
    """A guarded setup precondition failed."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SetupError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SetupError(f"manifest {path} must contain a JSON object")
    return value


def load_dependencies(path: Path = DEPENDENCIES_PATH) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("version") != VERSION:
        raise SetupError(f"unsupported dependency manifest version: {value.get('version')!r}")
    profiles = value.get("profiles")
    nodes = value.get("custom_nodes")
    if not isinstance(profiles, dict) or not isinstance(nodes, dict):
        raise SetupError("dependency manifest requires profiles and custom_nodes objects")
    return value


def load_models(path: Path = MODELS_PATH) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("version") != VERSION:
        raise SetupError(f"unsupported model manifest version: {value.get('version')!r}")
    if not isinstance(value.get("models"), list):
        raise SetupError("model manifest requires a models list")
    return value


def _resolved(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=False)
    except OSError:
        return Path(os.path.abspath(os.fspath(path)))


def _is_within(path: Path, root: Path) -> bool:
    try:
        _resolved(path).relative_to(_resolved(root))
        return True
    except ValueError:
        return False


def validate_relative_path(relative: str | os.PathLike[str], *, label: str = "path") -> Path:
    """Reject absolute paths, drive-qualified paths and ``..`` traversal."""

    raw = os.fspath(relative)
    candidate = Path(raw)
    # On Windows ``Path('/foo').is_absolute()`` is false because a drive is
    # missing, but it is still rooted and must not be accepted as a relative
    # destination.  Check the raw spelling as well as pathlib's platform
    # interpretation.
    normalized_parts = [part for part in raw.replace("\\", "/").split("/") if part]
    if not raw or candidate.is_absolute() or PureWindowsPath(raw).drive or raw.startswith(("/", "\\")):
        raise SetupError(f"{label} must be a relative path: {raw!r}")
    if any(part in (".", "..") for part in (*candidate.parts, *normalized_parts)):
        raise SetupError(f"{label} contains unsafe traversal: {raw!r}")
    return Path(*normalized_parts)


def safe_destination(root: Path, relative: str | os.PathLike[str], *, label: str = "destination") -> Path:
    rel = validate_relative_path(relative, label=label)
    destination = _resolved(root / rel)
    if not _is_within(destination, root):
        raise SetupError(f"{label} escapes selected root: {relative!r}")
    return destination


def _looks_like_comfy_root(path: Path) -> bool:
    return path.is_dir() and ((path / "main.py").is_file() or (path / "comfy").is_dir())


def detect_comfy_root(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """Resolve a ComfyUI root without scanning arbitrary drives."""

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for name in ("COMFYUI_PATH", "COMFY_BASE_PATH"):
        raw = os.environ.get(name)
        if raw:
            candidates.append(Path(raw))
    cwd = Path.cwd()
    candidates.extend((cwd, cwd / "ComfyUI"))
    for candidate in candidates:
        resolved = _resolved(candidate)
        if _looks_like_comfy_root(resolved):
            return resolved
    if explicit:
        raise SetupError(f"selected ComfyUI root is not a ComfyUI tree: {_resolved(Path(explicit))}")
    return None


def resolve_python(comfy_root: Path | None, explicit: str | os.PathLike[str] | None = None) -> Path:
    """Find portable embedded Python, venv Python, or the current interpreter."""

    candidates: list[Path] = []
    if explicit:
        selected = _resolved(Path(explicit))
        if not selected.is_file() or not os.access(selected, os.X_OK):
            raise SetupError(f"explicit Python interpreter is not usable: {selected}")
        return selected
    if comfy_root:
        root = _resolved(comfy_root)
        candidates.extend(
            (
                root / "python_embeded" / "python.exe",
                root.parent / "python_embeded" / "python.exe",
                root / ".venv" / "Scripts" / "python.exe",
                root / ".venv" / "bin" / "python",
                root.parent / ".venv" / "Scripts" / "python.exe",
                root.parent / ".venv" / "bin" / "python",
            )
        )
    candidates.append(Path(sys.executable))
    seen: set[str] = set()
    for candidate in candidates:
        resolved = _resolved(candidate)
        key = os.path.normcase(str(resolved))
        if key in seen:
            continue
        seen.add(key)
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    requested = str(explicit) if explicit else "the selected ComfyUI root"
    raise SetupError(f"could not locate a usable Python interpreter for {requested}")


def _run(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout: float | None = 120.0,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            [str(item) for item in argv],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SetupError(f"command failed to start: {' '.join(map(str, argv))}: {exc}") from exc
    if check and result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        raise SetupError(f"command returned {result.returncode}: {' '.join(map(str, argv))}\n{detail}")
    return result


def _git_output(destination: Path, *args: str) -> str | None:
    result = _run(("git", "-C", str(destination), *args), check=False, timeout=30.0)
    if result.returncode:
        return None
    return (result.stdout or "").strip()


def _norm_remote(url: str | None) -> str:
    if not url:
        return ""
    value = url.strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.split(":", 1)[1]
    return value.lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_protected_distribution(name: str) -> bool:
    normalized = str(name).lower().replace("_", "-")
    return any(normalized == prefix.rstrip("-") or normalized.startswith(prefix) for prefix in PROTECTED_GPU_PREFIXES)


def _python_has_module(python: Path, module: str) -> bool:
    probe = _run(
        (
            str(python),
            "-c",
            "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
            module,
        ),
        check=False,
        timeout=30.0,
    )
    return probe.returncode == 0


def _filtered_requirements_file(requirement: Path, python: Path) -> tuple[Path, bool]:
    """Create a temporary requirement file when cv2 is already available.

    Installing both ``opencv-python`` and ``opencv-python-headless`` can race
    over the same ``cv2`` files.  If the selected runtime already exposes cv2,
    keep the existing provider and omit only the direct headless requirement.
    The returned temporary path is owned by this call and may be deleted after
    pip completes.
    """

    try:
        lines = requirement.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SetupError(f"cannot read requirements file {requirement}: {exc}") from exc
    if not _python_has_module(python, "cv2"):
        return requirement, False
    kept = [line for line in lines if not line.strip().lower().startswith(("opencv-python-headless", "opencv-python==", "opencv-python>="))]
    if len(kept) == len(lines):
        return requirement, False
    fd, name = tempfile.mkstemp(prefix="universal-inpaint-req-", suffix=".txt")
    os.close(fd)
    temporary = Path(name)
    temporary.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return temporary, True


def _pip_plan_and_install(python: Path, requirement: Path) -> dict[str, Any]:
    """Resolve normal Python dependencies while proving GPU packages stay put."""

    filtered, temporary = _filtered_requirements_file(requirement, python)
    report_fd, report_name = tempfile.mkstemp(prefix="universal-inpaint-pip-", suffix=".json")
    os.close(report_fd)
    report = Path(report_name)
    locked = report.with_suffix(".requirements.txt")
    try:
        _run(
            (
                str(python),
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--report",
                str(report),
                "--disable-pip-version-check",
                "--no-input",
                "-r",
                str(filtered),
            ),
            timeout=1800.0,
        )
        try:
            document = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SetupError(f"pip dry-run did not produce a valid report for {requirement}: {exc}") from exc
        if not isinstance(document, dict) or not isinstance(document.get("install"), list):
            raise SetupError(f"pip dry-run report has no install transaction for {requirement}")
        changes: list[str] = []
        for item in document.get("install", []) if isinstance(document, dict) else []:
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            name = str(metadata.get("name", ""))
            if _is_protected_distribution(name):
                changes.append(f"{name}=={metadata.get('version', '?')}")
        if changes:
            raise SetupError(
                f"refusing {requirement.name}: pip resolver proposes protected GPU/CUDA changes ({', '.join(changes)}). "
                "Install a compatible runtime manually, then rerun the plan."
            )
        # Install exactly the complete resolved transaction, including its
        # transitive dependencies. A second unconstrained resolution could
        # select new GPU packages after the safety check.
        artifacts = []
        for item in document.get("install", []):
            name = str(item.get("metadata", {}).get("name", ""))
            info = item.get("download_info", {})
            url = str(info.get("url", ""))
            sha = str(info.get("archive_info", {}).get("hashes", {}).get("sha256", ""))
            if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                    or not re.fullmatch(r"[a-fA-F0-9]{64}", sha)
                    or urllib.parse.urlsplit(url).scheme != "https"
                    or any(c.isspace() for c in url)):
                raise SetupError(f"cannot safely lock resolved package {name!r}; require a hashed HTTPS archive")
            artifacts.append(f"{name} @ {url} --hash=sha256:{sha}")
        if not artifacts:
            return {"requirement": str(requirement), "status": "already_satisfied", "opencv_skipped": bool(temporary)}
        locked.write_text("\n".join(artifacts) + "\n", encoding="utf-8")
        _run(
            (
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--require-hashes",
                "--disable-pip-version-check",
                "--no-input",
                "-r",
                str(locked),
            ),
            timeout=1800.0,
        )
        return {"requirement": str(requirement), "status": "installed", "opencv_skipped": bool(temporary)}
    finally:
        report.unlink(missing_ok=True)
        locked.unlink(missing_ok=True)
        if temporary:
            filtered.unlink(missing_ok=True)


def _tree_digest(path: Path) -> str:
    """Hash relative file names and bytes without following symlink escapes."""

    digest = hashlib.sha256()
    for item in sorted(path.rglob("*"), key=lambda p: p.relative_to(path).as_posix()):
        rel = item.relative_to(path)
        if any(part in COPY_IGNORED or part.startswith(".env") or part.endswith(".egg-info") for part in rel.parts):
            continue
        if item.is_symlink():
            digest.update(f"L:{rel.as_posix()}:{os.readlink(item)}\n".encode())
            continue
        if item.is_file():
            digest.update(f"F:{rel.as_posix()}\n".encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


def _profile_names(requested: Sequence[str] | None, dependencies: Mapping[str, Any]) -> list[str]:
    available = dependencies.get("profiles", {})
    names = list(requested or ("editor",))
    unknown = [name for name in names if name not in available]
    if unknown:
        raise SetupError(f"unknown profile(s): {', '.join(unknown)}")
    return list(dict.fromkeys(names))


def _profile_union(profile_names: Sequence[str], dependencies: Mapping[str, Any], key: str) -> list[str]:
    values: list[str] = []
    for profile in profile_names:
        raw = dependencies["profiles"][profile].get(key, [])
        if not isinstance(raw, list):
            raise SetupError(f"profile {profile!r} has invalid {key} list")
        values.extend(str(item) for item in raw)
    return list(dict.fromkeys(values))


def _custom_node_status(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    rel = validate_relative_path(str(spec["destination"]), label="custom-node destination")
    destination = safe_destination(root, rel, label="custom-node destination")
    expected_remote = str(spec["expected_remote"])
    expected_revision = str(spec["revision"])
    record: dict[str, Any] = {
        "destination": str(destination),
        "repo": str(spec["repo"]),
        "expected_remote": expected_remote,
        "expected_revision": expected_revision,
    }
    if not destination.exists():
        record["status"] = "missing_clone"
        return record
    if destination.is_symlink() or not destination.is_dir():
        record["status"] = "unsafe_existing_path"
        return record
    remote = _git_output(destination, "remote", "get-url", "origin")
    revision = _git_output(destination, "rev-parse", "HEAD")
    record["remote"] = remote
    record["revision"] = revision
    if _norm_remote(remote) != _norm_remote(expected_remote):
        record["status"] = "wrong_remote"
    elif revision != expected_revision:
        record["status"] = "wrong_revision"
    else:
        record["status"] = "already_pinned"
    return record


def _workflow_status(root: Path, repo_root: Path, filename: str) -> dict[str, Any]:
    source = safe_destination(repo_root, Path("workflows") / filename, label="workflow source")
    destination = safe_destination(root, Path("user/default/workflows") / filename, label="workflow destination")
    result: dict[str, Any] = {"source": str(source), "destination": str(destination), "name": filename}
    if not source.is_file():
        result["status"] = "missing_source"
    elif not destination.exists():
        result["status"] = "copy"
    elif destination.is_file() and _sha256(source) == _sha256(destination):
        result["status"] = "already_identical"
    else:
        result["status"] = "conflict_no_overwrite"
    return result


def build_install_plan(
    *,
    comfy_root: Path,
    profile_names: Sequence[str],
    repo_root: Path = REPO_ROOT,
    dependencies: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dependencies = dependencies or load_dependencies()
    profiles = _profile_names(profile_names, dependencies)
    root = _resolved(comfy_root)
    repo = _resolved(repo_root)
    if not _looks_like_comfy_root(root):
        raise SetupError(f"selected ComfyUI root is not a ComfyUI tree: {root}")
    nodes = dependencies["custom_nodes"]
    node_ids = _profile_union(profiles, dependencies, "custom_nodes")
    node_records: list[dict[str, Any]] = []
    for node_id in node_ids:
        if node_id not in nodes:
            raise SetupError(f"profile references unknown custom node {node_id!r}")
        node_records.append(_custom_node_status(root, nodes[node_id]))
    # The public checkout is itself a ComfyUI custom-node package and is
    # normally cloned to ``custom_nodes/ComfyUI-Universal-Inpaint``.  Accept
    # the historical source-only layout as a read-only fallback so the same
    # installer can package an older delivery without hard-coded host paths.
    nested_source = repo / "custom_nodes" / "ths_universal_inpaint"
    addon_source = nested_source if nested_source.is_dir() else repo
    addon_destination = safe_destination(root, "custom_nodes/ComfyUI-Universal-Inpaint", label="add-on destination")
    legacy_destination = safe_destination(root, "custom_nodes/ths_universal_inpaint", label="legacy add-on destination")
    if not addon_source.is_dir():
        addon_status = "missing_source"
    elif not addon_destination.exists():
        addon_status = "copy"
    elif addon_destination.is_symlink() or not addon_destination.is_dir():
        addon_status = "unsafe_existing_path"
    elif _tree_digest(addon_source) == _tree_digest(addon_destination):
        addon_status = "already_identical"
    else:
        addon_status = "conflict_no_overwrite"
    legacy_status = "absent"
    if legacy_destination.exists():
        legacy_status = "legacy_duplicate_manual_migration_required"
    workflows = [_workflow_status(root, repo, name) for name in _profile_union(profiles, dependencies, "workflows")]
    requirements: list[str] = []
    for profile in profiles:
        requirements.extend(str(item) for item in dependencies["profiles"][profile].get("requirements", []))
    requirements = list(dict.fromkeys(requirements))
    requirement_paths = [str(safe_destination(repo, name, label="requirements file")) for name in requirements]
    errors: list[str] = []
    for record in node_records:
        if record["status"] in {"wrong_remote", "wrong_revision", "unsafe_existing_path"}:
            errors.append(f"custom node {record['destination']}: {record['status']}")
    for record in workflows:
        if record["status"] in {"missing_source", "conflict_no_overwrite"}:
            errors.append(f"workflow {record['name']}: {record['status']}")
    if addon_status in {"missing_source", "conflict_no_overwrite", "unsafe_existing_path"}:
        errors.append(f"add-on: {addon_status}")
    if legacy_status != "absent":
        errors.append(f"legacy add-on: {legacy_status}")
    missing_requirements = [path for path in requirement_paths if not Path(path).is_file()]
    errors.extend(f"requirements file missing: {path}" for path in missing_requirements)
    return {
        "product": "Universal Inpaint",
        "version": VERSION,
        "mode": "plan",
        "comfy_root": str(root),
        "repo_root": str(repo),
        "profiles": profiles,
        "add_on": {
            "source": str(addon_source),
            "destination": str(addon_destination),
            "status": addon_status,
            "legacy_destination": str(legacy_destination),
            "legacy_status": legacy_status,
        },
        "custom_nodes": node_records,
        "workflows": workflows,
        "requirements": requirement_paths,
        "dependency_manifest": dict(dependencies),
        "errors": errors,
        "would_mutate": not errors,
        "note": "No files, git checkouts, Python packages, workflows, or models are changed by a plan.",
    }


def _copy_tree_if_absent(source: Path, destination: Path) -> str:
    if destination.exists():
        if destination.is_dir() and not destination.is_symlink() and _tree_digest(source) == _tree_digest(destination):
            return "already_identical"
        raise SetupError(f"refusing to overwrite existing destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.setup-{os.getpid()}-{time.time_ns()}"
    try:
        shutil.copytree(source, temporary, symlinks=True, ignore=lambda _path, names: [name for name in names if name in COPY_IGNORED or name.startswith(".env") or name.endswith(".egg-info")])
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return "copied"


def _publish_file_no_clobber(source: Path, destination: Path) -> None:
    """Atomically publish a file while failing if another writer won the race."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        # A hard link is atomic and has O_EXCL-like no-clobber semantics on the
        # same filesystem (the normal case for a cache and its destination).
        os.link(source, destination)
    except FileExistsError as exc:
        raise SetupError(f"destination appeared during publish; refusing overwrite: {destination}") from exc
    except OSError:
        # Cross-device filesystems cannot link.  Fall back to an exclusive
        # create and stream copy; never use os.replace, which would clobber a
        # destination created between the preflight and publish.
        try:
            fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0))
        except FileExistsError as exc:
            raise SetupError(f"destination appeared during publish; refusing overwrite: {destination}") from exc
        try:
            with os.fdopen(fd, "wb") as target, source.open("rb") as handle:
                shutil.copyfileobj(handle, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
        except Exception:
            destination.unlink(missing_ok=True)
            raise
    finally:
        source.unlink(missing_ok=True)


def _copy_file_if_absent(source: Path, destination: Path) -> str:
    if destination.exists():
        if destination.is_file() and _sha256(source) == _sha256(destination):
            return "already_identical"
        raise SetupError(f"refusing to overwrite existing destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.setup-{os.getpid()}-{time.time_ns()}"
    try:
        shutil.copy2(source, temporary)
        _publish_file_no_clobber(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
        raise
    return "copied"


def _clone_pinned(destination: Path, spec: Mapping[str, Any]) -> str:
    current = _custom_node_status(destination.parent.parent, spec) if destination.exists() else None
    if current and current.get("status") == "already_pinned":
        return "already_pinned"
    if destination.exists():
        raise SetupError(f"refusing to mutate existing custom-node checkout: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.clone-{os.getpid()}-{time.time_ns()}"
    try:
        _run(("git", "clone", "--depth", "1", str(spec["repo"]), str(temporary)), timeout=300.0)
        _run(("git", "-C", str(temporary), "fetch", "--depth", "1", "origin", str(spec["revision"])), timeout=300.0)
        _run(("git", "-C", str(temporary), "checkout", "--detach", str(spec["revision"])), timeout=120.0)
        remote = _git_output(temporary, "remote", "get-url", "origin")
        revision = _git_output(temporary, "rev-parse", "HEAD")
        if _norm_remote(remote) != _norm_remote(str(spec["expected_remote"])) or revision != str(spec["revision"]):
            raise SetupError(f"cloned custom node did not resolve to expected remote/revision: {destination}")
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return "cloned"


def apply_install_plan(plan: Mapping[str, Any], *, python: Path) -> dict[str, Any]:
    errors = list(plan.get("errors") or [])
    if errors:
        raise SetupError("install plan has conflicts: " + "; ".join(str(item) for item in errors))
    root = Path(str(plan["comfy_root"]))
    repo = Path(str(plan["repo_root"]))
    dependencies = plan.get("dependency_manifest") or load_dependencies(repo / "config" / "dependencies.json")
    applied: list[dict[str, Any]] = []
    # Resolve/install Python dependencies first.  A protected-package refusal
    # then leaves the ComfyUI tree untouched; file/checkouts are only copied
    # after every pip dry-run guard passes.
    for requirement in plan["requirements"]:
        applied.append({"target": requirement, **_pip_plan_and_install(python, Path(requirement))})
    addon = plan["add_on"]
    applied.append({"target": addon["destination"], "status": _copy_tree_if_absent(Path(addon["source"]), Path(addon["destination"]))})
    node_specs = dependencies["custom_nodes"]
    for record in plan["custom_nodes"]:
        destination = Path(str(record["destination"]))
        node_id = next(key for key, value in node_specs.items() if safe_destination(root, value["destination"]) == destination)
        applied.append({"target": str(destination), "status": _clone_pinned(destination, node_specs[node_id])})
    for record in plan["workflows"]:
        applied.append({"target": record["destination"], "status": _copy_file_if_absent(Path(record["source"]), Path(record["destination"]))})
    result = dict(plan)
    result["mode"] = "apply"
    result["applied"] = applied
    result["would_mutate"] = True
    return result


def _model_entries(manifest: Mapping[str, Any], ids: Sequence[str] | None = None, profiles: Sequence[str] | None = None) -> list[dict[str, Any]]:
    all_models = manifest.get("models", [])
    wanted = set(ids or ())
    selected_profiles = set(profiles or ())
    result: list[dict[str, Any]] = []
    for item in all_models:
        if not isinstance(item, dict) or not item.get("id"):
            raise SetupError("invalid model entry in manifest")
        if wanted and item["id"] not in wanted:
            continue
        if selected_profiles and not selected_profiles.intersection(set(item.get("profiles", []))):
            continue
        result.append(item)
    if wanted:
        missing = sorted(wanted - {str(item["id"]) for item in result})
        if missing:
            raise SetupError(f"unknown model id(s): {', '.join(missing)}")
    if not result:
        raise SetupError("no models selected (use --model or --profile)")
    return result


def _model_files(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    files = model.get("files")
    if isinstance(files, list):
        destination_dir = str(model.get("destination_dir", ""))
        result = []
        for item in files:
            if not isinstance(item, dict) or not item.get("filename"):
                raise SetupError(f"model {model.get('id')} has invalid files entry")
            value = dict(model)
            value.update(item)
            value["destination"] = str(Path(destination_dir) / str(item["filename"]))
            result.append(value)
        return result
    return [dict(model)]


def _cache_path(cache_root: Path, model: Mapping[str, Any]) -> Path:
    repo = str(model["repo"]).replace("/", "__")
    revision = str(model["revision"])
    filename = validate_relative_path(str(model["filename"]), label="model filename")
    return safe_destination(cache_root, Path("hf") / repo / revision / (filename.as_posix() + ".part"), label="cache path")


def _entry_status(root: Path, cache_root: Path, model: Mapping[str, Any]) -> dict[str, Any]:
    destination = safe_destination(root, str(model["destination"]), label="model destination")
    expected_size = model.get("size_bytes")
    expected_sha = model.get("sha256")
    record = {
        "model": str(model["id"]),
        "repo": str(model["repo"]),
        "revision": str(model["revision"]),
        "filename": str(model["filename"]),
        "destination": str(destination),
        "size_bytes": expected_size,
        "sha256": expected_sha,
        "gated": bool(model.get("gated")),
        "access_url": model.get("access_url") or model.get("source"),
    }
    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            record["status"] = "unsafe_existing_path"
        elif expected_size is not None and destination.stat().st_size != int(expected_size):
            record["status"] = "conflict_size"
        elif expected_sha and _sha256(destination).lower() != str(expected_sha).lower():
            record["status"] = "conflict_checksum"
        else:
            record["status"] = "already_present"
    else:
        cache = _cache_path(cache_root, model)
        record["cache"] = str(cache)
        record["status"] = "cached_partial" if cache.is_file() else "missing"
        if cache.is_file() and expected_size is not None and cache.stat().st_size == int(expected_size):
            record["status"] = "cached_complete"
    return record


def build_download_plan(
    *,
    comfy_root: Path,
    model_ids: Sequence[str] | None = None,
    profile_names: Sequence[str] | None = None,
    cache_dir: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = manifest or load_models()
    root = _resolved(comfy_root)
    if not _looks_like_comfy_root(root):
        raise SetupError(f"selected ComfyUI root is not a ComfyUI tree: {root}")
    cache_root = _resolved(cache_dir or (root / "models" / ".universal-inpaint-cache"))
    entries: list[dict[str, Any]] = []
    for model in _model_entries(manifest, model_ids, profile_names):
        for file_model in _model_files(model):
            entries.append(_entry_status(root, cache_root, file_model))
    missing = [item for item in entries if item["status"] in {"missing", "cached_partial", "cached_complete"}]
    required = 0
    for item in missing:
        if item.get("size_bytes") is None or item["status"] == "cached_complete":
            continue
        remaining = int(item["size_bytes"])
        if item["status"] == "cached_partial" and item.get("cache"):
            try:
                remaining = max(0, remaining - Path(str(item["cache"])).stat().st_size)
            except OSError:
                pass
        required += remaining
    usage = shutil.disk_usage(root)
    return {
        "product": "Universal Inpaint",
        "version": VERSION,
        "mode": "plan",
        "comfy_root": str(root),
        "cache_dir": str(cache_root),
        "models": entries,
        "known_bytes_to_download": required,
        "free_bytes": int(usage.free),
        "disk_space_ok": bool(required <= usage.free),
        "note": "A plan performs no network request and never creates model directories.",
    }


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip() or None
    # ``hf auth login`` stores a token in the user's Hugging Face cache.  Read
    # it only into memory for the request; never print or persist it here.
    try:
        from huggingface_hub import get_token  # type: ignore

        token = get_token()
        if token:
            return str(token).strip() or None
    except Exception:
        pass
    cache_root = os.environ.get("HF_HOME")
    candidates = []
    if cache_root:
        candidates.append(Path(cache_root) / "token")
    candidates.append(Path.home() / ".cache" / "huggingface" / "token")
    for candidate in candidates:
        try:
            value = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return None


def _model_url(model: Mapping[str, Any]) -> str:
    filename = urllib.parse.quote(str(model["filename"]), safe="/")
    return f"{HF_BASE_URL}/{model['repo']}/resolve/{model['revision']}/{filename}?download=true"


class _HFRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward a bearer token to a different host (Hub CDN/Xet)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise SetupError("refusing a model download redirect to a non-HTTPS URL")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        old_host = urllib.parse.urlsplit(req.full_url).hostname
        new_host = urllib.parse.urlsplit(newurl).hostname
        if old_host and new_host and old_host.lower() != new_host.lower():
            redirected.headers.pop("Authorization", None)
            redirected.unredirected_hdrs.pop("Authorization", None)
        return redirected


def _download_one(root: Path, cache_root: Path, model: Mapping[str, Any], *, allow_gated: bool) -> dict[str, Any]:
    if bool(model.get("gated")):
        if not allow_gated:
            raise SetupError(
                f"model {model['id']} is gated; visit {model.get('access_url') or model.get('source')} and accept its terms yourself, "
                "then rerun with --allow-gated and HF_TOKEN (or an authenticated hf login)"
            )
        if not _hf_token():
            raise SetupError(
                f"model {model['id']} requires Hugging Face access. Set HF_TOKEN or run `hf auth login` after accepting "
                f"the licence at {model.get('access_url') or model.get('source')}"
            )
    destination = safe_destination(root, str(model["destination"]), label="model destination")
    if destination.exists():
        status = _entry_status(root, cache_root, model)["status"]
        if status == "already_present":
            return {"model": model["id"], "destination": str(destination), "status": status}
        raise SetupError(f"refusing to overwrite existing model destination: {destination} ({status})")
    cache = _cache_path(cache_root, model)
    cache.parent.mkdir(parents=True, exist_ok=True)
    expected_size = model.get("size_bytes")
    expected_sha = model.get("sha256")
    # A complete, verified cache is reusable without a second network request.
    # This is important for interrupted large downloads resumed by a later
    # invocation.
    if cache.is_file() and expected_size is not None and cache.stat().st_size == int(expected_size):
        cached_sha = _sha256(cache)
        if not expected_sha or cached_sha.lower() == str(expected_sha).lower():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.parent / f".{destination.name}.setup-{os.getpid()}-{time.time_ns()}"
            try:
                shutil.copy2(cache, temporary)
                _publish_file_no_clobber(temporary, destination)
            except Exception:
                if temporary.exists():
                    temporary.unlink(missing_ok=True)
                raise
            return {"model": model["id"], "destination": str(destination), "cache": str(cache), "status": "reused_cache", "size_bytes": int(expected_size), "sha256": cached_sha}
    partial_size = cache.stat().st_size if cache.is_file() else 0
    headers = {"User-Agent": f"Universal-Inpaint/{VERSION}"}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if partial_size:
        headers["Range"] = f"bytes={partial_size}-"
    request = urllib.request.Request(_model_url(model), headers=headers)
    try:
        if token:
            # urllib's global opener forwards request headers across redirects;
            # use a guarded opener whenever a bearer token is present.
            response = urllib.request.build_opener(_HFRedirectHandler()).open(request, timeout=120.0)
        else:
            response = urllib.request.urlopen(request, timeout=120.0)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403) and bool(model.get("gated")):
            raise SetupError(f"Hugging Face denied gated model access ({exc.code}); accept the licence and verify HF_TOKEN/hf login") from exc
        raise SetupError(f"download failed for {model['id']}: HTTP {exc.code}") from exc
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise SetupError(f"download failed for {model['id']}: {exc}") from exc
    status_code = int(getattr(response, "status", response.getcode() if hasattr(response, "getcode") else 200))
    if partial_size and status_code != 206:
        response.close()
        raise SetupError(f"server did not honor resume Range for {model['id']} (HTTP {status_code}); partial cache was preserved")
    if partial_size:
        content_range = ""
        try:
            content_range = str(response.headers.get("Content-Range", ""))
        except Exception:
            content_range = ""
        # RFC 7233: ``bytes START-END/TOTAL``.  Appending a 206 response with
        # an unexpected start would silently corrupt a large model cache.
        try:
            range_start = int(content_range.split(" ", 1)[1].split("-", 1)[0])
        except (IndexError, ValueError):
            response.close()
            raise SetupError(f"resume response for {model['id']} lacks a valid Content-Range; partial cache was preserved")
        if range_start != partial_size:
            response.close()
            raise SetupError(f"resume response for {model['id']} starts at {range_start}, expected {partial_size}; partial cache was preserved")
    mode = "ab" if partial_size else "wb"
    try:
        with response, cache.open(mode) as handle:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(block)
    except (OSError, urllib.error.URLError) as exc:
        raise SetupError(f"writing model cache failed for {model['id']}: {exc}") from exc
    actual_size = cache.stat().st_size
    if expected_size is not None and actual_size != int(expected_size):
        raise SetupError(f"size mismatch for {model['id']}: expected {expected_size}, got {actual_size}; partial cache preserved")
    actual_sha = _sha256(cache)
    if expected_sha and actual_sha.lower() != str(expected_sha).lower():
        raise SetupError(f"checksum mismatch for {model['id']}: expected {expected_sha}, got {actual_sha}; cache preserved")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SetupError(f"destination appeared during download; refusing overwrite: {destination}")
    temporary = destination.parent / f".{destination.name}.setup-{os.getpid()}-{time.time_ns()}"
    try:
        shutil.copy2(cache, temporary)
        _publish_file_no_clobber(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
        raise
    return {"model": model["id"], "destination": str(destination), "cache": str(cache), "status": "downloaded", "size_bytes": actual_size, "sha256": actual_sha}


def apply_download_plan(plan: Mapping[str, Any], *, manifest: Mapping[str, Any], allow_gated: bool) -> dict[str, Any]:
    if not plan.get("disk_space_ok"):
        raise SetupError("insufficient free disk space for the known model sizes")
    root = Path(str(plan["comfy_root"]))
    cache_root = Path(str(plan["cache_dir"]))
    wanted = {str(item["model"]) for item in plan.get("models", [])}
    models: list[dict[str, Any]] = []
    for model in _model_entries(manifest, wanted):
        models.extend(_model_files(model))
    applied: list[dict[str, Any]] = []
    for model in models:
        applied.append(_download_one(root, cache_root, model, allow_gated=allow_gated))
    result = dict(plan)
    result["mode"] = "apply"
    result["applied"] = applied
    return result


def check_runtime(*, comfy_root: Path | None, python: Path | None, profiles: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"product": "Universal Inpaint", "version": VERSION, "profiles": list(profiles)}
    if comfy_root is None:
        result.update({"status": "unknown", "comfy_root": None, "python": str(python) if python else None, "error": "ComfyUI root not found; pass --comfy-root"})
        return result
    root = _resolved(comfy_root)
    result["comfy_root"] = str(root)
    result["root_valid"] = _looks_like_comfy_root(root)
    if not result["root_valid"]:
        result["status"] = "invalid_root"
        return result
    interpreter = python or resolve_python(root)
    result["python"] = str(interpreter)
    probe = _run((str(interpreter), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}'); print(sys.version)"), check=False, timeout=30.0)
    result["python_probe_returncode"] = probe.returncode
    lines = (probe.stdout or "").splitlines()
    result["python_version"] = lines[-1] if lines else None
    version_line = lines[0] if lines else ""
    try:
        version_tuple = tuple(int(part) for part in version_line.split(".")[:3])
    except ValueError:
        version_tuple = (0, 0, 0)
    result["python_version_tuple"] = version_tuple
    result["python_supported"] = version_tuple >= (3, 10, 0)
    required_modules = {
        "editor": ("numpy", "PIL", "cv2"),
        "sam3": ("torch", "numpy", "PIL", "cv2"),
    }
    modules: dict[str, bool] = {}
    for profile in profiles:
        for module in required_modules.get(profile, ()):
            modules.setdefault(module, _python_has_module(interpreter, module))
    result["python_modules"] = modules
    missing_modules = sorted(name for name, present in modules.items() if not present)
    result["missing_modules"] = missing_modules
    custom_node_checks: dict[str, Any] = {}
    if "sam3" in profiles:
        sam3_root = root / "custom_nodes" / "comfyui-sam3"
        sam3_files = [sam3_root / "nodes" / "_model_cache.py", sam3_root / "nodes" / "sam3_model_patcher.py"]
        custom_node_checks["sam3"] = {"path": str(sam3_root), "present": sam3_root.is_dir(), "native_files": all(path.is_file() for path in sam3_files)}
    result["custom_nodes"] = custom_node_checks
    missing_nodes = [name for name, record in custom_node_checks.items() if not all(bool(value) for key, value in record.items() if key not in {"path"})]
    result["missing_custom_nodes"] = missing_nodes
    model_checks: dict[str, Any] = {}
    missing_models: list[str] = []
    try:
        model_manifest = load_models()
        for model in model_manifest.get("models", []):
            if not set(profiles).intersection(set(model.get("profiles", []))):
                continue
            for file_model in _model_files(model):
                destination = safe_destination(root, str(file_model["destination"]), label="model destination")
                expected_size = file_model.get("size_bytes")
                if not destination.is_file():
                    status = "missing"
                elif expected_size is not None and destination.stat().st_size != int(expected_size):
                    status = "conflict_size"
                else:
                    status = "present"
                key = f"{file_model['id']}:{file_model['filename']}"
                model_checks[key] = {"destination": str(destination), "status": status, "gated": bool(file_model.get("gated"))}
                if status != "present":
                    missing_models.append(key)
    except SetupError as exc:
        result["model_manifest_error"] = str(exc)
    result["models"] = model_checks
    result["missing_models"] = missing_models
    checks_ok = (
        probe.returncode == 0
        and bool(result["python_supported"])
        and not missing_modules
        and not missing_nodes
        and not missing_models
    )
    result["status"] = "ok" if checks_ok else ("python_probe_failed" if probe.returncode else "missing_capabilities")
    return result


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--comfy-root", type=Path, help="ComfyUI root (the directory containing main.py)")
    common.add_argument("--python", dest="python_path", type=Path, help="explicit Python interpreter")
    common.add_argument("--profile", action="append", dest="profiles", help="profile: editor, sam3 (repeatable)")
    common.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser = argparse.ArgumentParser(description=__doc__, parents=[common])
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command")
    # The same options are accepted after the subcommand too.  Suppressed
    # defaults keep a value supplied before the subcommand from being
    # overwritten by the subparser's implicit ``None``/``False`` default.
    subcommon = argparse.ArgumentParser(add_help=False)
    subcommon.add_argument("--comfy-root", type=Path, default=argparse.SUPPRESS)
    subcommon.add_argument("--python", dest="python_path", type=Path, default=argparse.SUPPRESS)
    subcommon.add_argument("--profile", action="append", dest="profiles", default=argparse.SUPPRESS)
    subcommon.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    for name in ("check", "plan", "install", "download"):
        command = sub.add_parser(name, parents=[subcommon], help=name)
        if name in ("install", "download"):
            command.add_argument("--apply", action="store_true", help="perform mutations; without it this is a read-only plan")
        if name == "download":
            command.add_argument("--model", action="append", dest="models", help="model id (repeatable)")
            command.add_argument("--cache-dir", type=Path)
            command.add_argument("--allow-gated", action="store_true", help="download a gated model only after you accepted its licence yourself")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    command = args.command or "plan"
    try:
        dependencies = load_dependencies()
        profiles = _profile_names(args.profiles, dependencies)
        root = detect_comfy_root(args.comfy_root)
        if command == "check":
            interpreter = resolve_python(root, args.python_path) if root or args.python_path else None
            result = check_runtime(comfy_root=root, python=interpreter, profiles=profiles)
        elif command in ("plan", "install"):
            if root is None:
                raise SetupError("ComfyUI root not found; pass --comfy-root")
            plan = build_install_plan(comfy_root=root, profile_names=profiles)
            if command == "install" and args.apply:
                interpreter = resolve_python(root, args.python_path)
                result = apply_install_plan(plan, python=interpreter)
            else:
                result = plan
        elif command == "download":
            if root is None:
                raise SetupError("ComfyUI root not found; pass --comfy-root")
            manifest = load_models()
            # ``editor`` is the safe default for install, but a model id is an
            # explicit selection and must not be filtered out by that default.
            download_profiles = args.profiles if args.profiles else None
            plan = build_download_plan(comfy_root=root, model_ids=args.models, profile_names=download_profiles, cache_dir=args.cache_dir, manifest=manifest)
            if args.apply:
                result = apply_download_plan(plan, manifest=manifest, allow_gated=bool(args.allow_gated))
            else:
                result = plan
        else:  # pragma: no cover - argparse restricts this branch.
            raise SetupError(f"unknown command: {command}")
        print(json.dumps(result, ensure_ascii=False, indent=None if args.json else 2, sort_keys=True))
        return 2 if (command == "check" and result.get("status") != "ok") or result.get("errors") else 0
    except SetupError as exc:
        error = {"product": "Universal Inpaint", "version": VERSION, "status": "error", "error": str(exc)}
        print(json.dumps(error, ensure_ascii=False, indent=None if args.json else 2), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
