"""Standalone additive custom-node package for universal masked H3 repair."""

from __future__ import annotations

# Pytest and a few lightweight smoke harnesses execute a direct repository
# entry point without first registering its package name.  Give relative
# imports the same package context that ComfyUI's loader supplies.
if not __package__:  # pragma: no cover - exercised by pytest collection
    import sys as _sys
    from pathlib import Path as _Path

    _module = _sys.modules.get(__name__)
    if _module is not None:
        _module.__path__ = [str(_Path(__file__).resolve().parent)]
        _sys.modules.setdefault("_ths_universal_inpaint", _module)
    __package__ = "_ths_universal_inpaint"
elif __package__ not in __import__("sys").modules:  # pragma: no cover
    import sys as _sys
    import types as _types
    from pathlib import Path as _Path

    _shim = _types.ModuleType(__package__)
    _shim.__file__ = str(_Path(__file__).resolve())
    _shim.__path__ = [str(_Path(__file__).resolve().parent)]
    _shim.__package__ = __package__
    _sys.modules[__package__] = _shim

from .repair_nodes import NODE_CLASS_MAPPINGS as _REPAIR_NODES
from .repair_nodes import NODE_DISPLAY_NAME_MAPPINGS as _REPAIR_NAMES

try:
    # Corrected-frame guidance stays in a small module so CPU/unit-test imports
    # remain independent from native H3/ComfyUI imports.
    from .corrected_frame_nodes import NODE_CLASS_MAPPINGS as _CORRECTED_FRAME_NODES
    from .corrected_frame_nodes import NODE_DISPLAY_NAME_MAPPINGS as _CORRECTED_FRAME_NAMES
except ModuleNotFoundError as exc:  # companion module may be installed later.
    if exc.name not in {f"{__package__}.corrected_frame_nodes", "corrected_frame_nodes"}:
        raise
    _CORRECTED_FRAME_NODES = {}
    _CORRECTED_FRAME_NAMES = {}

try:
    # The caption backend is optional during CPU-only installs/tests.  Keep the
    # package importable before the companion node module is installed while
    # merging its mappings automatically when it is present.
    from .auto_prompt_nodes import NODE_CLASS_MAPPINGS as _AUTO_PROMPT_NODES
    from .auto_prompt_nodes import NODE_DISPLAY_NAME_MAPPINGS as _AUTO_PROMPT_NAMES
except ModuleNotFoundError as exc:  # companion module may be installed later.
    if exc.name not in {f"{__package__}.auto_prompt_nodes", "auto_prompt_nodes"}:
        raise
    _AUTO_PROMPT_NODES = {}
    _AUTO_PROMPT_NAMES = {}

try:
    # The mask editor/front-end is maintained separately so it can evolve
    # without changing the repair/composite contract.  Keep this import local
    # and additive: a CPU-only test of repair_nodes.py does not need ComfyUI.
    from .mask_nodes import NODE_CLASS_MAPPINGS as _MASK_NODES
    from .mask_nodes import NODE_DISPLAY_NAME_MAPPINGS as _MASK_NAMES
except ModuleNotFoundError as exc:  # mask_nodes.py is supplied beside this file.
    if exc.name not in {f"{__package__}.mask_nodes", "mask_nodes"}:
        raise
    _MASK_NODES = {}
    _MASK_NAMES = {}

from .h3_window import THSUniversalH3Window

NODE_CLASS_MAPPINGS = {"THSUniversalH3Window": THSUniversalH3Window, **_REPAIR_NODES, **_CORRECTED_FRAME_NODES, **_MASK_NODES, **_AUTO_PROMPT_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_REPAIR_NAMES, **_CORRECTED_FRAME_NAMES, **_MASK_NAMES, **_AUTO_PROMPT_NAMES}

# ComfyUI discovers the optional browser bundle from this relative directory.
# It is harmless when the package is run in the CPU test harness.
WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
