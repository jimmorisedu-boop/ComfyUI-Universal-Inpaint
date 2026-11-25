"""ComfyUI custom-node package for interactive video mask selection and tracking."""

from __future__ import annotations

# Pytest and lightweight harnesses may execute the repository root directly
# without pre-registering a parent package. Give relative imports the same
# package context that ComfyUI's extension loader supplies.
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

from .mask_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# ComfyUI discovers the browser editor bundle from this relative directory.
WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
