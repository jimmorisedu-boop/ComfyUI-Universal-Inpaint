"""Load the direct-repository Comfy custom node as a test package.

ComfyUI imports a custom-node repository by executing its root ``__init__.py``
with the repository as the package search path.  Pytest needs the same package
context for the relative imports used by the addon modules.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "ths_universal_inpaint"


def _load_root_package() -> None:
    if PACKAGE_NAME in sys.modules:
        return
    package = types.ModuleType(PACKAGE_NAME)
    package.__file__ = str(ROOT / "__init__.py")
    package.__path__ = [str(ROOT)]
    package.__package__ = PACKAGE_NAME
    sys.modules[PACKAGE_NAME] = package
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec and spec.loader
    spec.loader.exec_module(package)


_load_root_package()
