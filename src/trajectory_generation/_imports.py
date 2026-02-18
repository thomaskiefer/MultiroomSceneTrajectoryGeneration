"""Shared dynamic import helpers for optional local floorplan modules."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable, Optional, TypeVar


T = TypeVar("T")


def import_with_tools_fallback(
    *,
    project_root: Path,
    import_fn: Callable[[], T],
    module_prefix: str = "floorplan",
    logger: Optional[object] = None,
) -> T:
    """
    Import using standard resolution first, then fallback to `<project_root>/tools`.

    This utility ensures temporary `sys.path` injection is cleaned up and partially
    imported modules are rolled back on import failures.
    """
    try:
        return import_fn()
    except ModuleNotFoundError as exc:
        if exc.name and not exc.name.startswith(module_prefix):
            raise

    tools_dir = project_root / "tools"
    tools_dir_str = str(tools_dir)
    injected = False
    preexisting_modules = {
        key: value
        for key, value in sys.modules.items()
        if key == module_prefix or key.startswith(f"{module_prefix}.")
    }
    if tools_dir_str not in sys.path:
        if logger is not None and hasattr(logger, "debug"):
            logger.debug("Falling back to local tools import via sys.path injection: %s", tools_dir)
        sys.path.insert(0, tools_dir_str)
        injected = True

    try:
        return import_fn()
    except Exception:
        for key in list(sys.modules):
            if key == module_prefix or key.startswith(f"{module_prefix}."):
                del sys.modules[key]
        sys.modules.update(preexisting_modules)
        raise
    finally:
        if injected:
            try:
                sys.path.remove(tools_dir_str)
            except ValueError:
                pass

