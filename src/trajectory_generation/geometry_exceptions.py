"""Shared geometry exception helpers."""

from __future__ import annotations

try:
    from shapely.errors import GEOSException as _ShapelyGEOSException
except ImportError:  # pragma: no cover
    _ShapelyGEOSException = None


def geometry_exceptions(*, include_runtime: bool = False) -> tuple[type[BaseException], ...]:
    """Return a standard tuple of recoverable geometry-related exceptions."""
    base: tuple[type[BaseException], ...] = (ValueError, TypeError, AttributeError)
    if include_runtime:
        base = base + (RuntimeError,)
    if _ShapelyGEOSException is not None:
        base = base + (_ShapelyGEOSException,)
    return base
