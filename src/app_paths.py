"""Helpers for resolving runtime storage paths for the application."""

from __future__ import annotations

import sys
from pathlib import Path


def get_app_root() -> Path:
    """Return the directory used for runtime app data.

    Returns:
        Directory beside the executable when frozen, otherwise the project root.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_icons_dir() -> Path:
    """Return the persistent icon cache directory."""
    return get_app_root() / "icons"


def get_records_dir() -> Path:
    """Return the persistent recordings output directory."""
    return get_app_root() / "records"
