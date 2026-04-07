"""Icon creator for tabler icons with transparent caching."""

from __future__ import annotations

from typing import cast

import customtkinter as ctk
from PIL import Image
from PIL.Image import Image as PILImage
from pytablericons import TablerIcons, outline_icon

from app_paths import get_icons_dir

ICON_SIZE = 32
ICONS_DIR = get_icons_dir()


def create_icon(
    icon_enum: outline_icon.OutlineIcon,
    size: int = ICON_SIZE,
    color: str = "#FFFFFF",
) -> ctk.CTkImage | None:
    """Create a CTkImage from a tabler icon with automatic caching.

    Args:
        icon_enum: OutlineIcon enum member (e.g., OutlineIcon.PLAYER_RECORD).
        size: Icon size in pixels (default 32).
        color: Icon color as hex string (default white).

    Returns:
        CTkImage object or None if icon cannot be loaded.

    Example:
        >>> icon = create_icon(outline_icon.OutlineIcon.PLAYER_RECORD)
        >>> button = ctk.CTkButton(parent, image=icon, text="")
    """
    ICONS_DIR.mkdir(parents=True, exist_ok=True)

    # Use enum name for cache file
    icon_name = icon_enum.name.lower()
    icon_path = ICONS_DIR / f"{icon_name}_{size}.png"

    # Load from cache or generate and cache
    if not icon_path.exists():
        try:
            tabler = TablerIcons()
            icon_image = cast(PILImage, tabler.load(icon_enum, size=size, color=color))
            icon_image.save(icon_path)
        except Exception:
            return None

    # Open cached image and convert to CTkImage
    try:
        img = Image.open(icon_path)
        return ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
    except Exception:
        return None
