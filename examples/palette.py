"""One colour palette for the logo, the diagrams, and the case-study maps.

Everything here is derived from the two colours in `docs/_static/polypix.svg`,
so the site, the explanatory diagrams, and the global maps stay recognisably
the same product. `docs/_static/polypix.css` mirrors the theme values.
"""

from __future__ import annotations

from typing import Any

# Straight from the logo.
NAVY_DEEP = "#0c1c3b"
NAVY = "#162240"
CYAN = "#2db7d4"

# Link and accent colours, sampled off the navy-to-cyan ramp so they carry
# enough contrast on white for body text.
PRIMARY = "#17607d"
SECONDARY = "#a8475c"

# Explanatory grid diagrams.
GRID_EDGE = "#8fa0b3"
GRID_CENTER = "#aab6c4"
COVERED_FILL = CYAN
COVERED_CENTER = "#12607a"
REGION_LINE = SECONDARY
MISSED_FILL = "#e0a33c"
LABEL = "#6f7f92"

# Global maps. These render on a light page, so the panel is a pale tint of the
# logo navy rather than the navy itself.
MAP_BACKGROUND = "white"
MAP_PANEL = "#f1f4f7"
MAP_TEXT = "#253545"
MAP_MUTED = "#4c5d6e"
MAP_RULE = "#aab4bf"
MAP_GRID = "#667788"

# Pale cyan up through the logo cyan into the logo navy. Lightness falls the
# whole way, which is what a sequential scale needs on a light background.
SEQUENTIAL = ["#eaf7fa", "#a7e0ec", CYAN, "#1a7290", "#14456a", NAVY_DEEP]

# Fast to slow, for revisit. Both ends stay legible on the pale panel.
DIVERGING = [CYAN, "#7fbfd6", "#b9a9b4", "#c98a5a", "#9c5a22"]


def sequential_colormap() -> Any:
    """Return the brand sequential colormap."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("polypix", SEQUENTIAL)


def diverging_colormap() -> Any:
    """Return the brand fast-to-slow colormap."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("polypix_fast_slow", DIVERGING)
