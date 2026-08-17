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

# Data colormaps are NOT brand colours.
#
# A colormap's job is to encode magnitude, and a ramp hand-mixed from two logo
# colours is neither perceptually uniform nor well ordered: the first attempt
# banded visibly and collapsed most of the range into one muddy mid-teal. These
# are ColorBrewer sequential schemes instead, picked from the blue and brown
# families so they still sit next to the logo, and both run light-to-dark
# because the maps render on a white page.
#
# Brand consistency lives in the chrome above and in the explanatory diagrams,
# where colour is categorical rather than quantitative.
COUNT_CMAP = "YlGnBu"
GAP_CMAP = "YlOrBr"


def sequential_colormap() -> Any:
    """Return the colormap for per-cell counts."""
    return COUNT_CMAP


def gap_colormap() -> Any:
    """Return the colormap for revisit gaps."""
    return GAP_CMAP
