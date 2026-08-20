"""Execute documentation examples and write their generated assets."""

from __future__ import annotations

from examples.communication_constellation import (
    build_documentation_assets as build_communications,
)
from examples.constellation import DOC_FIGURE_DIR
from examples.docs_diagrams import main as build_diagrams
from examples.earth_observation_constellation import (
    build_documentation_assets as build_earth_observation,
)


def main() -> None:
    build_diagrams()
    (DOC_FIGURE_DIR / "communications-availability.html").write_text(
        build_communications() + "\n"
    )
    (DOC_FIGURE_DIR / "earth-observation.html").write_text(
        build_earth_observation() + "\n"
    )


if __name__ == "__main__":
    main()
