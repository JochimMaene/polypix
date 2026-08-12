"""Execute documentation examples and write their generated assets."""

from __future__ import annotations

from examples.communication_constellation import (
    build_documentation_assets as build_communications,
)
from examples.communication_constellation import (
    documentation_html as communications_html,
)
from examples.constellation import DOC_FIGURE_DIR
from examples.earth_observation_constellation import (
    build_documentation_assets as build_earth_observation,
)
from examples.earth_observation_constellation import (
    documentation_html as earth_observation_html,
)


def main() -> None:
    build_communications()
    build_earth_observation()
    (DOC_FIGURE_DIR / "communications-availability.html").write_text(
        communications_html() + "\n"
    )
    (DOC_FIGURE_DIR / "earth-observation.html").write_text(
        earth_observation_html() + "\n"
    )


if __name__ == "__main__":
    main()
