"""Execute every documentation example and write its figures and measurements.

Zensical collects the files under ``docs/`` before rendering pages, so figures
written while a page renders are not copied into the built site. The examples
therefore run here, ahead of the build, and each documentation page embeds the
figure and measurements recorded by this step.
"""

from __future__ import annotations

from examples.communication_constellation import (
    build_documentation_assets as build_communications,
)
from examples.earth_observation_constellation import (
    build_documentation_assets as build_earth_observation,
)


def main() -> None:
    build_communications()
    build_earth_observation()


if __name__ == "__main__":
    main()
