"""Run the published matrix; execute from the repository root."""

import subprocess
import sys
from pathlib import Path

out = Path("benchmarks/results")
out.mkdir(exist_ok=True)
for order in (5, 6, 7):
    subprocess.run(
        [
            sys.executable,
            "benchmarks/competitors.py",
            "--order",
            str(order),
            "--output",
            str(out / f"comparison-{order}.json"),
        ],
        check=True,
    )
for workflow, regions in [("visibility", 10000), ("observation", 1000)]:
    for order in (5, 6, 7):
        subprocess.run(
            [
                sys.executable,
                "benchmarks/competitors.py",
                "--order",
                str(order),
                "--workflows",
                workflow,
                "--regions",
                str(regions),
                "--backends",
                "polypix-serial",
                "polypix-auto",
                "healpy",
                "hpgeom",
                "--output",
                str(out / f"large-{workflow}-{order}.json"),
            ],
            check=True,
        )
