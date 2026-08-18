"""Execute the doctest blocks in the getting-started guide.

The guide is the single source for the tutorial geometry. `docs_diagrams`
draws its figures from the namespace this module returns, so a figure cannot
show geometry that the page does not state, and the helper functions the page
tells readers to copy are the ones the figures actually use.

Sphinx checks the same blocks from the other side: `pixi run --environment docs
docs-doctest` runs them and compares every printed array against the real
output. Between the two, the code, the numbers, and the pictures move together
or the build fails.
"""

from __future__ import annotations

import doctest
import re
from functools import cache
from pathlib import Path
from typing import Any

GUIDE = Path(__file__).resolve().parent.parent / "docs" / "guide.md"

DOCTEST_BLOCK = re.compile(r"```\{doctest\}\n(?P<body>.*?)```", re.DOTALL)


def run_guide(path: Path = GUIDE) -> dict[str, Any]:
    """Run every doctest block in `path` in order and return the namespace."""
    blocks = DOCTEST_BLOCK.findall(path.read_text())
    if not blocks:
        message = f"no doctest blocks found in {path}"
        raise RuntimeError(message)

    parser = doctest.DocTestParser()
    namespace: dict[str, Any] = {}
    for body in blocks:
        for example in parser.get_examples(body):
            # "exec" rather than "single" so expression results stay unprinted.
            exec(compile(example.source, str(path), "exec"), namespace)
    return namespace


@cache
def guide() -> dict[str, Any]:
    """Return the guide namespace, running the page once per process."""
    return run_guide()
