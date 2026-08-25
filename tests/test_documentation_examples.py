"""Guards against documentation drifting away from runnable code.

Sphinx executes doctest blocks and resolves literalincludes during the docs
build. These tests cover the seam between the docs and generated figures: the
getting-started figures are drawn from the namespace of that page's own
doctest blocks, so renaming a variable in the guide must not leave a figure
reaching for a name that no longer exists.
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

GUIDE_LOOKUP = re.compile(r"""(?:page|guide\(\))\[["'](?P<name>[^"']+)["']\]""")


def readme_quick_start() -> str:
    """Return the Python block under the README's Quick start heading."""
    text = (REPO / "README.md").read_text()
    assert "## Quick start" in text, "the README no longer has a Quick start"
    block = re.search(
        r"```python\n(?P<body>.*?)```",
        text.split("## Quick start", 1)[1],
        re.DOTALL,
    )
    assert block is not None, "the README Quick start has no Python block"
    return block.group("body")


def test_readme_quick_start_prints_what_it_claims() -> None:
    """The README is not doctested, so run it and check its output comment."""
    source = readme_quick_start()
    claimed = re.search(r"print\(.*?\)\s*#\s*(?P<output>.+)", source)
    assert claimed is not None, "the README example should show its output"

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(source, "README.md", "exec"), {})
    assert captured.getvalue().strip() == claimed.group("output").strip()


def test_guide_fixtures_the_diagrams_ask_for_exist() -> None:
    """Every name the figures pull out of the guide is defined by the guide."""
    from examples.doc_snippets import run_guide

    source = (REPO / "examples" / "docs_diagrams.py").read_text()
    wanted = sorted(set(GUIDE_LOOKUP.findall(source)))
    assert wanted, "expected the diagrams to source their geometry from the guide"

    namespace = run_guide()
    missing = [name for name in wanted if name not in namespace]
    assert not missing, f"docs/guide.md no longer defines {missing}"


def test_diagram_examples_still_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drawing every figure exercises the guide namespace the figures read."""
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from examples import docs_diagrams

    monkeypatch.setattr(docs_diagrams, "DOC_FIGURE_DIR", tmp_path)
    docs_diagrams.main()
    assert (tmp_path / "sphere-levels.png").is_file()
    assert sorted(p.name for p in tmp_path.glob("*.svg")) == [
        "cell-at.svg",
        "center-sampling.svg",
        "cover-cap.svg",
        "cover-convex-polygon.svg",
        "cover-sweep.svg",
        "resolution-steps.svg",
    ]
