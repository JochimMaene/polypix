"""Guards against documentation drifting away from runnable code."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def test_guide_has_no_unchecked_python_blocks() -> None:
    """Tutorial Python belongs in doctest blocks so undefined names fail CI."""
    guide = (REPO / "docs" / "guide.md").read_text()
    assert "```python" not in guide


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


def test_diagram_examples_still_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behavior-sensitive figures still render from the current API."""
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from examples import docs_diagrams

    monkeypatch.setattr(docs_diagrams, "DOC_FIGURE_DIR", tmp_path)
    docs_diagrams.main()
    assert sorted(p.name for p in tmp_path.glob("*.svg")) == [
        "center-sampling.svg",
        "overlap-coverage.svg",
    ]


def test_static_documentation_figures_are_checked_in() -> None:
    figures = REPO / "docs" / "assets" / "generated"
    for name in (
        "cell-at.svg",
        "cover-cap.svg",
        "cover-convex-polygon.svg",
        "cover-sweep.svg",
        "resolution-steps.svg",
        "sphere-levels.png",
    ):
        assert (figures / name).is_file()
