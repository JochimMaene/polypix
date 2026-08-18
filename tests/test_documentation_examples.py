"""Guards against documentation drifting away from runnable code.

Published examples come from one of two places, neither hand-copied: a real
module pulled in with a `literalinclude` marker, or a doctest block that
`pixi run --environment docs docs-doctest` executes. These tests cover the
first kind, checking that the modules still run and that the markers the pages
ask for still exist.

They also cover the seam between the two. The getting-started figures are drawn
from the namespace of that page's own doctest blocks, so renaming a variable in
the guide must not leave a figure reaching for a name that no longer exists.
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

GUIDE_LOOKUP = re.compile(r"""(?:page|guide\(\))\[["'](?P<name>[^"']+)["']\]""")

INCLUDE = re.compile(
    r"```\{literalinclude\}\s+(?P<path>\S+)(?P<body>.*?)```",
    re.DOTALL,
)
START = re.compile(r'start-after:\s*"--8<-- \[start:(?P<name>[^\]]+)\]"')
END = re.compile(r'end-before:\s*"--8<-- \[end:(?P<name>[^\]]+)\]"')


def documentation_includes() -> list[tuple[Path, Path, str | None, str | None]]:
    """Return every literalinclude in the documentation with its markers."""
    found = []
    for page in sorted(DOCS.rglob("*.md")):
        text = page.read_text()
        for match in INCLUDE.finditer(text):
            target = (page.parent / match.group("path")).resolve()
            start = START.search(match.group("body"))
            end = END.search(match.group("body"))
            found.append(
                (
                    page,
                    target,
                    start.group("name") if start else None,
                    end.group("name") if end else None,
                )
            )
    return found


def test_documentation_has_literal_includes() -> None:
    assert documentation_includes(), "expected the docs to include real source files"


@pytest.mark.parametrize(
    ("page", "target", "start", "end"),
    documentation_includes(),
    ids=lambda value: getattr(value, "name", str(value)),
)
def test_included_source_and_markers_exist(
    page: Path,
    target: Path,
    start: str | None,
    end: str | None,
) -> None:
    assert target.is_file(), f"{page.name} includes missing file {target}"
    source = target.read_text()
    for kind, name in (("start", start), ("end", end)):
        if name is None:
            continue
        marker = f"--8<-- [{kind}:{name}]"
        assert marker in source, f"{page.name} wants {marker} in {target.name}"


def test_included_regions_are_not_empty() -> None:
    """A renamed marker would otherwise publish an empty code block."""
    for page, target, start, end in documentation_includes():
        if start is None or end is None:
            continue
        source = target.read_text()
        body = source.split(f"--8<-- [start:{start}]", 1)[1]
        body = body.split(f"--8<-- [end:{end}]", 1)[0]
        assert body.strip(), f"{page.name} includes an empty region for {start}"


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
        "cover-footprint.svg",
        "cover-sweep.svg",
        "resolution-steps.svg",
    ]
