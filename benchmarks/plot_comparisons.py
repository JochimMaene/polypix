"""Draw the documentation charts and CSV from benchmarks/results.

pixi run -e docs python benchmarks/plot_comparisons.py
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples import palette  # noqa: E402

RESULTS = ROOT / "benchmarks" / "results"
OUT = ROOT / "docs" / "assets" / "benchmarks"

SITE = palette.SITE_THEMES
NAMES = {
    "polypix": ("Polypix", "o"),
    "polypix-threaded": ("Polypix", "o"),
    "healpy": ("healpy", "s"),
    "hpgeom": ("HPGeom", "D"),
    "mocpy": ("MOCPy", "v"),
    "h3": ("H3", "^"),
    "s2": ("S2", "P"),
}
# Polypix wears the site's primary colour. The others are checked with a
# colour-vision simulator, so the libraries that share a chart stay apart in
# both themes; the markers back them up.
COLOURS = {
    "light": {
        "polypix": SITE["light"]["primary"],
        "healpy": "#eda100",
        "hpgeom": "#e87ba4",
        "mocpy": "#008300",
        "h3": "#6250d6",
        "s2": "#e34948",
    },
    "dark": {
        "polypix": SITE["dark"]["primary"],
        "healpy": "#c98500",
        "hpgeom": "#d55181",
        "mocpy": "#008300",
        "h3": "#9085e9",
        "s2": "#e66767",
    },
}
VERSION_KEY = {"polypix-threaded": "polypix", "s2": "s2geometry"}
WORKLOADS = {"visibility": "Visibility", "observation": "Observation"}
MODES = {
    "center": "Cell selected when its center is inside",
    "overlap": "Cell selected when it touches the shape",
}


def load(suite):
    """Records grouped by (library, workload, mode, level); fails on any error."""
    report = json.loads((RESULTS / f"{suite}.json").read_text())
    groups = defaultdict(list)
    for record in report["records"]:
        if "error" in record:
            raise SystemExit(f"{suite} has a failed run:\n{record['error']}")
        key = (record["library"], record["workload"], record["mode"], record["level"])
        groups[key].append(record)
    return report["environment"], groups


def summary(records):
    """Median, min and max over processes of each process's warm median, in ms."""
    per_process = [statistics.median(r["warm_seconds"]) * 1e3 for r in records]
    return {
        "warm_ms": statistics.median(per_process),
        "warm_min_ms": min(per_process),
        "warm_max_ms": max(per_process),
        "first_call_ms": statistics.median(r["first_seconds"] for r in records) * 1e3,
        "cells": records[0]["cells"],
        "mean_abs_error": records[0].get("mean_abs_error"),
        "mean_error": records[0].get("mean_error"),
    }


def colour(library, theme):
    return COLOURS[theme][library.removesuffix("-threaded")]


def label(library, versions):
    name = NAMES[library][0]
    version = versions.get(VERSION_KEY.get(library, library))
    name = f"{name} {version}" if version else name
    return f"{name}, threaded" if library.endswith("-threaded") else name


def shapes(workload, groups):
    """'128 circles' or '512 footprints', read from the records."""
    count = next(g[0]["shapes"] for key, g in groups.items() if key[1] == workload)
    kind = "circles" if workload == "visibility" else "footprints"
    return f"{count:,} {kind}"


def style(ax):
    ax.grid(linewidth=1.2)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def save(fig, name, theme):
    suffix = "" if theme == "light" else f"-{theme}"
    fig.savefig(OUT / f"{name}{suffix}.svg")
    plt.close(fig)


def healpix_chart(workload, groups, versions, theme):
    libraries = ["polypix", "polypix-threaded", "healpy", "hpgeom"]
    levels = sorted({key[3] for key in groups})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    style(ax)
    ax.grid(axis="x", visible=False)
    width = 0.2
    for i, library in enumerate(libraries):
        stats = [summary(groups[library, workload, "center", lv]) for lv in levels]
        x = [j + (i - (len(libraries) - 1) / 2) * width for j in range(len(levels))]
        heights = [s["warm_ms"] for s in stats]
        whiskers = [
            [s["warm_ms"] - s["warm_min_ms"] for s in stats],
            [s["warm_max_ms"] - s["warm_ms"] for s in stats],
        ]
        threaded = library.endswith("threaded")
        ax.bar(
            x,
            heights,
            width * 0.92,
            yerr=whiskers,
            color="none" if threaded else colour(library, theme),
            edgecolor=colour(library, theme),
            hatch="////" if threaded else None,
            label=label(library, versions),
            error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": SITE[theme]["muted"]},
        )
        for xi, s in zip(x, stats, strict=True):
            ax.annotate(
                f"{s['warm_ms']:.3g}",
                (xi, s["warm_max_ms"]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(
        range(len(levels)),
        [
            f"Order {lv}\n{groups['polypix', workload, 'center', lv][0]['cells']:,} cells"
            for lv in levels
        ],
    )
    ax.set_ylabel("Milliseconds per call (lower is better)")
    title = f"{WORKLOADS[workload]}: {shapes(workload, groups)}"
    ax.set_title(title, loc="left", fontsize=14)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, f"{workload}-healpix", theme)


def grids_chart(groups, versions, theme):
    """Time against grid size: one row per workload, one column per rule."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    handles = {}
    for row, workload in zip(axes, WORKLOADS, strict=True):
        for ax, mode in zip(row, MODES, strict=True):
            style(ax)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(shapes(workload, groups), loc="left", fontsize=10)
            for library in NAMES:
                levels = sorted(
                    key[3] for key in groups if key[:3] == (library, workload, mode)
                )
                if not levels:
                    continue
                stats = [summary(groups[library, workload, mode, lv]) for lv in levels]
                (handles[library],) = ax.plot(
                    [s["cells"] for s in stats],
                    [s["warm_ms"] for s in stats],
                    color=colour(library, theme),
                    marker=NAMES[library][1],
                    ms=6,
                    lw=1.6,
                )
        row[0].set_ylabel(f"{WORKLOADS[workload]}: ms per call")
    for ax, mode in zip(axes[0], MODES, strict=True):
        ax.annotate(
            MODES[mode],
            (0, 1.14),
            xycoords="axes fraction",
            fontsize=12,
            weight="bold",
        )
    for ax in axes[1]:
        ax.set_xlabel("Cells on the whole globe")
    order = [lib for lib in NAMES if lib in handles]
    fig.legend(
        [handles[lib] for lib in order],
        [label(lib, versions) for lib in order],
        loc="lower center",
        ncol=len(order),
        frameon=False,
    )
    fig.suptitle(
        "Time on each library's own grid (lower is better)",
        x=0.06,
        ha="left",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save(fig, "grids", theme)


def write_csv(suites):
    columns = ["suite", "library", "workload", "mode", "level"]
    fields = [
        "cells",
        "warm_ms",
        "warm_min_ms",
        "warm_max_ms",
        "first_call_ms",
        "mean_abs_error",
        "mean_error",
    ]
    with (OUT / "results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns + fields)
        for suite, groups in suites.items():
            for key in sorted(groups):
                stats = summary(groups[key])
                writer.writerow([suite, *key, *(stats[f] for f in fields)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.facecolor": "none",  # let the page show through
            "svg.fonttype": "none",
            "svg.hashsalt": "polypix",  # stable SVG ids, so reruns diff cleanly
        }
    )
    cross_env, cross_grid = load("cross-grid")
    healpix_env, healpix_large = load("healpix-large")
    for theme, site in SITE.items():
        colours = {
            "text.color": site["text"],
            "axes.labelcolor": site["text"],
            "xtick.color": site["text"],
            "ytick.color": site["text"],
            "axes.facecolor": site["surface"],
            "grid.color": site["background"],
            "legend.labelcolor": site["text"],
        }
        with plt.rc_context(colours):
            grids_chart(cross_grid, cross_env["versions"], theme)
            for workload in WORKLOADS:
                healpix_chart(workload, healpix_large, healpix_env["versions"], theme)
    write_csv({"cross-grid": cross_grid, "healpix-large": healpix_large})


if __name__ == "__main__":
    main()
