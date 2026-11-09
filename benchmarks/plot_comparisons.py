"""Render the recorded matrix as SVG charts and a downloadable results table."""

import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import palette

COLORS = [
    palette.NAVY_DEEP,
    palette.CYAN,
    palette.SECONDARY,
    palette.MISSED_FILL,
    palette.COVERED_CENTER,
    palette.GRID_EDGE,
    palette.MAP_MUTED,
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
NAMES = [
    "polypix-serial",
    "polypix-auto",
    "healpy",
    "hpgeom",
    "mocpy",
    "h3",
    "s2geometry",
]


def summarize(report, backend, workflow, cold=False):
    rows = [
        r
        for r in report["records"]
        if r["backend"] == backend and r["workflow"] == workflow
    ]
    if not rows or any("error" in row for row in rows):
        raise ValueError(f"Missing or failed results: {backend}/{workflow}")
    key = "cold_seconds" if cold else "seconds"
    values = [r[key] for r in rows]
    return statistics.median(values), min(values), max(values)


def style(ax):
    ax.set_facecolor(palette.MAP_PANEL)
    ax.grid(axis="y", color="white", linewidth=1.3)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, colors=palette.MAP_TEXT)


def main():
    reports = {
        p.stem: json.loads(p.read_text())
        for p in sorted(
            [
                *Path("benchmarks/results").glob("comparison-*.json"),
                *Path("benchmarks/results").glob("large-*.json"),
            ]
        )
    }
    if len(reports) != 9:
        raise ValueError("Run the complete nine-report matrix first")
    out = Path("docs/assets/benchmarks")
    out.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "text.color": palette.MAP_TEXT,
            "axes.labelcolor": palette.MAP_TEXT,
            "svg.fonttype": "none",
        }
    )
    versions = next(iter(reports.values()))["versions"]
    build_path = Path("benchmarks/results/polypix-build.json")
    if build_path.exists():
        versions["polypix"] = (
            versions.get("polypix") or json.loads(build_path.read_text())["version"]
        )
    labels = [
        name.replace("polypix-", "Polypix ")
        + " "
        + str(
            versions.get("polypix" if name.startswith("polypix") else name)
            or "checkout"
        )
        for name in NAMES
    ]
    for workflow in ("visibility", "observation"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
        for ax, cold in zip(axes, (True, False), strict=True):
            style(ax)
            for i, backend in enumerate(NAMES[:4]):
                stats = [
                    summarize(
                        reports[f"large-{workflow}-{order}"], backend, workflow, cold
                    )
                    for order in (5, 6, 7)
                ]
                # Throughput is whole workload completions per second, not regions/s.
                med = np.array([1 / s[0] for s in stats])
                errors = np.array(
                    [
                        [1 / s[0] - 1 / s[2] for s in stats],
                        [1 / s[1] - 1 / s[0] for s in stats],
                    ]
                )
                ax.bar(
                    np.arange(3) + (i - 1.5) * 0.19,
                    med,
                    width=0.18,
                    color=COLORS[i],
                    label=labels[i],
                    yerr=errors,
                    error_kw={"elinewidth": 0.7, "capsize": 2},
                )
            ax.set_xticks(
                range(3),
                [
                    "Order 5\n12,288 cells",
                    "Order 6\n49,152 cells",
                    "Order 7\n196,608 cells",
                ],
            )
            ax.set_title("First call" if cold else "Warmed calls")
            ax.set_ylim(bottom=0)
        axes[0].set_ylim(0, max(ax.dataLim.ymax for ax in axes) * 1.1)
        axes[0].set_ylabel("Workloads per second")
        workload = (
            "10,000 visibility caps"
            if workflow == "visibility"
            else "4,000 observation footprints"
        )
        fig.suptitle(f"{workload} — higher is better", x=0.08, ha="left", fontsize=17)
        axes[1].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            frameon=False,
            title="Library / version",
        )
        fig.subplots_adjust(left=0.08, right=0.73, bottom=0.19, top=0.81, wspace=0.07)
        fig.text(
            0.08,
            0.04,
            "Median of 5 processes; whiskers show min–max. Warm: median of 5 calls per process.",
            fontsize=10,
        )
        fig.savefig(out / f"{workflow}-throughput.svg")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        style(ax)
        for i, backend in enumerate(NAMES):
            xs, ys = [], []
            for order in (5, 6, 7):
                report = reports[f"comparison-{order}"]
                rows = [
                    r
                    for r in report["records"]
                    if r["backend"] == backend and r["workflow"] == workflow
                ]
                xs.append(rows[0]["mean_absolute_sample_error"])
                ys.append(summarize(report, backend, workflow)[0] * 1000)
            ax.plot(xs, ys, marker=MARKERS[i], color=COLORS[i], label=labels[i])
            for x, y, order in zip(xs, ys, (5, 6, 7), strict=True):
                ax.annotate(
                    str(order),
                    (x, y),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set_yscale("log")
        ax.set_xlabel(
            "Mean absolute count error"
            if workflow == "visibility"
            else "Sample misclassification fraction"
        )
        ax.set_ylabel("Warmed time (ms, logarithmic)")
        ax.set_title(
            f"{workflow.capitalize()}: speed and accuracy — lower / left is better",
            loc="left",
            pad=20,
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
        fig.subplots_adjust(left=0.10, right=0.70, bottom=0.19, top=0.85)
        fig.text(
            0.10,
            0.04,
            "128 caps / 512 footprints; 20,000 common samples. Labels: requested HEALPix order.",
            fontsize=9,
        )
        fig.savefig(out / f"{workflow}-accuracy.svg")
        plt.close(fig)
    with (out / "results.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case",
                "backend",
                "workflow",
                "warm_median_ms",
                "warm_min_ms",
                "warm_max_ms",
                "cold_median_ms",
                "median_peak_rss_mib",
                "sample_mae",
                "global_cells",
            ]
        )
        for name, report in reports.items():
            for backend, workflow in sorted(
                {(r["backend"], r["workflow"]) for r in report["records"]}
            ):
                row = next(
                    r
                    for r in report["records"]
                    if r["backend"] == backend and r["workflow"] == workflow
                )
                stats = summarize(report, backend, workflow)
                writer.writerow(
                    [
                        name,
                        backend,
                        workflow,
                        *[s * 1000 for s in stats],
                        summarize(report, backend, workflow, True)[0] * 1000,
                        statistics.median(
                            r["peak_process_rss_bytes"]
                            for r in report["records"]
                            if r["backend"] == backend and r["workflow"] == workflow
                        )
                        / 2**20,
                        row["mean_absolute_sample_error"],
                        row["global_cell_count"],
                    ]
                )


if __name__ == "__main__":
    main()
