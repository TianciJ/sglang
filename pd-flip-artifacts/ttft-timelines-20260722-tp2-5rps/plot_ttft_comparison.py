#!/usr/bin/env python3
"""Render baseline and state-machine TTFT timelines on a shared scale."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


HERE = Path(__file__).resolve().parent
C_LONG = "#167C80"
C_SHORT = "#D97706"
C_NEUTRAL = "#30343B"


def request_index(request_id: str) -> int:
    return int(request_id.rsplit("-", 1)[1])


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    required = {"request_id", "prompt_kind", "start_monotonic", "ttft_s"}
    for line_number, row in enumerate(rows, 1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
        if row["ttft_s"] < 0:
            raise ValueError(f"{path}:{line_number}: negative ttft_s")
    return sorted(rows, key=lambda row: request_index(row["request_id"]))


def normalize(rows: list[dict]) -> list[dict]:
    origin = min(row["start_monotonic"] for row in rows)
    return [
        {
            "index": request_index(row["request_id"]),
            "kind": row["prompt_kind"],
            "arrival": row["start_monotonic"] - origin,
            "prefill_complete": row["start_monotonic"] + row["ttft_s"] - origin,
            "ttft_s": row["ttft_s"],
        }
        for row in rows
    ]


datasets = [
    (
        "Clean upstream baseline (static 1P3D)",
        normalize(load_rows(HERE / "baseline_request_metrics.jsonl")),
    ),
    (
        "PD Flip state machine (1P3D to 2P2D)",
        normalize(load_rows(HERE / "state_machine_request_metrics.jsonl")),
    ),
]

expected_ids = [row["index"] for row in datasets[0][1]]
if len(expected_ids) != 40 or [row["index"] for row in datasets[1][1]] != expected_ids:
    raise ValueError("the two inputs must contain the same 40 request indices")

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "text.usetex": False,
        "axes.titleweight": "bold",
    }
)

fig, axes = plt.subplots(1, 2, figsize=(17, 12.5), sharex=True, sharey=True)
latest = max(row["prefill_complete"] for _, rows in datasets for row in rows)

for ax, (title, rows) in zip(axes, datasets):
    for row in rows:
        color = C_LONG if row["kind"] == "long" else C_SHORT
        y = row["index"]
        ax.plot(
            [row["arrival"], row["prefill_complete"]],
            [y, y],
            color=color,
            linewidth=2.15,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.scatter(
            row["arrival"],
            y,
            s=34,
            marker="o",
            facecolor="white",
            edgecolor=color,
            linewidth=1.45,
            zorder=3,
        )
        ax.scatter(
            row["prefill_complete"],
            y,
            s=34,
            marker="o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )

    mean_ttft = sum(row["ttft_s"] for row in rows) / len(rows)
    sorted_ttft = sorted(row["ttft_s"] for row in rows)
    p95_ttft = sorted_ttft[37]
    ax.set_title(
        f"{title}\nMean TTFT {mean_ttft * 1000:.1f} ms · P95 {p95_ttft * 1000:.1f} ms",
        fontsize=12.5,
        pad=12,
    )
    ax.set_xlim(-0.25, latest + 0.35)
    ax.set_ylim(max(expected_ids) + 0.75, min(expected_ids) - 0.75)
    ax.set_yticks(expected_ids)
    ax.set_yticklabels([f"{index:02d}" for index in expected_ids], fontsize=8.2)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.set_xlabel("Time since first actual request start (s)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", which="major", color="#C9CDD2", linewidth=0.8, alpha=0.75)
    ax.grid(axis="x", which="minor", color="#E7E9EC", linewidth=0.45, alpha=0.65)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="both", which="major", direction="out", length=4, width=0.8)
    ax.tick_params(axis="x", which="minor", direction="out", length=2, width=0.55)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#50545A")

axes[0].set_ylabel("Request index", fontsize=11, fontweight="bold")
fig.suptitle("Request Arrival to Prefill Completion", fontsize=16, fontweight="bold", y=0.995)

legend_handles = [
    mlines.Line2D([], [], color=C_LONG, linewidth=2.4, label="Long request"),
    mlines.Line2D([], [], color=C_SHORT, linewidth=2.4, label="Short request"),
    mlines.Line2D(
        [], [], color=C_NEUTRAL, marker="o", markerfacecolor="white",
        markeredgewidth=1.4, linewidth=0, label="Actual arrival (start_monotonic)"
    ),
    mlines.Line2D(
        [], [], color=C_NEUTRAL, marker="o", markerfacecolor=C_NEUTRAL,
        markeredgecolor="white", linewidth=0,
        label="Prefill complete (arrival + TTFT)"
    ),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.958),
    ncol=4,
    fontsize=9.2,
    frameon=True,
    facecolor="white",
    edgecolor="#A8ADB3",
    framealpha=1.0,
)

fig.tight_layout(rect=(0, 0, 1, 0.935), pad=1.0)
output = HERE / "baseline_vs_state_machine_ttft_timeline.png"
fig.savefig(output, dpi=300, facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"saved: {output}")
