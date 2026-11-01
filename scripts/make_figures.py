"""Summary figures for the README / write-ups (aggregate numbers only, no postings).

  python -m uv run --no-project --with matplotlib python -B scripts/make_figures.py

matplotlib is not in the training lock, so this runs in a throwaway uv environment.
Numbers come from outputs/full-e3/eval-test.json, training/EXPERIMENTS.md (val2) and
outputs/abstention/results.json (triage_plain_model on val2).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODEL, BASELINE = "#2a78d6", "#b8b7b1"

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 13, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


def macro_f1():
    groups = ["Test set\n(63 postings, scored once)", "Unseen employers\n(175 postings)"]
    model, base = [0.683, 0.609], [0.278, 0.287]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200, facecolor=SURFACE)
    style(ax)
    w = 0.34
    for i, (m, b) in enumerate(zip(model, base)):
        for x, v, color in ((i - w / 2 - 0.01, b, BASELINE), (i + w / 2 + 0.01, m, MODEL)):
            ax.bar(x, v, w, color=color, zorder=2)
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=13, color=INK)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_ylabel("Verdict macro-F1 (strong / stretch / reject)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MODEL), plt.Rectangle((0, 0), 1, 1, color=BASELINE)],
              labels=["Fine-tuned 0.8B decision model", "Always say “reject”"],
              frameon=False, loc="upper right", fontsize=12)
    fig.suptitle("A 0.8B model fine-tuned on one laptop GPU beats the obvious baseline", x=0.06, ha="left",
                 fontsize=15, fontweight="semibold")
    fig.text(0.06, 0.885, "Qwen3.5-0.8B, 4-bit LoRA, typed decision head, RTX 3050 (4 GB)",
             fontsize=11.5, color=MUTED)
    fig.subplots_adjust(top=0.82, left=0.1, right=0.97, bottom=0.17)
    fig.savefig(OUT / "macro-f1.png", facecolor=SURFACE)


def triage():
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200, facecolor=SURFACE)
    style(ax)
    ax.spines["bottom"].set_visible(False)
    rows = [("All 175 unseen postings", 175, 43), ("Model sends to a person", 63, 35)]
    for y, (label, total, good) in zip((1, 0), rows):
        ax.barh(y, good, 0.5, color=MODEL, zorder=3)
        ax.barh(y, total - good - 1, 0.5, left=good + 1, color=BASELINE, zorder=2)  # surface gap
        ax.text(-3, y, label, ha="right", va="center", fontsize=13)
        ax.text(total + 3, y, f"{total} postings, {good} good", va="center", fontsize=12, color=MUTED)
    ax.set_xlim(-100, 270)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle("Reading pile cut by 64%; 35 of 43 good postings kept", x=0.04, ha="left",
                 fontsize=15, fontweight="semibold")
    fig.text(0.04, 0.8, "Blue = postings the reference reviewer rated strong or stretch", fontsize=11.5, color=MUTED)
    fig.subplots_adjust(top=0.74, left=0.02, right=0.98, bottom=0.05)
    fig.savefig(OUT / "triage.png", facecolor=SURFACE)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    macro_f1()
    triage()
