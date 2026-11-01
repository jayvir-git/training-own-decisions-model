"""Render the rule-test summary as docs/figures/rule-tests.png.

  python -m uv run --no-project --with matplotlib python -B scripts/make_rule_tests_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "rule-tests" / "summary.json"
OUT = ROOT / "docs" / "figures" / "rule-tests.png"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PASS, WEAK = "#2a78d6", "#e34948"
LABELS = {  # what each edit was, in plain words
    "sponsorship": "Add “no visa sponsorship” → reject",
    "clearance": "Add “US citizens / clearance” → reject",
    "invariance": "Irrelevant edit → verdict unchanged",
    "years_up": "Add “5+ years required” → reject",
    "years_down": "3+ years → 1–2 years → not reject",
    "preferred": "Add “5+ years preferred” → unchanged",
}


# Published (passes, n) from training/RULE-TESTS.md, used when the private
# outputs/rule-tests/summary.json is absent (e.g. a public clone).
PUBLISHED = {"years_up": (75, 84), "years_down": (4, 9), "preferred": (10, 84),
             "sponsorship": (83, 83), "clearance": (82, 82), "invariance": (235, 239)}


def main():
    if SUMMARY.exists():
        families = json.loads(SUMMARY.read_text(encoding="utf-8"))["families"]
    else:
        families = {k: {"n": n, "pass_rate": p / n} for k, (p, n) in PUBLISHED.items()}
    order = [f for f in LABELS if f in families]
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 13, "text.color": INK})
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(length=0)
    for y, name in enumerate(reversed(order)):
        f = families[name]
        rate = f["pass_rate"] or 0
        ax.barh(y, rate, 0.56, color=PASS if rate >= 0.8 else WEAK, zorder=2)
        ax.text(rate + 0.015, y, f"{rate:.0%}  (n={f['n']})", va="center", fontsize=12, color=MUTED)
    ax.set_yticks(range(len(order)), [LABELS[n] for n in reversed(order)], color=INK)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([])
    fig.suptitle("Edit one sentence, check the model follows the rule", x=0.03, ha="left",
                 fontsize=15, fontweight="semibold")
    fig.text(0.03, 0.885, "Share of held-out postings where the answer moved as the policy says. "
             "Red = rule not learned.", fontsize=11.5, color=MUTED)
    fig.subplots_adjust(top=0.83, left=0.38, right=0.98, bottom=0.04)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)


if __name__ == "__main__":
    main()
