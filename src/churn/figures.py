"""Figures for the README, built from the committed report files.

The project's claim is that the deliverable is a decision rule rather than a
model, so the charts are about money and thresholds, not about ROC curves.

Run with ``python -m churn.figures``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import brandviz as bv

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "assets" / "figures"

#: Printable names for the strategy keys in metrics.json, in the order a
#: reader should meet them: the default first, the proposal last.
STRATEGY_LABELS = {
    "accuracy_0.5": "Default\nthreshold 0.5",
    "best_f1": "Best F1\nthreshold 0.18",
    "contact_everyone": "Contact\neveryone",
    "expected_value": "Per-customer\nbreak-even",
}
STRATEGY_ORDER = ("accuracy_0.5", "best_f1", "contact_everyone", "expected_value")


def strategy_value(metrics: dict) -> Path:
    """Realised campaign value under each decision rule.

    The bar that matters is the difference between the first and the last, so
    the first is drawn in the loss colour and the last in the accent, and the
    gap between them is annotated in money rather than left to subtraction.
    """
    strategies = metrics["strategies"]
    labels = [STRATEGY_LABELS[key] for key in STRATEGY_ORDER]
    values = np.array([strategies[key]["net_value"] for key in STRATEGY_ORDER])
    colors = [bv.COBALT if key == "expected_value" else bv.SLATE for key in STRATEGY_ORDER]
    colors[STRATEGY_ORDER.index("accuracy_0.5")] = bv.CRIMSON

    fig, ax = bv.panel(
        12.4, 5.6,
        title="The same model and the same budget, under four decision rules",
        subtitle="Net value on held-out customers: margin saved, less the cost of every offer made and every call placed",
    )
    positions = np.arange(len(values), dtype=float)
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_ylim(0, values.max() * 1.30)
    fig.canvas.draw()
    bv.bars(ax, positions, values, colors,
            labels=[f"£{value:,.0f}" for value in values])
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Net campaign value")
    ax.yaxis.set_major_formatter(lambda value, _: f"£{value / 1000:,.0f}k")

    default = strategies["accuracy_0.5"]["net_value"]
    best = strategies["expected_value"]["net_value"]
    bv.annotate(
        ax,
        f"£{best - default:,.0f} recovered, {(best - default) / default:.0%} more than the default,\n"
        f"by taking the threshold from retention economics rather than from accuracy",
        xy=(len(values) - 1, best * 1.02),
        xytext=(0.62, values.max() * 1.19), color=bv.IVORY,
    )
    bv.clean(ax)
    return bv.save(fig, FIGURES / "strategy_value.svg")


def break_even_spread(decisions: pd.DataFrame, metrics: dict) -> Path:
    """Why no single global threshold can express this decision.

    Each customer has their own break-even churn probability, because each has
    their own margin at risk. The histogram is the argument: a vertical line at
    0.5 sits far outside the entire distribution.
    """
    probabilities = decisions["break_even_probability"].to_numpy() * 100
    quantiles = metrics["break_even_probability"]

    fig, ax = bv.panel(
        12.4, 5.4,
        title="Every customer has their own break-even probability",
        subtitle="The churn probability at which a retention offer stops losing money, one value per held-out customer",
    )
    ax.hist(probabilities, bins=70, color=bv.COBALT, alpha=0.92, zorder=3)
    ax.set_xlabel("Break-even churn probability")
    ax.set_ylabel("Customers")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.set_xlim(0, max(probabilities.max() * 1.02, 52))

    bv.reference_line(ax, quantiles["p10"] * 100, f"10th pct {quantiles['p10']:.1%}",
                      horizontal=False, where=0.97, ha="left")
    bv.reference_line(ax, quantiles["p90"] * 100, f"90th pct {quantiles['p90']:.1%}",
                      horizontal=False, where=0.97, ha="left")
    ax.axvline(50, color=bv.CRIMSON, linewidth=2.0, zorder=4)
    bv.annotate(
        ax,
        "The 0.5 default sits here:\noutside the distribution entirely,\nfor every single customer",
        xy=(49.4, ax.get_ylim()[1] * 0.55),
        xytext=(34, ax.get_ylim()[1] * 0.72), color=bv.CRIMSON, ha="left",
    )
    bv.clean(ax)
    return bv.save(fig, FIGURES / "break_even_spread.svg")


def contact_decision(decisions: pd.DataFrame) -> Path:
    """The rule itself: contact when predicted risk clears this customer's bar.

    Two quantities per customer, so a scatter. The diagonal is the rule, and
    membership of a side is also shown by colour *and* named in the legend, so
    the split never rests on hue alone.
    """
    sample = decisions.sample(n=min(900, len(decisions)), random_state=20261002)
    contacted = sample["contact"].to_numpy(dtype=bool)

    fig, ax = bv.panel(
        12.4, 5.6,
        title="The decision rule, drawn",
        subtitle="A customer is contacted when their predicted churn risk clears the break-even bar their own margin sets",
    )
    # Clipped to where the customers actually are: the break-even bar never
    # exceeds ~22% here, and an axis running to 62% would be mostly empty.
    limit = 0.25
    ax.plot([0, limit], [0, limit], color=bv.STEEL, linewidth=1.6,
            linestyle=(0, (5, 4)), zorder=4)
    bv.annotate(ax, "risk = break-even\n(below this line, an offer loses money)",
                xy=(limit * 0.88, limit * 0.88), xytext=(limit * 0.52, 0.40),
                color=bv.STEEL, fontsize=9.5)

    for mask, colour, label in (
        (~contacted, bv.SLATE, "No offer: risk below this customer's bar"),
        (contacted, bv.COBALT, "Offer: risk clears the bar"),
    ):
        ax.scatter(sample.loc[mask, "break_even_probability"],
                   sample.loc[mask, "probability"], s=26, c=colour,
                   edgecolors=bv.CARBON, linewidths=0.8, alpha=0.9,
                   zorder=3 if colour == bv.SLATE else 5, label=label)

    ax.set_xlim(0, limit)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("This customer's break-even churn probability")
    ax.set_ylabel("Predicted churn probability")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(loc="upper right")
    bv.clean(ax, axis="both", spines=("top", "right"))
    return bv.save(fig, FIGURES / "contact_decision.svg")


def headline(metrics: dict):
    """The three numbers the README leads with."""
    strategies = metrics["strategies"]
    recovered = strategies["expected_value"]["net_value"] - strategies["accuracy_0.5"]["net_value"]
    quantiles = metrics["break_even_probability"]
    fig, _ = bv.kpi_strip([
        (f"£{recovered:,.0f}", "Recovered against the 0.5 default,\non the same model and the same budget"),
        (f"{quantiles['p10']:.1%}–{quantiles['p90']:.1%}",
         "Range of break-even churn probability\nacross customers: no global cutoff fits"),
        (f"{metrics['model']['roc_auc']:.3f}",
         "Model AUC, unchanged throughout.\nThe decision rule is the deliverable"),
    ])
    return bv.save(fig, FIGURES / "headline.svg")


def build_all() -> list[Path]:
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    decisions = pd.read_csv(REPORTS / "decisions.csv")
    return [
        headline(metrics),
        strategy_value(metrics),
        break_even_spread(decisions, metrics),
        contact_decision(decisions),
    ]


if __name__ == "__main__":
    for path in build_all():
        print(path.relative_to(ROOT))
