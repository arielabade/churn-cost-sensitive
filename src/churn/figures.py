"""README figures, drawn from the committed report files.

Role in the pipeline
--------------------
Presentation, after the pipeline. Reads ``reports/`` and writes SVGs to
``assets/figures/``; computes no number of its own, so the README and the
figures cannot disagree with the reports. The client-facing equivalent is
:mod:`churn.dashboard`.

Receives: ``reports/metrics.json``, ``rules_by_segment.csv``,
``sensitivity.csv``, ``decisions_last_month.csv``.
Returns: the SVG paths. Depends on: :mod:`churn.brandviz` (ABADE theme).

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
from .config import REPORTS, ROOT

FIGURES = ROOT / "assets" / "figures"

#: Printable names for the rules, in the order a reader should meet them:
#: the default first, the proposal last.
RULE_LABELS = {
    "limiar_0.5": "Default\nthreshold 0.5",
    "melhor_f1": "Best-F1\nthreshold",
    "valor_esperado": "Per-customer\nexpected value",
}
RULE_ORDER = ("limiar_0.5", "melhor_f1", "valor_esperado")


def brl(value: float, signed: bool = False) -> str:
    # "\\$" because two dollar signs in one matplotlib string switch it to
    # math mode and the text between them is typeset as an equation.
    text = f"R\\${abs(value):,.0f}"
    if value < 0:
        return "−" + text
    return ("+" + text) if signed else text


def headline(metrics: dict) -> Path:
    """The three numbers the README leads with."""
    s = metrics["strategies"]
    gap_f1 = metrics["bootstrap"]["valor_esperado_vs_melhor_f1"]
    recovered = s["valor_esperado"]["net_value"] - s["limiar_0.5"]["net_value"]
    fig, _ = bv.kpi_strip([
        (brl(recovered), "Left on the table by the 0.5 default\nover six held-out months"),
        (brl(gap_f1["gap"], signed=True),
         f"Over a tuned F1 cut-off (95% CI\n{brl(gap_f1['ci_low'])} to {brl(gap_f1['ci_high'])})"),
        (f"{metrics['models']['lightgbm']['test']['roc_auc']:.3f}",
         "Model AUC on months it never saw;\ncalibrated to the observed churn rate"),
    ])
    return bv.save(fig, FIGURES / "headline.svg")


def strategy_value(metrics: dict) -> Path:
    """Realised campaign value under each rule, on the same model and months."""
    s = metrics["strategies"]
    values = np.array([s[key]["net_value"] for key in RULE_ORDER])
    colors = [bv.CRIMSON, bv.SLATE, bv.COBALT]
    everyone = s["contatar_todos"]["net_value"]

    fig, ax = bv.panel(
        12.4, 5.6,
        title="Same model, same six months: only the contact rule changes",
        subtitle=f"Net value: margin saved, less every offer taken and every contact made. "
                 f"Contacting everyone: {brl(everyone)}",
    )
    positions = np.arange(len(values), dtype=float)
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_ylim(0, values.max() * 1.32)
    fig.canvas.draw()
    labels = [f"{brl(v)}\n{int(s[k]['contacted']):,} contacted · {int(s[k]['customers_saved']):,} saved"
              for k, v in zip(RULE_ORDER, values)]
    bv.bars(ax, positions, values, colors, labels=labels)
    ax.set_xticks(positions, [RULE_LABELS[k] for k in RULE_ORDER])
    ax.set_ylabel("Net campaign value")
    ax.yaxis.set_major_formatter(lambda value, _: ("−" if value < 0 else "") + f"R\\${abs(value) / 1000:,.0f}k")
    bv.clean(ax)
    return bv.save(fig, FIGURES / "strategy_value.svg")


def by_month(segments: pd.DataFrame) -> Path:
    """The comparison repeated in each test month: where the rule wins and where it ties."""
    months = segments[segments["segmento"] == "cutoff"]
    table = months.pivot(index="valor", columns="regra", values="net_value")
    categories = [pd.Timestamp(v).strftime("%b %Y") for v in table.index]
    gap = table["valor_esperado"] - table["melhor_f1"]
    wins = int((gap > 0).sum())
    closest = pd.Timestamp(gap.idxmin()).strftime("%B")
    # The title states what the data says, computed rather than typed: a
    # re-run that changes the result changes the sentence with it.
    title = (f"The rule beats the F1 cut-off in all {len(gap)} test months, by least in {closest}"
             if wins == len(gap) else f"The rule beats the F1 cut-off in {wins} of {len(gap)} test months")
    fig, ax = bv.panel(
        12.4, 5.6,
        title=title,
        subtitle=f"Net value per monthly campaign; gap over F1 from {brl(gap.min(), True)} to {brl(gap.max(), True)}",
    )
    ax.set_xlim(-0.6, len(categories) - 0.4)
    ax.set_ylim(0, table.to_numpy().max() * 1.2)
    fig.canvas.draw()
    bv.grouped_bars(ax, categories, {
        "Per-customer expected value": table["valor_esperado"].to_numpy(),
        "Best-F1 threshold": table["melhor_f1"].to_numpy(),
        "Default 0.5": table["limiar_0.5"].to_numpy(),
    })
    ax.set_ylabel("Net value")
    ax.yaxis.set_major_formatter(lambda value, _: ("−" if value < 0 else "") + f"R\\${abs(value) / 1000:,.0f}k")
    ax.legend(loc="upper left", ncols=3)
    bv.clean(ax)
    return bv.save(fig, FIGURES / "by_month.svg")


def _sensitivity_text(base: pd.DataFrame) -> tuple[str, str]:
    """Title and subtitle stating what the sweep shows, including where the rule loses."""
    table = base.pivot(index="taxa_aceite", columns="regra", values="net_value")
    positive = bool((table["valor_esperado"] > 0).all())
    f1_negative = table.index[table["melhor_f1"] < 0]
    title = ("The rule stays positive at every acceptance rate" if positive
             else "The rule loses money at low acceptance too")
    if len(f1_negative):
        title += f"; the F1 cut-off loses money at {f1_negative.max():.0%}"
    subtitle = "Net value over six test months; offer = VOLTA50 on each plan."
    behind = table[table["limiar_0.5"] > table["valor_esperado"]]
    for rate, row in behind.iterrows():
        subtitle += (f" At {rate:.0%} the 0.5 default edges ahead "
                     f"({brl(row['limiar_0.5'])} vs {brl(row['valor_esperado'])}).")
    return title, subtitle


def sensitivity(frame: pd.DataFrame) -> Path:
    """Net value as the acceptance rate moves, offer at its data-derived cost."""
    base = frame[(frame["multiplicador_oferta"] == 1.0) & (frame["peso_morto"] == 0.35)]
    title, subtitle = _sensitivity_text(base)
    fig, ax = bv.panel(12.4, 5.6, title=title, subtitle=subtitle)
    for rule, color, label in (("valor_esperado", bv.COBALT, "Per-customer expected value"),
                               ("melhor_f1", bv.AMBER, "Best-F1 threshold"),
                               ("limiar_0.5", bv.TEAL, "Default 0.5")):
        line = base[base["regra"] == rule].sort_values("taxa_aceite")
        ax.plot(line["taxa_aceite"], line["net_value"], color=color, marker="o", label=label, zorder=4)
    ax.axhline(0, color=bv.STEEL, linewidth=1.0, zorder=2)
    ax.set_xlabel("Acceptance rate among customers who would have left")
    ax.set_ylabel("Net value")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.yaxis.set_major_formatter(lambda value, _: ("−" if value < 0 else "") + f"R\\${abs(value) / 1000:,.0f}k")
    ax.legend(loc="upper left")
    bv.clean(ax, axis="both", spines=("top", "right"))
    return bv.save(fig, FIGURES / "sensitivity.svg")


def break_even_spread(decisions: pd.DataFrame, metrics: dict) -> Path:
    """Where each group's break-even bar sits, against the two global cut-offs.

    The bar is set by plan price, billing cycle and extra-member fees, so it
    takes a handful of values rather than a continuum: drawn as one bar per
    group, sized by how many subscribers sit there.
    """
    names = {"basico_anuncios": "Basic", "padrao": "Standard", "premium": "Premium"}
    groups = (decisions.assign(bar=decisions["break_even_probability"].round(3))
              .groupby(["plano", "bar"]).size().rename("n").reset_index())
    groups = groups[groups["n"] >= 0.01 * len(decisions)].sort_values("bar")
    f1 = metrics["best_f1_threshold"]
    fig, ax = bv.panel(
        12.4, 5.4,
        title=f"The break-even bar runs from {groups['bar'].min():.0%} to {groups['bar'].max():.0%}, "
              f"set by plan and price",
        subtitle="Churn probability at which the offer stops paying, October 2026 subscribers, "
                 "by plan (groups with ≥1% of subscribers)",
    )
    positions = np.arange(len(groups), dtype=float)
    ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.set_xlim(0, 0.55)
    fig.canvas.draw()
    labels = [f"{bar:.1%}  ({n:,} subscribers)" for bar, n in zip(groups["bar"], groups["n"])]
    bv.bars(ax, positions, groups["bar"].to_numpy(), bv.COBALT, horizontal=True, labels=labels)
    ax.set_yticks(positions, [names.get(p, p) for p in groups["plano"]])
    ax.axvline(f1, color=bv.AMBER, linewidth=2.0, zorder=4)
    ax.axvline(0.5, color=bv.CRIMSON, linewidth=2.0, zorder=4)
    bv.annotate(ax, f"Best-F1 cut-off {f1:.1%}: below every bar,\nso it contacts subscribers at a loss",
                xy=(f1, 0), xytext=(0.335, len(groups) / 2 - 0.5), color=bv.AMBER, arrow=False)
    bv.annotate(ax, "0.5 default: above every bar,\nmisses subscribers worth saving",
                xy=(0.5, 0), xytext=(0.495, 0.4), color=bv.CRIMSON, arrow=False, ha="right")
    ax.set_xlabel("Break-even churn probability")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    bv.clean(ax, axis="x")
    return bv.save(fig, FIGURES / "break_even_spread.svg")


def contact_decision(decisions: pd.DataFrame) -> Path:
    """The rule itself: contact when predicted risk clears this customer's bar."""
    sample = decisions.sample(n=min(1500, len(decisions)), random_state=20261002)
    contacted = sample["contact"].to_numpy(dtype=bool)
    fig, ax = bv.panel(
        12.4, 5.6,
        title="The decision rule, drawn",
        subtitle="A subscriber gets the offer when predicted risk clears the bar their own plan and price set",
    )
    ax.plot([0, 1], [0, 1], color=bv.STEEL, linewidth=1.6, linestyle=(0, (5, 4)), zorder=4)
    for mask, colour, label in (
        (~contacted, bv.SLATE, "No offer: risk below this subscriber's bar"),
        (contacted, bv.COBALT, "Offer: risk clears the bar"),
    ):
        ax.scatter(sample.loc[mask, "break_even_probability"], sample.loc[mask, "probability"], s=26, c=colour,
                   edgecolors=bv.CARBON, linewidths=0.8, alpha=0.9, zorder=3 if colour == bv.SLATE else 5,
                   label=label)
    ax.set_xlim(0.1, 0.32)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("This subscriber's break-even churn probability")
    ax.set_ylabel("Predicted churn probability")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(loc="upper left")
    bv.clean(ax, axis="both", spines=("top", "right"))
    return bv.save(fig, FIGURES / "contact_decision.svg")


def build_all() -> list[Path]:
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    segments = pd.read_csv(REPORTS / "rules_by_segment.csv")
    sens = pd.read_csv(REPORTS / "sensitivity.csv")
    decisions = pd.read_csv(REPORTS / "decisions_last_month.csv")
    return [
        headline(metrics),
        strategy_value(metrics),
        by_month(segments),
        sensitivity(sens),
        break_even_spread(decisions, metrics),
        contact_decision(decisions),
    ]


if __name__ == "__main__":
    for path in build_all():
        print(path.relative_to(ROOT))
