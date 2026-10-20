"""Evaluation: the same probabilities under five contact rules, in money.

Role in the pipeline
--------------------
Fifth stage. Takes calibrated probabilities for the test months and asks one
question five ways: *whom do we contact?* Each answer is booked with the
same campaign accounting and the same simulated acceptance, so the only thing
that differs between rows of the comparison is the rule.

Receives: the test frame (with ``receita_mensal`` and ``custo_oferta`` per
row), calibrated probabilities, the F1 threshold chosen on validation months.
Returns: DataFrames of campaign results overall, by segment, under
alternative economics, and a bootstrap interval for the gap that matters.
Depends on: :mod:`churn.economics` (rule and accounting), :mod:`churn.config`.

The five rules
--------------
``limiar_0.5``         the default of ``predict()``; no business input at all.
``melhor_f1``          the cut-off maximising F1 on the validation months.
                       Tuned, but blind to money.
``valor_esperado``     contact when this customer's expected value is positive,
                       i.e. when risk clears their own break-even bar.
``contatar_todos``     the ceiling on customers saved and the floor on value.
``contatar_ninguem``   zero by construction; the bar every campaign must clear.

Each test month is booked as an independent monthly campaign. A subscriber
can therefore be contacted in several months; a real programme would add a
cool-down, which would cost every rule roughly alike.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import (
    ACCEPTANCE_GRID, DEADWEIGHT_GRID, ECONOMICS, OFFER_MULTIPLIER_GRID, RANDOM_SEED, RetentionEconomics,
)
from .economics import acceptance_draw, campaign_value, customer_value, decide

RULES = ("limiar_0.5", "melhor_f1", "valor_esperado", "contatar_todos", "contatar_ninguem")
RULE_LABELS = {
    "limiar_0.5": "Limiar padrão 0,5",
    "melhor_f1": "Melhor F1",
    "valor_esperado": "Valor esperado por cliente",
    "contatar_todos": "Contatar todos",
    "contatar_ninguem": "Contatar ninguém",
}


def best_f1_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """F1-optimal global cut-off, searched on a grid.

    Called on the validation months, never on test: choosing the baseline's
    threshold on the data it is scored on would flatter the baseline.
    """
    grid = np.round(np.arange(0.01, 0.951, 0.005), 3)
    f1 = np.array([f1_score(y, probability >= t, zero_division=0) for t in grid])
    best = int(np.argmax(f1))
    return float(grid[best]), float(f1[best])


def decisions_for(
    frame: pd.DataFrame,
    probability: np.ndarray,
    f1_threshold: float,
    economics: RetentionEconomics = ECONOMICS,
    offer_multiplier: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """One decisions table per rule, all sharing values and offer costs.

    The expected-value rule is told the declared deadweight, because the
    accounting that judges it charges that deadweight.
    """
    offer = frame["custo_oferta"].to_numpy(dtype=float) * offer_multiplier
    base = decide(probability, frame["receita_mensal"].to_numpy(dtype=float), offer,
                  deadweight=economics.non_churner_acceptance, economics=economics)
    contact = {
        "limiar_0.5": probability >= 0.5,
        "melhor_f1": probability >= f1_threshold,
        "valor_esperado": base["contact"].to_numpy(),
        "contatar_todos": np.ones(len(frame), dtype=bool),
        "contatar_ninguem": np.zeros(len(frame), dtype=bool),
    }
    return {rule: base.assign(contact=mask) for rule, mask in contact.items()}


def compare(
    frame: pd.DataFrame,
    probability: np.ndarray,
    f1_threshold: float,
    economics: RetentionEconomics = ECONOMICS,
    offer_multiplier: float = 1.0,
    draw: np.ndarray | None = None,
) -> pd.DataFrame:
    """Campaign result per rule on the given rows."""
    draw = acceptance_draw(len(frame)) if draw is None else draw
    churned = frame["churn"].to_numpy()
    rows = []
    for rule, decisions in decisions_for(frame, probability, f1_threshold, economics, offer_multiplier).items():
        result = campaign_value(decisions, churned, economics, draw)
        rows.append({"regra": rule, **result})
    return pd.DataFrame(rows).set_index("regra")


def by_segment(
    frame: pd.DataFrame, probability: np.ndarray, f1_threshold: float, segment: str,
    draw: np.ndarray | None = None,
) -> pd.DataFrame:
    """The comparison repeated inside each value of ``segment``.

    The acceptance draw is computed once on the full test set and then sliced,
    so a customer has the same luck in the overall table and in their segment.
    """
    draw = acceptance_draw(len(frame)) if draw is None else draw
    parts = []
    for value, index in frame.groupby(segment, observed=True).indices.items():
        result = compare(frame.iloc[index], probability[index], f1_threshold, draw=draw[index])
        result["segmento"] = segment
        result["valor"] = str(value.date()) if hasattr(value, "date") else str(value)
        result["clientes"] = len(index)
        result["churners"] = int(frame["churn"].to_numpy()[index].sum())
        parts.append(result.reset_index())
    return pd.concat(parts, ignore_index=True)


def sensitivity(
    frame: pd.DataFrame, probability: np.ndarray, f1_threshold: float, draw: np.ndarray | None = None,
) -> pd.DataFrame:
    """Net value per rule as the two shakiest assumptions move.

    The expected-value rule re-derives its thresholds under each scenario (it
    is told the new economics); the fixed-threshold rules cannot, which is the
    point being measured.
    """
    draw = acceptance_draw(len(frame)) if draw is None else draw
    scenarios = [(a, m, ECONOMICS.non_churner_acceptance) for a in ACCEPTANCE_GRID for m in OFFER_MULTIPLIER_GRID]
    # Deadweight moves the answer more than either headline assumption, so
    # it gets its own sweep at the declared acceptance and offer.
    scenarios += [(ECONOMICS.acceptance_rate, 1.0, d) for d in DEADWEIGHT_GRID
                  if d != ECONOMICS.non_churner_acceptance]
    rows = []
    for acceptance, multiplier, deadweight in scenarios:
        economics = replace(ECONOMICS, acceptance_rate=acceptance, non_churner_acceptance=deadweight)
        result = compare(frame, probability, f1_threshold, economics, multiplier, draw)
        for rule, row in result.iterrows():
            rows.append({"taxa_aceite": acceptance, "multiplicador_oferta": multiplier, "peso_morto": deadweight,
                         "regra": rule, "net_value": row["net_value"], "contacted": row["contacted"],
                         "customers_saved": row["customers_saved"]})
    return pd.DataFrame(rows)


def contribution(decisions: pd.DataFrame, churned: np.ndarray, draw: np.ndarray,
                 economics: RetentionEconomics = ECONOMICS) -> np.ndarray:
    """Net value each row adds to the campaign, so totals can be resampled."""
    contacted = decisions["contact"].to_numpy(dtype=bool)
    churned = churned.astype(bool)
    accepted = draw < np.where(churned, economics.acceptance_rate, economics.non_churner_acceptance)
    value = decisions["value_at_risk"].to_numpy()
    offer = decisions["offer_cost"].to_numpy()
    gain = np.where(contacted & churned & accepted, value, 0.0)
    cost = np.where(contacted & accepted, offer, 0.0) + np.where(contacted, economics.contact_cost, 0.0)
    return gain - cost


def bootstrap_gap(
    frame: pd.DataFrame, probability: np.ndarray, f1_threshold: float,
    rule_a: str = "valor_esperado", rule_b: str = "melhor_f1", reps: int = 2000,
    draw: np.ndarray | None = None,
) -> dict[str, float]:
    """Bootstrap interval for net value(rule_a) - net value(rule_b).

    Resamples *subscribers*, not rows: the same subscriber appears in up to
    six test months, and their months are not independent draws. Both the
    customers and their simulated acceptance travel together.
    """
    draw = acceptance_draw(len(frame)) if draw is None else draw
    churned = frame["churn"].to_numpy()
    tables = decisions_for(frame, probability, f1_threshold)
    gap = contribution(tables[rule_a], churned, draw) - contribution(tables[rule_b], churned, draw)
    per_subscriber = pd.Series(gap).groupby(frame["conta_id"].to_numpy()).sum().to_numpy()

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(per_subscriber)
    totals = np.array([per_subscriber[rng.integers(0, n, n)].sum() for _ in range(reps)])
    return {
        "gap": float(per_subscriber.sum()),
        "ci_low": float(np.quantile(totals, 0.025)),
        "ci_high": float(np.quantile(totals, 0.975)),
        "share_positive": float((totals > 0).mean()),
        "subscribers": int(n),
    }


def break_even_summary(frame: pd.DataFrame) -> dict[str, float]:
    """Spread of break-even probabilities across the test rows."""
    decisions = decide(np.zeros(len(frame)), frame["receita_mensal"].to_numpy(dtype=float),
                       frame["custo_oferta"].to_numpy(dtype=float),
                       deadweight=ECONOMICS.non_churner_acceptance)
    threshold = decisions["break_even_probability"].replace(np.inf, np.nan)
    value = customer_value(frame["receita_mensal"].to_numpy(dtype=float))
    return {
        "p10": float(threshold.quantile(0.10)),
        "median": float(threshold.median()),
        "p90": float(threshold.quantile(0.90)),
        "min": float(threshold.min()),
        "max": float(threshold.max()),
        "never_worth_contacting": int(np.isinf(decisions["break_even_probability"]).sum()),
        "value_at_risk_p10": float(np.quantile(value, 0.10)),
        "value_at_risk_p90": float(np.quantile(value, 0.90)),
    }
