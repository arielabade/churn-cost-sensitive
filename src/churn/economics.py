"""Turning a probability into a decision.

The whole project is this file plus a model good enough to feed it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ECONOMICS


def customer_value(monthly_charges: np.ndarray) -> np.ndarray:
    """Contribution margin at risk if this customer leaves."""
    return (
        monthly_charges
        * ECONOMICS.gross_margin
        * ECONOMICS.expected_remaining_months
    )


def expected_value_of_contacting(probability: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Expected profit from making a retention offer to one customer.

        gain  = P(churn) * acceptance_rate * value_at_risk
        cost  = offer_cost * P(accept the offer) + contact_cost

    The offer is only paid out when accepted, so its cost is weighted by the
    chance of acceptance; the contact is paid regardless. Treating the offer as
    a certain cost understates the case, treating the gain as certain overstates
    it, and both mistakes are common.
    """
    gain = probability * ECONOMICS.acceptance_rate * value
    offer = ECONOMICS.offer_cost * probability * ECONOMICS.acceptance_rate
    return gain - offer - ECONOMICS.contact_cost


def break_even_probability(value: np.ndarray | float) -> np.ndarray | float:
    """Churn probability at which contacting breaks even.

    Solving expected_value_of_contacting = 0 for p:

        p = contact_cost / (acceptance_rate * (value - offer_cost))

    Note what this says: the threshold is PER CUSTOMER. A high-value customer
    is worth contacting at a far lower churn probability than a low-value one,
    which a single global cutoff cannot express.
    """
    denominator = ECONOMICS.acceptance_rate * (np.asarray(value) - ECONOMICS.offer_cost)
    with np.errstate(divide="ignore", invalid="ignore"):
        threshold = np.where(denominator > 0, ECONOMICS.contact_cost / denominator, np.inf)
    return threshold


def decide(probability: np.ndarray, monthly_charges: np.ndarray) -> pd.DataFrame:
    """Per-customer contact decision and its expected value."""
    value = customer_value(monthly_charges)
    expected = expected_value_of_contacting(probability, value)
    threshold = break_even_probability(value)
    return pd.DataFrame(
        {
            "probability": probability,
            "value_at_risk": value,
            "break_even_probability": threshold,
            "expected_value": expected,
            "contact": expected > 0,
        }
    )


def campaign_value(decisions: pd.DataFrame, actually_churned: np.ndarray) -> dict[str, float]:
    """Realised value of the campaign on held-out customers.

    Counts money only where the customer was genuinely going to churn and the
    offer was accepted. Everyone contacted still costs contact time, and the
    offer cost lands on accepted offers whether or not the customer was leaving
    — which is where naive business cases quietly lose their margin.
    """
    contacted = decisions["contact"].to_numpy()
    churned = actually_churned.astype(bool)
    rng = np.random.default_rng(0)
    accepted = rng.random(len(decisions)) < ECONOMICS.acceptance_rate

    saved = contacted & churned & accepted
    revenue_saved = float(decisions.loc[saved, "value_at_risk"].sum())
    offer_spend = float(ECONOMICS.offer_cost * (contacted & accepted).sum())
    contact_spend = float(ECONOMICS.contact_cost * contacted.sum())

    return {
        "contacted": int(contacted.sum()),
        "customers_saved": int(saved.sum()),
        "margin_saved": revenue_saved,
        "offer_spend": offer_spend,
        "contact_spend": contact_spend,
        "net_value": revenue_saved - offer_spend - contact_spend,
    }
