"""Turning a probability into a contact decision, and a decision into money.

Role in the pipeline
--------------------
This is the product. The model upstream only supplies ``P(churn)``; this
module decides who gets the retention offer and books what that decision is
worth on customers whose outcome is known.

Receives: calibrated churn probabilities, each customer's monthly revenue
(plan price in force at the cut-off, plus extra-member fees) and, optionally,
each customer's offer cost (the ``VOLTA50`` coupon priced on their plan).
Returns: per-customer decisions (``decide``) and campaign accounting
(``campaign_value``). Depends on: :mod:`churn.config` for the declared
economics only.

The algebra
-----------
For a customer with churn probability ``p``, margin at risk ``V`` and offer
cost ``O``, contacting is worth::

    E[value] = p * a * (V - O)          churner accepts: saved, offer paid
             - (1 - p) * d * O          non-churner accepts: deadweight
             - c                        contact, paid regardless

where ``a`` is the acceptance rate among churners, ``d`` among non-churners
and ``c`` the contact cost. Setting it to zero gives the break-even
probability::

    p* = (c + d * O) / (a * (V - O) + d * O)

With ``d = 0`` this is the original rule, ``c / (a * (V - O))``, and the tests
pin that case. The pipeline passes the declared ``d``, because the campaign
accounting has always charged the offer to non-churners who take it; a rule
that ignores a cost its own scorecard charges is mis-specified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ECONOMICS, RetentionEconomics


def customer_value(monthly_charges: np.ndarray, economics: RetentionEconomics = ECONOMICS) -> np.ndarray:
    """Contribution margin at risk if this customer leaves.

    Revenue is not what is lost: margin is, and only over a horizon short
    enough to be believed. Monthly revenue times margin times the horizon.
    """
    return (
        np.asarray(monthly_charges, dtype=float)
        * economics.gross_margin
        * economics.expected_remaining_months
    )


def _offer(offer_cost, economics: RetentionEconomics) -> np.ndarray | float:
    return economics.offer_cost if offer_cost is None else np.asarray(offer_cost, dtype=float)


def expected_value_of_contacting(
    probability: np.ndarray,
    value: np.ndarray,
    offer_cost: np.ndarray | float | None = None,
    deadweight: float = 0.0,
    economics: RetentionEconomics = ECONOMICS,
) -> np.ndarray:
    """Expected profit from making a retention offer to one customer.

    The offer is only paid out when accepted, so its cost is weighted by the
    chance of acceptance; the contact is paid regardless. Treating the offer as
    a certain cost understates the case, treating the gain as certain overstates
    it, and both mistakes are common.

    ``deadweight`` is the acceptance rate among customers who were not leaving.
    It defaults to 0, the textbook formula; the pipeline passes the declared
    rate, because a discount sent to a loyal customer is usually taken.
    """
    probability = np.asarray(probability, dtype=float)
    offer = _offer(offer_cost, economics)
    gain = probability * economics.acceptance_rate * value
    offer_to_churners = offer * probability * economics.acceptance_rate
    offer_to_loyal = offer * (1.0 - probability) * deadweight
    return gain - offer_to_churners - offer_to_loyal - economics.contact_cost


def break_even_probability(
    value: np.ndarray | float,
    offer_cost: np.ndarray | float | None = None,
    deadweight: float = 0.0,
    economics: RetentionEconomics = ECONOMICS,
) -> np.ndarray | float:
    """Churn probability at which contacting breaks even.

    Solving expected_value_of_contacting = 0 for p:

        p = (contact_cost + d * offer) / (acceptance * (value - offer) + d * offer)

    Note what this says: the threshold is PER CUSTOMER. A high-value customer
    is worth contacting at a far lower churn probability than a low-value one,
    which a single global cutoff cannot express.

    When the denominator is not positive, no probability makes the contact pay
    (the customer is worth less than the offer). The threshold is then
    infinite rather than negative or undefined, so ``p > threshold`` is False
    for every p and the customer is never contacted, by construction.
    """
    offer = _offer(offer_cost, economics)
    value = np.asarray(value, dtype=float)
    denominator = economics.acceptance_rate * (value - offer) + deadweight * offer
    numerator = economics.contact_cost + deadweight * offer
    with np.errstate(divide="ignore", invalid="ignore"):
        threshold = np.where(denominator > 0, numerator / denominator, np.inf)
    return threshold


def decide(
    probability: np.ndarray,
    monthly_charges: np.ndarray,
    offer_cost: np.ndarray | float | None = None,
    deadweight: float = 0.0,
    economics: RetentionEconomics = ECONOMICS,
) -> pd.DataFrame:
    """Per-customer contact decision and its expected value.

    Contact exactly when the expected value is positive, which is the same as
    the probability clearing that customer's own break-even bar.
    """
    probability = np.asarray(probability, dtype=float)
    value = customer_value(monthly_charges, economics)
    offer = np.broadcast_to(_offer(offer_cost, economics), probability.shape).astype(float)
    expected = expected_value_of_contacting(probability, value, offer, deadweight, economics)
    threshold = break_even_probability(value, offer, deadweight, economics)
    return pd.DataFrame(
        {
            "probability": probability,
            "value_at_risk": value,
            "offer_cost": offer,
            "break_even_probability": threshold,
            "expected_value": expected,
            "contact": expected > 0,
        }
    )


def acceptance_draw(n: int, seed: int = 0) -> np.ndarray:
    """One uniform draw per customer, shared by every rule being compared.

    The data cannot say who would have accepted an offer nobody made, so
    acceptance is simulated. Drawing it once and reusing it means every rule
    faces the same customers' luck: the comparison is fair even though the
    absolute figures are not a forecast.
    """
    return np.random.default_rng(seed).random(n)


def campaign_value(
    decisions: pd.DataFrame,
    actually_churned: np.ndarray,
    economics: RetentionEconomics = ECONOMICS,
    draw: np.ndarray | None = None,
) -> dict[str, float]:
    """Realised value of the campaign on held-out customers.

    Counts money only where the customer was genuinely going to churn and the
    offer was accepted. Everyone contacted still costs contact time, and the
    offer cost lands on accepted offers whether or not the customer was leaving
    — which is where naive business cases quietly lose their margin.

    Churners accept at ``acceptance_rate``; loyal customers at
    ``non_churner_acceptance``. The accounting always uses the declared rates,
    because it plays the part of the world, not of the rule.
    """
    contacted = decisions["contact"].to_numpy(dtype=bool)
    churned = np.asarray(actually_churned).astype(bool)
    draw = acceptance_draw(len(decisions)) if draw is None else np.asarray(draw)
    rate = np.where(churned, economics.acceptance_rate, economics.non_churner_acceptance)
    accepted = draw < rate

    offer = (
        decisions["offer_cost"].to_numpy(dtype=float)
        if "offer_cost" in decisions
        else np.full(len(decisions), economics.offer_cost)
    )
    saved = contacted & churned & accepted
    margin_saved = float(decisions["value_at_risk"].to_numpy()[saved].sum())
    offer_spend = float(offer[contacted & accepted].sum())
    contact_spend = float(economics.contact_cost * contacted.sum())

    return {
        "contacted": int(contacted.sum()),
        "customers_saved": int(saved.sum()),
        "margin_saved": margin_saved,
        "offer_spend": offer_spend,
        "contact_spend": contact_spend,
        "net_value": margin_saved - offer_spend - contact_spend,
    }
