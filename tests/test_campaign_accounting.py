"""The general rule reduces to the pinned one, and the accounting books what the rule assumes."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from churn.config import ECONOMICS
from churn.economics import (
    break_even_probability, campaign_value, customer_value, decide, expected_value_of_contacting,
)


def test_zero_deadweight_is_the_original_formula():
    value = customer_value(np.array([40.0, 65.0]))
    offer = np.array([40.0, 65.0])
    expected = ECONOMICS.contact_cost / (ECONOMICS.acceptance_rate * (value - offer))
    np.testing.assert_allclose(break_even_probability(value, offer, deadweight=0.0), expected)


def test_deadweight_raises_the_bar():
    """A discount that loyal customers also take makes every contact dearer."""
    value, offer = customer_value(np.array([40.0])), np.array([40.0])
    assert break_even_probability(value, offer, deadweight=0.35) > break_even_probability(value, offer)


def test_decision_flips_at_the_break_even_with_deadweight():
    charges, offer = np.array([47.9]), np.array([47.9])
    p_star = float(break_even_probability(customer_value(charges), offer, deadweight=0.35)[0])
    assert 0 < p_star < 1
    below = decide(np.array([p_star - 1e-6]), charges, offer, deadweight=0.35)["contact"].iloc[0]
    above = decide(np.array([p_star + 1e-6]), charges, offer, deadweight=0.35)["contact"].iloc[0]
    assert not below and above


def _decisions(contact, value=500.0, offer=40.0):
    n = len(contact)
    return pd.DataFrame({"contact": contact, "value_at_risk": np.full(n, value), "offer_cost": np.full(n, offer)})


def test_contacting_nobody_is_worth_exactly_zero():
    result = campaign_value(_decisions([False] * 4), np.array([1, 1, 0, 0]), draw=np.zeros(4))
    assert result["net_value"] == 0 and result["contacted"] == 0


def test_a_loyal_customer_who_takes_the_offer_costs_offer_plus_contact():
    result = campaign_value(_decisions([True]), np.array([0]), draw=np.array([0.0]))
    assert result["customers_saved"] == 0
    assert result["net_value"] == pytest.approx(-(40.0 + ECONOMICS.contact_cost))


def test_a_churner_who_accepts_is_saved_and_books_value_less_costs():
    result = campaign_value(_decisions([True]), np.array([1]), draw=np.array([0.0]))
    assert result["customers_saved"] == 1
    assert result["net_value"] == pytest.approx(500.0 - 40.0 - ECONOMICS.contact_cost)


def test_a_churner_who_declines_costs_only_the_contact():
    result = campaign_value(_decisions([True]), np.array([1]), draw=np.array([0.99]))
    assert result["net_value"] == pytest.approx(-ECONOMICS.contact_cost)


def test_rule_and_accounting_agree_on_average():
    """Over many simulated customers, realised value per contact converges to the rule's expectation."""
    n, p, value, offer = 400_000, 0.3, 500.0, 40.0
    rng = np.random.default_rng(1)
    churned = rng.random(n) < p
    result = campaign_value(_decisions(np.ones(n, bool), value, offer), churned, draw=rng.random(n))
    expected = expected_value_of_contacting(np.array([p]), np.array([value]), np.array([offer]),
                                            deadweight=ECONOMICS.non_churner_acceptance)[0]
    assert result["net_value"] / n == pytest.approx(expected, rel=0.02)


def test_accounting_uses_the_declared_rates():
    economics = replace(ECONOMICS, acceptance_rate=1.0, non_churner_acceptance=0.0)
    result = campaign_value(_decisions([True, True]), np.array([1, 0]), economics, draw=np.array([0.5, 0.5]))
    assert result["customers_saved"] == 1
    assert result["offer_spend"] == pytest.approx(40.0)
