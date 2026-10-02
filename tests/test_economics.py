"""The decision rule is the product. It is tested directly."""

import numpy as np
import pytest

from churn.config import ECONOMICS
from churn.economics import (
    break_even_probability, customer_value, decide, expected_value_of_contacting,
)


def test_customer_value_is_margin_over_the_horizon():
    # 100/month at 55% margin for 12 months.
    assert customer_value(np.array([100.0]))[0] == pytest.approx(100 * 0.55 * 12)


def test_contacting_a_customer_who_will_not_churn_loses_the_contact_cost():
    value = customer_value(np.array([100.0]))
    assert expected_value_of_contacting(np.array([0.0]), value)[0] == pytest.approx(
        -ECONOMICS.contact_cost
    )


def test_break_even_threshold_is_lower_for_more_valuable_customers():
    """The argument against a single global cutoff, in one assertion."""
    cheap = break_even_probability(customer_value(np.array([25.0])))
    pricey = break_even_probability(customer_value(np.array([110.0])))
    assert pricey < cheap


def test_a_customer_worth_less_than_the_offer_is_never_worth_contacting():
    """Saving them costs more than they are worth. The threshold is infinite."""
    tiny = np.array([ECONOMICS.offer_cost / 2])
    assert np.isinf(break_even_probability(tiny))


def test_decision_flips_exactly_at_the_break_even_probability():
    charges = np.array([80.0])
    threshold = float(break_even_probability(customer_value(charges))[0])
    just_below = decide(np.array([threshold - 1e-6]), charges)["contact"].iloc[0]
    just_above = decide(np.array([threshold + 1e-6]), charges)["contact"].iloc[0]
    assert not just_below
    assert just_above


def test_offer_cost_is_weighted_by_acceptance_not_charged_to_everyone():
    """The offer is only paid when taken; the contact is paid regardless."""
    value = customer_value(np.array([100.0]))
    probability = np.array([1.0])
    expected = (
        1.0 * ECONOMICS.acceptance_rate * value[0]
        - ECONOMICS.offer_cost * 1.0 * ECONOMICS.acceptance_rate
        - ECONOMICS.contact_cost
    )
    assert expected_value_of_contacting(probability, value)[0] == pytest.approx(expected)
