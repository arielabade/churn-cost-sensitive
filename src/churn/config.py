"""Retention economics.

A churn model does not decide who leaves. It decides who gets a retention
offer, and that is a question about money: the offer costs something, it works
only sometimes, and the customer it saves is worth their remaining margin.

Choosing a threshold by accuracy ignores all three. These numbers are what make
the threshold a business decision instead of a default.
"""

from __future__ import annotations

from dataclasses import dataclass

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
EXPECTED_ROWS = 7_043
TARGET = "Churn"
RANDOM_SEED = 20261002


@dataclass(frozen=True)
class RetentionEconomics:
    """What an intervention costs and what it saves.

    offer_cost
        Cost of the retention offer itself: a discount, a credit, an upgrade.
        Spent on everyone contacted, including the ones who were never leaving.
    contact_cost
        Agent time per contact, spent whether or not the offer is accepted.
    acceptance_rate
        Share of genuinely at-risk customers the offer actually retains. The
        single most optimistic assumption in any retention business case, and
        the one least often stated.
    gross_margin
        Share of revenue that is contribution, not revenue.
    expected_remaining_months
        Horizon over which a saved customer keeps paying. Beyond this the
        saving is speculative.
    """

    offer_cost: float = 45.0
    contact_cost: float = 6.0
    acceptance_rate: float = 0.35
    gross_margin: float = 0.55
    expected_remaining_months: int = 12


ECONOMICS = RetentionEconomics()

# Columns that describe the account, not the person.
NUMERIC = ("tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen")
CATEGORICAL = (
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
)
