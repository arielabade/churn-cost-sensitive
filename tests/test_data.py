import numpy as np
import pandas as pd

from churn.data import features, load, split


def test_blank_total_charges_belong_to_brand_new_customers():
    """11 rows have a blank TotalCharges. All of them have tenure 0.

    Filling with the column mean would invent eleven months of billing history
    for customers who have none.
    """
    frame = load()
    assert frame["TotalCharges"].notna().all()
    assert (frame.loc[frame["tenure"] == 0, "TotalCharges"] == 0).all()


def test_target_is_binary_and_the_original_column_is_gone():
    frame = load()
    assert "Churn" not in frame.columns
    assert set(frame["churned"].unique()) == {0, 1}


def test_customer_id_is_not_a_feature():
    assert "customerID" not in features(load()).columns


def test_split_preserves_the_churn_rate():
    frame = load()
    train, test = split(frame)
    assert abs(train["churned"].mean() - test["churned"].mean()) < 0.02
