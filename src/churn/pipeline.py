"""Train, then compare thresholds chosen three different ways."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

from .config import ECONOMICS, RANDOM_SEED
from .data import features, load, split
from .economics import campaign_value, customer_value, decide, expected_value_of_contacting

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def _fixed_threshold_campaign(probability, charges, churned, threshold: float) -> dict:
    """What a single global cutoff would have produced."""
    decisions = decide(probability, charges)
    decisions["contact"] = probability >= threshold
    result = campaign_value(decisions, churned)
    result["threshold"] = threshold
    return result


def run() -> dict:
    frame = load()
    train_frame, test_frame = split(frame)

    x_train, y_train = features(train_frame), train_frame["churned"].to_numpy()
    x_test, y_test = features(test_frame), test_frame["churned"].to_numpy()

    model = LGBMClassifier(
        n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=40,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85, reg_lambda=1.0,
        random_state=RANDOM_SEED, verbose=-1,
    ).fit(x_train, y_train)

    # Calibrated because the probability is multiplied by money downstream.
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic", cv=5)
    calibrated.fit(x_train, y_train)
    probability = calibrated.predict_proba(x_test)[:, 1]

    charges = test_frame["MonthlyCharges"].to_numpy()
    value_decisions = decide(probability, charges)

    # Three ways to pick who gets contacted.
    strategies = {
        # 1. The textbook default. No business input at all.
        "accuracy_0.5": _fixed_threshold_campaign(probability, charges, y_test, 0.5),
        # 2. The F1-optimal cutoff, found by sweeping. Still blind to money.
        "best_f1": None,
        # 3. Per-customer expected value: contact when it pays.
        "expected_value": {**campaign_value(value_decisions, y_test), "threshold": float("nan")},
        # 4. Contact everyone, as a floor.
        "contact_everyone": _fixed_threshold_campaign(probability, charges, y_test, 0.0),
    }

    grid = np.linspace(0.05, 0.95, 91)
    f1_scores = [f1_score(y_test, probability >= t) for t in grid]
    best_f1_threshold = float(grid[int(np.argmax(f1_scores))])
    strategies["best_f1"] = _fixed_threshold_campaign(
        probability, charges, y_test, best_f1_threshold
    )

    summary = {
        "rows": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "churn_rate": float(frame["churned"].mean()),
        "model": {
            "roc_auc": float(roc_auc_score(y_test, probability)),
            "brier": float(brier_score_loss(y_test, probability)),
            "best_f1_threshold": best_f1_threshold,
            "best_f1": float(max(f1_scores)),
        },
        "break_even_probability": {
            "median": float(value_decisions["break_even_probability"].median()),
            "p10": float(value_decisions["break_even_probability"].quantile(0.10)),
            "p90": float(value_decisions["break_even_probability"].quantile(0.90)),
        },
        "strategies": strategies,
    }

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    value_decisions.assign(churned=y_test).to_csv(REPORTS / "decisions.csv", index=False)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
