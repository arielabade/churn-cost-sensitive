"""Same data, same code, same numbers: twice in a row and against the committed reports."""

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import build_frame

from churn.config import ECONOMICS, PARQUET_DIR, REPORTS
from churn.economics import acceptance_draw
from churn.evaluate import best_f1_threshold, compare
from churn.model import LightGBMModel, LogisticModel, temporal_split

SPLIT = (dt.date(2024, 9, 1), dt.date(2024, 12, 1))


def _run_once(fixture_dir):
    frame = build_frame(fixture_dir)
    split = temporal_split(frame, *SPLIT)
    model = LightGBMModel().fit(split.train, split.valid)
    probability = model.predict(split.test)
    threshold, _ = best_f1_threshold(split.valid["churn"].to_numpy(), model.predict(split.valid))
    return frame, probability, compare(split.test, probability, threshold)


def test_features_model_and_campaign_are_bit_for_bit_reproducible(fixture_dir):
    frame_a, prob_a, rules_a = _run_once(fixture_dir)
    frame_b, prob_b, rules_b = _run_once(fixture_dir)
    assert frame_a.equals(frame_b)
    np.testing.assert_array_equal(prob_a, prob_b)
    assert rules_a.equals(rules_b)


def test_logistic_baseline_is_reproducible(fixture_frame):
    split = temporal_split(fixture_frame, *SPLIT)
    a = LogisticModel().fit(split.train, split.valid).predict(split.test)
    b = LogisticModel().fit(split.train, split.valid).predict(split.test)
    np.testing.assert_array_equal(a, b)


def test_acceptance_draw_is_fixed():
    """Every rule faces the same simulated customers' luck, run after run."""
    np.testing.assert_array_equal(acceptance_draw(1000), acceptance_draw(1000))


def test_committed_reports_match_the_declared_economics():
    """reports/ must have been produced with the assumptions now in config.py."""
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    for name in ("contact_cost", "acceptance_rate", "non_churner_acceptance", "gross_margin",
                 "expected_remaining_months"):
        assert metrics["economics"][name] == pytest.approx(getattr(ECONOMICS, name)), name


@pytest.mark.skipif(not (PARQUET_DIR / "contas.parquet").exists(), reason="generated data not present")
def test_committed_reports_match_the_generated_data():
    """The volume quoted in reports/ is the volume of the data on disk (same seed, same scale)."""
    import duckdb

    metrics = json.loads((REPORTS / "metrics.json").read_text())
    for table in ("contas", "assinaturas", "reproducoes", "eventos_app"):
        rows = duckdb.sql(f"SELECT count(*) FROM read_parquet('{Path(PARQUET_DIR) / table}.parquet')").fetchone()[0]
        assert rows == metrics["data"][table], table
