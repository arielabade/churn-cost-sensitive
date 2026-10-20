"""No feature may know anything that happened at or after its cut-off.

The test builds the world twice: once with all the data, once with the data
rewritten as it stood at the cut-off (later rows deleted, later timestamps
blanked, end-of-data snapshot columns scrambled). Every feature of every
subscriber at that cut-off must be identical in both worlds. Only the label
may differ, because only the label is allowed to look forward.
"""

import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest

from conftest import build_frame
from fixture_cinelume import truncate_at

from churn import features
from churn.config import PARQUET_DIR
from churn.model import temporal_split

CUTOFFS = (dt.date(2024, 6, 1), dt.date(2024, 11, 1), dt.date(2025, 2, 1))
#: The fixture spans 2024-01 .. 2025-03, so it gets its own split dates.
FIXTURE_SPLIT = (dt.date(2024, 9, 1), dt.date(2024, 12, 1))


def _assert_same_features(full: pd.DataFrame, truncated: pd.DataFrame) -> None:
    keys = ["cutoff", "conta_id"]
    full = full.sort_values(keys).reset_index(drop=True)
    truncated = truncated.sort_values(keys).reset_index(drop=True)
    assert len(full) > 0
    assert full[keys].equals(truncated[keys]), "the population at the cut-off depends on the future"
    for column in features.FEATURES:
        left, right = full[column], truncated[column]
        if column in features.CATEGORICAL:
            assert left.astype(str).equals(right.astype(str)), f"{column} leaks the future"
        else:
            np.testing.assert_allclose(left.to_numpy(float), right.to_numpy(float), rtol=1e-9, equal_nan=True,
                                       err_msg=f"{column} leaks the future")


@pytest.mark.parametrize("cutoff", CUTOFFS, ids=str)
def test_features_are_identical_when_the_future_is_deleted(fixture_dir, tmp_path, cutoff):
    full = build_frame(fixture_dir, cutoff, cutoff)
    past_only = truncate_at(fixture_dir, tmp_path / "past", dt.datetime.combine(cutoff, dt.time()))
    _assert_same_features(full, build_frame(past_only, cutoff, cutoff))


def test_the_truncation_really_removes_the_future(fixture_dir, fixture_frame, tmp_path):
    """Guard the guard: the truncated world must actually differ, or the test above proves nothing."""
    busiest = fixture_frame.groupby("cutoff")["churn"].sum().idxmax()
    cutoff = busiest.to_pydatetime()
    past = truncate_at(fixture_dir, tmp_path / "past", cutoff)
    assert len(pd.read_parquet(past / "reproducoes.parquet")) < len(pd.read_parquet(fixture_dir / "reproducoes.parquet"))
    full_label = build_frame(fixture_dir, cutoff.date(), cutoff.date())["churn"].sum()
    past_label = build_frame(past, cutoff.date(), cutoff.date())["churn"].sum()
    assert full_label > 0 and past_label == 0, "with the future deleted, nobody can be seen cancelling"


def test_splits_are_strictly_ordered_in_time(fixture_frame):
    split = temporal_split(fixture_frame, *FIXTURE_SPLIT)
    assert split.train["cutoff"].max() < split.valid["cutoff"].min()
    assert split.valid["cutoff"].min() <= split.valid["cutoff"].max() < split.test["cutoff"].min()


def test_last_training_label_closes_before_validation_starts(fixture_frame):
    """A training row's label covers one month after its cut-off; it must end before validation begins."""
    split = temporal_split(fixture_frame, *FIXTURE_SPLIT)
    last_label_end = split.train["cutoff"].max() + pd.DateOffset(months=1)
    assert last_label_end <= split.valid["cutoff"].min()


@pytest.mark.skipif(
    not (PARQUET_DIR / "contas.parquet").exists() or not os.environ.get("CINELUME_REAL_LEAKAGE_TEST"),
    reason="real-data variant: set CINELUME_REAL_LEAKAGE_TEST=1 after generating the data",
)
def test_no_leakage_on_the_generated_data(tmp_path):
    cutoff = dt.date(2024, 4, 1)
    full = build_frame(PARQUET_DIR, cutoff, cutoff)
    past = truncate_at(PARQUET_DIR, tmp_path / "past", dt.datetime.combine(cutoff, dt.time()))
    _assert_same_features(full, build_frame(past, cutoff, cutoff))
