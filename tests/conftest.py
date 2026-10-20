import datetime as dt
from pathlib import Path

import pytest

from fixture_cinelume import make_cinelume

from churn import features, ingest, staging


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    """A miniature Cinelume in Parquet, built once per test session."""
    return make_cinelume(tmp_path_factory.mktemp("cinelume") / "parquet")


def build_frame(parquet_dir: Path, first=dt.date(2024, 1, 1), last=dt.date(2025, 3, 1)):
    con = ingest.connect("parquet", parquet_dir)
    staging.build(con)
    features.build(con, first, last)
    return features.load_frame(con)


@pytest.fixture(scope="session")
def fixture_frame(fixture_dir):
    return build_frame(fixture_dir)
