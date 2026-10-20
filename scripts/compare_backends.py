"""Build the feature table from MySQL and check it equals the one built from Parquet.

The pipeline's staging SQL runs over ``raw_*`` views; this script proves the
views mean the same thing whichever backend fills them. Business tables come
from MySQL (``make mysql-load``); usage logs fall back to Parquet because they
are not loaded into MySQL by default.

    python scripts/compare_backends.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from churn import features, ingest, staging
from churn.config import DATA_DIR, PARQUET_DIR


def build(backend: str) -> tuple[pd.DataFrame, dict]:
    # On disk, like the pipeline's warehouse: staged tables of tens of millions
    # of rows do not fit in an in-memory database on a laptop.
    database = DATA_DIR / f"compare_{backend}.duckdb"
    database.unlink(missing_ok=True)
    con = ingest.connect(backend, PARQUET_DIR, database=database)
    provenance = dict(con.execute("SELECT tabela, origem FROM raw_provenance").fetchall())
    staging.build(con)
    features.build(con)
    frame = con.execute("SELECT * FROM features ORDER BY cutoff, conta_id").df()
    con.close()
    database.unlink(missing_ok=True)
    return frame, provenance


def main() -> int:
    mysql, provenance = build("mysql")
    print("from MySQL:", sorted(t for t, o in provenance.items() if o == "mysql"))
    print("from Parquet:", sorted(t for t, o in provenance.items() if o == "parquet"))
    parquet, _ = build("parquet")
    if mysql.shape != parquet.shape:
        print(f"shape differs: mysql {mysql.shape} vs parquet {parquet.shape}")
        return 1
    bad = []
    for column in parquet.columns:
        a, b = parquet[column], mysql[column]
        if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
            same = np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True)
        else:
            same = a.astype(str).equals(b.astype(str))
        if not same:
            bad.append(column)
    print(f"{len(parquet):,} rows x {parquet.shape[1]} columns compared; differing columns: {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
