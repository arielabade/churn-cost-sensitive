# Data

Generated locally, never committed (`data/cinelume/` is in `.gitignore`).

## Source

**Cinelume**, a fictional Brazilian streaming service, produced by the generator in its own
source folder (`CINELUME_SOURCE`). `make data` runs it unchanged with its default seed through
`scripts/generate_data.py`, which only adds intermediate scales. Everything is synthetic.

| Path | What |
| --- | --- |
| `cinelume/parquet/`, `cinelume/csv/` | the generator's exports, one file per table |
| `cinelume/warehouse.duckdb` | staged tables and the feature table, for inspection |
| `cinelume/features.parquet` | one row per subscriber per monthly cut-off |
| `cinelume/test_scores.parquet` | probabilities on the test months, for the post-hoc check |

Retention economics the dataset does not publish (contact cost, acceptance rates, margin,
horizon) are declared in `src/churn/config.py`. The acceptance outcome per subscriber does not
exist in the data, so the campaign accounting draws it once, with a fixed seed, and every rule
faces the same draw: the comparison is fair; the absolute figures are not a forecast.

Tables, grain, churn definition, what is missing and the data flaws found:
[docs/DATA_SOURCE.md](../docs/DATA_SOURCE.md).
