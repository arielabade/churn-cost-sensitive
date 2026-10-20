"""Declared assumptions, calendar and paths: the only numbers a reader should have to trust.

Role in the pipeline
--------------------
Every other module imports from here and nothing here imports from them. It
holds three kinds of constants:

1. **Retention economics the dataset does not publish** (contact cost,
   acceptance rate, deadweight, gross margin, horizon). They are declared, not
   measured, and they are the first thing to challenge in a business review.
2. **The calendar of the temporal design**: monthly cut-off dates and which of
   them train, validate and test the model.
3. **Paths**: where the generated Cinelume data lives and where reports go.

Receives: environment variables ``CINELUME_DATA`` and ``CINELUME_SOURCE``
(optional overrides). Returns: constants. Depends on: nothing in the package.

Economics the dataset *does* publish (plan prices with their validity dates,
the extra-member fee, the retention coupon) are read from the data by
:mod:`churn.staging`, not typed here. Currency is the dataset's: BRL.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

#: Generated Cinelume tables (``parquet/`` and ``csv/`` underneath).
DATA_DIR = Path(os.environ.get("CINELUME_DATA", ROOT / "data" / "cinelume"))
PARQUET_DIR = DATA_DIR / "parquet"

#: The Cinelume source folder (generator, schema.sql, docker-compose). Needed
#: only to regenerate data and start MySQL.
SOURCE_DIR = Path(
    os.environ.get("CINELUME_SOURCE", Path.home() / "Downloads" / "datasets-github" / "streaming")
)

RANDOM_SEED = 20261002
GENERATOR_SEED = 20231001  # the generator's own default seed, used unchanged
GENERATOR_SCALE = "x60"  # 60k accounts; see docs/DATA_SOURCE.md

CURRENCY = "R$"

# --------------------------------------------------------------------- calendar
#: Data runs 2023-10-01 .. 2026-10-31. The first cut-off leaves three months of
#: history for the usage windows; the last one is the latest month whose
#: outcome is fully observed (its label window closes on 2026-11-01 00:00 UTC).
FIRST_CUTOFF = dt.date(2024, 1, 1)
LAST_CUTOFF = dt.date(2026, 10, 1)

#: Train on the past, calibrate and pick thresholds on the next months, report
#: on months neither the model nor its thresholds saw. Each label window is one
#: calendar month, so the last training label (December 2025) closes before the
#: first validation cut-off opens: no outcome is shared across the split.
TRAIN_UNTIL = dt.date(2025, 12, 1)   # cut-offs 2024-01 .. 2025-12 (24 months)
VALID_UNTIL = dt.date(2026, 4, 1)    # cut-offs 2026-01 .. 2026-04 (4 months)
#                                      test: cut-offs 2026-05 .. 2026-10 (6 months)

#: The retention offer is the dataset's own win-back coupon ("Reconquista: 50%
#: por 2 meses"). Its terms are read from ``cupons``; only the code is fixed here.
RETENTION_COUPON = "VOLTA50"


@dataclass(frozen=True)
class RetentionEconomics:
    """What an intervention costs and what it saves, where the data is silent.

    offer_cost
        Fallback cost of the retention offer when no per-customer cost is
        supplied. The pipeline always supplies one, derived from the
        ``VOLTA50`` coupon and the customer's own plan price; the scalar exists
        so the algebra can be reasoned about, and tested, in isolation.
    contact_cost
        Agent and CRM time per contact, in BRL, spent whether or not the offer
        is accepted.
    acceptance_rate
        Share of genuinely at-risk customers the offer actually retains. The
        single most optimistic assumption in any retention case, and the one
        least often stated.
    non_churner_acceptance
        Share of contacted customers who were *not* leaving and take the
        discount anyway. Pure deadweight: they cost the offer and save nothing.
        Equal to ``acceptance_rate`` because that is what the campaign
        accounting of this project has always simulated; the rule now prices
        it too, instead of being judged by a cost it did not know about.
    gross_margin
        Share of revenue that is contribution. The dataset has no content or
        serving cost per subscriber, so this stays declared.
    expected_remaining_months
        Horizon over which a saved customer keeps paying. Beyond this the
        saving is speculative.
    """

    offer_cost: float = 45.0
    contact_cost: float = 6.0
    acceptance_rate: float = 0.35
    non_churner_acceptance: float = 0.35
    gross_margin: float = 0.55
    expected_remaining_months: int = 12


ECONOMICS = RetentionEconomics()

#: Sensitivity grid for the two assumptions a reviewer will push on first.
ACCEPTANCE_GRID = (0.15, 0.25, 0.35, 0.45, 0.55)
OFFER_MULTIPLIER_GRID = (0.5, 1.0, 1.5, 2.0)
DEADWEIGHT_GRID = (0.0, 0.1, 0.2, 0.35, 0.5)
