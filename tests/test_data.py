"""The data decisions, pinned. Each replaces a test from the Telco version of this project.

Telco test                                   Cinelume counterpart
-------------------------------------------  ------------------------------------------------
blank TotalCharges belong to new customers   missing values follow a declared policy
target is binary, original column gone       label is binary; end-of-data snapshots never ingested
customerID is not a feature                  identifiers, dates and labels are not features
split preserves the churn rate               population excludes pending cancellations and staff
"""

import numpy as np

from churn import features, ingest


def test_missing_values_follow_the_declared_policy(fixture_frame):
    """Counts are zero when nothing happened; recency and survey answers stay unknown.

    Filling "days since last play" with 0 would say a subscriber who never
    watched watched today, the Cinelume version of inventing billing history.
    """
    counts = ["horas_28d", "recusas_90d", "tentativas_cancelamento_28d", "tickets_90d", "emails_90d"]
    assert fixture_frame[counts].notna().all().all()
    assert (fixture_frame[counts] >= 0).all().all()
    never_surveyed = fixture_frame["dias_desde_nps"].isna()
    assert never_surveyed.any() and fixture_frame.loc[never_surveyed, "nps_ultima_nota"].isna().all()
    assert fixture_frame["dias_desde_ultimo_play"].dropna().ge(1).all(), "a play on the cut-off day leaked in"


def test_label_is_binary_and_snapshot_columns_are_never_ingested(fixture_frame):
    assert set(fixture_frame["churn"].unique()) <= {0, 1}
    assert fixture_frame["churn"].sum() > 0
    # status, plano_id and metodo_pagamento of assinaturas describe 31/10/2026.
    for table, column in (("assinaturas", "status"), ("assinaturas", "plano_id"),
                          ("assinaturas", "metodo_pagamento"), ("faturas", "status"),
                          ("tickets_suporte", "status")):
        assert column not in ingest.TABLES[table], f"{table}.{column} is a snapshot of the future"


def test_identifiers_dates_and_labels_are_not_features():
    forbidden = {"conta_id", "assinatura_id", "cutoff", "churn", "churn_involuntario", "custo_oferta"}
    assert not forbidden & set(features.FEATURES)
    assert len(features.FEATURES) == len(set(features.FEATURES))


def test_population_excludes_pending_cancellations_and_staff(fixture_dir, fixture_frame):
    import pandas as pd

    subs = pd.read_parquet(fixture_dir / "assinaturas.parquet")
    contas = pd.read_parquet(fixture_dir / "contas.parquet")
    staff = set(contas.loc[contas["email"].str.endswith("@cinelume.example"), "conta_id"])
    assert staff and not staff & set(fixture_frame["conta_id"])

    rows = fixture_frame.merge(subs[["assinatura_id", "cancelamento_solicitado_em"]], on="assinatura_id")
    asked_before = rows["cancelamento_solicitado_em"] < rows["cutoff"]
    assert not asked_before.any(), "a subscriber who had already asked to cancel was scored"
    assert np.all(rows.groupby(["cutoff", "conta_id"]).size() == 1)
