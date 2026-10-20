"""Pipeline: from the Cinelume tables to the contact decision and its value.

Role in the pipeline
--------------------
The orchestrator. Runs ingest → staging → features → model → evaluate under
the answer-key guard and writes every number the README and the dashboard
quote into ``reports/``. Nothing downstream recomputes a number; figures and
the dashboard only read these files, which is how they stay in agreement.

Receives: command-line options (backend, whether to rebuild features).
Returns: ``reports/metrics.json`` plus the CSV tables listed in
:data:`REPORT_FILES`. Depends on: every stage module; never on
:mod:`churn.validation`.

Run with ``python -m churn.pipeline [--backend mysql] [--rebuild]``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import evaluate, features, ingest, model, staging
from .config import (
    DATA_DIR, ECONOMICS, GENERATOR_SCALE, GENERATOR_SEED, PARQUET_DIR, REPORTS, RETENTION_COUPON,
)
from .economics import acceptance_draw
from .guard import answer_key_blocked

FEATURES_FILE = DATA_DIR / "features.parquet"
WAREHOUSE = DATA_DIR / "warehouse.duckdb"
REPORT_FILES = (
    "metrics.json", "rules_by_segment.csv", "sensitivity.csv", "feature_importance.csv",
    "decisions_last_month.csv", "monthly_population.csv",
)


def build_features(backend: str = "parquet") -> tuple[pd.DataFrame, dict]:
    """Ingest, stage and build the feature table; cache it as Parquet."""
    started = time.time()
    con = ingest.connect(backend, PARQUET_DIR, database=WAREHOUSE)
    volume = ingest.volume(con)
    provenance = dict(con.execute("SELECT tabela, origem FROM raw_provenance").fetchall())
    staging.build(con)
    features.build(con)
    con.execute(f"COPY features TO '{FEATURES_FILE}' (FORMAT parquet)")
    offer = con.execute("SELECT * FROM stg_retention_offer").df().iloc[0].to_dict()
    con.close()
    meta = {"volume": volume, "provenance": provenance, "offer": offer,
            "feature_build_seconds": round(time.time() - started, 1)}
    (DATA_DIR / "features_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return _read_features(), meta


def _build_features_only(backend: str) -> None:
    with answer_key_blocked():
        build_features(backend)


def _read_features() -> pd.DataFrame:
    frame = pd.read_parquet(FEATURES_FILE).sort_values(["cutoff", "conta_id"]).reset_index(drop=True)
    for column in features.CATEGORICAL:
        frame[column] = frame[column].astype("category")
    return frame


def _round(value):
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    return value


def run(backend: str = "parquet", rebuild: bool = False) -> dict:
    """Train, decide, evaluate and write the reports."""
    with answer_key_blocked():
        if rebuild or not FEATURES_FILE.exists():
            # In a child process: DuckDB's allocator keeps its peak memory
            # after the connection closes, and that peak plus the models'
            # was enough to get the run killed on a 16 GB laptop. A process
            # that exits hands everything back.
            context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(1, mp_context=context) as pool:
                pool.submit(_build_features_only, backend).result()
        frame = _read_features()
        meta = json.loads((DATA_DIR / "features_meta.json").read_text())

        split = model.temporal_split(frame)
        lightgbm = model.LightGBMModel().fit(split.train, split.valid)
        logistic = model.LogisticModel().fit(split.train, split.valid)

        y_valid = split.valid["churn"].to_numpy()
        p_valid, p_test = lightgbm.predict(split.valid), lightgbm.predict(split.test)
        p_test_logistic = logistic.predict(split.test)
        # Both baselines' cut-offs come from the validation months.
        f1_threshold, f1_valid = evaluate.best_f1_threshold(y_valid, p_valid)
        f1_threshold_logistic, _ = evaluate.best_f1_threshold(y_valid, logistic.predict(split.valid))

        test = split.test
        draw = acceptance_draw(len(test))
        rules = evaluate.compare(test, p_test, f1_threshold, draw=draw)
        rules_logistic = evaluate.compare(test, p_test_logistic, f1_threshold_logistic, draw=draw)
        segments = pd.concat(
            [evaluate.by_segment(test, p_test, f1_threshold, column, draw)
             for column in ("cutoff", "plano", "canal")],
            ignore_index=True,
        )
        sensitivity = evaluate.sensitivity(test, p_test, f1_threshold, draw)
        gap_f1 = evaluate.bootstrap_gap(test, p_test, f1_threshold, "valor_esperado", "melhor_f1", draw=draw)
        gap_default = evaluate.bootstrap_gap(test, p_test, f1_threshold, "valor_esperado", "limiar_0.5",
                                             draw=draw)

        def model_scores(m, part):
            # Ranking on the raw score: isotonic calibration is monotone but
            # piecewise flat, and the ties it creates would understate AUC.
            # Calibration on the calibrated probability, the number the
            # decision rule multiplies by money.
            y = part["churn"].to_numpy()
            ranking = model.scores(y, m.predict_raw(part))
            calibrated = model.scores(y, m.predict(part))
            return {"roc_auc": ranking["roc_auc"], "pr_auc": ranking["pr_auc"],
                    "brier": calibrated["brier"], "log_loss": calibrated["log_loss"],
                    "mean_predicted": calibrated["mean_predicted"], "observed_rate": calibrated["observed_rate"]}

        summary = {
            "data": {
                "source": "Cinelume (streaming simulado), gerado pelo gerador do próprio dataset",
                "scale": GENERATOR_SCALE, "seed": GENERATOR_SEED, **meta["volume"],
                "provenance": meta["provenance"],
                "feature_rows": int(len(frame)),
                "subscribers_scored": int(frame["conta_id"].nunique()),
                "cutoffs": int(frame["cutoff"].nunique()),
                "churn_rate": float(frame["churn"].mean()),
                "involuntary_rate": float(frame["churn_involuntario"].mean()),
            },
            "split": split.describe(),
            "economics": {**asdict(ECONOMICS), "currency": "BRL",
                          "retention_coupon": RETENTION_COUPON, "offer_terms": meta["offer"]},
            "models": {
                "lightgbm": {"valid": model_scores(lightgbm, split.valid), "test": model_scores(lightgbm, test)},
                "logistic": {"valid": model_scores(logistic, split.valid), "test": model_scores(logistic, test)},
            },
            "best_f1_threshold": f1_threshold,
            "best_f1_on_valid": f1_valid,
            "best_f1_threshold_logistic": f1_threshold_logistic,
            "break_even_probability": evaluate.break_even_summary(test),
            "strategies": {rule: row.to_dict() for rule, row in rules.iterrows()},
            "strategies_logistic": {rule: row.to_dict() for rule, row in rules_logistic.iterrows()},
            "bootstrap": {"valor_esperado_vs_melhor_f1": gap_f1, "valor_esperado_vs_limiar_0.5": gap_default},
            "uplift": {
                "available": False,
                "reason": "o dataset não tem grupo de controle nem tratamento de retenção aleatorizado; "
                          "reengajamento e winback são disparados por regra, não por sorteio",
            },
        }

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(_round(summary), indent=2, ensure_ascii=False, default=str))
    segments.to_csv(REPORTS / "rules_by_segment.csv", index=False, float_format="%.2f")
    sensitivity.to_csv(REPORTS / "sensitivity.csv", index=False, float_format="%.2f")
    lightgbm.importance().rename("ganho").to_csv(REPORTS / "feature_importance.csv", float_format="%.5f")

    # The contact list a CRM team would receive for the last test month.
    last = (test["cutoff"] == test["cutoff"].max()).to_numpy()
    tables = evaluate.decisions_for(test[last], p_test[last], f1_threshold)
    tables["valor_esperado"].assign(
        conta_id=test.loc[last, "conta_id"].to_numpy(),
        plano=test.loc[last, "plano"].astype(str).to_numpy(),
        canal=test.loc[last, "canal"].astype(str).to_numpy(),
        churn=test.loc[last, "churn"].to_numpy(),
        contato_melhor_f1=tables["melhor_f1"]["contact"].to_numpy(),
    ).to_csv(REPORTS / "decisions_last_month.csv", index=False, float_format="%.4f")

    frame.groupby("cutoff").agg(
        assinantes=("churn", "size"), churn=("churn", "mean"), churn_involuntario=("churn_involuntario", "mean"),
    ).to_csv(REPORTS / "monthly_population.csv", float_format="%.5f")

    # Scores on test rows, kept out of git, for the answer-key check.
    test[["cutoff", "conta_id", "churn", "plano", "canal"]].assign(
        probabilidade=p_test, probabilidade_logistica=p_test_logistic,
    ).to_parquet(DATA_DIR / "test_scores.parquet", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, decide and evaluate on the Cinelume data.")
    parser.add_argument("--backend", choices=("parquet", "mysql"), default="parquet")
    parser.add_argument("--rebuild", action="store_true", help="rebuild features even if cached")
    args = parser.parse_args()
    summary = run(args.backend, args.rebuild)
    print(json.dumps({k: summary[k] for k in ("split", "best_f1_threshold", "strategies")},
                     indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
