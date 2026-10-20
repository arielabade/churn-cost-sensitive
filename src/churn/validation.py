"""Post-hoc check: did the model recover what the simulation planted?

Role in the pipeline
--------------------
Last, and outside the pipeline. The Cinelume generator records its ground
truth in ``_gabarito/``: each account's latent traits (engagement, drift,
price sensitivity...), the dominant cause of every voluntary cancellation,
and which accounts are staff, QA or bots. This module reads that folder
**after** ``reports/`` is final and compares it with what the model did. It
is the only module allowed to; ``tests/test_answer_key_guard.py`` fails if
any training module names the folder or imports this one.

Nothing found here is fed back into features, target, cleaning or
hyperparameters. Misses are reported as misses.

Receives: ``data/cinelume/test_scores.parquet`` and ``features.parquet``
from the pipeline, ``reports/feature_importance.csv``, and the answer key
for the scale in :data:`churn.config.GENERATOR_SCALE`.
Returns: ``reports/validation.json``. Depends on: :mod:`churn.config`,
:mod:`churn.economics` (to know whom the rule contacted).

Run with ``python -m churn.validation`` (``make validate``).
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import DATA_DIR, ECONOMICS, GENERATOR_SCALE, REPORTS, SOURCE_DIR
from .economics import decide

ANSWER_KEY = SOURCE_DIR / "_gabarito"
DAY0 = dt.date(2023, 10, 1)

#: Dominant cause codes, as the generator assigns them (largest term of the
#: cancellation hazard for that subscriber on that day).
CAUSES = {
    0: "baixo engajamento", 1: "terminou uma série", 2: "série saiu do catálogo", 3: "reajuste de preço",
    4: "fim de promoção", 5: "problema técnico (buffering)", 6: "perdeu o compartilhador",
    7: "saída do lote licenciado",
}

#: Planted pattern -> the feature(s) that could carry it.
PATTERNS = {
    "P1 queda de uso antecede o cancelamento": ["reproducoes_28d", "horas_84d", "horas_28d", "tendencia_uso_28d",
                                                "dias_ativos_28d", "dias_desde_ultimo_play"],
    "P2 sinais no app (tela de planos, tentativa)": ["tentativas_cancelamento_28d", "tentativas_cancelamento_84d",
                                                     "telas_plano_28d"],
    "P9 reajuste de preço": ["aviso_reajuste_45d", "reajuste_90d_pct", "variacao_fatura_pct"],
    "P10 fim de promoção": ["desconto_termina_proxima_fatura", "desconto_ultima_fatura", "cupom_tipo"],
    "P11 canal de aquisição": ["canal"],
    "P12 buffering": ["buffering_por_reproducao_28d", "tickets_reproducao_90d"],
    "P13 perda do compartilhador": ["perfis", "membros_extra"],
    "P16 anual perto da renovação": ["dias_ate_renovacao", "periodicidade"],
    "P17 ticket sem solução": ["tickets_90d", "nota_suporte_180d"],
    "P5 fim de série (sem feature direta)": ["titulos_28d", "buscas_28d"],
}

#: Planted false clues: associated with churn, no direct effect.
FALSE_CLUES = {
    "F1 região": ["regiao"], "F4 idade": ["idade"],
    "F5 engajamento com interface": ["avaliacoes_positivas_84d", "adicoes_lista_28d", "taxa_abertura_email_90d",
                                     "taxa_abertura_push_90d"],
    "F6 nº de perfis / infantil": ["perfis", "perfis_infantis"],
    "FP2 e-mail de reengajamento": ["reengajamento_35d"],
}


def _test_decisions(scores: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    test = scores.merge(frame[["cutoff", "conta_id", "receita_mensal", "custo_oferta"]], on=["cutoff", "conta_id"])
    decisions = decide(test["probabilidade"].to_numpy(), test["receita_mensal"].to_numpy(),
                       test["custo_oferta"].to_numpy(), deadweight=ECONOMICS.non_churner_acceptance)
    test["contatado"] = decisions["contact"].to_numpy()
    test["decil_risco"] = test.groupby("cutoff")["probabilidade"].rank(pct=True, method="first").gt(0.9)
    return test


def latent_recovery(test: pd.DataFrame, latent: pd.DataFrame) -> dict:
    """Rank correlation between each subscriber's mean predicted risk and their latent traits."""
    per_account = test.groupby("conta_id")["probabilidade"].mean().rename("risco").reset_index()
    joined = per_account.merge(latent, on="conta_id")
    traits = {"eng_mu": "engajamento-base (esperado: negativo)", "drift": "deriva mensal (esperado: negativo)",
              "sens_preco": "sensibilidade a preço", "conexao": "qualidade de conexão",
              "risco_pag": "risco de pagamento (afeta churn involuntário, não o alvo)"}
    return {name: {"spearman": float(joined["risco"].corr(joined[name], method="spearman")), "leitura": label}
            for name, label in traits.items()}


def causes(test: pd.DataFrame, causes_frame: pd.DataFrame) -> list[dict]:
    """For churners in the test months: by true cause, how visible were they to the model?"""
    causes_frame = causes_frame.assign(
        data=pd.to_datetime(DAY0) + pd.to_timedelta(causes_frame["dia"], unit="D"))
    causes_frame["cutoff"] = causes_frame["data"].dt.to_period("M").dt.to_timestamp()
    churners = test[test["churn"] == 1].merge(causes_frame[["conta_id", "cutoff", "causa"]],
                                              on=["conta_id", "cutoff"], how="left")
    churners = churners.drop_duplicates(["conta_id", "cutoff"])
    loyal = test[test["churn"] == 0]
    rows = []
    for code, group in churners.groupby("causa"):
        # AUC of this cause's churners against all non-churners: can the
        # model tell *this kind* of leaver from someone who stays?
        y = np.r_[np.ones(len(group)), np.zeros(len(loyal))]
        p = np.r_[group["probabilidade"], loyal["probabilidade"]]
        rows.append({
            "causa": CAUSES.get(int(code), str(code)), "churners": int(len(group)),
            "risco_medio": float(group["probabilidade"].mean()),
            "auc_contra_fieis": float(roc_auc_score(y, p)) if len(group) >= 10 else None,
            "no_decil_de_maior_risco": float(group["decil_risco"].mean()),
            "contatados_pela_regra": float(group["contatado"].mean()),
        })
    rows.append({"causa": "(sem causa registrada)", "churners": int(churners["causa"].isna().sum())})
    return sorted(rows, key=lambda r: -r["churners"])


def patterns(importance: pd.Series) -> dict:
    rank = importance.rank(ascending=False).astype(int)

    def best(columns):
        present = [c for c in columns if c in rank.index]
        top = min(present, key=lambda c: rank[c])
        return {"melhor_feature": top, "posicao": int(rank[top]), "ganho": float(importance[present].sum())}

    return {"padroes": {name: best(cols) for name, cols in PATTERNS.items()},
            "pistas_falsas": {name: best(cols) for name, cols in FALSE_CLUES.items()},
            "total_features": int(len(importance))}


def special_accounts(test: pd.DataFrame, latent: pd.DataFrame) -> dict:
    kinds = {1: "funcionários", 2: "contas de QA", 3: "bots"}
    joined = test.merge(latent[["conta_id", "especial"]], on="conta_id")
    out = {}
    for code, name in kinds.items():
        rows = joined[joined["especial"] == code]
        out[name] = {"linhas_no_teste": int(len(rows)), "contas": int(rows["conta_id"].nunique()),
                     "contatados": int(rows["contatado"].sum())}
    return out


def reengagement_experiment(frame: pd.DataFrame) -> dict:
    """The e-mail the blind analysis took for rule-triggered is a randomised experiment.

    Eligible at a cut-off: opted in to marketing and 15 to 60 days since the
    last play. Treated: received the ``reengajamento`` e-mail that month (it
    goes out on the 1st, after the cut-off). Outcome: our label. Compared
    within inactivity bands, because churn rises with inactivity and the
    treated share is only flat inside the 15-60 day range; beyond 60 days the
    untreated group is diluted with accounts that never played, and a naive
    pooled comparison shows a spurious +1.2 pp "effect". The e-mail carries
    no offer, so this measures a contact alone.
    """
    import duckdb

    from .config import PARQUET_DIR

    sent = duckdb.sql(f"""
        SELECT DISTINCT conta_id, CAST(date_trunc('month', enviada_em) AS TIMESTAMP) AS cutoff
        FROM read_parquet('{PARQUET_DIR / 'notificacoes.parquet'}')
        WHERE campanha = 'reengajamento'
    """).df()
    days = frame["dias_desde_ultimo_play"]
    eligible = frame[(frame["aceita_marketing"] == 1) & (days > 14) & (days <= 60)]
    eligible = eligible.merge(sent.assign(email=1), on=["conta_id", "cutoff"], how="left")
    eligible["email"] = eligible["email"].fillna(0)
    eligible["faixa"] = pd.cut(eligible["dias_desde_ultimo_play"], [14, 16, 21, 30, 60])

    effect, variance, strata = 0.0, 0.0, []
    for band, group in eligible.groupby("faixa", observed=True):
        treated, control = group[group["email"] == 1]["churn"], group[group["email"] == 0]["churn"]
        weight = len(group) / len(eligible)
        p1, p0 = treated.mean(), control.mean()
        effect += weight * (p1 - p0)
        variance += weight ** 2 * (p1 * (1 - p1) / len(treated) + p0 * (1 - p0) / len(control))
        strata.append({"faixa_dias": str(band), "tratados": int(len(treated)), "controles": int(len(control)),
                       "churn_com_email": float(p1), "churn_sem_email": float(p0)})
    se = float(np.sqrt(variance))
    return {"elegiveis": int(len(eligible)), "share_tratado": float(eligible["email"].mean()),
            "efeito_pp": float(effect * 100), "ic95_pp": [float((effect - 1.96 * se) * 100),
                                                          float((effect + 1.96 * se) * 100)],
            "faixas": strata}


def millisecond_fix(metrics_key: dict) -> dict:
    import duckdb

    from .config import PARQUET_DIR

    fixed = duckdb.sql(f"""
        SELECT count(*) FROM read_parquet('{PARQUET_DIR / 'reproducoes.parquet'}') r
        JOIN read_parquet('{PARQUET_DIR / 'dispositivos.parquet'}') d USING (dispositivo_id)
        WHERE d.plataforma LIKE 'mobile%' AND r.versao_app >= '6.0.0'
          AND (r.segundos_assistidos > 5 * date_diff('second', r.inicio_em, r.fim_em)
               OR (r.fim_em IS NULL AND r.segundos_assistidos > 86400))
    """).fetchone()[0]
    planted = metrics_key.get("repro_ms")
    return {"corrigidas": int(fixed), "plantadas": planted, "cobertura": fixed / planted if planted else None}


def run() -> dict:
    scale = GENERATOR_SCALE
    latent = pd.read_parquet(ANSWER_KEY / f"latentes_{scale}.parquet")
    causes_frame = pd.read_parquet(ANSWER_KEY / f"causas_cancelamento_{scale}.parquet")
    metrics_key = json.loads((ANSWER_KEY / f"metricas_{scale}.json").read_text())

    frame = pd.read_parquet(DATA_DIR / "features.parquet")
    scores = pd.read_parquet(DATA_DIR / "test_scores.parquet")
    test = _test_decisions(scores, frame)
    importance = pd.read_csv(REPORTS / "feature_importance.csv", index_col=0)["ganho"]

    result = {
        "escala": scale,
        "latentes": latent_recovery(test, latent),
        "causas": causes(test, causes_frame),
        **patterns(importance),
        "contas_especiais": special_accounts(test, latent),
        "experimento_reengajamento": reengagement_experiment(frame),
        "correcao_ms": millisecond_fix(metrics_key),
        "imperfeicoes_nao_tratadas": {
            "variantes de dispositivos.tipo": "tipos_dispositivo conta grafias como tipos distintos",
            "variantes de tickets_suporte.categoria": metrics_key.get("tickets_categoria_variante"),
            "eventos duplicados": metrics_key.get("eventos_duplicados"),
            "eventos renomeados na 6.0 mobile (adicionar_lista)": metrics_key.get("eventos_renomeados"),
            "bots e QA na população": "ver contas_especiais",
        },
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=float))
