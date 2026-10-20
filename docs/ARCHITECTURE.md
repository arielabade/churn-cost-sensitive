# Architecture

How a row in the Cinelume database becomes a contact decision with a price on it. Every module
named here exists in `src/churn/`, and each one opens with a docstring stating its role, what it
receives, what it returns and what it depends on (`tests/test_documentation.py` checks both).

## End to end

```mermaid
flowchart LR
  subgraph SRC["Cinelume source folder (not edited)"]
    GEN["gerar_dados.py<br/>default seed"]
    SQL["schema.sql"]
    KEY[("_gabarito/<br/>answer key")]
  end
  WRAP["scripts/generate_data.py<br/>scale x60"] --> GEN
  GEN --> PQ[("data/cinelume/parquet<br/>+ csv")]
  GEN -.writes.-> KEY
  PQ -->|csv| LOAD["infra/load_mysql.sh"] --> MY[("MySQL 8<br/>cinelume-churn-mysql")]
  SQL --> MY

  subgraph PIPE["churn.pipeline  (inside guard.answer_key_blocked)"]
    ING["ingest<br/>raw_* views"] --> STG["staging<br/>stg_* tables"] --> FEAT["features<br/>1 row per subscriber × month"]
    FEAT --> MOD["model<br/>LightGBM + isotonic<br/>logistic baseline"]
    MOD --> EVAL["evaluate<br/>5 rules, segments,<br/>sensitivity, bootstrap"]
    CFG["config<br/>declared economics,<br/>calendar"] --> ECO["economics<br/>break-even rule,<br/>campaign accounting"]
    ECO --> EVAL
  end
  PQ --> ING
  MY -->|business tables| ING
  EVAL --> REP[("reports/")]
  REP --> FIG["figures → assets/figures/*.svg → README"]
  REP --> DASH["dashboard → docs/dashboard/ → GitHub Pages"]
  REP --> VAL["validation<br/>post-hoc only"]
  KEY --> VAL
  VAL --> REP
```

The dotted boundary matters: `_gabarito/` reaches only `validation`, after `reports/` is final.
The guard enforces this in three layers: a static test, a path check in `ingest`, and a runtime audit hook.

## Data model: source tables used and tables derived

Source tables (from `schema.sql`, only the columns the pipeline reads; the end-of-data snapshot
columns `assinaturas.plano_id/metodo_pagamento/status`, `faturas.status` and
`tickets_suporte.status` are never ingested) and the tables the pipeline creates in the DuckDB
warehouse (`data/cinelume/warehouse.duckdb`).

```mermaid
erDiagram
    canais ||--o{ contas : "adquire"
    contas ||--|{ perfis : "tem"
    contas ||--o{ dispositivos : "registra"
    contas ||--|{ assinaturas : "assina"
    assinaturas ||--o{ assinatura_eventos : "histórico"
    planos ||--o{ precos : "tem preço"
    assinaturas ||--o{ faturas : "cobra"
    faturas |o--o{ pagamentos : "paga"
    cupons ||--o{ cupom_resgates : "resgatado"
    cupom_resgates |o--o{ faturas : "desconta"
    perfis ||--o{ reproducoes : "assiste"
    dispositivos ||--o{ reproducoes : "toca em"
    perfis ||--o{ sessoes_app : "abre"
    perfis ||--o{ eventos_app : "gera"
    contas ||--o{ notificacoes : "recebe"
    contas ||--o{ tickets_suporte : "abre"
    contas ||--o{ nps : "responde"

    contas ||--|| stg_accounts : "limpa (UF, região, staff)"
    assinaturas ||--|| stg_subscriptions : "ordena reativações"
    assinatura_eventos ||--o{ stg_plan_events : "plano no tempo"
    precos ||--o{ stg_prices : "mensaliza"
    cupons ||--o| stg_retention_offer : "VOLTA50"
    faturas ||--|| stg_invoices : "família do cupom"
    reproducoes ||--|| stg_plays : "corrige ms"
    stg_plays }o--|| stg_usage_daily : "conta × dia"
    sessoes_app }o--|| stg_sessions_daily : "conta × dia"
    eventos_app }o--|| stg_app_daily : "conta × dia (recebido_em)"
    cutoffs ||--o{ population : "1º dia do mês"
    stg_subscriptions ||--o{ population : "ativa no corte"
    population ||--|| features : "atributos + rótulo"
    features ||--o| test_scores : "probabilidade (teste)"

    population {
        timestamp cutoff PK
        int conta_id PK
        int assinatura_id
        int churn "pede cancelamento no mês seguinte"
        int churn_involuntario "só para relatório"
    }
    features {
        timestamp cutoff PK
        int conta_id PK
        varchar plano "reconstruído dos eventos"
        double receita_mensal "preço vigente + membro extra"
        double custo_oferta "VOLTA50 no plano"
        double horas_28d
        double tendencia_uso_28d
        int tentativas_cancelamento_28d
        int dias_ate_renovacao
    }
```

## Module responsibilities

| Module | Stage | Responsibility | Reads | Writes |
| --- | --- | --- | --- | --- |
| `config` | all | Declared economics (contact cost, acceptance, deadweight, margin, horizon), calendar and split, paths | env vars | constants |
| `guard` | all | Keeps `_gabarito/` out of training: path check, audit hook | — | — |
| `ingest` | 1 | Typed `raw_*` views over Parquet or MySQL; snapshot columns never exposed | data files / MySQL | DuckDB views |
| `staging` | 2 | Cleaning and per-day aggregation: UF codes, staff flag, plan history, prices, **millisecond fix**, usage per account-day | `raw_*` | `stg_*` |
| `features` | 3 | Population at each monthly cut-off, ~70 features known before the cut-off, label for the next month | `stg_*` | `features` |
| `model` | 4 | Temporal split; LightGBM and logistic, each calibrated on the validation months | `features` | scores |
| `economics` | 5 | Break-even probability per subscriber, contact decision, campaign accounting | probabilities, prices | decisions, money |
| `evaluate` | 5 | Five rules on the same probabilities; by month, plan, channel; sensitivity; bootstrap | test frame | tables |
| `pipeline` | 1–5 | Orchestration; writes every quoted number | all of the above | `reports/` |
| `figures` | after | README SVGs, ABADE dark theme (`brandviz`) | `reports/` | `assets/figures/` |
| `dashboard` | after | Client page for GitHub Pages | `reports/` | `docs/dashboard/` |
| `validation` | post-hoc | Compares the finished model with the answer key | `reports/`, `_gabarito/` | `reports/validation.json` |

## One subscriber, from the database to the decision

Take a Premium subscriber on 1 October 2026 (the cut-off `t`).

1. **Ingest.** Their rows arrive through `raw_contas`, `raw_assinaturas`, `raw_reproducoes`… The
   column saying they are on Premium *today* (`assinaturas.plano_id`) is not among them: it is a
   photograph of 31/10/2026, after `t`.
2. **Staging.** Their plays on the mobile app 6.1 arrive in milliseconds and are divided by 1000
   (`stg_plays`), then summed per day (`stg_usage_daily`). Their app events are dated when the
   server received them, not by the phone's clock.
3. **Features at `t`.** The subscription started before `t`, has not ended and has no pending
   cancellation, so they are in `population`. Their plan at `t` is the last plan event before
   `t` (an upgrade in March: Premium). Their price is the Premium monthly price in force on
   1/10/2026 (R$64.90), plus extra members. Their usage over 7/28/84 days, the trend, days since the
   last play, cancellation attempts in the app, declined payments, days to renewal: every one of them
   uses rows strictly before `t`. The label is 1 if they ask to cancel between 1 and 31 October.
4. **Model.** LightGBM, trained on 2024–2025, scores them; the isotonic map fitted on
   January–April 2026 turns the score into a probability, say **24%**.
5. **Economics.** Margin at risk = 64.90 × 55% × 12 = R$428. Offer = `VOLTA50` (50% for two
   cycles) on their plan = R$64.90. With contact R$6, acceptance 35% and deadweight 35%, their
   break-even probability is (6 + 0.35 × 64.90) / (0.35 × (428 − 64.90) + 0.35 × 64.90) ≈ **19.2%**.
6. **Decision.** 24% > 19.2%: they are on October's contact list (`reports/decisions_last_month.csv`).
   A Basic subscriber with the same 24% would not be: their bar is 26.5%.
7. **Accounting.** On the held-out month, if they did ask to cancel and the simulated draw says they
   accept, the campaign books R$428 of margin saved, minus the offer and the contact. If they were
   not leaving and accept anyway, it books the R$64.90 discount as a loss.

## Where the non-obvious decisions are commented

| Decision | Where |
| --- | --- |
| Cut-off on the first of the month, label = next calendar month, splits that share no label window | `config` (calendar), `features` (docstring) |
| Snapshot columns never ingested | `ingest` (docstring, `TABLES`) |
| Millisecond fix measured against wall-clock time | `staging` (`stg_plays`) |
| Events dated by `recebido_em` | `staging` (docstring) |
| Counts filled with 0, recency left NULL | `features` (assemble block) |
| E-mail open rate `coalesce` (a leak the test caught) | `features` (`f_notif`) |
| Integer sums for bit-for-bit reproducibility | `staging` (`stg_usage_daily`) |
| Calibration on the validation months; AUC on the raw score | `model.fit`, `pipeline.model_scores` |
| Offer cost weighted by acceptance; deadweight; infinite threshold | `economics` |
| F1 cut-off chosen on validation, not test | `evaluate.best_f1_threshold` |
| Bootstrap by subscriber, not by row | `evaluate.bootstrap_gap` |
| Feature build in a child process (memory) | `pipeline.run` |
