"""Staging: clean, conform and pre-aggregate the raw tables into ``stg_*`` tables.

Role in the pipeline
--------------------
Second stage, between :mod:`churn.ingest` (``raw_*`` views) and
:mod:`churn.features` (one row per subscriber per cut-off). Staging does the
work that does not depend on the cut-off date: cleaning, joining profiles to
accounts, and collapsing tens of millions of log rows into one row per account
per day, so the feature step joins days rather than events.

Receives: a DuckDB connection with ``raw_*`` views.
Returns: nothing; creates the ``stg_*`` tables listed in :data:`STAGED`.
Depends on: :mod:`churn.config` (retention coupon code).

Every staged row keeps the instant at which the business could first have
known it (``ocorrido_em``, ``processado_em``, ``recebido_em``, ``dia``...).
That instant is what the feature step compares against the cut-off, and it is
why app events are dated by ``recebido_em`` (server clock) rather than
``ocorrido_em`` (device clock, which can be wrong and can be backfilled from an
offline queue): an event is information only from the moment it arrives.
"""

from __future__ import annotations

import duckdb

from .config import RETENTION_COUPON

#: State names and their abbreviations. ``contas.uf`` is documented as a
#: two-letter upper-case code but also arrives lower-cased and spelled out.
_UF = {
    "acre": "AC", "alagoas": "AL", "amapá": "AP", "amazonas": "AM", "bahia": "BA", "ceará": "CE",
    "distrito federal": "DF", "espírito santo": "ES", "goiás": "GO", "maranhão": "MA",
    "mato grosso": "MT", "mato grosso do sul": "MS", "minas gerais": "MG", "pará": "PA",
    "paraíba": "PB", "paraná": "PR", "pernambuco": "PE", "piauí": "PI", "rio de janeiro": "RJ",
    "rio grande do norte": "RN", "rio grande do sul": "RS", "rondônia": "RO", "roraima": "RR",
    "santa catarina": "SC", "são paulo": "SP", "sergipe": "SE", "tocantins": "TO",
}

#: App events worth keeping. ``impressao_titulo`` and ``clique_titulo`` are the
#: bulk of the table and describe browsing, not intent; leaving them out keeps
#: the daily table small without losing the signals that precede a cancellation.
INTENT_EVENTS = ("tentativa_cancelamento", "abrir_tela_planos", "busca", "avaliacao", "adicionar_lista")

STAGED = (
    "stg_accounts", "stg_subscriptions", "stg_plan_events", "stg_lifecycle_events", "stg_prices",
    "stg_retention_offer", "stg_invoices", "stg_payments", "stg_profiles", "stg_devices",
    "stg_usage_daily", "stg_sessions_daily", "stg_app_daily", "stg_notifications", "stg_tickets", "stg_nps",
    "stg_staff_coupons", "stg_plays",
)


def build(con: duckdb.DuckDBPyConnection) -> None:
    """Create every ``stg_*`` table. Idempotent: tables are replaced."""
    uf_values = ", ".join(f"('{name}', '{code}')" for name, code in _UF.items())

    con.execute(f"""
    CREATE OR REPLACE TABLE stg_accounts AS
    WITH uf_map(nome, sigla) AS (VALUES {uf_values})
    SELECT
        c.conta_id,
        c.criada_em,
        ca.codigo AS canal,
        coalesce(m.sigla, upper(trim(c.uf))) AS uf,
        CASE coalesce(m.sigla, upper(trim(c.uf)))
            WHEN 'SP' THEN 'sudeste' WHEN 'RJ' THEN 'sudeste' WHEN 'MG' THEN 'sudeste' WHEN 'ES' THEN 'sudeste'
            WHEN 'PR' THEN 'sul' WHEN 'SC' THEN 'sul' WHEN 'RS' THEN 'sul'
            WHEN 'GO' THEN 'centro_oeste' WHEN 'DF' THEN 'centro_oeste' WHEN 'MT' THEN 'centro_oeste'
            WHEN 'MS' THEN 'centro_oeste'
            WHEN 'AC' THEN 'norte' WHEN 'AM' THEN 'norte' WHEN 'AP' THEN 'norte' WHEN 'PA' THEN 'norte'
            WHEN 'RO' THEN 'norte' WHEN 'RR' THEN 'norte' WHEN 'TO' THEN 'norte'
            ELSE 'nordeste'
        END AS regiao,
        c.data_nascimento,
        c.aceita_marketing,
        (c.indicada_por_conta_id IS NOT NULL)::INTEGER AS indicada,
        -- Staff accounts use the company's own mail domain. They are not
        -- customers a retention budget should be spent on.
        (split_part(c.email, '@', 2) = 'cinelume.example')::INTEGER AS interna
    FROM raw_contas c
    JOIN raw_canais ca ON ca.canal_id = c.canal_aquisicao_id
    LEFT JOIN uf_map m ON m.nome = lower(trim(c.uf))
    """)

    con.execute("""
    CREATE OR REPLACE TABLE stg_subscriptions AS
    SELECT
        assinatura_id, conta_id, periodicidade, iniciada_em, trial_fim_em,
        cancelamento_solicitado_em, encerrada_em, tipo_encerramento, assinatura_anterior_id,
        row_number() OVER (PARTITION BY conta_id ORDER BY iniciada_em, assinatura_id) - 1
            AS assinaturas_anteriores
    FROM raw_assinaturas
    """)

    # The plan a subscriber is on at any instant is the plan named by the last
    # creation, reactivation, upgrade or downgrade before it. assinaturas.plano_id
    # is only the plan on the last day of the data.
    con.execute("""
    CREATE OR REPLACE TABLE stg_plan_events AS
    SELECT e.assinatura_id, e.ocorrido_em, e.plano_novo_id AS plano_id, e.tipo
    FROM raw_assinatura_eventos e
    WHERE e.tipo IN ('assinatura_criada', 'reativada', 'upgrade', 'downgrade')
      AND e.plano_novo_id IS NOT NULL
    """)

    con.execute("""
    CREATE OR REPLACE TABLE stg_lifecycle_events AS
    SELECT a.conta_id, e.assinatura_id, e.tipo, e.ocorrido_em
    FROM raw_assinatura_eventos e
    JOIN raw_assinaturas a USING (assinatura_id)
    WHERE e.tipo IN ('upgrade', 'downgrade', 'metodo_pagamento_alterado',
                     'inadimplencia_iniciada', 'inadimplencia_resolvida',
                     'membro_extra_adicionado', 'membro_extra_removido')
    """)

    # Monthly-equivalent list prices with their validity windows. An annual
    # price is spread over twelve months so the two cycles compare.
    con.execute("""
    CREATE OR REPLACE TABLE stg_prices AS
    SELECT
        p.plano_id, p.item, p.periodicidade, p.vigencia_inicio,
        coalesce(p.vigencia_fim, DATE '2999-12-31') AS vigencia_fim,
        p.valor_brl,
        p.valor_brl / CASE WHEN p.periodicidade = 'anual' THEN 12 ELSE 1 END AS valor_mensal
    FROM raw_precos p
    """)

    con.execute(f"""
    CREATE OR REPLACE TABLE stg_retention_offer AS
    SELECT codigo, tipo_desconto, valor_desconto, duracao_ciclos
    FROM raw_cupons WHERE codigo = '{RETENTION_COUPON}'
    """)

    con.execute("""
    CREATE OR REPLACE TABLE stg_invoices AS
    SELECT
        f.fatura_id, f.assinatura_id, f.conta_id, f.emitida_em, f.periodo_inicio, f.periodo_fim,
        f.valor_bruto, f.valor_desconto, f.valor_membro_extra, f.valor_total,
        f.cupom_resgate_id,
        u.codigo AS cupom_codigo,
        u.duracao_ciclos AS cupom_ciclos,
        CASE
            WHEN u.codigo IS NULL THEN 'nenhum'
            WHEN u.codigo LIKE 'BLACK%' OR u.codigo LIKE 'FERIAS%' OR u.codigo LIKE 'VERAO%' THEN 'sazonal'
            WHEN u.codigo = 'PARCEIRO3M' THEN 'parceria'
            WHEN u.codigo = 'INDICA30' THEN 'indicacao'
            WHEN u.codigo IN ('CINE20', 'MARATONA20') THEN 'social'
            WHEN u.codigo = 'VOLTA50' THEN 'reconquista'
            WHEN u.codigo = 'COLABORADOR' THEN 'colaborador'
            ELSE 'outro'
        END AS cupom_tipo,
        -- Position of this invoice among the ones the coupon discounts.
        CASE WHEN f.cupom_resgate_id IS NOT NULL THEN
            row_number() OVER (PARTITION BY f.cupom_resgate_id ORDER BY f.emitida_em, f.fatura_id)
        END AS cupom_ciclo
    FROM raw_faturas f
    LEFT JOIN raw_cupom_resgates r ON r.resgate_id = f.cupom_resgate_id
    LEFT JOIN raw_cupons u ON u.cupom_id = r.cupom_id
    """)

    con.execute("""
    CREATE OR REPLACE TABLE stg_payments AS
    SELECT conta_id, fatura_id, metodo, tipo, tentativa, processado_em, status
    FROM raw_pagamentos
    """)

    con.execute("""
    CREATE OR REPLACE TABLE stg_profiles AS
    SELECT perfil_id, conta_id, infantil, membro_extra, criado_em, excluido_em FROM raw_perfis
    """)

    con.execute("""
    CREATE OR REPLACE TABLE stg_devices AS
    SELECT dispositivo_id, conta_id, tipo, registrado_em FROM raw_dispositivos
    """)

    # Watch time. From app 6.0.0 the mobile apps report most plays in
    # MILLISECONDS in a column documented as seconds: a 45-minute episode
    # arrives as 2,700,000. Left alone, every mobile subscriber looks a
    # thousand times more engaged from February 2026, the model reads that as
    # loyalty, and its test-period probabilities fall to half the observed
    # churn rate. The fix is measured against each play's own wall-clock time:
    # divide by 1000 only where the reported value exceeds five times the
    # elapsed time (or a full day when the play has no end), because part of
    # the same slice is already in seconds. See docs/DATA_SOURCE.md.
    con.execute("""
    CREATE OR REPLACE TABLE stg_plays AS
    SELECT
        r.*,
        CASE
            WHEN d.plataforma LIKE 'mobile%' AND r.versao_app >= '6.0.0'
             AND (r.segundos_assistidos > 5 * date_diff('second', r.inicio_em, r.fim_em)
                  OR (r.fim_em IS NULL AND r.segundos_assistidos > 86400))
            THEN round(r.segundos_assistidos / 1000.0)
            ELSE r.segundos_assistidos
        END::BIGINT AS segundos
    FROM raw_reproducoes r
    LEFT JOIN raw_dispositivos d USING (dispositivo_id)
    """)

    # Plays are dated by their server-side start. A play running across the
    # midnight of a cut-off counts in full on its start day; at one row per
    # account-day the distortion is a few minutes and the table is 20x smaller.
    con.execute("""
    CREATE OR REPLACE TABLE stg_usage_daily AS
    SELECT
        p.conta_id,
        CAST(r.inicio_em AS DATE) AS dia,
        -- Integer sums on purpose: DuckDB adds DOUBLEs in parallel, in an
        -- order that changes between runs, and the last-bit differences were
        -- enough to break the reproducibility test. Integers add exactly.
        sum(r.segundos) AS segundos,
        count(*) AS reproducoes,
        sum(r.eventos_buffering) AS buffering,
        sum(round(r.percentual_concluido * 100)::BIGINT) AS pct_centesimos,
        count(DISTINCT r.titulo_id) AS titulos,
        count(DISTINCT r.perfil_id) AS perfis_ativos,
        count(DISTINCT r.dispositivo_id) AS dispositivos_ativos
    FROM stg_plays r
    JOIN raw_perfis p USING (perfil_id)
    GROUP BY ALL
    """)

    con.execute("""
    CREATE OR REPLACE TABLE stg_sessions_daily AS
    SELECT p.conta_id, CAST(s.inicio_em AS DATE) AS dia, count(*) AS sessoes
    FROM raw_sessoes_app s
    JOIN raw_perfis p USING (perfil_id)
    GROUP BY ALL
    """)

    intent = ", ".join(f"'{event}'" for event in INTENT_EVENTS)
    con.execute(f"""
    CREATE OR REPLACE TABLE stg_app_daily AS
    SELECT
        p.conta_id,
        CAST(e.recebido_em AS DATE) AS dia,
        count(*) FILTER (WHERE e.tipo_evento = 'tentativa_cancelamento') AS tentativas_cancelamento,
        count(*) FILTER (WHERE e.tipo_evento = 'abrir_tela_planos') AS telas_plano,
        count(*) FILTER (WHERE e.tipo_evento = 'busca') AS buscas,
        count(*) FILTER (WHERE e.tipo_evento = 'adicionar_lista') AS adicoes_lista,
        count(*) FILTER (WHERE e.tipo_evento = 'avaliacao' AND e.avaliacao = 'negativa') AS avaliacoes_negativas,
        count(*) FILTER (WHERE e.tipo_evento = 'avaliacao' AND e.avaliacao = 'positiva') AS avaliacoes_positivas
    FROM raw_eventos_app e
    JOIN raw_perfis p USING (perfil_id)
    WHERE e.tipo_evento IN ({intent})
    GROUP BY ALL
    """)

    con.execute("""
    CREATE OR REPLACE TABLE stg_notifications AS
    SELECT conta_id, canal, campanha, enviada_em, aberta_em FROM raw_notificacoes
    """)

    con.execute("""
    CREATE OR REPLACE TABLE stg_tickets AS
    SELECT conta_id, aberto_em, categoria, resolvido_em, nota_satisfacao FROM raw_tickets_suporte
    """)

    con.execute("""
    CREATE OR REPLACE TABLE stg_nps AS
    SELECT conta_id, respondida_em, nota FROM raw_nps
    """)

    # Coupon-based staff flag, kept with its date: it only counts from the
    # moment the coupon was redeemed.
    con.execute("""
    CREATE OR REPLACE TABLE stg_staff_coupons AS
    SELECT r.conta_id, min(r.resgatado_em) AS desde
    FROM raw_cupom_resgates r JOIN raw_cupons u USING (cupom_id)
    WHERE u.codigo = 'COLABORADOR'
    GROUP BY 1
    """)
