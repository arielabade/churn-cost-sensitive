"""Features: one row per active subscriber per monthly cut-off, and its label.

Role in the pipeline
--------------------
Third stage. Turns the ``stg_*`` tables into the modelling table
``features``: for every cut-off ``t`` (first day of a month, 00:00 UTC) and
every subscription active at ``t``, the facts the business knew at ``t`` and
whether the subscriber asked to cancel during the month that follows.

Receives: a DuckDB connection with the ``stg_*`` tables from
:mod:`churn.staging`, and the cut-off range from :mod:`churn.config`.
Returns: the ``features`` table in DuckDB (also as a DataFrame from
:func:`load_frame`). Depends on: :mod:`churn.config`, :mod:`churn.staging`.

The contract that the leakage test enforces
-------------------------------------------
Every feature of the row ``(subscriber, t)`` is computed from rows whose
knowledge instant is strictly before ``t``. Deleting every row dated at or
after ``t`` from every source table, and blanking every column that records
something later than ``t``, must leave that row's features unchanged
(``tests/test_temporal_leakage.py`` does exactly this). Only the two label
columns look forward, and only into ``[t, t + 1 month)``.

Who is in the population at ``t``
---------------------------------
Subscriptions that started before ``t``, have not ended before ``t`` and had
not already asked to cancel before ``t``. A pending cancellation is a
save-desk case with a near-certain outcome; scoring it would inflate every
metric and teach the model nothing. Staff accounts are excluded: a retention
budget is not spent on employees.

What the label is
-----------------
``churn = 1`` when the subscriber requests cancellation inside the month after
``t``: the voluntary decision a retention offer can change. Involuntary
churn (payment failure after retries) is kept as ``churn_involuntario`` for
reporting, but it is not the target: a discount does not fix a declined card.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from .config import FIRST_CUTOFF, LAST_CUTOFF

KEYS = ("cutoff", "conta_id", "assinatura_id")
LABEL = "churn"

#: Columns that describe the economics of the customer, used by the decision
#: rule. ``receita_mensal`` is also a model feature (as ``MonthlyCharges`` was).
ECONOMICS_COLUMNS = ("receita_mensal", "custo_oferta")

#: Segment columns for reporting. ``plano`` and ``canal`` are also features.
SEGMENTS = ("plano", "canal")

#: Region, not state: 27 states split 5% churn into cells too thin to learn
#: from, and the tree spends splits memorising them.
CATEGORICAL = ("plano", "periodicidade", "metodo_pagamento", "canal", "regiao", "cupom_tipo")

NUMERIC = (
    # tenure and history
    "dias_conta", "dias_assinatura", "assinaturas_anteriores", "em_trial", "reconquistado",
    "faturas_emitidas", "dias_ate_renovacao",
    # price and discounts
    "receita_mensal", "preco_plano_mensal", "membros_extra", "reajuste_90d_pct",
    "variacao_fatura_pct", "desconto_ultima_fatura", "desconto_termina_proxima_fatura",
    "aviso_reajuste_45d",
    # payment
    "recusas_90d", "expirados_90d", "chargebacks_365d", "em_inadimplencia", "inadimplencias_180d",
    "trocas_metodo_180d", "falha_pagamento_35d",
    # plan changes
    "upgrades_180d", "downgrades_180d",
    # usage and its trend
    "horas_7d", "horas_28d", "horas_84d", "tendencia_uso_28d", "queda_uso_7d",
    "dias_ativos_28d", "reproducoes_28d", "buffering_por_reproducao_28d", "titulos_28d",
    "pct_concluido_28d", "perfis_assistindo_28d", "dispositivos_assistindo_28d",
    "sessoes_7d", "sessoes_28d", "dias_desde_ultimo_play", "dias_desde_ultima_sessao",
    # intent signals in the app
    "tentativas_cancelamento_28d", "tentativas_cancelamento_84d", "telas_plano_28d",
    "buscas_28d", "adicoes_lista_28d", "avaliacoes_negativas_84d", "avaliacoes_positivas_84d",
    # household
    "perfis", "perfis_infantis", "dispositivos", "tipos_dispositivo",
    # relationship
    "tickets_90d", "tickets_reproducao_90d", "tickets_cobranca_90d", "tickets_cancelamento_90d",
    "nota_suporte_180d", "nps_ultima_nota", "dias_desde_nps", "emails_90d", "taxa_abertura_email_90d",
    "taxa_abertura_push_90d", "reengajamento_35d",
    # account
    "idade", "indicada", "aceita_marketing",
)

FEATURES = CATEGORICAL + NUMERIC


def _window(source: str, ts: str, days: int, aggregates: str, name: str, by_date: bool = False) -> str:
    """Aggregate ``source`` over ``[t - days, t)`` for every cut-off.

    The range join is against the small ``cutoffs`` table first, so each
    source row is touched once per cut-off whose window contains it (at most
    four for a 90-day window), never once per subscriber.
    """
    start = "c.cutoff_date" if by_date else "c.cutoff"
    return f"""
    CREATE OR REPLACE TEMP TABLE {name} AS
    SELECT c.cutoff, s.conta_id, {aggregates}
    FROM {source} s
    JOIN cutoffs c ON s.{ts} >= {start} - INTERVAL {days} DAY AND s.{ts} < {start}
    GROUP BY ALL
    """


def build(con: duckdb.DuckDBPyConnection, first=FIRST_CUTOFF, last=LAST_CUTOFF) -> None:
    """Create ``cutoffs``, ``population`` and ``features``."""
    con.execute(f"""
    CREATE OR REPLACE TABLE cutoffs AS
    SELECT
        CAST(t AS TIMESTAMP) AS cutoff,
        CAST(t AS DATE) AS cutoff_date,
        CAST(t + INTERVAL 1 MONTH AS TIMESTAMP) AS fim
    FROM generate_series(TIMESTAMP '{first}', TIMESTAMP '{last}', INTERVAL 1 MONTH) g(t)
    """)

    con.execute("""
    CREATE OR REPLACE TABLE population AS
    SELECT
        c.cutoff, c.cutoff_date, s.conta_id, s.assinatura_id, s.periodicidade, s.iniciada_em,
        s.trial_fim_em, s.assinaturas_anteriores,
        -- The only two columns allowed to look past the cut-off.
        coalesce(s.cancelamento_solicitado_em >= c.cutoff AND s.cancelamento_solicitado_em < c.fim, FALSE)::INTEGER
            AS churn,
        coalesce(s.tipo_encerramento = 'involuntario' AND s.encerrada_em >= c.cutoff AND s.encerrada_em < c.fim,
                 FALSE)::INTEGER AS churn_involuntario
    FROM cutoffs c
    JOIN stg_subscriptions s
      ON s.iniciada_em < c.cutoff
     AND (s.encerrada_em IS NULL OR s.encerrada_em >= c.cutoff)
     AND (s.cancelamento_solicitado_em IS NULL OR s.cancelamento_solicitado_em >= c.cutoff)
    JOIN stg_accounts a ON a.conta_id = s.conta_id AND a.interna = 0
    LEFT JOIN stg_staff_coupons st ON st.conta_id = s.conta_id AND st.desde < c.cutoff
    WHERE st.conta_id IS NULL
    -- One subscription per account and month: the most recent, if a
    -- reactivation ever overlaps the previous period.
    QUALIFY row_number() OVER (PARTITION BY c.cutoff, s.conta_id ORDER BY s.iniciada_em DESC) = 1
    """)

    # ---------------------------------------------------------------- as-of lookups
    # ASOF JOIN picks, per row, the latest event strictly before the cut-off.
    con.execute("""
    CREATE OR REPLACE TEMP TABLE f_plan AS
    SELECT b.cutoff, b.conta_id, p.plano_id
    FROM population b
    ASOF LEFT JOIN stg_plan_events p ON b.assinatura_id = p.assinatura_id AND b.cutoff > p.ocorrido_em
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE charges AS
    SELECT conta_id, processado_em, metodo FROM stg_payments WHERE tipo = 'cobranca'
    """)
    con.execute("""
    CREATE OR REPLACE TEMP TABLE f_method AS
    SELECT b.cutoff, b.conta_id, p.metodo AS metodo_pagamento
    FROM population b
    ASOF LEFT JOIN charges p ON b.conta_id = p.conta_id AND b.cutoff > p.processado_em
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE invoices_ranked AS
    SELECT *,
        row_number() OVER (PARTITION BY assinatura_id ORDER BY emitida_em, fatura_id) AS ciclo,
        lag(valor_total) OVER (PARTITION BY assinatura_id ORDER BY emitida_em, fatura_id) AS valor_total_anterior
    FROM stg_invoices
    """)
    con.execute("""
    CREATE OR REPLACE TEMP TABLE f_invoice AS
    SELECT
        b.cutoff, b.conta_id,
        coalesce(i.ciclo, 0) AS faturas_emitidas,
        date_diff('day', b.cutoff_date, i.periodo_fim) + 1 AS dias_ate_renovacao,
        coalesce(i.valor_total / nullif(i.valor_total_anterior, 0) - 1, 0) AS variacao_fatura_pct,
        coalesce((i.valor_desconto > 0)::INTEGER, 0) AS desconto_ultima_fatura,
        -- The next invoice is the first at full price: the classic churn point
        -- of an acquisition promotion.
        coalesce((i.cupom_ciclo IS NOT NULL AND i.cupom_ciclo >= i.cupom_ciclos)::INTEGER, 0)
            AS desconto_termina_proxima_fatura,
        coalesce(i.cupom_tipo, 'nenhum') AS cupom_tipo
    FROM population b
    ASOF LEFT JOIN invoices_ranked i ON b.assinatura_id = i.assinatura_id AND b.cutoff > i.emitida_em
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE dunning AS
    SELECT assinatura_id, ocorrido_em, tipo FROM stg_lifecycle_events
    WHERE tipo IN ('inadimplencia_iniciada', 'inadimplencia_resolvida')
    """)
    con.execute("""
    CREATE OR REPLACE TEMP TABLE f_dunning AS
    SELECT b.cutoff, b.conta_id, coalesce((d.tipo = 'inadimplencia_iniciada')::INTEGER, 0) AS em_inadimplencia
    FROM population b
    ASOF LEFT JOIN dunning d ON b.assinatura_id = d.assinatura_id AND b.cutoff > d.ocorrido_em
    """)

    for name, table in (("f_last_play", "stg_usage_daily"), ("f_last_session", "stg_sessions_daily")):
        column = "dias_desde_ultimo_play" if name == "f_last_play" else "dias_desde_ultima_sessao"
        con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {name} AS
        SELECT b.cutoff, b.conta_id, date_diff('day', u.dia, b.cutoff_date) AS {column}
        FROM population b
        ASOF LEFT JOIN {table} u ON b.conta_id = u.conta_id AND b.cutoff_date > u.dia
        """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE f_nps AS
    SELECT b.cutoff, b.conta_id, n.nota AS nps_ultima_nota,
           date_diff('day', n.respondida_em, b.cutoff) AS dias_desde_nps
    FROM population b
    ASOF LEFT JOIN stg_nps n ON b.conta_id = n.conta_id AND b.cutoff > n.respondida_em
    """)

    # ---------------------------------------------------------------- prices
    # Price in force at the cut-off for the plan the subscriber is on at the
    # cut-off, from the price table, not from a later invoice. Extra members
    # are billed per active extra profile where the plan allows it.
    con.execute("""
    CREATE OR REPLACE TEMP TABLE extra_profiles AS
    SELECT c.cutoff, p.conta_id, count(*) FILTER (WHERE p.membro_extra = 1) AS membros_extra,
           count(*) AS perfis, count(*) FILTER (WHERE p.infantil = 1) AS perfis_infantis
    FROM stg_profiles p
    JOIN cutoffs c ON p.criado_em < c.cutoff AND (p.excluido_em IS NULL OR p.excluido_em >= c.cutoff)
    GROUP BY ALL
    """)
    con.execute("""
    CREATE OR REPLACE TEMP TABLE f_price AS
    SELECT
        b.cutoff, b.conta_id,
        pl.codigo AS plano,
        pr.valor_mensal AS preco_plano_mensal,
        coalesce(pr.valor_mensal / nullif(pr90.valor_mensal, 0) - 1, 0) AS reajuste_90d_pct,
        CASE WHEN pl.permite_membro_extra = 1 THEN coalesce(x.membros_extra, 0) ELSE 0 END AS membros_extra,
        pr.valor_mensal
          + CASE WHEN pl.permite_membro_extra = 1
                 THEN coalesce(x.membros_extra, 0) * coalesce(me.valor_mensal, 0) ELSE 0 END AS receita_mensal,
        -- The retention offer priced on this customer's own plan: percentage
        -- off for N cycles. Annual subscribers get the same N monthly
        -- equivalents, not N years.
        CASE o.tipo_desconto
            WHEN 'percentual' THEN o.valor_desconto / 100.0 * o.duracao_ciclos * pr.valor_mensal
            ELSE o.valor_desconto * o.duracao_ciclos
        END AS custo_oferta,
        coalesce(x.perfis, 0) AS perfis,
        coalesce(x.perfis_infantis, 0) AS perfis_infantis
    FROM population b
    JOIN f_plan fp USING (cutoff, conta_id)
    LEFT JOIN raw_planos pl ON pl.plano_id = fp.plano_id
    LEFT JOIN stg_prices pr
      ON pr.item = 'assinatura' AND pr.plano_id = fp.plano_id AND pr.periodicidade = b.periodicidade
     AND b.cutoff_date BETWEEN pr.vigencia_inicio AND pr.vigencia_fim
    LEFT JOIN stg_prices pr90
      ON pr90.item = 'assinatura' AND pr90.plano_id = fp.plano_id AND pr90.periodicidade = b.periodicidade
     AND b.cutoff_date - INTERVAL 90 DAY BETWEEN pr90.vigencia_inicio AND pr90.vigencia_fim
    LEFT JOIN stg_prices me
      ON me.item = 'membro_extra' AND b.cutoff_date BETWEEN me.vigencia_inicio AND me.vigencia_fim
    LEFT JOIN extra_profiles x USING (cutoff, conta_id)
    CROSS JOIN stg_retention_offer o
    """)

    # ---------------------------------------------------------------- windows
    con.execute(_window("stg_payments", "processado_em", 90, """
        count(*) FILTER (WHERE s.status = 'recusado') AS recusas_90d,
        count(*) FILTER (WHERE s.status = 'expirado') AS expirados_90d""", "f_pay90"))
    con.execute(_window("stg_payments", "processado_em", 365, """
        count(*) FILTER (WHERE s.tipo = 'chargeback') AS chargebacks_365d""", "f_pay365"))
    con.execute(_window("stg_lifecycle_events", "ocorrido_em", 180, """
        count(*) FILTER (WHERE s.tipo = 'upgrade') AS upgrades_180d,
        count(*) FILTER (WHERE s.tipo = 'downgrade') AS downgrades_180d,
        count(*) FILTER (WHERE s.tipo = 'metodo_pagamento_alterado') AS trocas_metodo_180d,
        count(*) FILTER (WHERE s.tipo = 'inadimplencia_iniciada') AS inadimplencias_180d""", "f_life"))
    con.execute(_window("stg_usage_daily", "dia", 84, """
        sum(s.segundos) FILTER (WHERE s.dia >= c.cutoff_date - INTERVAL 7 DAY) / 3600.0 AS horas_7d,
        sum(s.segundos) FILTER (WHERE s.dia >= c.cutoff_date - INTERVAL 28 DAY) / 3600.0 AS horas_28d,
        sum(s.segundos) / 3600.0 AS horas_84d,
        count(*) FILTER (WHERE s.dia >= c.cutoff_date - INTERVAL 28 DAY) AS dias_ativos_28d,
        sum(s.reproducoes) FILTER (WHERE s.dia >= c.cutoff_date - INTERVAL 28 DAY) AS reproducoes_28d,
        sum(s.buffering) FILTER (WHERE s.dia >= c.cutoff_date - INTERVAL 28 DAY) AS buffering_28d,
        sum(s.titulos) FILTER (WHERE s.dia >= c.cutoff_date - INTERVAL 28 DAY) AS titulos_28d,
        sum(s.pct_centesimos) FILTER (WHERE s.dia >= c.cutoff_date - INTERVAL 28 DAY) / 100.0
          / sum(s.reproducoes) FILTER (WHERE s.dia >= c.cutoff_date - INTERVAL 28 DAY) AS pct_concluido_28d,
        max(s.perfis_ativos) FILTER (WHERE s.dia >= c.cutoff_date - INTERVAL 28 DAY) AS perfis_assistindo_28d,
        max(s.dispositivos_ativos) FILTER (WHERE s.dia >= c.cutoff_date - INTERVAL 28 DAY)
            AS dispositivos_assistindo_28d""", "f_usage", by_date=True))
    con.execute(_window("stg_sessions_daily", "dia", 28, """
        sum(s.sessoes) FILTER (WHERE s.dia >= c.cutoff_date - INTERVAL 7 DAY) AS sessoes_7d,
        sum(s.sessoes) AS sessoes_28d""", "f_sessions", by_date=True))
    con.execute(_window("stg_app_daily", "dia", 84, """
        sum(s.tentativas_cancelamento) FILTER (WHERE s.dia >= c.cutoff_date - INTERVAL 28 DAY)
            AS tentativas_cancelamento_28d,
        sum(s.tentativas_cancelamento) AS tentativas_cancelamento_84d,
        sum(s.telas_plano) FILTER (WHERE s.dia >= c.cutoff_date - INTERVAL 28 DAY) AS telas_plano_28d,
        sum(s.buscas) FILTER (WHERE s.dia >= c.cutoff_date - INTERVAL 28 DAY) AS buscas_28d,
        sum(s.adicoes_lista) FILTER (WHERE s.dia >= c.cutoff_date - INTERVAL 28 DAY) AS adicoes_lista_28d,
        sum(s.avaliacoes_negativas) AS avaliacoes_negativas_84d,
        sum(s.avaliacoes_positivas) AS avaliacoes_positivas_84d""", "f_app", by_date=True))
    con.execute(_window("stg_tickets", "aberto_em", 90, """
        count(*) AS tickets_90d,
        count(*) FILTER (WHERE s.categoria = 'reproducao') AS tickets_reproducao_90d,
        count(*) FILTER (WHERE s.categoria IN ('cobranca', 'pagamento')) AS tickets_cobranca_90d,
        count(*) FILTER (WHERE s.categoria = 'cancelamento') AS tickets_cancelamento_90d""", "f_tickets"))
    # A satisfaction score exists only once the ticket is resolved, so a
    # ticket resolved after the cut-off contributes nothing yet.
    con.execute(_window("stg_tickets", "resolvido_em", 180, """
        avg(s.nota_satisfacao) AS nota_suporte_180d""", "f_ticket_score"))
    # Opens count only if they happened before the cut-off: an email sent on
    # the 30th and opened on the 2nd is, at the cut-off, an unopened email.
    # The coalesce matters: without it a never-opened email (NULL) drops out
    # of the average while a later-opened one counts as 0, so the mere
    # existence of a future open would change the denominator. The leakage
    # test caught exactly that.
    con.execute(_window("stg_notifications", "enviada_em", 90, """
        count(*) FILTER (WHERE s.canal = 'email') AS emails_90d,
        avg(coalesce(s.aberta_em < c.cutoff, FALSE)::INTEGER) FILTER (WHERE s.canal = 'email')
            AS taxa_abertura_email_90d,
        avg(coalesce(s.aberta_em < c.cutoff, FALSE)::INTEGER) FILTER (WHERE s.canal = 'push')
            AS taxa_abertura_push_90d,
        count(*) FILTER (WHERE s.campanha = 'reengajamento' AND s.enviada_em >= c.cutoff - INTERVAL 35 DAY)
            AS reengajamento_35d,
        count(*) FILTER (WHERE s.campanha = 'aviso_reajuste' AND s.enviada_em >= c.cutoff - INTERVAL 45 DAY)
            AS aviso_reajuste_45d,
        count(*) FILTER (WHERE s.campanha = 'falha_pagamento' AND s.enviada_em >= c.cutoff - INTERVAL 35 DAY)
            AS falha_pagamento_35d""", "f_notif"))

    con.execute("""
    CREATE OR REPLACE TEMP TABLE f_devices AS
    SELECT c.cutoff, d.conta_id, count(*) AS dispositivos, count(DISTINCT d.tipo) AS tipos_dispositivo
    FROM stg_devices d JOIN cutoffs c ON d.registrado_em < c.cutoff
    GROUP BY ALL
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE f_winback AS
    SELECT DISTINCT b.cutoff, b.conta_id, 1 AS reconquistado
    FROM population b
    JOIN raw_cupom_resgates r ON r.assinatura_id = b.assinatura_id AND r.resgatado_em < b.cutoff
    JOIN raw_cupons u ON u.cupom_id = r.cupom_id AND u.codigo = 'VOLTA50'
    """)

    # ---------------------------------------------------------------- assemble
    # Missing-value policy, decided per feature rather than by a blanket fill:
    # * counts and sums: no rows in the window is a fact (zero), so 0;
    # * recency ("days since"), NPS score and age: no event is not zero, it is
    #   unknown, so they stay NULL. LightGBM routes NULL down its own branch;
    #   the logistic baseline imputes and adds a missing-indicator column.
    con.execute("""
    CREATE OR REPLACE TABLE features AS
    SELECT
        b.cutoff, conta_id, b.assinatura_id, b.churn, b.churn_involuntario,
        p.plano, b.periodicidade, coalesce(m.metodo_pagamento, 'sem_cobranca') AS metodo_pagamento,
        a.canal, a.regiao, i.cupom_tipo,

        date_diff('day', a.criada_em, b.cutoff) AS dias_conta,
        date_diff('day', b.iniciada_em, b.cutoff) AS dias_assinatura,
        b.assinaturas_anteriores,
        coalesce((b.trial_fim_em > b.cutoff)::INTEGER, 0) AS em_trial,
        coalesce(w.reconquistado, 0) AS reconquistado,
        i.faturas_emitidas, i.dias_ate_renovacao,

        p.receita_mensal, p.custo_oferta, p.preco_plano_mensal, p.membros_extra, p.reajuste_90d_pct,
        i.variacao_fatura_pct, i.desconto_ultima_fatura, i.desconto_termina_proxima_fatura,
        coalesce(n.aviso_reajuste_45d, 0) AS aviso_reajuste_45d,

        coalesce(y.recusas_90d, 0) AS recusas_90d,
        coalesce(y.expirados_90d, 0) AS expirados_90d,
        coalesce(y365.chargebacks_365d, 0) AS chargebacks_365d,
        d.em_inadimplencia,
        coalesce(l.inadimplencias_180d, 0) AS inadimplencias_180d,
        coalesce(l.trocas_metodo_180d, 0) AS trocas_metodo_180d,
        coalesce(n.falha_pagamento_35d, 0) AS falha_pagamento_35d,
        coalesce(l.upgrades_180d, 0) AS upgrades_180d,
        coalesce(l.downgrades_180d, 0) AS downgrades_180d,

        coalesce(u.horas_7d, 0) AS horas_7d,
        coalesce(u.horas_28d, 0) AS horas_28d,
        coalesce(u.horas_84d, 0) AS horas_84d,
        -- Trend: daily hours in the last 28 days against the 56 before them.
        -- The +0.1 h/day smoothing keeps a light user going from 0.02 to
        -- 0.01 h/day from looking like a collapse.
        (coalesce(u.horas_28d, 0) / 28 + 0.1)
          / ((coalesce(u.horas_84d, 0) - coalesce(u.horas_28d, 0)) / 56 + 0.1) - 1 AS tendencia_uso_28d,
        (coalesce(u.horas_7d, 0) / 7 + 0.1) / (coalesce(u.horas_28d, 0) / 28 + 0.1) - 1 AS queda_uso_7d,
        coalesce(u.dias_ativos_28d, 0) AS dias_ativos_28d,
        coalesce(u.reproducoes_28d, 0) AS reproducoes_28d,
        coalesce(u.buffering_28d / nullif(u.reproducoes_28d, 0), 0) AS buffering_por_reproducao_28d,
        coalesce(u.titulos_28d, 0) AS titulos_28d,
        u.pct_concluido_28d,
        coalesce(u.perfis_assistindo_28d, 0) AS perfis_assistindo_28d,
        coalesce(u.dispositivos_assistindo_28d, 0) AS dispositivos_assistindo_28d,
        coalesce(s.sessoes_7d, 0) AS sessoes_7d,
        coalesce(s.sessoes_28d, 0) AS sessoes_28d,
        lp.dias_desde_ultimo_play,
        ls.dias_desde_ultima_sessao,

        coalesce(e.tentativas_cancelamento_28d, 0) AS tentativas_cancelamento_28d,
        coalesce(e.tentativas_cancelamento_84d, 0) AS tentativas_cancelamento_84d,
        coalesce(e.telas_plano_28d, 0) AS telas_plano_28d,
        coalesce(e.buscas_28d, 0) AS buscas_28d,
        coalesce(e.adicoes_lista_28d, 0) AS adicoes_lista_28d,
        coalesce(e.avaliacoes_negativas_84d, 0) AS avaliacoes_negativas_84d,
        coalesce(e.avaliacoes_positivas_84d, 0) AS avaliacoes_positivas_84d,

        p.perfis, p.perfis_infantis,
        coalesce(dv.dispositivos, 0) AS dispositivos,
        coalesce(dv.tipos_dispositivo, 0) AS tipos_dispositivo,

        coalesce(t.tickets_90d, 0) AS tickets_90d,
        coalesce(t.tickets_reproducao_90d, 0) AS tickets_reproducao_90d,
        coalesce(t.tickets_cobranca_90d, 0) AS tickets_cobranca_90d,
        coalesce(t.tickets_cancelamento_90d, 0) AS tickets_cancelamento_90d,
        ts.nota_suporte_180d,
        np.nps_ultima_nota, np.dias_desde_nps,
        coalesce(n.emails_90d, 0) AS emails_90d,
        n.taxa_abertura_email_90d, n.taxa_abertura_push_90d,
        coalesce(n.reengajamento_35d, 0) AS reengajamento_35d,

        date_diff('year', a.data_nascimento, b.cutoff_date) AS idade,
        a.indicada, a.aceita_marketing
    FROM population b
    JOIN stg_accounts a USING (conta_id)
    LEFT JOIN f_price p USING (cutoff, conta_id)
    LEFT JOIN f_method m USING (cutoff, conta_id)
    LEFT JOIN f_invoice i USING (cutoff, conta_id)
    LEFT JOIN f_dunning d USING (cutoff, conta_id)
    LEFT JOIN f_last_play lp USING (cutoff, conta_id)
    LEFT JOIN f_last_session ls USING (cutoff, conta_id)
    LEFT JOIN f_nps np USING (cutoff, conta_id)
    LEFT JOIN f_pay90 y USING (cutoff, conta_id)
    LEFT JOIN f_pay365 y365 USING (cutoff, conta_id)
    LEFT JOIN f_life l USING (cutoff, conta_id)
    LEFT JOIN f_usage u USING (cutoff, conta_id)
    LEFT JOIN f_sessions s USING (cutoff, conta_id)
    LEFT JOIN f_app e USING (cutoff, conta_id)
    LEFT JOIN f_tickets t USING (cutoff, conta_id)
    LEFT JOIN f_ticket_score ts USING (cutoff, conta_id)
    LEFT JOIN f_notif n USING (cutoff, conta_id)
    LEFT JOIN f_devices dv USING (cutoff, conta_id)
    LEFT JOIN f_winback w USING (cutoff, conta_id)
    ORDER BY b.cutoff, conta_id
    """)


def load_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """The ``features`` table as a DataFrame with categorical dtypes set."""
    frame = con.execute("SELECT * FROM features ORDER BY cutoff, conta_id").df()
    for column in CATEGORICAL:
        frame[column] = frame[column].astype("category")
    return frame
