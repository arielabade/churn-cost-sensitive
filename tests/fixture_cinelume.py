"""A miniature Cinelume, written to Parquet, for tests that must run without the generator.

Same tables and columns as the real export (only the ones the pipeline reads,
plus the end-of-data snapshot columns it must ignore), a few hundred accounts
and a year and a half of history. It is not realistic; it is consistent:
every subscription has a creation event, invoices follow the billing cycle,
cancellations are preceded by falling usage and a cancellation attempt, so
features and labels are non-trivial and the model has something to learn.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

START = dt.datetime(2023, 10, 1)
END = dt.datetime(2025, 4, 1)
PRICES = {1: (18.90, 189.00), 2: (39.90, 399.00), 3: (55.90, 559.00)}
CHANNELS = ["organico", "busca_paga", "social_pago", "indicacao", "parceria_operadora", "tv"]


def _ts(rng, start: dt.datetime, end: dt.datetime) -> dt.datetime:
    seconds = int((end - start).total_seconds())
    return start + dt.timedelta(seconds=int(rng.integers(0, max(seconds, 1))))


def make_cinelume(directory: Path, n_accounts: int = 400, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict]] = {name: [] for name in (
        "contas", "perfis", "dispositivos", "assinaturas", "assinatura_eventos", "cupom_resgates",
        "faturas", "pagamentos", "notificacoes", "tickets_suporte", "nps", "sessoes_app",
        "eventos_app", "reproducoes")}
    ids = {name: 0 for name in rows}

    def nid(name):
        ids[name] += 1
        return ids[name]

    for conta in range(1, n_accounts + 1):
        created = _ts(rng, START, END - dt.timedelta(days=60))
        staff = conta % 97 == 0
        rows["contas"].append(dict(
            conta_id=conta, email=f"u{conta}@{'cinelume.example' if staff else 'mail.example.com'}",
            data_nascimento=dt.date(int(rng.integers(1950, 2005)), 1, 1) if rng.random() > 0.1 else None,
            uf=rng.choice(["SP", "sp", "São Paulo", "RJ", "MG", "BA", "RS"]), criada_em=created,
            canal_aquisicao_id=int(rng.integers(1, 7)), indicada_por_conta_id=None,
            aceita_marketing=int(rng.random() < 0.6),
            # end-of-data snapshot noise the pipeline must not read
        ))
        perfis = []
        for k in range(int(rng.integers(1, 4))):
            pid = nid("perfis")
            perfis.append(pid)
            rows["perfis"].append(dict(
                perfil_id=pid, conta_id=conta, infantil=int(k == 2), membro_extra=0,
                criado_em=created + dt.timedelta(minutes=k), excluido_em=None))
        devices = []
        for _ in range(int(rng.integers(1, 3))):
            did = nid("dispositivos")
            devices.append(did)
            rows["dispositivos"].append(dict(dispositivo_id=did, conta_id=conta,
                                             tipo=rng.choice(["tv", "celular", "web"]), plataforma="mobile_os_a",
                                             registrado_em=created + dt.timedelta(hours=1)))

        plan = int(rng.integers(1, 4))
        annual = rng.random() < 0.1
        method = rng.choice(["cartao", "pix", "boleto"], p=[0.7, 0.2, 0.1])
        engagement = rng.gamma(2.0, 1.0)
        will_churn = rng.random() < 0.8 / (1 + engagement)
        churn_at = _ts(rng, created + dt.timedelta(days=40), END + dt.timedelta(days=30)) if will_churn else None
        sub = nid("assinaturas")
        ended = churn_at + dt.timedelta(days=20) if churn_at else None
        rows["assinaturas"].append(dict(
            assinatura_id=sub, conta_id=conta, plano_id=int(rng.integers(1, 4)),
            periodicidade="anual" if annual else "mensal", metodo_pagamento=rng.choice(["cartao", "pix"]),
            status=rng.choice(["ativa", "encerrada"]), iniciada_em=created, trial_fim_em=None,
            cancelamento_solicitado_em=churn_at, encerrada_em=ended,
            tipo_encerramento="voluntario" if churn_at else None,
            renovacao_automatica=int(method == "cartao"), assinatura_anterior_id=None))
        rows["assinatura_eventos"].append(dict(
            evento_id=nid("assinatura_eventos"), assinatura_id=sub, tipo="assinatura_criada",
            ocorrido_em=created, plano_anterior_id=None, plano_novo_id=plan, origem="app"))
        if rng.random() < 0.3:
            when = _ts(rng, created, END)
            new_plan = min(3, plan + 1)
            rows["assinatura_eventos"].append(dict(
                evento_id=nid("assinatura_eventos"), assinatura_id=sub, tipo="upgrade", ocorrido_em=when,
                plano_anterior_id=plan, plano_novo_id=new_plan, origem="app"))
        if churn_at:
            rows["assinatura_eventos"].append(dict(
                evento_id=nid("assinatura_eventos"), assinatura_id=sub, tipo="cancelamento_solicitado",
                ocorrido_em=churn_at, plano_anterior_id=None, plano_novo_id=None, origem="app"))
        resgate = None
        if rng.random() < 0.3:
            resgate = nid("cupom_resgates")
            rows["cupom_resgates"].append(dict(resgate_id=resgate, cupom_id=int(rng.choice([2, 3, 4])),
                                               conta_id=conta, assinatura_id=sub, resgatado_em=created))
        if staff:
            rows["cupom_resgates"].append(dict(resgate_id=nid("cupom_resgates"), cupom_id=5, conta_id=conta,
                                               assinatura_id=sub, resgatado_em=created))

        stop = min(ended or END, END)
        cycle_start = created.date()
        cycle = 0
        while dt.datetime.combine(cycle_start, dt.time()) < stop:
            cycle += 1
            months = 12 if annual else 1
            month_index = cycle_start.month - 1 + months
            cycle_end = dt.date(cycle_start.year + month_index // 12, month_index % 12 + 1,
                                min(cycle_start.day, 28)) - dt.timedelta(days=1)
            gross = PRICES[plan][1 if annual else 0]
            discount = round(gross * 0.5, 2) if resgate and cycle <= 2 else 0.0
            fid = nid("faturas")
            emitted = dt.datetime.combine(cycle_start, dt.time(3))
            rows["faturas"].append(dict(
                fatura_id=fid, assinatura_id=sub, conta_id=conta, preco_id=plan,
                cupom_resgate_id=resgate if discount else None, periodo_inicio=cycle_start,
                periodo_fim=cycle_end, emitida_em=emitted, vencimento_em=cycle_start,
                valor_bruto=gross, valor_desconto=discount, valor_membro_extra=0.0,
                valor_total=round(gross - discount, 2), status=rng.choice(["paga", "vencida"])))
            declined = rng.random() < 0.08
            rows["pagamentos"].append(dict(
                pagamento_id=nid("pagamentos"), fatura_id=fid, conta_id=conta, metodo=method,
                tipo="cobranca", tentativa=1, processado_em=emitted + dt.timedelta(hours=1),
                valor_brl=gross - discount, status="recusado" if declined else "aprovado",
                codigo_recusa="saldo_insuficiente" if declined else None))
            cycle_start = cycle_end + dt.timedelta(days=1)

        day = created
        while day < stop:
            fading = churn_at is not None and (churn_at - day).days < 30
            rate = engagement * (0.3 if fading else 1.0)
            for _ in range(int(rng.poisson(min(rate, 4)))):
                perfil = int(rng.choice(perfis))
                device = int(rng.choice(devices))
                start = day + dt.timedelta(hours=int(rng.integers(8, 23)))
                session = nid("sessoes_app")
                rows["sessoes_app"].append(dict(sessao_id=session, perfil_id=perfil, dispositivo_id=device,
                                                versao_app="5.0.0", inicio_em=start,
                                                fim_em=start + dt.timedelta(hours=1), ip="10.0.0.1", regiao="Sudeste"))
                rows["reproducoes"].append(dict(
                    reproducao_id=nid("reproducoes"), perfil_id=perfil, sessao_id=session, dispositivo_id=device,
                    titulo_id=int(rng.integers(1, 50)), episodio_id=None, inicio_em=start,
                    fim_em=start + dt.timedelta(minutes=50), segundos_assistidos=int(rng.integers(300, 4000)),
                    percentual_concluido=float(rng.uniform(10, 100)), qualidade_video="1080p",
                    eventos_buffering=int(rng.poisson(0.3)), origem_autoplay=0, origem_recomendacao=0,
                    versao_app="5.0.0"))
                kind = rng.choice(["busca", "avaliacao", "adicionar_lista", "impressao_titulo"])
                # Device clock and server clock differ; received a little later.
                rows["eventos_app"].append(dict(
                    evento_id=nid("eventos_app"), sessao_id=session, perfil_id=perfil, seq=1, tipo_evento=kind,
                    ocorrido_em=start, recebido_em=start + dt.timedelta(minutes=int(rng.integers(0, 600))),
                    titulo_id=None, fileira=None, posicao=None, termo_busca=None,
                    avaliacao=rng.choice(["positiva", "negativa"]) if kind == "avaliacao" else None,
                    versao_app="5.0.0"))
            day += dt.timedelta(days=1)
        if churn_at:
            attempt = churn_at - dt.timedelta(days=int(rng.integers(1, 20)))
            rows["eventos_app"].append(dict(
                evento_id=nid("eventos_app"), sessao_id=0, perfil_id=perfis[0], seq=1,
                tipo_evento="tentativa_cancelamento", ocorrido_em=attempt,
                recebido_em=attempt + dt.timedelta(hours=2), titulo_id=None, fileira=None, posicao=None,
                termo_busca=None, avaliacao=None, versao_app="5.0.0"))

        for _ in range(int(rng.integers(0, 6))):
            sent = _ts(rng, created, END)
            opened = sent + dt.timedelta(days=int(rng.integers(0, 5))) if rng.random() < 0.4 else None
            rows["notificacoes"].append(dict(
                notificacao_id=nid("notificacoes"), conta_id=conta, perfil_id=None,
                canal=rng.choice(["email", "push"]),
                campanha=rng.choice(["destaques_semana", "reengajamento", "aviso_reajuste"]),
                titulo_id=None, enviada_em=sent, aberta_em=opened, clicada_em=None))
        if rng.random() < 0.3:
            opened_at = _ts(rng, created, END)
            resolved = opened_at + dt.timedelta(days=int(rng.integers(0, 10)))
            rows["tickets_suporte"].append(dict(
                ticket_id=nid("tickets_suporte"), conta_id=conta, aberto_em=opened_at, canal_atendimento="chat",
                categoria=rng.choice(["reproducao", "cobranca", "cancelamento"]), status="resolvido",
                resolvido_em=resolved, nota_satisfacao=int(rng.integers(1, 6))))
        if rng.random() < 0.2:
            rows["nps"].append(dict(resposta_id=nid("nps"), conta_id=conta,
                                    respondida_em=_ts(rng, created, END), nota=int(rng.integers(0, 11)),
                                    comentario=None, canal_pesquisa="email"))

    static = {
        "canais": pd.DataFrame({"canal_id": range(1, 7), "codigo": CHANNELS, "nome": CHANNELS,
                                "tipo": "pago", "pago": 1}),
        "planos": pd.DataFrame({"plano_id": [1, 2, 3], "codigo": ["basico_anuncios", "padrao", "premium"],
                                "nome": ["B", "P", "Pr"], "com_anuncios": [1, 0, 0],
                                "resolucao_maxima": ["720p", "1080p", "4k"], "telas_simultaneas": [2, 2, 4],
                                "max_perfis": 5, "permite_membro_extra": [0, 1, 1]}),
        "precos": pd.DataFrame(
            [dict(preco_id=i * 2 + j + 1, plano_id=plan, item="assinatura", periodicidade=per, valor_brl=value,
                  vigencia_inicio=dt.date(2023, 1, 1), vigencia_fim=None)
             for i, (plan, values) in enumerate(PRICES.items())
             for j, (per, value) in enumerate(zip(("mensal", "anual"), values))]
            + [dict(preco_id=99, plano_id=None, item="membro_extra", periodicidade="mensal", valor_brl=12.9,
                    vigencia_inicio=dt.date(2025, 3, 10), vigencia_fim=None)]),
        "cupons": pd.DataFrame({
            "cupom_id": [1, 2, 3, 4, 5], "codigo": ["VOLTA50", "BLACK50-2023", "PARCEIRO3M", "INDICA30", "COLABORADOR"],
            "descricao": "x", "tipo_desconto": "percentual", "valor_desconto": [50.0, 50.0, 50.0, 30.0, 100.0],
            "duracao_ciclos": [2, 3, 3, 1, 99], "valido_de": dt.date(2023, 1, 1), "valido_ate": None,
            "canal_id": None, "apenas_novos": 0}),
    }
    for name, frame in static.items():
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    for name, records in rows.items():
        pd.DataFrame.from_records(records).to_parquet(directory / f"{name}.parquet", index=False)
    return directory


#: Knowledge instant of each table: the column whose value says when a row
#: became known. Rows at or after a cut-off do not exist yet at that cut-off.
KNOWN_AT = {
    "contas": "criada_em", "perfis": "criado_em", "dispositivos": "registrado_em",
    "assinaturas": "iniciada_em", "assinatura_eventos": "ocorrido_em", "cupom_resgates": "resgatado_em",
    "faturas": "emitida_em", "pagamentos": "processado_em", "notificacoes": "enviada_em",
    "tickets_suporte": "aberto_em", "nps": "respondida_em", "sessoes_app": "inicio_em",
    "eventos_app": "recebido_em", "reproducoes": "inicio_em",
}

#: Columns that record something happening later than the row itself. At a
#: cut-off before that later instant, the column was still empty.
LATER = {
    "assinaturas": ("cancelamento_solicitado_em", "encerrada_em"),
    "perfis": ("excluido_em",),
    "notificacoes": ("aberta_em", "clicada_em"),
    "tickets_suporte": ("resolvido_em",),
}

#: End-of-data photographs. In the truncated world they are scrambled: if a
#: feature moved, the pipeline was reading the future through them.
SNAPSHOTS = {
    "assinaturas": ("plano_id", "metodo_pagamento", "status"),
    "faturas": ("status",),
    "tickets_suporte": ("status",),
}


def truncate_at(source: Path, target: Path, cutoff: dt.datetime, seed: int = 0) -> Path:
    """Rewrite every table as it would have looked at ``cutoff``.

    Done in DuckDB, streaming file to file, so it also works on the full
    generated data (tens of millions of rows) without loading it into memory.
    Snapshot columns are altered on every row (numbers rotated, text
    reversed): if any feature moved, the pipeline was reading them.
    """
    import duckdb

    target.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit = '2GB'")
    con.execute(f"SET temp_directory = '{target / '.tmp'}'")
    stamp = f"TIMESTAMP '{cutoff:%Y-%m-%d %H:%M:%S}'"
    for path in sorted(source.glob("*.parquet")):
        name = path.stem
        columns = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]
        select = []
        for column in columns:
            expression = column
            if column in LATER.get(name, ()):
                expression = f"CASE WHEN {column} >= {stamp} THEN NULL ELSE {column} END"
            if name == "assinaturas" and column == "tipo_encerramento":
                expression = f"CASE WHEN encerrada_em >= {stamp} THEN NULL ELSE {column} END"
            if name == "tickets_suporte" and column == "nota_satisfacao":
                expression = f"CASE WHEN resolvido_em >= {stamp} THEN NULL ELSE {column} END"
            if column in SNAPSHOTS.get(name, ()):
                expression = (f"(({column} % 3) + 1)" if column.endswith("_id") else f"reverse({column})")
            select.append(f"{expression} AS {column}")
        where = f"WHERE {KNOWN_AT[name]} < {stamp}" if name in KNOWN_AT else ""
        con.execute(f"COPY (SELECT {', '.join(select)} FROM read_parquet('{path}') {where}) "
                    f"TO '{target / path.name}' (FORMAT parquet)")
    con.close()
    return target
