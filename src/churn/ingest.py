"""Ingestion: expose the Cinelume tables to DuckDB as typed ``raw_*`` views.

Role in the pipeline
--------------------
First stage. Everything downstream is SQL over ``raw_<table>`` views, so the
rest of the pipeline does not know or care whether the rows came from the
MySQL database or from the generator's Parquet exports.

Receives: a backend name (``parquet`` or ``mysql``) and its location.
Returns: a DuckDB connection with one ``raw_<table>`` view per table used and
a ``raw_provenance`` table saying where each one was read from.
Depends on: :mod:`churn.config` for paths.

Two decisions live here rather than downstream
----------------------------------------------
* **Only the columns the pipeline is allowed to use are exposed.** Several
  columns are a photograph of 31/10/2026, not a history:
  ``assinaturas.plano_id``, ``assinaturas.metodo_pagamento``,
  ``assinaturas.status``, ``faturas.status`` and ``tickets_suporte.status``.
  Used as features they would tell the model the future. They are not
  ingested at all, so no later query can leak them by accident; plan and
  payment method at each cut-off are rebuilt from the event history instead.
* **Types are normalised in the view** (money as DOUBLE, instants as
  TIMESTAMP), because MySQL returns DECIMAL and Parquet returns double, and
  the staging SQL must produce identical numbers from either.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import duckdb

from .config import DATA_DIR, PARQUET_DIR
from .guard import assert_not_answer_key

#: Columns read per table, with the type each is normalised to.
TABLES: dict[str, dict[str, str]] = {
    "canais": {"canal_id": "INTEGER", "codigo": "VARCHAR"},
    "planos": {"plano_id": "INTEGER", "codigo": "VARCHAR", "permite_membro_extra": "INTEGER"},
    "precos": {
        "preco_id": "INTEGER", "plano_id": "INTEGER", "item": "VARCHAR", "periodicidade": "VARCHAR",
        "valor_brl": "DOUBLE", "vigencia_inicio": "DATE", "vigencia_fim": "DATE",
    },
    "contas": {
        "conta_id": "INTEGER", "email": "VARCHAR", "data_nascimento": "DATE", "uf": "VARCHAR",
        "criada_em": "TIMESTAMP", "canal_aquisicao_id": "INTEGER",
        "indicada_por_conta_id": "INTEGER", "aceita_marketing": "INTEGER",
    },
    "perfis": {
        "perfil_id": "INTEGER", "conta_id": "INTEGER", "infantil": "INTEGER", "membro_extra": "INTEGER",
        "criado_em": "TIMESTAMP", "excluido_em": "TIMESTAMP",
    },
    "dispositivos": {
        "dispositivo_id": "INTEGER", "conta_id": "INTEGER", "tipo": "VARCHAR", "plataforma": "VARCHAR",
        "registrado_em": "TIMESTAMP",
    },
    # plano_id, metodo_pagamento and status deliberately absent: end-of-data snapshots.
    "assinaturas": {
        "assinatura_id": "INTEGER", "conta_id": "INTEGER", "periodicidade": "VARCHAR",
        "iniciada_em": "TIMESTAMP", "trial_fim_em": "TIMESTAMP",
        "cancelamento_solicitado_em": "TIMESTAMP", "encerrada_em": "TIMESTAMP",
        "tipo_encerramento": "VARCHAR", "assinatura_anterior_id": "INTEGER",
    },
    "assinatura_eventos": {
        "evento_id": "BIGINT", "assinatura_id": "INTEGER", "tipo": "VARCHAR", "ocorrido_em": "TIMESTAMP",
        "plano_anterior_id": "INTEGER", "plano_novo_id": "INTEGER",
    },
    "cupons": {
        "cupom_id": "INTEGER", "codigo": "VARCHAR", "tipo_desconto": "VARCHAR",
        "valor_desconto": "DOUBLE", "duracao_ciclos": "INTEGER",
    },
    "cupom_resgates": {
        "resgate_id": "INTEGER", "cupom_id": "INTEGER", "conta_id": "INTEGER",
        "assinatura_id": "INTEGER", "resgatado_em": "TIMESTAMP",
    },
    # status deliberately absent: it is the invoice's state on 31/10/2026.
    "faturas": {
        "fatura_id": "BIGINT", "assinatura_id": "INTEGER", "conta_id": "INTEGER",
        "cupom_resgate_id": "INTEGER", "periodo_inicio": "DATE", "periodo_fim": "DATE",
        "emitida_em": "TIMESTAMP", "valor_bruto": "DOUBLE", "valor_desconto": "DOUBLE",
        "valor_membro_extra": "DOUBLE", "valor_total": "DOUBLE",
    },
    "pagamentos": {
        "pagamento_id": "BIGINT", "fatura_id": "BIGINT", "conta_id": "INTEGER", "metodo": "VARCHAR",
        "tipo": "VARCHAR", "tentativa": "INTEGER", "processado_em": "TIMESTAMP", "status": "VARCHAR",
    },
    "notificacoes": {
        "conta_id": "INTEGER", "canal": "VARCHAR", "campanha": "VARCHAR",
        "enviada_em": "TIMESTAMP", "aberta_em": "TIMESTAMP",
    },
    # status deliberately absent: it is the ticket's state on 31/10/2026.
    "tickets_suporte": {
        "conta_id": "INTEGER", "aberto_em": "TIMESTAMP", "categoria": "VARCHAR",
        "resolvido_em": "TIMESTAMP", "nota_satisfacao": "INTEGER",
    },
    "nps": {"conta_id": "INTEGER", "respondida_em": "TIMESTAMP", "nota": "INTEGER"},
    "sessoes_app": {"perfil_id": "INTEGER", "dispositivo_id": "INTEGER", "inicio_em": "TIMESTAMP"},
    "eventos_app": {
        "perfil_id": "INTEGER", "tipo_evento": "VARCHAR", "recebido_em": "TIMESTAMP", "avaliacao": "VARCHAR",
    },
    "reproducoes": {
        "perfil_id": "INTEGER", "dispositivo_id": "INTEGER", "titulo_id": "INTEGER",
        "inicio_em": "TIMESTAMP", "fim_em": "TIMESTAMP", "segundos_assistidos": "BIGINT",
        "percentual_concluido": "DOUBLE", "eventos_buffering": "INTEGER", "versao_app": "VARCHAR",
    },
}

#: Usage logs. At full scale these are tens of millions of rows each; loading
#: them into MySQL takes hours, so the MySQL backend falls back to Parquet for
#: any of them the database does not hold.
LOG_TABLES = ("sessoes_app", "eventos_app", "reproducoes")


@dataclass(frozen=True)
class MySQLSettings:
    """Connection to the Cinelume MySQL started by ``infra/docker-compose.yml``."""

    host: str = os.environ.get("CINELUME_MYSQL_HOST", "127.0.0.1")
    port: int = int(os.environ.get("CINELUME_MYSQL_PORT", "3317"))
    user: str = os.environ.get("CINELUME_MYSQL_USER", "root")
    password: str = os.environ.get("CINELUME_MYSQL_PASSWORD", "cinelume")
    database: str = "cinelume"

    def dsn(self) -> str:
        """Connection string in the form DuckDB's MySQL extension expects."""
        return (f"host={self.host} port={self.port} user={self.user} "
                f"password={self.password} database={self.database}")


def _select(table: str, source: str) -> str:
    columns = ", ".join(f"CAST({col} AS {kind}) AS {col}" for col, kind in TABLES[table].items())
    return f"SELECT {columns} FROM {source}"


def connect(
    backend: str = "parquet",
    parquet_dir: Path = PARQUET_DIR,
    database: Path | str = ":memory:",
    mysql: MySQLSettings | None = None,
    memory_limit: str = os.environ.get("CINELUME_DUCKDB_MEMORY", "4GB"),
) -> duckdb.DuckDBPyConnection:
    """Open DuckDB and register one ``raw_<table>`` view per table.

    ``database`` may be a file, so staged and feature tables persist between
    runs and can be inspected; ``:memory:`` is used by the tests.
    """
    # DuckDB reads files in C++, out of reach of the runtime guard's audit
    # hook, so the path is checked here, where it is handed over.
    assert_not_answer_key(parquet_dir)
    if str(database) != ":memory:":
        assert_not_answer_key(database)
    con = duckdb.connect(str(database))
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute("SET preserve_insertion_order = false")  # lets large aggregations spill
    con.execute("SET enable_progress_bar = false")
    # Spill to disk instead of failing when an aggregation outgrows memory,
    # including for in-memory databases.
    temp = DATA_DIR / "duckdb_tmp"
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{temp}'")

    provenance: list[tuple[str, str]] = []
    mysql_tables: set[str] = set()
    if backend == "mysql":
        settings = mysql or MySQLSettings()
        con.execute("INSTALL mysql; LOAD mysql;")
        con.execute(f"ATTACH '{settings.dsn()}' AS cinelume_mysql (TYPE mysql, READ_ONLY)")
        for table in TABLES:
            try:
                rows = con.execute(f"SELECT count(*) FROM cinelume_mysql.{table}").fetchone()[0]
            except duckdb.Error:
                rows = 0
            if rows:
                mysql_tables.add(table)
    elif backend != "parquet":
        raise ValueError(f"unknown backend {backend!r}; use 'parquet' or 'mysql'")

    for table in TABLES:
        if table in mysql_tables:
            source, origin = f"cinelume_mysql.{table}", "mysql"
        else:
            path = Path(parquet_dir) / f"{table}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found: generate the data first (make data)")
            source, origin = f"read_parquet('{path}')", "parquet"
        con.execute(f"CREATE OR REPLACE VIEW raw_{table} AS {_select(table, source)}")
        provenance.append((table, origin))

    con.execute("CREATE OR REPLACE TABLE raw_provenance (tabela VARCHAR, origem VARCHAR)")
    con.executemany("INSERT INTO raw_provenance VALUES (?, ?)", provenance)
    return con


def volume(con: duckdb.DuckDBPyConnection) -> dict[str, int | str]:
    """Row counts and date range of the ingested data, for the README and the dashboard."""
    counts = {
        table: int(con.execute(f"SELECT count(*) FROM raw_{table}").fetchone()[0])
        for table in ("contas", "assinaturas", "faturas", "pagamentos", "sessoes_app", "reproducoes", "eventos_app")
    }
    first, last = con.execute("SELECT min(criada_em), max(criada_em) FROM raw_contas").fetchone()
    counts["primeiro_cadastro"] = str(first.date())
    counts["ultimo_cadastro"] = str(last.date())
    return counts
