"""Client dashboard: a static page for GitHub Pages, built from ``reports/``.

Role in the pipeline
--------------------
Presentation, after the pipeline (and after the post-hoc validation, whose
summary it quotes in the method block). Reads the report files, writes
``docs/dashboard/index.html`` plus a local copy of plotly.js, and computes no
number of its own: every figure on the page is a field of ``reports/``. That
is what keeps the page, the README and the reports in agreement, and
``tests/test_documentation.py`` checks it.

Receives: ``reports/metrics.json``, ``rules_by_segment.csv``,
``sensitivity.csv``, ``validation.json``; the brand mark from
``assets/brand/header.svg``. Returns: the page path. Depends on: nothing in
the package beyond :mod:`churn.config`.

Written for the client, in Portuguese and in business language: titles state
the conclusion, the method sits in a collapsible block, and the page says in
plain sight that the data is simulated.

Run with ``python -m churn.dashboard``.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd
import plotly.offline

from .config import REPORTS, ROOT

OUT = ROOT / "docs" / "dashboard"
RULES = ("limiar_0.5", "melhor_f1", "valor_esperado", "contatar_todos", "contatar_ninguem")
RULE_NAMES = {
    "limiar_0.5": "Limiar padrão 0,5",
    "melhor_f1": "Melhor F1",
    "valor_esperado": "Valor esperado por cliente",
    "contatar_todos": "Contatar todos",
    "contatar_ninguem": "Contatar ninguém",
}
ROLES = {"limiar_0.5": "d05", "melhor_f1": "f1", "valor_esperado": "ev"}
PLANS = {"basico_anuncios": "Básico", "padrao": "Padrão", "premium": "Premium"}
CHANNELS = {"organico": "Orgânico", "busca_paga": "Busca paga", "social_pago": "Social pago",
            "indicacao": "Indicação", "parceria_operadora": "Parceria operadora", "tv": "TV"}
MONTHS = ("jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez")


# ------------------------------------------------------------------ formatting
def money(value: float, signed: bool = False) -> str:
    """R$ in the Brazilian convention: thousands with '.', no decimals."""
    text = f"{abs(value):,.0f}".replace(",", ".")
    sign = "−" if value < 0 else ("+" if signed else "")
    return f"{sign}R$\u00a0{text}"  # no line break between currency and amount


def money_short(value: float) -> str:
    """Millions written out, for numbers too large to read digit by digit."""
    if abs(value) >= 1e6:
        return f"{'−' if value < 0 else ''}R$\u00a0{abs(value) / 1e6:.1f}\u00a0milhões".replace(".", ",")
    return money(value)


def pct(value: float, digits: int = 1) -> str:
    """Percentage with a decimal comma."""
    return f"{value * 100:.{digits}f}%".replace(".", ",")


def integer(value: float) -> str:
    """Count with a dot for thousands."""
    return f"{int(value):,}".replace(",", ".")


def month_label(value: str) -> str:
    """Cut-off date as the client reads it: mai/26."""
    stamp = pd.Timestamp(value)
    return f"{MONTHS[stamp.month - 1]}/{stamp.year % 100:02d}"


# ------------------------------------------------------------------ content
def _segment_table(segments: pd.DataFrame, name: str, labels) -> pd.DataFrame:
    part = segments[segments["segmento"] == name]
    table = part.pivot(index="valor", columns="regra", values="net_value")
    table["gap_f1"] = table["valor_esperado"] - table["melhor_f1"]
    table["gap_05"] = table["valor_esperado"] - table["limiar_0.5"]
    table.index = [labels(v) for v in table.index]
    return table


def _segment_sentence(table: pd.DataFrame, noun: str) -> str:
    """Where the rule gains most over F1, where it ties, where it loses: from the numbers."""
    best = table["gap_f1"].idxmax()
    losing = table.index[table["gap_f1"] < 0].tolist()
    close = table.index[(table["gap_f1"] >= 0) & (table["gap_f1"] < 0.05 * table["melhor_f1"].abs())].tolist()
    parts = [f"ganha mais em {best} ({money(table.loc[best, 'gap_f1'], True)} sobre o F1)"]
    if close:
        parts.append(f"quase empata em {', '.join(close)}")
    if losing:
        parts.append(f"perde por pouco em {' e '.join(losing)} "
                     f"({', '.join(money(table.loc[x, 'gap_f1'], True) for x in losing)})")
    else:
        parts.append(f"vence o F1 em todos os {noun}")
    return "; ".join(parts)


def _bar_chart(table: pd.DataFrame, chart_id: str) -> dict:
    traces = []
    for rule in ("valor_esperado", "melhor_f1", "limiar_0.5"):
        traces.append({
            "type": "bar", "role": ROLES[rule], "name": RULE_NAMES[rule], "x": list(table.index),
            "y": [round(v, 2) for v in table[rule]],
            "hovertemplate": "%{x}<br>" + RULE_NAMES[rule] + ": R$ %{y:,.0f}<extra></extra>",
        })
    return {"id": chart_id, "traces": traces,
            "layout": {"barmode": "group", "yaxis": {"tickprefix": "R$ ", "separatethousands": True}}}


def _sensitivity_chart(frame: pd.DataFrame, chart_id: str, axis: str, fixed: dict, label: str, fmt) -> dict:
    part = frame
    for column, value in fixed.items():
        part = part[part[column] == value]
    traces = []
    for rule in ("valor_esperado", "melhor_f1", "limiar_0.5"):
        line = part[part["regra"] == rule].sort_values(axis)
        traces.append({
            "type": "scatter", "mode": "lines+markers", "role": ROLES[rule], "name": RULE_NAMES[rule],
            "x": [fmt(v) for v in line[axis]], "y": [round(v, 2) for v in line["net_value"]],
            "hovertemplate": label + " %{x}<br>" + RULE_NAMES[rule] + ": R$ %{y:,.0f}<extra></extra>",
        })
    return {"id": chart_id, "traces": traces, "zero_line": True,
            "layout": {"xaxis": {"title": {"text": label}, "type": "category"},
                       "yaxis": {"tickprefix": "R$ ", "separatethousands": True}}}


def _mark() -> str:
    """The ABADE 'A' from the brand header, so the page and the README share one symbol."""
    svg = (ROOT / "assets" / "brand" / "header.svg").read_text()
    path = re.search(r'<path transform="translate\(80\.0 118\.0\) scale\(1\.3200\)" d="([^"]+)"', svg).group(1)
    return (f'<svg class="mark" viewBox="0 0 128 100" aria-hidden="true">'
            f'<path d="{path}" fill="currentColor" fill-rule="evenodd"/></svg>')


def build() -> Path:
    """Write docs/dashboard/index.html (and plotly.js beside it) from reports/.

    Every number and every verdict in a title is read or derived from the
    report files here, so a re-run that changes the result changes the page.
    """
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    segments = pd.read_csv(REPORTS / "rules_by_segment.csv")
    sens = pd.read_csv(REPORTS / "sensitivity.csv")
    validation_file = REPORTS / "validation.json"
    validation = json.loads(validation_file.read_text()) if validation_file.exists() else None

    s = metrics["strategies"]
    ev, f1, d05 = s["valor_esperado"], s["melhor_f1"], s["limiar_0.5"]
    everyone = s["contatar_todos"]
    gap_f1 = metrics["bootstrap"]["valor_esperado_vs_melhor_f1"]
    gap_05 = metrics["bootstrap"]["valor_esperado_vs_limiar_0.5"]
    left = ev["net_value"] - d05["net_value"]
    model = metrics["models"]["lightgbm"]["test"]
    test = metrics["split"]["test"]
    econ = metrics["economics"]
    be = metrics["break_even_probability"]

    decisions = pd.read_csv(REPORTS / "decisions_last_month.csv")
    plan_bar = decisions.groupby("plano")["break_even_probability"].median()
    months = _segment_table(segments, "cutoff", month_label)
    plans = _segment_table(segments, "plano", lambda v: PLANS.get(v, v))
    channels = _segment_table(segments, "canal", lambda v: CHANNELS.get(v, v))

    base = sens[(sens["multiplicador_oferta"] == 1.0) & (sens["peso_morto"] == econ["non_churner_acceptance"])]
    by_acceptance = base.pivot(index="taxa_aceite", columns="regra", values="net_value")
    f1_negative = by_acceptance.index[by_acceptance["melhor_f1"] < 0]
    default_ahead = by_acceptance[by_acceptance["limiar_0.5"] > by_acceptance["valor_esperado"]]
    offer = sens[(sens["taxa_aceite"] == econ["acceptance_rate"]) & (sens["peso_morto"] == econ["non_churner_acceptance"])]
    by_offer = offer.pivot(index="multiplicador_oferta", columns="regra", values="net_value")
    grid = sens[sens["peso_morto"] == econ["non_churner_acceptance"]].pivot_table(
        index=["taxa_aceite", "multiplicador_oferta"], columns="regra", values="net_value")
    alternatives = grid[["melhor_f1", "limiar_0.5", "contatar_todos"]].max(axis=1)
    ev_wins = int((grid["valor_esperado"] >= alternatives).sum())

    charts = [
        {"id": "regras", "traces": [{
            "type": "bar", "roles": ["d05", "f1", "ev", "neutral"],
            "x": [RULE_NAMES[r] for r in ("limiar_0.5", "melhor_f1", "valor_esperado", "contatar_ninguem")],
            "y": [round(s[r]["net_value"], 2) for r in ("limiar_0.5", "melhor_f1", "valor_esperado", "contatar_ninguem")],
            "text": [money(s[r]["net_value"]) for r in ("limiar_0.5", "melhor_f1", "valor_esperado", "contatar_ninguem")],
            "textposition": "outside", "cliponaxis": False,
            "customdata": [[integer(s[r]["contacted"]), integer(s[r]["customers_saved"])]
                           for r in ("limiar_0.5", "melhor_f1", "valor_esperado", "contatar_ninguem")],
            "hovertemplate": "%{x}<br>Valor líquido: R$ %{y:,.0f}<br>Contatados: %{customdata[0]}"
                             "<br>Salvos: %{customdata[1]}<extra></extra>",
        }], "layout": {"showlegend": False, "yaxis": {"tickprefix": "R$ ", "separatethousands": True}}},
        _bar_chart(months, "meses"),
        _bar_chart(plans, "planos"),
        _bar_chart(channels, "canais"),
        _sensitivity_chart(sens, "aceite", "taxa_aceite",
                           {"multiplicador_oferta": 1.0, "peso_morto": econ["non_churner_acceptance"]},
                           "Taxa de aceite", lambda v: pct(v, 0)),
        _sensitivity_chart(sens, "oferta", "multiplicador_oferta",
                           {"taxa_aceite": econ["acceptance_rate"], "peso_morto": econ["non_churner_acceptance"]},
                           "Custo da oferta", lambda v: f"{v:g}×".replace(".", ",")),
        _sensitivity_chart(sens, "pesomorto", "peso_morto",
                           {"taxa_aceite": econ["acceptance_rate"], "multiplicador_oferta": 1.0},
                           "Fiéis que aceitam o desconto", lambda v: pct(v, 0)),
    ]

    # ---------------------------------------------------------------- text
    month_wins = int((months["gap_f1"] > 0).sum())
    seg_title = (f"A regra vence o F1 em todos os {len(months)} meses" if month_wins == len(months)
                 else f"A regra vence o F1 em {month_wins} de {len(months)} meses")
    seg_title += f"; a vantagem é menor no {plans['gap_f1'].idxmin()}"
    channels_behind = int((channels["gap_f1"] < 0).sum())
    if channels_behind:
        seg_title += f" e o F1 fica à frente em {channels_behind} {'canal' if channels_behind == 1 else 'canais'}"
    widest = plans["gap_f1"].idxmax()
    widest_code = {v: k for k, v in PLANS.items()}[widest]
    narrowest = plans["gap_f1"].idxmin()
    narrowest_code = {v: k for k, v in PLANS.items()}[narrowest]
    plan_note = (f"Onde o limiar do plano fica perto do corte de F1 ({narrowest}, "
                 f"{pct(plan_bar[narrowest_code], 0)} contra {pct(metrics['best_f1_threshold'])}), as duas regras "
                 f"contatam quase as mesmas pessoas; onde fica longe ({widest}, {pct(plan_bar[widest_code], 0)}), "
                 f"a diferença aparece.")
    f1_loses_at = (f"com aceite de {pct(f1_negative.max(), 0)} ele perde dinheiro, porque contata quem está "
                   f"abaixo do próprio limiar" if len(f1_negative)
                   else "ele contata quem está abaixo do próprio limiar")
    headline = (f"Decidir quem recebe a oferta pelo valor de cada assinante rendeu {money(ev['net_value'])} "
                f"em seis meses: {money(left)} a mais que o limiar padrão e "
                f"{money(gap_f1['gap'])} a mais que o melhor limiar de F1.")
    sens_title = "A regra continua positiva em todas as taxas de aceite"
    if len(f1_negative):
        sens_title += f"; o limiar de F1 passa a perder dinheiro com aceite de {pct(f1_negative.max(), 0)} ou menos"
    sens_note = ""
    for rate, row in default_ahead.iterrows():
        sens_note += (f" Com aceite de {pct(rate, 0)}, o limiar 0,5 fica um pouco à frente "
                      f"({money(row['limiar_0.5'])} contra {money(row['valor_esperado'])}): "
                      f"com pouca margem, contatar só os casos mais certos vence.")

    highlight = {"valor_esperado": " class='hl'"}
    rows = "".join(
        f"<tr{highlight.get(r, '')}><th scope='row'>{RULE_NAMES[r]}</th>"
        f"<td>{integer(s[r]['contacted'])}</td><td>{integer(s[r]['customers_saved'])}</td>"
        f"<td>{money(s[r]['offer_spend'] + s[r]['contact_spend'])}</td>"
        f"<td>{money(s[r]['margin_saved'])}</td><td class='num'>{money_short(s[r]['net_value'])}</td></tr>"
        for r in RULES
    )

    def segment_rows(table):
        return "".join(
            f"<tr><th scope='row'>{html.escape(str(i))}</th><td>{money(row['limiar_0.5'])}</td>"
            f"<td>{money(row['melhor_f1'])}</td><td>{money(row['valor_esperado'])}</td>"
            f"<td class='{'neg' if row['gap_f1'] < 0 else 'pos'}'>{money(row['gap_f1'], True)}</td></tr>"
            for i, row in table.iterrows())

    validation_html = ""
    if validation:
        lat = validation["latentes"]
        causes = [c for c in validation["causas"] if c.get("auc_contra_fieis") is not None]
        cause_rows = "".join(
            f"<tr><th scope='row'>{html.escape(c['causa'])}</th><td>{integer(c['churners'])}</td>"
            f"<td>{c['auc_contra_fieis']:.2f}".replace(".", ",") + "</td>"
            f"<td>{pct(c['no_decil_de_maior_risco'], 0)}</td></tr>" for c in causes)
        exp = validation["experimento_reengajamento"]
        bots = validation["contas_especiais"]["bots"]
        validation_html = f"""
        <h4>O modelo encontrou o que a simulação plantou?</h4>
        <p>Depois de os números acima estarem fechados, as previsões foram comparadas com a verdade da
        simulação, que nunca entrou no treino. O risco previsto acompanha o engajamento latente de cada
        assinante (correlação {lat['eng_mu']['spearman']:.2f}) e a queda de uso ao longo do tempo
        ({lat['drift']['spearman']:.2f}).</p>
        <div class="table-wrap"><table><caption>Capacidade de separar quem sai de quem fica, por causa real do cancelamento</caption>
        <thead><tr><th scope="col">Causa real</th><th scope="col">Cancelamentos no teste</th>
        <th scope="col">AUC</th><th scope="col">No decil de maior risco</th></tr></thead>
        <tbody>{cause_rows}</tbody></table></div>
        <p>Pontos cegos: quem cancela por ter terminado uma série ou por problema técnico quase não se
        distingue de quem fica. {integer(bots['contas'])} contas-robô chegaram ao período de teste; a regra
        não gastou oferta com nenhuma.</p>
        <p>O e-mail de reengajamento era um experimento sorteado, e a análise inicial não percebeu. Comparando
        sorteados e não sorteados com o mesmo tempo sem assistir, o efeito no cancelamento é
        {pct(exp['efeito_pp'] / 100, 2)} (IC 95% {pct(exp['ic95_pp'][0] / 100, 2)} a
        {pct(exp['ic95_pp'][1] / 100, 2)}): nenhum. Ele não traz oferta, então não mede a taxa de aceite.</p>"""

    page = TEMPLATE.format(
        mark=_mark(),
        headline=html.escape(headline),
        kpi_value=money(ev["net_value"]), kpi_left=money(left), kpi_f1=money(gap_f1["gap"], True),
        kpi_f1_ci=f"{money(gap_f1['ci_low'])} a {money(gap_f1['ci_high'])}",
        kpi_05_ci=f"{money(gap_05['ci_low'])} a {money(gap_05['ci_high'])}",
        contacted=integer(ev["contacted"]), saved=integer(ev["customers_saved"]),
        test_rows=integer(test["rows"]), test_subs=integer(test["subscribers"]),
        d05_contacted=integer(d05["contacted"]), d05_saved=integer(d05["customers_saved"]),
        f1_contacted=integer(f1["contacted"]), f1_value=money(f1["net_value"]), d05_value=money(d05["net_value"]),
        f1_threshold=pct(metrics["best_f1_threshold"]), be_min=pct(be["min"], 0), be_max=pct(be["max"], 0),
        everyone_value=money_short(everyone["net_value"]), everyone_saved=integer(everyone["customers_saved"]),
        rows=rows,
        months_sentence=_segment_sentence(months, "meses"),
        plans_sentence=_segment_sentence(plans, "planos"),
        channels_sentence=_segment_sentence(channels, "canais"),
        months_rows=segment_rows(months), plans_rows=segment_rows(plans), channels_rows=segment_rows(channels),
        seg_title=html.escape(seg_title), plan_note=html.escape(plan_note), f1_loses_at=html.escape(f1_loses_at),
        sens_title=html.escape(sens_title), sens_note=html.escape(sens_note),
        ev_wins=ev_wins, scenarios=len(grid),
        offer_double=money(by_offer.loc[2.0, "valor_esperado"]), offer_double_f1=money(by_offer.loc[2.0, "melhor_f1"]),
        acceptance=pct(econ["acceptance_rate"], 0), deadweight=pct(econ["non_churner_acceptance"], 0),
        contact_cost=money(econ["contact_cost"]).replace("R$ ", "R$ "), margin=pct(econ["gross_margin"], 0),
        horizon=econ["expected_remaining_months"],
        auc=f"{model['roc_auc']:.3f}".replace(".", ","), predicted=pct(model["mean_predicted"], 2),
        observed=pct(model["observed_rate"], 2),
        auc_logistic=f"{metrics['models']['logistic']['test']['roc_auc']:.3f}".replace(".", ","),
        logistic_value=money(metrics["strategies_logistic"]["valor_esperado"]["net_value"]),
        accounts=integer(metrics["data"]["contas"]), plays=integer(metrics["data"]["reproducoes"]),
        events=integer(metrics["data"]["eventos_app"]), feature_rows=integer(metrics["data"]["feature_rows"]),
        churn=pct(metrics["data"]["churn_rate"], 2),
        validation=validation_html,
        charts=json.dumps(charts, ensure_ascii=False),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plotly.min.js").write_text(plotly.offline.get_plotlyjs())
    path = OUT / "index.html"
    path.write_text(page)
    (ROOT / "docs" / ".nojekyll").write_text("")
    return path


TEMPLATE = """<!doctype html>
<html lang="pt-BR" data-theme="auto">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retenção Cinelume · ABADE</title>
<meta name="description" content="Quem deve receber a oferta de retenção: a regra de decisão por valor esperado, avaliada em seis meses que o modelo nunca viu. Dados simulados.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lato:wght@400;700;900&display=swap" rel="stylesheet">
<style>
:root {{
  --carbon: #050505; --ivory: #F6F5F0; --cobalt: #4353E6;
  --bg: var(--ivory); --surface: #FFFFFF; --ink: #050505; --muted: #5C6168; --line: #E2E0D7;
  --accent: #4353E6; --pos: #1F7A55; --neg: #B3324A; --warn-bg: #FFF4D6; --warn-ink: #5A4100;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #050505; --surface: #111214; --ink: #F6F5F0; --muted: #9BA2AA; --line: #24262B;
    --accent: #5B6CFF; --pos: #3DBB8A; --neg: #E8687D; --warn-bg: #2A2208; --warn-ink: #F2D78A;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #050505; --surface: #111214; --ink: #F6F5F0; --muted: #9BA2AA; --line: #24262B;
  --accent: #5B6CFF; --pos: #3DBB8A; --neg: #E8687D; --warn-bg: #2A2208; --warn-ink: #F2D78A;
}}
* {{ box-sizing: border-box; }}
html {{ -webkit-text-size-adjust: 100%; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.55 Lato, "Helvetica Neue", Arial, sans-serif; }}
header.top {{ background: var(--carbon); color: var(--ivory); }}
.wrap {{ max-width: 1120px; margin: 0 auto; padding: 0 16px; }}
.bar {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 0; }}
.brand {{ display: flex; align-items: center; gap: 12px; font-weight: 900; letter-spacing: .14em; }}
.mark {{ width: 34px; height: 27px; color: var(--ivory); }}
.brand small {{ font-weight: 400; letter-spacing: .02em; color: #9BA2AA; }}
button.theme {{ background: transparent; color: var(--ivory); border: 1px solid #3A3E46; border-radius: 999px; padding: 6px 12px; font: inherit; font-size: 14px; cursor: pointer; }}
.hero {{ padding: 28px 0 36px; }}
.hero h1 {{ font-size: clamp(24px, 4vw, 38px); line-height: 1.2; margin: 0 0 14px; font-weight: 900; max-width: 30ch; }}
.hero p {{ color: #C9CCD1; margin: 0; max-width: 70ch; }}
.warn {{ background: var(--warn-bg); color: var(--warn-ink); border-radius: 12px; padding: 12px 16px; margin: 20px 0 0; font-size: 15px; }}
.warn strong {{ font-weight: 900; }}
main section {{ padding: 36px 0 8px; border-bottom: 1px solid var(--line); }}
.eyebrow {{ color: var(--accent); font-weight: 900; font-size: 13px; letter-spacing: .12em; text-transform: uppercase; margin: 0 0 6px; }}
h2 {{ font-size: clamp(20px, 2.6vw, 27px); line-height: 1.25; margin: 0 0 10px; font-weight: 900; max-width: 40ch; }}
h3 {{ font-size: 18px; margin: 22px 0 6px; }}
h4 {{ font-size: 17px; margin: 22px 0 6px; }}
p {{ max-width: 72ch; }}
.lead {{ color: var(--muted); margin: 0 0 18px; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 22px 0 8px; }}
.kpi {{ background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 16px; }}
.kpi .v {{ font-size: clamp(22px, 3vw, 30px); font-weight: 900; line-height: 1.1; }}
.kpi .v.accent {{ color: var(--accent); }}
.kpi .l {{ color: var(--muted); font-size: 14px; margin-top: 6px; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 12px 12px 4px; margin: 14px 0; }}
.chart {{ width: 100%; height: 360px; }}
.grid3 {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
.grid3.g12 {{ grid-template-columns: minmax(0, 1fr) minmax(0, 2fr); }}
.grid3.g11 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.grid3 .chart {{ height: 300px; }}
.grid3 h3 {{ margin: 6px 6px 0; font-size: 15px; }}
.table-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 12px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 15px; background: var(--surface); }}
caption {{ text-align: left; color: var(--muted); font-size: 14px; padding: 0 0 8px; }}
th, td {{ padding: 9px 12px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
th:first-child {{ text-align: left; }}
thead th {{ color: var(--muted); font-weight: 700; font-size: 13px; }}
tr.hl th, tr.hl td {{ font-weight: 900; color: var(--accent); }}
td.pos {{ color: var(--pos); font-weight: 700; }} td.neg {{ color: var(--neg); font-weight: 700; }}
.rec {{ background: var(--carbon); color: var(--ivory); border-radius: 20px; padding: 24px; margin: 18px 0; }}
.rec h2 {{ color: var(--ivory); }}
.rec ol {{ margin: 0; padding-left: 20px; }} .rec li {{ margin: 8px 0; max-width: 72ch; }}
details {{ background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 14px 18px; margin: 18px 0 36px; }}
summary {{ cursor: pointer; font-weight: 900; }}
details[open] summary {{ margin-bottom: 8px; }}
footer {{ color: var(--muted); font-size: 14px; padding: 24px 0 40px; }}
footer a, main a {{ color: var(--accent); }}
@media (max-width: 860px) {{
  .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .grid3, .grid3.g12, .grid3.g11 {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 420px) {{ .kpis {{ grid-template-columns: 1fr; }} .chart {{ height: 320px; }} }}
</style>
</head>
<body>
<header class="top">
  <div class="wrap">
    <div class="bar">
      <div class="brand">{mark}<span>ABADE <small>· Retenção Cinelume</small></span></div>
      <button class="theme" type="button" id="theme" aria-label="Alternar tema claro ou escuro">Tema</button>
    </div>
    <div class="hero">
      <p class="eyebrow" style="color:#8C98FF">Resumo executivo</p>
      <h1>{headline}</h1>
      <p>Mesmo modelo, mesmos {test_rows} assinantes-mês de maio a outubro de 2026, meses que o modelo nunca viu.
      Só muda a regra que decide quem recebe a oferta de retenção.</p>
      <div class="warn" role="note"><strong>Dados simulados.</strong> A Cinelume é um streaming fictício; todos os
      clientes, valores e eventos foram gerados por um simulador. Os números ilustram o método e não descrevem
      nenhuma empresa real.</div>
    </div>
  </div>
</header>
<main class="wrap">
  <section aria-labelledby="s-kpi">
    <h2 id="s-kpi">Os números que importam</h2>
    <div class="kpis">
      <div class="kpi"><div class="v accent">{kpi_value}</div><div class="l">Valor líquido da campanha com a regra por cliente, em seis meses</div></div>
      <div class="kpi"><div class="v">{kpi_left}</div><div class="l">Deixados na mesa pelo limiar padrão de 0,5 (IC 95%: {kpi_05_ci})</div></div>
      <div class="kpi"><div class="v">{kpi_f1}</div><div class="l">Sobre o melhor limiar de F1 (IC 95%: {kpi_f1_ci})</div></div>
      <div class="kpi"><div class="v">{contacted}</div><div class="l">Contatos para salvar {saved} assinantes</div></div>
    </div>
  </section>

  <section aria-labelledby="s-mesa">
    <p class="eyebrow">O limiar padrão</p>
    <h2 id="s-mesa">O limiar de 0,5 contata só {d05_contacted} assinantes e deixa {kpi_left} na mesa</h2>
    <p class="lead">O corte de 0,5 é o que sai de um modelo sem nenhuma informação de negócio. Ele só oferece
    desconto a quem tem mais de 50% de chance de sair; mas, com a economia da Cinelume, a oferta já se paga a
    partir de {be_min} a {be_max} de risco, conforme o plano. O limiar de F1 ({f1_threshold}) erra para o outro
    lado: contata {f1_contacted} pessoas, várias abaixo da conta que fecha.</p>
    <div class="card"><div class="chart" id="regras" role="img" aria-label="Valor líquido por regra de decisão"></div></div>
  </section>

  <section aria-labelledby="s-regras">
    <p class="eyebrow">Comparação das regras</p>
    <h2 id="s-regras">Contatar todos salva mais gente e destrói valor: {everyone_saved} salvos por {everyone_value}</h2>
    <p class="lead">O objetivo não é salvar o maior número de assinantes; é gastar o orçamento onde a oferta se
    paga. Quem não ia sair também aceita o desconto, e esse custo cai sobre cada contato desnecessário.</p>
    <div class="table-wrap"><table>
      <caption>Seis campanhas mensais (mai–out/2026), mesma aceitação simulada para todas as regras</caption>
      <thead><tr><th scope="col">Regra</th><th scope="col">Contatados</th><th scope="col">Salvos</th>
      <th scope="col">Custo (oferta + contato)</th><th scope="col">Margem salva</th><th scope="col">Valor líquido</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
  </section>

  <section aria-labelledby="s-seg">
    <p class="eyebrow">Por mês, plano e canal</p>
    <h2 id="s-seg">{seg_title}</h2>
    <p class="lead">Por mês, a regra {months_sentence}. Por plano, {plans_sentence}. Por canal, {channels_sentence}.
    {plan_note}</p>
    <div class="card"><h3>Por mês de teste</h3><div class="chart" id="meses" role="img" aria-label="Valor líquido por mês e regra"></div></div>
    <div class="grid3 g12">
      <div class="card"><h3>Por plano</h3><div class="chart" id="planos" role="img" aria-label="Valor líquido por plano e regra"></div></div>
      <div class="card"><h3>Por canal de aquisição</h3><div class="chart" id="canais" role="img" aria-label="Valor líquido por canal e regra"></div></div>
    </div>
    <details><summary>Tabelas por segmento</summary>
      <div class="table-wrap"><table><caption>Por mês</caption><thead><tr><th scope="col">Mês</th><th scope="col">Limiar 0,5</th><th scope="col">Melhor F1</th><th scope="col">Regra por cliente</th><th scope="col">Regra − F1</th></tr></thead><tbody>{months_rows}</tbody></table></div>
      <div class="table-wrap"><table><caption>Por plano</caption><thead><tr><th scope="col">Plano</th><th scope="col">Limiar 0,5</th><th scope="col">Melhor F1</th><th scope="col">Regra por cliente</th><th scope="col">Regra − F1</th></tr></thead><tbody>{plans_rows}</tbody></table></div>
      <div class="table-wrap"><table><caption>Por canal</caption><thead><tr><th scope="col">Canal</th><th scope="col">Limiar 0,5</th><th scope="col">Melhor F1</th><th scope="col">Regra por cliente</th><th scope="col">Regra − F1</th></tr></thead><tbody>{channels_rows}</tbody></table></div>
    </details>
  </section>

  <section aria-labelledby="s-sens">
    <p class="eyebrow">Sensibilidade</p>
    <h2 id="s-sens">{sens_title}</h2>
    <p class="lead">Taxa de aceite e custo da oferta são premissas, não medições. A regra por cliente recalcula
    seus limiares quando elas mudam; os cortes fixos, não. Em {ev_wins} de {scenarios} cenários de aceite × custo
    da oferta ela é a melhor regra.{sens_note} Com a oferta custando o dobro, a regra ainda rende {offer_double};
    o F1, {offer_double_f1}.</p>
    <div class="card"><h3>Taxa de aceite entre quem ia sair</h3><div class="chart" id="aceite" role="img" aria-label="Valor líquido por taxa de aceite"></div></div>
    <div class="grid3 g11">
      <div class="card"><h3>Custo da oferta (1× = VOLTA50)</h3><div class="chart" id="oferta" role="img" aria-label="Valor líquido por custo da oferta"></div></div>
      <div class="card"><h3>Fiéis que aceitam o desconto</h3><div class="chart" id="pesomorto" role="img" aria-label="Valor líquido por peso morto"></div></div>
    </div>
  </section>

  <section class="rec" aria-labelledby="s-rec">
    <p class="eyebrow" style="color:#8C98FF">Recomendação</p>
    <h2 id="s-rec">Trocar o corte fixo pela regra por cliente e medir a taxa de aceite antes de escalar</h2>
    <ol>
      <li><strong>Adotar a regra de valor esperado</strong> na lista mensal de retenção: oferta só para quem tem
      risco acima do limiar do próprio plano ({be_min} a {be_max}). Com o mesmo modelo, rendeu {kpi_value} contra
      {d05_value} do limiar padrão.</li>
      <li><strong>Rodar um teste com grupo de controle</strong> antes de escalar: a taxa de aceite ({acceptance})
      e a fração de fiéis que aceitam o desconto ({deadweight}) são as premissas que mais mexem no resultado, e
      nenhum dado atual as mede.</li>
      <li><strong>Atualizar mensalmente</strong> a calibração do modelo com o último mês fechado: um erro de unidade
      no app 6.0 (tempo assistido gravado em milissegundos, não em segundos) bastou para derrubar as probabilidades pela metade.</li>
      <li><strong>Não usar o limiar de F1 como atalho</strong>: {f1_loses_at}.</li>
    </ol>
  </section>

  <details>
    <summary>Método, premissas e limites</summary>
    <h4>Dados</h4>
    <p>{accounts} contas simuladas, 37 meses (out/2023 a out/2026), {plays} reproduções e {events} eventos de
    aplicativo. Cada linha do modelo é um assinante ativo no primeiro dia de um mês ({feature_rows} linhas);
    o alvo é pedir o cancelamento naquele mês (taxa média de {churn}). Quem já pediu cancelamento e contas internas
    ficam de fora.</p>
    <h4>Validação no tempo</h4>
    <p>Treino de jan/2024 a dez/2025; calibração e escolha do limiar de F1 de jan a abr/2026; teste de mai a
    out/2026. Cada atributo usa só o que a empresa sabia antes da data de corte, e um teste automático reconstrói
    os dados como estavam naquela data para provar isso.</p>
    <h4>Modelo</h4>
    <p>LightGBM com calibração isotônica: AUC de {auc} no teste, risco médio previsto de {predicted} contra
    {observed} observado. Regressão logística de referência: AUC {auc_logistic}; com a mesma regra, renderia
    {logistic_value}.</p>
    <h4>Economia</h4>
    <p>Margem em risco = receita mensal (preço do plano em vigor, mais membro extra) × margem de {margin} × {horizon}
    meses. A oferta é o cupom de reconquista da própria Cinelume (50% por 2 ciclos), custando cerca de uma
    mensalidade. Premissas declaradas: aceite de {acceptance} entre quem ia sair, {deadweight} entre os fiéis,
    contato de {contact_cost}. A aceitação é sorteada uma vez e vale para todas as regras.</p>
    {validation}
    <h4>Limites</h4>
    <p>Não há experimento de retenção com oferta nos dados: a regra prevê quem sai, não quem muda de ideia por
    causa da oferta. Cada mês é tratado como campanha independente, sem período de carência entre contatos.
    Os dados são simulados.</p>
  </details>
</main>
<footer class="wrap">
  <p>ABADE · Caso de retenção com dados simulados da Cinelume. Código, dados de origem e método completo em
  <a href="https://github.com/arielabade/churn-cost-sensitive">github.com/arielabade/churn-cost-sensitive</a>.</p>
</footer>
<script src="plotly.min.js"></script>
<script>
const CHARTS = {charts};
const THEMES = {{
  light: {{ ink: "#050505", muted: "#5C6168", grid: "#E2E0D7", zero: "#9AA0A8",
           ev: "#4353E6", f1: "#B07A12", d05: "#1F8F6A", neutral: "#B9BDC3", hover: "#FFFFFF" }},
  dark:  {{ ink: "#F6F5F0", muted: "#9BA2AA", grid: "#24262B", zero: "#5C6168",
           ev: "#5B6CFF", f1: "#BD8620", d05: "#1F9B73", neutral: "#3A3E46", hover: "#111214" }}
}};
function currentTheme() {{
  const attr = document.documentElement.dataset.theme;
  if (attr === "light" || attr === "dark") return attr;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}}
function render() {{
  const t = THEMES[currentTheme()];
  const narrow = window.innerWidth < 640;
  for (const spec of CHARTS) {{
    const el = document.getElementById(spec.id);
    if (!el) continue;
    const data = spec.traces.map(tr => {{
      const c = JSON.parse(JSON.stringify(tr));
      const color = c.roles ? c.roles.map(r => t[r]) : t[c.role];
      delete c.role; delete c.roles;
      if (c.type === "bar") {{ c.marker = {{ color: color, line: {{ width: 0 }} }}; c.textfont = {{ color: t.ink, family: "Lato" }}; }}
      else {{ c.line = {{ color: color, width: 2 }}; c.marker = {{ color: color, size: 8, line: {{ color: t.hover, width: 2 }} }}; }}
      return c;
    }});
    const base = spec.layout || {{}};
    const layout = Object.assign({{}}, base, {{
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: {{ family: "Lato, Arial, sans-serif", color: t.muted, size: 13 }},
      margin: {{ l: 70, r: 12, t: data.length > 1 ? (el.clientWidth < 560 ? 92 : 56) : 24, b: narrow ? 70 : 48 }},
      bargap: 0.28, bargroupgap: 0.06, barcornerradius: 4,
      hovermode: "closest", hoverlabel: {{ bgcolor: t.hover, bordercolor: t.grid, font: {{ color: t.ink, family: "Lato" }} }},
      legend: {{ orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: {{ color: t.ink }} }},
      xaxis: Object.assign({{ gridcolor: t.grid, linecolor: t.grid, tickfont: {{ color: t.muted }}, automargin: true }}, base.xaxis || {{}}),
      yaxis: Object.assign({{ gridcolor: t.grid, zerolinecolor: t.zero, zerolinewidth: 1, tickfont: {{ color: t.muted }}, automargin: true }}, base.yaxis || {{}})
    }});
    if (layout.xaxis.title) layout.xaxis.title = Object.assign({{}}, layout.xaxis.title, {{ font: {{ color: t.muted }} }});
    Plotly.react(el, data, layout, {{ responsive: true, displayModeBar: false, locale: "pt-BR" }});
  }}
}}
document.getElementById("theme").addEventListener("click", () => {{
  document.documentElement.dataset.theme = currentTheme() === "dark" ? "light" : "dark";
  try {{ localStorage.setItem("abade-theme", document.documentElement.dataset.theme); }} catch (e) {{}}
  render();
}});
try {{ const saved = localStorage.getItem("abade-theme"); if (saved) document.documentElement.dataset.theme = saved; }} catch (e) {{}}
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
let resizeTimer; window.addEventListener("resize", () => {{ clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 150); }});
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print(build().relative_to(ROOT))
