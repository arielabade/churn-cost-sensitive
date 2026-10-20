"""The documents say what the code and the reports say.

* No money figure in the README or the dashboard that reports/ cannot produce.
* Every module states its role, inputs, outputs and dependencies.
* docs/ARCHITECTURE.md names exactly the modules that exist.
"""

import ast
import json
import re
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PACKAGE = ROOT / "src" / "churn"


def _numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, (int, float)):
        yield float(value)


@pytest.fixture(scope="module")
def allowed_amounts() -> set[int]:
    """Every whole-real amount the reports can justify, including the gaps between rules."""
    values = list(_numbers(json.loads((REPORTS / "metrics.json").read_text())))
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    for table in metrics["strategies"].values():
        values.append(table["offer_spend"] + table["contact_spend"])
    s = metrics["strategies"]
    values += [a["net_value"] - b["net_value"] for a, b in combinations(s.values(), 2)]

    segments = pd.read_csv(REPORTS / "rules_by_segment.csv")
    values += segments["net_value"].tolist()
    for _, group in segments.groupby(["segmento", "valor"]):
        nets = group["net_value"].tolist()
        values += [a - b for a, b in combinations(nets, 2)] + [b - a for a, b in combinations(nets, 2)]
    sensitivity = pd.read_csv(REPORTS / "sensitivity.csv")
    values += sensitivity["net_value"].tolist()
    return {int(round(abs(v))) for v in values}


def _amounts(text: str, thousands: str) -> list[int]:
    pattern = r"R\$(?:\s|&nbsp;| )?([0-9]{1,3}(?:" + re.escape(thousands) + r"[0-9]{3})+|[0-9]+)(?![0-9.,]*k)"
    return [int(match.replace(thousands, "")) for match in re.findall(pattern, text)]


def _close(amount: int, allowed: set[int]) -> bool:
    return any(abs(amount - a) <= 1 for a in allowed)


def test_every_amount_in_the_readme_comes_from_the_reports(allowed_amounts):
    text = (ROOT / "README.md").read_text()
    amounts = _amounts(text, ",")
    assert len(amounts) > 20
    stray = [a for a in amounts if a > 10 and not _close(a, allowed_amounts)]
    assert not stray, f"README quotes amounts no report produces: {stray}"


def test_every_amount_in_the_dashboard_comes_from_the_reports(allowed_amounts):
    page = ROOT / "docs" / "dashboard" / "index.html"
    text = page.read_text()
    text = text[: text.index("<script")]  # the chart data is the reports themselves
    amounts = _amounts(text, ".")
    assert len(amounts) > 20
    stray = [a for a in amounts if a > 10 and not _close(a, allowed_amounts)]
    assert not stray, f"dashboard quotes amounts no report produces: {stray}"


def test_readme_and_dashboard_quote_the_same_headline():
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    s = metrics["strategies"]
    left = round(s["valor_esperado"]["net_value"] - s["limiar_0.5"]["net_value"])
    gap = round(metrics["bootstrap"]["valor_esperado_vs_melhor_f1"]["gap"])
    readme = (ROOT / "README.md").read_text()
    page = (ROOT / "docs" / "dashboard" / "index.html").read_text()
    for amount in (left, gap, round(s["valor_esperado"]["net_value"])):
        assert f"R${amount:,}" in readme
        assert f"R$ {amount:,}".replace(",", ".") in page
    auc = metrics["models"]["lightgbm"]["test"]["roc_auc"]
    assert f"{auc:.3f}" in readme and f"{auc:.3f}".replace(".", ",") in page


def test_simulated_data_is_stated_on_the_dashboard():
    page = (ROOT / "docs" / "dashboard" / "index.html").read_text()
    assert "Dados simulados" in page and 'role="note"' in page


MODULES = sorted(p for p in PACKAGE.glob("*.py"))


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_every_module_states_its_place_in_the_architecture(path):
    doc = ast.get_docstring(ast.parse(path.read_text())) or ""
    for part in ("Role in the pipeline", "Receives", "Returns", "Depends on"):
        assert part in doc, f"{path.name} docstring lacks '{part}'"


def test_architecture_document_matches_the_code():
    text = (ROOT / "docs" / "ARCHITECTURE.md").read_text()
    table = text[text.index("## Module responsibilities"):text.index("## One subscriber")]
    documented = set(re.findall(r"^\| `([a-z_]+)` \|", table, flags=re.M))
    existing = {p.stem for p in MODULES} - {"__init__", "brandviz"}
    assert documented == existing, f"documented {documented ^ existing} differs from the code"
    assert "brandviz" in text
