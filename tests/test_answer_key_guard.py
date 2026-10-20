"""The simulation's answer key (``_gabarito``) never reaches training code.

It may only be read by ``churn.validation``, the post-hoc check that runs
after the reports are final. Three layers are tested: what the code says,
what the ingestion boundary accepts, and what the process opens at run time.
"""

import ast
from pathlib import Path

import pytest

from churn.guard import ANSWER_KEY_DIRNAME, AnswerKeyAccessError, answer_key_blocked
from churn.ingest import connect

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "churn"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
#: The guard has to name the folder it guards; the validation module is the
#: one sanctioned reader. Every other module is training code.
ALLOWED = {"guard.py", "validation.py"}


def _training_modules():
    modules = [p for p in PACKAGE.glob("*.py") if p.name not in ALLOWED]
    modules += [p for p in SCRIPTS.glob("*.py")]
    assert len(modules) >= 8, "the scan found suspiciously few modules"
    return modules


@pytest.mark.parametrize("path", _training_modules(), ids=lambda p: p.name)
def test_training_code_never_names_the_answer_key(path):
    text = path.read_text(encoding="utf-8").lower()
    assert "gabarito" not in text, f"{path.name} mentions the answer key"


@pytest.mark.parametrize("path", _training_modules(), ids=lambda p: p.name)
def test_training_code_never_imports_the_validation_module(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [alias.name for alias in node.names]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        assert not any(name.split(".")[-1] == "validation" for name in names), (
            f"{path.name} imports churn.validation, which reads the answer key"
        )


def test_ingestion_refuses_a_data_path_inside_the_answer_key(tmp_path):
    folder = tmp_path / ANSWER_KEY_DIRNAME
    folder.mkdir()
    with pytest.raises(AnswerKeyAccessError):
        connect("parquet", folder)


def test_runtime_guard_blocks_python_level_reads(tmp_path):
    secret = tmp_path / ANSWER_KEY_DIRNAME / "latentes.csv"
    secret.parent.mkdir()
    secret.write_text("conta_id,sens_preco\n1,0.9\n")
    import pandas as pd

    with answer_key_blocked():
        with pytest.raises(AnswerKeyAccessError):
            open(secret).read()
        with pytest.raises(AnswerKeyAccessError):
            pd.read_csv(secret)
        with pytest.raises(AnswerKeyAccessError):
            list(secret.parent.iterdir())
    # Outside the block the post-hoc check can read it.
    assert open(secret).read().startswith("conta_id")


def test_the_whole_training_path_runs_under_the_guard(fixture_dir):
    """Ingest, staging, features and both models, end to end, with the guard on."""
    import datetime as dt

    from conftest import build_frame

    from churn.model import LightGBMModel, LogisticModel, temporal_split

    with answer_key_blocked():
        frame = build_frame(fixture_dir)
        split = temporal_split(frame, dt.date(2024, 9, 1), dt.date(2024, 12, 1))
        for model in (LightGBMModel(), LogisticModel()):
            assert model.fit(split.train, split.valid).predict(split.test).shape == (len(split.test),)
