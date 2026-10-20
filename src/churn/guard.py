"""Guard: the simulation's answer key must never reach training.

Role in the pipeline
--------------------
Cross-cutting. The Cinelume generator writes its ground truth (latent traits
per account, true causes of each cancellation) to a ``_gabarito`` folder.
Using it as a feature, a target or a tuning signal would turn the case into
a circular exercise. It is read in exactly one place, the post-hoc check in
:mod:`churn.validation`, after every number in ``reports/`` is final.

Receives: nothing. Returns: :func:`answer_key_blocked`, a context manager the
pipeline wraps around training, and :func:`assert_not_answer_key`, a path
check used at the ingestion boundary. Depends on: nothing in the package.

Three layers, because no single one is enough
---------------------------------------------
1. **Static:** ``tests/test_answer_key_guard.py`` fails if any module other
   than this one and :mod:`churn.validation` mentions the folder, or imports
   the validation module.
2. **Boundary:** :func:`churn.ingest.connect` refuses a data path inside the
   answer-key folder. DuckDB and pyarrow open files in C++, below the reach of
   Python's audit hooks, so the place that hands paths to them checks first.
3. **Runtime:** inside :func:`answer_key_blocked`, a Python audit hook raises
   on any ``open``/``listdir``/``scandir`` of a path in that folder, which
   catches pandas CSV readers, ``json.load`` and friends.
"""

from __future__ import annotations

import contextlib
import os
import sys

ANSWER_KEY_DIRNAME = "_gabarito"
_AUDITED = {"open", "os.listdir", "os.scandir", "glob.glob", "os.walk", "pathlib.Path.glob"}
_active = False
_installed = False


class AnswerKeyAccessError(PermissionError):
    """Raised when training code touches the simulation's answer key."""


def _is_answer_key(path) -> bool:
    try:
        text = os.fsdecode(path) if not isinstance(path, int) else ""
    except TypeError:
        return False
    parts = os.path.normpath(os.path.abspath(text)).split(os.sep) if text else []
    return ANSWER_KEY_DIRNAME in parts


def assert_not_answer_key(path) -> None:
    """Raise if ``path`` lies inside an answer-key folder."""
    if _is_answer_key(path):
        raise AnswerKeyAccessError(f"training code may not read the answer key: {path}")


def _hook(event: str, args: tuple) -> None:
    if _active and event in _AUDITED and args and _is_answer_key(args[0]):
        raise AnswerKeyAccessError(f"{event} on the answer key during training: {args[0]}")


@contextlib.contextmanager
def answer_key_blocked():
    """Block Python-level access to the answer key for the duration of the block.

    Audit hooks cannot be removed once added, so one hook is installed for the
    life of the process and switched on and off by a flag.
    """
    global _active, _installed
    if not _installed:
        sys.addaudithook(_hook)
        _installed = True
    previous, _active = _active, True
    try:
        yield
    finally:
        _active = previous
