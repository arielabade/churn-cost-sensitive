"""Model: a calibrated churn probability, and a plain baseline to beat.

Role in the pipeline
--------------------
Fourth stage. Takes the ``features`` frame, splits it by time, fits LightGBM
on the training months, calibrates it on the validation months and scores the
test months. A logistic regression goes through the same steps as the
reference a reviewer will ask for.

Receives: the frame from :func:`churn.features.load_frame`.
Returns: :class:`Split` (the three time slices) and :class:`ChurnModel`
objects whose ``predict`` gives calibrated probabilities.
Depends on: :mod:`churn.config` (calendar, seed), :mod:`churn.features`
(feature lists).

Why each choice
---------------
* **Temporal split, not random.** The same subscriber appears in many months;
  a random split puts their June in training and their July in test and
  rewards memorising people. Training months come strictly before validation
  months, which come strictly before test months.
* **Isotonic calibration on the validation months.** The probability is
  multiplied by money downstream, so a score that ranks well but is
  mis-scaled moves every break-even decision. AUC cannot see that. Fitting the
  calibrator on later, unseen months (rather than on the training rows the
  trees already memorised) also absorbs some of the drift between periods.
* **No class weights, no resampling.** Both distort the probability that the
  decision rule needs to be honest; imbalance is a ranking problem LightGBM
  handles, and the calibrator fixes the scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_SEED, TRAIN_UNTIL, VALID_UNTIL
from .features import CATEGORICAL, FEATURES, LABEL, NUMERIC


@dataclass
class Split:
    """Three consecutive time slices of the feature frame."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> dict:
        return {
            name: {
                "rows": int(len(part)),
                "subscribers": int(part["conta_id"].nunique()),
                "first_cutoff": str(part["cutoff"].min().date()),
                "last_cutoff": str(part["cutoff"].max().date()),
                "churn_rate": float(part[LABEL].mean()),
            }
            for name, part in (("train", self.train), ("valid", self.valid), ("test", self.test))
        }


def temporal_split(frame: pd.DataFrame, train_until=TRAIN_UNTIL, valid_until=VALID_UNTIL) -> Split:
    """Split by cut-off date: past for training, next months for validation, last months for test."""
    cutoff = frame["cutoff"].dt.date
    return Split(
        train=frame[cutoff <= train_until].reset_index(drop=True),
        valid=frame[(cutoff > train_until) & (cutoff <= valid_until)].reset_index(drop=True),
        test=frame[cutoff > valid_until].reset_index(drop=True),
    )


class ChurnModel:
    """A fitted scorer plus an isotonic map from its score to a probability."""

    name = "model"

    def _fit_scorer(self, x: pd.DataFrame, y: np.ndarray) -> None:
        raise NotImplementedError

    def _score(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> "ChurnModel":
        self._fit_scorer(train[list(FEATURES)], train[LABEL].to_numpy())
        # out_of_bounds="clip": a test score outside the validation range gets
        # the nearest calibrated value instead of NaN.
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.calibrator.fit(self._score(valid[list(FEATURES)]), valid[LABEL].to_numpy())
        return self

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        return self._score(frame[list(FEATURES)])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated churn probability for the month after each row's cut-off."""
        return self.calibrator.predict(self.predict_raw(frame))


class LightGBMModel(ChurnModel):
    name = "lightgbm"

    def _fit_scorer(self, x, y):
        self.estimator = LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=200,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_SEED, n_jobs=4, deterministic=True, force_row_wise=True, verbose=-1,
        ).fit(x, y)

    def _score(self, x):
        return self.estimator.predict_proba(x)[:, 1]

    def importance(self) -> pd.Series:
        """Total split gain per feature, normalised to sum to one."""
        gain = self.estimator.booster_.feature_importance(importance_type="gain")
        return pd.Series(gain / gain.sum(), index=list(FEATURES)).sort_values(ascending=False)


class LogisticModel(ChurnModel):
    """Reference model: median-imputed, standardised, one-hot, L2 logistic.

    The missing-indicator columns let it tell "never surveyed" from an average
    NPS score, the distinction LightGBM gets for free from its NULL branch.
    """

    name = "logistic"

    def _fit_scorer(self, x, y):
        numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
        categorical = make_pipeline(
            SimpleImputer(strategy="constant", fill_value="ausente"),
            OneHotEncoder(handle_unknown="ignore", min_frequency=50),
        )
        # Heavy-tailed, non-negative counts and hours get log1p so one binge
        # does not dominate the scale. Which columns is decided once, on the
        # training rows, and reused: deciding per call would transform test
        # rows differently from the rows the coefficients were fitted on.
        self.log_columns = [c for c in NUMERIC if x[c].min() >= 0 and x[c].quantile(0.99) > 20]
        self.estimator = make_pipeline(
            ColumnTransformer([("num", numeric, list(NUMERIC)), ("cat", categorical, list(CATEGORICAL))]),
            LogisticRegression(C=1.0, max_iter=3000),
        ).fit(self._prepare(x), y)

    def _prepare(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        for column in CATEGORICAL:
            x[column] = x[column].astype(object)
        for column in self.log_columns:
            x[column] = np.log1p(x[column].clip(lower=0))
        return x

    def _score(self, x):
        return self.estimator.predict_proba(self._prepare(x))[:, 1]


def scores(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Ranking (AUC, PR-AUC) and calibration (Brier, log loss) on one slice."""
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6))),
        "mean_predicted": float(np.mean(probability)),
        "observed_rate": float(np.mean(y)),
    }
