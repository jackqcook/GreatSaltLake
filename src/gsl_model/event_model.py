"""Two-stage dust-event model (Section 6): stage 1 estimates the probability
that a dust event is occurring; stage 2 estimates the magnitude of the PM10
enhancement above background, conditional on an event. This is a model-free
(non-physics) cross-check: if it agrees with the physics model's lake
contribution and with the Section 7 physical-sanity checks, that's independent
evidence the physics model isn't just fitting noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gsl_model.baselines import FEATURE_COLS, add_ml_features
from gsl_model.evaluate import detect_events, EVENT_PM10_THRESHOLD

EVENT_FEATURE_COLS = [c for c in FEATURE_COLS if c not in ("pm10_lag1", "pm10_lag24")]


def label_events(df: pd.DataFrame, threshold: float = EVENT_PM10_THRESHOLD) -> np.ndarray:
    events = detect_events(df, threshold=threshold)
    label = np.zeros(len(df), dtype=int)
    for ev in events:
        label[ev.start_idx:ev.end_idx + 1] = 1
    return label


@dataclass
class EventModel:
    classifier: object
    magnitude_model: object
    feature_cols: list

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        d = add_ml_features(df)
        X = d[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        return self.classifier.predict_proba(X.to_numpy())[:, 1]

    def predict_enhancement(self, df: pd.DataFrame) -> np.ndarray:
        d = add_ml_features(df)
        X = d[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        return self.magnitude_model.predict(X.to_numpy())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Combined prediction: P_bg + Pr(event) * predicted enhancement."""
        p_event = self.predict_proba(df)
        enhancement = self.predict_enhancement(df)
        bg = df["pm10_background"].to_numpy()
        return bg + p_event * np.clip(enhancement, 0, None)


def fit_event_model(train_df: pd.DataFrame, feature_cols: list[str] = EVENT_FEATURE_COLS) -> EventModel:
    d = add_ml_features(train_df)
    d["event"] = label_events(train_df)
    d["enhancement"] = d["target_next"] - d["pm10_background"]

    cols = feature_cols + ["event"]
    sub_cls = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
    X_cls, y_cls = sub_cls[feature_cols].to_numpy(), sub_cls["event"].to_numpy()

    classifier = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=0, n_jobs=4, eval_metric="logloss",
    )
    classifier.fit(X_cls, y_cls)

    event_rows = d[d["event"] == 1]
    cols_mag = feature_cols + ["enhancement"]
    sub_mag = event_rows[cols_mag].replace([np.inf, -np.inf], np.nan).dropna()
    X_mag, y_mag = sub_mag[feature_cols].to_numpy(), sub_mag["enhancement"].to_numpy()

    magnitude_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        random_state=0, n_jobs=4,
    )
    magnitude_model.fit(X_mag, y_mag)

    return EventModel(classifier=classifier, magnitude_model=magnitude_model, feature_cols=feature_cols)
