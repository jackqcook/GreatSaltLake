"""Baselines on the model ladder (Section 5): persistence, a meteorology-only
additive statistical model, and a pure ML (XGBoost) model. All predict the
same target as the physics model -- P[t+1], one-step-ahead -- from the same
driver set, so they're directly comparable on the same test rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb

FEATURE_COLS = [
    "pm10_lag1", "pm10_lag24", "pm10_background",
    "wind_speed_ms", "wind_dir_sin", "wind_dir_cos",
    "rh_pct", "temp_c", "precip_mm", "precip_48h_mm",
    "mixing_height_m", "exposed_playa_km2", "lake_elev_ft_navd88",
    "local_hour", "month", "is_weekend",
]


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    rad = np.radians(d["wind_dir_deg"])
    d["wind_dir_sin"] = np.sin(rad)
    d["wind_dir_cos"] = np.cos(rad)
    d["is_weekend"] = d["is_weekend"].astype(float)
    d["target_next"] = d["pm10_ugm3"].shift(-1)
    return d


def _clean_xy(df: pd.DataFrame, feature_cols: list[str]):
    cols = feature_cols + ["target_next"]
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    return sub[feature_cols].to_numpy(), sub["target_next"].to_numpy()


def persistence_predict(df: pd.DataFrame) -> np.ndarray:
    return df["pm10_ugm3"].to_numpy()


@dataclass
class FittedBaseline:
    name: str
    model: object
    feature_cols: list

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        d = add_ml_features(df)
        X = d[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        X_filled = X.fillna(X.median())
        return self.model.predict(X_filled.to_numpy())


def fit_gam_baseline(train_df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS) -> FittedBaseline:
    """Additive spline model: SplineTransformer per continuous feature + Ridge
    regression on the concatenated basis == a standard, interpretable
    generalized-additive-model approximation (Section 5, item 2)."""
    d = add_ml_features(train_df)
    X, y = _clean_xy(d, feature_cols)

    spline_cols = list(range(len(feature_cols)))
    pre = ColumnTransformer([
        ("spline", SplineTransformer(n_knots=6, degree=3, include_bias=False), spline_cols),
    ])
    model = Pipeline([
        ("scale", StandardScaler()),
        ("features", pre),
        ("ridge", Ridge(alpha=1.0)),
    ])
    model.fit(X, y)
    return FittedBaseline(name="meteorology_gam", model=model, feature_cols=feature_cols)


def fit_xgboost_baseline(
    train_df: pd.DataFrame, val_df: pd.DataFrame | None = None,
    feature_cols: list[str] = FEATURE_COLS,
) -> FittedBaseline:
    d = add_ml_features(train_df)
    X, y = _clean_xy(d, feature_cols)

    model = xgb.XGBRegressor(
        n_estimators=600, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, random_state=0, n_jobs=4,
    )
    eval_set = None
    if val_df is not None:
        dv = add_ml_features(val_df)
        Xv, yv = _clean_xy(dv, feature_cols)
        eval_set = [(Xv, yv)]
        model.set_params(early_stopping_rounds=50)
        model.fit(X, y, eval_set=eval_set, verbose=False)
    else:
        model.fit(X, y)
    return FittedBaseline(name="xgboost", model=model, feature_cols=feature_cols)
