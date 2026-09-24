"""Hybrid residual model (Section 5, item 5): the physics model makes the
first prediction; XGBoost is trained to predict what the physics model still
gets wrong (the residual), using the same driver set. Final prediction =
physics + learned correction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from gsl_model import physics_model as pm
from gsl_model.baselines import FEATURE_COLS, add_ml_features


@dataclass
class HybridModel:
    theta: np.ndarray
    residual_model: object
    feature_cols: list

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        physics_pred = pm.predict(df, self.theta)
        d = add_ml_features(df)
        X = d[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        X_filled = X.fillna(X.median())
        correction = self.residual_model.predict(X_filled.to_numpy())
        return physics_pred + correction

    def physics_component(self, df: pd.DataFrame) -> np.ndarray:
        return pm.predict(df, self.theta)


def fit_hybrid(
    train_df: pd.DataFrame,
    theta: np.ndarray,
    val_df: pd.DataFrame | None = None,
    feature_cols: list[str] = FEATURE_COLS,
) -> HybridModel:
    d = add_ml_features(train_df)
    physics_pred_train = pm.predict(train_df, theta)
    d["physics_pred"] = physics_pred_train
    d["residual_target"] = d["target_next"] - d["physics_pred"]

    cols = feature_cols + ["residual_target"]
    sub = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
    X, y = sub[feature_cols].to_numpy(), sub["residual_target"].to_numpy()

    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.5, random_state=0, n_jobs=4,
    )
    if val_df is not None:
        dv = add_ml_features(val_df)
        physics_pred_val = pm.predict(val_df, theta)
        dv["physics_pred"] = physics_pred_val
        dv["residual_target"] = dv["target_next"] - dv["physics_pred"]
        subv = dv[cols].replace([np.inf, -np.inf], np.nan).dropna()
        Xv, yv = subv[feature_cols].to_numpy(), subv["residual_target"].to_numpy()
        model.set_params(early_stopping_rounds=50)
        model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    else:
        model.fit(X, y)

    return HybridModel(theta=theta, residual_model=model, feature_cols=feature_cols)
