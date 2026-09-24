"""Block-bootstrap uncertainty for the physics model's fitted parameters
(Section 7: "Use uncertainty intervals or bootstrap estimates for fitted
physical parameters. A point estimate of U_c or beta without uncertainty can
look more precise than the data justify.").

We resample contiguous BLOCKS of hours (not individual hours) with
replacement, because hourly PM10/weather is strongly autocorrelated; an
ordinary i.i.d. bootstrap would understate uncertainty by pretending adjacent
hours are independent draws.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gsl_model import physics_model as pm

DEFAULT_BLOCK_HOURS = 24 * 14  # 2-week blocks


@dataclass
class BootstrapResult:
    param_names: list
    samples: np.ndarray  # (n_reps, n_params)

    def summary(self) -> pd.DataFrame:
        rows = []
        for i, name in enumerate(self.param_names):
            col = self.samples[:, i]
            rows.append({
                "param": name,
                "median": np.median(col),
                "p05": np.percentile(col, 5),
                "p95": np.percentile(col, 95),
                "std": np.std(col),
            })
        return pd.DataFrame(rows)


def block_bootstrap_physics_params(
    df_train: pd.DataFrame,
    n_reps: int = 15,
    block_hours: int = DEFAULT_BLOCK_HOURS,
    event_weight_power: float = 0.25,
    x0: np.ndarray | None = None,
    seed: int = 0,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    n = len(df_train)
    n_blocks_needed = int(np.ceil(n / block_hours))
    block_starts_pool = np.arange(0, max(1, n - block_hours))

    samples = []
    for rep in range(n_reps):
        chosen_starts = rng.choice(block_starts_pool, size=n_blocks_needed, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block_hours, n)) for s in chosen_starts])
        idx = idx[idx < n]
        resampled = df_train.iloc[idx].reset_index(drop=True)
        try:
            fitted = pm.fit(resampled, x0=x0, event_weight_power=event_weight_power)
            samples.append(fitted.theta)
        except Exception:
            continue

    return BootstrapResult(param_names=pm.PARAM_NAMES, samples=np.array(samples))
