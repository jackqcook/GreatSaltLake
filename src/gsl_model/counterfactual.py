"""Lake-level counterfactual scenarios (Section 8: "Run lake-level
counterfactuals. Hold the observed weather fixed and vary the
lake-elevation/exposed-area input.").

For each scenario we hold the ENTIRE observed weather record fixed (same wind,
precipitation, mixing height, timing of every storm) and substitute a
different, constant lake elevation -> a different constant exposed-playa area
-> a different A_eff(t) feeding the fitted E_GSL term. This isolates what the
model attributes specifically to lake state, independent of any particular
year's weather.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gsl_model import physics_model as pm
from gsl_model.acquisition.usgs_lake import exposed_playa_area
from gsl_model.config import REFERENCE_HEALTHY_ELEV_FT_NAVD88


@dataclass
class Scenario:
    name: str
    elevation_ft: float
    description: str


def default_scenarios(observed_min_elev_ft: float, observed_max_elev_ft: float) -> list[Scenario]:
    return [
        Scenario(
            "healthy_reference", REFERENCE_HEALTHY_ELEV_FT_NAVD88,
            f"Lake held at {REFERENCE_HEALTHY_ELEV_FT_NAVD88:.0f} ft NAVD88, "
            "the Strike Team's minimum healthy reference elevation (0 km^2 exposed "
            "playa by this model's definition).",
        ),
        Scenario(
            "observed_max_in_record", observed_max_elev_ft,
            f"Lake held at the highest elevation observed in the analysis "
            f"period ({observed_max_elev_ft:.1f} ft).",
        ),
        Scenario(
            "observed_min_in_record", observed_min_elev_ft,
            f"Lake held at the lowest (record-low-era) elevation observed in "
            f"the analysis period ({observed_min_elev_ft:.1f} ft).",
        ),
        Scenario(
            "further_decline_5ft", observed_min_elev_ft - 5.0,
            f"Lake 5 ft below the observed record low "
            f"({observed_min_elev_ft - 5.0:.1f} ft) -- a forward-looking "
            "what-if for continued decline.",
        ),
        Scenario(
            "recovery_plus_5ft", observed_min_elev_ft + 5.0,
            f"Lake 5 ft above the observed record low "
            f"({observed_min_elev_ft + 5.0:.1f} ft) -- a partial-recovery "
            "what-if.",
        ),
    ]


def run_scenarios(
    df: pd.DataFrame, theta: np.ndarray, scenarios: list[Scenario],
) -> pd.DataFrame:
    """For each scenario, compute the model's lake-dust contribution term
    E_GSL(t-tau)/H(t) over the WHOLE observed weather record, holding exposed
    playa area constant at that scenario's value. Returns one summary row per
    scenario plus the observed (actual-history) baseline.
    """
    rows = []

    baseline_contrib = pm.lake_contribution(df, theta)
    rows.append(_summarize("observed_history", "Actual historical lake elevation trajectory.",
                            baseline_contrib))

    for sc in scenarios:
        override = np.full(len(df), exposed_playa_area(np.array([sc.elevation_ft]))[0])
        contrib = pm.lake_contribution(df, theta, exposed_playa_km2_override=override)
        rows.append(_summarize(sc.name, sc.description, contrib))

    return pd.DataFrame(rows)


def _summarize(name: str, description: str, contrib: np.ndarray) -> dict:
    finite = np.isfinite(contrib)
    return {
        "scenario": name,
        "description": description,
        "mean_lake_contribution_ugm3_per_h": float(np.nanmean(contrib[finite])),
        "p95_lake_contribution_ugm3_per_h": float(np.nanpercentile(contrib[finite], 95)),
        "max_lake_contribution_ugm3_per_h": float(np.nanmax(contrib[finite])),
        "hours_with_nonzero_contribution": int(np.sum(contrib[finite] > 0.01)),
        "frac_hours_with_nonzero_contribution": float(np.mean(contrib[finite] > 0.01)),
    }


if __name__ == "__main__":
    import json
    from gsl_model.config import DATA_PROCESSED, MODELS_DIR, TABLES_DIR

    df = pd.read_parquet(DATA_PROCESSED / "hourly_analysis_table.parquet")
    theta = np.load(MODELS_DIR / "physics_theta.npy")

    obs_min = df["lake_elev_ft_navd88"].min()
    obs_max = df["lake_elev_ft_navd88"].max()
    scenarios = default_scenarios(obs_min, obs_max)
    summary = run_scenarios(df, theta, scenarios)
    print(summary.to_string(index=False))
    summary.to_csv(TABLES_DIR / "lake_level_counterfactuals.csv", index=False)
