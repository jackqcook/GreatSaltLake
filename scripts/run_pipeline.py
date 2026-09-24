"""End-to-end pipeline: build data -> detect & plot events -> fit the full
model ladder (persistence, GAM, XGBoost, physics, hybrid, two-stage event
model) -> evaluate on a held-out, event-aware test period -> physical sanity
checks -> lake-level counterfactuals -> figures + tables for the write-up.

Run with:  PYTHONPATH=src python3 scripts/run_pipeline.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gsl_model import physics_model as pm
from gsl_model import baselines, hybrid, event_model, evaluate, plotting, counterfactual, uncertainty
from gsl_model.config import DATA_PROCESSED, MODELS_DIR, TABLES_DIR, FIGURES_DIR, PRIMARY_SITE
from gsl_model.acquisition.usgs_lake import exposed_playa_area


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Weight on squared error for the physics-model fit, applied to the SQUARED
# residual as y_next ** EVENT_WEIGHT_POWER. Plain OLS (power=0) is dominated by
# the ~98% of quiet-air hours and pushes the wind threshold U_c to the edge of
# its bounds, effectively turning the lake-dust term off; see physics_model.fit
# docstring. power=0.25 is a moderate compromise (verified empirically):
# a real, non-degenerate lake-dust signal, at a modest cost in overall
# one-step-ahead accuracy that the hybrid residual model recovers.
EVENT_WEIGHT_POWER = 0.25


def main():
    log("Loading hourly analysis table...")
    df = pd.read_parquet(DATA_PROCESSED / "hourly_analysis_table.parquet")
    log(f"  {len(df)} hourly rows, {df['datetime_local'].min()} to {df['datetime_local'].max()}")

    # ------------------------------------------------------------------
    # Event detection + diagnostic plots (Section 8 MVP step)
    # ------------------------------------------------------------------
    log("Detecting dust episodes and plotting top events...")
    events_all = evaluate.detect_events(df)
    log(f"  {len(events_all)} episodes >= {evaluate.EVENT_PM10_THRESHOLD} ug/m3 detected")
    plotting.plot_top_events(df, events_all, n=30, prefix="episode")
    events_table = pd.DataFrame([{
        "start": df["datetime_local"].iloc[e.start_idx],
        "end": df["datetime_local"].iloc[e.end_idx],
        "peak_time": df["datetime_local"].iloc[e.peak_idx],
        "peak_pm10": e.peak_value,
        "duration_h": e.end_idx - e.start_idx + 1,
    } for e in events_all]).sort_values("peak_pm10", ascending=False)
    events_table.to_csv(TABLES_DIR / "dust_episodes.csv", index=False)

    plotting.plot_lake_elevation_and_exposure(df, save_path=FIGURES_DIR / "lake_elevation_exposure.png")

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    split = evaluate.time_based_event_aware_split(df, train_frac=0.70, val_frac=0.15)
    log(f"Split: train={len(split.train)} val={len(split.val)} test={len(split.test)}")

    # ------------------------------------------------------------------
    # Model ladder
    # ------------------------------------------------------------------
    log("[1/5] Persistence baseline (no fitting)...")

    log("[2/5] Fitting meteorology-only GAM baseline...")
    gam = baselines.fit_gam_baseline(split.train)

    log("[3/5] Fitting XGBoost baseline...")
    xgb_model = baselines.fit_xgboost_baseline(split.train, split.val)

    log(f"[4/5] Fitting physics state-space model (event_weight_power={EVENT_WEIGHT_POWER})...")
    fitted_physics = pm.fit(split.train, verbose=1, event_weight_power=EVENT_WEIGHT_POWER)
    log(f"  fitted params: {fitted_physics.as_dict()}")
    log(f"  converged: {fitted_physics.success} ({fitted_physics.message})")
    np.save(MODELS_DIR / "physics_theta.npy", fitted_physics.theta)
    with open(MODELS_DIR / "physics_params.json", "w") as f:
        json.dump({k: float(v) for k, v in fitted_physics.as_dict().items()}, f, indent=2)

    boundary_flags = evaluate.flag_boundary_parameters(fitted_physics.theta)
    with open(TABLES_DIR / "physics_boundary_parameter_flags.txt", "w") as f:
        f.write("\n".join(boundary_flags) if boundary_flags else "none")
    if boundary_flags:
        log(f"  WARNING - parameters at bounds: {boundary_flags}")

    log("Block-bootstrapping physics parameter uncertainty (15 reps)...")
    boot = uncertainty.block_bootstrap_physics_params(
        split.train, n_reps=15, event_weight_power=EVENT_WEIGHT_POWER, x0=fitted_physics.theta,
    )
    boot_summary = boot.summary()
    boot_summary.to_csv(TABLES_DIR / "physics_param_bootstrap_uncertainty.csv", index=False)
    log("\n" + boot_summary.to_string(index=False))

    log("[5/5] Fitting hybrid residual model...")
    hybrid_model = hybrid.fit_hybrid(split.train, fitted_physics.theta, split.val)

    log("Fitting two-stage event model...")
    evt_model = event_model.fit_event_model(split.train)

    # ------------------------------------------------------------------
    # Predictions on test set
    # ------------------------------------------------------------------
    log("Generating test-set predictions for all models...")
    test = split.test
    y_next = test["pm10_ugm3"].shift(-1).to_numpy()

    preds = {
        "persistence": baselines.persistence_predict(test),
        "meteorology_gam": gam.predict(test),
        "xgboost": xgb_model.predict(test),
        "physics": pm.predict(test, fitted_physics.theta),
        "hybrid_physics_plus_ml": hybrid_model.predict(test),
        "event_model": evt_model.predict(test),
    }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    log("Scoring models on the held-out test period...")
    rows = []
    for name, pred in preds.items():
        rows.append({
            "model": name,
            "MAE": evaluate.mae(pred, y_next),
            "RMSE": evaluate.rmse(pred, y_next),
            "bias": evaluate.bias(pred, y_next),
            "event_recall": evaluate.event_recall(pred, y_next),
            "event_precision": evaluate.event_precision(pred, y_next),
        })
    metrics_df = pd.DataFrame(rows).sort_values("RMSE")
    metrics_df.to_csv(TABLES_DIR / "model_comparison_test.csv", index=False)
    log("\n" + metrics_df.to_string(index=False))

    plotting.plot_metric_bar(metrics_df, "MAE", save_path=FIGURES_DIR / "model_comparison_mae.png",
                              title="Test-set MAE by model")
    plotting.plot_metric_bar(metrics_df, "RMSE", save_path=FIGURES_DIR / "model_comparison_rmse.png",
                              title="Test-set RMSE by model")

    # Peak error / timing for physics + hybrid on held-out events
    test_events = evaluate.detect_events(test.reset_index(drop=True))
    for name in ["physics", "hybrid_physics_plus_ml", "xgboost"]:
        pe = evaluate.peak_error_and_timing(preds[name], test["pm10_ugm3"].to_numpy(), test_events)
        pe.to_csv(TABLES_DIR / f"peak_error_timing_{name}_test.csv", index=False)
        if len(pe):
            log(f"  {name}: median |peak error|={pe['peak_error'].abs().median():.1f} ug/m3, "
                f"median |timing offset|={pe['timing_offset_h'].abs().median():.1f} h "
                f"(n={len(pe)} held-out events)")

    # ------------------------------------------------------------------
    # Physical sanity checks (Section 7)
    # ------------------------------------------------------------------
    log("Running physical sanity checks on the physics model...")
    lake_contrib_test = pm.lake_contribution(test, fitted_physics.theta)
    sanity = evaluate.physical_sanity_checks(test, lake_contrib_test)
    with open(TABLES_DIR / "physical_sanity_checks.json", "w") as f:
        json.dump(sanity, f, indent=2)
    for k, v in sanity.items():
        log(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Event-window comparison plots on the test period (model ladder overlay)
    # ------------------------------------------------------------------
    log("Plotting model-ladder overlays for the largest held-out test events...")
    test_reset = test.reset_index(drop=True)
    top_test_events = sorted(test_events, key=lambda e: -e.peak_value)[:8]
    overlay_preds = {
        "Physics": preds["physics"], "Hybrid": preds["hybrid_physics_plus_ml"],
        "XGBoost": preds["xgboost"], "Persistence": preds["persistence"],
    }
    for i, ev in enumerate(top_test_events, 1):
        plotting.plot_event(
            test_reset, ev, pred_series=overlay_preds, pad_hours=36,
            save_path=FIGURES_DIR / f"test_event_ladder_{i:02d}.png",
            title_suffix="[held-out test]",
        )

    # ------------------------------------------------------------------
    # Lake-level counterfactuals (full record, using the fitted physics model)
    # ------------------------------------------------------------------
    log("Running lake-level counterfactual scenarios...")
    obs_min = df["lake_elev_ft_navd88"].min()
    obs_max = df["lake_elev_ft_navd88"].max()
    scenarios = counterfactual.default_scenarios(obs_min, obs_max)
    cf_summary = counterfactual.run_scenarios(df, fitted_physics.theta, scenarios)
    cf_summary.to_csv(TABLES_DIR / "lake_level_counterfactuals.csv", index=False)
    log("\n" + cf_summary.to_string(index=False))
    plotting.plot_counterfactuals(cf_summary, save_path=FIGURES_DIR / "lake_level_counterfactuals.png")

    log("Pipeline complete. Tables in results/tables, figures in results/figures, "
        "fitted physics params in results/models.")


if __name__ == "__main__":
    main()
