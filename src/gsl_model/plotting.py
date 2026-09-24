"""Event diagnostic plots (Section 8: "Plot and label 20-50 major episodes")
and model-comparison / counterfactual figures for the write-up.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from gsl_model.config import FIGURES_DIR
from gsl_model.evaluate import EventBlock


def plot_event(df: pd.DataFrame, ev: EventBlock, pred_series: dict | None = None,
               pad_hours: int = 24, save_path=None, title_suffix: str = ""):
    lo = max(0, ev.start_idx - pad_hours)
    hi = min(len(df), ev.end_idx + pad_hours)
    seg = df.iloc[lo:hi]
    t = seg["datetime_local"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    ax.plot(t, seg["pm10_ugm3"], color="black", lw=1.3, label="Observed PM10")
    if pred_series:
        for name, arr in pred_series.items():
            ax.plot(t, np.asarray(arr)[lo:hi], lw=1.0, alpha=0.85, label=name)
    ax.axhspan(0, 0, color="none")
    ax.axvspan(df["datetime_local"].iloc[ev.start_idx], df["datetime_local"].iloc[ev.end_idx],
               color="orange", alpha=0.12)
    ax.set_ylabel("PM10 (ug/m3)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"Event peak {ev.peak_value:.0f} ug/m3 at "
                 f"{df['datetime_local'].iloc[ev.peak_idx]:%Y-%m-%d %H:%M} {title_suffix}")

    ax = axes[1]
    sc = ax.scatter(t, seg["wind_speed_ms"], c=seg["wind_dir_deg"], cmap="twilight",
                     s=10, vmin=0, vmax=360)
    ax.set_ylabel("Wind speed (m/s)")
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Wind dir (deg)")

    ax = axes[2]
    ax.bar(t, seg["precip_mm"], width=0.03, color="steelblue")
    ax.set_ylabel("Precip (mm/h)")

    ax = axes[3]
    ax.plot(t, seg["exposed_playa_km2"], color="saddlebrown", lw=1.2)
    ax.set_ylabel("Exposed playa (km2)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.autofmt_xdate()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
    return fig


def plot_top_events(df: pd.DataFrame, events: list[EventBlock], n: int = 30,
                     pred_series: dict | None = None, prefix: str = "event"):
    top = sorted(events, key=lambda e: -e.peak_value)[:n]
    out_dir = FIGURES_DIR / "events"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, ev in enumerate(top, 1):
        d = df["datetime_local"].iloc[ev.peak_idx]
        fname = out_dir / f"{prefix}_{i:02d}_{d:%Y%m%d}.png"
        plot_event(df, ev, pred_series=pred_series, save_path=fname)
    return [str(out_dir / f"{prefix}_{i:02d}_*.png") for i in range(1, len(top) + 1)]


def plot_model_ladder_timeseries(df: pd.DataFrame, preds: dict, lo: int, hi: int, save_path=None):
    seg = df.iloc[lo:hi]
    t = seg["datetime_local"]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(t, seg["pm10_ugm3"], color="black", lw=1.4, label="Observed")
    for name, arr in preds.items():
        ax.plot(t, np.asarray(arr)[lo:hi], lw=1.0, alpha=0.85, label=name)
    ax.set_ylabel("PM10 (ug/m3)")
    ax.legend(fontsize=9, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
        plt.close(fig)
    return fig


def plot_metric_bar(metrics_df: pd.DataFrame, metric: str, save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(metrics_df["model"], metrics_df[metric], color="teal")
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
        plt.close(fig)
    return fig


def plot_counterfactuals(cf_df: pd.DataFrame, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(cf_df["scenario"], cf_df["mean_lake_contribution_ugm3_per_h"], color="darkorange")
    ax.set_ylabel("Mean lake contribution (ug/m3 per hour)")
    ax.set_title("Lake-level counterfactuals: mean modeled GSL dust contribution")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
        plt.close(fig)
    return fig


def plot_lake_elevation_and_exposure(df: pd.DataFrame, save_path=None):
    d = df.drop_duplicates("date_local")[["date_local", "lake_elev_ft_navd88", "exposed_playa_km2"]]
    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(d["date_local"], d["lake_elev_ft_navd88"], color="steelblue")
    ax1.set_ylabel("Lake elevation (ft NAVD88)", color="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(d["date_local"], d["exposed_playa_km2"], color="saddlebrown")
    ax2.set_ylabel("Exposed playa (km2, vs. 4198 ft ref.)", color="saddlebrown")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
        plt.close(fig)
    return fig
