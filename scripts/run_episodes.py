"""Stage 4: direct HYSPLIT runs over the 30 validation episodes with the central
run's hourly emissions, and the comparison the guide asks for.

- Monitors are compared with HYSPLIT air concentrations at the monitor, never
  with R. Observed PM10 includes every source (urban, wildfire, other playas), so
  the modeled lake-dust concentration is a contribution to it, not a prediction
  of all of it; PM10 alone cannot attribute dust to the lake (guide).
- Scores are reported separately for held-out events, held-out monitors, and
  both, using the design fixed in validation/heldout_design.json before any
  comparison. Nothing in the model is tuned to PM10, so every episode is
  out-of-sample for calibration; the held-out split is kept as specified.
- Each run also counts valley arrival directly from its particles, which checks
  the unit-source superposition (hysplit_transport) against an ordinary HYSPLIT
  run with the same emissions."""
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from gsl_dust import boundaries, decisions, episode_runs, hysplit_run, hysplit_transport
from gsl_dust.config import DATA_INTERIM, ROOT, TABLES


def _one(args):
    ep, e_zone, hours, zones, valley = args
    conc, r_direct = episode_runs.run_episode(ep, e_zone, hours, zones, valley)
    return ep["id"], conc, r_direct


def main(workers=6):
    design = episode_runs.episodes()
    hours = pd.date_range(pd.Timestamp(decisions.SIM_START), pd.Timestamp(decisions.SIM_END), freq="h", inclusive="left")
    e_zone = np.load(DATA_INTERIM / "central_e_zone.npy")
    zones, _ = hysplit_run.zones()
    valley = boundaries.valley_receptor_utm()
    jobs = [(ep, e_zone, hours, zones, valley) for ep in design["episodes"]]
    concs, direct = [], {}
    with ProcessPoolExecutor(workers) as ex:
        for eid, conc, r in ex.map(_one, jobs):
            concs.append(conc.assign(episode=eid))
            direct[eid] = r
            print(f"episode {eid}: max modeled lake dust at a monitor {conc.conc_ug_m3.max() if len(conc) else 0:.1f} ug/m3",
                  flush=True)
    conc = pd.concat(concs, ignore_index=True)
    conc.to_csv(TABLES / "episode_model_conc.csv", index=False)

    # superposition check over the same windows
    tr = hysplit_transport.Transfer(hours, e_zone.shape[1])
    r_sup = pd.Series(tr.arrival(e_zone), index=hours)
    rows = []
    for ep in design["episodes"]:
        t0, t1 = episode_runs._window(ep)
        d = direct[ep["id"]]
        rows.append({"episode": ep["id"], "R_direct_kg": float(d[(d.index >= t0) & (d.index < t1)].sum()) if len(d) else 0.0,
                     "R_superposition_kg": float(r_sup[(r_sup.index >= t0) & (r_sup.index < t1)].sum())})
    pd.DataFrame(rows).to_csv(TABLES / "superposition_check.csv", index=False)

    # model vs monitors
    obs = pd.read_parquet(ROOT / "data" / "raw" / "aqs" / "pm10_2022_wasatch.parquet")
    obs["time"] = pd.to_datetime(obs.time, utc=True)
    m = conc.merge(obs[["site_id", "time", "pm10"]], on=["site_id", "time"], how="inner")
    held_ev, held_mon = set(design["heldout_episode_ids"]), set(design["heldout_monitors"])
    m["heldout_event"] = m.episode.isin(held_ev)
    m["heldout_monitor"] = m.site_id.isin(held_mon)
    windows = {ep["id"]: (pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"])) for ep in design["episodes"]}
    m["in_event"] = [windows[e][0] <= t <= windows[e][1] for e, t in zip(m.episode, m.time)]
    m.to_csv(TABLES / "episode_model_vs_monitor.csv", index=False)
    scores = []
    for name, sel in {"all": m.in_event, "held-out events": m.in_event & m.heldout_event,
                      "held-out monitors": m.in_event & m.heldout_monitor,
                      "held-out events at held-out monitors": m.in_event & m.heldout_event & m.heldout_monitor}.items():
        s = m[sel]
        if len(s) < 3:
            continue
        per_ev = s.groupby(["episode", "site_id"]).agg(obs_peak=("pm10", "max"), mod_peak=("conc_ug_m3", "max"))
        scores.append({"subset": name, "n_hours": len(s), "n_event_monitor_pairs": len(per_ev),
                       "corr_hourly": float(np.corrcoef(s.pm10, s.conc_ug_m3)[0, 1]) if s.conc_ug_m3.std() > 0 else np.nan,
                       "median_modeled_share_of_observed": float((s.conc_ug_m3 / s.pm10.clip(lower=1)).median()),
                       "pairs_with_modeled_peak_over_50": int((per_ev.mod_peak > 50).sum()),
                       "obs_peak_median": float(per_ev.obs_peak.median()), "mod_peak_median": float(per_ev.mod_peak.median())})
    pd.DataFrame(scores).to_csv(TABLES / "validation_scores.csv", index=False)
    print(pd.DataFrame(scores).round(3).to_string())


if __name__ == "__main__":
    main()
