"""Stage 3.2: unit-source HYSPLIT runs for every lakebed zone over the simulation
period, with HRRR weather streamed through a small disk buffer.

For each 4-day emission chunk: make sure its HRRR/ARL days exist (the next chunk's
days download in the background meanwhile), run every zone in parallel, keep only
the per-zone arrival tables, then delete weather days no longer needed, except the
days around the Stage 4 validation episodes. Resumable: finished zone-chunks are
skipped.

    python scripts/run_hysplit_transfer.py [--start 2022-04-30] [--chunks N] [--workers 14]
"""
import argparse
import json
import threading
import time
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from gsl_dust import boundaries, decisions, hrrr_arl, hysplit_run
from gsl_dust.config import ROOT, TABLES

LOG = ROOT / "logs" / "hysplit_transfer.log"
STATS = TABLES / "hysplit_run_stats.csv"


def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def keep_days():
    design = json.loads((ROOT / "validation" / "heldout_design.json").read_text())
    days = set()
    for e in design["episodes"]:
        s, en = pd.Timestamp(e["start"]).floor("D"), pd.Timestamp(e["end"]).floor("D")
        days.update(pd.date_range(s - pd.Timedelta(days=1), en, freq="D"))
    return days


def ensure_met(days):
    for d in days:
        hrrr_arl.write_day(d, log=log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--chunks", type=int, default=None)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()

    all_chunks = list(hysplit_run.chunks(pd.Timestamp(decisions.SIM_START), pd.Timestamp(decisions.SIM_END)))
    if a.start:
        s = pd.Timestamp(a.start, tz="UTC")
        all_chunks = [c for c in all_chunks if c[0] >= s]
    if a.chunks:
        all_chunks = all_chunks[:a.chunks]
    zones, zone_of_cell = hysplit_run.zones()
    valley = boundaries.valley_receptor_utm()
    sim_hours = pd.date_range(pd.Timestamp(decisions.SIM_START), pd.Timestamp(decisions.SIM_END), freq="h", inclusive="left")
    windy = hysplit_run.windy_hours(sim_hours, zone_of_cell, len(zones))
    log(f"zone-hours where some parameter set could emit: {windy.mean():.1%}")
    keep = keep_days()
    log(f"start: {len(all_chunks)} chunks x {len(zones)} zones, {a.workers} workers")

    prefetch = None
    for k, (t0, t1) in enumerate(all_chunks):
        days = hysplit_run.met_days(t0, t1)
        if prefetch is not None:
            prefetch.join()
        ensure_met(days)
        if k + 1 < len(all_chunks):
            nxt = hysplit_run.met_days(*all_chunks[k + 1])
            prefetch = threading.Thread(target=ensure_met, args=(nxt,), daemon=True)
            prefetch.start()
        tic = time.time()
        stats = []
        with ProcessPoolExecutor(a.workers) as ex:
            a0 = int((t0 - sim_hours[0]) / pd.Timedelta(hours=1))
            a1 = int((t1 - sim_hours[0]) / pd.Timedelta(hours=1))
            futs = [ex.submit(hysplit_run.run_zone, SimpleNamespace(**z), t0, t1, valley, windy[a0:a1, z["zone"]])
                    for z in zones.to_dict("records")]
            for f in as_completed(futs):
                r = f.result()
                if r:
                    stats.append(r)
        if stats:
            st = pd.DataFrame(stats)
            st.to_csv(STATS, mode="a", header=not STATS.exists(), index=False)
            log(f"chunk {t0:%Y-%m-%d}: {len(st)} zones in {(time.time() - tic) / 60:.1f} min, "
                f"mean arrived fraction {st.arrived_frac.mean():.3f}, max dump {st.dump_mb.max():.0f} MB")
        # drop weather no later chunk needs, except validation-episode days
        later = set(d for c in all_chunks[k + 1:k + 2] for d in hysplit_run.met_days(*c))
        for d in days:
            if d not in later and d not in keep:
                p = hrrr_arl.ARL_DIR / f"hrrr_utah_{d:%Y%m%d}.arl"
                p.unlink(missing_ok=True)
    log("done")


if __name__ == "__main__":
    main()
