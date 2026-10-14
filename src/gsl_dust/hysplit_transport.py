"""Stage 3.2 transport: valley arrival from HYSPLIT unit-source runs.

hysplit_run.py stores, per lakebed zone and release hour, the mass that first
entered the valley in each later hour per kg released. Because HYSPLIT's
transport and deposition are linear in mass, the arrival for any hourly emission
series E[t, zone] (beta*S summed over the zone's cells) is

    dR/dt [h] = sum over zones z, release hours t:  E[t, z] * A[z, t, h]

and the same with arsenic emissions for dR_As/dt."""
import numpy as np
import pandas as pd

from gsl_dust import hysplit_run


class Transfer:
    def __init__(self, hours: pd.DatetimeIndex, n_zones: int, table: pd.DataFrame = None,
                 diameter_um=hysplit_run.CENTRAL_DIAMETER_UM):
        t = hysplit_run.load_transfer(diameter_um=diameter_um) if table is None else table
        h0 = hours[0]
        rel = ((pd.to_datetime(t.release, utc=True) - h0) / pd.Timedelta(hours=1)).to_numpy().astype(np.int64)
        ent = ((pd.to_datetime(t.entry, utc=True) - h0) / pd.Timedelta(hours=1)).to_numpy().astype(np.int64)
        keep = (rel >= 0) & (rel < len(hours)) & (ent >= 0)
        self.hours, self.n_zones = hours, n_zones
        self.zone = t.zone.to_numpy().astype(np.int64)[keep]  # empty per-zone files can make this column float
        self.rel, self.ent = rel[keep], ent[keep]
        self.mass = t.mass.to_numpy()[keep]
        # which (zone, release hour) pairs HYSPLIT has covered
        self.covered = np.zeros((len(hours), n_zones), bool)
        runs = hysplit_run.OUT.glob("*/zone*.parquet") if table is None else []
        for f in runs:
            t0 = pd.to_datetime(f.parent.name, format="%Y%m%d%H").tz_localize("UTC")
            a = int((t0 - h0) / pd.Timedelta(hours=1))
            z = int(f.stem[4:])
            self.covered[max(a, 0):max(a + hysplit_run.EMIT_HOURS, 0), z] = True
        # arrival fraction per release hour and zone (kg reaching the valley per kg released)
        self.frac = np.zeros((len(hours), n_zones))
        np.add.at(self.frac, (self.rel, self.zone), self.mass)

    def coverage(self):
        return float(self.covered.mean())

    def arrival(self, e_zone: np.ndarray) -> np.ndarray:
        """Hourly kg entering the valley for emissions e_zone (T, Z) in kg/h.
        Arrivals after the last simulated hour are dropped (they belong to the
        next period)."""
        w = self.mass * e_zone[self.rel, self.zone]
        inside = self.ent < len(self.hours)
        return np.bincount(self.ent[inside], w[inside], len(self.hours))
