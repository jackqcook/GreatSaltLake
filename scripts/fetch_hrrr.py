"""Download hourly HRRR inputs for the simulation period (resumable)."""
import sys
import time

import pandas as pd

from gsl_dust import decisions, hrrr


def main(workers=24):
    months = pd.period_range(decisions.SIM_START.replace(tzinfo=None),
                             (decisions.SIM_END - pd.Timedelta(hours=1)).replace(tzinfo=None), freq="M")
    for m in months:
        t0 = time.time()
        failed = hrrr.download_month(str(m), workers=workers)
        print(f"{m} took {(time.time() - t0) / 60:.1f} min, {len(failed)} failed hours", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
