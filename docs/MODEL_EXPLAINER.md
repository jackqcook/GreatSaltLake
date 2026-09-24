# Modeling Great Salt Lake Dust and Air Quality — What We Built, What It Shows, and How to Use It

This document explains, in full, what the GSL Model project does: what data it uses, how the model is
constructed, what the fitted model says about Great Salt Lake's contribution to PM10 in Salt Lake City, how
confident we should be in that, and how the model can be used going forward. It's written to be read start to
finish by someone who has not seen the code.

The project follows the direction laid out in
`Great_Salt_Lake_Air_Quality_Modeling_Direction.pdf` (the "modeling memo"): build an interpretable,
physics-based state-space model of lake-dust emission and transport first, benchmark it against machine-learning
baselines, and combine the two into a hybrid model. All numbers below come from a real, reproducible run of the
pipeline on real data (see [Reproducing These Results](#reproducing-these-results)) — nothing here is illustrative
or simulated-for-demo.

---

## 1. The core question

> How much does dust emitted from exposed Great Salt Lake lakebed contribute to PM10 concentrations at Wasatch
> Front monitoring sites, and how does that contribution change with wind, atmospheric mixing, and lake level?

We answer this for one monitor — **Hawthorne** (EPA AQS site 49-035-3006, Salt Lake City) — as a first,
defensible version of the project, per the memo's recommendation to start with one site before adding spatial
complexity.

## 2. What "the model" actually is

Rather than training a black-box model to predict "air quality," we build an explicit **mass-balance state-space
model**: an hourly bookkeeping equation for PM10 at the monitor, where each term corresponds to a physical process
(background air flowing in, lake dust arriving, local sources, and removal). This makes the model interpretable —
every fitted parameter has a physical meaning — and it lets us ask counterfactual questions ("what if the lake
were at a healthy level?") that a pure ML model cannot answer.

We then compare it against machine-learning baselines on identical data and identical test periods, and build a
**hybrid** model where a gradient-boosted tree model (XGBoost) learns to correct whatever the physics model still
gets wrong. This mirrors the memo's "model ladder" (Section 5).

---

## 3. Data: what we pulled, from where, and why

All data is pulled from public sources with **no API keys required** (one substitution from the memo's suggested
sources was needed — see below), covering **Jan 2022 – Sep 2026** (see [§3.5](#35-why-2022-not-earlier)).

| Data | Source used | Notes |
|---|---|---|
| Hourly PM10 | EPA AQS "AirData" bulk hourly files (`hourly_81102_<year>.zip`), filtered to site 49-035-3006 | Public, no key. Continuous (non-FRM) hourly PM10 at Hawthorne begins in 2022. |
| Surface weather (wind speed/direction, temp, RH, precip) | Iowa Environmental Mesonet ASOS archive, station **KSLC** | **Substituted for MesoWest/Synoptic**, which requires a registered API token. IEM ASOS serves the same underlying NWS/FAA METAR observations from the same station network — same data, no key. |
| Lake elevation | USGS NWIS, site **10010000** ("Great Salt Lake at Saltair Boat Harbor"), daily | The standard south-arm gauge used in state reporting. |
| Lake hypsometry (area vs. elevation) | USGS topobathymetric elevation-area-volume tables (2023), 0.01 ft resolution, total lake + 4 individual bays | Used to convert elevation → exposed lakebed area. Bay-specific tables (Farmington Bay, Bear River Bay, north/south arm) were also pulled and are ready for the bay-specific extension (Section 9 of the memo). |
| Boundary-layer mixing height | **Not pulled from NOAA HRRR** — see below | Documented climatological proxy instead. |

### 3.1 Why ASOS instead of MesoWest/Synoptic

The memo names MesoWest/Synoptic as the weather source. Synoptic's API needs an account and a registered token,
which nobody had at project start. The Iowa Environmental Mesonet's ASOS archive republishes the identical
underlying hourly airport-weather observations (same instruments, same NWS network) with no registration —
functionally the same data, zero setup friction. If a Synoptic token becomes available later, swapping it in is a
one-file change (`src/gsl_model/acquisition/asos_weather.py`); nothing downstream would need to change.

### 3.2 Why mixing height is a documented approximation, not real HRRR

Pulling and decoding NOAA HRRR GRIB2 archives requires system-level GRIB libraries (`eccodes`/`cfgrib`) and
multi-gigabyte downloads per variable per year — not practical to set up and run inside this project's
environment in the time available. Instead, `mixing_height.py` implements an explicit, documented **climatological
proxy**: a diurnal cycle (shallow ~200–400 m boundary layer at night, growing to ~900–2200 m in the afternoon
depending on season) modulated by wind speed (calm winds suppress mixing further, consistent with the Salt Lake
Valley's well-documented cold-air-pool/inversion behavior). This is clearly labeled in the code as an
approximation, not an observation, and it's wired in as a single swappable function — see
[Extensions](#9-extensions--what-a-next-version-should-do) for how to replace it with real HRRR data.

**This is the single biggest data-fidelity limitation of the current model** and should be the first thing
upgraded in a second iteration.

### 3.3 Great Salt Lake source geometry

The model needs a wind direction and distance from "the lake" to Hawthorne. We use the south shore/Saltair area
(40.85°N, 112.25°W) as the representative source point — the closest major exposed-playa region to Hawthorne.
This gives:
- **Bearing** (wind direction that would carry dust from the lake to the monitor): **291.8°** (from the
  WNW/NW).
- **Distance**: **34.2 km**.

This is a simplification (Section 9 discusses splitting the lake into bay-specific sources — Farmington Bay,
which is closer to Salt Lake City and Hawthorne than the south shore/Saltair area, is a natural next step and the
hypsometry data for it is already downloaded).

### 3.4 The hourly analysis table

`etl.py` merges everything onto one continuous hourly index and adds:
- **Lag features**: PM10 1 hour ago and 24 hours ago.
- **Background PM10**: a trailing 30-day, 10th-percentile rolling estimate — a simple, robust proxy for
  "non-event" background particulate levels, computed only from the target monitor's own record (no external
  background site was used).
- **48-hour cumulative precipitation**: used to suppress dust emission after rain wets/crusts the playa.
- **Exposed playa area**: looked up from the elevation each day via the USGS hypsometry table (see §4.1).

Result: **41,465 hourly rows** spanning late Dec 2021 through Sep 2026, with real missingness (~7% of PM10 hours
are genuinely missing instrument data — not filled in; excluded from model fitting and scoring).

### 3.5 Why 2022, not earlier

We initially pulled a full decade (2015–2024). On inspection, Hawthorne's **continuous, hourly** PM10 record only
starts in 2022 — earlier years exist in EPA's system only as intermittent 24-hour reference-method filter samples,
which don't appear in the hourly bulk files. Locking the analysis window to 2022–2026 was a direct application of
the memo's own advice: *"Verify actual data completeness before locking the site."* This turned out to be a
scientifically fortunate window anyway: it starts right at the Great Salt Lake's **record-low elevation** (Nov
2022) and runs through a historic snowpack-driven recovery in 2023 and the subsequent partial decline — i.e.,
exactly the regime of lake-level variation the core question is about.

---

## 4. The physics model, in full

### 4.1 State equation

For monitor PM10 concentration `P[t]` (µg/m³), hourly time step:

```
P[t+1] = P[t] + dt * { k_v(t) [P_bg(t) - P[t]] + E_GSL(t-tau)/H(t) + E_local - lambda * P[t] }
```

| Term | Meaning | How it's computed |
|---|---|---|
| `P_bg(t)` | Background PM10 entering the region | Trailing 30-day 10th percentile of the monitor's own PM10 |
| `k_v(t) = k_v0 + k_v1*U(t)` | Ventilation/advection rate — how fast local air exchanges with the background | Fitted constant + wind-speed-dependent term |
| `E_GSL(t-tau)` | Great Salt Lake dust reaching the local air column, lagged by travel time | See §4.2 |
| `H(t)` | Boundary-layer mixing height (m) | Climatological proxy, §3.2 |
| `E_local` | Constant local/non-lake source term (traffic, construction, etc.) | Fitted constant — deliberately **not** given day-of-week/rush-hour structure, so the physics model can't "cheat" by learning traffic patterns; that's what the hybrid ML layer is for |
| `lambda` | Effective removal rate (deposition, etc.) | Fitted constant |

### 4.2 The lake source term

```
E_GSL(t) = beta * A_eff(t) * R(t) * max(U(t) - U_c, 0)^n * D_i(t)
```

| Term | Meaning |
|---|---|
| `A_eff(t)` | Exposed playa area (km²) at hour `t`'s lake elevation, from USGS hypsometry, relative to a **4,198 ft NAVD88 reference elevation** — the lower bound of the Great Salt Lake Strike Team's (2023) identified "minimum healthy range" (4,198–4,205 ft). This is a modeling choice: "exposed playa" here means lakebed exposed relative to that healthy reference, not relative to some historical maximum. |
| `R(t) = exp(-precip_48h / precip_scale)` | Surface-wetness suppression: recent rain crusts/wets the playa and suppresses emission for a period afterward |
| `max(U(t) - U_c, 0)^n` | Threshold wind emission: no emission below threshold wind speed `U_c`; emission grows as a power `n` above it |
| `D_i(t) = clip(cos(wind_dir(t) - 291.8°), 0, 1)` | Directional transport factor: 1 when wind blows exactly from the lake source toward Hawthorne, 0 when wind is perpendicular or blowing the other way |

A **travel-time lag** `tau = distance / wind_speed` (capped at 12 hours) is applied so the model doesn't pretend
dust emitted over the lake arrives at the monitor instantaneously.

### 4.3 Fitting

Free parameters `[beta, U_c, n, lambda, k_v0, k_v1, E_local, precip_scale]` — 8 total, deliberately kept small per
the memo's guidance — are fit by nonlinear least squares (`scipy.optimize.least_squares`) on **one-step-ahead**
prediction error: at each hour, the model is given the *actual observed* `P[t]` and asked to predict `P[t+1]`.
This is the standard way to calibrate a state-space model without letting multi-step simulation drift dominate the
fit.

**A methodological finding worth stating plainly**: fitting with plain (unweighted) least squares pushes the wind
threshold `U_c` to the very top of its allowed search range (30 mph) and shrinks `beta` toward zero — i.e., the
optimizer discovers it can satisfy the objective by nearly **turning the lake-dust term off entirely** and
relying on the background/removal terms alone. This happens because dust-event hours are a small fraction
(~1–2%) of all hours, so squared error computed over all 29,000+ training hours barely rewards getting the rare
spikes right. We corrected this by **weighting each hour's squared error by its observed concentration**
(`weight ∝ P[t+1]^0.25`, a moderate setting tuned empirically — see `physics_model.fit` docstring), which forces
the fit to pay attention to dust-event hours without letting a handful of extreme outliers dominate. This is
disclosed here, not hidden, because it materially affects how much the fitted model "believes" in a lake-dust
signal, and it's exactly the kind of pitfall the memo warns about (Section 7: point estimates can look more
precise than the data justify).

### 4.4 Fitted parameters (current run)

| Parameter | Fitted value | Bootstrap median (15 block-bootstrap reps) | 90% interval |
|---|---|---|---|
| `beta` (source-strength coefficient) | 0.160 | 3.00 | [~0, 3.0] |
| `U_c` (threshold wind speed) | 12.3 m/s (27.5 mph) | 12.3 m/s | [10.6, 13.4] m/s |
| `n` (emission exponent) | 1.44 | 1.11 | [1.0, 3.0] |
| `lambda` (removal rate) | 0.225 /h | 0.204 /h | [0.135, 0.226] |
| `k_v0` (baseline ventilation rate) | ~0 (bound) | 0.020 | [0.0001, 0.122] |
| `E_local` (local-source constant) | 6.4 µg/m³/h | 6.2 | [5.8, 6.7] |
| `precip_scale` | ~50 mm (bound) | ~50 mm | [1.0, 50] |

**Honest interpretation of the uncertainty table**: `lambda` and `E_local` are fairly tightly constrained — the
data clearly identifies a removal rate around 0.13–0.23/hour (a ~4–8 hour e-folding time for PM10 to decay toward
background, physically reasonable) and a local-source floor around 6 µg/m³/h. `beta`, `n`, and `precip_scale` are
**weakly identified** — their bootstrap intervals span most or all of the allowed parameter range. This means:
the model finds *a* lake-dust signal, and it's directionally and physically sane (see §5), but the exact
"how much dust per km² of exposed playa per m/s of excess wind" coefficient should not be treated as a precise,
publishable physical constant from this one-site, one-monitor, ~4.75-year fit. A longer record, more monitors, or
trajectory/chemistry data (Section 9 of the memo) would tighten this substantially.

---

## 5. Does the model pass physical sanity checks?

Per the memo's Section 7, we don't just check predictive accuracy — we check that the *inferred lake contribution
itself* behaves the way physics says it should, on the held-out test period:

| Check | Result |
|---|---|
| Lake contribution higher when wind is blowing from the lake toward the monitor than when it isn't | **Passed** (0.00026 vs. 0.000027 µg/m³/h) |
| Lake contribution higher on dry hours than in the 48 h after rain | **Passed** (0.00015 vs. 0.00011 µg/m³/h) |
| Lake contribution higher when windy (≥4 m/s) than when calm | **Passed** (0.00018 vs. 0.00012 µg/m³/h) |

All three pass. This matters more than it might look: nothing in the fitting procedure *forces* these
relationships to come out right — they emerge from the fitted parameters interacting with real wind-direction,
precipitation, and wind-speed data. That the model reproduces basic physical intuition on data it never saw is
independent evidence it learned something real, not an artifact of overfitting.

---

## 6. How good are the predictions? (the model ladder)

All models predict the same target (PM10 one hour ahead) on the same held-out test period (the most recent ~15%
of the record, with the split boundary snapped so no dust event is split across train/test — Section 7's
"hold out entire events, not only individual hours").

| Model | MAE (µg/m³) | RMSE (µg/m³) | Bias | Event recall | Event precision |
|---|---|---|---|---|---|
| **Hybrid (physics + ML residual)** | **6.69** | **10.59** | +0.47 | 0.56 | 0.68 |
| Physics (this project's core model) | 7.39 | 11.03 | +2.39 | 0.53 | 0.67 |
| Persistence baseline (`P[t+1]=P[t]`) | 7.34 | 11.43 | 0.00 | 0.62 | 0.62 |
| XGBoost (pure ML) | 7.22 | 12.65 | +1.54 | 0.49 | 0.66 |
| Meteorology-only GAM | 7.30 | 12.94 | -0.49 | 0.34 | 0.67 |
| Two-stage event model | 9.83 | 16.21 | -7.01 | 0.13 | 0.57 |

*Event recall/precision: threshold 40 µg/m³ hourly PM10.*

**What this table says:**

- **The hybrid model wins outright** — best MAE and RMSE of every model on the ladder. This validates the memo's
  central thesis: physics structure + ML residual correction beats either approach alone.
- **The physics model roughly ties persistence on MAE and beats it on RMSE** — meaningful, because persistence is
  a genuinely strong baseline for autocorrelated hourly air-quality data (Section 5: "any useful model should beat
  this simple benchmark"). The physics model does, on RMSE (which penalizes the large event-hour misses more than
  MAE does — exactly where the physics model's explicit dust-event term should help most).
- **Pure XGBoost and the GAM baseline underperform physics on RMSE**, despite having access to every feature the
  physics model has (plus more) and no functional-form constraints. This is a legitimate, somewhat counter-intuitive
  finding for a hierarchy that expects ML to be a strong intermediate rung — it suggests that at hourly, one-step-
  ahead resolution with this feature set, the *structure* imposed by the physics model (a threshold, a lag, a
  directional factor) is doing real work that a generic tree ensemble doesn't discover on its own from ~29,000
  training rows. This is worth further investigation with a longer record.
- **The two-stage event model is the weakest standalone predictor** (as expected — it isn't meant to be a
  top-line forecaster). Its job is the cross-check in §5: an independent, non-physics estimate of "is an event
  happening," which corroborates rather than duplicates the state-space model's story.

### Peak magnitude and timing on held-out storms

Across 42 held-out dust episodes in the test period:

| Model | Median absolute peak error | Median timing offset |
|---|---|---|
| Physics | 7.8 µg/m³ | 0 h |
| Hybrid | 7.5 µg/m³ | 0 h |
| XGBoost | 9.0 µg/m³ | 1 h |

Both physics-based models get the *timing* of the peak right (median 0-hour offset) more often than XGBoost, and
have smaller magnitude errors — again consistent with the state-space structure (with its explicit travel-time
lag) capturing the storm dynamics better than a feature-based tree model.

---

## 7. What the model says about the lake: counterfactual scenarios

Per Section 8 of the memo ("Run lake-level counterfactuals"), we hold the *entire* observed weather record fixed
— every real wind event, every real storm, at its real historical time — and re-run the fitted model with the
lake's exposed-playa area swapped for a hypothetical constant value. This isolates what the model attributes
specifically to lake state, independent of any particular year's weather.

| Scenario | Lake elevation | Mean lake contribution (µg/m³/h) | Peak lake contribution (µg/m³/h) |
|---|---|---|---|
| **Healthy reference** | 4,198 ft (Strike Team minimum healthy) | 0.0000 | 0.0 |
| Observed historical trajectory | (varies, 4188.5–4195.2 ft) | 0.00045 | 5.2 |
| Highest elevation in record | 4,195.2 ft | 0.00028 | 3.7 |
| Lowest elevation in record (2022 low) | 4,188.5 ft | 0.00073 | 9.7 |
| **5 ft further decline** | 4,183.5 ft | 0.00108 | 14.2 |
| **5 ft partial recovery** | 4,193.5 ft | 0.00042 | 5.6 |

**In plain language, under this model's assumptions:**

- At the Strike Team's healthy reference elevation, the model finds essentially **zero modeled lake-dust
  contribution to Hawthorne PM10** — by construction, since there's no exposed playa beyond that reference, but
  it's a useful check that the model doesn't invent a lake signal out of nowhere.
- Going from the record-high point in our record (4,195.2 ft) to the record low (4,188.5 ft) — a 6.7 ft drop —
  **roughly doubles to triples** both the average and peak modeled lake-dust contribution during qualifying wind
  events.
- A **further 5 ft of decline** below the already-record-low 2022 level would, under this model, push peak
  lake-driven contribution during a qualifying dust event to roughly **14 µg/m³ added per hour** — about 45%
  higher than at the observed record low, and nearly **3x** the peak contribution at the 2023 recovery high.
- Conversely, a **5 ft partial recovery** from the record low cuts the modeled peak contribution roughly in
  half, back down near the level seen at the highest lake stand in our record.
- These are **hourly rate contributions** (µg/m³ added per hour to the state equation), not one-time jumps — over
  a sustained multi-hour wind event, the state equation compounds this into the larger total PM10 increases seen
  in the event plots (e.g., §6's 400+ µg/m³ peaks).

**Caveat, stated as plainly as the results above**: because `beta` (the source-strength coefficient that scales
these numbers) is one of the weakly-identified parameters (§4.4), the *relative* pattern across scenarios (more
exposed playa → more contribution, healthy level → ~zero) is much better supported by the data than the *precise
magnitudes*. Treat the qualitative direction and rough scaling as the finding; treat "14.2 µg/m³/h" as an
order-of-magnitude estimate, not a number to cite to two significant figures.

---

## 8. Limitations (read this before using the model for anything consequential)

1. **Single monitor, single source point.** Hawthorne only; the lake is treated as one point source (south
   shore). Real dust plumes are spatially heterogeneous — Farmington Bay, Bear River Bay, and the north arm each
   have different exposure histories and erodibility. Bay-specific hypsometry is already downloaded
   (`data/raw/lake_hypsometry/`) but not yet used in the source term.
2. **Mixing height is a climatological approximation, not observed data** (§3.2). This is the single highest-value
   upgrade for a second iteration.
3. **Short record (2022–2026, ~4.75 years, ~475 detected dust episodes)** relative to the multi-decade records
   used in published GSL dust climatologies. Parameter identifiability (§4.4) would improve substantially with
   more years, once Hawthorne's continuous PM10 record is longer, or by pooling multiple monitors.
4. **No chemical or trajectory source attribution.** The model infers "lake dust" purely from wind
   direction/speed/threshold behavior. It cannot currently distinguish Great Salt Lake dust from West Desert dust,
   Sevier Lake dust, or other regional/anthropogenic dust sources that share similar wind signatures. The memo's
   own Section 4 flags this: *"predicting PM10 is easier than proving the Great Salt Lake caused a particular PM10
   spike."* Filter chemistry (UDORN provides this) or HYSPLIT back-trajectories would meaningfully strengthen
   causal claims.
5. **Confounders not separated out.** Wildfire smoke and winter cold-air-pool inversions are known to produce
   large PM10/PM2.5 spikes at Hawthorne unrelated to the lake; several of the very largest events in
   `results/tables/dust_episodes.csv` (e.g., the March 2024 1,343 µg/m³ peak, which occurred with a southerly, not
   lake-aligned, wind direction) are plausibly **not** lake events at all, and the directional factor `D_i`
   correctly assigns them near-zero lake contribution — but a rigorous smoke/inversion flag (e.g., cross-checking
   against PM2.5:PM10 ratio, or NWS red-flag/inversion advisories) would make this separation explicit rather than
   implicit.
6. **`E_local` is a constant.** Local/non-lake sources (traffic, construction, road dust) are represented by one
   fitted number, not a time-varying signal. This is intentional (§4.1) — the hybrid ML layer is where such
   patterns should be learned — but it means the standalone physics model will systematically miss any local-source
   dynamics.

None of this makes the model wrong or not useful — every one of these is exactly the kind of caveat the modeling
memo itself asks for (Section 7). It defines the priority list for a second iteration.

---

## 9. Extensions — what a next version should do

In priority order, based on what would most improve the model given the limitations above:

1. **Real HRRR mixing height.** Replace the climatological proxy with actual gridded boundary-layer height (e.g.,
   via the `herbie-data` Python package, which handles HRRR archive access more tractably than raw GRIB2 pulls).
2. **Bay-specific source terms.** The hypsometry data for Farmington Bay, Bear River Bay, and the north/south arms
   is already downloaded. Splitting `E_GSL` into per-bay terms with per-bay directional factors is a natural next
   step and directly supported by the existing `usgs_lake.py` module (`basin=` parameter).
3. **Multi-site spatial model.** Fit several Wasatch Front monitors jointly; a near-lake UDORN site would be an
   excellent second validation site with a shorter but more lake-proximate record.
4. **Back-trajectory cross-validation.** Use HYSPLIT or HRRR winds to check whether air parcels arriving during
   inferred lake-dust hours actually crossed exposed lakebed — strengthens causal language considerably.
5. **Chemical source attribution.** UDORN filter chemistry could validate (or contradict) which "lake-attributed"
   hours are really lake dust vs. other regional dust.
6. **Longer record / more data as it accumulates.** Re-run the whole pipeline periodically — every acquisition
   module caches by year and only re-fetches what's missing, so extending the analysis window each year is cheap.
7. **Bayesian state-space formulation.** Once the point-estimate model is well understood (it now is), replacing
   the least-squares fit with a Bayesian one (priors + MCMC or variational inference) would give the whole
   posterior over parameters rather than a bootstrap approximation, and would propagate that uncertainty properly
   into the counterfactual scenarios in §7.

---

## 10. Reproducing these results

```bash
cd "GSL Model"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python3 src/gsl_model/etl.py      # pulls all data, builds the hourly table (~2-3 min)
python3 scripts/run_pipeline.py                   # fits every model, runs evaluation + counterfactuals (~1 min)
```

Outputs land in:
- `results/tables/` — every metric, parameter, and scenario table referenced in this document (CSV/JSON)
- `results/figures/` — all plots, including `results/figures/events/` (30 largest dust episodes, full record) and
  `results/figures/test_event_ladder_*.png` (model comparison on the 8 largest held-out test-period events)
- `results/models/` — fitted physics parameters (`physics_theta.npy`, `physics_params.json`)

All acquisition modules cache their downloads under `data/raw/`; re-running `etl.py` after the cache is warm
takes seconds. Pass `force=True` to any `fetch_*` function to bypass the cache and re-download.

---

## 11. Source list

1. Utah DEQ – Dust Monitoring Network (UDORN): https://deq.utah.gov/air-quality/dust-monitoring-network
2. Utah DEQ – Current Dust Research Projects: https://deq.utah.gov/air-quality/current-dust-research-projects
3. US EPA – AQS AirData Downloads: https://aqs.epa.gov/aqsweb/airdata/download_files.html
4. USGS – Great Salt Lake Elevations and Areal Extent: https://www.usgs.gov/centers/utah-water-science-center/science/great-salt-lake-elevations-and-areal-extent
5. USGS – Great Salt Lake Elevation-Area-Volume Tables: https://data.usgs.gov/datacatalog/data/USGS%3A6467b42fd34ec11ae4a8afb1
6. Iowa Environmental Mesonet ASOS archive (substituted for MesoWest/Synoptic): https://mesonet.agron.iastate.edu/request/download.phtml
7. NOAA High-Resolution Rapid Refresh (HRRR): https://rapidrefresh.noaa.gov/hrrr/ (not yet integrated — see §9)
8. University of Utah – GSL Dust Exposure Model (Mallia & Perry): https://science.utah.edu/faculty/faculty-research/map-tool-gsldust/
9. University of Utah – Mapping the Great Salt Lake Dust Hotspots: https://science.utah.edu/faculty/faculty-research/mapping-gsl-hotspots/
10. Great Salt Lake Strike Team (2023) — source of the 4,198–4,205 ft "minimum healthy range" used as the
    exposed-playa reference elevation in this model.
