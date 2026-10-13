# Great Salt Lake dust model

Predicts PM10 dust and its arsenic moving from Great Salt Lake sediment into the Salt Lake Valley. Outputs are arrival mass and
arrival rate; the model stops before health effects.

The modeling standard is [`docs/GSL_Dust_Model_Guide.pdf`](docs/GSL_Dust_Model_Guide.pdf). The stage-by-stage plan built on it is
[`docs/GSL_Dust_Model_Implementation_Plan.pdf`](docs/GSL_Dust_Model_Implementation_Plan.pdf).

## Layout (by the guide's stages)

| Stage | Module | What it does |
| --- | --- | --- |
| 1 | `lake_levels.py`, `datums.py` | Daily lake level by arm (USGS 10010000, 10010100), NGVD29 ft to NAVD88 |
| 1 | `grid.py`, `boundaries.py` | 1 km lattice (NAD83 / UTM 12N); valley receptor; railroad causeway (arm divide) |
| 1 | `exposure.py` | Terrain, arms, per-cell hypsometry, X_i(t), check against USGS area tables |
| 1 | `hrrr.py` | Hourly HRRR inputs on the lattice (winds rotated to true north) |
| 2 | `emission.py`, `erodibility.py`, `chemistry.py` | Threshold law and K calibration; physics phi; arsenic kriging and eta bounds |
| 3.1 | `ode.py`, `transfer.py` | Exact hourly solution of S -> I -> R, R_As; documented kV, kL |
| 4 | `validation.py`, `validation/heldout_design.json` | AQS PM10, regional episodes, held-out design (fixed) |
| 1-5 | `model.py` | The chain, month by month; scenarios reuse identical weather |
| all | `params.py`, `decisions.py` | Parameter values/ranges/sources; the user's open decisions (provisional) |

## Running

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" scikit-learn
python -m pytest                      # unit and chain tests
python scripts/fetch_hrrr.py          # hourly HRRR for the simulation period (~2 h, resumable)
python scripts/stage1_exposure.py     # needs the USGS terrain files in data/raw/tbdem/
python scripts/run_model.py           # central run, lake-level scenarios, uncertainty ensembles
```

The terrain (USGS topobathymetric model, ScienceBase item `646d0ed2d34ee02593fb50a7`) needs a ScienceBase sign-in to
download. Put `Great_Salt_Lake_TBDEM.tif.ovr`, `.tfw`, `.tif.aux.xml` and `.tif.xml` in `data/raw/tbdem/`.

## Status

- Stages 1, 2 and 3.1 are implemented and tested. Stage 3.2 (HYSPLIT driven by HRRR) and the Stage 4 comparison wait on a
  HYSPLIT install.
- Values marked `interim=True` in `params.py` stand in for Great Salt Lake field data (PI-SWERL fluxes) that hasn't been
  obtained. Replace them first.
- Choices in `decisions.py` are provisional until confirmed.
