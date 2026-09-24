# Great Salt Lake Dust & Air Quality Model

A physics-informed, data-driven model of Great Salt Lake dust's contribution to PM10 at Salt Lake City's
Hawthorne monitor, plus ML baselines and a hybrid physics+ML model, following the direction in
`Great_Salt_Lake_Air_Quality_Modeling_Direction.pdf`.

**Start here: [docs/MODEL_EXPLAINER.md](docs/MODEL_EXPLAINER.md)** — a full write-up of the data, the model,
the results, what they mean for the lake, the limitations, and next steps.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python3 src/gsl_model/etl.py   # pull data + build hourly analysis table
python3 scripts/run_pipeline.py                # fit all models, evaluate, run counterfactuals
```

## Layout

```
src/gsl_model/
  config.py              site definitions, paths, physical reference constants
  acquisition/            data pulls (EPA AQS PM10, ASOS weather, USGS lake state, mixing-height proxy)
  etl.py                  builds the hourly analysis table
  physics_model.py        the state-space dust/PM10 model (fit, predict, simulate)
  baselines.py             persistence / GAM / XGBoost baselines
  hybrid.py                physics + ML-residual hybrid model
  event_model.py           two-stage event-probability / magnitude model
  evaluate.py               splits, metrics, physical sanity checks
  uncertainty.py            block-bootstrap parameter uncertainty
  counterfactual.py         lake-level scenario analysis
  plotting.py                event & comparison figures

scripts/run_pipeline.py   end-to-end: data -> fit everything -> evaluate -> counterfactuals -> figures
data/                      raw (cached downloads) and processed (analysis table) data
results/                   tables, figures, and fitted model parameters from the last pipeline run
docs/MODEL_EXPLAINER.md    the full write-up
```

## Notes

- All data sources are free and need no API key (MesoWest/Synoptic was substituted with the equivalent free
  Iowa Environmental Mesonet ASOS archive; see the write-up for why).
- Every `fetch_*` function in `acquisition/` caches to `data/raw/`; re-runs are fast.
- The analysis window is 2022–2026 (Hawthorne's continuous hourly PM10 record starts in 2022) — see
  docs/MODEL_EXPLAINER.md §3.5 for why.
