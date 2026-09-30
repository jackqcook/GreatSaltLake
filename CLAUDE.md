# Great Salt Lake dust model

## Modeling standard
`docs/GSL_Dust_Model_Guide.pdf` is the modeling standard for this project. All modeling work follows it: its equations, compartments (S, I, R, R_As in kg), outputs (arrival mass and rate; stop before health effects), component methods, assumptions to test, and build order.

- Do not substitute methods, re-phase the build order, or promote fallbacks to defaults. The only alternatives allowed are the ones the guide names ("otherwise physics" for phi, "measured or bounded" for eta, "particle bins or uncertainty bounds" for size effects).
- When data can't support a guide method, report it as a data-readiness gap against that method.
- The implementation plan is `docs/GSL_Dust_Model_Implementation_Plan.pdf` (source: `docs/plan_assets/plan_source.html`, figures: `docs/plan_assets/make_figures.py`). Keep it consistent with the guide.

## Datums
USGS gauges 10010000 (south arm) and 10010100 (north arm) report parameter 62614 = NGVD29 feet. The USGS topobathymetric DEM is NAVD88 meters. NAVD88 = NGVD29 + 3.47585958 ft at Saltair (USGS/VERTCON). Convert before combining.

## History
The earlier single-monitor PM10 model (v1) is superseded; it remains in git history at commit 165a2a8 for reference only.
