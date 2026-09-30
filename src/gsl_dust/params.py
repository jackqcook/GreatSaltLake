"""Every physical parameter with a central value, an uncertainty range, and where
it comes from. `interim=True` marks values that stand in for Great Salt Lake
field data the guide calls for but that has not been obtained yet; they are the
first things to replace. Ranges feed the Section 8 uncertainty ensemble."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Param:
    value: float
    low: float
    high: float
    units: str
    source: str
    interim: bool = False
    log: bool = False  # sample log-uniformly


P = {
    # --- friction velocity over exposed lakebed (Stage 1.3) -------------------
    "z0_playa": Param(3e-4, 1e-4, 1e-3, "m",
        "Aerodynamic roughness of smooth playa (typical range). Used because HRRR's static "
        "land-surface classes give the lakebed 0 (water) or 5-7 cm (land) roughness.", interim=True, log=True),

    # --- threshold law (Stage 2.2): guide says calibrate against field fluxes -
    "u_star_t": Param(0.36, 0.24, 0.56, "m/s",
        "PM10 threshold friction velocity, PI-SWERL, Lordsburg Playa NM (Van Pelt et al. 2020, "
        "PMC7728642): 0.24-0.56 m/s across delta/lake/beach surfaces. Not GSL: replace with GSL "
        "PI-SWERL (e.g., Powell, Univ. of Utah MS thesis).", interim=True),
    "p": Param(3.0, 2.0, 4.0, "-",
        "Wind-sensitivity exponent. No GSL flux-vs-u* series yet to fit it.", interim=True),
    "flux_ref": Param(3e-6, 5.6e-7, 2.5e-5, "kg m-2 s-1",
        "PM10 flux from emitting surface at u*_ref with a full reservoir. Range = PI-SWERL maximum "
        "fluxes at Lordsburg Playa, lake crust (0.56 mg/m2/s) to disturbed delta (25 mg/m2/s).",
        interim=True, log=True),
    "u_star_ref": Param(0.8, 0.7, 0.9, "m/s",
        "Friction velocity of the PI-SWERL top step at which flux_ref applies (approximate).", interim=True),

    # --- sediment reservoir (Stage 2.3): guide says independent estimates -----
    "s0_areal": Param(0.026, 0.005, 0.1, "kg m-2",
        "Initial PM10-emittable sediment per lakebed area: ~1 mm loose layer x 1300 kg/m3 x 2% PM10.",
        interim=True, log=True),
    "g_areal": Param(0.026, 0.0, 0.1, "kg m-2 yr-1",
        "Replenishment per lakebed area (crust breakdown, redeposition). Unconstrained; central value "
        "renews the reservoir about once a year.", interim=True),

    # --- erodibility phi, physics branch (Stage 2.1) --------------------------
    "phi_surface": Param(0.09, 0.05, 0.20, "-",
        "Fraction of exposed lakebed that emits (crust disturbed/absent). Univ. of Utah hotspot "
        "mapping: ~9% (175 km2) of exposed lakebed emits."),
    "wet_depth_mm": Param(2.0, 1.0, 5.0, "mm",
        "Effective recent rain that holds the surface above the 4% soil-moisture shutoff found by "
        "PI-SWERL on GSL playa.", interim=True),
    "dry_tau_days": Param(3.0, 1.0, 7.0, "days", "Surface drying e-folding time after rain.", interim=True),
    "fringe_m": Param(0.3, 0.1, 0.6, "m",
        "Height above the water line kept too moist to emit by capillary rise.", interim=True),
    "snow_m": Param(0.01, 0.005, 0.02, "m", "Snow depth above which the surface cannot emit."),

    # --- chemistry (Stage 2.4) -------------------------------------------------
    "eta": Param(1.3, 0.8, 2.0, "-",
        "Arsenic enrichment in airborne PM10 relative to <63 um sediment. Bounded, as the guide allows "
        "(no PM10-size-matched samples exist). USGS 2018-19 sites with both fractions show <63 um / "
        ">=63 um arsenic ratios of 0.84-1.33 (median 1.31); the upper bound allows further enrichment "
        "below 63 um."),

    # --- ODE prototype transfer (Stage 3.1) ------------------------------------
    "v_dep": Param(0.01, 0.003, 0.03, "m/s", "PM10 dry deposition velocity (typical range)."),
    "mix_depth": Param(1000.0, 500.0, 2000.0, "m", "Mixed-layer depth for the deposition rate."),
}


def central():
    return {k: v.value for k, v in P.items()}


def interim_names():
    return sorted(k for k, v in P.items() if v.interim)
