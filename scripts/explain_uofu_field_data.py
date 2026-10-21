"""One labeled figure summarizing the University of Utah field data (Perry, Crosman & Hoch, 2019):
the 2016-18 lakebed surface survey (Appendix C) and PM10 arsenic (Appendix G)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyproj import Transformer
from gsl_dust.config import DATA_INTERIM, DATA_RAW, FIGURES
from gsl_dust.grid import LATTICE

plt.rcParams.update({"font.family": "Helvetica", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
NAVY, RUST, RED, TAN, GREEN, GREY = "#16324a", "#b5651d", "#c0392b", "#d9b77e", "#5b8c3a", "#9aa3a8"
NAMES = {"DU1": "South shore (Saltair)", "DU2": "Antelope Is. / W. Farmington", "DU3": "Farmington Bay (east)",
         "DU4": "Southwest shore", "DU5": "Bear River Bay", "DU6": "West shore, south arm", "DU7": "Far west shore",
         "DU8": "Ogden Bay", "DU9": "Northwest (Gunnison Bay)", "DU10": "North shore, north arm"}
AREA_MI2 = {"DU1": 15.2, "DU2": 56.2, "DU3": 43.9, "DU4": 54.9, "DU5": 87.6, "DU6": 63.1, "DU7": 66.1, "DU8": 28.5,
            "DU9": 257.1, "DU10": 85.0}  # report Table 2.1


def main():
    c = pd.read_csv(DATA_RAW / 'fieldflux' / 'perry2019_appendixC_surface.csv'); c['du'] = c.su.str.extract(r'(DU\d+)')
    c['hot'] = (c.NV == 1) & (c.VF == 1) & ((c.NC == 1) | (c.ESC == 1))
    c['veg'] = (c.V == 1) | (c.SV == 1)
    tr = Transformer.from_crs('EPSG:4326', 'EPSG:26912', always_xy=True)
    c['x'], c['y'] = tr.transform(c.lon.values, c.lat.values)
    a = pd.read_csv(DATA_RAW / 'fieldflux' / 'perry2019_appendixG_pm10_arsenic.csv'); a['du'] = a.site.str.extract(r'(DU\d+)')
    t = np.load(DATA_INTERIM / 'arms_20m.npy', mmap_mode='r'); lake = np.asarray(t[::10, ::10]) > 0
    x0 = LATTICE.x0 / 1e3; y1 = LATTICE.y1 / 1e3; ext = (x0, x0 + t.shape[1] * 0.02, y1 - t.shape[0] * 0.02, y1)

    fig = plt.figure(figsize=(12, 9.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], hspace=0.42, wspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(np.where(lake, 1, np.nan), extent=ext, cmap=matplotlib.colors.ListedColormap(['#eef1f4']), interpolation='nearest')
    other = c[~c.hot & ~c.veg]
    ax.scatter(other.x / 1e3, other.y / 1e3, s=2, color=TAN, label='crusted or no visible fines (protected)')
    ax.scatter(c[c.veg].x / 1e3, c[c.veg].y / 1e3, s=2, color=GREEN, label='vegetated')
    ax.scatter(c[c.hot].x / 1e3, c[c.hot].y / 1e3, s=5, color=RED, label='dust hot spot (bare, loose fines, no protective crust)')
    for du, (x, y) in (c.groupby('du')[['x', 'y']].mean() / 1e3).iterrows():
        ax.text(x, y, du[2:], fontsize=9, fontweight='bold', color=NAVY, ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.25', fc='white', ec=NAVY, lw=0.8))
    ax.set(xlim=(318, 432), ylim=(4498, 4626), aspect='equal', xlabel='Easting (km, UTM 12N)', ylabel='Northing (km)',
           title="(a) 5,300 field observations, 2016–18\ngrey = lakebed; circled = decision unit")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fontsize=7.5, markerscale=3, frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(True)

    ax = fig.add_subplot(gs[0, 1])
    g = c.groupby('du').hot.mean().rename('share').to_frame()
    g['area'] = [AREA_MI2[d] * 2.59 * g.loc[d, 'share'] for d in g.index]
    g = g.sort_values('area'); y = np.arange(len(g))
    ax.barh(y, g.area, color=[RED if v > 15 else '#e2a59c' for v in g.area])
    for i, (d, r) in enumerate(g.iterrows()):
        ax.text(r.area + 0.8, i, f"{r.area:.1f} km²  ({r.share * 100:.0f}% of the unit)" if r.area < 1 else
                f"{r.area:.0f} km²  ({r.share * 100:.0f}% of the unit)", va='center', fontsize=7.5)
    ax.set(yticks=y, yticklabels=[f"{NAMES[d]} ({d[2:]})" for d in g.index], xlabel='Dust hot-spot area (km²)',
           title='(b) Where the dust-prone ground is', xlim=(0, 85))

    ax = fig.add_subplot(gs[1, 1])
    order = a.groupby('du').pm10_as_ppm.median().sort_values().index
    for i, d in enumerate(order):
        v = a[a.du == d].pm10_as_ppm
        ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), s=12, color=RUST, alpha=0.75)
        ax.plot([v.median()] * 2, [i - 0.3, i + 0.3], color=NAVY, lw=2)
    l1 = ax.axvline(18.5, color=GREY, ls='--', lw=1.2, label='arsenic in lakebed sediment (≈ 18.5 mg/kg, USGS)')
    l2 = ax.axvline(3.0, color=RED, ls=':', lw=1.4, label='EPA industrial soil screening level (3.0 mg/kg)')
    ax.legend(handles=[l1, l2], loc='upper center', bbox_to_anchor=(0.45, -0.13), fontsize=7.5, frameon=False)
    ax.set(yticks=range(len(order)), yticklabels=[f"{NAMES[d]} ({d[2:]})" for d in order], xlim=(0, 90),
           ylim=(-0.7, len(order) - 0.2), xlabel='Arsenic in PM10-size dust (mg/kg); dots = sites, bar = median',
           title='(c) Arsenic in the dust each area would emit')
    fig.savefig(FIGURES / 'uofu_field_data_explained.png', dpi=170, bbox_inches='tight')


if __name__ == '__main__':
    main()
