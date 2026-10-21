"""Map of the 1 km model grid: lakebed cells by arm, shoreline cells, and the
10 km HYSPLIT source zones, with a zoom on Farmington Bay."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from gsl_dust.exposure import CellHypsometry
from gsl_dust.config import DATA_INTERIM, FIGURES
from gsl_dust.grid import LATTICE
from gsl_dust import boundaries

plt.rcParams.update({"font.family": "Helvetica", "font.size": 9})
h = CellHypsometry.load(DATA_INTERIM / 'cell_hypsometry.npz')
arm = np.argmax(h.n_lakebed, axis=1)
full = h.lakebed_fraction > 0.999
counts = {k: int(m.sum()) for k, m in {'nf': (arm == 0) & full, 'ns': (arm == 0) & ~full,
                                        'sf': (arm == 1) & full, 'ss': (arm == 1) & ~full}.items()}
cat = np.full(LATTICE.shape, np.nan)
cat.ravel()[h.cells] = np.where(arm == 0, np.where(full, 0, 1), np.where(full, 2, 3))
x0, y1 = LATTICE.x0 / 1e3, LATTICE.y1 / 1e3
ext = (x0, x0 + LATTICE.nx, y1 - LATTICE.ny, y1)
cmap = ListedColormap(['#2f6ea6', '#9cc0e0', '#c27c2c', '#ecc48f'])
county = boundaries.salt_lake_county_utm()
polys = county.geoms if county.geom_type == 'MultiPolygon' else [county]

fig, ax = plt.subplots(1, 2, figsize=(13, 7.2), gridspec_kw={'width_ratios': [1, 1.05]})
a = ax[0]
a.imshow(cat, extent=ext, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation='nearest')
a.add_patch(Rectangle((ext[0], ext[2]), LATTICE.nx, LATTICE.ny, fill=False, ec='black', lw=1.2))
for g in polys:
    xs, ys = g.exterior.xy
    a.plot(np.array(xs) / 1e3, np.array(ys) / 1e3, color='#c0392b', lw=1.2)
a.text(432, 4482, 'Salt Lake\nCounty', color='#c0392b', fontsize=8.5, ha='center')
for v in np.arange(320, 431, 10):
    a.axvline(v, color='#666', lw=0.4, alpha=0.5)
for v in np.arange(4500, 4625, 10):
    a.axhline(v, color='#666', lw=0.4, alpha=0.5)
a.add_patch(Rectangle((395, 4515), 25, 30, fill=False, ec='black', lw=1.6, ls='--'))
a.text(407.5, 4547, 'zoom area (b)', ha='center', fontsize=8)
a.set(xlim=(315, 450), ylim=(4470, 4628), aspect='equal', xlabel='Easting (km, UTM 12N)', ylabel='Northing (km)',
      title=f'(a) The 1 km model grid: {LATTICE.nx} × {LATTICE.ny} = {LATTICE.nx * LATTICE.ny:,} cells,\n'
            f'{len(h.cells):,} of them lakebed (thin grey lines = 10 km HYSPLIT zones)')
a.legend(handles=[Patch(color='#2f6ea6', label=f"north arm, fully lakebed ({counts['nf']:,} cells)"),
                  Patch(color='#9cc0e0', label=f"north arm, shoreline ({counts['ns']:,} cells)"),
                  Patch(color='#c27c2c', label=f"south arm + bays, fully lakebed ({counts['sf']:,} cells)"),
                  Patch(color='#ecc48f', label=f"south arm + bays, shoreline ({counts['ss']:,} cells)")],
         loc='lower left', fontsize=7.5, framealpha=0.95)
b = ax[1]
b.imshow(cat, extent=ext, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation='nearest')
for v in np.arange(395, 421, 1):
    b.axvline(v, color='white', lw=0.5)
for v in np.arange(4515, 4546, 1):
    b.axhline(v, color='white', lw=0.5)
for v in np.arange(400, 421, 10):
    b.axvline(v, color='black', lw=1.6)
for v in np.arange(4520, 4546, 10):
    b.axhline(v, color='black', lw=1.6)
for g in polys:
    xs, ys = g.exterior.xy
    b.plot(np.array(xs) / 1e3, np.array(ys) / 1e3, color='#c0392b', lw=1.6)
b.set(xlim=(395, 420), ylim=(4515, 4545), aspect='equal', xlabel='Easting (km)', ylabel='Northing (km)',
      title='(b) Zoom on Farmington Bay: each small square is one 1 km cell;\nthick black lines = 10 km HYSPLIT source zones')
b.text(397, 4517, 'each white-outlined square = 1 km × 1 km\n(up to 100 cells per HYSPLIT zone)', fontsize=8,
       bbox=dict(fc='white', ec='none', alpha=0.85))
fig.tight_layout()
fig.savefig(FIGURES / 'model_grid_map.png', dpi=170, bbox_inches='tight')
