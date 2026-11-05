"""Plot P95 twCRPSS and BSS maps alongside lead-time skill curves."""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.gridspec import GridSpec
from netCDF4 import Dataset as NC4Dataset
import geopandas as gpd
import cartopy.io.shapereader as shpreader

# ==============================================================
# CONFIGURATION
# ==============================================================
BASE = "/cw3e/mead/projects/cwp190/projects/post-processing"

LEAD_TIME = 72
PERCENTILE = 95

# The twCRPSS column is intentionally fixed at P95.
TWCRPSS_COL = "twCRPSS_P95"

MAP_DIR = os.path.join(BASE, f"output/v21/map/lead_time_{LEAD_TIME}/p95")
QPE_ROOT = BASE + "/data/CNRFC/QPE-from-official-website/QPE-with-lat-lon-R"

MAP_SOURCES = ["WestWRF", "GEFS", "SS"]
MAP_SOURCE_LABELS = {"WestWRF": "West-WRF", "GEFS": "GEFS", "SS": "DL+SS"}

LINE_SOURCES = ["WestWRF", "GEFS", "DL", "SS"]
LINE_LABELS = {"WestWRF": "West-WRF", "GEFS": "GEFS", "DL": "DL", "SS": "DL+SS"}
LINE_COLORS = {"WestWRF": "#1f77b4", "GEFS": "#ff7f0e", "DL": "#2ca02c", "SS": "#d62728"}
LINE_MARKERS = {"WestWRF": "o", "GEFS": "s", "DL": "^", "SS": "D"}

ALL_LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72,
                   78, 84, 90, 96, 102, 108, 114, 120, 126, 132,
                   138, 144, 150, 156, 162, 168]

METRICS_CSV_TMPL = {
    "WestWRF": BASE + "/output/v17/v17.3/westwrf/lead_time_{lt}/evaluation_metrics.csv",
    "GEFS":    BASE + "/output/v19/gefs/lead_time_{lt}/evaluation_metrics.csv",
    "SS":      BASE + "/output/v17/v17.0/lead_time_{lt}/ss/plot/evaluation_metrics.csv",
    "DL":      BASE + "/output/v21/DL-out-3/lead_time_{lt}/evaluation_metrics.csv",
}

METRICS_CSV_TWCRPSS_TMPL = {
    "WestWRF": BASE + "/output/v17/v17.3/westwrf/lead_time_{lt}/evaluation_metrics_twcrpss.csv",
    "GEFS":    BASE + "/output/v19/gefs/lead_time_{lt}/evaluation_metrics_twcrpss.csv",
    "SS":      BASE + "/output/v17/v17.0/lead_time_{lt}/ss/plot/evaluation_metrics_twcrpss.csv",
    "DL":      BASE + "/output/v21/DL-out-3/lead_time_{lt}/evaluation_metrics_twcrpss.csv",
}

OUT_PNG = os.path.join(BASE, "output/v21", f"figure2_p95.png")

# ==============================================================
# LOAD METRIC MAPS AND SPATIAL GRID
# ==============================================================
def load_metric(source, metric):
    path = os.path.join(MAP_DIR, f"{source}_p{PERCENTILE}_{metric}.npz")
    d = np.load(path)
    return d[metric]

twcrpss_maps = {s: load_metric(s, "twCRPSS") for s in MAP_SOURCES}
bss_maps     = {s: load_metric(s, "BSS")     for s in MAP_SOURCES}

obs_pack = np.load(os.path.join(MAP_DIR, f"obs_p{PERCENTILE}_map.npz"))
lat_idx0, lat_idx1, lon_idx0, lon_idx1 = obs_pack["index_range"].tolist()

qpe_files = sorted(glob.glob(os.path.join(QPE_ROOT, "qpe.*.wgs84.nc")))
assert len(qpe_files) > 0, f"No QPE files found under {QPE_ROOT}"
nc = NC4Dataset(qpe_files[0], "r")
lat_full = np.array(nc.variables["lat"][:])
lon_full = np.array(nc.variables["lon"][:])
nc.close()

lat_sub = lat_full[lat_idx0:lat_idx1 + 1]
lon_sub = lon_full[lon_idx0:lon_idx1 + 1]

if lat_sub[0] > lat_sub[-1]:
    lat_sub = lat_sub[::-1]
    for d in (twcrpss_maps, bss_maps):
        for s in MAP_SOURCES:
            d[s] = d[s][::-1, :]
if lon_sub[0] > lon_sub[-1]:
    lon_sub = lon_sub[::-1]
    for d in (twcrpss_maps, bss_maps):
        for s in MAP_SOURCES:
            d[s] = d[s][:, ::-1]

dlat = float(np.abs(np.mean(np.diff(lat_sub))))
dlon = float(np.abs(np.mean(np.diff(lon_sub))))
extent = [lon_sub.min(), lon_sub.max(), lat_sub.min(), lat_sub.max()]

states_shp = shpreader.natural_earth(
    resolution="50m", category="cultural", name="admin_1_states_provinces_lakes"
)
states_reader = shpreader.Reader(states_shp)
us_state_geoms = [
    r.geometry for r in states_reader.records() if r.attributes.get("admin") == "United States of America"
]
gdf_states = gpd.GeoDataFrame(geometry=us_state_geoms, crs="EPSG:4326")

countries_shp = shpreader.natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
countries_reader = shpreader.Reader(countries_shp)
us_country_geom = next(
    r.geometry for r in countries_reader.records() if r.attributes.get("NAME") == "United States of America"
)

# ==============================================================
# BUILD GEOGRAPHIC MASK
# ==============================================================
def build_us_mask(country_geom, lon_1d, lat_1d):
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    inside = np.zeros(lon_2d.shape, dtype=bool)
    polys = list(country_geom.geoms) if country_geom.geom_type == "MultiPolygon" else [country_geom]
    for poly in polys:
        pminx, pminy, pmaxx, pmaxy = poly.bounds
        bbox_mask = (lon_2d >= pminx) & (lon_2d <= pmaxx) & (lat_2d >= pminy) & (lat_2d <= pmaxy)
        if not bbox_mask.any():
            continue
        idx = np.where(bbox_mask)
        pts = np.column_stack([lon_2d[idx], lat_2d[idx]])
        ext_path = MplPath(np.array(poly.exterior.coords))
        pt_inside = ext_path.contains_points(pts)
        for interior in poly.interiors:
            hole_path = MplPath(np.array(interior.coords))
            pt_inside &= ~hole_path.contains_points(pts)
        inside[idx] |= pt_inside
    return inside

us_mask = build_us_mask(us_country_geom, lon_sub, lat_sub)
for d in (twcrpss_maps, bss_maps):
    for s in MAP_SOURCES:
        d[s] = np.where(us_mask, d[s], np.nan)

combined_finite = np.zeros(us_mask.shape, dtype=bool)
for d in (twcrpss_maps, bss_maps):
    for s in MAP_SOURCES:
        combined_finite |= np.isfinite(d[s])

valid_rows = np.where(combined_finite.any(axis=1))[0]
valid_cols = np.where(combined_finite.any(axis=0))[0]
BUFFER_DEG = 0.15
lat_lo = lat_sub[valid_rows[0]] - BUFFER_DEG
lat_hi = 44.0
lon_lo = lon_sub[valid_cols[0]] - BUFFER_DEG
lon_hi = lon_sub[valid_cols[-1]] + BUFFER_DEG

# ==============================================================
# LOAD LEAD-TIME VERIFICATION METRICS
# ==============================================================
def load_metrics_all_leadtimes(csv_template_dict, source):
    pattern = csv_template_dict[source]
    rows = []
    for lt in ALL_LEAD_TIMES:
        path = pattern.format(lt=lt)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["lead_time"] = lt
            rows.append(df)
        else:
            print(f"  [MISSING {source}] lead_time={lt}: {path}")
    if rows:
        return pd.concat(rows, ignore_index=True).sort_values("lead_time")
    return pd.DataFrame()

print("Loading BSS metrics CSVs (evaluation_metrics.csv, P95 column) ...")
bss_line_metrics = {s: load_metrics_all_leadtimes(METRICS_CSV_TMPL, s) for s in LINE_SOURCES}
for s in LINE_SOURCES:
    print(f"  {s}: {len(bss_line_metrics[s])} rows")

print("Loading twCRPSS metrics CSVs (evaluation_metrics_twcrpss.csv) ...")
twcrpss_line_metrics = {s: load_metrics_all_leadtimes(METRICS_CSV_TWCRPSS_TMPL, s) for s in LINE_SOURCES}
for s in LINE_SOURCES:
    print(f"  {s}: {len(twcrpss_line_metrics[s])} rows")

from matplotlib.gridspec import GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ==============================================================
# CREATE COMPOSITE FIGURE
# ==============================================================
fig = plt.figure(figsize=(20, 8))
outer = GridSpec(1, 2, width_ratios=[3, 2], wspace=0.18, figure=fig)

maps_gs = GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[0], wspace=0.08, hspace=0.15)
lines_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.35)

map_axes = np.empty((2, 3), dtype=object)
for r in range(2):
    for c in range(3):
        map_axes[r, c] = fig.add_subplot(maps_gs[r, c])

line_axes = [fig.add_subplot(lines_gs[0, 0]), fig.add_subplot(lines_gs[1, 0])]

PANEL_LABELS = list("abcdefgh")

def annotate_panel(ax, label):
    ax.text(-0.05, 1.08, f"({label})", transform=ax.transAxes,
             fontsize=18, va="top", ha="left")

for j, s in enumerate(MAP_SOURCES):
    ax = map_axes[0, j]
    m = np.ma.masked_invalid(twcrpss_maps[s])
    im = ax.imshow(m, origin="lower", extent=extent, aspect="auto",
                    cmap="bwr", vmin=-1, vmax=1)
    gdf_states.boundary.plot(ax=ax, color="black", linewidth=0.6)
    ax.set_xlim(lon_lo, lon_hi)
    ax.set_ylim(lat_lo, lat_hi)
    ax.set_title(MAP_SOURCE_LABELS[s], fontsize=18)
    if j == 0:
        ax.set_ylabel("Latitude", fontsize=18)
    dom_mean = np.nanmean(twcrpss_maps[s])
    ax.text(0.03, 0.03, f"twCRPSS = {dom_mean:.3f}", transform=ax.transAxes,
            fontsize=12, va="bottom", ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    ax.tick_params(axis="both", labelsize=14)
    annotate_panel(ax, PANEL_LABELS[j])

divider = make_axes_locatable(map_axes[0, 2])
cax0 = divider.append_axes("right", size="5%", pad=0.08)
cbar0 = fig.colorbar(im, cax=cax0)
cbar0.set_label("twCRPSS", fontsize=18)
cbar0.ax.tick_params(labelsize=14)

for j, s in enumerate(MAP_SOURCES):
    ax = map_axes[1, j]
    m = np.ma.masked_invalid(bss_maps[s])
    im2 = ax.imshow(m, origin="lower", extent=extent, aspect="auto",
                     cmap="bwr", vmin=-1, vmax=1)
    gdf_states.boundary.plot(ax=ax, color="black", linewidth=0.6)
    ax.set_xlim(lon_lo, lon_hi)
    ax.set_ylim(lat_lo, lat_hi)
    if j == 0:
        ax.set_ylabel("Latitude", fontsize=18)
    dom_mean = np.nanmean(bss_maps[s])
    ax.text(0.03, 0.03, f"BSS = {dom_mean:.3f}", transform=ax.transAxes,
            fontsize=12, va="bottom", ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    ax.set_xlabel("Longitude", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    annotate_panel(ax, PANEL_LABELS[3 + j])

divider2 = make_axes_locatable(map_axes[1, 2])
cax1 = divider2.append_axes("right", size="5%", pad=0.08)
cbar1 = fig.colorbar(im2, cax=cax1)
cbar1.set_label("BSS", fontsize=18)
cbar1.ax.tick_params(labelsize=14)

ax = line_axes[0]
for s in LINE_SOURCES:
    df = twcrpss_line_metrics[s]
    if len(df) == 0 or TWCRPSS_COL not in df.columns:
        continue
    x, y = df["lead_time"].values, df[TWCRPSS_COL].values.astype(float)
    mask = np.isfinite(y)
    ax.plot(x[mask], y[mask], marker=LINE_MARKERS[s], color=LINE_COLORS[s],
            label=LINE_LABELS[s], linewidth=1.5, markersize=4)
ax.set_xlabel("Lead time (h)", fontsize=18)
ax.set_ylabel("twCRPSS (P95)", fontsize=18)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.set_title("twCRPSS (P95) vs Lead Time", fontsize=18)
ax.grid(True, linestyle="--", alpha=0.4)
ax.tick_params(axis="both", labelsize=14)
ax.legend(fontsize=12, frameon=False)
annotate_panel(ax, PANEL_LABELS[6])

ax = line_axes[1]
bss_col = f"BSS_P{PERCENTILE}"
for s in LINE_SOURCES:
    df = bss_line_metrics[s]
    if len(df) == 0 or bss_col not in df.columns:
        continue
    x, y = df["lead_time"].values, df[bss_col].values.astype(float)
    mask = np.isfinite(y)
    ax.plot(x[mask], y[mask], marker=LINE_MARKERS[s], color=LINE_COLORS[s],
            label=LINE_LABELS[s], linewidth=1.5, markersize=4)
ax.set_xlabel("Lead time (h)", fontsize=18)
ax.set_ylabel(f"BSS (P{PERCENTILE})", fontsize=18)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.set_title(f"Brier Skill Score (P{PERCENTILE}) vs Lead Time", fontsize=18)
ax.grid(True, linestyle="--", alpha=0.4)
ax.tick_params(axis="both", labelsize=14)
ax.legend(fontsize=12, frameon=False)
annotate_panel(ax, PANEL_LABELS[7])

plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print(f"\nSaved: {OUT_PNG}")