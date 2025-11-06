"""
windOverlay.py
======================================
Fetches live wind data from data.gov.sg, processes it using 
Inverse Distance Weighting (IDW) to create a continuous wind grid,
and visualizes the results as a scalar heatmap and a quiver plot.

Outputs: 
- wind_grid_*.npz (for use in optimizationMain.py)
- wind_scalar_*.png
- wind_quiver_*.png
"""

import json
import time
import math
import datetime as dt
import numpy as np
import requests
import matplotlib.pyplot as plt
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon
from shapely import vectorized
from typing import Dict, Any, List, Tuple

# ---------------------------
# Config
# ---------------------------
GRID_SIZE_M = 400          # grid resolution for IDW interpolation/plotting (m)
IDW_POWER   = 2.0          # Inverse-distance weighting power (e.g., 2.0)
MIN_STATIONS = 8           # Require at least this many stations for interpolation

# Quiver plot configuration (arrows)
QUIVER_GRID_KM   = 3.0     # Target arrow spacing (one arrow every N km)
QUIVER_MIN_SPEED = 0.8     # Drop very small arrows (m/s)
QUIVER_WIDTH     = 0.002   # Arrow line width
QUIVER_ALPHA     = 0.95    # Arrow opacity
QUIVER_BG_HEATMAP = False  # Keep a faint heatmap behind arrows?

# Hardcoded Singapore bounds for interpolation (WGS84)
LON_MIN, LON_MAX = 103.60, 104.42
LAT_MIN, LAT_MAX = 1.15, 1.48
EXTENT = (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)

# ---------------------------
# Helpers
# ---------------------------

def biggest_polygon(geom: Any) -> Any:
    """Returns the largest component polygon if geom is a MultiPolygon."""
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom

def fetch_json(api_url: str) -> Dict[str, Any]:
    """Fetches JSON data from a URL with error handling."""
    try:
        r = requests.get(api_url, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"Error fetching data from {api_url}: {e}")
        return {}

def parse_wind_readings() -> Tuple[List[Tuple[str, float, float, float, float]], str]:
    """Fetches and parses wind speed and direction data."""
    js_s = fetch_json("https://api.data.gov.sg/v1/environment/wind-speed")
    js_d = fetch_json("https://api.data.gov.sg/v1/environment/wind-direction")
    
    stations_s = {item['station_id']: item['value'] for reading in js_s.get('items', []) for item in reading.get('readings', [])}
    stations_d = {item['station_id']: item['value'] for reading in js_d.get('items', []) for item in reading.get('readings', [])}
    metadata = {item['id']: {'lat': item['location']['latitude'], 'lon': item['location']['longitude']} for item in js_s.get('metadata', {}).get('stations', [])}
    
    timestamp = js_s.get('items', [{}])[0].get('timestamp', 'N/A')
    
    readings = []
    for sid, speed in stations_s.items():
        if sid in stations_d and sid in metadata:
            direction = stations_d[sid]
            lat = metadata[sid]['lat']
            lon = metadata[sid]['lon']
            readings.append((sid, lat, lon, speed, direction))
            
    return readings, timestamp

def _grid_latlon_mesh(minx: float, miny: float, maxx: float, maxy: float, cell_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a mesh grid of (LAT, LON) coordinates for interpolation."""
    # Approximate degrees per meter for small area
    lat0 = 0.5 * (miny + maxy)
    dlat = cell_m / 111320.0
    dlon = cell_m / (111320.0 * np.cos(np.deg2rad(lat0)))
    
    lats = np.arange(miny, maxy, dlat)
    lons = np.arange(minx, maxx, dlon)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    return LAT, LON

def idw_interpolate(xy: np.ndarray, values: np.ndarray, grid_lon: np.ndarray, grid_lat: np.ndarray, power: float = 2.0, eps: float = 1e-12) -> np.ndarray:
    """Performs Inverse Distance Weighting interpolation."""
    gx = grid_lon.ravel()
    gy = grid_lat.ravel()
    
    # Calculate distances (d^power)
    dx = gx[None, :] - xy[:, 0:1] # N x M
    dy = gy[None, :] - xy[:, 1:1] # N x M
    d = np.hypot(dx, dy)
    
    # Avoid division by zero for exact station points
    d_safe = np.where(d < eps, eps, d)
    weights = 1.0 / (d_safe ** power)

    # Normalize weights and apply to values
    sum_weights = weights.sum(axis=0)
    weights_norm = weights / sum_weights[None, :]
    
    interpolated = np.dot(values.T, weights_norm)

    return interpolated.reshape(grid_lon.shape)

# ---------------------------
# Wind Field Processing
# ---------------------------

def process_wind_field(readings: List[Tuple[str, float, float, float, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Interpolates wind data onto a grid and prepares it for visualization/export.
    Returns: lon, lat, u_grid, v_grid, timestamp
    """
    if len(readings) < MIN_STATIONS:
        raise ValueError(f"Not enough valid stations found ({len(readings)}). Minimum required: {MIN_STATIONS}")

    print(f"Interpolating from {len(readings)} valid stations...")
    
    # Extract data for interpolation
    station_data = np.array([(lon, lat, speed, direction) for _, lat, lon, speed, direction in readings])
    station_xy = station_data[:, 0:2] # (lon, lat)
    station_speed = station_data[:, 2] 
    station_dir_from_deg = station_data[:, 3] 
    
    # Convert speed/direction (from) to u/v (to) components
    # Angle is 'from North (0) clockwise', need 'to East (0) counter-clockwise' 
    # Dir FROM is wind source direction. Wind TO is wind direction vector.
    # Angle TO = (Dir FROM + 180) % 360
    angle_to_rad = np.deg2rad((station_dir_from_deg + 180) % 360)
    station_u = station_speed * np.sin(angle_to_rad) # U (East/West component)
    station_v = station_speed * np.cos(angle_to_rad) # V (North/South component)
    
    # Create target grid
    grid_lat, grid_lon = _grid_latlon_mesh(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, GRID_SIZE_M)
    
    # Interpolate U and V components separately
    u_grid = idw_interpolate(station_xy, station_u, grid_lon, grid_lat, power=IDW_POWER)
    v_grid = idw_interpolate(station_xy, station_v, grid_lon, grid_lat, power=IDW_POWER)

    return grid_lon, grid_lat, u_grid, v_grid

def prepare_quiver_data(grid_lon: np.ndarray, grid_lat: np.ndarray, u_grid: np.ndarray, v_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Subsamples the grid data for a clean quiver plot.
    """
    # Determine subsampling step based on target grid spacing
    grid_res_m = GRID_SIZE_M
    target_km = QUIVER_GRID_KM
    step = max(1, int(target_km * 1000 / grid_res_m))
    
    # Apply subsampling
    q_lon = grid_lon[::step, ::step]
    q_lat = grid_lat[::step, ::step]
    q_u = u_grid[::step, ::step]
    q_v = v_grid[::step, ::step]
    
    # Filter out very low speeds
    q_spd = np.hypot(q_u, q_v)
    mask = q_spd >= QUIVER_MIN_SPEED
    
    # Flatten and mask for plotting
    return q_lon[mask], q_lat[mask], q_u[mask], q_v[mask]

# ---------------------------
# Main Execution
# ---------------------------

def main():
    readings, timestamp_str = parse_wind_readings()
    print(f"--- Wind Data Fetch & Overlay ---")
    print(f"Timestamp: {timestamp_str}")

    try:
        grid_lon, grid_lat, u_grid, v_grid = process_wind_field(readings)
    except ValueError as e:
        print(f"FATAL: {e}"); return
    
    # --- Data Export (For optimizationMain.py) ---
    wind_grid = np.stack([u_grid, v_grid], axis=-1)
    # Stacked wind_grid shape: (rows, cols, 2)
    export_name = f"wind_grid_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez_compressed(export_name, wind_grid=wind_grid)
    print(f"\n✅ Wind grid data saved to {export_name}")

    # --- Setup Plotting ---
    sg = ox.geocode_to_gdf("Singapore")
    poly = biggest_polygon(sg.iloc[0].geometry)
    
    # 1. Scalar Plot (Speed Heatmap)
    spd_grid = np.hypot(u_grid, v_grid)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    im = ax.imshow(spd_grid, origin="lower", extent=EXTENT, cmap="turbo")
    plt.colorbar(im, ax=ax, label="Wind Speed (m/s)")
    gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326").boundary.plot(ax=ax, color='k', lw=1.0, alpha=0.7)
    ax.set_title(f"Singapore Wind — Scalar Heatmap\n{timestamp_str} SGT (Res: {GRID_SIZE_M}m)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(True, lw=0.3, alpha=0.3)
    f1 = f"wind_scalar_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.tight_layout(); plt.savefig(f1, dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"✅ Scalar plot saved to {f1}")

    # 2. Quiver Plot (Arrows)
    q_lon, q_lat, q_u, q_v = prepare_quiver_data(grid_lon, grid_lat, u_grid, v_grid)
    
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    if QUIVER_BG_HEATMAP:
        ax.imshow(spd_grid, origin="lower", extent=EXTENT, cmap="turbo", alpha=0.35)

    # Normalize arrow lengths for visual contrast
    max_spd = np.nanpercentile(np.hypot(q_u, q_v), 98) 
    scale_factor = 45 * (max_spd / 4.0) # Empirical scale factor
    
    # Arrows
    ax.quiver(q_lon, q_lat, q_u, q_v,
              scale=scale_factor, width=QUIVER_WIDTH, alpha=QUIVER_ALPHA,
              pivot="mid", color='k')

    gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326").boundary.plot(ax=ax, color='k', lw=1.0, alpha=0.7)
    ax.set_title(f"Singapore Wind — Quiver Plot\n{timestamp_str} SGT (Spacing: ≈{QUIVER_GRID_KM} km, Min: {QUIVER_MIN_SPEED} m/s)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(True, lw=0.3, alpha=0.3)
    f2 = f"wind_quiver_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.tight_layout(); plt.savefig(f2, dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"✅ Quiver plot saved to {f2}")

if __name__ == "__main__":
    main()