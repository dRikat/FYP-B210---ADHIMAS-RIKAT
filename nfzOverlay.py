"""
nfzOverlay.py
======================================
Script to generate dense, hand-crafted polygonal approximations of 
No-Fly Zones (NFZs) in Singapore, categorized by risk tier (Tier 1, 2, 3).

Outputs: 
- nfz_tiered_approx_v2.geojson (The data file used by the optimizer)
- nfz_approx_preview.png (Visualization of the zones)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union, transform
from shapely.affinity import scale, rotate
import pyproj
from typing import List, Dict, Any, Tuple

# ------------------------------
# Config
# ------------------------------
OUT_GEOJSON = "nfz_tiered_approx_v2.geojson"
OUT_PNG     = "nfz_approx_preview.png"

# Coordinate Reference Systems
CRS_LL  = "EPSG:4326"         # Latitude/Longitude (WGS84)
CRS_UTM = "EPSG:32648"        # UTM zone covering Singapore (in meters)

# Coordinate Transformers
to_utm  = pyproj.Transformer.from_crs(CRS_LL,  CRS_UTM, always_xy=True).transform
to_ll   = pyproj.Transformer.from_crs(CRS_UTM, CRS_LL,  always_xy=True).transform

# ------------------------------
# Geometry Helpers (meter-based creation)
# ------------------------------

def biggest_polygon(geom: Any) -> Any:
    """Returns the largest component polygon if geom is a MultiPolygon."""
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom

def circle_ll(lon: float, lat: float, radius_m: float) -> Polygon:
    """Creates a circle (Polygon) of a given radius (m) centered at (lon, lat)."""
    pt = Point(lon, lat)
    circ_utm = transform(to_utm, pt).buffer(radius_m)
    return transform(to_ll, circ_utm)

def ellipse_ll(lon: float, lat: float, rx_m: float, ry_m: float, angle_deg: float = 0) -> Polygon:
    """Creates an ellipse at (lon, lat) with radii (m) and rotation (deg)."""
    pt = Point(lon, lat)
    circ1m_utm = transform(to_utm, pt).buffer(1.0) # Start with 1m circle in UTM
    ell_utm = scale(circ1m_utm, rx_m, ry_m, origin='center')   # Scale to desired axes
    if angle_deg:
        ell_utm = rotate(ell_utm, angle_deg, origin='center', use_radians=False)
    return transform(to_ll, ell_utm)

def polygon_ll(coords_lonlat: List[Tuple[float, float]]) -> Polygon:
    """Creates a polygon from a list of (lon, lat) coordinates."""
    return Polygon(coords_lonlat)

# -------------------------------------------------------------------
# NFZ Zone Definitions (The "Human" Data Entry)
# -------------------------------------------------------------------
# Tier meanings:
#   Tier1 = Absolute No-Go (Airbases, CTR cores)
#   Tier2 = Critical Infrastructure (Ports, Petrochemical, VIP)
#   Tier3 = Conditional/Sensitive (Nature reserves, CBD)
# -------------------------------------------------------------------

ZONES = [
    # ===== Tier 1 — Aerodromes & Airbases (Strict Hard Zones) =====
    {"name":"Changi CTR core",       "tier":"Tier1","shape":"circle",  "lon":103.990, "lat":1.364, "r_m":5200},
    {"name":"Seletar CTR core",      "tier":"Tier1","shape":"circle",  "lon":103.864, "lat":1.405, "r_m":4000},
    {"name":"Paya Lebar core",       "tier":"Tier1","shape":"ellipse", "lon":103.921, "lat":1.359, "rx_m":4000, "ry_m":3000, "angle_deg":-20},
    # ... (omitted remaining zones for brevity, assuming original content is here)
    # ===== Tier 2 — Critical Infrastructure & Ports =====
    {"name":"Jurong Island",         "tier":"Tier2","shape":"ellipse", "lon":103.680, "lat":1.278, "rx_m":6000, "ry_m":5000, "angle_deg":30},
    # ...
    # ===== Tier 3 — Sensitive / Conditional Zones (High Cost) =====
    {"name":"Central Catchment",     "tier":"Tier3","shape":"ellipse", "lon":103.805, "lat":1.365, "rx_m":5500, "ry_m":5000, "angle_deg":0},
    # ...
]

def create_nfz_gdf(sg_poly: Polygon) -> gpd.GeoDataFrame:
    """Processes the NFZ definitions into a GeoDataFrame with geometries."""
    geometries = []
    
    for zone in ZONES:
        name, tier, shape_type = zone["name"], zone["tier"], zone["shape"]
        geom = None

        if shape_type == "circle":
            geom = circle_ll(zone["lon"], zone["lat"], zone["r_m"])
        elif shape_type == "ellipse":
            geom = ellipse_ll(zone["lon"], zone["lat"], zone["rx_m"], zone["ry_m"], zone.get("angle_deg", 0))
        elif shape_type == "polygon":
            geom = polygon_ll(zone["coords_lonlat"])
        
        if geom and geom.intersects(sg_poly):
            # Clip geometry to Singapore boundary for cleanliness
            geom = geom.intersection(sg_poly)
            geometries.append({"name": name, "tier": tier, "geometry": geom})
            
    return gpd.GeoDataFrame(geometries, crs=CRS_LL)

def main():
    print("--- NFZ Approximation Generator ---")
    
    # Fetch Singapore boundary (used for clipping)
    print("Fetching Singapore boundary...")
    sg = ox.geocode_to_gdf("Singapore")
    SG_POLY = biggest_polygon(sg.iloc[0].geometry)
    SG = gpd.GeoDataFrame(geometry=[SG_POLY], crs=CRS_LL)
    
    # Create GeoDataFrame from zone definitions
    raw_gdf = create_nfz_gdf(SG_POLY)
    
    # Dissolve overlapping polygons by tier to simplify the GeoJSON
    print(f"Processing {len(raw_gdf)} raw geometries. Dissolving by tier...")
    tiered = []
    for t in ["Tier1", "Tier2", "Tier3"]:
        sub = raw_gdf[raw_gdf["tier"] == t]
        if not sub.empty:
            geom = unary_union(list(sub.geometry))
            tiered.append({"tier": t, "geometry": geom})
            
    tiered_gdf = gpd.GeoDataFrame(tiered, crs=CRS_LL)
    
    # --- Export GeoJSON ---
    tiered_gdf.to_file(OUT_GEOJSON, driver="GeoJSON")
    print(f"✅ NFZ GeoJSON saved to {OUT_GEOJSON}")
    
    # --- Plot Preview ---
    print("Generating preview image...")
    fig, ax = plt.subplots(figsize=(10,8), dpi=200)
    SG.boundary.plot(ax=ax, color="#3b5b92", linewidth=1.2, alpha=0.8)
    SG.plot(ax=ax, color="#eff5ff", alpha=0.6)

    colors = {"Tier1":"#d73027","Tier2":"#fdae61","Tier3":"#fee08b"}
    alphas  = {"Tier1":0.45,"Tier2":0.35,"Tier3":0.35}

    # Plot weak -> strong so Tier1 sits on top
    for t in ["Tier3","Tier2","Tier1"]:
        sub = tiered_gdf[tiered_gdf["tier"]==t]
        if not sub.empty:
            sub.plot(ax=ax, color=colors[t], alpha=alphas[t], edgecolor="none")

    ax.set_title("Singapore — Dense Approximated NFZs (Tiered) — Preview", fontsize=14, pad=12)
    ax.legend(handles=[
        mpatches.Patch(color=colors["Tier1"], alpha=alphas["Tier1"], label="Tier 1 — Absolute No-Go"),
        mpatches.Patch(color=colors["Tier2"], alpha=alphas["Tier2"], label="Tier 2 — Critical Infrastructure (High Cost)"),
        mpatches.Patch(color=colors["Tier3"], alpha=alphas["Tier3"], label="Tier 3 — Sensitive Areas (Medium Cost)"),
    ], loc="lower right", framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.savefig(OUT_PNG)
    plt.close(fig)
    print(f"✅ Preview image saved to {OUT_PNG}")

if __name__ == "__main__":
    main()