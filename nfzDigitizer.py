"""
nfzDigitizer.py
======================================
Interactive digitizer for hand-tracing custom No-Fly Zone (NFZ) polygons 
over a map of Singapore and saving them to a GeoJSON file.

Usage: Run the script, click on the map to define vertices, press 'z' to 
       complete the polygon, press 't' then '1', '2', or '3' to set the tier,
       and 's' to save the final GeoJSON and a preview image.
"""

import json
import sys
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import geopandas as gpd
import numpy as np
import osmnx as ox
import contextily as cx
from shapely.geometry import Polygon
from shapely.ops import unary_union

# -----------------------------
# Config
# -----------------------------
OUT_GEOJSON = "nfz_manual.geojson"
OUT_PNG = "nfz_manual_preview.png"
BASEMAP = cx.providers.OpenStreetMap.Mapnik  # Base map tile set

# -----------------------------
# Helpers
# -----------------------------

def singapore_outline() -> gpd.GeoDataFrame:
    """Fetches and cleans the main Singapore boundary polygon."""
    gdf = ox.geocode_to_gdf("Singapore")
    geom = gdf.iloc[0].geometry
    if geom.geom_type == "MultiPolygon":
        # Only take the largest (main island)
        geom = max(geom.geoms, key=lambda g: g.area)
    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    return gdf

@dataclass
class DraftPoly:
    """A draft polygon being constructed by the user."""
    xs: list
    ys: list
    tier: int = 1  # 1, 2, or 3 (default to highest risk)

    def to_polygon(self) -> Any:
        """Converts the collected points to a Shapely Polygon."""
        if len(self.xs) >= 3:
            coords = list(zip(self.xs, self.ys))
            return Polygon(coords)
        return None

# -----------------------------
# UI Class (Handles interaction and drawing)
# -----------------------------

class Digitizer:
    def __init__(self, ax: plt.Axes, sg_boundary: gpd.GeoDataFrame):
        self.ax = ax
        self.sg_boundary = sg_boundary
        self.current = DraftPoly([], [], tier=1)
        self.completed: List[DraftPoly] = []
        
        # Matplotlib artists for drawing points/polygons
        self.pts_artist = None
        self.poly_artist = None
        self.ui_text = ax.figure.text(0.02, 0.98, "", transform=ax.figure.transFigure, 
                                      fontsize=10, verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        # Connect event handlers
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.ax.figure.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.update_ui_hint("Ready. Click to add points. Keys: [Z] close, [R] reset, [T] set tier, [S] save, [Q] quit.")
        self.ax.figure.canvas.draw()

    def update_ui_hint(self, message: str):
        """Updates the text hint in the corner of the figure."""
        tier_color = {1: 'red', 2: 'orange', 3: 'yellow'}
        tier_info = f"CURRENT TIER: {self.current.tier} (Color: {tier_color[self.current.tier]})"
        self.ui_text.set_text(f"{tier_info}\n---\n{message}")
        self.ax.figure.canvas.draw()

    def draw_current(self):
        """Draws the points and the current, incomplete polygon."""
        if self.pts_artist: self.pts_artist.remove()
        if self.poly_artist: self.poly_artist.remove()
        
        # 1. Draw points
        if self.current.xs:
            self.pts_artist, = self.ax.plot(self.current.xs, self.current.ys, 'o', color='k', ms=4, zorder=100)

        # 2. Draw connecting line/polygon preview
        if len(self.current.xs) >= 2:
            vertices = list(zip(self.current.xs, self.current.ys))
            codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1)
            path = Path(vertices, codes)
            
            face_color = {1: '#d73027', 2: '#fdae61', 3: '#fee08b'}.get(self.current.tier, 'gray')
            
            self.poly_artist = PathPatch(path, facecolor=face_color, edgecolor='k', lw=1, alpha=0.5, zorder=50)
            self.ax.add_patch(self.poly_artist)

        self.ax.figure.canvas.draw()

    def close_current(self):
        """Finalizes the current draft polygon and starts a new one."""
        if len(self.current.xs) >= 3:
            self.completed.append(self.current)
            self.update_ui_hint(f"Polygon Tier {self.current.tier} completed ({len(self.current.xs)} vertices). Starting new Tier 1 draft.")
        else:
            self.update_ui_hint("Need at least 3 points to close a polygon.")
            
        # Clear artists and start new draft
        if self.pts_artist: self.pts_artist.remove()
        if self.poly_artist: self.poly_artist.remove()
        self.current = DraftPoly([], [], tier=1)
        self.draw_completed()

    def reset_current(self):
        """Clears the points of the current draft."""
        self.current = DraftPoly([], [], tier=self.current.tier)
        self.update_ui_hint("Current draft reset. Click to start a new polygon.")
        if self.pts_artist: self.pts_artist.remove()
        if self.poly_artist: self.poly_artist.remove()
        self.ax.figure.canvas.draw()

    def draw_completed(self):
        """Redraws all completed polygons."""
        # Clear all polygons
        for patch in self.ax.patches:
            if patch is not self.sg_boundary.patches[0]: # Don't remove the SG background
                 patch.remove()

        # Redraw all completed polygons
        for poly_data in self.completed:
            geom = poly_data.to_polygon()
            if geom:
                x, y = geom.exterior.xy
                face_color = {1: '#d73027', 2: '#fdae61', 3: '#fee08b'}.get(poly_data.tier, 'gray')
                patch = PathPatch(Path(list(zip(x,y))), facecolor=face_color, edgecolor='k', lw=1, alpha=0.5, zorder=50)
                self.ax.add_patch(patch)
        self.draw_current()
        
    def save(self):
        """Saves all completed polygons to a GeoJSON file."""
        if not self.completed:
            self.update_ui_hint("Nothing to save. Complete at least one polygon first (press 'z').")
            return
            
        polygons = []
        for poly_data in self.completed:
            geom = poly_data.to_polygon()
            if geom:
                polygons.append({"tier": f"Tier{poly_data.tier}", "geometry": geom})

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(polygons, crs="EPSG:4326")
        
        # Dissolve (merge) polygons by tier
        dissolved = []
        for t in ["Tier1", "Tier2", "Tier3"]:
            sub = gdf[gdf["tier"] == t]
            if not sub.empty:
                geom = unary_union(list(sub.geometry))
                dissolved.append({"tier": t, "geometry": geom})

        out = gpd.GeoDataFrame(dissolved, crs="EPSG:4326")
        
        # Export
        out.to_file(OUT_GEOJSON, driver="GeoJSON")
        self.ax.figure.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
        self.update_ui_hint(f"Saved {OUT_GEOJSON} and {OUT_PNG}")

    # -----------------------------
    # Event Handlers
    # -----------------------------

    def on_click(self, event: Any):
        """Handles mouse clicks to add points."""
        if event.inaxes != self.ax: return
        
        self.current.xs.append(event.xdata)
        self.current.ys.append(event.ydata)
        self.draw_current()
        self.update_ui_hint(f"Added point. Total vertices: {len(self.current.xs)}. Press [Z] to close polygon.")

    def on_key(self, event: Any):
        """Handles key presses for commands."""
        key = event.key.lower()
        if key == "z":
            self.close_current()
            self.draw_completed()
        elif key == "r":
            self.reset_current()
        elif key == "s":
            self.save()
        elif key == "q":
            plt.close(self.ax.figure)
        elif key == "t":
            self.update_ui_hint("Press 1, 2, or 3 to set the TIER for current polygon")
        elif key in ("1", "2", "3"):
            self.current.tier = int(key)
            self.draw_current()
            self.update_ui_hint(f"Tier set to {key}. Add points or press [Z] to close.")
            
# ----------------------------
# Main
# ----------------------------

def main():
    print("--- NFZ Manual Digitizer ---")
    sg = singapore_outline()
    xmin, ymin, xmax, ymax = sg.total_bounds

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    
    # Plot Singapore outline as background
    sg.boundary.plot(ax=ax, color="#3b5b92", linewidth=1.2, alpha=0.8)
    sg.plot(ax=ax, color="#eff5ff", alpha=0.5)

    # Add basemap (can be slow)
    print("Fetching basemap tiles (this might take a moment)...")
    try:
        cx.add_basemap(ax, crs=sg.crs, source=BASEMAP)
    except Exception as e:
        print(f"Warning: Could not add basemap: {e}")

    ax.set_xlim(xmin - 0.05, xmax + 0.05)
    ax.set_ylim(ymin - 0.05, ymax + 0.05)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.set_title("NFZ Manual Digitizer — Singapore", fontsize=14, pad=12)

    # Initialize and run the digitizer UI
    Digitizer(ax, sg)
    plt.show()

if __name__ == "__main__":
    main()