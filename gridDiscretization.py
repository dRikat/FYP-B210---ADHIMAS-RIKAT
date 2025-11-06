"""
gridDiscretization.py
======================================
Utility script for tuning the optimal grid cell size (h). 
It scores grid resolutions based on quantization error and runtime proxies 
(like total cell count) using a road-only corridor model.

Dependencies: numpy, pandas, matplotlib, optimizationMain
"""
import math
import pathlib
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Clean import from core module ---
try:
    # Ensure the core optimization module is on the path if needed
    sys.path.append(str(pathlib.Path(__file__).resolve().parent))
    from optimizationMain import Config, create_groundbiased_grid_hard_corridor, CellType
except ImportError as e:
    print(f"FATAL: Could not import core components from optimizationMain.py. Ensure it is in the same directory. Error: {e}")
    sys.exit(1)

HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE / "discretization_analysis"
OUT_DIR.mkdir(exist_ok=True)

# --- Configuration for this tuner ---
CELL_SIZES = list(range(10, 301, 20))     # Cell sizes to test (in meters)

# The corridor radius is a core setting, taken from the imported Config
R = Config.CORRIDOR_RADIUS_M

def score(h: int) -> Dict[str, Any]:
    """Runs a single discretization test for cell size 'h'."""
    Config.CELL_SIZE = h  # The grid builder reads this global setting

    print(f"Testing cell size h={h}m...")
    grid = create_groundbiased_grid_hard_corridor() # builds SpatialGrid
    
    road_mask = grid._road_mask
    if road_mask is None:
        raise RuntimeError("Grid builder failed to set road mask.")

    free_cells = int(road_mask.sum())
    cells_total = grid.rows * grid.cols
    h_m = grid.cell_m

    # --- Proxies ---
    # 1. Quantization Error Proxy (based on 45-degree corner of a road segment)
    corridor_cells_across = (2.0 * R) / h_m
    mean_quant_err_m = (h_m / 4) * (1 / math.sqrt(2)) # A conservative, simplified proxy

    # 2. Estimated Total Road Length (L_est, meters)
    # Total Free Cells * (Area of one cell) / (Width of road corridor)
    # Area = h_m^2
    # Width = 2*R
    L_est_m = (free_cells * h_m**2) / (2.0 * R)

    # 3. Runtime Proxy (Total Cells)
    runtime_proxy_cells = cells_total

    return {
        "cell_size_m": h_m,
        "cells_total": cells_total,
        "free_cells": free_cells,
        "L_est_m": L_est_m,
        "mean_quant_err_m": mean_quant_err_m,
        "runtime_proxy_cells": runtime_proxy_cells
    }

def main():
    """Main execution block to run the grid size sweep and plot results."""
    
    print(f"--- Grid Discretization Tuner ---")
    print(f"Corridor Half-Width (R) is set to {R}m (from optimizationMain.Config)")
    
    results = [score(h) for h in CELL_SIZES]
    df = pd.DataFrame(results)

    # --- Export Results ---
    csvp = OUT_DIR / "discretization_results.csv"
    df.to_csv(csvp, index=False)
    print(f"\n✅ Results saved to: {csvp}")

    # --- Generate Charts ---
    print("Generating charts...")
    
    # 1. Quantization Error
    plt.figure(figsize=(6,4))
    plt.plot(df["cell_size_m"], df["mean_quant_err_m"], marker="o")
    plt.title("Quantization Error Proxy vs Cell Size"); plt.xlabel("Cell Size (m)"); plt.ylabel("Mean Error Proxy (m)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(); plt.savefig(OUT_DIR / "quant_error_vs_cellsize.png", dpi=160); plt.close()

    # 2. Estimated Total Road Length
    plt.figure(figsize=(6,4))
    plt.plot(df["cell_size_m"], df["L_est_m"], marker="o")
    plt.title("Estimated Total Road Length vs Cell Size"); plt.xlabel("Cell Size (m)"); plt.ylabel("L_est (m)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(); plt.savefig(OUT_DIR / "L_est_vs_cellsize.png", dpi=160); plt.close()

    # 3. Runtime Proxy (Total Cells)
    plt.figure(figsize=(6,4))
    plt.plot(df["cell_size_m"], df["cells_total"], marker="o")
    plt.title("Runtime Proxy (Total Cells) vs Cell Size"); plt.xlabel("Cell Size (m)"); plt.ylabel("Total Cells (Log Scale)")
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(); plt.savefig(OUT_DIR / "cells_vs_cellsize.png", dpi=160); plt.close()

    print("--- Analysis Complete ---")

if __name__ == "__main__":
    main()