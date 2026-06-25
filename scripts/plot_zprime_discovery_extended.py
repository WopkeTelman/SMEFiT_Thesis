#!/usr/bin/env python3
"""
plot_zprime_discovery_extended.py

Analytically compute and plot the Z' discovery region over an extended
(gZH, mZp) grid, keeping gZl fixed at base value.

Saves to results/zprime_corrected_scan_scan/plots/discovery_region_extended.png
without touching the existing discovery_region.png.
"""

import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(Path(__file__).parent))

from models.zprime import ZPrimeModel
from scripts.run_pipeline import _build_K_Ci   # reuse existing K/Ci builder


def q_to_sigma(q, ndof):
    if q <= 0:
        return 0.0
    p = chi2_dist.sf(q, df=ndof)
    return float(norm_dist.isf(p / 2)) if p > 0 else 99.0


def compute_sigma_grid(coupling_grid, mass_grid, gZl_fixed, K, Ci, ops, ndof):
    """Analytic UV significance over (gZH, mZp) grid with gZl fixed."""
    nc, nm = len(coupling_grid), len(mass_grid)
    sigma = np.zeros((nc, nm))
    for ic, gZH in enumerate(coupling_grid):
        for im, mZp in enumerate(mass_grid):
            model   = ZPrimeModel(gZH=gZH, gZl=gZl_fixed, mZp=mZp)
            c_truth = np.array([model.eft_coefficients().get(op, 0.0) for op in ops])
            Kc      = K @ c_truth
            q       = float(Kc @ Ci @ Kc)
            sigma[ic, im] = q_to_sigma(q, ndof)
    return sigma


def main():
    gZl_fixed = 0.04     # held constant across the scan
    gZH_min   = 0.001
    gZH_max   = 3.5
    mZp_min   = 0.3
    mZp_max   = 38.0
    ng, nm    = 90, 120

    base_params = {"gZH": 0.12, "gZl": gZl_fixed, "mZp": 1.0}
    ops = ZPrimeModel(**base_params).OPERATORS
    ndof = 2   # gZH and gZl both free in UV fit

    print(f"Building K, Ci  (ops: {ops})")
    K, Ci = _build_K_Ci(ops)
    print(f"  K: {K.shape}   Ci: {Ci.shape}")

    coupling_grid = list(np.linspace(gZH_min, gZH_max, ng))
    mass_grid     = list(np.linspace(mZp_min, mZp_max, nm))

    print(f"Computing {ng}×{nm} grid ...")
    sigma_uv = compute_sigma_grid(coupling_grid, mass_grid,
                                  gZl_fixed, K, Ci, ops, ndof)
    sigma_uv = np.clip(sigma_uv, 0, 50)

    mass_arr     = np.array(mass_grid)
    coupling_arr = np.array(coupling_grid)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.contourf(mass_arr, coupling_arr, sigma_uv,
                levels=[5.0, 1e4], colors=["#BDD7EE"], alpha=0.7)
    ax.contourf(mass_arr, coupling_arr, sigma_uv,
                levels=[0, 3.0],   colors=["#DCDCDC"], alpha=0.6)
    ax.contourf(mass_arr, coupling_arr, sigma_uv,
                levels=[3.0, 5.0], colors=["#E8F4FD"], alpha=0.5)

    cs5 = ax.contour(mass_arr, coupling_arr, sigma_uv,
                     levels=[5.0], colors=["#1f77b4"], linewidths=[2.5])
    ax.clabel(cs5, fmt=r"$5\sigma$", fontsize=11, inline=True, inline_spacing=8)

    cs3 = ax.contour(mass_arr, coupling_arr, sigma_uv,
                     levels=[3.0], colors=["#1f77b4"], linewidths=[1.5],
                     linestyles=["--"])
    ax.clabel(cs3, fmt=r"$3\sigma$", fontsize=11, inline=True, inline_spacing=8)



    ax.set_xlabel(r"$m_{Z'}$ [TeV]", fontsize=14)
    ax.set_ylabel(r"$|g_{ZH}|$",     fontsize=14)
    ax.set_title(r"FCC-ee  discovery reach  (Z'$,\ g_{Zl}=0.04$ fixed)", fontsize=13)
    ax.set_xlim(mZp_min, mZp_max)
    ax.set_ylim(0, gZH_max)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
    ax.grid(True, which="major", alpha=0.15, lw=0.5)

    ax.text(0.97, 0.97, r"Discoverable ($>5\sigma$)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="#1f77b4", style="italic")
    ax.text(0.97, 0.03, "Not discoverable",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color="gray", style="italic")

    handles = [
        Line2D([0],[0], color="#1f77b4", lw=2.5, label=r"UV: $5\sigma$ discovery"),
        Line2D([0],[0], color="#1f77b4", lw=1.5, ls="--", label=r"UV: $3\sigma$ evidence"),
    ]
    ax.legend(handles=handles, fontsize=10, loc="upper left",
              framealpha=0.9, edgecolor="gray")

    plt.tight_layout()

    out_dir = PIPELINE / "results" / "zprime_corrected_scan_scan" / "plots"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "discovery_region_extended.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
