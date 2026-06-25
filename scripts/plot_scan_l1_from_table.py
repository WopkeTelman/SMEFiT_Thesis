"""Regenerate scan-l1 plots from an existing scan_l1_table.txt."""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import LinearNDInterpolator


def load_table(path):
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            results.append({
                "gWH":          float(parts[0]),
                "mWp":          float(parts[1]),
                "sigma_l1_16":  float(parts[2]),
                "sigma_l1_50":  float(parts[3]),
                "sigma_l1_84":  float(parts[4]),
                "sigma_l1_tc_16": float(parts[5]),
                "sigma_l1_tc_50": float(parts[6]),
                "sigma_l1_tc_84": float(parts[7]),
            })
    return results


def plot_1d(results, coupling_grid, mass_grid, coupling_key, mass_key,
            plt_dir, tag, n_reps):
    nc    = len(coupling_grid)
    ncols = min(nc, 3)
    nrows = (nc + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.2 * nrows), squeeze=False)
    for ic, g_val in enumerate(coupling_grid):
        ax  = axes[ic // ncols][ic % ncols]
        pts = sorted([r for r in results if abs(r[coupling_key] - g_val) < 1e-4],
                     key=lambda r: r[mass_key])
        if not pts:
            ax.set_visible(False)
            continue
        masses = [p[mass_key] for p in pts]
        ax.fill_between(masses,
                        [p["sigma_l1_16"] for p in pts],
                        [p["sigma_l1_84"] for p in pts],
                        alpha=0.25, color="steelblue", label="L1 68% (no TC)")
        ax.plot(masses, [p["sigma_l1_50"] for p in pts],
                color="steelblue", lw=2.0, label="L1 median (no TC)")
        ax.fill_between(masses,
                        [p["sigma_l1_tc_16"] for p in pts],
                        [p["sigma_l1_tc_84"] for p in pts],
                        alpha=0.20, color="crimson", label="L1 68% (TC aggressive)")
        ax.plot(masses, [p["sigma_l1_tc_50"] for p in pts],
                color="crimson", lw=2.0, ls="--", label="L1 median (TC aggressive)")
        for thr, col, lbl in [(5.0, "gold", r"5$\sigma$"), (3.0, "orange", r"3$\sigma$")]:
            ax.axhline(thr, color=col, lw=1.2, ls="--")
            ax.text(masses[-1] * 0.97, thr + 0.15, lbl, color=col, ha="right", fontsize=9)
        ax.set_xlabel(f"{mass_key}  [TeV]", fontsize=11)
        ax.set_ylabel(r"$\sigma_\mathrm{UV}$", fontsize=11)
        ax.set_title(fr"${coupling_key} = {g_val:.4f}$", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
    for idx in range(nc, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle(f"Discovery reach — {tag}  (L1: {n_reps} reps, TC aggressive)",
                 fontsize=12)
    plt.tight_layout()
    out1 = os.path.join(plt_dir, "significance_vs_mass.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.savefig(out1.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  Saved: {out1}")
    plt.close()


def plot_2d(results, coupling_grid, mass_grid, coupling_key, mass_key,
            plt_dir, tag, n_reps):
    if len(coupling_grid) < 3 or len(mass_grid) < 3:
        print("  Skipping 2D plot (grid too small)")
        return

    # Build interpolation points
    pts_xy = np.array([[r[mass_key], r[coupling_key]] for r in results])
    SIGMA_CAP = 30.0

    def _interp_field(key):
        vals = np.array([min(r[key], SIGMA_CAP) for r in results])
        return LinearNDInterpolator(pts_xy, vals)

    interp = {k: _interp_field(k) for k in
              ["sigma_l1_16", "sigma_l1_50", "sigma_l1_84",
               "sigma_l1_tc_16", "sigma_l1_tc_50", "sigma_l1_tc_84"]}

    m_fine = np.linspace(min(mass_grid), max(mass_grid), 300)
    c_fine = np.linspace(min(coupling_grid), max(coupling_grid), 300)
    MM, CC = np.meshgrid(m_fine, c_fine)
    pts_fine = np.column_stack([MM.ravel(), CC.ravel()])

    def _eval(key):
        v = interp[key](pts_fine)
        v = np.where(np.isnan(v), 0.0, v)
        return v.reshape(MM.shape)

    G = {k: _eval(k) for k in interp}

    fig2, ax = plt.subplots(figsize=(8, 5.5))
    for thr, ls_med, ls_tc in [(5.0, "-", "--"), (3.0, "-.", ":")]:
        band_notc = (G["sigma_l1_84"] >= thr) & (G["sigma_l1_16"] < thr)
        if band_notc.any():
            ax.contourf(m_fine, c_fine, band_notc.astype(float),
                        levels=[0.5, 1.5], colors=["steelblue"], alpha=0.20)
        band_tc = (G["sigma_l1_tc_84"] >= thr) & (G["sigma_l1_tc_16"] < thr)
        if band_tc.any():
            ax.contourf(m_fine, c_fine, band_tc.astype(float),
                        levels=[0.5, 1.5], colors=["crimson"], alpha=0.15)
        try:
            cs = ax.contour(m_fine, c_fine, G["sigma_l1_50"], levels=[thr],
                            colors=["steelblue"], linewidths=2.0, linestyles=[ls_med])
            ax.clabel(cs, fmt=f"{thr:.0f}σ", fontsize=8)
        except Exception:
            pass
        try:
            cs = ax.contour(m_fine, c_fine, G["sigma_l1_tc_50"], levels=[thr],
                            colors=["crimson"], linewidths=2.0, linestyles=[ls_tc])
            ax.clabel(cs, fmt=f"{thr:.0f}σ TC", fontsize=8)
        except Exception:
            pass

    handles = [
        Line2D([0], [0], color="steelblue", lw=2,       label="L1 median (no TC)"),
        Line2D([0], [0], color="crimson",   lw=2, ls="--", label="L1 median (TC aggressive)"),
        Patch(facecolor="steelblue", alpha=0.3, label="L1 68% band (no TC)"),
        Patch(facecolor="crimson",   alpha=0.3, label="L1 68% band (TC aggressive)"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper right")
    ax.set_xlabel(f"{mass_key}  [TeV]", fontsize=12)
    ax.set_ylabel(coupling_key, fontsize=12)
    ax.set_title(f"Discovery reach — {tag}  (n_reps={n_reps}, TC aggressive)",
                 fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out2 = os.path.join(plt_dir, "discovery_reach_l1.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.savefig(out2.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  Saved: {out2}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", required=True, help="Path to scan_l1_table.txt")
    p.add_argument("--plt-dir", required=True, help="Output directory for plots")
    p.add_argument("--tag", default="wprime_constrained", help="Label for titles")
    p.add_argument("--coupling-key", default="gWH")
    p.add_argument("--mass-key", default="mWp")
    p.add_argument("--n-reps", type=int, default=50)
    args = p.parse_args()

    os.makedirs(args.plt_dir, exist_ok=True)
    results = load_table(args.table)
    print(f"  Loaded {len(results)} grid points from {args.table}")

    coupling_grid = sorted(set(round(r[args.coupling_key], 6) for r in results))
    mass_grid     = sorted(set(round(r[args.mass_key],     6) for r in results))
    print(f"  coupling_grid ({len(coupling_grid)}): {coupling_grid}")
    print(f"  mass_grid     ({len(mass_grid)}): {mass_grid}")

    plot_1d(results, coupling_grid, mass_grid,
            args.coupling_key, args.mass_key,
            args.plt_dir, args.tag, args.n_reps)
    plot_2d(results, coupling_grid, mass_grid,
            args.coupling_key, args.mass_key,
            args.plt_dir, args.tag, args.n_reps)


if __name__ == "__main__":
    main()
