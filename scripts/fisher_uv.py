#!/usr/bin/env python3
"""
UV Fisher information heatmap — Linear and Quadratic, side by side.

Fixes the bias in smefit's built-in Fisher for UV fits: smefit evaluates the
Jacobian at g=0 (one coupling set to 1, all others zeroed), zeroing out all
bilinear matching terms like O3pl1 = -0.25·gWH·gWLf/mWp².

  Linear   — Jacobian via finite differences at the UV truth point.
  Quadratic — Jacobian averaged over the NS posterior (loads fit_results.json
               from the _UVcoup NS run).  Captures how sensitivity varies as
               the Jacobian changes across the posterior for bilinear matchings.

Usage (recommended — tag encodes model + couplings):
    python scripts/fisher_uv.py --tag wprime_gwh050_mwp100
    python scripts/fisher_uv.py --tag wprime_gwh012_mwp020

Usage (explicit params, default gWLf = gWqf = gWH/3):
    python scripts/fisher_uv.py --model wprime --gWH 0.5  --mWp 10.0
    python scripts/fisher_uv.py --model wprime --gWH 0.12 --mWp 2.0 --gWLf 0.04 --gWqf 0.04
    python scripts/fisher_uv.py --model zprime --gZH 0.12 --mZp 7.5

Optional flags:
    --ns-draws N   number of NS posterior draws to use for quadratic (default 500)
    --out PATH     override the output file path
    --no-title     omit the figure title
"""
import argparse
import dataclasses
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE   = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE / "scripts"))
sys.path.insert(0, str(PIPELINE))

from run_pipeline import DATA_INFO, DATASETS, _build_K_Ci, SM_DATA, THEORY  # noqa: E402

RESULTS = PIPELINE / "results"

ENERGY_GROUPS = {
    "91 GeV":  [ds for ds, _ in DATA_INFO["91 GeV (Z-pole)"]],
    "161 GeV": [ds for ds, _ in DATA_INFO["161 GeV (WW thr.)"]],
    "240 GeV": [ds for ds, _ in DATA_INFO["240 GeV"]],
    "365 GeV": [ds for ds, _ in DATA_INFO["365 GeV"]],
}
GROUP_ORDER = ["91 GeV", "161 GeV", "240 GeV", "365 GeV"]

UV_LATEX = {
    "gWH":    r"$g_{WH}$",
    "gWLf11": r"$g_{WLf,11}$",
    "gWLf22": r"$g_{WLf,22}$",
    "gWLf33": r"$g_{WLf,33}$",
    "gWqf33": r"$g_{Wqf,33}$",
    "gZH":    r"$g_{ZH}$",
    "gZl":    r"$g_{Zl}$",
}


# ── Model construction ────────────────────────────────────────────────────────

def _build_model(model_name, **kwargs):
    if model_name == "wprime_1g":
        from models.wprime_1g import WPrime1gModel
        return WPrime1gModel(g=kwargs["g"], mWp=kwargs["mWp"])
    if model_name == "wprime":
        from models.wprime import WPrimeModel
        gWLf = kwargs.get("gWLf") or kwargs["gWH"] / 3
        gWqf = kwargs.get("gWqf") or kwargs["gWH"] / 3
        return WPrimeModel(gWH=kwargs["gWH"], gWLf11=gWLf, gWLf22=gWLf,
                           gWLf33=gWLf, gWqf33=gWqf, mWp=kwargs["mWp"])
    if model_name == "wprime_constrained_v3":
        from models.wprime_constrained_v3 import WPrimeConstrainedV3Model
        gWLf = kwargs.get("gWLf") or kwargs["gWH"] / 3
        return WPrimeConstrainedV3Model(gWH=kwargs["gWH"], gWLf11=gWLf, gWLf22=gWLf,
                                        gWLf33=gWLf, gWqf33=gWLf, mWp=kwargs["mWp"])
    if model_name == "wprime_constrained_v2":
        from models.wprime_constrained_v2 import WPrimeConstrainedV2Model
        gWLf = kwargs.get("gWLf") or kwargs["gWH"] / 3
        return WPrimeConstrainedV2Model(gWH=kwargs["gWH"], gWLf11=gWLf, gWLf22=gWLf,
                                        gWLf33=gWLf, gWqf33=gWLf, mWp=kwargs["mWp"])
    if model_name == "wprime_constrained":
        from models.wprime_constrained import WPrimeConstrainedModel
        gWLf = kwargs.get("gWLf") or kwargs["gWH"] / 3
        return WPrimeConstrainedModel(gWH=kwargs["gWH"], gWLf11=gWLf, gWLf22=gWLf,
                                      gWLf33=gWLf, gWqf33=gWLf, mWp=kwargs["mWp"])
    if model_name == "wprime_universal":
        from models.wprime_universal import WPrimeUniversalModel
        return WPrimeUniversalModel(gWH=kwargs["gWH"], mWp=kwargs["mWp"])
    if model_name == "zprime":
        from models.zprime import ZPrimeModel
        return ZPrimeModel(gZH=kwargs["gZH"], gZl=kwargs.get("gZl", kwargs["gZH"] / 3),
                           mZp=kwargs["mZp"])
    if model_name == "zprime_constrained":
        from models.zprime_constrained import ZPrimeConstrainedModel
        return ZPrimeConstrainedModel(gZH=kwargs["gZH"],
                                      gZl=kwargs.get("gZl", kwargs["gZH"] / 3),
                                      mZp=kwargs["mZp"])
    if model_name == "zprime2":
        from models.zprime2 import ZPrimeModel as ZPrimeV2Model
        return ZPrimeV2Model(gZH=kwargs["gZH"], gZl=kwargs.get("gZl", kwargs["gZH"] / 3),
                             mZp=kwargs["mZp"])
    if model_name == "zprime2_constrained":
        from models.zprime2_constrained import ZPrimeV2ConstrainedModel
        return ZPrimeV2ConstrainedModel(gZH=kwargs["gZH"],
                                        gZl=kwargs.get("gZl", kwargs["gZH"] / 3),
                                        mZp=kwargs["mZp"])
    raise ValueError(f"Unknown model: {model_name!r}")


def _parse_tag(tag):
    patterns = [
        (r"^(wprime_1g)_g(\d+)_mwp(\d+)",              "g",   "mWp"),
        (r"^(wprime_constrained_v3)_gwh(\d+)_mwp(\d+)", "gWH", "mWp"),
        (r"^(wprime_constrained_v2)_gwh(\d+)_mwp(\d+)", "gWH", "mWp"),
        (r"^(wprime_constrained)_gwh(\d+)_mwp(\d+)",    "gWH", "mWp"),
        (r"^(wprime_universal)_gwh(\d+)_mwp(\d+)",      "gWH", "mWp"),
        (r"^(wprime)_gwh(\d+)_mwp(\d+)",                "gWH", "mWp"),
        (r"^(zprime2_constrained)_gzh(\d+)_mzp(\d+)",  "gZH", "mZp"),
        (r"^(zprime2)_gzh(\d+)_mzp(\d+)",              "gZH", "mZp"),
        (r"^(zprime_constrained)_gzh(\d+)_mzp(\d+)",   "gZH", "mZp"),
        (r"^(zprime)_gzh(\d+)_mzp(\d+)",               "gZH", "mZp"),
    ]
    for pat, ck, mk in patterns:
        m = re.match(pat, tag)
        if m:
            return m.group(1), {ck: int(m.group(2)) / 100, mk: int(m.group(3)) / 10}
    raise ValueError(
        f"Cannot parse tag {tag!r}. "
        "Expected e.g. wprime_gwh050_mwp100 or zprime_gzh012_mzp075."
    )


# ── Shared infrastructure ─────────────────────────────────────────────────────

def _build_K_ds_rows(model):
    """Build K, Ci and dataset→row-range map. Shared by linear and quadratic."""
    ops  = model.OPERATORS
    K, Ci = _build_K_Ci(ops)
    row, ds_rows = 0, {}
    for ds in DATASETS:
        name    = ds["name"]
        sm_path = f"{SM_DATA}/{name}.yaml"
        th_path = f"{THEORY}/{name}.json"
        if not os.path.exists(sm_path) or not os.path.exists(th_path):
            continue
        sm = yaml.safe_load(open(sm_path))
        dc = sm["data_central"]
        n  = len(dc) if isinstance(dc, list) else 1
        ds_rows[name] = (row, row + n)
        row += n
    return K, Ci, ds_rows


def _load_K_quad(ops, ds_rows):
    """Load K^(2)[k,a,b] from 'op_a*op_b' keys in theory LO dicts.

    Returns a (n_data, n_ops, n_ops) array, or None if nothing is found.
    Only pairs where both operators are in `ops` are included.
    """
    n_data  = sum(r1 - r0 for r0, r1 in ds_rows.values())
    n_ops   = len(ops)
    op_idx  = {op: i for i, op in enumerate(ops)}
    ops_set = set(ops)
    K_quad  = np.zeros((n_data, n_ops, n_ops))
    found   = False

    row = 0
    for ds in DATASETS:
        name    = ds["name"]
        sm_path = f"{SM_DATA}/{name}.yaml"
        th_path = f"{THEORY}/{name}.json"
        if not os.path.exists(sm_path) or not os.path.exists(th_path):
            continue
        sm = yaml.safe_load(open(sm_path))
        dc = sm["data_central"]
        n  = len(dc) if isinstance(dc, list) else 1
        lo = json.load(open(th_path)).get("LO", {})
        for key, val in lo.items():
            if "*" not in key:
                continue
            parts = key.split("*")
            if len(parts) != 2:
                continue
            op_a, op_b = parts
            if op_a not in ops_set or op_b not in ops_set:
                continue
            ia, ib = op_idx[op_a], op_idx[op_b]
            v = np.array(val[:n], dtype=float) if isinstance(val, list) else np.full(n, float(val))
            K_quad[row:row + n, ia, ib] += v
            if ia != ib:
                K_quad[row:row + n, ib, ia] += v
            found = True
        row += n

    if not found:
        print("  [K_quad] No quadratic corrections found — K^(2) = 0.")
        return None
    nnz = int(np.count_nonzero(K_quad))
    print(f"  [K_quad] Loaded: {nnz} non-zero entries out of {K_quad.size} "
          f"({100*nnz/K_quad.size:.1f}%)")
    return K_quad


def _jacobian(cls, fields, uv_params, ops, eps=1e-6):
    """Central finite-difference Jacobian dc/dg at the point given by fields."""
    J = np.zeros((len(ops), len(uv_params)))
    for j, par in enumerate(uv_params):
        kp = {**fields, par: fields[par] + eps}
        km = {**fields, par: fields[par] - eps}
        cp = np.array([cls(**kp).eft_coefficients().get(op, 0.) for op in ops])
        cm = np.array([cls(**km).eft_coefficients().get(op, 0.) for op in ops])
        J[:, j] = (cp - cm) / (2 * eps)
    return J


def _fisher_from_KJ(KJ, Ci, ds_rows, n_uv):
    """Accumulate the per-group diagonal Fisher from KJ (n_obs × n_uv)."""
    F = np.zeros((n_uv, len(GROUP_ORDER)))
    for ig, grp in enumerate(GROUP_ORDER):
        for ds in ENERGY_GROUPS[grp]:
            if ds not in ds_rows:
                continue
            r0, r1 = ds_rows[ds]
            KJ_d  = KJ[r0:r1]
            Ci_d  = Ci[r0:r1, r0:r1]
            F[:, ig] += np.diag(KJ_d.T @ Ci_d @ KJ_d)
    return F


def _normalise(F):
    row_sums = F.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.
    return 100. * F / row_sums


def _all_fields(model):
    return ({f.name: getattr(model, f.name) for f in dataclasses.fields(model)}
            if dataclasses.is_dataclass(model) else vars(model).copy())


# ── Fisher computation ────────────────────────────────────────────────────────

def compute_fisher_lin(model, K, Ci, ds_rows):
    """Linear Fisher: Jacobian at the UV truth point."""
    ops       = model.OPERATORS
    uv_params = model.uv_param_names()
    J         = _jacobian(type(model), _all_fields(model), uv_params, ops)
    return _normalise(_fisher_from_KJ(K @ J, Ci, ds_rows, len(uv_params)))


def compute_fisher_quad(model, K, K_quad, Ci, ds_rows, tag, n_draws=500):
    """
    Exact smefit compute_quadratic formula, evaluated at the UV truth point.

    Replicates smefit's fisher.py compute_quadratic with corrected inputs:
      lin_corr[i,k]      = K^(1)[k,a] J_ai(g_truth)          — truth-point Jacobian
      quad_corr[i,k]     = K^(2)[k,a,b] J_ai J_bi             — diagonal Q at truth
      off_diag_corr[i,j,k] = K^(2)[k,a,b] J_ai J_bj (i≠j)   — off-diagonal Q at truth
      posterior_g        = actual NS UV coupling samples
      delta_th           = BSM_signal − mean(K c(g_s))         — data residual

    Result = quad_correction + lin_fisher  (matching smefit's quad_fisher + lin_fisher).
    """
    from run_pipeline import _build_delta

    fit_path = RESULTS / tag / "fits" / f"{tag}_UVcoup" / "fit_results.json"
    if not fit_path.exists():
        print(f"  [quad] NS file not found at {fit_path} — skipping quadratic panel.")
        return None

    r         = json.load(open(fit_path))
    uv_params = model.uv_param_names()
    ops       = model.OPERATORS
    cls       = type(model)
    base      = _all_fields(model)
    n_uv      = len(uv_params)
    n_data    = K.shape[0]

    # ── Corrected Jacobian and Q tensors at truth ────────────────────────────
    J_truth  = _jacobian(cls, base, uv_params, ops)         # (n_ops, n_uv)
    lin_corr = (K @ J_truth).T                              # (n_uv, n_data)

    if K_quad is not None:
        # Q_full[i,j,k] = K^(2)[k,a,b] J[a,i] J[b,j]
        Q_full = np.einsum("kab,ai,bj->ijk", K_quad, J_truth, J_truth,
                           optimize="optimal")              # (n_uv, n_uv, n_data)
        quad_corr = np.array([Q_full[i, i, :] for i in range(n_uv)])  # (n_uv, n_data)
        off_diag_corr = Q_full.copy()                                # no factor of 2 — matches
        for i in range(n_uv):                                        # smefit new_QuadraticCorrections
            off_diag_corr[i, i, :] = 0.0
    else:
        quad_corr     = np.zeros((n_uv, n_data))
        off_diag_corr = np.zeros((n_uv, n_uv, n_data))

    # ── NS posterior samples ─────────────────────────────────────────────────
    n_raw  = len(r["samples"][uv_params[0]])
    stride = max(1, n_raw // n_draws)
    idxs   = list(range(0, n_raw, stride))
    n_used = len(idxs)
    print(f"  [quad] {n_used}/{n_raw} NS draws (stride {stride})")

    posterior_g = np.array(                                 # (n_used, n_uv)
        [[r["samples"][p][s] for p in uv_params] for s in idxs])

    # Displacements from truth: the smefit formula is derived for variables
    # measured from the reference point (c=0 in SMEFT space).  Here our
    # reference is g_truth, so all posterior moments must use Δg = g - g_truth.
    truth_vec   = np.array([base[p] for p in uv_params])   # (n_uv,)
    posterior_x = posterior_g - truth_vec[np.newaxis, :]   # (n_used, n_uv)

    c_mean  = np.mean(posterior_x,    axis=0)               # (n_uv,)
    c2_mean = np.mean(posterior_x**2, axis=0)               # (n_uv,)

    # tmp[r,j,k] = Σ_i Δg_r[i] · off_diag_corr[i,j,k]
    tmp   = np.einsum("ri,ijk->rjk", posterior_x, off_diag_corr,
                      optimize="optimal")                   # (n_used, n_uv, n_data)
    A_all = np.mean(tmp, axis=0)                            # (n_uv, n_data)
    B_all = (np.einsum("rj,rjk->jk", posterior_x, tmp,
                       optimize="optimal") / n_used)        # (n_uv, n_data)
    D_all = (np.einsum("rjk,rjl->jkl", tmp, tmp,
                       optimize="optimal") / n_used)        # (n_uv, n_data, n_data)

    # ── delta_th = BSM_signal − mean(K c(g_s) + K^(2) c(g_s)^2) ────────────
    # Mirrors smefit: delta_th = Commondata − mean(smeft_predictions),
    # where smeft_predictions includes both linear and quadratic SMEFT terms.
    proj_dir = str(PIPELINE / "projections" / tag)
    bsm_signal = _build_delta(proj_dir)                     # (n_data,)
    mean_sigma = np.zeros(n_data)
    for s in idxs:
        fields_s = {**base, **{p: r["samples"][p][s] for p in uv_params}}
        c_s = np.array([cls(**fields_s).eft_coefficients().get(op, 0.) for op in ops])
        mean_sigma += K @ c_s
        if K_quad is not None:
            mean_sigma += np.einsum("kab,a,b->k", K_quad, c_s, c_s,
                                    optimize="optimal")
    mean_sigma /= n_used
    delta_th = bsm_signal - mean_sigma                      # (n_data,)

    # ── smefit formula per dataset ───────────────────────────────────────────
    F = np.zeros((n_uv, len(GROUP_ORDER)))

    for ig, grp in enumerate(GROUP_ORDER):
        for ds_name in ENERGY_GROUPS[grp]:
            if ds_name not in ds_rows:
                continue
            r0, r1   = ds_rows[ds_name]
            sl       = slice(r0, r1)
            qc       = quad_corr[:, sl]                     # (n_uv, ndat)
            lc       = lin_corr[:, sl]                      # (n_uv, ndat)
            inv_c    = Ci[sl, sl]                           # (ndat, ndat)
            delta    = delta_th[sl]                         # (ndat,)
            A        = A_all[:, sl]                         # (n_uv, ndat)
            B        = B_all[:, sl]                         # (n_uv, ndat)
            D        = D_all[:, sl, sl]                     # (n_uv, ndat, ndat)

            fisher_mat = (
                - qc @ inv_c @ delta
                - delta @ inv_c @ qc.T
                + lc @ inv_c @ A.T
                + A @ inv_c @ lc.T
                + 2 * c_mean @ (lc @ inv_c @ qc.T + qc @ inv_c @ lc.T)
                + 2 * (B @ inv_c @ qc.T + qc @ inv_c @ B.T)
                + 4 * c2_mean @ qc @ inv_c @ qc.T
                + np.einsum("ikl,kl->i", D, inv_c, optimize="optimal")
            )
            quad_diag = np.diag(fisher_mat)                 # (n_uv,)
            lin_diag  = np.diag(lc @ inv_c @ lc.T)         # (n_uv,)
            F[:, ig] += quad_diag + lin_diag                # quad + linear = smefit total

    return _normalise(F)


# ── Plotting — matches smefit fisher.py appearance ───────────────────────────

def _make_cmap_norm():
    base = plt.get_cmap("Blues")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "trunc_blues", base(np.linspace(0, 0.8, 100)))
    norm = mcolors.BoundaryNorm(np.arange(110, step=10), cmap.N)
    return cmap, norm


def _plot_values(ax, F_norm, cmap, norm):
    n_uv, n_grp = F_norm.shape
    for i in range(n_uv):
        for j in range(n_grp):
            val = F_norm[i, j]
            x, y = j, n_uv - 1 - i
            if val > 0:
                ax.text(x, y, f"{val:.1f}", va="center", ha="center", fontsize=10)
                ax.add_patch(Polygon(
                    [[x-.5, y-.5], [x+.5, y-.5], [x+.5, y+.5], [x-.5, y+.5]],
                    closed=True, ec="grey", color=cmap(norm(val)),
                ))
    ax.set_xlim(-0.5, n_grp - 0.5)
    ax.set_ylim(-0.5, n_uv  - 0.5)
    ax.set_aspect("equal", adjustable="box")


def _set_ticks(ax, n_uv, n_grp, y_labels, x_labels):
    yt, xt = np.arange(n_uv), np.arange(n_grp)
    ax.set_yticks(yt, labels=y_labels[::-1], fontsize=15)
    ax.set_xticks(xt, labels=x_labels, rotation=90, fontsize=15)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(which="major", top=False, bottom=False, left=False)
    ax.set_xticks(xt - 0.5, minor=True)
    ax.set_yticks(yt - 0.5, minor=True)
    ax.tick_params(which="minor", bottom=False)
    ax.grid(visible=True, which="minor", alpha=0.2)


def plot_heatmap(F_lin, F_quad, uv_params, title, out_path, figsize=None):
    cmap, norm = _make_cmap_norm()
    y_labels   = [UV_LATEX.get(p, p) for p in uv_params]
    two_panels = F_quad is not None

    if figsize is None:
        figsize = (12, 7) if two_panels else (7, 8)

    fig = plt.figure(figsize=figsize)

    if two_panels:
        # GridSpec with equal-width panel columns + narrow colorbar column.
        # make_axes_locatable steals space from the host axes, causing the
        # right panel to be smaller; explicit width ratios avoid this.
        gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.08], wspace=0.5)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        cax = fig.add_subplot(gs[2])
    else:
        gs  = fig.add_gridspec(1, 2, width_ratios=[1, 0.08], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        cax = fig.add_subplot(gs[1])

    _plot_values(ax1, F_lin, cmap, norm)
    _set_ticks(ax1, len(uv_params), len(GROUP_ORDER), y_labels, GROUP_ORDER)
    ax1.set_title("Linear", fontsize=20, y=-0.18)

    if two_panels:
        _plot_values(ax2, F_quad, cmap, norm)
        _set_ticks(ax2, len(uv_params), len(GROUP_ORDER), y_labels, GROUP_ORDER)
        ax2.set_title("Quadratic", fontsize=20, y=-0.18)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("Normalized Value", fontsize=20, labelpad=30, rotation=270)

    fig.subplots_adjust(top=0.9)
    if title:
        plt.suptitle(title, fontsize=18, y=0.98)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out,                     bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"),  bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def _make_title(model):
    if hasattr(model, "mWp"):
        if hasattr(model, "g"):
            coupling = rf"$g={model.g}$"
        else:
            coupling = rf"$g_{{WH}}={model.gWH}$"
        return rf"Fisher information: W$'$ ({coupling}, $m_{{W'}}={model.mWp}$ TeV)"
    if hasattr(model, "mZp"):
        return (r"Fisher information: Z$'$ "
                rf"($g_{{ZH}}={model.gZH}$, $m_{{Z'}}={model.mZp}$ TeV)")
    return "Fisher information"


def _default_out(tag):
    plots_dir = RESULTS / tag / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / "fisher_uv.png"
    report_meta = RESULTS / tag / "reports" / f"Report_{tag}_UVcoup" / "meta"
    copy = report_meta / f"fisher_heatmap_{tag}_UVcoup_corrected.png" if report_meta.exists() else None
    return str(out), str(copy) if copy else None


def _print_table(label, F_norm, uv_params):
    print(f"\n{label}:")
    print(f"{'':12s}" + "".join(f"{g:>10s}" for g in GROUP_ORDER))
    for i, p in enumerate(uv_params):
        print(f"{p:12s}" + "".join(f"{F_norm[i,j]:10.1f}" for j in range(len(GROUP_ORDER))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag",      help="Pipeline run tag, e.g. wprime_gwh050_mwp100")
    ap.add_argument("--model",    choices=["wprime_1g", "wprime", "wprime_constrained",
                                           "wprime_universal", "zprime", "zprime_constrained",
                                           "zprime2", "zprime2_constrained"])
    ap.add_argument("--g",    type=float, help="universal coupling for wprime_1g")
    ap.add_argument("--gWH",  type=float)
    ap.add_argument("--mWp",  type=float)
    ap.add_argument("--gWLf", type=float, default=None, help="default: gWH/3")
    ap.add_argument("--gWqf", type=float, default=None, help="default: gWH/3")
    ap.add_argument("--gZH",  type=float)
    ap.add_argument("--gZl",  type=float, default=None, help="default: gZH/3")
    ap.add_argument("--mZp",  type=float)
    ap.add_argument("--ns-draws", type=int, default=500, metavar="N",
                    help="NS posterior draws for quadratic Fisher (default 500)")
    ap.add_argument("--out",      help="Override output path")
    ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args()

    if args.tag:
        model_name, params = _parse_tag(args.tag)
        model = _build_model(model_name, **params)
        tag   = args.tag
    elif args.model:
        model = _build_model(args.model, g=args.g, gWH=args.gWH, mWp=args.mWp,
                             gWLf=args.gWLf, gWqf=args.gWqf,
                             gZH=args.gZH, gZl=args.gZl, mZp=args.mZp)
        if args.model == "wprime_1g":
            tag = f"wprime_1g_g{int(args.g*100):03d}_mwp{int(args.mWp*10):03d}"
        else:
            gv = args.gWH or args.gZH
            mv = args.mWp or args.mZp
            gk = "gwh" if "wprime" in args.model else "gzh"
            mk = "mwp" if "wprime" in args.model else "mzp"
            tag = f"{args.model}_{gk}{int(gv*100):03d}_{mk}{int(mv*10):03d}"
    else:
        ap.error("Provide --tag or --model + coupling/mass flags.")

    print(f"\nModel : {model}")
    print(f"Truth : {model.uv_truth()}")

    K, Ci, ds_rows = _build_K_ds_rows(model)
    uv_params      = model.uv_param_names()

    print("\n[Linear Fisher] evaluating Jacobian at UV truth point ...")
    F_lin = compute_fisher_lin(model, K, Ci, ds_rows)
    _print_table("Linear [%]", F_lin, uv_params)

    print("\n[Quadratic Fisher] loading K^(2) and averaging over NS posterior ...")
    K_quad = _load_K_quad(model.OPERATORS, ds_rows)
    F_quad = compute_fisher_quad(model, K, K_quad, Ci, ds_rows, tag, n_draws=args.ns_draws)
    if F_quad is not None:
        _print_table("Quadratic [%]", F_quad, uv_params)

    title = "" if args.no_title else _make_title(model)

    if args.out:
        plot_heatmap(F_lin, F_quad, uv_params, title, args.out)
    else:
        primary, report_copy = _default_out(tag)
        plot_heatmap(F_lin, F_quad, uv_params, title, primary)
        if report_copy:
            plot_heatmap(F_lin, F_quad, uv_params, title, report_copy)


if __name__ == "__main__":
    main()
