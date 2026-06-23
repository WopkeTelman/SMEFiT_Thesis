"""
scripts/fingerprint_comparison.py

Plots the W' vs Z' operator fingerprints side by side.

For each model the Fisher-gradient fingerprint is:

    n_i = (F c)_i / || F c ||

where c_i are the EFT Wilson coefficients at the truth point,
F = K^T C^{-1} K is the Fisher information matrix, and F c is the
gradient of chi2 with respect to the operator coefficients.  Using the
full gradient (rather than just the diagonal of F) correctly accounts
for operator correlations.

The cosine similarity is computed in the *shared data space*:

    cos(theta) = s_W^T C^{-1} s_Z / sqrt(lambda_W * lambda_Z)

where s = K c is the BSM signal vector in the FCC-ee observable space
and lambda = s^T C^{-1} s is the non-centrality (expected chi^2).
Both models are evaluated against the same FCC-ee dataset and the same
inverse covariance C^{-1}, so cos(theta) is a genuine measure of how
similar the predicted BSM deviations look to the experiment.

cos = 1  : models produce identical FCC-ee deviations (indistinguishable)
cos = 0  : orthogonal deviations (maximally distinguishable)
cos = -1 : opposite-sign deviations (also fully distinguishable)

Operators are ordered: W'-only | shared | Z'-only.

Usage:
    python scripts/fingerprint_comparison.py
    python scripts/fingerprint_comparison.py --gstar 0.18 --mWp 10.0 --mZp 10.0
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE / "scripts"))
sys.path.insert(0, str(PIPELINE))

from models.wprime_constrained import WPrimeConstrainedModel
from models.zprime_constrained import ZPrimeConstrainedModel

# Import the shared K/Ci builder from run_pipeline so both models are
# evaluated against the exact same FCC-ee dataset and covariance.
from run_pipeline import _build_K_Ci

OP_LABELS = {
    "O3pQ3":   r"$\mathcal{O}_{3pQ3}$",
    "O3pl1":   r"$\mathcal{O}_{3pl1}$",
    "O3pl2":   r"$\mathcal{O}_{3pl2}$",
    "O3pl3":   r"$\mathcal{O}_{3pl3}$",
    "OQl13":   r"$\mathcal{O}_{Ql}^{13}$",
    "OQl1M":   r"$\mathcal{O}_{Ql}^{1(-)}$",
    "Obp":     r"$\mathcal{O}_{bW}$",
    "Oll1111": r"$\mathcal{O}_{ll}^{1111}$",
    "Oll1122": r"$\mathcal{O}_{ll}^{1122}$",
    "Oll1221": r"$\mathcal{O}_{ll}^{1221}$",
    "Oll1331": r"$\mathcal{O}_{ll}^{1331}$",
    "OpBox":   r"$\mathcal{O}_{\varphi,\mathrm{box}}$",
    "OpQM":    r"$\mathcal{O}_{\varphi Q}^{(-)}$",
    "Otap":    r"$\mathcal{O}_{t\varphi}$",
    "Otp":     r"$\mathcal{O}_{tW}$",
    "OpD":     r"$\mathcal{O}_{\varphi D}$",
    "Opl1":    r"$\mathcal{O}_{\varphi l_1}$",
    "Opl2":    r"$\mathcal{O}_{\varphi l_2}$",
    "Opl3":    r"$\mathcal{O}_{\varphi l_3}$",
}


def compute_fingerprint(model, K, Ci):
    """
    Compute the Fisher-gradient fingerprint for a UV model.

    Parameters
    ----------
    model : UV model instance with .OPERATORS and .eft_coefficients()
    K     : (n_obs, n_ops) theory matrix for this model's operators,
            built from the shared FCC-ee dataset
    Ci    : (n_obs, n_obs) inverse covariance from the same dataset

    Returns
    -------
    n    : (n_ops,) unit-norm Fisher-gradient fingerprint
    lam  : non-centrality  lambda = c^T F c  (expected chi2 for this signal)
    ops  : list of operator names
    c    : (n_ops,) EFT Wilson coefficients at the truth point
    """
    ops = model.OPERATORS
    c   = np.array([model.eft_coefficients().get(op, 0.0) for op in ops])
    F   = K.T @ Ci @ K          # (n_ops, n_ops) Fisher matrix
    lam = float(c @ F @ c)      # non-centrality parameter

    # Full Fisher gradient: direction of steepest chi2 ascent in operator space.
    # Using F c rather than c * sqrt(diag F) correctly accounts for operator
    # correlations (off-diagonal F elements).
    v    = F @ c
    norm = np.linalg.norm(v)
    n    = v / norm if norm > 0 else np.zeros_like(v)
    return n, lam, ops, c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gstar", type=float, default=0.18)
    p.add_argument("--mWp",   type=float, default=10.0)
    p.add_argument("--mZp",   type=float, default=10.0)
    p.add_argument("--gZl",   type=float, default=None,
                   help="Z' lepton coupling (default: same as --gstar)")
    args = p.parse_args()

    g   = args.gstar
    mWp = args.mWp
    mZp = args.mZp
    gZl = args.gZl if args.gZl is not None else g

    # W' gauge-universal scenario: gWLf = gWqf = gWH / 3.
    # Setting all couplings equal would overestimate lepton/quark operators by 3x.
    gWLf = g / 3.0
    wp_model = WPrimeConstrainedModel(
        gWH=g, gWLf11=gWLf, gWLf22=gWLf, gWLf33=gWLf, gWqf33=gWLf, mWp=mWp
    )
    zp_model = ZPrimeConstrainedModel(gZH=g, gZl=gZl, mZp=mZp)

    ops_wp = wp_model.OPERATORS
    ops_zp = zp_model.OPERATORS

    # ── Build K matrices from the *same* FCC-ee dataset ───────────────────────
    # Using a joint operator list ensures both models see the same rows (data
    # points) and the same C^{-1}, making the cosine similarity well-defined.
    all_ops_joint = sorted(set(ops_wp) | set(ops_zp))
    K_joint, Ci_joint = _build_K_Ci(all_ops_joint)

    wp_cols = [all_ops_joint.index(op) for op in ops_wp]
    zp_cols = [all_ops_joint.index(op) for op in ops_zp]
    K_wp = K_joint[:, wp_cols]   # (n_obs, n_wp_ops)
    K_zp = K_joint[:, zp_cols]   # (n_obs, n_zp_ops)

    n_wp, lam_wp, ops_wp, c_wp = compute_fingerprint(wp_model, K_wp, Ci_joint)
    n_zp, lam_zp, ops_zp, c_zp = compute_fingerprint(zp_model, K_zp, Ci_joint)

    sigma_wp = float(np.sqrt(max(lam_wp, 0.0)))
    sigma_zp = float(np.sqrt(max(lam_zp, 0.0)))

    # ── Data-space cosine similarity ──────────────────────────────────────────
    # s = K c is the predicted BSM signal deviation in observable space.
    # cos(theta) = s_W^T C^{-1} s_Z / sqrt(lambda_W * lambda_Z)
    # This is the physically correct metric: it measures how alike the two
    # models look to the FCC-ee experiment.
    s_wp = K_wp @ c_wp
    s_zp = K_zp @ c_zp
    inner   = float(s_wp @ Ci_joint @ s_zp)
    cos_sim = inner / (sigma_wp * sigma_zp) if sigma_wp > 0 and sigma_zp > 0 else 0.0
    cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_sim)))

    # ── Operator grouping for bar chart ───────────────────────────────────────
    wp_set = set(ops_wp)
    zp_set = set(ops_zp)
    shared  = [op for op in ops_wp if op in zp_set]
    wp_only = [op for op in ops_wp if op not in zp_set]
    zp_only = [op for op in ops_zp if op not in wp_set]
    plot_ops = wp_only + shared + zp_only

    def expand(ops_model, vals):
        d = dict(zip(ops_model, vals))
        return np.array([d.get(op, 0.0) for op in plot_ops])

    n_wp_full = expand(ops_wp, n_wp)
    n_zp_full = expand(ops_zp, n_zp)

    print(f"\nW' model: gWH={g}, gWLf=gWqf={gWLf:.4f}, mWp={mWp} TeV")
    print(f"  lambda={lam_wp:.2f}  sqrt(lambda)={sigma_wp:.2f}")
    print(f"Z' model: gZH={g}, gZl={gZl:.4f}, mZp={mZp} TeV")
    print(f"  lambda={lam_zp:.2f}  sqrt(lambda)={sigma_zp:.2f}")
    print(f"\nData-space cosine similarity:  cos(theta) = {cos_sim:.4f}")
    print(f"Angle (data space):            theta = {angle_deg:.1f} deg")
    print(f"\nW'-only operators: {wp_only}")
    print(f"Shared operators:  {shared}")
    print(f"Z'-only operators: {zp_only}")
    print(f"\nNote: sqrt(lambda_Z) / sqrt(lambda_W) = {sigma_zp/sigma_wp:.2f}")
    print(f"  Z' has higher FCC-ee sensitivity because OpD and Opl enter")
    print(f"  Z-pole observables where FCC-ee has per-mille-level precision.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                             gridspec_kw={"width_ratios": [3, 1]})

    ax = axes[0]
    x     = np.arange(len(plot_ops))
    width = 0.38

    n_w = len(wp_only)
    n_s = len(shared)

    ax.axvspan(-0.5,           n_w - 0.5,
               color="#d0e8ff", alpha=0.35, zorder=0, label="W'-only")
    ax.axvspan(n_w - 0.5,      n_w + n_s - 0.5,
               color="#e0e0e0", alpha=0.50, zorder=0, label="Shared")
    ax.axvspan(n_w + n_s - 0.5, len(plot_ops) - 0.5,
               color="#ffe0d0", alpha=0.35, zorder=0, label="Z'-only")

    ax.bar(x - width/2, n_wp_full, width, color="#1f77b4", zorder=3,
           label=fr"W' ($g_{{WH}}={g}$, $g_{{Wf}}={gWLf:.3f}$, "
                 fr"$m_{{W'}}={mWp:.0f}$ TeV, $\sqrt{{\lambda}}={sigma_wp:.1f}$)")
    ax.bar(x + width/2, n_zp_full, width, color="#d62728", zorder=3,
           label=fr"Z' ($g_{{ZH}}=g_{{Zl}}={g}$, "
                 fr"$m_{{Z'}}={mZp:.0f}$ TeV, $\sqrt{{\lambda}}={sigma_zp:.1f}$)")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [OP_LABELS.get(op, op) for op in plot_ops],
        rotation=45, ha="right", fontsize=9
    )
    ax.set_ylabel(
        r"Fisher-gradient fingerprint $\,\tilde{n}_i = (Fc)_i\,/\,\|Fc\|$",
        fontsize=10
    )
    ax.set_title(
        fr"Operator fingerprint: W' vs Z' at FCC-ee  "
        fr"($\cos\theta_\mathrm{{data}} = {cos_sim:.3f}$, "
        fr"$\theta = {angle_deg:.1f}^\circ$)",
        fontsize=11
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_xlim(-0.6, len(plot_ops) - 0.4)

    # --- right: angle diagram ---
    ax2 = axes[1]
    theta = np.radians(angle_deg)
    ax2.annotate("", xy=(np.cos(0), np.sin(0)), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2.5))
    ax2.annotate("", xy=(np.cos(theta), np.sin(theta)), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.5))
    arc_t = np.linspace(0, theta, 60)
    ax2.plot(0.4 * np.cos(arc_t), 0.4 * np.sin(arc_t), "k-", lw=1.2)
    ax2.text(0.45 * np.cos(theta / 2), 0.45 * np.sin(theta / 2),
             fr"$\theta={angle_deg:.0f}^\circ$", fontsize=11,
             ha="center", va="center")
    ax2.text(1.05, 0.0, r"W'", fontsize=11, color="#1f77b4",
             ha="left", va="center")
    ax2.text(np.cos(theta) + 0.05, np.sin(theta),
             r"Z'", fontsize=11, color="#d62728", ha="left", va="center")
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-0.3, 1.3)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title(
        fr"$\cos\theta = {cos_sim:.3f}$" + "\n"
        r"(data space: $s = Kc$," + "\n"
        r"$\cos\theta = s_W^\top C^{-1} s_Z / \sqrt{\lambda_W\lambda_Z}$)",
        fontsize=9
    )

    plt.tight_layout()

    out_dir = PIPELINE / "results" / "fingerprint_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"gwh{int(g*100):03d}_gzl{int(gZl*100):03d}_mwp{int(mWp*10):03d}_mzp{int(mZp*10):03d}"
    for ext in ["png", "pdf"]:
        path = out_dir / f"fingerprint_{tag}.{ext}"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
