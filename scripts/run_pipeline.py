#!/usr/bin/env python3
"""
fullpipeline/scripts/run_pipeline.py

Full BSM discovery pipeline for FCC-ee smefit analysis.
Fully self-contained: projections, fits, report, and scan all in one command.

Single-point run:
    python run_pipeline.py --model zprime --gZH 0.12 --gZl 0.04 --mZp 1.0
    python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0

Discovery scan (analytic, fast):
    python run_pipeline.py --model zprime --gZH 0.12 --gZl 0.04 --mZp 1.0 --scan
    python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0 --scan

Skip the slow NS fit (report + significance still produced):
    python run_pipeline.py --model zprime --gZH 0.12 --gZl 0.04 --mZp 1.0 --no-ns

Separated-projection mode (per-operator signal + posterior):
    python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0 --sepproj OQl13 OpQM
    python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0 --sepproj  (all operators)

Steps (single-point run):
    0. Generate BSM pseudo-data  (data_BSM = data_SM + Σ K_eff_j * c_j, RGE-consistent)
    1. SM-only baseline fit      (analytic, ~1 min)
    2. Full SMEFT fit            (analytic, ~1 min)
    3. UV coupling NS fit        (nested sampling, ~15-30 min, skip with --no-ns)
    4. PCA                       (Fisher eigenvectors from theory DB)
    5. Profile likelihood table  + significance bar chart
    6. smefit report             (posterior histograms, 2D contours, chi2 table)

Scan mode adds:
    7. Loop over (coupling, mass) grid  (analytic fits only, fast)
    8. Discovery region table    (discovery_table.txt)
    9. Discovery region plot     (discovery_region.png)

All outputs under fullpipeline/results/<tag>/
"""

import os, sys, argparse, yaml, json, subprocess, hashlib, functools
import numpy as np
from pathlib import Path
from scipy.stats import chi2 as chi2_dist, norm as norm_dist

# ── Paths ──────────────────────────────────────────────────────────────────────
PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

DB      = "/data/theorie/wtelman/smefit_database"
SMEFIT  = "/data/theorie/wtelman/miniconda3/envs/smefit-dev/bin/smefit"
SM_DATA = f"{DB}/commondata_projections_L0"
THEORY  = f"{DB}/theory"

ENV = {**os.environ,
       "MPLCONFIGDIR": "/data/theorie/wtelman/.mplconfig",
       "XDG_CACHE_HOME": "/data/theorie/wtelman/.cache"}

# FCC-ee datasets used in every fit
# Commented-out entries have negative covariance eigenvalues and are excluded.
DATASETS = [
    # ── Z pole ────────────────────────────────────────────────────────────────
    {"name": "FCCee_Wwidth",               "order": "LO"},
    {"name": "FCCee_Zdata",                "order": "LO"},
    # FCCee_alphaEW removed from database (Feb 2026, merged into FCCee_Zdata)

    # ── WW threshold / cross-sections ─────────────────────────────────────────
    {"name": "FCCee_ww_161GeV",            "order": "LO"},
    {"name": "FCCee_ww_240GeV",            "order": "LO"},
    {"name": "FCCee_ww_365GeV",            "order": "LO"},
    # optimised WW observables — negative covariance eigenvalues, excluded:
    # {"name": "FCCee_161_ww_leptonic_optim_obs", "order": "LO"},
    # {"name": "FCCee_161_ww_semilep_optim_obs",  "order": "LO"},
    # {"name": "FCCee_240_ww_leptonic_optim_obs", "order": "LO"},
    # {"name": "FCCee_240_ww_semilep_optim_obs",  "order": "LO"},
    # {"name": "FCCee_365_ww_leptonic_optim_obs", "order": "LO"},
    # {"name": "FCCee_365_ww_semilep_optim_obs",  "order": "LO"},

    # ── W branching ratio ─────────────────────────────────────────────────────
    {"name": "FCCee_Brw",                  "order": "LO"},

    # ── EWPO @ 240 GeV ────────────────────────────────────────────────────────
    {"name": "FCCee_Rb_240GeV",            "order": "LO"},
    {"name": "FCCee_Rc_240GeV",            "order": "LO"},
    {"name": "FCCee_Rmu_240GeV",           "order": "LO"},
    {"name": "FCCee_Rtau_240GeV",          "order": "LO"},
    {"name": "FCCee_ee_240GeV",            "order": "LO"},
    {"name": "FCCee_ee_Afb_240GeV",        "order": "LO"},
    {"name": "FCCee_bb_Afb_240GeV",        "order": "LO"},
    {"name": "FCCee_cc_Afb_240GeV",        "order": "LO"},
    {"name": "FCCee_mumu_Afb_240GeV",      "order": "LO"},
    {"name": "FCCee_tautau_Afb_240GeV",    "order": "LO"},
    {"name": "FCCee_sigmaHad_240GeV",      "order": "LO"},

    # ── EWPO @ 365 GeV ────────────────────────────────────────────────────────
    {"name": "FCCee_Rb_365GeV",            "order": "LO"},
    {"name": "FCCee_Rc_365GeV",            "order": "LO"},
    {"name": "FCCee_Rmu_365GeV",           "order": "LO"},
    {"name": "FCCee_Rtau_365GeV",          "order": "LO"},
    {"name": "FCCee_ee_365GeV",            "order": "LO"},
    {"name": "FCCee_ee_Afb_365GeV",        "order": "LO"},
    {"name": "FCCee_bb_Afb_365GeV",        "order": "LO"},
    {"name": "FCCee_cc_Afb_365GeV",        "order": "LO"},
    {"name": "FCCee_mumu_Afb_365GeV",      "order": "LO"},
    {"name": "FCCee_tautau_Afb_365GeV",    "order": "LO"},
    {"name": "FCCee_sigmaHad_365GeV",      "order": "LO"},
    # FCCee_365_tt_optim_obs excluded: all 8 central values = 0.0, causes divide-by-zero in theory covmat

    # ── ZH @ 240 GeV ──────────────────────────────────────────────────────────
    {"name": "FCCee_zh_240GeV",            "order": "NLO_EW"},
    {"name": "FCCee_zh_WW_240GeV",         "order": "NLO_EW"},
    {"name": "FCCee_zh_ZZ_240GeV",         "order": "NLO_EW"},
    {"name": "FCCee_zh_aZ_240GeV",         "order": "NLO_EW"},
    {"name": "FCCee_zh_aa_240GeV",         "order": "NLO_EW"},
    {"name": "FCCee_zh_tautau_240GeV",     "order": "NLO_EW"},
    {"name": "FCCee_zh_mumu_240GeV",       "order": "NLO_EW"},
    {"name": "FCCee_240_H_HADR",           "order": "LO"},

    # ── ZH / vvH @ 365 GeV ────────────────────────────────────────────────────
    {"name": "FCCee_zh_365GeV",            "order": "NLO_EW"},
    {"name": "FCCee_zh_WW_365GeV",         "order": "NLO_EW"},
    {"name": "FCCee_zh_ZZ_365GeV",         "order": "NLO_EW"},
    {"name": "FCCee_zh_aZ_365GeV",         "order": "NLO_EW"},
    {"name": "FCCee_zh_aa_365GeV",         "order": "NLO_EW"},
    {"name": "FCCee_zh_tautau_365GeV",     "order": "NLO_EW"},
    {"name": "FCCee_zh_mumu_365GeV",       "order": "NLO_EW"},
    {"name": "FCCee_365_H_HADR",           "order": "LO"},
    {"name": "FCCee_vvh_WW_365GeV",        "order": "LO"},
    {"name": "FCCee_vvh_ZZ_365GeV",        "order": "LO"},
    {"name": "FCCee_vvh_aZ_365GeV",        "order": "LO"},
    {"name": "FCCee_vvh_aa_365GeV",        "order": "LO"},
    {"name": "FCCee_vvh_tautau_365GeV",    "order": "LO"},
]

DS_NAMES  = [d["name"]  for d in DATASETS]
DS_ORDER  = {d["name"]: d.get("order", "LO") for d in DATASETS}

def _model_mass(model) -> float:
    """Return the BSM mass in TeV for any model (W', Z', CompositeHiggs, ...)."""
    return getattr(model, "mWp", None) or getattr(model, "mZp", None) or getattr(model, "m_rho", 1.0)

def make_rge(mWp_TeV: float = 1.0) -> dict | None:
    """RGE settings block, or None when RGE is disabled via --no-rge."""
    if not CFG.use_rge:
        return None
    return {"init_scale": mWp_TeV * 1000.0, "obs_scale": "dynamic",
            "smeft_accuracy": "integrate", "yukawa": "top", "adm_QCD": False}

class _Config:
    """Pipeline-wide settings. Mutate once at startup via CLI args; read everywhere else."""
    use_theory_covmat:    bool = True
    theory_cov_variant:   str  = "theory_cov_aggressive"  # aggressive | conservative | current
    use_rge:              bool = True

CFG = _Config()

# Backwards-compatible alias used in _make_base_runcard and _build_K_Ci
def USE_THEORY_COVMAT():
    return CFG.use_theory_covmat

# Dataset groups for the smefit report
DATA_INFO = {
    "91 GeV (Z-pole)": [["FCCee_Zdata",    "Z pole data"],
                         ["FCCee_Wwidth",   "W width"]],
    "161 GeV (WW thr.)": [["FCCee_ww_161GeV", "WW @ 161 GeV"],
                           ["FCCee_Brw",       "Br(W) @ 161 GeV"]],
    "240 GeV": [["FCCee_ww_240GeV",          "WW @ 240 GeV"],
                ["FCCee_ee_240GeV",          "ee @ 240 GeV"],
                ["FCCee_ee_Afb_240GeV",      "ee Afb @ 240 GeV"],
                ["FCCee_Rb_240GeV",          "Rb @ 240 GeV"],
                ["FCCee_Rc_240GeV",          "Rc @ 240 GeV"],
                ["FCCee_Rmu_240GeV",         "Rmu @ 240 GeV"],
                ["FCCee_Rtau_240GeV",        "Rtau @ 240 GeV"],
                ["FCCee_bb_Afb_240GeV",      "bb Afb @ 240 GeV"],
                ["FCCee_cc_Afb_240GeV",      "cc Afb @ 240 GeV"],
                ["FCCee_mumu_Afb_240GeV",    "mumu Afb @ 240 GeV"],
                ["FCCee_sigmaHad_240GeV",    "sigma_had @ 240 GeV"],
                ["FCCee_tautau_Afb_240GeV",  "tautau Afb @ 240 GeV"],
                ["FCCee_zh_240GeV",          "ZH @ 240 GeV"],
                ["FCCee_zh_WW_240GeV",       "ZH->WW @ 240 GeV"],
                ["FCCee_zh_ZZ_240GeV",       "ZH->ZZ @ 240 GeV"],
                ["FCCee_zh_aZ_240GeV",       "ZH->aZ @ 240 GeV"],
                ["FCCee_zh_aa_240GeV",       "ZH->aa @ 240 GeV"],
                ["FCCee_zh_tautau_240GeV",   "ZH->tautau @ 240 GeV"],
                ["FCCee_zh_mumu_240GeV",     "ZH->mumu @ 240 GeV"],
                ["FCCee_240_H_HADR",         "H hadronic @ 240 GeV"]],
    "365 GeV": [["FCCee_ww_365GeV",          "WW @ 365 GeV"],
                ["FCCee_ee_365GeV",          "ee @ 365 GeV"],
                ["FCCee_ee_Afb_365GeV",      "ee Afb @ 365 GeV"],
                ["FCCee_Rb_365GeV",          "Rb @ 365 GeV"],
                ["FCCee_Rc_365GeV",          "Rc @ 365 GeV"],
                ["FCCee_Rmu_365GeV",         "Rmu @ 365 GeV"],
                ["FCCee_Rtau_365GeV",        "Rtau @ 365 GeV"],
                ["FCCee_bb_Afb_365GeV",      "bb Afb @ 365 GeV"],
                ["FCCee_cc_Afb_365GeV",      "cc Afb @ 365 GeV"],
                ["FCCee_mumu_Afb_365GeV",    "mumu Afb @ 365 GeV"],
                ["FCCee_sigmaHad_365GeV",    "sigma_had @ 365 GeV"],
                ["FCCee_tautau_Afb_365GeV",  "tautau Afb @ 365 GeV"],
                ["FCCee_zh_365GeV",          "ZH @ 365 GeV"],
                ["FCCee_zh_WW_365GeV",       "ZH->WW @ 365 GeV"],
                ["FCCee_zh_ZZ_365GeV",       "ZH->ZZ @ 365 GeV"],
                ["FCCee_zh_aZ_365GeV",       "ZH->aZ @ 365 GeV"],
                ["FCCee_zh_aa_365GeV",       "ZH->aa @ 365 GeV"],
                ["FCCee_zh_tautau_365GeV",   "ZH->tautau @ 365 GeV"],
                ["FCCee_zh_mumu_365GeV",     "ZH->mumu @ 365 GeV"],
                ["FCCee_365_H_HADR",         "H hadronic @ 365 GeV"],
                ["FCCee_vvh_WW_365GeV",      "vvH->WW @ 365 GeV"],
                ["FCCee_vvh_ZZ_365GeV",      "vvH->ZZ @ 365 GeV"],
                ["FCCee_vvh_aZ_365GeV",      "vvH->aZ @ 365 GeV"],
                ["FCCee_vvh_aa_365GeV",      "vvH->aa @ 365 GeV"],
                ["FCCee_vvh_tautau_365GeV",  "vvH->tautau @ 365 GeV"]],
}

# LaTeX labels per operator
LATEX = {
    # W' operators
    "OpBox":   r"$c_{H\Box}$",    "Op":      r"$c_H$",
    "Obp":     r"$c_{HB}$",       "Otp":     r"$c_{Ht}$",
    "Otap":    r"$c_{H\tilde{t}}$","OpQM":   r"$c_{HQ}^{(1)}$",
    "O3pQ3":   r"$c_{Hq,33}^{(3)}$","O3pl1": r"$c_{Hl,11}^{(3)}$",
    "O3pl2":   r"$c_{Hl,22}^{(3)}$","O3pl3": r"$c_{Hl,33}^{(3)}$",
    "OQQ1":    r"$c_{QQ}^{(1)}$", "OQQ8":    r"$c_{QQ}^{(8)}$",
    "OQl13":   r"$c_{Ql,13}$",    "OQl1M":   r"$c_{Ql,1M}$",
    "OQl33":   r"$c_{Ql,33}$",    "OQl3M":   r"$c_{Ql,3M}$",
    "Oll1111": r"$c_{ll,1111}$",  "Oll1221": r"$c_{ll,1221}$",
    "Oll1122": r"$c_{ll,1122}$",  "Oll1133": r"$c_{ll,1133}$",
    "Oll1331": r"$c_{ll,1331}$",  "Oll2222": r"$c_{ll,2222}$",
    "Oll2233": r"$c_{ll,2233}$",  "Oll2332": r"$c_{ll,2332}$",
    "Oll3333": r"$c_{ll,3333}$",
    # Z' operators
    "OpD":     r"$c_{HD}$",
    "Opl1":    r"$c_{Hl,1}^{(1)}$","Opl2":  r"$c_{Hl,2}^{(1)}$",
    "Opl3":    r"$c_{Hl,3}^{(1)}$",
}


# ── Step 0: Projection generation via smefit PROJ ─────────────────────────────

def generate_projections(model, proj_dir: str, rc_dir: str = None, noise_level: str = "L0"):
    """
    Generate BSM pseudo-data using smefit PROJ (closure test mode, noise=L0).

    Builds a PROJ runcard from model.eft_coefficients() and calls:
        smefit PROJ <runcard> --noise L0

    smefit PROJ with RGE enabled applies the same RGE-evolved K matrix as the
    A/NS fits, so the injected signal is exactly self-consistent with what
    smefit predicts for c_j = c_truth.  Posteriors centre on the truth values.

    Output: one yaml per dataset in proj_dir/, named {dataset}.yaml.
    """
    os.makedirs(proj_dir, exist_ok=True)
    if rc_dir is None:
        rc_dir = proj_dir
    os.makedirs(rc_dir, exist_ok=True)

    c_eft = model.eft_coefficients()

    print(f"\n  [Step 0] Generating projections via smefit PROJ for {model}")
    print(f"    EFT coefficients:")
    for op, c in c_eft.items():
        print(f"      {op:12s}: {c:+.4e} TeV^-2")

    # Build PROJ runcard
    # Each operator is expressed as a fixed value (constrain: True, value: c)
    proj_rc = {
        "projections_path": proj_dir,
        "commondata_path":  SM_DATA,
        "theory_path":      THEORY,
        "datasets":         DATASETS,
        "coefficients":     {op: {"constrain": True, "value": float(c)}
                             for op, c in c_eft.items() if abs(c) > 1e-15},
        "use_quad":         True,
        "use_t0":           False,
        "use_theory_covmat": False,
        **( {"rge": make_rge(_model_mass(model))} if make_rge(_model_mass(model)) is not None else {} ),
    }

    rc_path = f"{rc_dir}/proj_{os.path.basename(proj_dir)}.yaml"
    with open(rc_path, "w") as f:
        yaml.dump(proj_rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"    Running: smefit PROJ {rc_path} --noise {noise_level}")
    r = subprocess.run([SMEFIT, "PROJ", rc_path, "--noise", noise_level], env=ENV)
    if r.returncode != 0:
        raise RuntimeError(f"smefit PROJ failed (exit {r.returncode})")

    n_written = sum(1 for ds in DS_NAMES
                    if os.path.exists(f"{proj_dir}/{ds}.yaml"))
    print(f"    {n_written}/{len(DS_NAMES)} datasets written to: {proj_dir}")


# ── Runcard builders ───────────────────────────────────────────────────────────

def _datasets_with_theory_cov():
    """Return DATASETS with theory_cov variant injected so smefit uses the right variant.

    smefit selects the theory covmat key as f"theory_cov_{ds.get('theory_cov','current')}".
    Without this, use_theory_covmat=True silently defaults to theory_cov_current regardless
    of CFG.theory_cov_variant. Ignored when use_theory_covmat=False.
    """
    if not CFG.use_theory_covmat:
        return DATASETS
    variant = CFG.theory_cov_variant.replace("theory_cov_", "")  # e.g. "aggressive"
    return [{**ds, "theory_cov": variant} for ds in DATASETS]


def _make_base_runcard(result_id, data_path, ops, use_quad=False, uv=False, mWp_TeV=1.0):
    return {
        "result_ID":         result_id,
        "result_path":       None,
        "data_path":         data_path,
        "theory_path":       THEORY,
        "use_quad":          use_quad,
        "use_t0":            CFG.use_theory_covmat,   # t0 only needed when theory covmat is on
        "use_theory_covmat": CFG.use_theory_covmat,
        "uv_couplings":      uv,
        "n_samples":         10000,
        "datasets":          _datasets_with_theory_cov(),
        **( {"rge": make_rge(mWp_TeV)} if make_rge(mWp_TeV) is not None else {} ),
    }

def make_smonly_runcard(result_id, ops, mWp_TeV=1.0):
    """SM-only fit: uses SM data (not BSM) → baseline posterior centered at zero."""
    rc = _make_base_runcard(result_id, SM_DATA, ops, mWp_TeV=mWp_TeV)
    rc["coefficients"] = {op: {"min": -1.0, "max": 1.0} for op in ops}
    return rc

def make_smeft_runcard(result_id, data_path, model, free_ops=None):
    """SMEFT fit: uses BSM data with priors scaled to the model's signal size.

    free_ops: if given, only these operators are free in the fit (subset test).
              The pseudo-data still contains contributions from all model operators.
    """
    ops   = free_ops if free_ops is not None else model.OPERATORS
    truth = model.eft_coefficients()
    # Set prior width to 20× the truth value (min 0.01, max 1.0) so posteriors
    # are well-resolved around the signal without being artificially narrow.
    rc = _make_base_runcard(result_id, data_path, ops, mWp_TeV=_model_mass(model))
    rc["coefficients"] = {
        op: {"min": -max(0.01, min(1.0, 20 * abs(truth.get(op, 0.01)))),
             "max":  max(0.01, min(1.0, 20 * abs(truth.get(op, 0.01))))}
        for op in ops
    }
    return rc

def make_uv_runcard(result_id, data_path, model, use_quad=True):
    rc = _make_base_runcard(result_id, data_path, model.OPERATORS,
                            use_quad=use_quad, uv=True, mWp_TeV=_model_mass(model))
    rc["coefficients"]        = model.uv_coeff_block()
    rc["nlive"]               = 300
    rc["lepsilon"]            = 0.01
    rc["target_evidence_unc"] = 0.1
    rc["target_post_unc"]     = 0.1
    rc["frac_remain"]         = 0.01
    return rc

def make_report_runcard(tag, proj_dir, fit_dir, report_dir, model, free_ops=None):
    """
    Build a smefit report runcard (smefit R) for this pipeline run.
    Compares SM-only, SMEFT, and UV fits with truth overlay.

    free_ops: if set, report only shows these operators (subset fit mode).
    """
    ops        = free_ops if free_ops is not None else model.OPERATORS
    truth      = model.eft_coefficients()
    sm_id      = f"{tag}_SMonly"
    bsm_id     = f"{tag}_BSMclosure_SMEFT"
    uv_id      = f"{tag}_UVcoup"

    # Only include EFT fits (SM-only and SMEFT) in the main report.
    # The UV coupling NS fit uses UV couplings (gZH, gZl) as free parameters
    # rather than EFT operators, so smefit's operator-level plots (Fisher,
    # correlations, PCA, histograms) would fail or be misleading if it's included.
    # UV fit significance is already captured in Step 5's profile likelihood table.
    result_ids, fit_labels = [], []
    for fid, label in [(sm_id,  "SM-only hypothesis"),
                       (bsm_id, f"SMEFT fit ({tag})")]:
        if os.path.exists(f"{fit_dir}/{fid}/fit_results.json"):
            result_ids.append(fid)
            fit_labels.append(label)

    if not result_ids:
        print("  [R] No completed fits found — skipping report")
        return None

    # Operator info block (name + LaTeX label)
    coeff_info = [[op, LATEX.get(op, op)] for op in ops]

    # Truth overlay for posterior histograms: None for SM-only, truth for others
    truth_point = {op: float(truth.get(op, 0.0)) for op in ops}
    closure_truth  = [None if fid == sm_id else truth_point  for fid in result_ids]
    closure_labels = [""   if fid == sm_id else f"UV truth: {model}" for fid in result_ids]
    closure_colors = ["tab:gray" if fid == sm_id else "red"  for fid in result_ids]
    closure_styles = ["--"] * len(result_ids)

    # 2D contour operators: pick the top operators by signal size
    sorted_ops = sorted(ops, key=lambda op: abs(truth.get(op, 0.0)), reverse=True)
    contour_ops = sorted_ops[:min(6, len(sorted_ops))]

    # contours_2d in smefit loops over ALL fits and indexes their posterior
    # DataFrames by the same dofs_show.  UV NS fits have UV couplings as
    # free parameters (gZH, gZl), while analytic EFT fits have operator
    # names.  Mixing both in result_ids with a shared dofs_show always causes
    # a KeyError — so disable contours_2d when the report contains both EFT
    # and UV fits.  Posterior histograms and scatter plots still work fine.
    has_ns_fit = os.path.exists(f"{fit_dir}/{uv_id}/fit_results.json")
    has_eft_fit = any(
        os.path.exists(f"{fit_dir}/{fid}/fit_results.json")
        for fid in [sm_id, bsm_id]
    )
    # Enable contours_2d only when ALL fits are UV-type (no EFT analytic fits)
    can_do_contours = has_ns_fit and not has_eft_fit

    # Always include contours_2d for EFT fits (analytic fits have Gaussian covariance)
    # Show all operators — let smefit decide the layout
    coeff_plots = {
        "contours_2d": {
            "show":             True,
            "confidence_level": 95,
            "dofs_show":        ops,
        }
    }

    rc = {
        "name":        f"Report_{tag}",
        "title":       f"BSM discovery: {model} | {tag}",
        "result_IDs":  result_ids,
        "fit_labels":  fit_labels,
        "report_path": report_dir,
        "result_path": fit_dir,
        "summary":     False,
        "summary_only": False,
        "coefficients_plots": {
            **coeff_plots,
            "scatter_plot": {
                "figsize": [10, max(6, len(ops))],
                "x_min":   -0.05,
                "x_max":   0.05,
                "lin_thr": 0.001,
                "x_log":   False,
            },
            "posterior_histograms": {
                "nrows":               -1,
                "disjointed_lists":    False,
                "bins":                45,
                "show_closure_truth":  True,
                "closure_truth_points": closure_truth,
                "closure_line_color":  closure_colors,
                "closure_line_style":  closure_styles,
                "show_closure_legend": True,
                "closure_line_labels": closure_labels,
            },
            "logo": False,
        },
        "chi2_plots": {
            "table": True,
            "plot_experiment": {"figsize": [12, 10]},
            "plot_distribution": {"figsize": [8, 5]},
        },
        "data_vs_theory": {
            "panel":           "pull",
            "include_sm":      True,
            "include_best_fit": True,
            "per_dataset":     False,
        },
        "correlations": {
            "thr_show": 0.1,
            "fit_list": [fid for fid in [bsm_id] if fid in result_ids],
        },
        "PCA": {
            "table": True,
            "plot":  {"heatmap": True, "figsize": [max(10, len(ops) + 2), 8]},
            "fit_list": [fid for fid in [bsm_id] if fid in result_ids],
        },
        "fisher": {
            "norm":         "coeff",
            "summary_only": False,
            "log":          True,
            "plot":         {"title": True, "figsize": [14, 10], "summary_only": True},
        },
        "plot":      False,
        "coeff_info": {"benchmark_smeft": coeff_info},
        "data_info": DATA_INFO,
    }
    return rc


def make_uv_report_runcard(tag, fit_dir, report_dir, model):
    """
    Build a smefit report runcard for the UV coupling NS fit only.
    Shows UV parameter posteriors (gWH, gWLf, ...) with truth lines,
    2D contours, chi2 and data-vs-theory plots.
    Mirrors the structure of Report_gwh008_UVcoup_RGE.yaml.
    """
    uv_id     = f"{tag}_UVcoup"
    uv_params = model.uv_param_names() if hasattr(model, "uv_param_names") else []
    truth_uv  = model.uv_truth()       if hasattr(model, "uv_truth")       else {}

    if not os.path.exists(f"{fit_dir}/{uv_id}/fit_results.json"):
        return None

    # coeff_info: LaTeX labels for UV parameters
    uv_latex = {
        "gWH":    r"$g_{WH}$",
        "gWLf11": r"$g_{WLf,11}$",
        "gWLf22": r"$g_{WLf,22}$",
        "gWLf33": r"$g_{WLf,33}$",
        "gWqf33": r"$g_{Wqf,33}$",
        "gZH":    r"$g_{ZH}$",
        "gZl":    r"$g_{Zl}$",
    }
    coeff_info = [[p, uv_latex.get(p, p)] for p in uv_params]
    truth_point = {p: float(truth_uv.get(p, 0.0)) for p in uv_params}

    rc = {
        "name":        f"Report_{tag}_UVcoup",
        "title":       f"W' UV coupling fit — {tag}",
        "result_IDs":  [uv_id],
        "fit_labels":  [f"BSM UV closure ({tag})"],
        "report_path": report_dir,
        "result_path": fit_dir,
        "summary":     False,
        "summary_only": False,
        "coefficients_plots": {
            "contours_2d": (
                {
                    "show":             True,
                    "confidence_level": 95,
                    "dofs_show":        uv_params[:min(6, len(uv_params))],
                }
                if len(uv_params) >= 2 else None
            ),
            "posterior_histograms": {
                "nrows":               -1,
                "disjointed_lists":    False,
                "bins":                60,
                "show_closure_truth":  True,
                "closure_truth_points": [truth_point],
                "closure_line_color":  ["red"],
                "closure_line_style":  ["--"],
                "show_closure_legend": True,
                "closure_line_labels": [f"UV truth ({tag})"],
            },
            "logo": False,
        },
        "chi2_plots": {
            "table": True,
            "plot_experiment":   {"figsize": [12, 8]},
            "plot_distribution": {"figsize": [8, 5]},
        },
        "fisher": {
            "norm":         "coeff",
            "summary_only": False,
            "log":          False,
            "plot":         {"title": True, "figsize": [10, 6], "summary_only": True},
        },
        "data_vs_theory": {
            "panel":            "pull",
            "include_sm":       True,
            "include_best_fit": True,
            "per_dataset":      False,
        },
        "plot":      False,
        "coeff_info": {"default": coeff_info},
        "data_info": DATA_INFO,
    }
    return rc


# ── Step 0b: Single-operator projection (sepproj) ─────────────────────────────

def generate_sepproj_projection(model, op, proj_dir, rc_dir):
    """
    Generate BSM pseudo-data injecting only one operator at its UV truth value.
    Used by --sepproj to isolate per-operator signal.
    """
    c_truth = model.eft_coefficients()
    c_op    = c_truth.get(op, 0.0)
    if abs(c_op) < 1e-15:
        print(f"  [sepproj] {op}: truth value ≈ 0 — skipping")
        return False

    os.makedirs(proj_dir, exist_ok=True)
    os.makedirs(rc_dir,   exist_ok=True)

    proj_rc = {
        "projections_path": proj_dir,
        "commondata_path":  SM_DATA,
        "theory_path":      THEORY,
        "datasets":         DATASETS,
        "coefficients":     {op: {"constrain": True, "value": float(c_op)}},
        "use_quad":         True,
        "use_t0":           False,
        "use_theory_covmat": False,
        **( {"rge": make_rge(_model_mass(model))} if make_rge(_model_mass(model)) is not None else {} ),
    }
    rc_path = f"{rc_dir}/proj_sepproj_{op}.yaml"
    with open(rc_path, "w") as f:
        yaml.dump(proj_rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  Running: smefit PROJ {rc_path} --noise L0  [{op}={c_op:.4e}]")
    r = subprocess.run([SMEFIT, "PROJ", rc_path, "--noise", "L0"], env=ENV)
    if r.returncode != 0:
        print(f"  WARN: smefit PROJ failed for {op} (exit {r.returncode})")
        return False
    return True


def make_sepproj_fit_runcard(result_id, data_path, op):
    """Analytic fit runcard with a single operator free; wide flat prior."""
    rc = _make_base_runcard(result_id, data_path, [op])
    rc["coefficients"] = {op: {"min": -100.0, "max": 100.0}}
    return rc


def make_sepproj_report_runcard(tag, op, sm_id, bsm_id, fit_dir, report_dir, model):
    """smefit R runcard comparing SM-only vs BSM-closure for a single operator."""
    c_truth = float(model.eft_coefficients().get(op, 0.0))
    latex   = LATEX.get(op, op)

    result_ids, fit_labels = [], []
    cl_truth, cl_colors, cl_labels, cl_styles = [], [], [], []

    for fid, label in [(sm_id,  f"SM-only ({op} free)"),
                       (bsm_id, f"BSM-closure (sep. {op})")]:
        if not os.path.exists(f"{fit_dir}/{fid}/fit_results.json"):
            continue
        result_ids.append(fid)
        fit_labels.append(label)
        cl_styles.append("--")
        if fid == sm_id:
            cl_truth.append(None)
            cl_colors.append("tab:gray")
            cl_labels.append("")
        else:
            cl_truth.append({op: c_truth})
            cl_colors.append("red")
            cl_labels.append(f"UV truth ({op}={c_truth:.2e})")

    if not result_ids:
        return None

    return {
        "name":        f"Report_{tag}_sepproj_{op}",
        "title":       f"{op} sep. projection — {tag}",
        "result_IDs":  result_ids,
        "fit_labels":  fit_labels,
        "report_path": report_dir,
        "result_path": fit_dir,
        "summary":     False,
        "summary_only": False,
        "coefficients_plots": {
            "posterior_histograms": {
                "nrows":                1,
                "disjointed_lists":     False,
                "bins":                 60,
                "show_closure_truth":   True,
                "closure_truth_points": cl_truth,
                "closure_line_color":   cl_colors,
                "closure_line_style":   cl_styles,
                "show_closure_legend":  True,
                "closure_line_labels":  cl_labels,
            },
            "logo": False,
        },
        "chi2_plots": {
            "table":             True,
            "plot_experiment":   {"figsize": [12, 6]},
            "plot_distribution": {"figsize": [8, 5]},
        },
        "data_vs_theory": {
            "panel":            "pull",
            "include_sm":       True,
            "include_best_fit": True,
            "per_dataset":      False,
        },
        "plot":       False,
        "coeff_info": {"default": [[op, latex]]},
        "data_info":  DATA_INFO,
    }


# ── Sepproj pipeline ───────────────────────────────────────────────────────────

def run_sepproj(model, tag, sepproj_ops, skip_existing=False):
    """
    Separated-projection pipeline for selected operators.

    For each operator in sepproj_ops:
        0. Generate single-op BSM projection (only this op at UV truth)
        1. SM-only analytic fit  (1 op free, SM data)
        2. BSM-closure analytic fit  (1 op free, single-op BSM data)
        3. smefit R report (posterior histogram with truth line)

    Outputs under: fullpipeline/results/<tag>/sepproj/
        projections/<op>/   — single-op BSM pseudo-data
        runcards/           — PROJ + A + R runcards
        fits/               — smefit A results
        reports/            — smefit R reports
    """
    out        = str(PIPELINE / "results" / tag)
    sep_base   = f"{out}/sepproj"
    fit_dir    = f"{sep_base}/fits"
    rc_dir     = f"{sep_base}/runcards"
    report_dir = f"{sep_base}/reports"

    for d in [fit_dir, rc_dir, report_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  SEPARATED-PROJECTION MODE  (--sepproj)")
    print(f"  Tag      : {tag}")
    print(f"  Operators: {sepproj_ops}")
    print(f"  Out      : {sep_base}")
    print(f"{'='*64}")

    # Warn upfront about operators not in the model
    valid_ops   = set(model.OPERATORS)
    invalid_ops = [op for op in sepproj_ops if op not in valid_ops]
    if invalid_ops:
        print(f"\n  WARNING: the following operators are not part of {model} and will be skipped:")
        for op in invalid_ops:
            print(f"    {op}  (not in model.OPERATORS)")
        print(f"  Valid operators: {sorted(valid_ops)}\n")
    sepproj_ops = [op for op in sepproj_ops if op in valid_ops]

    for op in sepproj_ops:
        print(f"\n  ── {op} {'─'*(50 - len(op))}")
        proj_dir = f"{sep_base}/projections/{op}"

        # ── Step 0: single-op projection ─────────────────────────────────────
        if os.path.exists(proj_dir) and os.listdir(proj_dir):
            print(f"  [Step 0] Projection for {op} exists — skipping")
        else:
            ok = generate_sepproj_projection(model, op, proj_dir, rc_dir)
            if not ok:
                print(f"  [Step 0] FAILED for {op} — skipping this operator")
                continue

        # ── Step 1: SM-only fit ───────────────────────────────────────────────
        sm_id = f"{tag}_sepproj_{op}_SMonly"
        sm_rc = f"{rc_dir}/{sm_id}.yaml"
        rc    = make_sepproj_fit_runcard(sm_id, SM_DATA, op)
        rc["result_path"] = fit_dir
        with open(sm_rc, "w") as f:
            yaml.dump(rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        if not os.path.exists(f"{fit_dir}/{sm_id}/fit_results.json") or not skip_existing:
            _run_smefit("A", sm_rc, f"SM-only ({op} free, SM data)", fit_dir)
        else:
            print(f"  [A] SM-only for {op} exists — skipping")

        # ── Step 2: BSM-closure fit ───────────────────────────────────────────
        bsm_id = f"{tag}_sepproj_{op}_BSMclosure"
        bsm_rc = f"{rc_dir}/{bsm_id}.yaml"
        rc     = make_sepproj_fit_runcard(bsm_id, proj_dir, op)
        rc["result_path"] = fit_dir
        with open(bsm_rc, "w") as f:
            yaml.dump(rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        if not os.path.exists(f"{fit_dir}/{bsm_id}/fit_results.json") or not skip_existing:
            _run_smefit("A", bsm_rc, f"BSM-closure ({op} free, sep. proj.)", fit_dir)
        else:
            print(f"  [A] BSM-closure for {op} exists — skipping")

        # ── Step 3: smefit report ─────────────────────────────────────────────
        rep_rc = make_sepproj_report_runcard(tag, op, sm_id, bsm_id,
                                             fit_dir, report_dir, model)
        if rep_rc is not None:
            rep_rc_path = f"{rc_dir}/Report_{tag}_sepproj_{op}.yaml"
            with open(rep_rc_path, "w") as f:
                yaml.dump(rep_rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            _run_smefit("R", rep_rc_path, f"Report for {op} sepproj")
        else:
            print(f"  [R] No completed fits for {op} — report skipped")

    print(f"\n{'='*64}")
    print(f"  SEPPROJ COMPLETE")
    print(f"  Reports : {report_dir}/")
    print(f"{'='*64}\n")


# ── smefit runner ──────────────────────────────────────────────────────────────

def _run_smefit(mode, rc_path, label, fit_dir=None):
    from datetime import datetime
    rc = yaml.safe_load(open(rc_path))
    if fit_dir is not None:
        rc["result_path"] = fit_dir
    with open(rc_path, "w") as f:
        yaml.dump(rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    t0 = datetime.now()
    print(f"\n  [{mode}] {label}  [{t0.strftime('%H:%M:%S')}]", flush=True)
    r = subprocess.run([SMEFIT, mode, rc_path], env=ENV)
    t1 = datetime.now()
    elapsed = (t1 - t0).seconds
    status = "OK" if r.returncode == 0 else f"FAILED ({r.returncode})"
    print(f"  [{mode}] {status}  (elapsed: {elapsed//60}m {elapsed%60}s)  [{t1.strftime('%H:%M:%S')}]", flush=True)
    return r.returncode


# ── K matrix and covariance (model-agnostic) ───────────────────────────────────

_K_CI_CACHE: dict = {}

def _build_K_Ci(ops):
    """
    K (n_obs × n_ops): linearized theory matrix from smefit theory DB.
    Ci (n_obs × n_obs): block-diagonal inverse covariance (stat + sys + theory_cov).
    Results are cached in-process keyed on (ops, use_theory_covmat).
    """
    cache_key = (tuple(ops), CFG.use_theory_covmat)
    if cache_key in _K_CI_CACHE:
        return _K_CI_CACHE[cache_key]

    from scipy.linalg import block_diag

    K_blocks, Ci_blocks = [], []

    for ds in DS_NAMES:
        sm_path = f"{SM_DATA}/{ds}.yaml"
        th_path = f"{THEORY}/{ds}.json"
        if not os.path.exists(sm_path) or not os.path.exists(th_path):
            continue

        sm  = yaml.safe_load(open(sm_path))
        th  = json.load(open(th_path))
        order = DS_ORDER.get(ds, "LO")
        lo  = th.get(order, th.get("LO", {}))
        dc  = sm["data_central"]
        dc  = [dc] if not isinstance(dc, list) else list(dc)
        n   = len(dc)

        # Statistical covariance — use 0 for zero stat errors (no artificial fallback)
        stat = sm.get("statistical_error", [0.0] * n)
        stat = list(stat) if isinstance(stat, list) else [stat] * n
        C_ds = np.diag([float(e)**2 for e in stat])

        # Systematic covariance: YAML stores shape (n_sys, n_data), so C += S.T @ S
        sys_mat = sm.get("systematics", None)
        if sys_mat is not None:
            S = np.array(sys_mat, dtype=float)
            if S.ndim == 1:
                S = S.reshape(1, n)   # single systematic: (1, n_data)
            if S.ndim == 2:
                C_ds += S.T @ S       # (n_data, n_sys) @ (n_sys, n_data) → (n_data, n_data)

        # Theory covariance (skipped when --no-theory-covmat)
        if CFG.use_theory_covmat:
            th_cov = th.get(CFG.theory_cov_variant, None)
            if th_cov is not None:
                T = np.array(th_cov, dtype=float)
                if T.shape == (n, n):
                    C_ds += T

        # K rows
        K_ds = np.zeros((n, len(ops)))
        for j, op in enumerate(ops):
            k = lo.get(op, 0.0)
            if isinstance(k, list):
                K_ds[:, j] = [float(k[i]) if i < len(k) else 0.0 for i in range(n)]
            else:
                K_ds[:, j] = float(k)
        K_blocks.append(K_ds)

        try:
            Ci_ds = np.linalg.inv(C_ds)
        except np.linalg.LinAlgError:
            Ci_ds = np.diag(1.0 / np.maximum(np.diag(C_ds), 1e-30))
        Ci_blocks.append(Ci_ds)

    result = np.vstack(K_blocks), block_diag(*Ci_blocks)
    _K_CI_CACHE[cache_key] = result
    return result


def _build_cov_exp_chol():
    """
    Cholesky factor L of the block-diagonal experimental covariance (stat+sys only,
    no theory covmat). Used to draw L1 noise vectors: ε ~ N(0, C_exp) → ε = L @ z.
    Cached after first call.
    """
    if hasattr(_build_cov_exp_chol, "_cache"):
        return _build_cov_exp_chol._cache
    C_blocks = []
    for ds in DATASETS:
        sm_path = f"{SM_DATA}/{ds['name']}.yaml"
        if not os.path.exists(sm_path):
            continue
        sm   = yaml.safe_load(open(sm_path))
        dc   = sm["data_central"]
        dc   = [dc] if not isinstance(dc, list) else list(dc)
        n    = len(dc)
        stat = sm.get("statistical_error", [0.0] * n)
        stat = list(stat) if isinstance(stat, list) else [stat] * n
        C    = np.diag([max(float(e) ** 2, 1e-30) for e in stat])
        sys_mat = sm.get("systematics", None)
        if sys_mat is not None:
            S = np.array(sys_mat, dtype=float)
            if S.ndim == 1:
                S = S.reshape(1, n)
            C += S.T @ S
        C_blocks.append(C)
    from scipy.linalg import block_diag as _blkd
    L = np.linalg.cholesky(_blkd(*C_blocks))
    _build_cov_exp_chol._cache = L
    return L


def _build_delta(proj_dir):
    """BSM - SM signal vector from projection directory."""
    delta = []
    for ds in DS_NAMES:
        sm_path  = f"{SM_DATA}/{ds}.yaml"
        bsm_path = f"{proj_dir}/{ds}.yaml"
        if not os.path.exists(sm_path) or not os.path.exists(bsm_path):
            continue
        sm_dc  = yaml.safe_load(open(sm_path)) ["data_central"]
        bsm_dc = yaml.safe_load(open(bsm_path))["data_central"]
        scalar = not isinstance(sm_dc, list)
        sm_dc  = [sm_dc]  if scalar else list(sm_dc)
        bsm_dc = [bsm_dc] if scalar else list(bsm_dc)
        delta.extend(float(b) - float(s) for s, b in zip(sm_dc, bsm_dc))
    return np.array(delta)


# ── Step 4: PCA ────────────────────────────────────────────────────────────────

def _proj_hash(proj_dir: str) -> str:
    """MD5 of all projection yaml files — used to detect stale PCA cache."""
    h = hashlib.md5()
    for name in sorted(os.listdir(proj_dir)):
        if name.endswith(".yaml"):
            h.update(open(f"{proj_dir}/{name}", "rb").read())
    return h.hexdigest()

def _run_pca(model, proj_dir, pca_dir, free_ops=None):
    ops = free_ops if free_ops is not None else model.OPERATORS

    # Check if cached PCA is still valid (same projections + same covmat setting)
    hash_file = f"{pca_dir}/proj_hash.txt"
    current_hash = _proj_hash(proj_dir) + f"_covmat={CFG.use_theory_covmat}"
    cached_files = ["K_fit.npy", "C_inv.npy", "data_delta.npy",
                    "eigenvalues.npy", "eigenvectors.npy"]
    if (os.path.exists(hash_file)
            and open(hash_file).read().strip() == current_hash
            and all(os.path.exists(f"{pca_dir}/{f}") for f in cached_files)):
        print(f"\n  [PCA] Cache valid — skipping recomputation")
        return

    print(f"\n  [PCA] Building K matrix  (ops: {ops})")
    K, Ci = _build_K_Ci(ops)
    delta = _build_delta(proj_dir)
    print(f"    K: {K.shape}   chi2_SM: {float(delta @ Ci @ delta):.2f}")

    F = K.T @ Ci @ K
    evals, evecs = np.linalg.eigh(F)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]

    chi2_sm = float(delta @ Ci @ delta)
    np.save(f"{pca_dir}/K_fit.npy",        K)
    np.save(f"{pca_dir}/C_inv.npy",        Ci)
    np.save(f"{pca_dir}/data_delta.npy",   delta)
    np.save(f"{pca_dir}/eigenvalues.npy",  evals)
    np.save(f"{pca_dir}/eigenvectors.npy", evecs)
    with open(f"{pca_dir}/chi2_SM.txt", "w") as f:
        f.write(f"chi2_SM = {chi2_sm:.6f}\n")
    with open(hash_file, "w") as f:
        f.write(current_hash)

    print(f"    Top-5 eigenvalues: {evals[:5]}")


# ── Significance helpers ───────────────────────────────────────────────────────

def _q_to_sigma(q, k):
    if q <= 0:
        return 0.0
    p = chi2_dist.sf(q, df=k)
    return float(norm_dist.isf(p / 2)) if p > 0 else 99.0


def _point_significance(model, delta, K, Ci, ops):
    """
    Core significance computation given prebuilt K, Ci, delta.
    Returns a flat dict with q, ndof, sigma for each method.
    Shared by _analytic_significance and run_scan to avoid duplication.
    """
    chi2_sm = float(delta @ Ci @ delta)
    F       = K.T @ Ci @ K
    g       = K.T @ Ci @ delta

    evals, evecs = np.linalg.eigh(F)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]

    rank   = np.linalg.matrix_rank(F)
    q_full = float(g @ np.linalg.pinv(F) @ g)

    best_q, best_op = max((g[i]**2 / max(F[i,i], 1e-30), ops[i]) for i in range(len(ops)))

    k2 = min(2, len(evals))
    k5 = min(5, len(evals))
    q2 = sum((evecs[:,i]@g)**2 / max(evals[i],1e-30) for i in range(k2))
    q4 = sum((evecs[:,i]@g)**2 / max(evals[i],1e-30) for i in range(k5))

    c_truth = np.array([model.eft_coefficients().get(op, 0.0) for op in ops])
    res_uv  = delta - K @ c_truth
    q_uv    = chi2_sm - float(res_uv @ Ci @ res_uv)
    ndof_uv = len(model.uv_param_names()) if hasattr(model, "uv_param_names") else 2

    return {
        "chi2_SM":      chi2_sm,
        "best_q":       best_q,   "best_op":    best_op,
        "sigma_best1":  _q_to_sigma(best_q, 1),
        "q_pca2":       float(q2), "sigma_pca2": _q_to_sigma(q2, k2),
        "q_pca5":       float(q4), "sigma_pca5": _q_to_sigma(q4, k5),
        "q_full":       q_full,    "sigma_full": _q_to_sigma(q_full, rank),
        "rank":         rank,
        "q_uv":         q_uv,      "sigma_uv":   _q_to_sigma(q_uv, ndof_uv),
        "ndof_uv":      ndof_uv,
        "evals":        evals,     "evecs":      evecs,
        "F":            F,         "g":          g,
    }


def _analytic_significance(model, proj_dir):
    """
    Fast analytic profile likelihood from theory DB.
    Returns dict: method -> (sigma, note).  No smefit call needed.
    """
    ops   = model.OPERATORS
    K, Ci = _build_K_Ci(ops)
    delta = _build_delta(proj_dir)
    s     = _point_significance(model, delta, K, Ci, ops)

    return {
        "chi2_SM":   s["chi2_SM"],
        "Best 1-op": (_q_to_sigma(s["best_q"], 1), s["best_op"]),
        "PCA k=2":   (s["sigma_pca2"], ""),
        "PCA k=5":   (s["sigma_pca5"], ""),
        "Full SMEFT":(_q_to_sigma(s["q_full"], s["rank"]), f"rank={s['rank']}"),
        "UV (truth)":(s["sigma_uv"], ""),
    }


# ── Step 5: Profile likelihood summary ────────────────────────────────────────

def _compute_summary(model, tag, fit_dir, pca_dir, plt_dir, out, free_ops=None):
    ops     = free_ops if free_ops is not None else model.OPERATORS
    all_ops = model.OPERATORS  # always the full 27-op list (K matrix columns)

    if not os.path.exists(f"{pca_dir}/K_fit.npy"):
        print("  [Summary] PCA data not found, skipping")
        return

    K     = np.load(f"{pca_dir}/K_fit.npy")
    Ci    = np.load(f"{pca_dir}/C_inv.npy")
    delta = np.load(f"{pca_dir}/data_delta.npy")
    evals = np.load(f"{pca_dir}/eigenvalues.npy")
    evecs = np.load(f"{pca_dir}/eigenvectors.npy")

    chi2_sm = float(delta @ Ci @ delta)
    F = K.T @ Ci @ K
    g = K.T @ Ci @ delta

    # K may be restricted to free_ops (when a new freeops tag creates its own pca dir)
    # or full (all_ops) when loaded from an existing tag's pca dir.
    if K.shape[1] == len(all_ops):
        # Full K: subset to active ops via index
        op_idx = [all_ops.index(op) for op in ops]
        F_sub  = F[np.ix_(op_idx, op_idx)]
        g_sub  = g[op_idx]
        K_full = K
    else:
        # Restricted K (free_ops only): F and g are already in ops space
        F_sub  = F
        g_sub  = g
        K_full, _ = _build_K_Ci(all_ops)

    results = {}

    # Best single operator (restricted to active ops)
    best_q, best_op = max(
        (g_sub[j]**2 / max(F_sub[j, j], 1e-30), ops[j]) for j in range(len(ops))
    )
    results["Best 1-op"] = (best_q, 1, _q_to_sigma(best_q, 1), best_op)

    # PCA k=2 and k=5 (capped at number of available eigenvectors)
    # PCA uses the full g (all operators contribute to the eigenvectors)
    n_evecs = evecs.shape[1]
    for k in [2, 5]:
        if k > n_evecs:
            continue
        q_k = sum((evecs[:,i]@g)**2 / max(evals[i], 1e-30) for i in range(k))
        results[f"PCA k={k}"] = (float(q_k), k, _q_to_sigma(q_k, k), "")

    # Full SMEFT (restricted to active ops)
    rank   = np.linalg.matrix_rank(F_sub)
    q_full = float(g_sub @ np.linalg.pinv(F_sub) @ g_sub)
    results["Full SMEFT"] = (q_full, rank, _q_to_sigma(q_full, rank), f"rank={rank}/{len(ops)}")

    # UV NS fit (if available)
    uv_path = f"{fit_dir}/{tag}_UVcoup/fit_results.json"
    if os.path.exists(uv_path):
        r_uv    = json.load(open(uv_path))
        lnBF_uv = r_uv["logz"] - (-chi2_sm / 2)
        # c_truth must span all_ops to match K's column layout
        c_truth = np.array([model.eft_coefficients().get(op, 0.0) for op in all_ops])
        res_uv  = delta - K_full @ c_truth
        q_uv    = chi2_sm - float(res_uv @ Ci @ res_uv)
        ndof    = len(model.uv_param_names()) if hasattr(model, "uv_param_names") else 2
        results["UV coupling"] = (q_uv, ndof, _q_to_sigma(q_uv, ndof), f"lnBF={lnBF_uv:.2f}")

    # Bayes evidence from analytic fits
    for fit_id, label in [(f"{tag}_SMonly", "SM-only"),
                          (f"{tag}_BSMclosure_SMEFT", "SMEFT BSM")]:
        path = f"{fit_dir}/{fit_id}/fit_results.json"
        if os.path.exists(path):
            lnBF = json.load(open(path))["logz"] - (-chi2_sm / 2)
            results[label + " (Bayes)"] = (None, None, None, f"lnBF={lnBF:.2f}")

    # Print + save table
    print(f"\n  {'Method':<22} {'q':>8}  {'ndof':>5}  {'sigma':>7}  note")
    print(f"  {'-'*62}")
    lines = [f"# Profile likelihood summary: {tag}",
             f"# model: {model}",
             f"# chi2_SM = {chi2_sm:.4f}",
             f"# {'Method':<22} {'q':>8}  {'ndof':>5}  {'sigma':>7}  note"]
    for method, (q, k, sig, note) in results.items():
        if q is not None:
            print(f"  {method:<22} {q:>8.2f}  {k:>5d}  {sig:>6.2f}σ  {note}")
            lines.append(f"  {method:<22} {q:>8.2f}  {k:>5}  {sig:>6.2f}  {note}")
        else:
            print(f"  {method:<22} {'—':>8}  {'—':>5}  {'—':>7}  {note}")
            lines.append(f"  {method:<22} — — — {note}")

    with open(f"{out}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Saved: {out}/summary.txt")

    # Significance bar chart
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods_pl = [m for m, (q, k, s, _) in results.items() if s is not None]
    sigs_pl    = [results[m][2] for m in methods_pl]
    colors     = ["steelblue" if s >= 5 else "tomato" for s in sigs_pl]

    fig, ax = plt.subplots(figsize=(8, max(4, len(methods_pl) * 0.55)))
    ax.barh(methods_pl, sigs_pl, color=colors)
    ax.axvline(5.0, color="gold",   lw=2,   ls="--", label=r"5$\sigma$ discovery")
    ax.axvline(3.0, color="orange", lw=1.5, ls=":",  label=r"3$\sigma$ evidence")
    ax.set_xlabel(r"Significance $\sigma$ (profile likelihood)", fontsize=11)
    ax.set_title(f"Discovery significance\n{model}  |  {tag}", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plt_dir}/significance_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plt_dir}/significance_summary.png")

    # Analytic UV Fisher heatmap (no NS required) — always uses full K and all_ops
    _plot_uv_fisher_analytic(K_full, Ci, model, all_ops, plt_dir, tag)

    # PCA dimensionality: must use the same K that produced evals/evecs (may be restricted)
    _plot_pca_dimensionality(K, Ci, delta, evals, evecs, ops, plt_dir, tag)

    return results


# ── Step 5b: UV coupling posterior plots ──────────────────────────────────────

def _plot_uv_posteriors(model, tag, fit_dir, plt_dir):
    """
    Plot posterior distributions of UV coupling parameters from the NS fit.
    Produces:
      - uv_posteriors_1d.png  : 1D marginal histograms per UV parameter
      - uv_posteriors_2d.png  : 2D scatter / contour for each pair of UV params
    """
    uv_path = f"{fit_dir}/{tag}_UVcoup/fit_results.json"
    if not os.path.exists(uv_path):
        print(f"\n  [UV plots] Skipped — UV fit not found at {uv_path}")
        return

    import json, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    r         = json.load(open(uv_path))
    uv_params = model.uv_param_names() if hasattr(model, "uv_param_names") else list(r["free_parameters"])
    truth     = model.uv_truth() if hasattr(model, "uv_truth") else {}
    samples   = {k: np.array(r["samples"][k]) for k in uv_params if k in r["samples"]}

    if not samples:
        print(f"\n  [UV plots] No UV parameter samples found in fit_results.json")
        return

    n = len(samples)
    param_list = list(samples.keys())

    # ── 1D marginal histograms ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    for i, param in enumerate(param_list):
        ax  = axes[0, i]
        arr = samples[param]
        ax.hist(arr, bins=60, color="#4C72B0", alpha=0.75, density=True, edgecolor="none")

        # Truth line
        if param in truth:
            ax.axvline(truth[param], color="crimson", lw=2.0, ls="--",
                       label=f"truth = {truth[param]:.4f}")
            ax.legend(fontsize=9)

        # Mean line
        ax.axvline(arr.mean(), color="#444", lw=1.2, ls=":",
                   label=f"mean = {arr.mean():.4f}")

        ax.set_xlabel(f"${param}$", fontsize=13)
        ax.set_ylabel("posterior density", fontsize=11)
        ax.set_title(f"UV posterior: {param}", fontsize=11)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.grid(True, alpha=0.2, lw=0.5)

    fig.suptitle(f"UV coupling posteriors — {tag}", fontsize=12, y=1.01)
    plt.tight_layout()
    out1 = f"{plt_dir}/uv_posteriors_1d.png"
    plt.savefig(out1, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out1}")

    # ── 2D scatter plots for each pair ───────────────────────────────────────
    if n < 2:
        return

    from itertools import combinations
    pairs = list(combinations(param_list, 2))
    ncols = min(3, len(pairs))
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4.0 * nrows), squeeze=False)

    for idx, (px, py) in enumerate(pairs):
        ax  = axes[idx // ncols][idx % ncols]
        ax.scatter(samples[px], samples[py], s=1.5, alpha=0.25, color="#4C72B0", rasterized=True)

        # Truth point
        if px in truth and py in truth:
            ax.scatter([truth[px]], [truth[py]], color="crimson", s=60, zorder=5,
                       marker="*", label="truth")
            ax.legend(fontsize=9, markerscale=1.5)

        # 68% and 95% contours via KDE
        try:
            from scipy.stats import gaussian_kde
            xarr, yarr = samples[px], samples[py]
            kde  = gaussian_kde(np.vstack([xarr, yarr]))
            xg   = np.linspace(xarr.min(), xarr.max(), 80)
            yg   = np.linspace(yarr.min(), yarr.max(), 80)
            XX, YY = np.meshgrid(xg, yg)
            ZZ   = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            # find contour levels for 68% and 95%
            zflat   = np.sort(ZZ.ravel())[::-1]
            cumsum  = np.cumsum(zflat) / zflat.sum()
            lev68   = zflat[np.searchsorted(cumsum, 0.68)]
            lev95   = zflat[np.searchsorted(cumsum, 0.95)]
            ax.contour(XX, YY, ZZ, levels=[lev95, lev68],
                       colors=["#4C72B0", "#1a3a6b"], linewidths=[1.0, 1.5],
                       linestyles=["--", "-"])
        except Exception:
            pass

        ax.set_xlabel(f"${px}$", fontsize=12)
        ax.set_ylabel(f"${py}$", fontsize=12)
        ax.set_title(f"{px} vs {py}", fontsize=10)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.grid(True, alpha=0.2, lw=0.5)

    # Hide unused axes
    for idx in range(len(pairs), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"UV coupling 2D posteriors — {tag}", fontsize=12)
    plt.tight_layout()
    out2 = f"{plt_dir}/uv_posteriors_2d.png"
    plt.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2}")

    # Print summary
    print(f"\n  UV posterior summary (truth | mean ± std):")
    for param in param_list:
        arr = samples[param]
        tv  = truth.get(param, float("nan"))
        print(f"    {param:12s}: truth={tv:.4f}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"median={np.median(arr):.4f}")


# ── PCA dimensionality: cumulative chi2 vs k ──────────────────────────────────

def _plot_pca_dimensionality(K, Ci, delta, evals, evecs, ops, plt_dir, tag):
    """
    Plot cumulative chi2 improvement q(k) = sum_{i=1}^{k} (e_i^T g)^2 / lambda_i
    as a function of PCA dimension k. The elbow identifies the effective
    dimensionality of the SMEFT operator basis needed to describe the signal.
    Also plots the individual per-component contribution.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g    = K.T @ Ci @ delta          # gradient vector in operator space

    # Only use eigenvalues significantly above zero (rank of Fisher matrix)
    threshold = 1e-6 * evals.max() if evals.max() > 0 else 1.0
    valid_idx = np.where(evals > threshold)[0]   # already sorted descending
    evals_v   = evals[valid_idx]
    evecs_v   = evecs[:, valid_idx]
    n         = len(valid_idx)

    projections  = np.array([(evecs_v[:, i] @ g)**2 / evals_v[i]
                              for i in range(n)])
    total        = float(delta @ Ci @ delta)      # q_SMEFT — ground truth total
    cumulative   = np.cumsum(projections)
    cumulative_pct = 100.0 * cumulative / max(total, 1e-30)
    individual_pct = 100.0 * projections / max(total, 1e-30)

    # Convert cumulative to sigma for reference
    from scipy.stats import norm as norm_dist, chi2 as chi2_dist
    def q_to_sigma(q, k):
        if q <= 0 or k <= 0: return 0.0
        p = chi2_dist.sf(q, df=k)
        return float(norm_dist.isf(p / 2)) if p > 0 else 99.0

    sigma_k = np.array([q_to_sigma(float(cumulative[i]), i + 1) for i in range(n)])
    sigma_k = np.clip(sigma_k, 0, 50)

    ks = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: cumulative chi2 %
    axes[0].plot(ks, cumulative_pct, "o-", color="#1f77b4", ms=4, lw=1.5)
    axes[0].axhline(95, color="gray", lw=1, ls="--", label="95%")
    axes[0].axhline(99, color="gray", lw=1, ls=":",  label="99%")
    axes[0].set_xlabel("PCA components k", fontsize=12)
    axes[0].set_ylabel("Cumulative $q(k)$ [% of total]", fontsize=12)
    axes[0].set_title("Cumulative signal fraction", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(1, n)
    axes[0].set_ylim(0, 101)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: per-component contribution (bar)
    axes[1].bar(ks, individual_pct, color="#1f77b4", alpha=0.7, width=0.8)
    axes[1].set_xlabel("PCA component k", fontsize=12)
    axes[1].set_ylabel("Individual contribution [%]", fontsize=12)
    axes[1].set_title("Per-component signal fraction", fontsize=11)
    axes[1].set_xlim(0.5, min(n + 0.5, 20.5))
    axes[1].grid(True, axis="y", alpha=0.3)

    # Panel 3: per-component chi2 on log scale — shows where signal truly stops
    axes[2].bar(ks, projections, color="#ff7f0e", alpha=0.8, width=0.8)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("PCA component k", fontsize=12)
    axes[2].set_ylabel(r"$\Delta\chi^2$ per component", fontsize=12)
    axes[2].set_title("Per-component $\\chi^2$ (log scale)", fontsize=11)
    axes[2].set_xlim(0.5, n + 0.5)
    axes[2].grid(True, axis="y", alpha=0.3, which="both")

    fig.suptitle(f"PCA effective dimensionality — {tag}", fontsize=12, y=1.01)
    plt.tight_layout()
    path = f"{plt_dir}/pca_dimensionality.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Analytic UV Fisher heatmap ────────────────────────────────────────────────

def _plot_uv_fisher_analytic(K, Ci, model, ops, plt_dir, tag):
    """
    Compute and plot the UV Fisher information matrix analytically.

    F_UV^(dataset)[i,j] = (K_d J)^T Ci_d (K_d J)  where J = dc/dtheta (Jacobian)

    Rows = UV parameters, columns = energy groups (91/161/240/365 GeV).
    Saved as uv_fisher_analytic.png. No NS fit required.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not hasattr(model, "uv_param_names"):
        return
    uv_params = model.uv_param_names()
    if not uv_params:
        return

    # Numerical Jacobian: dc_i / d(theta_j)  shape (n_ops, n_uv)
    eps = 1e-6
    truth = model.uv_truth()
    n_ops = len(ops)
    n_uv  = len(uv_params)
    J = np.zeros((n_ops, n_uv))
    for j, par in enumerate(uv_params):
        kwargs_p = {**{k: float(v) for k, v in truth.items()}}
        kwargs_m = {**kwargs_p}
        kwargs_p[par] = truth[par] + eps
        kwargs_m[par] = truth[par] - eps
        from models.wprime            import WPrimeModel
        from models.wprime_constrained import WPrimeConstrainedModel
        from models.zprime            import ZPrimeModel
        from models.zprime_constrained import ZPrimeConstrainedModel
        cls = type(model)
        c_p = np.array([cls(**kwargs_p).eft_coefficients().get(op, 0.0) for op in ops])
        c_m = np.array([cls(**kwargs_m).eft_coefficients().get(op, 0.0) for op in ops])
        J[:, j] = (c_p - c_m) / (2 * eps)

    # KJ: shape (n_obs, n_uv)
    KJ = K @ J

    # Per-dataset UV Fisher diagonal (variance of each UV param constrained by dataset)
    # Group by energy following DATASETS order
    energy_groups = {
        "91 GeV\n(Z-pole)":   ["FCCee_Zdata", "FCCee_Wwidth"],
        "161 GeV\n(WW thr.)": ["FCCee_ww_161GeV"],
        "240 GeV":            ["FCCee_ww_240GeV","FCCee_Rb_240GeV","FCCee_Rc_240GeV",
                               "FCCee_Rmu_240GeV","FCCee_Rtau_240GeV",
                               "FCCee_bb_Afb_240GeV","FCCee_cc_Afb_240GeV",
                               "FCCee_mumu_Afb_240GeV","FCCee_sigmaHad_240GeV",
                               "FCCee_tautau_Afb_240GeV","FCCee_zh_240GeV",
                               "FCCee_zh_WW_240GeV","FCCee_zh_ZZ_240GeV",
                               "FCCee_zh_aZ_240GeV","FCCee_zh_aa_240GeV",
                               "FCCee_zh_tautau_240GeV"],
        "365 GeV":            ["FCCee_ww_365GeV","FCCee_Rb_365GeV","FCCee_Rc_365GeV",
                               "FCCee_Rmu_365GeV","FCCee_Rtau_365GeV",
                               "FCCee_bb_Afb_365GeV","FCCee_cc_Afb_365GeV",
                               "FCCee_mumu_Afb_365GeV","FCCee_sigmaHad_365GeV",
                               "FCCee_tautau_Afb_365GeV","FCCee_zh_365GeV",
                               "FCCee_zh_WW_365GeV","FCCee_zh_ZZ_365GeV",
                               "FCCee_zh_aa_365GeV","FCCee_zh_tautau_365GeV"],
    }

    # Build dataset row index from DATASETS order
    import yaml as _yaml
    proj_dir = str(PIPELINE / "projections" / tag)
    row = 0
    ds_row = {}
    for d in DATASETS:
        name = d["name"]
        yp = f"{proj_dir}/{name}.yaml"
        if os.path.exists(yp):
            with open(yp) as f:
                info = _yaml.safe_load(f)
            n = info.get("num_data", 1)
        else:
            n = 1
        ds_row[name] = (row, row + n)
        row += n

    group_names = list(energy_groups.keys())
    n_groups = len(group_names)

    # F_UV[param, group] = sum of diagonal Fisher entries for that group
    F_uv = np.zeros((n_uv, n_groups))
    for ig, (grp, ds_list) in enumerate(energy_groups.items()):
        for ds in ds_list:
            if ds not in ds_row:
                continue
            r0, r1 = ds_row[ds]
            KJ_d  = KJ[r0:r1, :]
            Ci_d  = Ci[r0:r1, r0:r1]
            F_d   = KJ_d.T @ Ci_d @ KJ_d   # (n_uv, n_uv)
            F_uv[:, ig] += np.diag(F_d)

    # Normalise each UV param row to 100%
    row_sums = F_uv.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    F_norm = 100.0 * F_uv / row_sums

    uv_latex = {
        "gWH":    r"$g_{WH}$",
        "gWLf11": r"$g_{WLf,11}$",
        "gWLf22": r"$g_{WLf,22}$",
        "gWLf33": r"$g_{WLf,33}$",
        "gWqf33": r"$g_{Wqf,33}$",
        "gZH":    r"$g_{ZH}$",
        "gZl":    r"$g_{Zl}$",
    }
    ylabels = [uv_latex.get(p, p) for p in uv_params]

    fig, ax = plt.subplots(figsize=(7, max(3, n_uv * 0.9 + 1.5)))
    im = ax.imshow(F_norm, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(n_groups)); ax.set_xticklabels(group_names, fontsize=10)
    ax.set_yticks(range(n_uv));    ax.set_yticklabels(ylabels, fontsize=11)
    for i in range(n_uv):
        for j in range(n_groups):
            ax.text(j, i, f"{F_norm[i,j]:.1f}", ha="center", va="center",
                    fontsize=9, color="white" if F_norm[i,j] > 60 else "black")
    plt.colorbar(im, ax=ax, label="Normalised Fisher [%]")
    ax.set_title(f"Analytic UV Fisher: {tag}", fontsize=11)
    plt.tight_layout()
    path = f"{plt_dir}/uv_fisher_analytic.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Step 6: smefit report ──────────────────────────────────────────────────────

def _run_report(tag, rc_dir, report_dir, proj_dir, fit_dir, model, free_ops=None):
    os.makedirs(report_dir, exist_ok=True)

    # ── EFT report (SM-only vs SMEFT BSM) ────────────────────────────────────
    report_id = f"Report_{tag}"
    rc_path   = f"{rc_dir}/{report_id}.yaml"
    rc        = make_report_runcard(tag, proj_dir, fit_dir, report_dir, model,
                                    free_ops=free_ops)
    with open(rc_path, "w") as f:
        yaml.dump(rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"\n  [R] Generating EFT smefit report...")
    r = subprocess.run([SMEFIT, "R", rc_path], env=ENV)
    status = "OK" if r.returncode == 0 else f"FAILED ({r.returncode})"
    print(f"  {status}  ->  {report_dir}/{report_id}/")

    # ── UV coupling report (UV NS fit only, UV parameter posteriors) ──────────
    uv_rc = make_uv_report_runcard(tag, fit_dir, report_dir, model)
    if uv_rc is not None:
        uv_report_id = f"Report_{tag}_UVcoup"
        uv_rc_path   = f"{rc_dir}/{uv_report_id}.yaml"
        with open(uv_rc_path, "w") as f:
            yaml.dump(uv_rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"\n  [R] Generating UV coupling smefit report...")
        r2 = subprocess.run([SMEFIT, "R", uv_rc_path], env=ENV)
        status2 = "OK" if r2.returncode == 0 else f"FAILED ({r2.returncode})"
        print(f"  {status2}  ->  {report_dir}/{uv_report_id}/")
    else:
        print(f"\n  [R] UV report skipped — UV fit not found")

    return r.returncode


# ── Step 7-9: Discovery scan ───────────────────────────────────────────────────

def run_scan(model_cls, base_params, tag_prefix, coupling_key, coupling_grid,
             mass_key, mass_grid, plt_dir, out_dir):
    """
    Scan over (coupling, mass) grid using analytic profile likelihood.
    No smefit calls — fast enough for a full 2D grid.

    Parameters
    ----------
    model_cls     : class (WPrimeModel or ZPrimeModel)
    base_params   : dict of default model parameters
    tag_prefix    : e.g. "zprime" or "wprime"
    coupling_key  : parameter name for the coupling axis, e.g. "gZH"
    coupling_grid : list of coupling values
    mass_key      : parameter name for the mass axis, e.g. "mZp"
    mass_grid     : list of mass values [TeV]
    """
    os.makedirs(plt_dir, exist_ok=True)
    os.makedirs(out_dir,  exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  DISCOVERY SCAN  [{tag_prefix}]")
    print(f"  {coupling_key}: {coupling_grid}")
    print(f"  {mass_key}:     {mass_grid}")
    print(f"{'='*64}\n")

    # K and Ci are the same for all scan points (fixed operator set)
    dummy = model_cls(**base_params)
    ops   = dummy.OPERATORS
    K, Ci = _build_K_Ci(ops)

    results = []   # list of (coupling, mass, sigma_dict)

    n_total = len(coupling_grid) * len(mass_grid)
    n_done  = 0

    for g_val in coupling_grid:
        for m_val in mass_grid:
            n_done += 1
            params = {**base_params, coupling_key: g_val, mass_key: m_val}
            # For W': scale all fermion couplings with gWH so the full
            # coupling dependence (gWH × gWLf and gWH × gWqf) is captured.
            # This makes the signal scale as g² / m² and gives a diagonal
            # contour in the (m, g) plane.
            if "gWLf11" in params:
                ratio = g_val / base_params.get(coupling_key, g_val)
                for fkey in ["gWLf11", "gWLf22", "gWLf33", "gWqf33"]:
                    if fkey in params:
                        params[fkey] = base_params[fkey] * ratio
            model  = model_cls(**params)

            # Generate projections
            _glf = base_params.get("gWLf11", base_params.get("gZl", base_params.get(coupling_key, g_val)))
            _gqf = base_params.get("gWqf33", _glf)
            _g0  = base_params.get(coupling_key, g_val)
            _lfr = int(round(_glf / _g0 * 100)) if _g0 else 100
            _qfr = int(round(_gqf / _g0 * 100)) if _g0 else 100
            scan_tag  = (f"{tag_prefix}_scan_{coupling_key}{int(g_val*100):03d}"
                         f"_{mass_key}{int(m_val*10):03d}_lfr{_lfr:03d}_qfr{_qfr:03d}")
            proj_dir  = str(PIPELINE / "projections" / scan_tag)
            if not os.path.isdir(proj_dir) or not os.listdir(proj_dir):
                generate_projections(model, proj_dir)

            # Analytic significance (shared with _analytic_significance)
            delta = _build_delta(proj_dir)
            s     = _point_significance(model, delta, K, Ci, ops)
            chi2_sm     = s["chi2_SM"]
            sigma_best1 = s["sigma_best1"]
            sigma_pca2  = s["sigma_pca2"]
            sigma_full  = s["sigma_full"]
            sigma_uv    = s["sigma_uv"]

            results.append({
                coupling_key: g_val, mass_key: m_val,
                "sigma_best1": sigma_best1,
                "sigma_pca2":  sigma_pca2,
                "sigma_full":  sigma_full,
                "sigma_uv":    sigma_uv,
                "chi2_SM":     chi2_sm,
            })

            disc = "DISCOVERY" if sigma_uv >= 5 else ("evidence" if sigma_uv >= 3 else "")
            print(f"  [{n_done:2d}/{n_total}] {coupling_key}={g_val:.3f}  {mass_key}={m_val:.1f} TeV  "
                  f"σ(UV)={sigma_uv:.2f}  σ(SMEFT)={sigma_full:.2f}  {disc}")

    # Save table
    table_path = f"{out_dir}/discovery_table.txt"
    header = (f"# Discovery scan: {tag_prefix}\n"
              f"# {coupling_key:>8}  {mass_key:>8}  "
              f"{'sigma_best1':>12}  {'sigma_pca2':>10}  {'sigma_full':>10}  {'sigma_uv':>10}  {'chi2_SM':>10}\n")
    with open(table_path, "w") as f:
        f.write(header)
        for r in results:
            f.write(f"  {r[coupling_key]:>8.4f}  {r[mass_key]:>8.2f}  "
                    f"{r['sigma_best1']:>12.3f}  {r['sigma_pca2']:>10.3f}  "
                    f"{r['sigma_full']:>10.3f}  {r['sigma_uv']:>10.3f}  "
                    f"{r['chi2_SM']:>10.2f}\n")
    print(f"\n  Saved: {table_path}")

    # Discovery region plot (from scan grid)
    _plot_discovery_region(results, coupling_key, mass_key,
                           coupling_grid, mass_grid, tag_prefix, plt_dir,
                           base_params=base_params)

    # Extended analytic discovery plot (corner-to-corner, auto-computed bounds)
    _plot_discovery_extended(K, Ci, model_cls, base_params, coupling_key,
                             mass_key, ops, tag_prefix, plt_dir, coupling_grid)

    return results


def _plot_discovery_extended(K, Ci, model_cls, base_params, coupling_key,
                             mass_key, ops, tag_prefix, plt_dir, coupling_grid):
    """
    Analytic extended discovery region with auto-computed bounds so the
    5sigma contour spans corner to corner. Saved as discovery_region_extended.png.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
    from scipy.stats import chi2 as chi2_dist, norm as norm_dist

    g0 = base_params[coupling_key]
    m0 = base_params[mass_key]
    model_ref = model_cls(**base_params)
    ndof_uv   = len(model_ref.uv_param_names()) if hasattr(model_ref, "uv_param_names") else 2

    # Reference q at injection point (Asimov: q = c^T K^T Ci K c)
    c_ref = np.array([model_ref.eft_coefficients().get(op, 0.0) for op in ops])
    q_ref = float((K @ c_ref) @ Ci @ (K @ c_ref))
    if q_ref <= 0:
        print("  [extended plot] q_ref <= 0, skipping.")
        return

    # SMEFT rank
    F = K.T @ Ci @ K
    ndof_full = int(np.sum(np.linalg.eigvalsh(F) > 1e-8 * np.linalg.eigvalsh(F).max()))

    # 5sigma q-threshold (two-sided, matching _q_to_sigma)
    p_5s      = 2.0 * norm_dist.sf(5.0)
    q_5s_uv   = chi2_dist.ppf(1.0 - p_5s, df=ndof_uv)

    # Auto grid: q scales as (g/g0)^4 * (m0/m)^4
    # At gWH_max, 5sigma boundary is at:  m = m0*(gWH_max/g0)*(q_ref/q_5s)^0.25
    gWH_max  = max(coupling_grid)
    mWp_max  = m0 * (gWH_max / g0) * (q_ref / q_5s_uv) ** 0.25 * 1.15
    mWp_min  = max(0.3, m0 * 0.03)

    ng, nm = 80, 100
    coupling_arr = np.linspace(0.001, gWH_max, ng)
    mass_arr     = np.linspace(mWp_min, mWp_max, nm)

    # Coupling ratios (preserved across scan)
    ratios = {}
    for fkey in ["gWLf11", "gWLf22", "gWLf33", "gWqf33", "gZl"]:
        if fkey in base_params and fkey != coupling_key:
            ratios[fkey] = base_params[fkey] / g0

    sigma_uv   = np.zeros((ng, nm))
    sigma_full = np.zeros((ng, nm))

    for ic, g in enumerate(coupling_arr):
        for im, m in enumerate(mass_arr):
            params = {**base_params, coupling_key: g, mass_key: m}
            for fkey, r in ratios.items():
                params[fkey] = g * r
            model = model_cls(**params)
            c = np.array([model.eft_coefficients().get(op, 0.0) for op in ops])
            q = float((K @ c) @ Ci @ (K @ c))
            sigma_uv[ic, im]   = _q_to_sigma(q, ndof_uv)
            sigma_full[ic, im] = _q_to_sigma(q, ndof_full)

    sigma_uv   = np.clip(sigma_uv,   0, 50)
    sigma_full = np.clip(sigma_full, 0, 50)

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

    cs5f = ax.contour(mass_arr, coupling_arr, sigma_full,
                      levels=[5.0], colors=["#ff7f0e"], linewidths=[1.5],
                      linestyles=[":"])
    ax.clabel(cs5f, fmt=r"$5\sigma$ (SMEFT)", fontsize=9, inline=True)

    model_label = (tag_prefix.replace("wprime_constrained", "W' (constrained)")
                             .replace("wprime", "W'")
                             .replace("zprime", "Z'"))
    mkey_tex = r"m_{W'}" if "Wp" in mass_key else r"m_{Z'}"
    ckey_tex = r"|g_{WH}|" if "gWH" in coupling_key else r"|g_{ZH}|"

    ax.set_xlabel(f"${mkey_tex}$ [TeV]", fontsize=14)
    ax.set_ylabel(f"${ckey_tex}$",       fontsize=14)
    ax.set_title(f"FCC-ee  discovery reach  ({model_label})", fontsize=13)
    ax.set_xlim(mWp_min, mWp_max)
    ax.set_ylim(0, gWH_max)
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
        Line2D([0],[0], color="#ff7f0e", lw=1.5, ls=":", label=r"Full SMEFT: $5\sigma$"),
    ]
    ax.legend(handles=handles, fontsize=10, loc="upper left",
              framealpha=0.9, edgecolor="gray")

    plt.tight_layout()
    path = f"{plt_dir}/discovery_region_extended.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_discovery_region(results, coupling_key, mass_key,
                           coupling_grid, mass_grid, tag_prefix, plt_dir,
                           base_params=None):
    """
    Clean single-panel discovery contour plot, styled like publication figures.
    x-axis = mass [TeV], y-axis = coupling |g|.
    Shaded region = discoverable (sigma >= 5).
    Smooth contour lines at 3sigma (dashed) and 5sigma (solid).
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    nc = len(coupling_grid)
    nm = len(mass_grid)

    # Build 2D grids (rows=coupling, cols=mass) — mass on x, coupling on y
    sigma_uv   = np.zeros((nc, nm))
    sigma_full = np.zeros((nc, nm))

    for r in results:
        ic = coupling_grid.index(r[coupling_key])
        im = mass_grid.index(r[mass_key])
        sigma_uv[ic, im]   = r["sigma_uv"]
        sigma_full[ic, im] = r["sigma_full"]

    mass_arr     = np.array(mass_grid)
    coupling_arr = np.array(coupling_grid)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Filled background: light blue = discoverable, light grey = not
    ax.contourf(mass_arr, coupling_arr, sigma_uv,
                levels=[5.0, 1e4],
                colors=["#ADD8E6"], alpha=0.5)
    ax.contourf(mass_arr, coupling_arr, sigma_uv,
                levels=[0, 5.0],
                colors=["#DCDCDC"], alpha=0.5)

    # 5sigma discovery contour (solid, bold)
    cs5 = ax.contour(mass_arr, coupling_arr, sigma_uv,
                     levels=[5.0], colors=["#1f77b4"], linewidths=[2.5])
    ax.clabel(cs5, fmt=r"$5\sigma$", fontsize=10, inline=True)

    # 3sigma evidence contour (dashed)
    cs3 = ax.contour(mass_arr, coupling_arr, sigma_uv,
                     levels=[3.0], colors=["#1f77b4"], linewidths=[1.5],
                     linestyles=["--"])
    ax.clabel(cs3, fmt=r"$3\sigma$", fontsize=10, inline=True)

    # Full SMEFT contour overlaid for comparison (thinner, different colour)
    cs5f = ax.contour(mass_arr, coupling_arr, sigma_full,
                      levels=[5.0], colors=["#ff7f0e"], linewidths=[1.5],
                      linestyles=[":"])
    ax.clabel(cs5f, fmt=r"$5\sigma$ (SMEFT)", fontsize=9, inline=True)

    model_label = tag_prefix.replace("wprime", "W'").replace("zprime", "Z'")
    ax.set_xlabel(f"$m_{{\\rm {model_label}}}$ [TeV]", fontsize=13)
    ax.set_ylabel(f"$|{coupling_key}|$", fontsize=13)
    ax.set_title(f"FCC-ee discovery reach: {model_label}", fontsize=12)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(True, which="major", alpha=0.2, lw=0.5)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#1f77b4", lw=2.5, label=r"UV: $5\sigma$ discovery"),
        Line2D([0], [0], color="#1f77b4", lw=1.5, ls="--", label=r"UV: $3\sigma$ evidence"),
        Line2D([0], [0], color="#ff7f0e", lw=1.5, ls=":", label=r"Full SMEFT: $5\sigma$"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left")

    # Annotate regions
    ax.text(0.97, 0.97, "Discoverable", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="#1f77b4",
            style="italic")
    ax.text(0.97, 0.03, "Not discoverable", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, color="gray",
            style="italic")

    # Coupling assumption annotation
    if base_params is not None:
        g0 = base_params.get(coupling_key, None)
        if g0 and g0 != 0:
            other_keys = [k for k in base_params
                          if k not in (coupling_key, "mWp", "mZp")]
            ratios = {k: base_params[k] / g0 for k in other_keys
                      if isinstance(base_params[k], float)}
            if ratios:
                ratio_str = ",  ".join(
                    f"$g_{{\\rm {k.replace('gW','').replace('gZ','')}}} = "
                    f"{v:.2g}\\,|g_{{\\rm {coupling_key[1:]}}}|$"
                    for k, v in ratios.items()
                )
                ax.text(0.03, 0.03,
                        f"Fixed ratios:  {ratio_str}",
                        transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=7.5, color="#444",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="gray", alpha=0.7))

    # Clip x-axis to where the 5σ contour exits the bottom of the plot
    # (first mass where the lowest coupling drops below 5σ) — removes vertical artifact
    x_min_idx = next((im for im in range(nm) if sigma_uv[0, im] < 5.0), nm - 1)
    ax.set_xlim(mass_arr[x_min_idx], mass_arr[-1])
    ax.set_ylim(coupling_arr[0], coupling_arr[-1])

    plt.tight_layout()
    path = f"{plt_dir}/discovery_region.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Also save a clean summary: which points are discoverable
    print(f"\n  5σ discovery region (UV method):")
    for r in sorted(results, key=lambda x: (x[mass_key], x[coupling_key])):
        disc = "✓ DISCOVERY" if r["sigma_uv"] >= 5 else ("  evidence" if r["sigma_uv"] >= 3 else "  below threshold")
        print(f"    {coupling_key}={r[coupling_key]:.3f}  {mass_key}={r[mass_key]:.1f} TeV  "
              f"σ={r['sigma_uv']:.2f}  {disc}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(model, tag: str, skip_existing: bool = False,
                 run_ns: bool = True, uv_analytic: bool = False,
                 run_report: bool = True, inject_tag: str = None,
                 free_ops: list = None):
    """
    Full end-to-end pipeline for a single model point.

    Steps:
        0. Generate projections (auto if missing)
        1. SM-only analytic fit
        2. Full SMEFT analytic fit
        3. UV NS fit  (skip with run_ns=False)
        4. PCA
        5. Profile likelihood table + bar chart
        6. smefit report  (skip with run_report=False)

    inject_tag: if set, use pseudo-data from this existing tag instead of
                generating new projections. Enables robustness tests, e.g.
                inject W' data but fit with Z' UV model.
    free_ops:   if set, only these operators are free in the SMEFT fit step.
                PCA and UV steps still use all model operators.
    """
    # Use injected pseudo-data if specified (robustness test)
    inj_tag    = inject_tag or tag
    proj_dir   = str(PIPELINE / "projections" / inj_tag)
    out        = str(PIPELINE / "results"     / tag)
    rc_dir     = f"{out}/runcards"
    fit_dir    = f"{out}/fits"
    pca_dir    = f"{out}/pca"
    plt_dir    = f"{out}/plots"
    report_dir = f"{out}/reports"

    for d in [proj_dir, rc_dir, fit_dir, pca_dir, plt_dir]:
        os.makedirs(d, exist_ok=True)

    ops      = model.OPERATORS
    fit_ops  = free_ops if free_ops is not None else ops

    print(f"\n{'='*64}")
    print(f"  BSM DISCOVERY PIPELINE")
    print(f"  Fit model : {model}")
    if inject_tag:
        print(f"  Inject tag: {inject_tag}  (ROBUSTNESS TEST — wrong model fit)")
    if free_ops is not None:
        print(f"  FREE OPS  : {free_ops}  (subset fit — {len(free_ops)}/{len(ops)} operators)")
    print(f"  Tag   : {tag}")
    print(f"  Ops   : {fit_ops}")
    print(f"  Out   : {out}")
    print(f"  NS fit: {'yes' if run_ns else 'no (--no-ns)'}  |  "
          f"Report: {'yes' if run_report else 'no (--no-report)'}")
    print(f"{'='*64}")

    from datetime import datetime
    def _step(msg): print(f"\n  {msg}  [{datetime.now().strftime('%H:%M:%S')}]", flush=True)

    # ── Step 0: Projections ──────────────────────────────────────────────────
    if inject_tag:
        if not os.path.exists(proj_dir) or not os.listdir(proj_dir):
            raise RuntimeError(f"inject-tag projections not found at {proj_dir}. "
                               f"Run the {inject_tag} pipeline first.")
        _step(f"[Step 0] Using existing projections from {inject_tag}")
    elif not os.listdir(proj_dir):
        _step("[Step 0] Generating projections...")
        generate_projections(model, proj_dir, rc_dir=rc_dir)
    else:
        _step("[Step 0] Projections exist — skipping")

    # ── Step 1: SM-only fit ──────────────────────────────────────────────────
    _step("[Step 1] SM-only fit")
    sm_id = f"{tag}_SMonly"
    sm_rc = f"{rc_dir}/{sm_id}.yaml"
    rc = make_smonly_runcard(sm_id, fit_ops, mWp_TeV=_model_mass(model))   # uses SM_DATA, not BSM proj_dir
    rc["result_path"] = fit_dir
    with open(sm_rc, "w") as f:
        yaml.dump(rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    if not os.path.exists(f"{fit_dir}/{sm_id}/fit_results.json") or not skip_existing:
        _run_smefit("A", sm_rc, "SM-only baseline fit (SM data)", fit_dir)
    else:
        print(f"\n  [A] SM-only fit exists — skipping")

    # ── Step 2: Full SMEFT fit ───────────────────────────────────────────────
    _step("[Step 2] BSM closure SMEFT fit")
    bsm_id = f"{tag}_BSMclosure_SMEFT"
    bsm_rc = f"{rc_dir}/{bsm_id}.yaml"
    rc = make_smeft_runcard(bsm_id, proj_dir, model, free_ops=free_ops)  # BSM data, tight priors
    rc["result_path"] = fit_dir
    with open(bsm_rc, "w") as f:
        yaml.dump(rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    if not os.path.exists(f"{fit_dir}/{bsm_id}/fit_results.json") or not skip_existing:
        _run_smefit("A", bsm_rc, "Full SMEFT fit", fit_dir)
    else:
        print(f"\n  [A] Full SMEFT fit exists — skipping")

    # ── Step 3: UV coupling fit (NS or analytic A) ───────────────────────────
    _step("[Step 3] UV coupling fit")
    if uv_analytic:
        uv_id = f"{tag}_UVcoup"
        uv_rc = f"{rc_dir}/{uv_id}.yaml"
        rc = make_uv_runcard(uv_id, proj_dir, model, use_quad=False)
        rc["result_path"] = fit_dir
        with open(uv_rc, "w") as f:
            yaml.dump(rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        if not os.path.exists(f"{fit_dir}/{uv_id}/fit_results.json") or not skip_existing:
            _run_smefit("A", uv_rc, "UV coupling analytic fit (--uv-a)", fit_dir)
        else:
            print(f"\n  [A] UV analytic fit exists — skipping")
    elif run_ns:
        uv_id = f"{tag}_UVcoup"
        uv_rc = f"{rc_dir}/{uv_id}.yaml"
        rc = make_uv_runcard(uv_id, proj_dir, model)
        rc["result_path"] = fit_dir
        with open(uv_rc, "w") as f:
            yaml.dump(rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        if not os.path.exists(f"{fit_dir}/{uv_id}/fit_results.json") or not skip_existing:
            _run_smefit("NS", uv_rc, "UV coupling NS fit  (~15-30 min)", fit_dir)
        else:
            print(f"\n  [NS] UV fit exists — skipping")
    else:
        print(f"\n  [NS] Skipped (--no-ns)")

    # ── Step 4: PCA ──────────────────────────────────────────────────────────
    _step("[Step 4] PCA")
    _run_pca(model, proj_dir, pca_dir, free_ops=free_ops)

    # ── Step 5: Profile likelihood summary ───────────────────────────────────
    _step("[Step 5] Profile likelihood summary")
    _compute_summary(model, tag, fit_dir, pca_dir, plt_dir, out, free_ops=free_ops)

    # ── Step 6: smefit report ─────────────────────────────────────────────────
    if run_report:
        _step("[Step 6] Generating smefit report")
        _run_report(tag, rc_dir, report_dir, proj_dir, fit_dir, model, free_ops=free_ops)
    else:
        _step("[Step 6] Report skipped (--no-report)")

    print(f"\n{'='*64}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Summary : {out}/summary.txt")
    print(f"  Plot    : {plt_dir}/significance_summary.png")
    if run_report:
        print(f"  Report  : {report_dir}/Report_{tag}/")
    print(f"{'='*64}\n")


# ── Step 7b: L1 scan with theory-covmat band ──────────────────────────────────

def run_scan_l1(model_cls, base_params, tag_prefix, coupling_key, coupling_grid,
                mass_key, mass_grid, plt_dir, out_dir, n_reps=100, seed=42):
    """
    Discovery reach scan with two uncertainty bands — identical to --scan but adds:

      L1 band   — for each grid point: n_reps analytic noise draws
                  ε ~ N(0, C_exp), delta_r = delta_L0 + ε. Reports 16th/50th/84th
                  percentile of sigma_UV across replicas.
      TC band   — same L0 Asimov data refit with theory_cov_aggressive added to Ci.

    Uses the SAME L0 projections and scan_tag format as run_scan, so existing
    projections from --scan are reused automatically. No extra smefit PROJ calls
    if --scan has already been run.

    Outputs (results/<tag>_l1scan/):
      significance_vs_mass.png  — 1D per-coupling panel: L1 band + TC line
      discovery_reach_l1.png    — 2D (mass, coupling) contours with L1 + TC bands

    Usage:
        python run_pipeline.py --model wprime_constrained --gWH 0.5 --mWp 7.0 --scan-l1
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    os.makedirs(plt_dir, exist_ok=True)
    os.makedirs(out_dir,  exist_ok=True)

    dummy = model_cls(**base_params)
    ops   = dummy.OPERATORS

    # ── Build the three covariance objects (once, shared across all points) ──
    print(f"\n[L1 scan] Building Ci (no theory covmat) ...")
    _K_CI_CACHE.clear()
    CFG.use_theory_covmat = False
    K_no_tc, Ci_no_tc = _build_K_Ci(ops)

    print(f"[L1 scan] Building Ci (theory_cov_aggressive) ...")
    _K_CI_CACHE.clear()
    CFG.use_theory_covmat  = True
    CFG.theory_cov_variant = "theory_cov_aggressive"
    K_tc, Ci_tc = _build_K_Ci(ops)
    _K_CI_CACHE.clear()
    CFG.use_theory_covmat  = False   # restore

    print(f"\n[L1 scan] Grid: {coupling_key}={coupling_grid}")
    print(f"          Grid: {mass_key}={mass_grid}")
    print(f"          n_reps={n_reps}  seed={seed}\n")

    results = []
    n_total = len(coupling_grid) * len(mass_grid)
    n_done  = 0

    for g_val in coupling_grid:
        for m_val in mass_grid:
            n_done += 1
            # ── build model ──────────────────────────────────────────────────
            params = {**base_params, coupling_key: g_val, mass_key: m_val}
            if "gWLf11" in params:
                ratio = g_val / base_params.get(coupling_key, g_val)
                for fk in ["gWLf11", "gWLf22", "gWLf33", "gWqf33"]:
                    if fk in params:
                        params[fk] = base_params[fk] * ratio
            model = model_cls(**params)

            # ── per-grid-point parent directory ──────────────────────────────
            scan_tag  = (f"{tag_prefix}_l1scan_{coupling_key}{int(g_val*100):03d}"
                         f"_{mass_key}{int(m_val*10):03d}")
            point_dir = PIPELINE / "projections" / scan_tag

            def _has_datasets(d):
                return (os.path.isdir(d) and
                        any(f.endswith(".yaml") and not f.startswith("proj_")
                            for f in os.listdir(d)))

            # ── generate n_reps L1 replicas via smefit PROJ --noise L1 ───────
            sigma_no_tc_reps, sigma_tc_reps = [], []
            for r in range(n_reps):
                rep_dir = str(point_dir / f"rep{r:03d}")
                if not _has_datasets(rep_dir):
                    generate_projections(model, rep_dir, noise_level="L1")

                delta_r     = _build_delta(rep_dir)
                sigma_no_tc_reps.append(
                    _point_significance(model, delta_r, K_no_tc, Ci_no_tc, ops)["sigma_uv"])
                sigma_tc_reps.append(
                    _point_significance(model, delta_r, K_tc,    Ci_tc,    ops)["sigma_uv"])

            p16,    p50,    p84    = np.percentile(sigma_no_tc_reps, [16, 50, 84])
            p16_tc, p50_tc, p84_tc = np.percentile(sigma_tc_reps,    [16, 50, 84])

            pt = {
                coupling_key:        g_val,
                mass_key:            m_val,
                "sigma_l1_16":       p16,
                "sigma_l1_50":       p50,
                "sigma_l1_84":       p84,
                "sigma_l1_tc_16":    p16_tc,
                "sigma_l1_tc_50":    p50_tc,
                "sigma_l1_tc_84":    p84_tc,
            }
            results.append(pt)
            print(f"  [{n_done:3d}/{n_total}] {coupling_key}={g_val:.3f}  {mass_key}={m_val:.2f} TeV"
                  f"  σ_L1(noTC)=[{p16:.2f},{p50:.2f},{p84:.2f}]"
                  f"  σ_L1(TC)=[{p16_tc:.2f},{p50_tc:.2f},{p84_tc:.2f}]")

    # ── Save table ────────────────────────────────────────────────────────────
    tbl = os.path.join(out_dir, "scan_l1_table.txt")
    with open(tbl, "w") as f:
        f.write(f"# L1 scan (smefit PROJ --noise L1): {tag_prefix}  n_reps={n_reps}  TC=aggressive\n")
        f.write(f"# {coupling_key:>8}  {mass_key:>8}  "
                f"{'l1_p16':>8}  {'l1_p50':>8}  {'l1_p84':>8}  "
                f"{'tc_p16':>8}  {'tc_p50':>8}  {'tc_p84':>8}\n")
        for r in results:
            f.write(f"  {r[coupling_key]:>8.4f}  {r[mass_key]:>8.2f}  "
                    f"{r['sigma_l1_16']:>8.3f}  {r['sigma_l1_50']:>8.3f}  {r['sigma_l1_84']:>8.3f}  "
                    f"{r['sigma_l1_tc_16']:>8.3f}  {r['sigma_l1_tc_50']:>8.3f}  {r['sigma_l1_tc_84']:>8.3f}\n")
    print(f"\n  Saved: {tbl}")

    # ── 1D plot: significance vs mass per coupling ────────────────────────────
    nc    = len(coupling_grid)
    ncols = min(nc, 3)
    nrows = (nc + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.2 * nrows), squeeze=False)
    for ic, g_val in enumerate(coupling_grid):
        ax  = axes[ic // ncols][ic % ncols]
        pts = sorted([r for r in results if abs(r[coupling_key] - g_val) < 1e-6],
                     key=lambda r: r[mass_key])
        if not pts:
            ax.set_visible(False); continue
        masses = [p[mass_key]         for p in pts]
        ax.fill_between(masses,
                        [p["sigma_l1_16"]    for p in pts],
                        [p["sigma_l1_84"]    for p in pts],
                        alpha=0.25, color="steelblue", label="L1 68% (no TC)")
        ax.plot(masses, [p["sigma_l1_50"]    for p in pts],
                color="steelblue", lw=2.0,           label="L1 median (no TC)")
        ax.fill_between(masses,
                        [p["sigma_l1_tc_16"] for p in pts],
                        [p["sigma_l1_tc_84"] for p in pts],
                        alpha=0.20, color="crimson",   label="L1 68% (TC aggressive)")
        ax.plot(masses, [p["sigma_l1_tc_50"] for p in pts],
                color="crimson",   lw=2.0, ls="--",  label="L1 median (TC aggressive)")
        for thr, col, lbl in [(5.0, "gold", r"5$\sigma$"), (3.0, "orange", r"3$\sigma$")]:
            ax.axhline(thr, color=col, lw=1.2, ls="--")
            ax.text(masses[-1] * 0.97, thr + 0.15, lbl, color=col, ha="right", fontsize=9)
        ax.set_xlabel(f"{mass_key}  [TeV]", fontsize=11)
        ax.set_ylabel(r"$\sigma_\mathrm{UV}$", fontsize=11)
        ax.set_title(fr"${coupling_key} = {g_val:.3f}$", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
    for idx in range(nc, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle(f"Discovery reach — {tag_prefix}  (L1: {n_reps} reps, TC aggressive)",
                 fontsize=12)
    plt.tight_layout()
    out1 = os.path.join(plt_dir, "significance_vs_mass.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.savefig(out1.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  Saved: {out1}")
    plt.close()

    # ── 2D reach contour plot ─────────────────────────────────────────────────
    if len(coupling_grid) >= 3 and len(mass_grid) >= 3:
        nc2, nm2 = len(coupling_grid), len(mass_grid)
        mg = np.array(mass_grid)
        cg = np.array(coupling_grid)

        def _grid(key):
            arr = np.full((nc2, nm2), np.nan)
            for r in results:
                ic = coupling_grid.index(r[coupling_key])
                im = mass_grid.index(r[mass_key])
                arr[ic, im] = r[key]
            return arr

        G_l1_16    = _grid("sigma_l1_16")
        G_l1_50    = _grid("sigma_l1_50")
        G_l1_84    = _grid("sigma_l1_84")
        G_l1_tc_16 = _grid("sigma_l1_tc_16")
        G_l1_tc_50 = _grid("sigma_l1_tc_50")
        G_l1_tc_84 = _grid("sigma_l1_tc_84")

        fig2, ax = plt.subplots(figsize=(8, 5.5))
        for thr in [5.0, 3.0]:
            # no-TC: blue band where 5σ boundary fluctuates
            band_notc = (G_l1_84 >= thr) & (G_l1_16 < thr)
            if band_notc.any():
                ax.contourf(mg, cg, band_notc.astype(float),
                            levels=[0.5, 1.5], colors=["steelblue"], alpha=0.20)
            # TC: red band
            band_tc = (G_l1_tc_84 >= thr) & (G_l1_tc_16 < thr)
            if band_tc.any():
                ax.contourf(mg, cg, band_tc.astype(float),
                            levels=[0.5, 1.5], colors=["crimson"], alpha=0.15)
            try:
                cs = ax.contour(mg, cg, G_l1_50, levels=[thr],
                                colors=["steelblue"], linewidths=2.0)
                ax.clabel(cs, fmt=f"{thr:.0f}σ", fontsize=8)
            except Exception:
                pass
            try:
                cs = ax.contour(mg, cg, G_l1_tc_50, levels=[thr],
                                colors=["crimson"], linewidths=2.0, linestyles="--")
                ax.clabel(cs, fmt=f"{thr:.0f}σ TC", fontsize=8)
            except Exception:
                pass
        handles = [
            Line2D([0],[0], color="steelblue", lw=2,    label="L1 median (no TC)"),
            Line2D([0],[0], color="crimson",   lw=2, ls="--", label="L1 median (TC aggressive)"),
            Patch(facecolor="steelblue", alpha=0.3, label="L1 68% band (no TC)"),
            Patch(facecolor="crimson",   alpha=0.3, label="L1 68% band (TC aggressive)"),
        ]
        ax.legend(handles=handles, fontsize=9, loc="upper right")
        ax.set_xlabel(f"{mass_key}  [TeV]", fontsize=12)
        ax.set_ylabel(coupling_key, fontsize=12)
        ax.set_title(f"Discovery reach — {tag_prefix}  (n_reps={n_reps}, TC aggressive)",
                     fontsize=11)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        out2 = os.path.join(plt_dir, "discovery_reach_l1.png")
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        plt.savefig(out2.replace(".png", ".pdf"), bbox_inches="tight")
        print(f"  Saved: {out2}")
        plt.close()

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FCC-ee BSM discovery pipeline — single point or scan",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=["wprime", "zprime", "wprime_constrained", "wprime_constrained_v2", "wprime_constrained_v3", "zprime_constrained", "comphiggs", "wprime_universal", "wprime_1g"], required=True,
                   help="UV model to USE FOR FITTING (injection model set by --inject-tag)")
    p.add_argument("--tag",   default=None, help="Run tag (auto if omitted)")
    p.add_argument("--inject-tag", default=None,
                   help="Use pseudo-data from this existing tag instead of generating new projections. "
                        "Enables robustness tests: e.g. inject W' data, fit with Z' UV model.")
    p.add_argument("--scan",     action="store_true",
                   help="Run discovery scan over (coupling, mass) grid (L0 Asimov, analytic)")
    p.add_argument("--scan-l1", action="store_true",
                   help="Run L1 scan: L1 noise band + theory_cov_aggressive band on reach contour")
    p.add_argument("--n-l1-reps", type=int, default=100,
                   help="Number of analytic L1 noise replicas per grid point for --scan-l1 (default 100)")
    p.add_argument("--no-ns", action="store_true",
                   help="Skip the slow UV NS fit (Steps 1,2,4,5,6 still run)")
    p.add_argument("--uv-a", action="store_true",
                   help="Run UV fit with fast analytic (A) mode instead of NS")
    p.add_argument("--no-report", action="store_true",
                   help="Skip smefit report generation (Step 6)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip fits whose fit_results.json already exists (resume mode)")
    p.add_argument("--no-rge", action="store_true",
                   help="Disable SMEFT RGE running (Wilson coefficients used at matching scale)")
    p.add_argument("--no-theory-covmat", action="store_true",
                   help="Exclude theory covariance matrix from fits and analytic significance "
                        "(most optimistic scenario — treats theory predictions as exact)")
    p.add_argument("--theory-cov", default="aggressive",
                   choices=["aggressive", "conservative", "current"],
                   help="Theory covariance variant to use (default: aggressive)")
    p.add_argument("--sepproj", nargs="*", metavar="OP",
                   help="Run separated-projection mode: inject only one operator at a time "
                        "and fit with only that operator free. Specify operators explicitly "
                        "(e.g. --sepproj OQl13 OpQM) or omit operators to run all model operators. "
                        "Outputs to results/<tag>/sepproj/. Cannot be combined with --scan.")
    p.add_argument("--free-ops", nargs="+", metavar="OP", default=None,
                   help="Restrict SMEFT fit to only these operators. Pseudo-data still contains "
                        "contributions from all model operators. Use with --inject-tag to reuse "
                        "existing pseudo-data. E.g. --free-ops OQl13 OQl1m O3pQ3 OpQM O3pl2")
    # W' parameters
    p.add_argument("--gWH",  type=float, default=0.12)
    p.add_argument("--mWp",  type=float, default=1.0)
    p.add_argument("--gWLf", type=float, default=None, help="default: gWH/3")
    p.add_argument("--gWqf", type=float, default=None, help="default: gWH/3")
    p.add_argument("--g",    type=float, default=None, help="single universal coupling for wprime_1g")
    # Z' parameters
    p.add_argument("--gZH",   type=float, default=0.12)
    p.add_argument("--gZl",   type=float, default=0.04)
    p.add_argument("--mZp",   type=float, default=1.0)
    # Composite Higgs model parameters
    p.add_argument("--g_rho", type=float, default=2.0,  help="CHM strong-sector coupling")
    p.add_argument("--m_rho", type=float, default=10.0, help="CHM resonance mass [TeV]")
    return p.parse_args()


if __name__ == "__main__":
    from models.wprime import WPrimeModel
    from models.wprime_constrained import WPrimeConstrainedModel
    from models.wprime_constrained_v2 import WPrimeConstrainedV2Model
    from models.wprime_constrained_v3 import WPrimeConstrainedV3Model
    from models.wprime_universal import WPrimeUniversalModel
    from models.wprime_1g import WPrime1gModel
    from models.zprime import ZPrimeModel
    from models.zprime_constrained import ZPrimeConstrainedModel
    from models.comphiggs import CompHiggsModel

    args = parse_args()

    # Apply theory covmat setting globally before any computation
    if args.no_rge:
        CFG.use_rge = False
        print("  [config] RGE DISABLED (--no-rge)")
    if args.no_theory_covmat:
        CFG.use_theory_covmat = False
        print("  [config] Theory covariance matrix DISABLED (--no-theory-covmat)")
    CFG.theory_cov_variant = f"theory_cov_{args.theory_cov}"
    print(f"  [config] Theory covariance variant: {CFG.theory_cov_variant}")

    if args.model == "wprime":
        gWLf       = args.gWLf if args.gWLf is not None else args.gWH / 3
        gWqf       = args.gWqf if args.gWqf is not None else args.gWH / 3
        model      = WPrimeModel(gWH=args.gWH, gWLf11=gWLf, gWLf22=gWLf,
                                 gWLf33=gWLf, gWqf33=gWqf, mWp=args.mWp)
        base_params = {"gWH": args.gWH, "gWLf11": gWLf, "gWLf22": gWLf,
                       "gWLf33": gWLf, "gWqf33": gWqf, "mWp": args.mWp}
        tag         = args.tag or f"wprime_gwh{int(args.gWH*100):03d}_mwp{int(args.mWp*10):03d}"
        coupling_key, mass_key = "gWH", "mWp"
        model_cls   = WPrimeModel
        # Scan grid for W': scale around the injected point so the 5sigma
        # contour (g ∝ m) is well-sampled regardless of input parameters.
        g0, m0 = args.gWH, args.mWp
        coupling_grid = sorted(set(round(v, 4) for v in np.linspace(g0 * 0.25, g0 * 3.0, 9).tolist()))
        mass_grid     = sorted(set(round(v, 2) for v in np.linspace(m0 * 0.1,  m0 * 3.0, 11).tolist()))
        scan_params   = {**base_params}

    elif args.model == "wprime_constrained":
        gWLf       = args.gWLf if args.gWLf is not None else args.gWH / 3
        gWqf       = args.gWqf if args.gWqf is not None else args.gWH / 3
        model      = WPrimeConstrainedModel(gWH=args.gWH, gWLf11=gWLf, gWLf22=gWLf,
                                            gWLf33=gWLf, gWqf33=gWqf, mWp=args.mWp)
        base_params = {"gWH": args.gWH, "gWLf11": gWLf, "gWLf22": gWLf,
                       "gWLf33": gWLf, "gWqf33": gWqf, "mWp": args.mWp}
        tag         = args.tag or f"wprime_constrained_gwh{int(args.gWH*100):03d}_mwp{int(args.mWp*10):03d}"
        coupling_key, mass_key = "gWH", "mWp"
        model_cls   = WPrimeConstrainedModel
        g0, m0 = args.gWH, args.mWp
        coupling_grid = sorted(set(round(v, 4) for v in np.linspace(g0 * 0.25, g0 * 3.0, 9).tolist()))
        mass_grid     = sorted(set(round(v, 2) for v in np.linspace(m0 * 0.1,  m0 * 3.0, 11).tolist()))
        scan_params   = {**base_params}

    elif args.model == "wprime_constrained_v2":
        gWLf       = args.gWLf if args.gWLf is not None else args.gWH / 3
        gWqf       = args.gWqf if args.gWqf is not None else args.gWH / 3
        model      = WPrimeConstrainedV2Model(gWH=args.gWH, gWLf11=gWLf, gWLf22=gWLf,
                                              gWLf33=gWLf, gWqf33=gWqf, mWp=args.mWp)
        base_params = {"gWH": args.gWH, "gWLf11": gWLf, "gWLf22": gWLf,
                       "gWLf33": gWLf, "gWqf33": gWqf, "mWp": args.mWp}
        tag         = args.tag or f"wprime_constrained_v2_gwh{int(args.gWH*100):03d}_mwp{int(args.mWp*10):03d}"
        coupling_key, mass_key = "gWH", "mWp"
        model_cls   = WPrimeConstrainedV2Model
        g0, m0 = args.gWH, args.mWp
        coupling_grid = sorted(set(round(v, 4) for v in np.linspace(g0 * 0.25, g0 * 3.0, 9).tolist()))
        mass_grid     = sorted(set(round(v, 2) for v in np.linspace(m0 * 0.1,  m0 * 3.0, 11).tolist()))
        scan_params   = {**base_params}

    elif args.model == "wprime_constrained_v3":
        gWLf       = args.gWLf if args.gWLf is not None else args.gWH / 3
        gWqf       = args.gWqf if args.gWqf is not None else args.gWH / 3
        model      = WPrimeConstrainedV3Model(gWH=args.gWH, gWLf11=gWLf, gWLf22=gWLf,
                                              gWLf33=gWLf, gWqf33=gWqf, mWp=args.mWp)
        base_params = {"gWH": args.gWH, "gWLf11": gWLf, "gWLf22": gWLf,
                       "gWLf33": gWLf, "gWqf33": gWqf, "mWp": args.mWp}
        tag         = args.tag or f"wprime_constrained_v3_gwh{int(args.gWH*100):03d}_mwp{int(args.mWp*10):03d}"
        coupling_key, mass_key = "gWH", "mWp"
        model_cls   = WPrimeConstrainedV3Model
        g0, m0 = args.gWH, args.mWp
        coupling_grid = sorted(set(round(v, 4) for v in np.linspace(g0 * 0.25, g0 * 3.0, 9).tolist()))
        mass_grid     = sorted(set(round(v, 2) for v in np.linspace(m0 * 0.1,  m0 * 3.0, 11).tolist()))
        scan_params   = {**base_params}

    elif args.model == "zprime":
        model       = ZPrimeModel(gZH=args.gZH, gZl=args.gZl, mZp=args.mZp)
        base_params = {"gZH": args.gZH, "gZl": args.gZl, "mZp": args.mZp}
        tag         = args.tag or f"zprime_gzh{int(args.gZH*100):03d}_mzp{int(args.mZp*10):03d}"
        coupling_key, mass_key = "gZH", "mZp"
        model_cls   = ZPrimeModel
        # Scan grid for Z': scale around the injected point
        g0, m0 = args.gZH, args.mZp
        coupling_grid = sorted(set(round(v, 4) for v in np.linspace(g0 * 0.25, g0 * 3.0, 9).tolist()))
        mass_grid     = sorted(set(round(v, 2) for v in np.linspace(m0 * 0.1,  m0 * 3.0, 11).tolist()))
        scan_params   = {**base_params}

    elif args.model == "zprime_constrained":
        model       = ZPrimeConstrainedModel(gZH=args.gZH, gZl=args.gZl, mZp=args.mZp)
        base_params = {"gZH": args.gZH, "gZl": args.gZl, "mZp": args.mZp}
        tag         = args.tag or f"zprime_constrained_gzh{int(args.gZH*100):03d}_mzp{int(args.mZp*10):03d}"
        coupling_key, mass_key = "gZH", "mZp"
        model_cls   = ZPrimeConstrainedModel
        g0, m0 = args.gZH, args.mZp
        coupling_grid = sorted(set(round(v, 4) for v in np.linspace(g0 * 0.25, g0 * 3.0, 9).tolist()))
        mass_grid     = sorted(set(round(v, 2) for v in np.linspace(m0 * 0.1,  m0 * 3.0, 11).tolist()))
        scan_params   = {**base_params}

    elif args.model == "comphiggs":
        model       = CompHiggsModel(g_rho=args.g_rho, m_rho=args.m_rho)
        base_params = {"g_rho": args.g_rho, "m_rho": args.m_rho}
        tag         = args.tag or f"comphiggs_grho{int(args.g_rho*100):03d}_mrho{int(args.m_rho*10):03d}"
        coupling_key, mass_key = "g_rho", "m_rho"
        model_cls   = CompHiggsModel
        g0, m0 = args.g_rho, args.m_rho
        coupling_grid = sorted(set(round(v, 4) for v in np.linspace(g0 * 0.25, g0 * 3.0, 9).tolist()))
        mass_grid     = sorted(set(round(v, 2) for v in np.linspace(m0 * 0.1,  m0 * 3.0, 11).tolist()))
        scan_params   = {**base_params}

    elif args.model == "wprime_universal":
        model       = WPrimeUniversalModel(gWH=args.gWH, mWp=args.mWp)
        base_params = {"gWH": args.gWH, "mWp": args.mWp}
        tag         = args.tag or f"wprime_universal_gwh{int(args.gWH*100):03d}_mwp{int(args.mWp*10):03d}"
        coupling_key, mass_key = "gWH", "mWp"
        model_cls   = WPrimeUniversalModel
        g0, m0 = args.gWH, args.mWp
        coupling_grid = sorted(set(round(v, 4) for v in np.linspace(g0 * 0.25, g0 * 3.0, 9).tolist()))
        mass_grid     = sorted(set(round(v, 2) for v in np.linspace(m0 * 0.1,  m0 * 3.0, 11).tolist()))
        scan_params   = {**base_params}

    elif args.model == "wprime_1g":
        g_val       = args.g if args.g is not None else args.gWH
        model       = WPrime1gModel(g=g_val, mWp=args.mWp)
        base_params = {"g": g_val, "mWp": args.mWp}
        tag         = args.tag or f"wprime_1g_g{int(g_val*100):03d}_mwp{int(args.mWp*10):03d}"
        coupling_key, mass_key = "g", "mWp"
        model_cls   = WPrime1gModel
        g0, m0 = g_val, args.mWp
        coupling_grid = sorted(set(round(v, 4) for v in np.linspace(g0 * 0.25, g0 * 3.0, 9).tolist()))
        mass_grid     = sorted(set(round(v, 2) for v in np.linspace(m0 * 0.1,  m0 * 3.0, 11).tolist()))
        scan_params   = {**base_params}

    if args.sepproj is not None:
        # Separated-projection mode: per-operator signal isolation
        ops_to_do = args.sepproj if args.sepproj else model.OPERATORS
        run_sepproj(model, tag, ops_to_do, skip_existing=args.skip_existing)

    elif args.scan:
        # Discovery scan — analytic only, no NS
        scan_out  = str(PIPELINE / "results" / f"{tag}_scan")
        scan_plt  = f"{scan_out}/plots"
        os.makedirs(scan_plt, exist_ok=True)

        scan_params_fixed = {k: v for k, v in base_params.items()
                             if k not in (coupling_key, mass_key)}

        run_scan(
            model_cls    = model_cls,
            base_params  = base_params,
            tag_prefix   = args.model,
            coupling_key = coupling_key,
            coupling_grid= coupling_grid,
            mass_key     = mass_key,
            mass_grid    = mass_grid,
            plt_dir      = scan_plt,
            out_dir      = scan_out,
        )

    elif args.scan_l1:
        # L1 scan: L1 noise band + theory_cov_aggressive band
        scan_out = str(PIPELINE / "results" / f"{tag}_l1scan")
        scan_plt = f"{scan_out}/plots"
        os.makedirs(scan_plt, exist_ok=True)

        run_scan_l1(
            model_cls    = model_cls,
            base_params  = base_params,
            tag_prefix   = args.model,
            coupling_key = coupling_key,
            coupling_grid= coupling_grid,
            mass_key     = mass_key,
            mass_grid    = mass_grid,
            plt_dir      = scan_plt,
            out_dir      = scan_out,
            n_reps       = args.n_l1_reps,
        )

    else:
        # Single-point full pipeline
        run_pipeline(
            model,
            tag,
            skip_existing = args.skip_existing,
            run_ns        = not args.no_ns and not args.uv_a,
            uv_analytic   = args.uv_a,
            run_report    = not args.no_report,
            inject_tag    = args.inject_tag,
            free_ops      = args.free_ops,
        )
