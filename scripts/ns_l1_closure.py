"""
ns_l1_closure.py
----------------
NS-based L1 closure test for BSM UV models.

For each of N_rep level-1 replicas:
  1. Generate d^(r) = d_BSM^central + ε^(r),  ε ~ N(0, C_exp)
  2. Write fluctuated central values into a temporary projection directory
  3. Run smefit NS on the fluctuated data (full nonlinear, quadratic-EFT fit)
  4. Extract UV coupling posterior: (g_fit^(r), σ_g^(r))
  5. Compute UV pull: P^(r) = (g_fit^(r) - g_truth) / σ_g^(r)

If the BSM closure test is genuine, P^(r) ~ N(0,1).

Unlike bsm_closure_metrics.py (which uses the analytic Fisher approximation
and is self-consistent by construction), this test is NOT circular: NS
solves the actual quadratic-EFT nonlinear inverse problem using the full
likelihood, so the pull distribution carries real information.

Note on covariance: replicas are generated from C_total = C_exp + C_th
(loaded from pca/C_inv.npy). NS fits inherit use_theory_covmat=True and
theory_cov_variant=theory_cov_aggressive from the base runcard, so noise
and fit covariance are consistent.

Note on Z2 degeneracy: for models where all operators scale as g^2 the NS
posterior is bimodal at ±|g_truth|. The script folds the samples to |g|
before computing pulls in that case (auto-detected from uv_param_names).
For CHM the Otap operator is linear in g_rho, which breaks Z2 so no
folding is needed.

Usage:
    python scripts/ns_l1_closure.py --tag comphiggs_grho200_mrho1000 --nrep 50
    python scripts/ns_l1_closure.py --tag comphiggs_grho200_mrho1000 --nrep 50 --collect-only
    python scripts/ns_l1_closure.py --tag wprime_constrained_gwh012_mwp030 --nrep 20
"""
import sys, os, argparse, yaml, json, shutil, subprocess, copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm as norm_dist
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

# absolute path to smefit binary — avoids PATH issues when launched via nohup
# or ProcessPoolExecutor workers that don't inherit the conda environment PATH
SMEFIT_BIN = str(Path(sys.executable).parent / "smefit")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(tag):
    """Infer model class and truth params from the result tag."""
    from models.comphiggs               import CompHiggsModel
    from models.wprime_constrained      import WPrimeConstrainedModel
    from models.wprime_constrained_v2   import WPrimeConstrainedV2Model
    from models.wprime_constrained_v3   import WPrimeConstrainedV3Model
    from models.wprime_universal        import WPrimeUniversalModel
    from models.zprime_constrained      import ZPrimeConstrainedModel

    # strip any leading subdirectory prefix (e.g. "closure_contour/scan/comphiggs_...")
    tag = Path(tag).name

    if tag.startswith("comphiggs"):
        parts = tag.split("_")
        g   = float(parts[1].replace("grho", "")) / 100.0
        m   = float(parts[2].replace("mrho", "")) / 10.0
        return CompHiggsModel(g_rho=g, m_rho=m)
    if tag.startswith("wprime_constrained_v3"):
        parts = tag.split("_")
        g   = float(parts[3].replace("gwh", "")) / 100.0
        m   = float(parts[4].replace("mwp", "")) / 10.0
        f   = 1.0 / 3.0
        return WPrimeConstrainedV3Model(gWH=g, gWLf11=g*f, gWLf22=g*f,
                                        gWLf33=g*f, gWqf33=g*f, mWp=m)
    if tag.startswith("wprime_constrained_v2"):
        # e.g. wprime_constrained_v2_gwh050_mwp050
        parts = tag.split("_")
        g   = float(parts[3].replace("gwh", "")) / 100.0
        m   = float(parts[4].replace("mwp", "")) / 10.0
        f   = 1.0 / 3.0
        return WPrimeConstrainedV2Model(gWH=g, gWLf11=g*f, gWLf22=g*f,
                                        gWLf33=g*f, gWqf33=g*f, mWp=m)
    if tag.startswith("wprime_constrained"):
        # e.g. wprime_constrained_gwh012_mwp030 -> ['wprime','constrained','gwh012','mwp030']
        parts = tag.split("_")
        g   = float(parts[2].replace("gwh", "")) / 100.0
        m   = float(parts[3].replace("mwp", "")) / 10.0
        f   = 1.0 / 3.0
        return WPrimeConstrainedModel(gWH=g, gWLf11=g*f, gWLf22=g*f,
                                      gWLf33=g*f, gWqf33=g*f, mWp=m)
    if tag.startswith("wprime_gwh"):
        # e.g. wprime_gwh050_mwp050 -> gWH=0.50, mWp=5.0 TeV, full 5-param model (gWLf=gWH/3)
        from models.wprime import WPrimeModel
        parts = tag.split("_")
        g   = float(parts[1].replace("gwh", "")) / 100.0
        m   = float(parts[2].replace("mwp", "")) / 10.0
        f   = 1.0 / 3.0
        return WPrimeModel(gWH=g, gWLf11=g*f, gWLf22=g*f, gWLf33=g*f, gWqf33=g*f, mWp=m)
    if tag.startswith("zprime_constrained"):
        # e.g. zprime_constrained_gzh012_mzp075 -> ['zprime','constrained','gzh012','mzp075']
        parts = tag.split("_")
        g   = float(parts[2].replace("gzh", "")) / 100.0
        m   = float(parts[3].replace("mzp", "")) / 10.0
        f   = 1.0 / 3.0
        return ZPrimeConstrainedModel(gZH=g, gZl=g*f, mZp=m)
    if tag.startswith("zprime_gzh"):
        # e.g. zprime_gzh012_mzp030 -> gZH=0.12, gZl=0.04 (=gZH/3), mZp=3.0
        from models.zprime import ZPrimeModel
        parts = tag.split("_")
        g   = float(parts[1].replace("gzh", "")) / 100.0
        m   = float(parts[2].replace("mzp", "")) / 10.0
        f   = 1.0 / 3.0
        return ZPrimeModel(gZH=g, gZl=g*f, mZp=m)
    raise ValueError(f"Cannot infer model from tag: {tag}")


def load_data_central(proj_dir, datasets):
    """
    Read data_central from each dataset yaml in runcard order.
    Returns (d_central, slices) where slices maps dataset name -> slice.
    """
    d_parts, slices, idx = [], {}, 0
    for ds in datasets:
        name = ds["name"]
        dat  = yaml.safe_load(open(Path(proj_dir) / f"{name}.yaml"))
        vals = dat["data_central"]
        if isinstance(vals, (int, float)):
            vals = [vals]
        d_parts.append(np.array(vals, dtype=float))
        slices[name] = slice(idx, idx + len(vals))
        idx += len(vals)
    return np.concatenate(d_parts), slices


def _replace_data_central(src_text, new_vals):
    """
    Rewrite data_central in raw yaml text without round-tripping through
    yaml.dump (which changes formatting and breaks smefit's loader).

    Handles two formats:
      scalar:  "data_central: 1.234\n"
      list:    "data_central:\n- 1.0\n- 2.0\n..."
    """
    import re

    # scalar case: single value on the same line
    scalar_pat = re.compile(r'^(data_central:\s*)[\d.eE+\-]+(\s*)$', re.MULTILINE)
    if scalar_pat.search(src_text):
        return scalar_pat.sub(
            lambda m: f"{m.group(1)}{repr(float(new_vals[0]))}{m.group(2)}",
            src_text,
        )

    # list case: "data_central:\n" followed by "- value\n" lines
    # find the block and replace each "- old_val" line with "- new_val"
    list_header = re.compile(r'^data_central:\s*\n', re.MULTILINE)
    m = list_header.search(src_text)
    if m is None:
        raise ValueError("Cannot find data_central block in yaml")

    start = m.end()
    item_pat = re.compile(r'^- [\d.eE+\-]+\n', re.MULTILINE)
    new_lines = [f"- {repr(float(v))}\n" for v in new_vals]
    out, pos, idx = src_text[:start], start, 0
    for item_m in item_pat.finditer(src_text, start):
        if item_m.start() != pos:
            break   # non-item line → end of data_central block
        out += new_lines[idx]
        pos  = item_m.end()
        idx += 1
        if idx == len(new_lines):
            break
    out += src_text[pos:]
    return out


def write_replica_projections(proj_dir_src, datasets, d_rep, slices, rep_dir):
    """
    Copy projection directory, replacing only the data_central values via
    raw text substitution — avoids yaml.dump round-trip which corrupts smefit's
    expected file format.
    """
    rep_dir = Path(rep_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    # copy all files (symlink-safe)
    for f in Path(proj_dir_src).iterdir():
        shutil.copy2(f, rep_dir / f.name)
    # patch only data_central in each active dataset using raw text replacement
    for ds in datasets:
        name     = ds["name"]
        src_path = Path(proj_dir_src) / f"{name}.yaml"
        new_vals = d_rep[slices[name]]
        src_text = src_path.read_text(encoding="utf-8")
        new_text = _replace_data_central(src_text, new_vals)
        (rep_dir / f"{name}.yaml").write_text(new_text, encoding="utf-8")


def make_ns_runcard(base_rc_path, rep_proj_dir, result_id, result_path, smeft=False):
    """
    Clone base NS runcard, redirect data_path and result_ID.
    For UV runs: theory covmat settings inherited from base runcard (aggressive).
    For SMEFT runs: override to use theory_cov_aggressive so noise and fit are consistent.
    """
    rc = yaml.safe_load(open(base_rc_path))
    rc["result_ID"]  = result_id
    rc["result_path"] = str(result_path)
    rc["data_path"]  = str(rep_proj_dir)
    if smeft:
        # BSMclosure_SMEFT runcard has use_theory_covmat=False by default;
        # override to match the noise drawn from C_aggressive
        rc["use_theory_covmat"] = True
        rc["use_t0"]            = True
        rc["datasets"] = [{**ds, "theory_cov": "aggressive"}
                          for ds in rc.get("datasets", [])]
    # reduce NS budget for speed (closure test needs breadth, not depth)
    rc["nlive"]               = rc.get("nlive", 200)
    rc["n_samples"]           = rc.get("n_samples", 5000)
    rc["target_post_unc"]     = rc.get("target_post_unc", 0.2)
    rc["target_evidence_unc"] = rc.get("target_evidence_unc", 0.2)
    return rc


def run_ns(runcard_path):
    """Run smefit NS on the given runcard. Returns exit code."""
    result = subprocess.run(
        [SMEFIT_BIN, "NS", str(runcard_path)],
        capture_output=True, text=True
    )
    return result.returncode, result.stderr


def extract_uv_posterior(fits_dir, result_id, uv_params):
    """
    Load NS posterior from fit_results.json and return
    {param: (mean, std, samples)} for each UV parameter.
    """
    path = Path(fits_dir) / result_id / "fit_results.json"
    if not path.exists():
        return None
    res  = json.load(open(path))
    out  = {}
    samples_all = res.get("samples", {})
    for p in uv_params:
        if p not in samples_all:
            continue
        s = np.array(samples_all[p])
        out[p] = {"mean": float(np.mean(s)), "std": float(np.std(s, ddof=1)),
                  "samples": s}
    return out


def needs_fold(model):
    """
    Return True if all UV operators are even in every coupling (Z2 symmetry).
    CHM has Otap ~ g (linear) → no fold needed.
    W' universal-g has all ops ~ g^2 → fold needed.
    """
    import dataclasses
    try:
        block = model.uv_coeff_block()
    except Exception:
        return False
    ops   = model.OPERATORS
    truth = model.uv_truth()
    for p in model.uv_param_names():
        g0 = truth[p]
        if g0 == 0:
            continue
        # check if c(-g) == c(+g) for all operators
        kw = dataclasses.asdict(model)
        kw_neg = {**kw, p: -g0}
        try:
            mdl_neg = type(model)(**kw_neg)
        except TypeError:
            return False
        c_pos = model.eft_coefficients()
        c_neg = mdl_neg.eft_coefficients()
        if any(abs(c_pos[op] - c_neg[op]) > 1e-10 for op in ops if op in c_pos):
            return False   # at least one operator is odd → no Z2
    return True


# ── main run loop ─────────────────────────────────────────────────────────────

def run_replicas(tag, n_rep, seed, out_dir, dry_run=False, smeft=False):
    result_root = PIPELINE / "results" / tag
    pca_dir     = result_root / "pca"
    fits_dir    = out_dir / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)

    # load PCA matrices
    Ci    = np.load(pca_dir / "C_inv.npy")
    C     = np.linalg.inv(Ci)
    L     = np.linalg.cholesky(C)

    if smeft:
        # free-EFT run: use BSMclosure_SMEFT runcard (27 free EFT operators)
        rc_candidates = [
            r for r in (result_root / "runcards").glob("*BSMclosure_SMEFT.yaml")
            if not r.name.startswith("Report_")
        ]
        if not rc_candidates:
            raise FileNotFoundError(f"No BSMclosure_SMEFT runcard found under {result_root}/runcards/")
    else:
        # find the full-model UVcoup NS runcard (exclude Report_, *_universal, *_sym, *_gWHonly)
        rc_candidates = [
            r for r in (result_root / "runcards").glob("*UVcoup.yaml")
            if not r.name.startswith("Report_")
            and "_universal" not in r.name
            and "_sym"       not in r.name
            and "_gWHonly"   not in r.name
        ]
        if not rc_candidates:
            raise FileNotFoundError(f"No UVcoup runcard found under {result_root}/runcards/")
    base_rc_path = rc_candidates[0]
    base_rc      = yaml.safe_load(open(base_rc_path))

    proj_dir_central = base_rc["data_path"]
    datasets         = base_rc["datasets"]

    d_central, slices = load_data_central(proj_dir_central, datasets)
    n_data = len(d_central)
    assert n_data == Ci.shape[0], \
        f"Data length mismatch: yaml gives {n_data}, C_inv is {Ci.shape[0]}×{Ci.shape[0]}"

    rng = np.random.default_rng(seed)

    print(f"\nGenerating and fitting {n_rep} L1 replicas for {tag}")
    print(f"  n_data={n_data}  base_rc={base_rc_path.name}")
    print(f"  Output: {out_dir}\n")

    completed = []
    for r in range(n_rep):
        rep_id   = f"{tag}_l1rep{r:03d}"
        fits_out = fits_dir / rep_id
        done_file = fits_out / "fit_results.json"

        if done_file.exists():
            print(f"  rep {r:03d}: already done, skipping")
            completed.append(r)
            continue

        # generate replica
        noise = L @ rng.standard_normal(n_data)
        d_rep = d_central + noise

        # write projection files
        rep_proj_dir = out_dir / "projections" / f"rep{r:03d}"
        write_replica_projections(proj_dir_central, datasets, d_rep,
                                  slices, rep_proj_dir)

        # write NS runcard
        rc_rep  = make_ns_runcard(base_rc_path, rep_proj_dir, rep_id,
                                  fits_dir, smeft=smeft)
        rc_path = out_dir / "runcards" / f"{rep_id}.yaml"
        rc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rc_path, "w") as fh:
            yaml.dump(rc_rep, fh, default_flow_style=False, sort_keys=False)

        if dry_run:
            print(f"  rep {r:03d}: runcard written → {rc_path}")
            continue

        print(f"  rep {r:03d}: running NS ...", end="", flush=True)
        ret, stderr = run_ns(rc_path)
        if ret != 0:
            print(f" FAILED (exit {ret})")
            print(stderr[-500:] if stderr else "")
        else:
            print(" done")
            completed.append(r)

    return completed


# ── collect and plot ──────────────────────────────────────────────────────────

def collect_pulls(tag, out_dir, model):
    fits_dir  = out_dir / "fits"
    uv_params = model.uv_param_names()
    truth     = model.uv_truth()
    fold      = needs_fold(model)

    pulls_per_param = {p: [] for p in uv_params}
    sigma_per_param = {p: [] for p in uv_params}
    n_loaded = 0

    for rep_dir in sorted(fits_dir.iterdir()):
        if not rep_dir.is_dir():
            continue
        posteriors = extract_uv_posterior(fits_dir, rep_dir.name, uv_params)
        if posteriors is None:
            continue
        n_loaded += 1
        for p in uv_params:
            if p not in posteriors:
                continue
            s     = posteriors[p]["samples"]
            g_tr  = truth[p]
            if fold and g_tr > 0:
                s = np.abs(s)
                g_tr = abs(g_tr)
            mu    = float(np.mean(s))
            sigma = float(np.std(s, ddof=1))
            if sigma > 0:
                pulls_per_param[p].append((mu - g_tr) / sigma)
                sigma_per_param[p].append(sigma)

    print(f"\nCollected {n_loaded} successful NS fits")
    return pulls_per_param, sigma_per_param


def plot_ns_pulls(pulls_per_param, sigma_per_param, model, out_dir):
    uv_params = model.uv_param_names()
    truth     = model.uv_truth()
    n_params  = len(uv_params)

    ncols = 3
    nrows = (n_params + ncols - 1) // ncols
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    # flatten and hide any unused axes
    axes_flat = np.array(axes_grid).flatten()
    for ax in axes_flat[n_params:]:
        ax.set_visible(False)
    axes = axes_flat[:n_params]

    for ax, p in zip(axes, uv_params):
        pulls = np.array(pulls_per_param[p])
        n = len(pulls)
        if n == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        bins = np.linspace(-4, 4, min(20, max(8, n // 3)))
        ax.hist(pulls, bins=bins, density=True, alpha=0.7,
                color="#2196F3", edgecolor="white",
                label=fr"NS L1 pulls  ($n={n}$)")
        x = np.linspace(-4, 4, 300)
        ax.plot(x, norm_dist.pdf(x), "r-", lw=2.5,
                label=r"$\mathcal{N}(0,1)$  [target]")

        mu  = float(pulls.mean())
        sig = float(pulls.std(ddof=1))
        ax.axvline(mu, color="#2196F3", lw=1.5, ls="--")

        # μ/σ summary — upper right
        ax.text(0.97, 0.95,
                fr"$\mu = {mu:+.3f}$" + "\n" + fr"$\sigma = {sig:.3f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

        # legend — upper left, pinned explicitly
        ax.legend(fontsize=10, loc="upper left")

        # truth / sigma_NS box — below the legend
        sigma_vals = np.array(sigma_per_param[p])
        ax.text(0.03, 0.72,
                fr"truth $= {truth[p]:.3f}$" + "\n"
                fr"$\sigma_{{NS}}$ mean $= {sigma_vals.mean():.4f}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

        ax.set_xlim(-4.5, 4.5)
        ax.set_xlabel(fr"Pull  $P = (g_{{fit}} - g_{{truth}}) / \sigma_g$",
                      fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(fr"NS L1 closure — UV parameter $\theta = $ {p}", fontsize=12)

    model_label = repr(model).split("(")[0]
    fig.suptitle(fr"NS-based L1 pull distribution — {model_label}  "
                 fr"(closed CT $\Rightarrow$ $\mu\approx0$, $\sigma\approx1$)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = out_dir / "ns_l1_pulls"
    plt.savefig(f"{out}.png", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}.png")


def plot_ns_pulls_pooled(pulls_per_param, model, out_dir):
    """Single histogram of all UV parameter pulls pooled together."""
    all_pulls = np.concatenate([np.array(v) for v in pulls_per_param.values() if len(v) > 0])
    n = len(all_pulls)
    if n == 0:
        return

    mu  = float(all_pulls.mean())
    sig = float(all_pulls.std(ddof=1))
    n_params = len([v for v in pulls_per_param.values() if len(v) > 0])

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(-4, 4, min(30, max(10, n // 5)))
    ax.hist(all_pulls, bins=bins, density=True, alpha=0.7,
            color="#2196F3", edgecolor="white",
            label=fr"Pooled UV pulls  ($n={n}$,  {n_params} params × {n // n_params} reps)")

    x = np.linspace(-4.5, 4.5, 300)
    ax.plot(x, norm_dist.pdf(x), "r-", lw=2.5, label=r"$\mathcal{N}(0,1)$  [target]")
    ax.axvline(mu, color="#2196F3", lw=1.5, ls="--")

    frac2 = float((np.abs(all_pulls) > 2).mean()) * 100
    frac1 = float((np.abs(all_pulls) > 1).mean()) * 100
    ax.text(0.97, 0.95,
            fr"$\mu = {mu:+.3f}$" + "\n"
            fr"$\sigma = {sig:.3f}$" + "\n"
            fr"$|P|>2\sigma$: {frac2:.1f}% (exp. 5%)" + "\n"
            fr"$|P|>1\sigma$: {frac1:.1f}% (exp. 32%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

    ax.set_xlim(-4.5, 4.5)
    ax.set_xlabel(r"Pull  $P = (\theta_\mathrm{fit} - \theta_\mathrm{truth}) / \sigma_\theta$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    model_label = repr(model).split("(")[0]
    ax.set_title(fr"NS L1 closure — pooled UV pulls — {model_label}", fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = out_dir / "ns_l1_pulls_pooled"
    plt.savefig(f"{out}.png", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}.png")


def collect_eft_pulls(out_dir, model):
    """
    Collect per-replica EFT operator pulls from the same NS fit_results.json files.
    Truth values come from model.eft_coefficients(). Operators with zero truth
    and zero posterior std are skipped.
    Returns {op: array_of_pulls}.
    """
    fits_dir   = out_dir / "fits"
    truth_eft  = model.eft_coefficients()
    ops        = [op for op, v in truth_eft.items()]
    pulls_dict = {op: [] for op in ops}
    n_loaded   = 0

    for rep_dir in sorted(fits_dir.iterdir()):
        if not rep_dir.is_dir():
            continue
        path = rep_dir / "fit_results.json"
        if not path.exists():
            continue
        samples = json.load(open(path)).get("samples", {})
        n_loaded += 1
        for op in ops:
            if op not in samples:
                continue
            s     = np.array(samples[op])
            sigma = float(np.std(s, ddof=1))
            if sigma == 0:
                continue
            pull = (float(np.mean(s)) - truth_eft[op]) / sigma
            pulls_dict[op].append(pull)

    print(f"EFT pulls collected from {n_loaded} fits")
    return {op: np.array(v) for op, v in pulls_dict.items() if len(v) > 0}


def plot_eft_pulls(eft_pulls, model, out_dir, smeft=False):
    """
    Two plots:
      1. Histogram of all EFT pulls pooled together vs N(0,1)
      2. Per-operator bar chart of mean pull ± std
    """
    truth_eft = model.eft_coefficients()
    ops       = list(eft_pulls.keys())
    if not ops:
        print("No EFT pull data to plot.")
        return

    n_reps = max(len(eft_pulls[op]) for op in ops)
    # ── 1. pooled histogram ───────────────────────────────────────────────────
    all_pulls = np.concatenate([eft_pulls[op] for op in ops])
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(-4, 4, 25)
    legend_label = "EFT pulls"
    ax.hist(all_pulls, bins=bins, density=True, alpha=0.7,
            color="#4CAF50", edgecolor="white",
            label=legend_label)
    x = np.linspace(-4, 4, 300)
    ax.plot(x, norm_dist.pdf(x), "r-", lw=2.5, label=r"$\mathcal{N}(0,1)$  [target]")
    mu  = float(all_pulls.mean())
    sig = float(all_pulls.std(ddof=1))
    ax.axvline(mu, color="#4CAF50", lw=1.5, ls="--")
    ax.text(0.97, 0.95,
            fr"$\mu = {mu:+.3f}$" + "\n" + fr"$\sigma = {sig:.3f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))
    ax.set_xlim(-4.5, 4.5)
    ax.set_xlabel(r"Pull  $P = (c_{\rm fit} - c_{\rm truth}) / \sigma_c$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("NS L1 closure — EFT operators (pooled)", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    out = out_dir / "ns_l1_eft_pulls"
    plt.savefig(f"{out}.png", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}.png")

    # ── 2. per-operator bar chart ─────────────────────────────────────────────
    means = np.array([eft_pulls[op].mean() for op in ops])
    stds  = np.array([eft_pulls[op].std(ddof=1) for op in ops])
    fig, ax = plt.subplots(figsize=(max(10, len(ops) * 0.5), 5))
    xpos = np.arange(len(ops))
    ax.bar(xpos, means, yerr=stds, color="#4CAF50", alpha=0.7,
           ecolor="black", capsize=3, width=0.7)
    ax.axhline(0,  color="black", lw=0.8)
    ax.axhline(+1, color="red",   lw=1.0, ls="--", alpha=0.6)
    ax.axhline(-1, color="red",   lw=1.0, ls="--", alpha=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(ops, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"Mean pull $\pm$ std", fontsize=11)
    ax.set_title("NS L1 closure — EFT operator pulls per coefficient", fontsize=11)
    plt.tight_layout()
    out2 = out_dir / "ns_l1_eft_pull_bars"
    plt.savefig(f"{out2}.png", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out2}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}.png")


def print_summary(pulls_per_param, model):
    truth = model.uv_truth()
    print("\n" + "=" * 55)
    print(f"{'param':<15} {'n':>5} {'mean pull':>10} {'std pull':>10}")
    print("-" * 55)
    for p, pulls in pulls_per_param.items():
        if not pulls:
            print(f"{p:<15} {'—':>5}")
            continue
        arr = np.array(pulls)
        print(f"{p:<15} {len(arr):>5} {arr.mean():>+10.3f} {arr.std(ddof=1):>10.3f}")
    print("=" * 55)
    print("Expected for genuine closure: mean≈0, std≈1")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag",   required=True,
                   help="Result tag, e.g. comphiggs_grho200_mrho1000")
    p.add_argument("--nrep",  type=int, default=50)
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--dry-run",      action="store_true",
                   help="Write runcards but do not run NS")
    p.add_argument("--collect-only", action="store_true",
                   help="Skip generation/fitting; just collect existing results")
    p.add_argument("--smeft",        action="store_true",
                   help="Free-EFT closure: use BSMclosure_SMEFT runcard (27 free operators), "
                        "gives 27x50=1350 independent EFT pulls")
    args = p.parse_args()

    suffix  = "_l1closure_smeft" if args.smeft else "_l1closure"
    out_dir = PIPELINE / "results" / f"{args.tag}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.tag)
    print(f"Model : {model!r}")
    print(f"UV params : {model.uv_param_names()}")
    print(f"UV truth  : {model.uv_truth()}")
    if not args.smeft:
        print(f"Z2 fold   : {needs_fold(model)}")

    if not args.collect_only:
        run_replicas(args.tag, args.nrep, args.seed, out_dir,
                     dry_run=args.dry_run, smeft=args.smeft)

    if not args.dry_run:
        if not args.smeft:
            pulls, sigmas = collect_pulls(args.tag, out_dir, model)
            print_summary(pulls, model)
            plot_ns_pulls(pulls, sigmas, model, out_dir)
            plot_ns_pulls_pooled(pulls, model, out_dir)

        eft_pulls = collect_eft_pulls(out_dir, model)
        plot_eft_pulls(eft_pulls, model, out_dir, smeft=args.smeft)

        print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
