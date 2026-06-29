# SMEFiT Thesis

Analysis code and results for the MSc thesis *Discovering New Physics at the FCC-ee with the SMEFT* (Wopke Telman, University of Amsterdam).

## Repository structure

| Directory | Description |
|-----------|-------------|
| `models/` | W' and Z' UV model definitions |
| `scripts/` | Analysis scripts |
| `runcards/` | smefit runcards per benchmark |
| `figures/` | All thesis figures |

---

## Scripts

| Script | Description |
|--------|-------------|
| `run_pipeline.py` | Master pipeline: pseudo-data → fits → significance → report |
| `fisher_uv.py` | Corrected UV Fisher information matrix |
| `ns_l1_closure.py` | L1 closure test with nested sampling |
| `fingerprint_comparison.py` | W' vs Z' signal fingerprint comparison |
| `plot_scan_l1_from_table.py` | Theory-covariance comparison scan plot |
| `plot_zprime_discovery_extended.py` | Extended Z' discovery reach plot |
| `theory_vs_data.py` | Data vs theory comparison plot |
| `zdata_comparison.py` | Z data comparison plot |

---

## Runcards

| Directory | Benchmark |
|-----------|-----------|
| `runcards/wprime_gwh050_mwp050/` | W', g=0.50, m=5 TeV (primary) |
| `runcards/wprime_gwh050_mwp010/` | W', g=0.50, m=1 TeV |
| `runcards/wprime_constrained_v3_gwh050_mwp050/` | W' 14-op constrained, g=0.50, m=5 TeV |
| `runcards/zprime_gzh050_mzp050/` | Z', g=0.50, m=5 TeV |
| `runcards/robustness_wprime_data_zprime_fit_benchmark/` | Cross-injection test (W' data, Z' fit) |

---

## Figures

- `eft_closure_histograms.png`
- `eft_post_14.png`
- `uv_closure_posteriors.png`
- `pull_eft14.png`
- `Pull_Uv.png`
- `fisher_uv_wprime.png`
- `discovery_reach_wprime.png`
- `discovery_reach_theorycov.png`
- `data_vs_theory.png`
- `fingerprint_wprime_zprime.png`
- `Zprime_gZH050.png`
- `robustness_wprime_zprime.png`
- `discovery_reach_zprime.png`

---

## Quick start

```bash
conda activate smefit-dev
cd scripts/
python run_pipeline.py --model wprime --gWH 0.50 --mWp 5.0
```
