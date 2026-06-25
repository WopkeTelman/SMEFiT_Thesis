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

| File | Script | Description |
|------|--------|-------------|
| `eft_closure_histograms.png` | `smefit R` | EFT posterior histograms — W' g=0.50, m=1 TeV |
| `eft_post_14.png` | `smefit R` | EFT posteriors — 14-op constrained W', m=5 TeV |
| `uv_closure_posteriors.png` | `smefit R` | UV coupling posteriors — W' g=0.50, m=1 TeV |
| `pull_eft14.png` | `ns_l1_closure.py --smeft` | L1 pull distribution — 14-op free EFT |
| `Pull_Uv.png` | `ns_l1_closure.py` | L1 pull distribution — UV coupling |
| `fisher_uv_wprime.png` | `fisher_uv.py` | UV Fisher matrix by √s tier — W' g=0.50, m=5 TeV |
| `discovery_reach_wprime.png` | `run_pipeline.py --scan` | W' discovery reach in (g, m) plane |
| `discovery_reach_theorycov.png` | `plot_scan_l1_from_table.py` | ⚠️ Not yet generated — TC aggressive vs no TC |
| `data_vs_theory.png` | `smefit R` | ⚠️ Not yet generated — boundary point g=1.28, m=9.3 TeV |
| `fingerprint_wprime_zprime.png` | `fingerprint_comparison.py` | W' vs Z' signal fingerprint |
| `Zprime_gZH050.png` | `smefit R` | Z' EFT posteriors — g=0.50, m=5 TeV |
| `robustness_wprime_zprime.png` | `smefit R` | Cross-injection: W' data fit with Z' model |
| `discovery_reach_zprime.png` | `plot_zprime_discovery_extended.py` | Z' discovery reach |

---

## Quick start

```bash
conda activate smefit-dev
cd scripts/
python run_pipeline.py --model wprime --gWH 0.50 --mWp 5.0
```
