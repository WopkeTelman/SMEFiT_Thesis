# SMEFiT Thesis

Analysis code for the MSc thesis *Discovering New Physics at the FCC-ee with the SMEFT* (Wopke Telman, University of Amsterdam).

## Contents

| Directory | Description |
|-----------|-------------|
| `models/` | W' and Z' UV model definitions (matching relations, operator lists) |
| `scripts/` | Core analysis scripts |
| `runcards/` | smefit runcards for the W' benchmark at g=0.50, m=5 TeV |

## Scripts

| Script | Description |
|--------|-------------|
| `run_pipeline.py` | Master pipeline: pseudo-data → fits → significance → report |
| `fisher_uv.py` | Corrected UV Fisher information matrix |
| `ns_l1_closure.py` | L1 closure test with nested sampling |
| `fingerprint_comparison.py` | W' vs Z' signal fingerprint comparison |

## Usage

```bash
conda activate smefit-dev
cd scripts/
python run_pipeline.py --model wprime --gWH 0.50 --mWp 5.0
```
