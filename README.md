# qTDA (PCE-VQE) Reproducibility Repository

Code and artifacts for the manuscript `paper.tex`:
**Depth-Efficient Quantum Topological Data Analysis for Financial Crash Early Warning**.

## Project Layout

- `paper.tex` - manuscript source (primary paper file)
- `src/` - core qTDA/PCE implementation modules
- `results/` - generated numeric outputs used by manuscript tables/claims
- `images/` - generated figures used by manuscript
- `notebooks/` - analysis notebooks used for figure/result support
- `scripts/` - runnable experiment/verification entry scripts
- `data/` - project datasets (including S&P 500 CSV)
- `docs/` - paper-adjacent project notes/experiment plans
- `scripts/verify_paper_claims.py` - CE vs ripser verification + timing/statistics
- `scripts/finish_updated_experiments.py` - chronological/OOD/hardware estimate runs
- `scripts/run_kappa_comparison.py` - kappa=2 vs kappa=3 fixed-budget comparison
- `scripts/analyze_deflation_failure.py` - beta1=4 deflation sensitivity sweeps
- `scripts/run_barren_plateau_extended.py` - n=14,16 barren-plateau extension
- `REPRODUCE.md` - concise command checklist
- `archived/` - legacy/non-essential files moved out of the active workflow

## Quick Reproduction

Run from repository root:

```bash
python scripts/verify_paper_claims.py
python scripts/finish_updated_experiments.py
python scripts/run_kappa_comparison.py
python scripts/analyze_deflation_failure.py
python scripts/run_barren_plateau_extended.py
python scripts/generate_figures.py
```

Primary outputs:
- `results/verification_results.json`
- `results/classification_chronological.json`
- `results/classification_ood.json`
- `results/kappa_comparison.json`
- `results/deflation_failure_analysis.json`
- `results/barren_plateau_extended.json`
- `images/figure_*.png`

## Notes on Scope

- The repository is organized around the current paper workflow and artifacts.
- Legacy drafts, temporary notebooks, and cache-like files are moved under `archived/` to keep the active tree review-ready.
- If you need older exploratory files, check `archived/` first.
