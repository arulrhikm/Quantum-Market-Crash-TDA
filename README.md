# qTDA (PCE-VQE) Reproducibility Repository

Code and artifacts for the manuscript `paper.tex`:
**Depth-Efficient Quantum Topological Data Analysis for Financial Crash Early Warning**.

## Project Layout

- `paper.tex` - manuscript source (primary paper file)
- `1_Code/` - core qTDA/PCE implementation modules
- `3_Results/` - generated numeric outputs used by manuscript tables/claims
- `Figures/` - generated figures used by manuscript
- `Notebooks/` - analysis notebooks used for figure/result support
- `verify_paper_claims.py` - CE vs ripser verification + timing/statistics
- `finish_updated_experiments.py` - chronological/OOD/hardware estimate runs
- `run_kappa_comparison.py` - kappa=2 vs kappa=3 fixed-budget comparison
- `analyze_deflation_failure.py` - beta1=4 deflation sensitivity sweeps
- `run_barren_plateau_extended.py` - n=14,16 barren-plateau extension
- `REPRODUCE.md` - concise command checklist
- `archived/` - legacy/non-essential files moved out of the active workflow

## Quick Reproduction

Run from repository root:

```bash
python verify_paper_claims.py
python finish_updated_experiments.py
python run_kappa_comparison.py
python analyze_deflation_failure.py
python run_barren_plateau_extended.py
python generate_figures.py
```

Primary outputs:
- `3_Results/verification_results.json`
- `3_Results/classification_chronological.json`
- `3_Results/classification_ood.json`
- `3_Results/kappa_comparison.json`
- `3_Results/deflation_failure_analysis.json`
- `3_Results/barren_plateau_extended.json`
- `Figures/fig_*.png`

## Notes on Scope

- The repository is organized around the current paper workflow and artifacts.
- Legacy drafts, temporary notebooks, and cache-like files are moved under `archived/` to keep the active tree review-ready.
- If you need older exploratory files, check `archived/` first.
