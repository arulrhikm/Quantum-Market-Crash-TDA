# Reproduce Key Paper Outputs

## Core verification
- `python scripts/verify_paper_claims.py`
- Output: `results/verification_results.json`

## Figure generation
- `python scripts/generate_figures.py`
- Outputs: `images/figure_*.png`

## Finishing experiments (E1, E1b, E6, E7)
- `python scripts/finish_updated_experiments.py`
- Outputs:
  - `results/classification_chronological.json`
  - `results/classification_ood.json`
  - `results/hardware_estimate.json`
  - `results/hyperparameters.json`
  - `images/figure_06_classification_roc_split.png`

## Kappa comparison close-out (E4)
- `python scripts/run_kappa_comparison.py`
- Output: `results/kappa_comparison.json`

## Deflation failure analysis close-out
- `python scripts/analyze_deflation_failure.py`
- Output: `results/deflation_failure_analysis.json`

## Barren plateau extension (E5)
- `python scripts/run_barren_plateau_extended.py`
- Output: `results/barren_plateau_extended.json`
