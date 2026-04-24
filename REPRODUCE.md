# Reproduce Key Paper Outputs

## Core verification
- `python verify_paper_claims.py`
- Output: `3_Results/verification_results.json`

## Figure generation
- `python generate_figures.py`
- Outputs: `Figures/fig_*.png`

## Finishing experiments (E1, E1b, E6, E7)
- `python finish_updated_experiments.py`
- Outputs:
  - `3_Results/classification_chronological.json`
  - `3_Results/classification_ood.json`
  - `3_Results/hardware_estimate.json`
  - `3_Results/hyperparameters.json`
  - `Figures/fig_roc_classification_split.png`
