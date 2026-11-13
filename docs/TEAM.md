# Team and completed tasks

## Balakarev Sergey — Project Management & Core Architecture

Experiment coordination and the overall architecture of the research code.

- Planned the study stages: the main comparison, additional models, transfer across corruption families, and augmentation sensitivity experiments.
- Defined a shared comparison protocol for four models, with consistent fine-tuning settings, three seeds, and the same CIFAR-10-C evaluation.
- Organized model construction, training, and evaluation in `scripts/cv_robustness.py` and `scripts/cv_robustness_v2.py`.

## Sultan Abayev — Backend Development & Data Processing

Experiment execution and data preparation.

- Implemented CIFAR-10 and CIFAR-10-C loading, array access, batch preparation, image resizing, and normalization.
- Implemented noise, blur, and digital augmentations, including variations in augmentation probability and intensity.
- Computed accuracy and relative accuracy drop, and saved each run's results to JSON.
- Set up experiment queues and job distribution across GPUs in the `run_A`–`run_D` notebooks.

## Erasyl Umirbay — Frontend, Testing & Analytics

Experiment presentation, result validation, and analysis. Frontend work in this project covers the notebooks, tables, and figures used to explore the results.

- Prepared notebook layouts, run instructions, summary reports, and result visualizations.
- Aggregated results across seeds, compared models, and computed confidence intervals and Welch tests in `scripts/analyze.py` and `scripts/analyze_v2.py`.
- Prepared figures for corruption types and severity, transfer across families, augmentation sensitivity, and the relationship between clean accuracy and robustness.
- Checked JSON structure and consistency between computed metrics and summary tables, and used `QUICK_TEST` for short execution checks.

The results are stored in `results/` and `analysis/`: 72 runs covering 24 model/condition combinations, with three seeds each. Reproduction instructions are available in the [README](../README.md).
