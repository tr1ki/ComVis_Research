# Team and completed tasks

## Balakarev Sergey — Project Coordination & Experimental Design

Study design, experiment coordination, and the shared comparison protocol.

- Planned the study stages: the main comparison, additional models, transfer across corruption families, and augmentation sensitivity experiments.
- Defined a shared comparison protocol for four models, with consistent fine-tuning settings, three seeds, and the same CIFAR-10-C evaluation.
- Coordinated the training and evaluation workflow across `scripts/cv_robustness.py` and `scripts/cv_robustness_v2.py`.

## Sultan Abayev — Model Training & Data Processing

Model fine-tuning, experiment execution, and data preparation.

- Implemented CIFAR-10 and CIFAR-10-C loading, array access, batch preparation, image resizing, and normalization.
- Implemented noise, blur, and digital augmentations, including variations in augmentation probability and intensity.
- Implemented the fine-tuning loop, optimizer, learning-rate schedule, and CUDA mixed precision.
- Computed accuracy and relative accuracy drop, and saved each run's results to JSON.
- Set up experiment queues and job distribution across GPUs in the `run_A`–`run_D` notebooks.

## Erasyl Umirbay — Evaluation, Statistical Analysis & Visualization

Model evaluation, result validation, statistical analysis, and visualization.

- Prepared notebook layouts, run instructions, summary reports, and result visualizations.
- Aggregated results across seeds, compared models, and computed confidence intervals and Welch tests in `scripts/analyze.py` and `scripts/analyze_v2.py`.
- Prepared figures for corruption types and severity, transfer across families, augmentation sensitivity, and the relationship between clean accuracy and robustness.
- Checked JSON structure and consistency between computed metrics and summary tables, and used `QUICK_TEST` for short execution checks.

The results are stored in `results/` and `analysis/`: 72 runs covering 24 model/condition combinations, with three seeds each. Reproduction instructions are available in the [README](../README.md).
