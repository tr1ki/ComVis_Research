# CNN vs ViT: corruption robustness on CIFAR-10-C

We study how data augmentation during fine-tuning affects the robustness of CNNs and Vision Transformers. The experiments compare ResNet-50, ConvNeXt-T, DeiT-S, and Swin-T with ImageNet-1k pretrained weights. Models are trained on CIFAR-10 and evaluated on clean images and CIFAR-10-C.

The repository contains 72 results: 24 combinations of model and training condition, with three seeds each. Datasets and fine-tuned weights are not included.

English is the working language for documentation, notebooks, code comments, and reports.

## Team

| Member | Role |
|---|---|
| **Balakarev Sergey** | Project Coordination & Experimental Design |
| **Sultan Abayev** | Model Training & Data Processing |
| **Erasyl Umirbay** | Evaluation, Statistical Analysis & Visualization |

### Completed tasks

**Balakarev Sergey** handled study design, experiment planning and coordination, and the shared model comparison protocol.

**Sultan Abayev** handled model fine-tuning, dataset loading and preparation, augmentation, batch processing, JSON result storage, and GPU job queues.

**Erasyl Umirbay** handled notebook and result presentation, metric and table checks, statistical analysis, and figures.

See [docs/TEAM.md](docs/TEAM.md) for the detailed task breakdown.

## Project structure

```text
.
├── README.md
├── docs/
│   └── TEAM.md                     # roles and completed tasks
├── notebooks/
│   ├── round1_single_run.ipynb     # ResNet-50 and DeiT-S: baseline, noise+blur, AugMix
│   ├── run_A_new_models.ipynb      # ConvNeXt-T and Swin-T
│   ├── run_B_transfer.ipynb        # transfer across corruption families
│   ├── run_C_probability.ipynb     # augmentation probability
│   └── run_D_intensity.ipynb       # augmentation intensity
├── scripts/
│   ├── cv_robustness.py            # first round of experiments
│   ├── cv_robustness_v2.py         # four models and additional conditions
│   ├── analyze.py                 # main tables and figures
│   └── analyze_v2.py              # four-model comparison and extended analysis
├── results/
│   └── *.json                     # 72 files: model, condition, and seed
└── analysis/
    ├── results_summary.md         # main results
    ├── results_summary_v2.md      # extended analysis
    ├── table_main.csv
    └── fig_*.png                  # seven figures
```

## Experiments

The models come from `timm`: `resnet50.a1_in1k`, `convnext_tiny.fb_in1k`, `deit_small_patch16_224.fb_in1k`, and `swin_tiny_patch4_window7_224.ms_in1k`.

All runs use AdamW, a learning rate of `1e-4`, weight decay of `0.05`, batch size `128`, and 6 epochs, with warm-up over the first 5% of steps followed by a cosine schedule. Label smoothing is `0.1`, and AMP is enabled on CUDA. Images are resized from 32×32 to 224×224 using bilinear interpolation. Seeds are `0`, `1`, and `2`.

ResNet-50 and DeiT-S are evaluated with baseline augmentation (crop + flip), noise+blur (`corrupt_aug`), AugMix, individual noise, blur, and digital families, and variations in noise+blur probability and intensity. ConvNeXt-T and Swin-T have results for baseline and `corrupt_aug`. AugMix uses a restricted set of operations without JSD loss.

Each run is evaluated on 15 CIFAR-10-C corruptions at five severity levels. The main metric is relative accuracy drop: `(clean accuracy − corrupted accuracy) / clean accuracy`. The seen and unseen groups refer to noise+blur and weather+digital families relative to the `corrupt_aug` condition. Test images are never used for training.

## Running the project

The notebooks are designed for Kaggle with internet access enabled. The `run_A`–`run_D` experiments used two NVIDIA Tesla T4 GPUs, with one job per GPU.

Attach the [CIFAR-10 Python](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python) and [CIFAR10-C](https://www.kaggle.com/datasets/faigarvn/cifar10-c) datasets. If they are not attached, the code downloads them automatically. Set `QUICK_TEST = True` for a short check, then switch it back to `False` and use Save & Run All for the full experiment. Results are saved to `results/*.json`; the `run_A`–`run_D` notebooks also create a ZIP archive.

To regenerate the reports locally from the saved results:

```bash
python -m pip install numpy scipy matplotlib
python -X utf8 scripts/analyze.py results analysis
python -X utf8 scripts/analyze_v2.py results analysis
```

Training requires PyTorch, torchvision, and timm. Before running v2 locally, set `CV_DATA_DIR` to the dataset storage directory; select the model and condition through `CV_MODEL` and `CV_CONDITION`. Library versions are not pinned yet. Repeating a run with the same model, condition, and seed overwrites the corresponding JSON file.

## Data and sources

CIFAR-10 is credited to Krizhevsky (2009), and CIFAR-10-C to Hendrycks and Dietterich (ICLR 2019). The datasets are not redistributed with the code. CIFAR-10-C is downloaded from [Zenodo](https://zenodo.org/records/2535967), and pretrained weights are downloaded through `timm` from the Hugging Face Hub.
