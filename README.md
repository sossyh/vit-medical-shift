# ViT Reliability and Explainability Under Medical Distribution Shift

**Author:** Sosna Worku Achamyeleh | George Washington University  
**Thesis Chapter** | 2026

---

## Overview

This project investigates how Vision Transformers (ViT) behave under hospital-level distribution shift in chest X-ray classification, comparing reliability and explainability against a ResNet50 baseline.

**Training data:** NIH ChestX-ray14 (94,875 images, 14 pathology labels)  
**OOD test data:** CheXpert validation set (202 frontal chest X-rays, Stanford Hospital)

---

## Key Findings

| Model | ID AUC (NIH) | OOD AUC (CheXpert) | AUC Drop | ID ECE | OOD ECE |
|-------|-------------|-------------------|----------|--------|---------|
| ViT-B/16 | **0.800** | **0.774** | 0.027 | 0.006 | 0.040 |
| ResNet50 | 0.745 | 0.729 | **0.016** | 0.003 | 0.035 |

**RQ1:** ViT achieves higher accuracy but degrades more under distribution shift than ResNet.  
**RQ2:** MC-Dropout uncertainty fails to reliably detect OOD samples. ECE degrades 6-13x under shift.  
**RQ3:** ViT attention maps are diffuse and clinically unfocused. ResNet GradCAM produces more spatially meaningful activations.

---

## Project Structure

```
vit-medical-shift/
├── src/                        ← reusable Python modules
│   ├── utils.py                ← helpers, NIH_LABELS
│   ├── dataset.py              ← NIHChestDataset, CheXpertDataset
│   ├── model.py                ← get_vit, get_resnet, mc_dropout_predict
│   ├── train.py                ← training loop, checkpointing
│   ├── evaluate.py             ← AUC, ECE, reliability diagrams
│   ├── explainability.py       ← attention rollout, GradCAM, IoU
│   └── download_data.py        ← Kaggle download helpers
├── notebooks/                  ← Colab notebooks (one per experiment)
│   ├── 00_setup_and_download.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_distribution_shift.ipynb
│   └── 04_explainability.ipynb
├── configs/                    ← YAML hyperparameter configs
│   ├── vit_nih.yaml
│   └── resnet_nih.yaml
├── results/
│   ├── figures/                ← all experiment figures
│   └── metrics/                ← CSV and JSON result files
├── requirements.txt
└── .gitignore
```

---

## Datasets

| Dataset | Source | Use |
|---------|--------|-----|
| NIH ChestX-ray14 | [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) | Training + ID evaluation |
| CheXpert | [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/) | OOD evaluation |

Data is not included in this repo. Download instructions are in `notebooks/00_setup_and_download.ipynb`.

---

## Setup

**Requirements:** Python 3.10+, Google Colab Pro (T4 GPU recommended)

```bash
git clone https://github.com/sossyh/vit-medical-shift.git
cd vit-medical-shift
pip install -r requirements.txt
```

**To reproduce results:**
1. Run `notebooks/00_setup_and_download.ipynb` once to download data to Google Drive
2. Run notebooks 01-04 in order

Each notebook starts with:
```python
!git clone https://github.com/sossyh/vit-medical-shift.git
import sys
sys.path.insert(0, '/content/vit-medical-shift')
```

---

## Results

### Baseline Performance (NIH ChestX-ray14)

| Disease | ViT AUC | ResNet AUC |
|---------|---------|------------|
| Cardiomegaly | 0.869 | 0.728 |
| Effusion | 0.855 | 0.834 |
| Emphysema | 0.910 | 0.834 |
| Pneumothorax | 0.847 | 0.773 |
| **Mean** | **0.800** | **0.745** |

### Distribution Shift (NIH → CheXpert)

The most affected condition under shift is Pneumothorax:
- ViT: 0.847 → 0.620 (drop -0.227)
- ResNet: 0.773 → 0.665 (drop -0.107)

### Explainability

ViT attention maps are diffuse across the full image, while ResNet GradCAM produces spatially focused activations aligned with pathology regions for conditions like Atelectasis and Effusion.

---

## Models

| Model | Parameters | Pretrained | Fine-tuned |
|-------|-----------|------------|------------|
| ViT-B/16 | 85.8M | ImageNet-21k | NIH ChestX-ray14 |
| ResNet50 | 23.5M | ImageNet | NIH ChestX-ray14 |

---

## Citation

```
@misc{worku2026vit,
  author    = {Sosna Worku Achamyeleh},
  title     = {Reliability and Explainability of Vision Transformers
               Under Medical Distribution Shift},
  year      = {2026},
  school    = {George Washington University},
  note      = {https://github.com/sossyh/vit-medical-shift}
}
```
