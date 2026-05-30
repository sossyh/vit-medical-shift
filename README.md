
# ViT Reliability and Explainability Under Medical Distribution Shift

Thesis research project investigating how Vision Transformers behave 
under hospital-level distribution shift in chest X-ray classification.

## Datasets
- Training: NIH ChestX-ray14
- OOD Test: CheXpert

## Structure
- `notebooks/` — Colab notebooks (one per experiment week)
- `src/` — reusable Python modules
- `configs/` — YAML hyperparameter configs
- `results/` — figures and metrics (checkpoints excluded)

## Setup
```bash
pip install -r requirements.txt
```




# Research Proposal

## Title
Reliability and Explainability of Vision Transformers Under
Medical Distribution Shift

## Author
Sosna Worku

---

## 1. Problem Statement

Deep learning models for medical image analysis frequently
fail when deployed in clinical settings different from where
they were trained. This phenomenon — known as distribution
shift — occurs because different hospitals use different
scanners, imaging protocols, and patient populations. A model
trained at one institution may achieve high accuracy internally
but degrade significantly when tested externally.

Vision Transformers (ViTs) have recently emerged as a
promising alternative to Convolutional Neural Networks (CNNs)
for medical imaging tasks. However, their reliability and
explainability under distribution shift remains poorly
understood. This research addresses that gap.

---

## 2. Research Questions

1. How does ViT performance degrade under hospital-level
   distribution shift compared to a ResNet baseline?

2. Does uncertainty estimation (MC-Dropout) reliably flag
   cases where the model fails under distribution shift?

3. Do ViT attention maps remain clinically meaningful when
   the model is tested on out-of-distribution data?

---

## 3. Methodology

### Datasets
- Training: NIH ChestX-ray14 (94,875 chest X-rays, 14 pathology labels,
  collected at NIH Clinical Center)
- OOD Test: CheXpert (224,316 chest X-rays collected at
  Stanford Hospital) — different scanner, protocol, population

### Models
- ViT-B/16: Vision Transformer with 16x16 patch size,
  pretrained on ImageNet-21k, fine-tuned on NIH
- ResNet50: CNN baseline, pretrained on ImageNet,
  fine-tuned on NIH

### Experiments
1. Baseline performance (in-distribution)
   - Train both models on NIH
   - Evaluate AUC, ECE on NIH validation set

2. Distribution shift analysis
   - Evaluate both models on CheXpert (OOD)
   - Measure AUC drop, calibration degradation
   - Compare MC-Dropout uncertainty on ID vs OOD

3. Explainability analysis
   - Extract ViT attention rollout maps
   - Compute GradCAM for ResNet
   - Measure IoU with NIH ground truth bounding boxes
   - Compare attention quality ID vs OOD

---

## 4. Preliminary Results

Early experiments on a 27,140-image subset show:

| Model    | Mean AUC (ID) | ECE    |
|----------|---------------|--------|
| ViT-B/16 | 0.765         | 0.010  |
| ResNet50 | 0.679         | 0.001  |

ViT outperforms ResNet on 12 of 14 pathologies, with
particularly strong performance on structural findings
(Cardiomegaly: +0.257, Fibrosis: +0.292). ResNet shows
slightly better calibration on in-distribution data.

These early results suggest ViT learns more generalizable
representations, which may translate to better robustness
under distribution shift — a hypothesis to be tested in
the next phase of experiments.

---

## 5. Expected Contributions

1. Empirical comparison of ViT vs ResNet reliability under
   real-world medical distribution shift

2. Analysis of uncertainty estimation quality under shift —
   does the model know when it doesn't know?

3. Qualitative and quantitative evaluation of attention map
   faithfulness under distribution shift

4. Reproducible codebase and trained models released publicly
   at github.com/sossyh/vit-medical-shift

---

## 6. Timeline

| Week | Task                                    |
|------|-----------------------------------------|
| 1    | Data preparation and exploration        |
| 2    | Baseline training (ViT + ResNet)        |
| 3    | Distribution shift experiments          |
| 4    | Explainability analysis + thesis writing|

---

## 7. Tools and Resources

- Compute: Google Colab Pro (NVIDIA T4 GPU)
- Framework: PyTorch, timm, HuggingFace
- Datasets: NIH ChestX-ray14, CheXpert (both public)
- Code: github.com/sossyh/vit-medical-shift
