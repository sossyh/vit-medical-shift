
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