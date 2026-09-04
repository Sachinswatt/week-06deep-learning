# week-06deep-learning

# Integrative Capstone Project — Week 6

A complete data science pipeline combining supervised and unsupervised
learning: predicting breast cancer diagnosis (malignant vs. benign) and
independently testing whether the same structure is discoverable via
unsupervised clustering.

## Problem

Binary classification of tumors as malignant or benign from 30 numeric
features derived from digitized biopsy images, using the public
[Breast Cancer Wisconsin (Diagnostic) dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)
(UCI ML Repository, accessed via scikit-learn — 569 samples, 30 features).

## Contents

```
code/
  capstone_pipeline.py   # Full pipeline: acquisition -> cleaning -> EDA -> modeling -> evaluation
  requirements.txt
figures/
  class_balance.png, correlation_heatmap.png, feature_distributions.png,
  confusion_logreg.png, confusion_rf.png, roc_curves.png,
  feature_importance.png, pca_clustering.png
outputs/
  results_summary.json   # All metrics from the actual run
report/
  Capstone_Report.docx   # Full write-up: problem, methodology, EDA, modeling, evaluation, insights, reflection
```

## Approach

- **Supervised**: Logistic Regression (interpretable baseline) and a
  grid-search-tuned Random Forest, evaluated with accuracy, precision,
  recall, F1, ROC-AUC, confusion matrices, and 5-fold cross-validation.
- **Unsupervised**: PCA (dimensionality reduction/visualization) + K-Means
  clustering (k=2), evaluated with silhouette score and Adjusted Rand Index
  against the true labels (labels used only for evaluation, never for
  fitting).

## Running it

```bash
pip install -r code/requirements.txt
cd code
python capstone_pipeline.py
```

## Results (from an actual run)

| Model | Accuracy | ROC-AUC | Malignant Recall |
|---|---|---|---|
| Logistic Regression | 98.25% | 0.995 | 97.62% |
| Random Forest (tuned) | 95.61% | 0.993 | 92.86% |

Unsupervised K-Means (no labels used): Adjusted Rand Index = 0.654,
Silhouette score = 0.343 — the clustering recovers most of the diagnosis
structure without ever seeing the labels.

See `report/Capstone_Report.docx` for the full analysis, including why the
simpler model outperformed the ensemble, feature importance findings, and
a critical reflection on further work.
