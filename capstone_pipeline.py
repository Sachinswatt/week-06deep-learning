"""
capstone_pipeline.py
---------------------
Integrative Capstone Project: Breast Cancer Diagnosis — Predictive and
Descriptive Analysis.

Combines the full data science pipeline covered across the internship:
  1. Data acquisition
  2. Preprocessing / cleaning
  3. Exploratory data analysis (EDA)
  4. Supervised modeling (Logistic Regression + Random Forest classification)
  5. Unsupervised modeling (PCA + KMeans clustering)
  6. Evaluation (classification metrics, ROC-AUC, clustering agreement)

Dataset: Breast Cancer Wisconsin (Diagnostic) dataset — a public, well-known
UCI Machine Learning Repository benchmark, available directly via
scikit-learn (sklearn.datasets.load_breast_cancer), so no external download
or API key is required. 569 samples, 30 numeric features computed from
digitized images of fine needle aspirate (FNA) biopsies, binary target
(malignant / benign).

Run:
    pip install -r requirements.txt
    python capstone_pipeline.py

Outputs are written to ../figures and ../outputs.
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, silhouette_score,
    adjusted_rand_score,
)

SEED = 42
FIG_DIR = "../figures"
OUT_DIR = "../outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)


# =============================================================================
# 1. DATA ACQUISITION
# =============================================================================
def acquire_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
    return df, data.feature_names.tolist()


# =============================================================================
# 2. PREPROCESSING / CLEANING
# =============================================================================
def preprocess(df, feature_names):
    report = {}
    report["n_rows"] = int(df.shape[0])
    report["n_features"] = len(feature_names)
    report["missing_values_total"] = int(df[feature_names].isna().sum().sum())
    report["duplicate_rows"] = int(df.duplicated().sum())
    report["class_balance"] = df["diagnosis"].value_counts().to_dict()

    # No missing values / duplicates expected in this curated dataset, but we
    # check programmatically rather than assuming, and would drop/impute here
    # if any were found.
    df = df.drop_duplicates()

    X = df[feature_names].values
    y = df["target"].values  # 0 = malignant, 1 = benign

    return df, X, y, report


# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# =============================================================================
def run_eda(df, feature_names):
    # Class balance plot
    plt.figure(figsize=(5, 4))
    order = df["diagnosis"].value_counts()
    sns.barplot(x=order.index, y=order.values, hue=order.index, palette="Set2", legend=False)
    plt.ylabel("Count"); plt.title("Class Balance: Diagnosis")
    for i, v in enumerate(order.values):
        plt.text(i, v + 3, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/class_balance.png", dpi=160)
    plt.close()

    # Correlation heatmap (mean-level features only, for readability)
    mean_features = [f for f in feature_names if f.startswith("mean")]
    corr = df[mean_features].corr()
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": 0.8}, xticklabels=True, yticklabels=True)
    plt.title("Correlation Matrix — Mean-Level Features")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=160)
    plt.close()

    # Distribution of a few key features by class
    key_feats = ["mean radius", "mean texture", "mean concavity", "mean smoothness"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, feat in zip(axes.flat, key_feats):
        sns.kdeplot(data=df, x=feat, hue="diagnosis", fill=True, alpha=0.4, ax=ax, palette="Set2")
        ax.set_title(feat)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_distributions.png", dpi=160)
    plt.close()

    eda_summary = {
        "most_correlated_pair": None,
        "class_balance": df["diagnosis"].value_counts().to_dict(),
    }
    corr_abs = corr.abs().copy()
    vals = corr_abs.values.copy()
    np.fill_diagonal(vals, 0)
    max_idx = np.unravel_index(np.argmax(vals), vals.shape)
    eda_summary["most_correlated_pair"] = [
        mean_features[max_idx[0]], mean_features[max_idx[1]], float(vals[max_idx])
    ]
    return eda_summary


# =============================================================================
# 4. SUPERVISED MODELING
# =============================================================================
def supervised_modeling(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # --- Logistic Regression (baseline, interpretable) ---
    logreg = LogisticRegression(max_iter=5000, random_state=SEED)
    logreg.fit(X_train_s, y_train)
    y_pred_lr = logreg.predict(X_test_s)
    y_proba_lr = logreg.predict_proba(X_test_s)[:, 1]

    results["logistic_regression"] = evaluate_classifier(y_test, y_pred_lr, y_proba_lr)

    # --- Random Forest (non-linear, feature importance) with small grid search ---
    param_grid = {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}
    rf = RandomForestClassifier(random_state=SEED)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train_s, y_train)
    best_rf = grid.best_estimator_
    y_pred_rf = best_rf.predict(X_test_s)
    y_proba_rf = best_rf.predict_proba(X_test_s)[:, 1]

    results["random_forest"] = evaluate_classifier(y_test, y_pred_rf, y_proba_rf)
    results["random_forest"]["best_params"] = grid.best_params_

    # 5-fold cross-validation (on Random Forest) for robustness check
    cv_scores = cross_val_score(best_rf, scaler.transform(X), y, cv=5, scoring="accuracy")
    results["random_forest"]["cv_accuracy_mean"] = float(cv_scores.mean())
    results["random_forest"]["cv_accuracy_std"] = float(cv_scores.std())

    # Confusion matrices
    plot_confusion(y_test, y_pred_lr, "Logistic Regression", f"{FIG_DIR}/confusion_logreg.png")
    plot_confusion(y_test, y_pred_rf, "Random Forest", f"{FIG_DIR}/confusion_rf.png")

    # ROC curves (both models on one plot)
    plt.figure(figsize=(6, 5.5))
    for name, proba in [("Logistic Regression", y_proba_lr), ("Random Forest", y_proba_rf)]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Test Set"); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/roc_curves.png", dpi=160)
    plt.close()

    # Feature importance (Random Forest)
    importances = pd.Series(best_rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    top15 = importances.head(15)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=top15.values, y=top15.index, hue=top15.index, palette="viridis", legend=False)
    plt.xlabel("Importance"); plt.title("Top 15 Feature Importances — Random Forest")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_importance.png", dpi=160)
    plt.close()

    results["top_features"] = top15.head(5).to_dict()

    return results, (X_test_s, y_test), scaler, best_rf


def evaluate_classifier(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "classification_report": classification_report(y_true, y_pred, target_names=["malignant", "benign"], digits=4),
    }


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4.3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["malignant", "benign"], yticklabels=["malignant", "benign"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix — {title}")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# =============================================================================
# 5. UNSUPERVISED MODELING
# =============================================================================
def unsupervised_modeling(X, y):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_s)
    explained_var = pca.explained_variance_ratio_

    kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(X_s)

    sil_score = silhouette_score(X_s, cluster_labels)
    ari = adjusted_rand_score(y, cluster_labels)

    # PCA scatter colored by true diagnosis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for lbl, name, color in [(0, "malignant", "#d62728"), (1, "benign", "#2ca02c")]:
        mask = y == lbl
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, alpha=0.6, s=25, color=color)
    axes[0].set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% var)")
    axes[0].set_title("PCA Projection — Colored by True Diagnosis")
    axes[0].legend()

    for c in [0, 1]:
        mask = cluster_labels == c
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {c}", alpha=0.6, s=25)
    axes[1].set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% var)")
    axes[1].set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% var)")
    axes[1].set_title("PCA Projection — Colored by K-Means Cluster")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/pca_clustering.png", dpi=160)
    plt.close()

    # Cross-tab of clusters vs true labels
    crosstab = pd.crosstab(
        pd.Series(cluster_labels, name="cluster"),
        pd.Series(y, name="true_label").map({0: "malignant", 1: "benign"})
    )

    return {
        "pca_explained_variance_ratio": explained_var.tolist(),
        "silhouette_score": float(sil_score),
        "adjusted_rand_index": float(ari),
        "cluster_vs_true_crosstab": crosstab.to_dict(),
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    df, feature_names = acquire_data()
    df, X, y, cleaning_report = preprocess(df, feature_names)
    eda_summary = run_eda(df, feature_names)
    supervised_results, test_data, scaler, best_rf = supervised_modeling(X, y, feature_names)
    unsupervised_results = unsupervised_modeling(X, y)

    all_results = {
        "cleaning_report": cleaning_report,
        "eda_summary": eda_summary,
        "supervised_results": {
            k: v for k, v in supervised_results.items() if k != "top_features"
        },
        "top_features": supervised_results["top_features"],
        "unsupervised_results": unsupervised_results,
    }

    with open(f"{OUT_DIR}/results_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("=" * 70)
    print("CLEANING REPORT:", cleaning_report)
    print("=" * 70)
    print("EDA SUMMARY:", eda_summary)
    print("=" * 70)
    print("LOGISTIC REGRESSION:")
    print(supervised_results["logistic_regression"]["classification_report"])
    print("RANDOM FOREST:")
    print(supervised_results["random_forest"]["classification_report"])
    print("Random Forest best params:", supervised_results["random_forest"]["best_params"])
    print("Random Forest CV accuracy: %.4f +/- %.4f" % (
        supervised_results["random_forest"]["cv_accuracy_mean"],
        supervised_results["random_forest"]["cv_accuracy_std"]))
    print("=" * 70)
    print("UNSUPERVISED (KMeans + PCA):", unsupervised_results["silhouette_score"],
          unsupervised_results["adjusted_rand_index"])
    print("PCA explained variance:", unsupervised_results["pca_explained_variance_ratio"])


if __name__ == "__main__":
    main()
