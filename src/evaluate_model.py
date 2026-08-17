import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, classification_report, precision_recall_curve, average_precision_score

DATA_PATH = os.path.join("data", "processed", "customers_features.csv")
MODELS_DIR = "models"
REPORTS_DIR = "reports"


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    best_model_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
    return model, scaler, feature_names, best_model_name


def get_test_split():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["CustomerID", "Churn"])
    y = df["Churn"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


def plot_confusion_matrix(y_test, y_pred, save_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(y_test, y_proba, roc_auc, save_path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_precision_recall_curve(y_test, y_proba, save_path):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return pr_auc


def plot_feature_importance(model, feature_names, save_path, top_n=15):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(7, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", color="steelblue")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return importance_df


def run_pipeline():
    model, scaler, feature_names, best_model_name = load_artifacts()
    X_test, y_test = get_test_split()

    if best_model_name == "Logistic Regression":
        X_test_input = scaler.transform(X_test)
    else:
        X_test_input = X_test

    y_pred = model.predict(X_test_input)
    y_proba = model.predict_proba(X_test_input)[:, 1]

    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(os.path.join(MODELS_DIR, "model_comparison.json")) as f:
        results = json.load(f)
    roc_auc = results[best_model_name]["roc_auc"]

    plot_confusion_matrix(y_test, y_pred, os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_test, y_proba, roc_auc, os.path.join(REPORTS_DIR, "roc_curve.png"))
    pr_auc = plot_precision_recall_curve(y_test, y_proba, os.path.join(REPORTS_DIR, "precision_recall_curve.png"))
    importance_df = plot_feature_importance(
        model, feature_names, os.path.join(REPORTS_DIR, "feature_importance.png")
    )

    print(f"Evaluated model: {best_model_name}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    if importance_df is not None:
        print("\nTop features:")
        print(importance_df.to_string(index=False))

    print(f"\nPlots saved -> {REPORTS_DIR}/")


if __name__ == "__main__":
    run_pipeline()
