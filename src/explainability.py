import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import shap

DATA_PATH = os.path.join("data", "processed", "customers_features.csv")
MODELS_DIR = "models"
REPORTS_DIR = "reports"


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    best_model_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
    return model, scaler, feature_names, best_model_name


def get_sample_data(feature_names, sample_size=300):
    df = pd.read_csv(DATA_PATH)
    X = df.reindex(columns=feature_names, fill_value=0)
    sample = X.sample(n=min(sample_size, len(X)), random_state=42)
    return sample, df.loc[sample.index, "CustomerID"]


def build_explainer(model, best_model_name, background_data):
    if best_model_name in ["Random Forest", "Decision Tree", "Gradient Boosting", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, background_data)
    return explainer


def plot_summary(shap_values, X_sample, save_path):
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_bar_importance(shap_values, X_sample, save_path):
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def explain_single_customer(explainer, X_sample, customer_ids, target_id, feature_names, save_path):
    idx_list = list(customer_ids[customer_ids == target_id].index)
    if not idx_list:
        return None
    row_idx = X_sample.index.get_loc(idx_list[0])

    shap_values = explainer(X_sample)
    values = shap_values.values[row_idx]

    if values.ndim > 1:
        values = values[:, 1]

    contrib_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values,
    }).sort_values("shap_value", key=abs, ascending=False).head(10)

    plt.figure(figsize=(7, 5))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in contrib_df["shap_value"]]
    plt.barh(contrib_df["feature"], contrib_df["shap_value"], color=colors)
    plt.xlabel("SHAP value (impact on churn probability)")
    plt.title(f"Feature Contributions for {target_id}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return contrib_df


def run_pipeline():
    model, scaler, feature_names, best_model_name = load_artifacts()
    X_sample, customer_ids = get_sample_data(feature_names)

    background = X_sample if best_model_name == "Logistic Regression" else X_sample.iloc[:50]
    explainer = build_explainer(model, best_model_name, background)

    shap_values_raw = explainer(X_sample)
    values = shap_values_raw.values
    if values.ndim == 3:
        values = values[:, :, 1]

    os.makedirs(REPORTS_DIR, exist_ok=True)
    plot_summary(values, X_sample, os.path.join(REPORTS_DIR, "shap_summary.png"))
    plot_bar_importance(values, X_sample, os.path.join(REPORTS_DIR, "shap_bar_importance.png"))

    print(f"SHAP explainability generated for model: {best_model_name}")
    print(f"Sample size used: {len(X_sample)}")

    target_id = customer_ids.iloc[0]
    contrib_df = explain_single_customer(
        explainer, X_sample, customer_ids, target_id, feature_names,
        os.path.join(REPORTS_DIR, "shap_single_customer.png"),
    )

    if contrib_df is not None:
        print(f"\nTop SHAP contributors for {target_id}:")
        print(contrib_df.to_string(index=False))

    print(f"\nPlots saved -> {REPORTS_DIR}/")


if __name__ == "__main__":
    run_pipeline()
