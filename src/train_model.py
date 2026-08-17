import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier

DATA_PATH = os.path.join("data", "processed", "customers_features.csv")
MODELS_DIR = "models"
RESULTS_PATH = os.path.join(MODELS_DIR, "model_comparison.json")


def load_features(path=DATA_PATH):
    return pd.read_csv(path)


def split_data(df):
    drop_cols = ["CustomerID", "Churn"]
    X = df.drop(columns=drop_cols)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
        ),
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
    }


def run_pipeline():
    df = load_features()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    models = get_models()
    results = {}
    trained_models = {}

    for name, model in models.items():
        if name in ["Logistic Regression"]:
            model.fit(X_train_scaled, y_train)
            metrics = evaluate_model(model, X_test_scaled, y_test)
        else:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)

        results[name] = metrics
        trained_models[name] = model
        print(f"{name}: {metrics}")

    best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = trained_models[best_model_name]
    print(f"\nBest model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']})")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "churn_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(list(X_train.columns), os.path.join(MODELS_DIR, "feature_names.pkl"))
    joblib.dump(best_model_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved best model -> {MODELS_DIR}/churn_model.pkl")
    print(f"Saved comparison results -> {RESULTS_PATH}")

    return results, best_model_name


if __name__ == "__main__":
    run_pipeline()
