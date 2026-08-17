import pandas as pd
import numpy as np

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
    confusion_matrix,
)
from xgboost import XGBClassifier


def detect_id_columns(df):
    id_like = []
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            continue
        if df[col].nunique() == len(df):
            id_like.append(col)
    return id_like


def clean_generic(df):
    df = df.drop_duplicates().copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)

    return df


def prepare_features(df, target_col, positive_value, drop_cols=None):
    drop_cols = drop_cols or []
    work_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()

    y = (work_df[target_col].astype(str) == str(positive_value)).astype(int)
    X = work_df.drop(columns=[target_col])

    categorical_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    low_card_cols = [c for c in categorical_cols if X[c].nunique() <= 30]
    high_card_cols = [c for c in categorical_cols if X[c].nunique() > 30]

    X = X.drop(columns=high_card_cols)

    if low_card_cols:
        X = pd.get_dummies(X, columns=low_card_cols, drop_first=True)

    X = X.select_dtypes(include=[np.number, bool])
    X = X.astype(float)

    return X, y, list(X.columns), high_card_cols


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
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
    }, y_pred, y_proba


def train_compare_generic(X, y, progress_callback=None):
    if y.nunique() < 2:
        raise ValueError("Target column must have at least two distinct classes.")

    test_size = 0.2 if len(X) >= 50 else 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models()
    results = {}
    trained_models = {}
    predictions = {}

    for name, model in models.items():
        if progress_callback:
            progress_callback(name)

        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            metrics, y_pred, y_proba = evaluate_model(model, X_test_scaled, y_test)
        else:
            model.fit(X_train, y_train)
            metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)

        results[name] = metrics
        trained_models[name] = model
        predictions[name] = {"y_test": y_test, "y_pred": y_pred, "y_proba": y_proba}

    best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = trained_models[best_model_name]

    return {
        "results": results,
        "trained_models": trained_models,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "scaler": scaler,
        "predictions": predictions,
        "feature_names": list(X.columns),
        "X_test": X_test,
        "y_test": y_test,
    }


def get_feature_importance(model, feature_names, top_n=15):
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

    return importance_df


def predict_new_data(model, scaler, feature_names, best_model_name, new_df, drop_cols=None):
    drop_cols = drop_cols or []
    work_df = new_df.drop(columns=[c for c in drop_cols if c in new_df.columns], errors="ignore").copy()

    categorical_cols = [c for c in work_df.columns if not pd.api.types.is_numeric_dtype(work_df[c]) and not pd.api.types.is_bool_dtype(work_df[c])]
    if categorical_cols:
        work_df = pd.get_dummies(work_df, columns=categorical_cols, drop_first=True)

    work_df = work_df.reindex(columns=feature_names, fill_value=0)
    work_df = work_df.astype(float)

    if best_model_name == "Logistic Regression":
        input_data = scaler.transform(work_df)
    else:
        input_data = work_df

    probabilities = model.predict_proba(input_data)[:, 1]
    return probabilities
