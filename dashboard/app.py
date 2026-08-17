import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import sys
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import generic_pipeline as gp
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

DATA_PATH = os.path.join("data", "processed", "customers_features.csv")
RAW_PATH = os.path.join("data", "processed", "customers_clean.csv")
MODELS_DIR = "models"

REQUIRED_FILES = [
    os.path.join("data", "processed", "customers_clean.csv"),
    os.path.join("data", "processed", "customers_features.csv"),
    os.path.join(MODELS_DIR, "churn_model.pkl"),
    os.path.join(MODELS_DIR, "scaler.pkl"),
    os.path.join(MODELS_DIR, "feature_names.pkl"),
    os.path.join(MODELS_DIR, "best_model_name.pkl"),
]

PIPELINE_STEPS = [
    ("Generating dataset", os.path.join("src", "generate_data.py")),
    ("Cleaning data", os.path.join("src", "data_preprocessing.py")),
    ("Engineering features", os.path.join("src", "feature_engineering.py")),
    ("Training models", os.path.join("src", "train_model.py")),
]


def ensure_pipeline_artifacts():
    if all(os.path.exists(f) for f in REQUIRED_FILES):
        return

    project_root = os.path.join(os.path.dirname(__file__), "..")
    with st.spinner("First-time setup: generating data and training model, this takes a minute..."):
        for label, script in PIPELINE_STEPS:
            script_path = os.path.join(project_root, script)
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=project_root,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                st.error(f"Setup failed at step: {label}")
                st.code(result.stderr)
                st.stop()


ensure_pipeline_artifacts()


@st.cache_data
def load_data():
    features_df = pd.read_csv(DATA_PATH)
    clean_df = pd.read_csv(RAW_PATH)
    return features_df, clean_df


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    best_model_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
    return model, scaler, feature_names, best_model_name


def get_risk_level(probability):
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    return "LOW"


def predict_all(features_df, model, scaler, feature_names, best_model_name):
    input_df = features_df.reindex(columns=feature_names, fill_value=0)
    if best_model_name == "Logistic Regression":
        input_scaled = scaler.transform(input_df)
        probabilities = model.predict_proba(input_scaled)[:, 1]
    else:
        probabilities = model.predict_proba(input_df)[:, 1]
    return probabilities


st.title("Customer Churn Prediction & Business Intelligence")
st.caption("End-to-end churn analytics platform")

features_df, clean_df = load_data()
model, scaler, feature_names, best_model_name = load_model_artifacts()
probabilities = predict_all(features_df, model, scaler, feature_names, best_model_name)

clean_df = clean_df.reset_index(drop=True)
clean_df["Churn_Probability"] = np.round(probabilities * 100, 1)
clean_df["Risk_Level"] = [get_risk_level(p) for p in probabilities]

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Executive Overview", "Customer Analysis", "Churn Analysis", "Prediction", "Upload & Train Custom Model"]
)

with tab1:
    total_customers = len(clean_df)
    current_churn_rate = (clean_df["Churn"] == "Yes").mean() * 100
    high_risk_customers = (clean_df["Risk_Level"] == "HIGH").sum()
    revenue_at_risk = clean_df.loc[clean_df["Risk_Level"] == "HIGH", "MonthlyCharges"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Current Churn Rate", f"{current_churn_rate:.1f}%")
    col3.metric("High-Risk Customers", f"{high_risk_customers:,}")
    col4.metric("Est. Revenue at Risk", f"${revenue_at_risk:,.0f}/mo")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        churn_counts = clean_df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        fig = px.pie(churn_counts, names="Churn", values="Count", title="Churn Distribution", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        risk_counts = clean_df["Risk_Level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig = px.bar(
            risk_counts, x="Risk Level", y="Count", title="Predicted Risk Distribution",
            color="Risk Level", color_discrete_map={"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_left, col_right = st.columns(2)

    with col_left:
        fig = px.histogram(clean_df, x="Age", color="Churn", barmode="overlay", title="Age Distribution by Churn")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = px.histogram(clean_df, x="TenureMonths", color="Churn", barmode="overlay", title="Tenure Distribution by Churn")
        st.plotly_chart(fig, use_container_width=True)

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        fig = px.box(clean_df, x="ContractType", y="MonthlyCharges", color="Churn", title="Monthly Charges by Contract Type")
        st.plotly_chart(fig, use_container_width=True)

    with col_right2:
        payment_churn = clean_df.groupby(["PaymentMethod", "Churn"]).size().reset_index(name="Count")
        fig = px.bar(payment_churn, x="PaymentMethod", y="Count", color="Churn", barmode="group", title="Churn by Payment Method")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    col_left, col_right = st.columns(2)

    with col_left:
        contract_churn = clean_df.groupby("ContractType")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
        contract_churn.columns = ["ContractType", "ChurnRate"]
        fig = px.bar(contract_churn, x="ContractType", y="ChurnRate", title="Churn Rate by Contract Type (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        internet_churn = clean_df.groupby("InternetService")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
        internet_churn.columns = ["InternetService", "ChurnRate"]
        fig = px.bar(internet_churn, x="InternetService", y="ChurnRate", title="Churn Rate by Internet Service (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Support Calls vs Churn")
    support_churn = clean_df.groupby("SupportCalls")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
    support_churn.columns = ["SupportCalls", "ChurnRate"]
    fig = px.line(support_churn, x="SupportCalls", y="ChurnRate", markers=True, title="Churn Rate by Number of Support Calls (%)")
    st.plotly_chart(fig, use_container_width=True)

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).head(10)
        fig = px.bar(importance_df, x="importance", y="feature", orientation="h", title="Top Churn Drivers")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Predict Churn for a Customer")

    input_mode = st.radio("Choose input method", ["Select existing customer", "Upload CSV"], horizontal=True)

    if input_mode == "Select existing customer":
        selected_id = st.selectbox("Select Customer ID", clean_df["CustomerID"].tolist())
        customer_row = clean_df[clean_df["CustomerID"] == selected_id].iloc[0]
        feature_row = features_df[features_df["CustomerID"] == selected_id].iloc[0]

        probability = customer_row["Churn_Probability"]
        risk_level = customer_row["Risk_Level"]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Churn Probability", f"{probability}%")
            color = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}[risk_level]
            st.markdown(f"### Risk Level: {color} {risk_level}")

        with col2:
            factors = []
            if feature_row.get("TenureMonths", 99) < 12:
                factors.append("Short customer tenure")
            if feature_row.get("MonthlyCharges", 0) > 90:
                factors.append("High monthly charges")
            if feature_row.get("SupportCalls", 0) > 3:
                factors.append("Frequent support requests")
            if feature_row.get("ContractType_Two Year", 0) == 0 and feature_row.get("ContractType_One Year", 0) == 0:
                factors.append("Month-to-month contract")
            if feature_row.get("Complaints", 0) > 1:
                factors.append("Multiple complaints filed")
            if not factors:
                factors.append("No major risk factors identified")

            st.markdown("**Major Risk Factors:**")
            for f in factors:
                st.markdown(f"- {f}")

            if risk_level == "HIGH":
                st.warning("Recommended Action: Offer a discounted annual contract and proactive customer support.")
            elif risk_level == "MEDIUM":
                st.info("Recommended Action: Monitor account and offer loyalty incentives.")
            else:
                st.success("Recommended Action: No immediate action required.")

    else:
        uploaded_file = st.file_uploader("Upload customer data (CSV)", type=["csv"])
        if uploaded_file is not None:
            upload_df = pd.read_csv(uploaded_file)
            input_df = upload_df.reindex(columns=feature_names, fill_value=0)

            if best_model_name == "Logistic Regression":
                input_scaled = scaler.transform(input_df)
                upload_probabilities = model.predict_proba(input_scaled)[:, 1]
            else:
                upload_probabilities = model.predict_proba(input_df)[:, 1]

            result_df = pd.DataFrame({
                "CustomerID": upload_df.get("CustomerID", pd.Series(range(len(upload_df)))),
                "Churn_Probability": np.round(upload_probabilities * 100, 1),
                "Risk_Level": [get_risk_level(p) for p in upload_probabilities],
            }).sort_values("Churn_Probability", ascending=False)

            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                "Download Predictions",
                result_df.to_csv(index=False),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

    st.markdown("---")
    st.markdown("### Top 20 Highest Risk Customers")
    top_risk = clean_df.sort_values("Churn_Probability", ascending=False).head(20)
    st.dataframe(
        top_risk[["CustomerID", "TenureMonths", "ContractType", "MonthlyCharges", "Churn_Probability", "Risk_Level"]],
        use_container_width=True,
    )

with tab5:
    st.subheader("Upload Your Own Dataset")
    st.caption("Upload any CSV with a binary target column and this app will clean it, engineer features automatically, train and compare 5 models, and let you generate predictions — all using your data instead of the built-in demo dataset.")

    custom_file = st.file_uploader("Upload dataset (CSV)", type=["csv"], key="custom_upload")

    if custom_file is not None:
        custom_df = pd.read_csv(custom_file)
        st.markdown(f"**Preview** ({custom_df.shape[0]} rows, {custom_df.shape[1]} columns)")
        st.dataframe(custom_df.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Select the target column (what you want to predict)", custom_df.columns.tolist())
        with col2:
            positive_options = custom_df[target_col].dropna().unique().tolist() if target_col else []
            positive_value = st.selectbox("Which value means 'Yes / Positive / Churned'?", positive_options)

        auto_id_cols = gp.detect_id_columns(custom_df)
        drop_cols = st.multiselect(
            "Columns to exclude from training (IDs, names, etc.)",
            [c for c in custom_df.columns if c != target_col],
            default=[c for c in auto_id_cols if c != target_col],
        )

        if st.button("Train Model on This Dataset", type="primary"):
            with st.spinner("Cleaning data and training 5 models..."):
                clean_custom_df = gp.clean_generic(custom_df)
                X, y, custom_feature_names, dropped_high_card = gp.prepare_features(
                    clean_custom_df, target_col, positive_value, drop_cols
                )

                if dropped_high_card:
                    st.info(f"Dropped high-cardinality text columns automatically: {', '.join(dropped_high_card)}")

                try:
                    custom_result = gp.train_compare_generic(X, y)
                except ValueError as e:
                    st.error(str(e))
                    custom_result = None

            if custom_result:
                st.session_state["custom_result"] = custom_result
                st.session_state["custom_drop_cols"] = drop_cols
                st.session_state["custom_target_col"] = target_col
                st.session_state["custom_raw_df"] = custom_df
                st.success(f"Training complete. Best model: {custom_result['best_model_name']}")

    if "custom_result" in st.session_state:
        result = st.session_state["custom_result"]

        st.markdown("---")
        st.markdown("### Model Comparison")
        comparison_df = pd.DataFrame(result["results"]).T
        st.dataframe(comparison_df, use_container_width=True)

        best_name = result["best_model_name"]
        st.markdown(f"**Best model: {best_name}** (selected by ROC-AUC)")

        col1, col2 = st.columns(2)
        with col1:
            importance_df = gp.get_feature_importance(result["best_model"], result["feature_names"])
            if importance_df is not None:
                fig = px.bar(importance_df, x="importance", y="feature", orientation="h", title="Top Feature Importances")
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            preds = result["predictions"][best_name]
            cm = pd.DataFrame(
                confusion_matrix(preds["y_test"], preds["y_pred"]),
                index=["Actual: No", "Actual: Yes"],
                columns=["Predicted: No", "Predicted: Yes"],
            )
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Predict on New Data Using This Model")
        predict_file = st.file_uploader("Upload new records to predict (same column structure)", type=["csv"], key="custom_predict_upload")

        if predict_file is not None:
            new_df = pd.read_csv(predict_file)
            probs = gp.predict_new_data(
                result["best_model"],
                result["scaler"],
                result["feature_names"],
                best_name,
                new_df,
                drop_cols=st.session_state["custom_drop_cols"] + [st.session_state["custom_target_col"]],
            )

            pred_result_df = new_df.copy()
            pred_result_df["Predicted_Probability_%"] = np.round(probs * 100, 1)
            pred_result_df["Risk_Level"] = [get_risk_level(p) for p in probs]
            pred_result_df = pred_result_df.sort_values("Predicted_Probability_%", ascending=False)

            st.dataframe(pred_result_df, use_container_width=True)
            st.download_button(
                "Download Predictions",
                pred_result_df.to_csv(index=False),
                file_name="custom_predictions.csv",
                mime="text/csv",
            )
