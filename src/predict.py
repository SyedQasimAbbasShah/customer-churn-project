import pandas as pd
import numpy as np
import os
import joblib

MODELS_DIR = "models"


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    best_model_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
    return model, scaler, feature_names, best_model_name


def prepare_input(customer_row, feature_names):
    row = customer_row.reindex(feature_names, fill_value=0)
    return pd.DataFrame([row])


def get_risk_level(probability):
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    return "LOW"


def get_risk_factors(customer_row):
    factors = []

    if customer_row.get("TenureMonths", 99) < 12:
        factors.append("Short customer tenure")

    if customer_row.get("MonthlyCharges", 0) > 90:
        factors.append("High monthly charges")

    if customer_row.get("SupportCalls", 0) > 3:
        factors.append("Frequent support requests")

    if customer_row.get("ContractType_Two Year", 0) == 0 and customer_row.get("ContractType_One Year", 0) == 0:
        factors.append("Month-to-month contract")

    if customer_row.get("Payment_Delay_Rate", 0) > 0.3:
        factors.append("Frequent payment delays")

    if customer_row.get("Complaints", 0) > 1:
        factors.append("Multiple complaints filed")

    if not factors:
        factors.append("No major risk factors identified")

    return factors


def get_recommendation(risk_level, factors):
    if risk_level == "HIGH":
        if "Month-to-month contract" in factors:
            return "Offer a discounted annual contract and proactive customer support."
        return "Assign to retention team for immediate proactive outreach."
    elif risk_level == "MEDIUM":
        return "Monitor account and offer loyalty incentives."
    return "No immediate action required. Continue standard engagement."


def predict_single_customer(customer_row, customer_id="N/A"):
    model, scaler, feature_names, best_model_name = load_artifacts()

    input_df = prepare_input(customer_row, feature_names)

    if best_model_name == "Logistic Regression":
        input_scaled = scaler.transform(input_df)
        probability = model.predict_proba(input_scaled)[0][1]
    else:
        probability = model.predict_proba(input_df)[0][1]

    risk_level = get_risk_level(probability)
    factors = get_risk_factors(customer_row)
    recommendation = get_recommendation(risk_level, factors)

    return {
        "customer_id": customer_id,
        "churn_probability": round(probability * 100, 1),
        "risk_level": risk_level,
        "risk_factors": factors,
        "recommendation": recommendation,
    }


def predict_batch(df):
    model, scaler, feature_names, best_model_name = load_artifacts()

    ids = df["CustomerID"] if "CustomerID" in df.columns else pd.Series(range(len(df)))
    feature_df = df.reindex(columns=feature_names, fill_value=0)

    if best_model_name == "Logistic Regression":
        input_scaled = scaler.transform(feature_df)
        probabilities = model.predict_proba(input_scaled)[:, 1]
    else:
        probabilities = model.predict_proba(feature_df)[:, 1]

    results = pd.DataFrame({
        "CustomerID": ids,
        "Churn_Probability": np.round(probabilities * 100, 1),
        "Risk_Level": [get_risk_level(p) for p in probabilities],
    })

    return results.sort_values("Churn_Probability", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join("data", "processed", "customers_features.csv"))
    sample_customer = df.iloc[0]
    result = predict_single_customer(sample_customer, customer_id=sample_customer["CustomerID"])

    print(f"Customer ID: {result['customer_id']}")
    print(f"Churn Probability: {result['churn_probability']}%")
    print(f"Risk Level: {result['risk_level']}")
    print("Major Risk Factors:")
    for i, factor in enumerate(result["risk_factors"], 1):
        print(f"  {i}. {factor}")
    print(f"Recommended Action: {result['recommendation']}")

    print("\nBatch prediction (top 5 highest risk):")
    batch_results = predict_batch(df)
    print(batch_results.head(5).to_string(index=False))
