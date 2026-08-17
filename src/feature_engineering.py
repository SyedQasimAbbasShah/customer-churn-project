import pandas as pd
import numpy as np
import os

IN_PATH = os.path.join("data", "processed", "customers_clean.csv")
OUT_PATH = os.path.join("data", "processed", "customers_features.csv")


def load_clean_data(path=IN_PATH):
    return pd.read_csv(path)


def add_business_features(df):
    df["Customer_Lifetime_Months"] = df["TenureMonths"]

    df["Average_Monthly_Spend"] = np.where(
        df["TenureMonths"] > 0,
        df["TotalCharges"] / df["TenureMonths"],
        df["MonthlyCharges"],
    )

    df["Support_Calls_Per_Month"] = np.where(
        df["TenureMonths"] > 0,
        df["SupportCalls"] / df["TenureMonths"],
        df["SupportCalls"],
    )

    df["Payment_Delay_Rate"] = df["PaymentDelayDays"] / (df["TenureMonths"] + 1)

    service_flags = (
        (df["PhoneService"] == "Yes").astype(int)
        + (df["InternetService"] != "No").astype(int)
        + (df["StreamingService"] == "Yes").astype(int)
    )
    df["Service_Count"] = service_flags

    expected_usage = df["MonthlyUsageGB"].median()
    df["Usage_Change_Percentage"] = (
        (df["MonthlyUsageGB"] - expected_usage) / expected_usage * 100
    )

    df["High_Value_Customer"] = (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)).astype(int)
    df["Is_New_Customer"] = (df["TenureMonths"] <= 6).astype(int)
    df["Has_Complaints"] = (df["Complaints"] > 0).astype(int)

    return df


def encode_categoricals(df):
    binary_map = {"Yes": 1, "No": 0}
    df["PhoneService"] = df["PhoneService"].map(binary_map)
    df["StreamingService"] = df["StreamingService"].map(binary_map)
    df["Churn"] = df["Churn"].map(binary_map)

    df = pd.get_dummies(
        df,
        columns=["Gender", "ContractType", "InternetService", "PaymentMethod"],
        drop_first=True,
    )
    return df


def run_pipeline():
    df = load_clean_data()
    df = add_business_features(df)
    df = encode_categoricals(df)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved engineered features -> {OUT_PATH}")
    print(f"Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    run_pipeline()
