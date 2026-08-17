import pandas as pd
import numpy as np
import os

RAW_PATH = os.path.join("data", "raw", "customers.csv")
OUT_PATH = os.path.join("data", "processed", "customers_clean.csv")


def load_data(path=RAW_PATH):
    return pd.read_csv(path)


def drop_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(subset="CustomerID", keep="first")
    after = len(df)
    print(f"Removed {before - after} duplicate rows")
    return df


def handle_missing_values(df):
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["MonthlyCharges"] * df["TenureMonths"]
    )
    df = df.dropna(subset=["Age", "TenureMonths", "MonthlyCharges"])
    return df


def fix_inconsistent_data(df):
    df["Gender"] = df["Gender"].str.strip().str.title()
    df["ContractType"] = df["ContractType"].str.strip()
    df["InternetService"] = df["InternetService"].str.strip()
    df["PaymentMethod"] = df["PaymentMethod"].str.strip()
    df["Churn"] = df["Churn"].str.strip().str.title()
    return df


def cap_outliers(df, columns):
    for col in columns:
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=q1, upper=q99)
    return df


def validate_ranges(df):
    df = df[(df["Age"] >= 18) & (df["Age"] <= 100)]
    df = df[df["TenureMonths"] >= 0]
    df = df[df["MonthlyCharges"] > 0]
    return df


def run_pipeline():
    df = load_data()
    print(f"Loaded {len(df)} rows")

    df = drop_duplicates(df)
    df = handle_missing_values(df)
    df = fix_inconsistent_data(df)
    df = cap_outliers(df, ["MonthlyCharges", "TotalCharges", "MonthlyUsageGB", "PaymentDelayDays"])
    df = validate_ranges(df)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned data -> {OUT_PATH}")
    print(f"Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    run_pipeline()
