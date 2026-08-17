import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

CLEAN_PATH = os.path.join("data", "processed", "customers_clean.csv")
REPORTS_DIR = "reports"


def load_clean_data(path=CLEAN_PATH):
    return pd.read_csv(path)


def plot_churn_distribution(df, save_path):
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Churn", hue="Churn", palette="Set2", legend=False)
    plt.title("Churn Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_correlation_matrix(df, save_path):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return corr


def plot_churn_by_tenure(df, save_path):
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="TenureMonths", hue="Churn", kde=True, element="step")
    plt.title("Churn by Customer Tenure")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_monthly_charges_vs_churn(df, save_path):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn", palette="Set3", legend=False)
    plt.title("Monthly Charges vs Churn")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_churn_by_contract(df, save_path):
    contract_churn = df.groupby("ContractType")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
    contract_churn.columns = ["ContractType", "ChurnRate"]
    plt.figure(figsize=(6, 4))
    sns.barplot(data=contract_churn, x="ContractType", y="ChurnRate", hue="ContractType", palette="Set1", legend=False)
    plt.title("Churn Rate by Contract Type (%)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def customer_segmentation(df):
    def spend_tier(x):
        if x < 40:
            return "Low Spend"
        elif x < 90:
            return "Medium Spend"
        return "High Spend"

    def tenure_tier(x):
        if x < 12:
            return "New"
        elif x < 48:
            return "Established"
        return "Loyal"

    df["Spend_Segment"] = df["MonthlyCharges"].apply(spend_tier)
    df["Tenure_Segment"] = df["TenureMonths"].apply(tenure_tier)

    segment_summary = (
        df.groupby(["Spend_Segment", "Tenure_Segment"])["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="ChurnRate")
    )
    return df, segment_summary


def plot_segmentation(segment_summary, save_path):
    pivot = segment_summary.pivot(index="Tenure_Segment", columns="Spend_Segment", values="ChurnRate")
    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title("Churn Rate (%) by Customer Segment")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def answer_eda_questions(df):
    answers = {}

    tenure_corr = df["TenureMonths"].corr((df["Churn"] == "Yes").astype(int))
    answers["tenure_effect"] = f"Correlation between tenure and churn: {tenure_corr:.3f} (negative means longer tenure reduces churn)"

    charges_corr = df["MonthlyCharges"].corr((df["Churn"] == "Yes").astype(int))
    answers["charges_effect"] = f"Correlation between monthly charges and churn: {charges_corr:.3f}"

    contract_rates = df.groupby("ContractType")["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
    answers["highest_churn_contract"] = f"Highest churn contract type: {contract_rates.idxmax()} ({contract_rates.max():.1f}%)"

    support_corr = df["SupportCalls"].corr((df["Churn"] == "Yes").astype(int))
    answers["support_effect"] = f"Correlation between support calls and churn: {support_corr:.3f}"

    payment_rates = df.groupby("PaymentMethod")["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
    answers["highest_churn_payment"] = f"Highest churn payment method: {payment_rates.idxmax()} ({payment_rates.max():.1f}%)"

    df["AgeGroup"] = pd.cut(df["Age"], bins=[17, 30, 45, 60, 100], labels=["18-30", "31-45", "46-60", "60+"])
    age_rates = df.groupby("AgeGroup", observed=True)["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
    answers["highest_churn_age"] = f"Highest churn age group: {age_rates.idxmax()} ({age_rates.max():.1f}%)"

    return answers


def run_pipeline():
    df = load_clean_data()
    os.makedirs(REPORTS_DIR, exist_ok=True)

    plot_churn_distribution(df, os.path.join(REPORTS_DIR, "eda_churn_distribution.png"))
    plot_correlation_matrix(df, os.path.join(REPORTS_DIR, "eda_correlation_matrix.png"))
    plot_churn_by_tenure(df, os.path.join(REPORTS_DIR, "eda_churn_by_tenure.png"))
    plot_monthly_charges_vs_churn(df, os.path.join(REPORTS_DIR, "eda_monthly_charges_vs_churn.png"))
    plot_churn_by_contract(df, os.path.join(REPORTS_DIR, "eda_churn_by_contract.png"))

    df, segment_summary = customer_segmentation(df)
    plot_segmentation(segment_summary, os.path.join(REPORTS_DIR, "eda_customer_segmentation.png"))

    answers = answer_eda_questions(df)

    print("EDA Key Findings:")
    for key, value in answers.items():
        print(f"- {value}")

    print(f"\nPlots saved -> {REPORTS_DIR}/")

    return df, segment_summary, answers


if __name__ == "__main__":
    run_pipeline()
