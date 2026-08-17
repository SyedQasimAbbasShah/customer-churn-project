import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 5000

customer_id = [f"CUST-{10000+i}" for i in range(N)]
age = np.random.randint(18, 75, N)
gender = np.random.choice(["Male", "Female"], N)

tenure_months = np.random.randint(1, 72, N)
contract_type = np.random.choice(["Month-to-Month", "One Year", "Two Year"], N, p=[0.55, 0.25, 0.20])

internet_service = np.random.choice(["DSL", "Fiber Optic", "No"], N, p=[0.35, 0.45, 0.20])
phone_service = np.random.choice(["Yes", "No"], N, p=[0.9, 0.1])
streaming_service = np.random.choice(["Yes", "No"], N, p=[0.5, 0.5])

monthly_charges = np.round(np.random.normal(65, 25, N).clip(15, 150), 2)
total_charges = np.round(monthly_charges * tenure_months * np.random.uniform(0.9, 1.05, N), 2)

monthly_usage_gb = np.round(np.random.normal(180, 90, N).clip(1, 600), 1)
sessions_per_month = np.random.randint(1, 60, N)

support_calls = np.random.poisson(1.5, N)
complaints = np.random.poisson(0.4, N)

payment_method = np.random.choice(
    ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"], N
)
payment_delay_days = np.random.poisson(3, N)

churn_score = (
    (contract_type == "Month-to-Month").astype(int) * 2.2
    + (tenure_months < 12).astype(int) * 1.8
    + (support_calls > 3).astype(int) * 1.5
    + (complaints > 1).astype(int) * 1.3
    + (monthly_charges > 90).astype(int) * 1.1
    + (payment_delay_days > 5).astype(int) * 1.0
    + (payment_method == "Electronic Check").astype(int) * 0.8
    - (tenure_months > 48).astype(int) * 2.0
    - (contract_type == "Two Year").astype(int) * 2.0
    + np.random.normal(0, 1.2, N)
)

churn_prob = 1 / (1 + np.exp(-(churn_score - 3)))
churn = (np.random.rand(N) < churn_prob).astype(int)
churn_label = np.where(churn == 1, "Yes", "No")

df = pd.DataFrame({
    "CustomerID": customer_id,
    "Age": age,
    "Gender": gender,
    "TenureMonths": tenure_months,
    "ContractType": contract_type,
    "InternetService": internet_service,
    "PhoneService": phone_service,
    "StreamingService": streaming_service,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "MonthlyUsageGB": monthly_usage_gb,
    "SessionsPerMonth": sessions_per_month,
    "SupportCalls": support_calls,
    "Complaints": complaints,
    "PaymentMethod": payment_method,
    "PaymentDelayDays": payment_delay_days,
    "Churn": churn_label,
})

missing_idx = np.random.choice(df.index, size=int(0.02 * N), replace=False)
df.loc[missing_idx, "TotalCharges"] = np.nan

dup_rows = df.sample(20, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

os.makedirs("data/raw", exist_ok=True)
output_path = os.path.join("data", "raw", "customers.csv")
df.to_csv(output_path, index=False)

print(f"Generated {len(df)} rows -> {output_path}")
print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
