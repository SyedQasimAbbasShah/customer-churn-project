import pandas as pd
import sqlite3
import os

DATA_PATH = os.path.join("data", "processed", "customers_clean.csv")
DB_PATH = os.path.join("data", "processed", "churn_database.db")


def create_database(csv_path=DATA_PATH, db_path=DB_PATH):
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)
    df.to_sql("customers", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Database created -> {db_path}")
    print(f"Loaded {len(df)} rows into 'customers' table")
    return db_path


def run_query(query, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result


QUERIES = {
    "overall_churn_rate": """
        SELECT
            ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
            COUNT(*) AS total_customers
        FROM customers
    """,
    "churn_by_contract": """
        SELECT
            ContractType,
            COUNT(*) AS total_customers,
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
            ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
        FROM customers
        GROUP BY ContractType
        ORDER BY churn_rate_pct DESC
    """,
    "revenue_at_risk_by_segment": """
        SELECT
            ContractType,
            InternetService,
            SUM(MonthlyCharges) AS monthly_revenue,
            SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END) AS revenue_at_risk
        FROM customers
        GROUP BY ContractType, InternetService
        ORDER BY revenue_at_risk DESC
    """,
    "high_support_call_churn": """
        SELECT
            SupportCalls,
            COUNT(*) AS total_customers,
            ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
        FROM customers
        GROUP BY SupportCalls
        ORDER BY SupportCalls
    """,
    "top_paying_churned_customers": """
        SELECT CustomerID, TenureMonths, ContractType, MonthlyCharges, SupportCalls
        FROM customers
        WHERE Churn = 'Yes'
        ORDER BY MonthlyCharges DESC
        LIMIT 20
    """,
    "payment_method_analysis": """
        SELECT
            PaymentMethod,
            COUNT(*) AS total_customers,
            ROUND(AVG(PaymentDelayDays), 2) AS avg_payment_delay,
            ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
        FROM customers
        GROUP BY PaymentMethod
        ORDER BY churn_rate_pct DESC
    """,
}


def run_all_queries(db_path=DB_PATH):
    results = {}
    for name, query in QUERIES.items():
        results[name] = run_query(query, db_path)
    return results


if __name__ == "__main__":
    create_database()

    results = run_all_queries()
    for name, df in results.items():
        print(f"\n{name.replace('_', ' ').title()}")
        print(df.to_string(index=False))
