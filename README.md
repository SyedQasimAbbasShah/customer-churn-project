# Customer Churn Prediction & Business Intelligence System

End-to-end data science project that predicts customer churn, explains the key drivers behind it, and presents everything through an interactive Streamlit dashboard.

## Tech Stack
Python 3.13, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, Plotly, Streamlit

## Project Structure
```
customer-churn-project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
├── src/
│   ├── generate_data.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── explainability.py
│   ├── database.py
│   ├── generate_report.py
│   └── predict.py
├── models/
├── dashboard/
│   └── app.py
├── reports/
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux / Mac
pip install -r requirements.txt
```

## Pipeline (run in order)
```bash
python src/generate_data.py
python src/data_preprocessing.py
python src/eda.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate_model.py
python src/explainability.py
python src/database.py
python src/generate_report.py
python src/predict.py
```

## Dashboard
```bash
streamlit run dashboard/app.py
```

## Notebooks
The same pipeline is also available as Jupyter notebooks for step-by-step, exploratory use:
```bash
jupyter notebook notebooks/
```
- `01_data_cleaning.ipynb` - load raw data and run the cleaning pipeline
- `02_eda.ipynb` - churn distribution, correlation matrix, tenure/contract/segment analysis
- `03_feature_engineering.ipynb` - business feature creation and categorical encoding
- `04_model_training.ipynb` - train/compare models, evaluate the best one, SHAP explainability

## SQL Layer
`src/database.py` loads the cleaned dataset into a local SQLite database (`data/processed/churn_database.db`) and runs business-facing SQL queries (churn by contract, revenue at risk by segment, payment method analysis, etc.). Swap the `sqlite3` connection for a `psycopg2`/`SQLAlchemy` PostgreSQL connection to point at a production database.

## Explainability
`src/explainability.py` uses SHAP to explain the best model's predictions globally (summary/bar plots) and for individual customers, going beyond feature importance to show *why* a specific customer is at risk.

## Business Report
`src/generate_report.py` produces a polished PDF (`reports/business_report.pdf`) combining the executive summary, model comparison table, EDA charts, model diagnostics, segmentation heatmap, and retention recommendations — ready to share with non-technical stakeholders.

## Status
Complete end-to-end project: data generation → cleaning → EDA → feature engineering → model training/comparison → evaluation (incl. PR-AUC) → SHAP explainability → SQL analytics layer → PDF business report → prediction → Streamlit dashboard → Jupyter notebooks.
