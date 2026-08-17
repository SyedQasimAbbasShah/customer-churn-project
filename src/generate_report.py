import pandas as pd
import numpy as np
import os
import json
import joblib

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

DATA_PATH = os.path.join("data", "processed", "customers_clean.csv")
MODELS_DIR = "models"
REPORTS_DIR = "reports"
OUTPUT_PATH = os.path.join(REPORTS_DIR, "business_report.pdf")


def load_data():
    df = pd.read_csv(DATA_PATH)
    with open(os.path.join(MODELS_DIR, "model_comparison.json")) as f:
        model_results = json.load(f)
    best_model_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
    return df, model_results, best_model_name


def build_metrics_table(model_results):
    header = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    rows = [header]
    for name, metrics in model_results.items():
        rows.append([
            name,
            f"{metrics['accuracy']:.3f}",
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1']:.3f}",
            f"{metrics['roc_auc']:.3f}",
            f"{metrics.get('pr_auc', 0):.3f}",
        ])
    return rows


def image_if_exists(path, width=6.2 * inch, height=3.4 * inch):
    if os.path.exists(path):
        return Image(path, width=width, height=height)
    return None


def run_pipeline():
    df, model_results, best_model_name = load_data()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    doc = SimpleDocTemplate(OUTPUT_PATH, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=22, spaceAfter=6)
    heading_style = ParagraphStyle("HeadingStyle", parent=styles["Heading2"], spaceBefore=16, spaceAfter=8)
    body_style = styles["Normal"]

    story = []

    story.append(Paragraph("Customer Churn Prediction & Business Intelligence Report", title_style))
    story.append(Paragraph("Automated analysis of customer churn drivers, model performance, and retention recommendations", body_style))
    story.append(Spacer(1, 16))

    total_customers = len(df)
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    avg_monthly_charge = df["MonthlyCharges"].mean()
    revenue_at_risk = df.loc[df["Churn"] == "Yes", "MonthlyCharges"].sum()

    story.append(Paragraph("Executive Summary", heading_style))
    summary_table_data = [
        ["Total Customers", f"{total_customers:,}"],
        ["Current Churn Rate", f"{churn_rate:.1f}%"],
        ["Average Monthly Charges", f"${avg_monthly_charge:.2f}"],
        ["Monthly Revenue at Risk", f"${revenue_at_risk:,.2f}"],
        ["Best Performing Model", best_model_name],
    ]
    summary_table = Table(summary_table_data, colWidths=[2.6 * inch, 2.6 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)

    story.append(Paragraph("Model Performance Comparison", heading_style))
    metrics_table_data = build_metrics_table(model_results)
    metrics_table = Table(metrics_table_data, colWidths=[1.3 * inch] + [0.85 * inch] * 6)
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(metrics_table)

    story.append(PageBreak())

    story.append(Paragraph("Churn Overview", heading_style))
    for img_path in ["eda_churn_distribution.png", "eda_correlation_matrix.png"]:
        img = image_if_exists(os.path.join(REPORTS_DIR, img_path))
        if img:
            story.append(img)
            story.append(Spacer(1, 12))

    story.append(Paragraph("Churn Drivers", heading_style))
    for img_path in ["eda_churn_by_contract.png", "eda_churn_by_tenure.png", "feature_importance.png"]:
        img = image_if_exists(os.path.join(REPORTS_DIR, img_path))
        if img:
            story.append(img)
            story.append(Spacer(1, 12))

    story.append(PageBreak())

    story.append(Paragraph("Model Diagnostics", heading_style))
    for img_path in ["confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png"]:
        img = image_if_exists(os.path.join(REPORTS_DIR, img_path), width=4.5 * inch, height=3.2 * inch)
        if img:
            story.append(img)
            story.append(Spacer(1, 10))

    story.append(Paragraph("Customer Segmentation", heading_style))
    seg_img = image_if_exists(os.path.join(REPORTS_DIR, "eda_customer_segmentation.png"))
    if seg_img:
        story.append(seg_img)

    story.append(PageBreak())

    story.append(Paragraph("Recommendations", heading_style))
    recommendations = [
        "Prioritize retention outreach for month-to-month contract customers, which show the highest churn rate.",
        "Offer discounted annual or two-year contract upgrades to high-risk, high-spend customers.",
        "Investigate service quality for customers with 4 or more support calls, where churn risk rises sharply.",
        "Review electronic check as a payment method, since it correlates with higher churn and payment delays.",
        "Deploy the trained model in the Streamlit dashboard for ongoing, real-time churn monitoring.",
    ]
    for rec in recommendations:
        story.append(Paragraph(f"- {rec}", body_style))
        story.append(Spacer(1, 4))

    doc.build(story)
    print(f"Business report generated -> {OUTPUT_PATH}")


if __name__ == "__main__":
    run_pipeline()
