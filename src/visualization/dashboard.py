"""Streamlit dashboard — interactive churn analytics and live model inference.

Run with:
    streamlit run src/visualization/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src.data.generator import SyntheticDataGenerator
from src.data.preprocessor import TARGET_COLUMN, DataPreprocessor
from src.evaluation.metrics import BusinessMetrics, ModelEvaluator
from src.features.engineering import FeatureEngineer
from src.models.classical import RandomForestModel, XGBoostModel
from src.models.classical.logistic_regression import LogisticRegressionModel
from src.visualization.plots import ChurnPlotter

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
)

plotter = ChurnPlotter()
evaluator = ModelEvaluator(BusinessMetrics())


# ──────────────────────────────────────────────
# Cached data & model loading
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Generating synthetic dataset…")
def load_data(n_samples: int = 10_000, seed: int = 42) -> pd.DataFrame:
    df = SyntheticDataGenerator(random_state=seed).generate(n_samples)
    return FeatureEngineer().fit_transform(df)


@st.cache_resource(show_spinner="Training models…")
def train_models(
    n_samples: int,
) -> tuple[list, pd.DataFrame, pd.DataFrame, pd.Series]:
    df = load_data(n_samples)
    y = df[TARGET_COLUMN]
    X = df.drop(
        columns=[TARGET_COLUMN, "customer_id", "monthly_charge_tier", "tenure_segment"],
        errors="ignore",
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )
    preprocessor = DataPreprocessor()
    X_train_p, _ = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    models_list = [
        LogisticRegressionModel(),
        RandomForestModel(n_estimators=100),
        XGBoostModel(n_estimators=200, early_stopping_rounds=15),
    ]
    for m in models_list:
        m.fit(X_train_p, y_train, X_val_p, y_val)

    return models_list, X_test_p, preprocessor, y_test


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
n_samples = st.sidebar.selectbox("Dataset size", [5_000, 10_000, 25_000], index=1)
page = st.sidebar.radio(
    "Navigation",
    ["📊 EDA", "🤖 Model Comparison", "🔍 Live Prediction"],
)

df = load_data(n_samples)
models, X_test_p, preprocessor, y_test = train_models(n_samples)

# ──────────────────────────────────────────────
# EDA Page
# ──────────────────────────────────────────────
if page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.caption(f"Dataset: {n_samples:,} synthetic customers")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Churn Rate", f"{df['churned'].mean():.1%}")
    col3.metric("Avg Monthly Charge", f"{df['monthly_charges'].mean():.0f} PLN")
    col4.metric("Avg Tenure", f"{df['tenure_months'].mean():.0f} months")

    st.plotly_chart(plotter.churn_distribution(df["churned"]), use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        seg = st.selectbox("Segment column", ["contract_type", "payment_method", "internet_service", "segment"])
        st.plotly_chart(plotter.churn_by_segment(df, seg), use_container_width=True)
    with col_right:
        st.plotly_chart(
            plotter.feature_importance(
                df[["tenure_months", "monthly_charges", "num_support_calls",
                    "num_products", "total_charges"]].corrwith(df["churned"]).abs(),
                title="Raw Feature Correlation with Churn",
            ),
            use_container_width=True,
        )

# ──────────────────────────────────────────────
# Model Comparison Page
# ──────────────────────────────────────────────
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")

    results = []
    probas = []
    for m in models:
        proba = m.predict_proba(X_test_p)
        result = evaluator.evaluate(proba, y_test, m.metadata.name)
        results.append(result)
        probas.append((m.metadata.name, y_test.to_numpy(), proba))

    comparison = evaluator.compare(results)
    st.dataframe(
        comparison.style.background_gradient(
            subset=["roc_auc", "f1", "business_savings_pln"], cmap="Greens"
        ),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plotter.roc_curves(probas), use_container_width=True)
    with col2:
        st.plotly_chart(plotter.precision_recall_curves(probas), use_container_width=True)

    best_model = models[2]  # XGBoost
    best_proba = best_model.predict_proba(X_test_p)
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            plotter.score_distribution(best_proba, y_test.to_numpy()),
            use_container_width=True,
        )
    with col4:
        if hasattr(best_model, "feature_importance"):
            st.plotly_chart(
                plotter.feature_importance(
                    best_model.feature_importance(),
                    title="XGBoost Feature Importance (Gain)",
                    top_n=15,
                ),
                use_container_width=True,
            )

# ──────────────────────────────────────────────
# Live Prediction Page
# ──────────────────────────────────────────────
elif page == "🔍 Live Prediction":
    st.title("🔍 Live Customer Churn Prediction")
    st.caption("Fill in customer attributes and get an instant churn score from all models.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            monthly = st.number_input("Monthly charges (PLN)", 18.0, 120.0, 65.0, step=1.0)
            total = st.number_input("Total charges (PLN)", 0.0, 10_000.0, monthly * tenure)

        with col2:
            contract = st.selectbox("Contract type", ["month-to-month", "one_year", "two_year"])
            payment = st.selectbox(
                "Payment method",
                ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
            )
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])

        with col3:
            n_products = st.slider("Number of products", 1, 6, 2)
            n_calls = st.slider("Support calls", 0, 10, 1)
            tech_support = st.checkbox("Has tech support")
            senior = st.checkbox("Senior citizen")
            partner = st.checkbox("Has partner")

        submitted = st.form_submit_button("Predict Churn", type="primary")

    if submitted:
        raw_input = pd.DataFrame([{
            "customer_id": "LIVE_001",
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "contract_type": contract,
            "payment_method": payment,
            "internet_service": internet,
            "segment": "Consumer",
            "num_products": n_products,
            "num_support_calls": n_calls,
            "has_tech_support": tech_support,
            "has_online_backup": False,
            "is_senior_citizen": senior,
            "has_partner": partner,
            "has_dependents": False,
        }])

        raw_input = FeatureEngineer().fit_transform(raw_input)
        X_input = raw_input.drop(
            columns=[TARGET_COLUMN, "customer_id", "monthly_charge_tier", "tenure_segment"],
            errors="ignore",
        )
        X_proc = preprocessor.transform(X_input)

        st.subheader("Results")
        cols = st.columns(len(models))
        for col, model in zip(cols, models):
            proba = float(model.predict_proba(X_proc)[0])
            color = "🔴" if proba >= 0.5 else "🟢"
            col.metric(
                label=model.metadata.name.replace("_", " ").title(),
                value=f"{proba:.1%}",
                delta=f"{'HIGH RISK' if proba >= 0.5 else 'LOW RISK'}",
                delta_color="inverse",
            )

        avg_proba = np.mean([m.predict_proba(X_proc)[0] for m in models])
        st.progress(float(avg_proba), text=f"Ensemble churn probability: **{avg_proba:.1%}**")

        if avg_proba >= 0.6:
            st.error("⚠️ High churn risk — recommend immediate retention intervention")
        elif avg_proba >= 0.4:
            st.warning("⚡ Moderate risk — monitor and consider proactive outreach")
        else:
            st.success("✅ Low churn risk — customer appears engaged")
