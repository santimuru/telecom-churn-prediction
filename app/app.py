"""
app.py — Streamlit dashboard: Telecom Customer Churn Prediction
"""

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "model_meta.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_churn.csv")

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction · Telecom",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta


@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        import requests
        r = requests.get(DATASET_URL, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(__import__("io").StringIO(r.text))
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)
    return df


model, meta = load_model()
df = load_data()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/phone-office.png", width=60)
    st.title("Telecom Churn\nPrediction")
    st.caption("Portfolio · Santiago Martínez")
    st.divider()
    page = st.radio(
        "Section",
        ["📊 Model Metrics", "🔍 Feature Importance", "🎯 Simulator", "📈 Segmentation"],
        label_visibility="collapsed",
    )
    st.divider()
    if meta:
        st.caption(f"Model: **{meta['model_name']}**")
        st.caption(f"Dataset: {meta['n_samples']:,} customers")
        st.caption(f"Churn rate: {meta['churn_rate']*100:.1f}%")

# ─── Guard: model not trained ────────────────────────────────────────────────
if model is None:
    st.error(
        "Model not found. Please run:\n\n"
        "```bash\npython src/train.py\n```"
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 1. MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Model Metrics":
    st.header("📊 Model Metrics")
    st.caption(
        "Comparison of 3 models trained on the IBM Telco Customer Churn dataset. "
        "Best model selected by AUC-ROC."
    )

    best = meta["best_metrics"]
    all_metrics = meta["metrics"]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", f"{best['accuracy']:.1%}")
    with col2:
        st.metric("Precision", f"{best['precision']:.1%}")
    with col3:
        st.metric("Recall", f"{best['recall']:.1%}")
    with col4:
        st.metric("F1 Score", f"{best['f1']:.1%}")
    with col5:
        st.metric("AUC-ROC", f"{best['roc_auc']:.3f}", delta="best model")

    st.divider()

    st.subheader("Model comparison")
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "Model": f"{'★ ' if name == best['name'] else ''}{name}",
            "Accuracy": f"{m['accuracy']:.1%}",
            "Precision": f"{m['precision']:.1%}",
            "Recall": f"{m['recall']:.1%}",
            "F1": f"{m['f1']:.1%}",
            "AUC-ROC": f"{m['roc_auc']:.3f}",
        })
    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    fig = go.Figure()
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    colors = ["#636EFA", "#EF553B", "#00CC96"]

    for i, (name, m) in enumerate(all_metrics.items()):
        fig.add_trace(go.Bar(
            name=name,
            x=metric_labels,
            y=[m[k] for k in metric_names],
            marker_color=colors[i],
            text=[f"{m[k]:.3f}" for k in metric_names],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1.1], title="Score"),
        title="Model metrics comparison",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 **Methodology note:** The selected model balances high Recall (catching real churners) "
        "with reasonable Precision, using AUC-ROC as the selection metric. "
        "In production, the decision threshold is tuned based on the cost of false positives vs. false negatives."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Importance":
    st.header("🔍 Feature Importance")
    st.caption("Variables with the highest predictive power for churn probability.")

    imp_df = meta["feature_importance"]
    top_n = st.slider("Top N features", 5, 30, 15)
    top = imp_df.head(top_n).sort_values("importance")

    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Blues",
        labels={"importance": "Relative importance", "feature": "Feature"},
        title=f"Top {top_n} churn predictors",
        height=max(400, top_n * 28),
    )
    fig.update_layout(coloraxis_showscale=False, yaxis=dict(tickfont=dict(size=11)))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Full table")
    st.dataframe(
        imp_df.assign(importance=imp_df["importance"].round(4)),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Simulator":
    st.header("🎯 Churn Probability Simulator")
    st.caption("Enter a customer's characteristics to predict their churn probability.")

    with st.form("simulator_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Customer profile")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has partner", ["No", "Yes"])
            dependents = st.selectbox("Has dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            st.subheader("Contracted services")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            online_sec = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            device_prot = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

        with col3:
            st.subheader("Billing & contract")
            contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 65.0, step=0.5)
            total = st.slider("Total charges ($)", 0.0, 9000.0, float(monthly * tenure), step=10.0)

        submitted = st.form_submit_button("Calculate churn probability", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_prot,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }])

        prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        st.divider()
        col_gauge, col_detail = st.columns([1, 1])

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 48}},
                delta={"reference": meta["churn_rate"] * 100, "suffix": "% (base rate)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#d62728" if prob > 0.5 else "#2ca02c"},
                    "steps": [
                        {"range": [0, 30], "color": "#d4edda"},
                        {"range": [30, 60], "color": "#fff3cd"},
                        {"range": [60, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
                title={"text": "Churn Probability"},
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_detail:
            if pred == 1:
                st.error(f"⚠️ **HIGH CHURN RISK** ({prob:.1%})")
                st.markdown("**Detected risk factors:**")
                if contract == "Month-to-month":
                    st.markdown("- Month-to-month contract (no commitment)")
                if internet == "Fiber optic":
                    st.markdown("- Fiber optic (high-competition segment)")
                if tenure < 12:
                    st.markdown("- Low tenure (< 12 months)")
                if payment == "Electronic check":
                    st.markdown("- Electronic check payment")
            else:
                st.success(f"✅ **LOW CHURN RISK** ({prob:.1%})")
                st.markdown("**Retention factors:**")
                if contract in ["One year", "Two year"]:
                    st.markdown("- Long-term contract")
                if tenure > 24:
                    st.markdown("- Loyal customer (>24 months)")
                if payment in ["Bank transfer (automatic)", "Credit card (automatic)"]:
                    st.markdown("- Automatic payment (low friction)")

            st.metric("Prediction", "CHURN" if pred == 1 else "RETAIN", delta=f"{abs(prob - meta['churn_rate']):.1%} vs average")
            st.metric("Dataset average churn rate", f"{meta['churn_rate']:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Segmentation":
    st.header("📈 Segmentation by Key Variables")
    st.caption("Churn distribution across different customer segments.")

    if df is None:
        st.warning("Dataset not found. Run `python src/train.py` to download it.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Contract", "Tenure", "Services", "Billing"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            churn_contract = df.groupby("Contract")["Churn_bin"].mean().reset_index()
            churn_contract.columns = ["Contract", "Churn Rate"]
            fig = px.bar(
                churn_contract, x="Contract", y="Churn Rate",
                color="Churn Rate", color_continuous_scale="RdYlGn_r",
                title="Churn rate by contract type",
                text=churn_contract["Churn Rate"].apply(lambda x: f"{x:.1%}"),
            )
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.histogram(
                df, x="Contract", color="Churn",
                barmode="group", title="Customer distribution by contract type",
                color_discrete_map={"Yes": "#d62728", "No": "#2ca02c"},
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.info("📌 **Month-to-month** customers churn ~3x more than annual contract customers.")

    with tab2:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 6, 12, 24, 48, 72],
            labels=["0-6m", "6-12m", "12-24m", "24-48m", "48-72m"],
        )
        churn_tenure = df.groupby("tenure_group", observed=True)["Churn_bin"].mean().reset_index()
        churn_tenure.columns = ["Tenure", "Churn Rate"]

        fig = px.line(
            churn_tenure, x="Tenure", y="Churn Rate",
            markers=True, title="Churn rate by customer tenure",
            labels={"Churn Rate": "Churn Rate"},
        )
        fig.add_hline(
            y=meta["churn_rate"], line_dash="dash",
            annotation_text=f"Average ({meta['churn_rate']:.1%})", line_color="gray",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.info("📌 Churn is concentrated in the first 12 months — critical window for onboarding strategies.")

    with tab3:
        col1, col2 = st.columns(2)
        service_cols = {
            "InternetService": "Internet Service",
            "TechSupport": "Tech Support",
            "OnlineSecurity": "Online Security",
            "StreamingTV": "Streaming TV",
        }
        for i, (col, label) in enumerate(service_cols.items()):
            with col1 if i % 2 == 0 else col2:
                ch = df.groupby(col)["Churn_bin"].mean().reset_index()
                ch.columns = [label, "Churn Rate"]
                fig = px.bar(
                    ch, x=label, y="Churn Rate",
                    title=f"Churn rate by {label}",
                    color="Churn Rate", color_continuous_scale="RdYlGn_r",
                    text=ch["Churn Rate"].apply(lambda x: f"{x:.1%}"),
                )
                fig.update_layout(coloraxis_showscale=False, yaxis=dict(tickformat=".0%"), height=300)
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(
                df, x="Churn", y="MonthlyCharges",
                color="Churn",
                color_discrete_map={"Yes": "#d62728", "No": "#2ca02c"},
                title="Monthly charges: Churn vs Retained",
                labels={"MonthlyCharges": "Monthly Charges ($)"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            churn_payment = df.groupby("PaymentMethod")["Churn_bin"].mean().reset_index()
            churn_payment.columns = ["Payment Method", "Churn Rate"]
            churn_payment["Payment Method"] = churn_payment["Payment Method"].str.replace(" (automatic)", " (auto)", regex=False)
            fig = px.bar(
                churn_payment.sort_values("Churn Rate", ascending=True),
                x="Churn Rate", y="Payment Method",
                orientation="h", title="Churn rate by payment method",
                color="Churn Rate", color_continuous_scale="RdYlGn_r",
                text=churn_payment.sort_values("Churn Rate")["Churn Rate"].apply(lambda x: f"{x:.1%}"),
            )
            fig.update_layout(coloraxis_showscale=False, xaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig, use_container_width=True)

        st.info("📌 Fiber optic + Electronic check = highest risk combo. High monthly charges correlate with churn.")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Portfolio project · [Santiago Martínez](https://santimuru.github.io) · "
    "Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)"
)
