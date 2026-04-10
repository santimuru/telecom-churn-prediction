"""
app.py — Streamlit dashboard: Telecom Customer Churn Prediction
"""

import os
import sys
import warnings
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "model_meta.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_churn.csv")

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction · Telecom",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .churn-high { color: #d62728; font-weight: bold; font-size: 2rem; }
    .churn-low  { color: #2ca02c; font-weight: bold; font-size: 2rem; }
    .section-title { font-size: 1.3rem; font-weight: 600; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


# ─── Load artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta


DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)

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
        "Sección",
        ["📊 Métricas del Modelo", "🔍 Feature Importance", "🎯 Simulador", "📈 Segmentación"],
        label_visibility="collapsed",
    )
    st.divider()
    if meta:
        st.caption(f"Modelo: **{meta['model_name']}**")
        st.caption(f"Dataset: {meta['n_samples']:,} clientes")
        st.caption(f"Churn rate: {meta['churn_rate']*100:.1f}%")

# ─── Guard: modelo no entrenado ───────────────────────────────────────────────
if model is None:
    st.error(
        "⚠️ Modelo no encontrado. Ejecutá primero:\n\n"
        "```bash\npython src/train.py\n```"
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 1. MÉTRICAS DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Métricas del Modelo":
    st.header("📊 Métricas del Modelo")
    st.caption(
        "Comparación de 3 modelos entrenados sobre el dataset IBM Telco Customer Churn. "
        "Se selecciona el de mayor AUC-ROC."
    )

    best = meta["best_metrics"]
    all_metrics = meta["metrics"]

    # KPIs del mejor modelo
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
        st.metric("AUC-ROC", f"{best['roc_auc']:.3f}", delta="↑ mejor modelo")

    st.divider()

    # Tabla comparativa
    st.subheader("Comparación de modelos")
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "Modelo": f"{'★ ' if name == best['name'] else ''}{name}",
            "Accuracy": f"{m['accuracy']:.1%}",
            "Precision": f"{m['precision']:.1%}",
            "Recall": f"{m['recall']:.1%}",
            "F1": f"{m['f1']:.1%}",
            "AUC-ROC": f"{m['roc_auc']:.3f}",
        })
    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Gráfico de barras comparativo
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
        title="Comparación de métricas por modelo",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 **Nota metodológica:** El modelo elegido balancea Recall alto (detectar churners reales) "
        "con Precision razonable, priorizando AUC-ROC como métrica de selección. "
        "En producción, el umbral de decisión se ajusta según el costo de falsos positivos/negativos."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Importance":
    st.header("🔍 Feature Importance")
    st.caption("Variables con mayor poder predictivo sobre la probabilidad de churn.")

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
        labels={"importance": "Importancia relativa", "feature": "Variable"},
        title=f"Top {top_n} variables predictoras de churn",
        height=max(400, top_n * 28),
    )
    fig.update_layout(coloraxis_showscale=False, yaxis=dict(tickfont=dict(size=11)))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Tabla completa")
    st.dataframe(
        imp_df.assign(importance=imp_df["importance"].round(4)),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. SIMULADOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Simulador":
    st.header("🎯 Simulador de Probabilidad de Churn")
    st.caption("Ingresá las características de un cliente para ver su probabilidad de darse de baja.")

    with st.form("simulator_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Perfil del cliente")
            gender = st.selectbox("Género", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Tiene pareja", ["No", "Yes"])
            dependents = st.selectbox("Tiene dependientes", ["No", "Yes"])
            tenure = st.slider("Antigüedad (meses)", 0, 72, 12)

        with col2:
            st.subheader("Servicios contratados")
            phone = st.selectbox("Servicio telefónico", ["Yes", "No"])
            multiple_lines = st.selectbox("Múltiples líneas", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Servicio de Internet", ["DSL", "Fiber optic", "No"])
            online_sec = st.selectbox("Seguridad online", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Backup online", ["No", "Yes", "No internet service"])
            device_prot = st.selectbox("Protección de dispositivo", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Soporte técnico", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming películas", ["No", "Yes", "No internet service"])

        with col3:
            st.subheader("Facturación y contrato")
            contract = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Factura paperless", ["Yes", "No"])
            payment = st.selectbox(
                "Método de pago",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly = st.slider("Cargo mensual ($)", 18.0, 120.0, 65.0, step=0.5)
            total = st.slider("Cargo total acumulado ($)", 0.0, 9000.0, float(monthly * tenure), step=10.0)

        submitted = st.form_submit_button("Calcular probabilidad de churn", use_container_width=True)

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
                title={"text": "Probabilidad de Churn"},
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_detail:
            if pred == 1:
                st.error(f"⚠️ **ALTO RIESGO DE CHURN** ({prob:.1%})")
                st.markdown("**Factores de riesgo detectados:**")
                if contract == "Month-to-month":
                    st.markdown("- Contrato mensual (sin fidelización)")
                if internet == "Fiber optic":
                    st.markdown("- Fiber optic (alta competencia)")
                if tenure < 12:
                    st.markdown("- Antigüedad baja (< 12 meses)")
                if payment == "Electronic check":
                    st.markdown("- Pago con cheque electrónico")
            else:
                st.success(f"✅ **BAJO RIESGO DE CHURN** ({prob:.1%})")
                st.markdown("**Factores de retención:**")
                if contract in ["One year", "Two year"]:
                    st.markdown("- Contrato de largo plazo")
                if tenure > 24:
                    st.markdown("- Cliente fidelizado (>24 meses)")
                if payment in ["Bank transfer (automatic)", "Credit card (automatic)"]:
                    st.markdown("- Pago automático (menor fricción)")

            st.metric("Predicción", "CHURN" if pred == 1 else "RETIENE", delta=f"{abs(prob - meta['churn_rate']):.1%} vs promedio")
            st.metric("Churn rate promedio del dataset", f"{meta['churn_rate']:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SEGMENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Segmentación":
    st.header("📈 Segmentación por Variables Clave")
    st.caption("Distribución de churn según diferentes segmentos del dataset.")

    if df is None:
        st.warning("Dataset no encontrado. Ejecutá `python src/train.py` para descargarlo.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Contrato", "Antigüedad", "Servicios", "Facturación"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            churn_contract = df.groupby("Contract")["Churn_bin"].mean().reset_index()
            churn_contract.columns = ["Contrato", "Churn Rate"]
            fig = px.bar(
                churn_contract, x="Contrato", y="Churn Rate",
                color="Churn Rate", color_continuous_scale="RdYlGn_r",
                title="Churn rate por tipo de contrato",
                text=churn_contract["Churn Rate"].apply(lambda x: f"{x:.1%}"),
            )
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.histogram(
                df, x="Contract", color="Churn",
                barmode="group", title="Distribución de clientes por contrato",
                color_discrete_map={"Yes": "#d62728", "No": "#2ca02c"},
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.info("📌 Los clientes **Month-to-month** tienen un churn rate ~3x más alto que los de contrato anual.")

    with tab2:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 6, 12, 24, 48, 72],
            labels=["0-6m", "6-12m", "12-24m", "24-48m", "48-72m"],
        )
        churn_tenure = df.groupby("tenure_group", observed=True)["Churn_bin"].mean().reset_index()
        churn_tenure.columns = ["Antigüedad", "Churn Rate"]

        fig = px.line(
            churn_tenure, x="Antigüedad", y="Churn Rate",
            markers=True, title="Churn rate por antigüedad del cliente",
            labels={"Churn Rate": "Tasa de Churn"},
        )
        fig.add_hline(
            y=meta["churn_rate"], line_dash="dash",
            annotation_text=f"Promedio ({meta['churn_rate']:.1%})", line_color="gray",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.info("📌 El churn se concentra en los primeros 12 meses — crítico para estrategias de onboarding.")

    with tab3:
        col1, col2 = st.columns(2)
        service_cols = {
            "InternetService": "Internet",
            "TechSupport": "Soporte Técnico",
            "OnlineSecurity": "Seg. Online",
            "StreamingTV": "Streaming TV",
        }
        for i, (col, label) in enumerate(service_cols.items()):
            with col1 if i % 2 == 0 else col2:
                ch = df.groupby(col)["Churn_bin"].mean().reset_index()
                ch.columns = [label, "Churn Rate"]
                fig = px.bar(
                    ch, x=label, y="Churn Rate",
                    title=f"Churn rate por {label}",
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
                title="Cargo mensual: Churn vs Retención",
                labels={"MonthlyCharges": "Cargo Mensual ($)"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            churn_payment = df.groupby("PaymentMethod")["Churn_bin"].mean().reset_index()
            churn_payment.columns = ["Método de Pago", "Churn Rate"]
            churn_payment["Método de Pago"] = churn_payment["Método de Pago"].str.replace(" (automatic)", " (auto)", regex=False)
            fig = px.bar(
                churn_payment.sort_values("Churn Rate", ascending=True),
                x="Churn Rate", y="Método de Pago",
                orientation="h", title="Churn rate por método de pago",
                color="Churn Rate", color_continuous_scale="RdYlGn_r",
                text=churn_payment.sort_values("Churn Rate")["Churn Rate"].apply(lambda x: f"{x:.1%}"),
            )
            fig.update_layout(coloraxis_showscale=False, xaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig, use_container_width=True)

        st.info("📌 Fiber optic + Electronic check = combo de mayor riesgo. Clientes con altos cargos mensuales churnan más.")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Proyecto de portfolio · [Santiago Martínez](https://santimuru.github.io) · "
    "Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)"
)
