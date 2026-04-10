"""
train.py — Entrena y evalúa modelos de churn prediction.
Descarga el dataset IBM Telco Customer Churn, preprocesa y serializa el mejor modelo.

Uso:
    python src/train.py
"""

import os
import sys
import warnings
import joblib
import requests
import pandas as pd
import numpy as np
from io import StringIO

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "telco_churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
PREP_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "model_meta.pkl")

# ─── Dataset URL (IBM Telco, repositorio público) ─────────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)


def download_data():
    if os.path.exists(DATA_RAW):
        print(f"[✓] Dataset ya existe en {DATA_RAW}")
        return pd.read_csv(DATA_RAW)

    print("[↓] Descargando dataset IBM Telco Customer Churn...")
    r = requests.get(DATASET_URL, timeout=30)
    r.raise_for_status()
    os.makedirs(os.path.dirname(DATA_RAW), exist_ok=True)
    with open(DATA_RAW, "w", encoding="utf-8") as f:
        f.write(r.text)
    print(f"[✓] Guardado en {DATA_RAW}")
    return pd.read_csv(StringIO(r.text))


def preprocess(df: pd.DataFrame):
    df = df.copy()

    # TotalCharges tiene espacios vacíos en lugar de NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Columna sin valor predictivo
    df.drop(columns=["customerID"], inplace=True)

    return df


def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    return preprocessor, cat_cols, num_cols


def get_feature_names(preprocessor, cat_cols, num_cols):
    ohe = preprocessor.named_transformers_["cat"]
    cat_features = ohe.get_feature_names_out(cat_cols).tolist()
    return num_cols + cat_features


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "name": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }
    return metrics


def main():
    # 1. Datos
    df = download_data()
    print(f"[i] Shape: {df.shape}")
    df = preprocess(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Preprocessor
    preprocessor, cat_cols, num_cols = build_preprocessor(X_train)

    # 3. Modelos a comparar
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    }

    results = []
    trained_pipelines = {}

    for name, clf in models.items():
        print(f"[~] Entrenando {name}...")
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test, name)
        results.append(metrics)
        trained_pipelines[name] = pipe
        print(f"    AUC-ROC: {metrics['roc_auc']}  F1: {metrics['f1']}")

    # 4. Seleccionar mejor modelo por AUC-ROC
    best = max(results, key=lambda r: r["roc_auc"])
    best_model = trained_pipelines[best["name"]]
    print(f"\n[★] Mejor modelo: {best['name']}  (AUC-ROC: {best['roc_auc']})")

    # 5. Feature importance
    clf = best_model.named_steps["clf"]
    prep_fitted = best_model.named_steps["prep"]
    feature_names = get_feature_names(prep_fitted, cat_cols, num_cols)

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # 6. Guardar
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    meta = {
        "model_name": best["name"],
        "metrics": {r["name"]: r for r in results},
        "best_metrics": best,
        "feature_importance": importance_df,
        "feature_names_raw": X.columns.tolist(),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "churn_rate": round(y.mean(), 4),
        "n_samples": len(df),
    }
    joblib.dump(meta, META_PATH)

    print(f"[✓] Modelo guardado en {MODEL_PATH}")
    print(f"[✓] Metadata guardada en {META_PATH}")
    print("\n[Top 10 features:]")
    print(importance_df.head(10).to_string(index=False))

    print("\n[Métricas por modelo:]")
    for r in results:
        print(f"  {r['name']:25s} | ACC: {r['accuracy']} | AUC: {r['roc_auc']} | F1: {r['f1']}")


if __name__ == "__main__":
    main()
