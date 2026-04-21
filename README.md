# Telecom Customer Churn Prediction

> Predicting which telecom customers will churn — and why — using machine learning.
> **Live demo:** [telecom-churn-prediction-santimuru.streamlit.app](https://telecom-churn-prediction-santimuru.streamlit.app)

---

## Overview

Customer churn is one of the most critical challenges in the telecom industry.
This project builds an end-to-end ML pipeline that:

1. **Trains and compares** 3 models (Logistic Regression, Random Forest, Gradient Boosting) on the IBM Telco Customer Churn dataset
2. **Identifies key churn drivers** using feature importance analysis
3. **Exposes a live dashboard** where you can explore insights and simulate individual customer risk

In a real-world deployment at a cable/telecom company, I applied similar models and achieved a **13% churn reduction** through targeted retention campaigns driven by model outputs.

---

## Live Dashboard

| Section               | What you'll find                                                 |
| --------------------- | ---------------------------------------------------------------- |
| 📊 Model Metrics      | Accuracy, Precision, Recall, F1, AUC-ROC — with model comparison |
| 🔍 Feature Importance | Which variables drive churn predictions most                     |
| 🎯 Simulator          | Enter any customer's characteristics → get churn probability     |
| 📈 Segmentation       | Churn breakdown by contract, tenure, services, billing           |

---

## Tech Stack

- **Python 3.11**
- **Scikit-learn** — model training, preprocessing pipeline
- **Streamlit** — interactive dashboard
- **Plotly** — visualizations
- **Pandas / NumPy** — data processing

---

## Dataset

IBM Telco Customer Churn — 7,043 customers, 21 features.

- `customerID` — unique customer identifier
- `tenure` — months as a customer
- `Contract` — month-to-month, one year, two year
- `MonthlyCharges`, `TotalCharges`
- Internet, phone, streaming services
- `Churn` — target variable (Yes/No)

The script auto-downloads the dataset from the public IBM repository on first run.

---

## Project Structure

```
telecom-churn-prediction/
├── app/
│   └── app.py              # Streamlit dashboard
├── src/
│   └── train.py            # Model training script
├── notebook/
│   └── exploratory_analysis.ipynb   # EDA & modeling walkthrough
├── data/
│   └── telco_churn.csv     # Downloaded on first run
├── models/
│   ├── churn_model.pkl     # Best trained model
│   └── model_meta.pkl      # Metrics, feature importance, metadata
├── requirements.txt
└── README.md
```

---

## Setup & Run

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/telecom-churn-prediction.git
cd telecom-churn-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train.py
```

This will:

- Download the IBM Telco dataset (~500KB)
- Train 3 models and compare them
- Save the best model to `models/`

### 4. Run the Streamlit app

```bash
streamlit run app/app.py
```

---

## Key Results

| Model               | Accuracy | Recall | F1   | AUC-ROC |
| ------------------- | -------- | ------ | ---- | ------- |
| Logistic Regression | ~80%     | ~76%   | ~62% | ~0.84   |
| Random Forest       | ~79%     | ~80%   | ~62% | ~0.83   |
| Gradient Boosting   | ~81%     | ~74%   | ~63% | ~0.85   |

> Actual values depend on the run — see the dashboard for live metrics.

**Top churn predictors:**

- Contract type (month-to-month = high risk)
- Tenure (new customers churn more)
- Internet service type (Fiber optic)
- Payment method (Electronic check)
- Monthly charges

---

## About

Built by [Santiago Martínez](https://santimuru.github.io) — Data Analyst with 6+ years of experience in telecom, e-commerce, and consulting.

Previously deployed churn prediction models in a production environment at a cable operator, integrating model outputs into CRM workflows to drive retention campaigns.

---

## License

MIT
