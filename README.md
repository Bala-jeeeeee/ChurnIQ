# 📊 ChurnIQ – Customer Churn Analytics & Prediction Dashboard

A production-ready Streamlit application for customer churn analytics and ML-powered prediction.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
streamlit run app.py
```

## 📁 Project Structure

```
churniq/
├── app.py                  # Main entry point & auth routing
├── config.py               # Constants, paths, feature lists
├── requirements.txt
├── data/
│   └── customer_churn_business_dataset.csv
├── models/                 # Auto-created after training
├── core/
│   ├── database.py         # SQLite auth & prediction history
│   ├── preprocess.py       # Data loading & sklearn pipeline
│   ├── trainer.py          # 5-model training loop + evaluation
│   └── predictor.py        # Inference engine (single & batch)
├── ui/
│   └── charts.py           # Plotly chart factory
└── pages/
    ├── home.py             # Dataset overview & business context
    ├── eda.py              # EDA with tabs & visualisations
    ├── training.py         # Model training & evaluation
    └── prediction.py       # Single & batch prediction
```

## 📄 Pages

| Page | Description |
|------|-------------|
| **Login / Register** | Secure auth with PBKDF2-SHA256 hashed passwords |
| **Home** | Dataset overview, schema, business context |
| **EDA** | KPIs, interactive charts, business insights |
| **Training** | Train 5 models, compare metrics, view curves |
| **Prediction** | Single customer or batch CSV prediction |

## 🤖 Models

- Logistic Regression
- Random Forest
- Gradient Boosting
- Decision Tree
- XGBoost

Best model auto-selected by ROC-AUC and saved for prediction.

## ⚙️ Requirements

- Python 3.10+
- Streamlit 1.40+
- Pandas	2.x
- NumPy	1.x
- Plotly	5.x
- Scikit-Learn	1.x
- XGBoost	2.x
- SQLite3	Built-in
- Joblib	1.x
- Matplotlib	3.x
- Seaborn	0.13+

