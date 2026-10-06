# AI-Based E-Commerce Customer Churn Prediction and Retention Recommendation System

An AI/ML decision-support system designed to identify at-risk e-commerce customers from behavioral data and prescribe targeted retention strategies.

---

## 🚀 Quick Start Guide

### 1. Environment Setup & Dependency Installation
Open PowerShell in the project root directory and run:
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Model & Generate Artifacts
```powershell
python train_model.py
```
This trains the balanced Random Forest classifier, benchmarks Gradient Boosting, and automatically generates:
- `models/churn_model.joblib`: Serialized inference pipeline (StandardScaler + Balanced Random Forest)
- `outputs/metrics.json`: Test-set evaluation metrics
- `outputs/feature_importance.csv` & `feature_importance.png`: Behavioral driver rankings
- `outputs/confusion_matrix.png`: Model classification performance heatmap
- `outputs/churn_distribution.png`: Class balance visualization
- `outputs/sample_predictions.csv`: Sample customer inferences with risk levels and retention actions

### 3. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Or directly using the virtual environment:
```powershell
.\venv\Scripts\streamlit run app.py
```
Open your browser at: `http://localhost:8501`

---

## 🌟 Key Application Features

1. **Behavioral Customer Profiling & Real-Time Churn Predictor:**
   - 10 core behavioral parameters: Tenure, Order Count, Average Order Value (AOV), Total Spend, Recency (inactivity), Returns, Discount Usage, Support Tickets, Categories Explored, Primary Device.
   - Quick Archetype Presets (🔴 High Risk, 🟢 Loyal Active, 🟡 Medium Risk, Custom).
   - Real-time churn probability gauge with color-coded risk levels:
     - 🟢 **Low Risk (< 33%)**: Retained / Healthy
     - 🟡 **Medium Risk (33% – 66%)**: Early Attrition Warning
     - 🔴 **High Risk (≥ 66%)**: Critical Defection Alert
   - Dynamic Behavioral Friction Alerts (detects excessive inactivity, repeated returns, multiple complaints, or high discount dependency).

2. **Tailored Retention Strategies:**
   - Targeted action plans mapped to each risk tier (incentives, retention cadence, preferred communication channels, customer care follow-up).

3. **Model Evaluation & Performance Benchmarking:**
   - Evaluated on hold-out test set (stratified 75/25 split, 375 test customers).
   - Full classification metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
   - Confusion matrix visualization and comparison with baseline Gradient Boosting.

4. **Feature Importance & Behavioral Driver Analysis:**
   - Breakdown of top predictive signals (`total_spend`, `recency_days`, `tenure_days`, `num_orders`).
   - Business interpretations of churn drivers.

5. **Batch Predictions & Dataset Explorer:**
   - Interactive table of sample customer inferences with risk filtering.
   - One-click CSV export of filtered results.

---

## 📊 Actual Model Performance Metrics (Test Set)

| Metric | Random Forest (Primary) | Gradient Boosting (Baseline) |
|---|---|---|
| **Accuracy** | **75.47%** | 80.53% |
| **Precision** | **48.76%** | 63.33% |
| **Recall** | **66.29%** | 42.70% |
| **F1-Score** | **0.5619** | 0.5101 |
| **ROC-AUC** | **0.7854** | 0.7762 |

*Note: Random Forest with `class_weight='balanced'` was chosen as the primary deployment model because it achieves significantly higher Recall (66.29% vs 42.70%) and higher ROC-AUC (0.7854 vs 0.7762), preventing high-value customer churn from going undetected.*

---

## 📂 Project Structure

```text
ecommerce_churn_ai_submission/
│
├── data/
│   └── customers.csv             # Customer behavioral dataset (1,500 records)
│
├── models/
│   └── churn_model.joblib        # Trained scikit-learn pipeline (Scaler + Random Forest)
│
├── outputs/
│   ├── churn_distribution.png    # Dataset target distribution plot
│   ├── confusion_matrix.png      # Test set confusion matrix heatmap
│   ├── feature_importance.csv    # Numerical feature importance scores
│   ├── feature_importance.png    # Feature importance bar chart
│   ├── metrics.json              # Model evaluation metrics
│   ├── project_metadata.json     # Project & dataset metadata
│   └── sample_predictions.csv    # Sample customer risk scores and retention actions
│
├── reports/
│   ├── final_report.docx         # Mini project report (Word)
│   └── final_report.pdf          # Mini project report (PDF)
│
├── app.py                        # Streamlit web application
├── train_model.py                # Model training and artifact generation pipeline
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation and run instructions
```

---

*Prototype note: bundled demonstration data is synthetic and is included to make this prototype runnable immediately. Replace it with an approved real dataset if your college requires a public real-world source.*
