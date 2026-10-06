import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customers.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Dataset
df = pd.read_csv(DATA_PATH)
features = [
    "tenure_days",
    "num_orders",
    "avg_order_value",
    "total_spend",
    "recency_days",
    "num_returns",
    "discount_usage_rate",
    "support_tickets",
    "product_categories",
    "mobile_user",
]
X = df[features]
y = df["churn"]

# 2. Train-Test Split (Stratified 75/25)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Model Training
# Primary Model: Balanced Random Forest
rf_pipeline = Pipeline([
    ("scale", StandardScaler()),
    (
        "model",
        RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
            max_depth=8,
        ),
    ),
])
rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)
rf_prob = rf_pipeline.predict_proba(X_test)[:, 1]

# Comparison Model: Gradient Boosting
gb_pipeline = Pipeline([
    ("scale", StandardScaler()),
    (
        "model",
        GradientBoostingClassifier(
            n_estimators=150,
            random_state=42,
            max_depth=4,
        ),
    ),
])
gb_pipeline.fit(X_train, y_train)
gb_pred = gb_pipeline.predict(X_test)
gb_prob = gb_pipeline.predict_proba(X_test)[:, 1]

# 4. Evaluation Metrics
rf_metrics = {
    "accuracy": float(accuracy_score(y_test, rf_pred)),
    "precision": float(precision_score(y_test, rf_pred, zero_division=0)),
    "recall": float(recall_score(y_test, rf_pred, zero_division=0)),
    "f1": float(f1_score(y_test, rf_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, rf_prob)),
}

gb_metrics = {
    "accuracy": float(accuracy_score(y_test, gb_pred)),
    "precision": float(precision_score(y_test, gb_pred, zero_division=0)),
    "recall": float(recall_score(y_test, gb_pred, zero_division=0)),
    "f1": float(f1_score(y_test, gb_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, gb_prob)),
}

all_metrics = {
    "Random Forest": rf_metrics,
    "Gradient Boosting": gb_metrics,
}

# 5. Save Primary Model
joblib.dump(rf_pipeline, MODELS_DIR / "churn_model.joblib")

# 6. Save Metrics JSON
with open(OUTPUTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, indent=2)

# 7. Compute & Save Feature Importance
rf_model = rf_pipeline.named_steps["model"]
fi_df = pd.DataFrame({
    "feature": features,
    "importance": rf_model.feature_importances_,
}).sort_values("importance", ascending=False)
fi_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)

# 8. Generate Visualizations
# A. Feature Importance Plot
plt.figure(figsize=(9, 5))
sns.barplot(
    data=fi_df,
    x="importance",
    y="feature",
    palette="Blues_r",
    hue="feature",
    legend=False,
)
plt.title("Random Forest Feature Importance", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Importance Score", fontsize=11)
plt.ylabel("Feature", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=300)
plt.close()

# B. Confusion Matrix Plot
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Retained (0)", "Churned (1)"],
    yticklabels=["Retained (0)", "Churned (1)"],
    cbar=False,
)
plt.title("Random Forest Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Predicted Class", fontsize=11)
plt.ylabel("Actual Class", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=300)
plt.close()

# C. Churn Distribution Plot
plt.figure(figsize=(6, 4.5))
churn_counts = df["churn"].value_counts().rename({0: "Retained (0)", 1: "Churned (1)"})
ax = sns.barplot(
    x=churn_counts.index,
    y=churn_counts.values,
    palette=["#2b6cb0", "#c53030"],
    hue=churn_counts.index,
    legend=False,
)
plt.title("Customer Churn Distribution", fontsize=13, fontweight="bold", pad=12)
plt.ylabel("Customer Count", fontsize=11)
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 7),
        textcoords="offset points",
        fontsize=10,
    )
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "churn_distribution.png", dpi=300)
plt.close()

# 9. Update Project Metadata
metadata = {
    "dataset_type": "synthetic demonstration dataset",
    "customers": len(df),
    "churn_rate": float(df["churn"].mean()),
    "best_model": "Random Forest",
    "metrics": all_metrics,
}
with open(OUTPUTS_DIR / "project_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

# 10. Generate Sample Predictions
sample_df = df.head(50).copy()
sample_X = sample_df[features]
sample_probs = rf_pipeline.predict_proba(sample_X)[:, 1]
sample_df["churn_probability"] = sample_probs
sample_df["risk_level"] = [
    "High" if p >= 0.66 else ("Medium" if p >= 0.33 else "Low")
    for p in sample_probs
]
sample_df["retention_action"] = [
    "Personalized offer + follow-up"
    if r == "High"
    else ("Loyalty reminder" if r == "Medium" else "Regular engagement")
    for r in sample_df["risk_level"]
]
sample_df.to_csv(OUTPUTS_DIR / "sample_predictions.csv", index=False)

print("Training and artifact generation complete!")
print("Random Forest Metrics:", rf_metrics)
print("Gradient Boosting Metrics:", gb_metrics)

