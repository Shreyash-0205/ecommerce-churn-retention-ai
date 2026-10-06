import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & Global Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Retention Diagnostic & Churn Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_PATH = BASE_DIR / "models" / "churn_model.joblib"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"
FEATURE_IMP_PATH = BASE_DIR / "outputs" / "feature_importance.csv"
SAMPLE_PRED_PATH = BASE_DIR / "outputs" / "sample_predictions.csv"
DATA_PATH = BASE_DIR / "data" / "customers.csv"

FEATURES = [
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

# Custom CSS for Modern Analytics Dashboard Look
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .dashboard-header {
        padding: 1.2rem 1.8rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.15);
    }
    .dashboard-header h1 {
        color: #f8fafc;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .dashboard-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }
    .header-pill {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .form-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }

    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        text-align: left;
    }
    .kpi-title {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.2rem 0;
        letter-spacing: -0.03em;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #64748b;
    }

    .risk-banner-low {
        background: #f0fdf4;
        border: 2px solid #86efac;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1.5rem 0;
    }
    .risk-banner-med {
        background: #fefce8;
        border: 2px solid #fde047;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1.5rem 0;
    }
    .risk-banner-high {
        background: #fef2f2;
        border: 2px solid #fca5a5;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1.5rem 0;
    }

    .insight-row {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    .insight-badge {
        font-weight: 700;
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 6px;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .nba-card {
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
        border-left: 5px solid #2563eb;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-top: 0.8rem;
    }

    .model-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper Functions
def safe_dataframe(df, stretch=True):
    """Render dataframe cleanly without deprecation warnings."""
    try:
        st.dataframe(df, width="stretch" if stretch else "content")
    except TypeError:
        st.dataframe(df, use_container_width=stretch)


def safe_image(img, caption=None, stretch=True, width=None):
    """Render image cleanly without deprecation warnings."""
    kwargs = {}
    if caption:
        kwargs["caption"] = caption
    if width is not None:
        kwargs["width"] = width
        st.image(img, **kwargs)
        return
    try:
        st.image(img, width="stretch" if stretch else "content", **kwargs)
    except TypeError:
        st.image(img, use_container_width=stretch, **kwargs)


# ---------------------------------------------------------
# Load Cached Model and Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_churn_model():
    if not MODELS_PATH.exists():
        return None
    return joblib.load(MODELS_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    if not FEATURE_IMP_PATH.exists():
        return None
    return pd.read_csv(FEATURE_IMP_PATH)


@st.cache_data
def load_sample_predictions():
    if not SAMPLE_PRED_PATH.exists():
        return None
    return pd.read_csv(SAMPLE_PRED_PATH)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


model = load_churn_model()
metrics_data = load_metrics()
fi_data = load_feature_importance()

# ---------------------------------------------------------
# Top Dashboard Brand Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="dashboard-header">
        <span class="header-pill">Operational Decision Support</span>
        <h1>Customer Churn Diagnostic & Retention Engine</h1>
        <p>Real-time behavioral risk scoring, automated segment profiling, and prescriptive next best actions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        "Model file not found. Please run `python train_model.py` to generate `models/churn_model.joblib`."
    )
    st.stop()

# ---------------------------------------------------------
# Preset Archetype Data Definitions
# ---------------------------------------------------------
ARCHETYPES = {
    "🔴 At-Risk Churner": {
        "tenure": 116,
        "orders": 2,
        "aov": 673.0,
        "spend": 1230.0,
        "recency": 161,
        "returns": 1,
        "discount": 0.60,
        "tickets": 2,
        "categories": 1,
        "mobile": "Desktop",
    },
    "🟢 Loyal Active Customer": {
        "tenure": 780,
        "orders": 17,
        "aov": 600.0,
        "spend": 10750.0,
        "recency": 16,
        "returns": 0,
        "discount": 0.04,
        "tickets": 0,
        "categories": 5,
        "mobile": "Mobile",
    },
    "🟡 Cooling-Off Periodic Buyer": {
        "tenure": 225,
        "orders": 4,
        "aov": 504.0,
        "spend": 2118.0,
        "recency": 54,
        "returns": 1,
        "discount": 0.36,
        "tickets": 0,
        "categories": 7,
        "mobile": "Desktop",
    },
}

# Initialize session state for all inputs if not set
defaults = ARCHETYPES["🔴 At-Risk Churner"]
if "inp_tenure" not in st.session_state:
    st.session_state["inp_tenure"] = defaults["tenure"]
if "inp_orders" not in st.session_state:
    st.session_state["inp_orders"] = defaults["orders"]
if "inp_aov" not in st.session_state:
    st.session_state["inp_aov"] = defaults["aov"]
if "inp_spend" not in st.session_state:
    st.session_state["inp_spend"] = defaults["spend"]
if "inp_recency" not in st.session_state:
    st.session_state["inp_recency"] = defaults["recency"]
if "inp_returns" not in st.session_state:
    st.session_state["inp_returns"] = defaults["returns"]
if "inp_discount" not in st.session_state:
    st.session_state["inp_discount"] = defaults["discount"]
if "inp_tickets" not in st.session_state:
    st.session_state["inp_tickets"] = defaults["tickets"]
if "inp_categories" not in st.session_state:
    st.session_state["inp_categories"] = defaults["categories"]
if "inp_mobile" not in st.session_state:
    st.session_state["inp_mobile"] = defaults["mobile"]


def apply_archetype(name):
    cfg = ARCHETYPES[name]
    st.session_state["inp_tenure"] = cfg["tenure"]
    st.session_state["inp_orders"] = cfg["orders"]
    st.session_state["inp_aov"] = cfg["aov"]
    st.session_state["inp_spend"] = cfg["spend"]
    st.session_state["inp_recency"] = cfg["recency"]
    st.session_state["inp_returns"] = cfg["returns"]
    st.session_state["inp_discount"] = cfg["discount"]
    st.session_state["inp_tickets"] = cfg["tickets"]
    st.session_state["inp_categories"] = cfg["categories"]
    st.session_state["inp_mobile"] = cfg["mobile"]


# ---------------------------------------------------------
# Archetype Quick-Load Action Bar
# ---------------------------------------------------------
st.markdown("##### ⚡ Test Customer Profiles (Click to Load Test Scenarios):")
pcol1, pcol2, pcol3, pcol4 = st.columns([1, 1, 1, 0.8])
with pcol1:
    if st.button("🔴 Load High-Risk Profile", use_container_width=True):
        apply_archetype("🔴 At-Risk Churner")
        st.rerun()
with pcol2:
    if st.button("🟢 Load Loyal Profile", use_container_width=True):
        apply_archetype("🟢 Loyal Active Customer")
        st.rerun()
with pcol3:
    if st.button("🟡 Load Cooling-Off Profile", use_container_width=True):
        apply_archetype("🟡 Cooling-Off Periodic Buyer")
        st.rerun()
with pcol4:
    if st.button("🔄 Reset Default", use_container_width=True):
        apply_archetype("🔴 At-Risk Churner")
        st.rerun()

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main-Page 2-Column Customer Input Form
# ---------------------------------------------------------
st.markdown("### 📝 Customer Behavioral Profile Form")
st.caption(
    "Adjust customer behavior metrics across purchasing history and service engagement to evaluate churn risk."
)

with st.container():
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown("#### 💳 Purchasing History & Recency")
        tenure_days = st.slider(
            "Account Tenure (Days Active)",
            min_value=30,
            max_value=1000,
            value=st.session_state["inp_tenure"],
            key="inp_tenure",
            help="Total duration in days since customer registration.",
        )
        recency_days = st.slider(
            "Days Since Last Order (Recency)",
            min_value=1,
            max_value=300,
            value=st.session_state["inp_recency"],
            key="inp_recency",
            help="Inactivity window in days since last recorded purchase.",
        )
        num_orders = st.slider(
            "Total Completed Orders",
            min_value=1,
            max_value=40,
            value=st.session_state["inp_orders"],
            key="inp_orders",
            help="Total lifetime completed transactions.",
        )
        avg_order_value = st.number_input(
            "Average Order Value (₹)",
            min_value=100.0,
            max_value=5000.0,
            value=float(st.session_state["inp_aov"]),
            step=50.0,
            key="inp_aov",
            help="Mean basket value per transaction.",
        )
        total_spend = st.number_input(
            "Total Historical Spend (₹)",
            min_value=100.0,
            max_value=100000.0,
            value=float(st.session_state["inp_spend"]),
            step=100.0,
            key="inp_spend",
            help="Cumulative gross transaction value.",
        )

    with col_right:
        st.markdown("#### 📱 Engagement, Friction & Platform")
        discount_rate = st.slider(
            "Promotional Discount Ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state["inp_discount"]),
            step=0.05,
            format="%.2f",
            key="inp_discount",
            help="Fraction of past orders completed with discount codes (0.0 = none, 1.0 = all).",
        )
        num_returns = st.slider(
            "Returned Orders Count",
            min_value=0,
            max_value=10,
            value=st.session_state["inp_returns"],
            key="inp_returns",
            help="Total items or orders returned for refund/exchange.",
        )
        support_tickets = st.slider(
            "Customer Support Inquiries",
            min_value=0,
            max_value=10,
            value=st.session_state["inp_tickets"],
            key="inp_tickets",
            help="Number of service complaints or support tickets logged.",
        )
        categories = st.slider(
            "Product Categories Explored",
            min_value=1,
            max_value=8,
            value=st.session_state["inp_categories"],
            key="inp_categories",
            help="Catalog department diversity purchased across.",
        )
        mobile_device = st.selectbox(
            "Primary Access Platform",
            ["Mobile", "Desktop"],
            index=0 if st.session_state["inp_mobile"] == "Mobile" else 1,
            key="inp_mobile",
            help="Dominant device used for browsing and checkout.",
        )

# Large Analyze Customer Button
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
run_analysis = st.button(
    "🔍 Analyze Customer Retention Risk",
    type="primary",
    use_container_width=True,
    help="Evaluate current customer behavioral parameters through the trained ML pipeline.",
)

# ---------------------------------------------------------
# Model Inference Execution
# ---------------------------------------------------------
input_payload = {
    "tenure_days": tenure_days,
    "num_orders": num_orders,
    "avg_order_value": avg_order_value,
    "total_spend": total_spend,
    "recency_days": recency_days,
    "num_returns": num_returns,
    "discount_usage_rate": discount_rate,
    "support_tickets": support_tickets,
    "product_categories": categories,
    "mobile_user": 1 if mobile_device == "Mobile" else 0,
}
input_df = pd.DataFrame([input_payload])[FEATURES]
churn_probability = float(model.predict_proba(input_df)[0, 1])
risk_score = int(round(churn_probability * 100))

# Risk Tier Determination
if churn_probability >= 0.66:
    risk_level = "High"
    banner_class = "risk-banner-high"
    banner_title = "🔴 HIGH CHURN RISK — Critical Churn Risk"
    banner_desc = "The customer exhibits severe disengagement signals including prolonged order inactivity, customer care friction, or collapsing purchase frequency. Immediate intervention is recommended to mitigate customer churn."
    status_tag = "Critical Churn Risk"
    status_color = "#dc2626"
elif churn_probability >= 0.33:
    risk_level = "Medium"
    banner_class = "risk-banner-med"
    banner_title = "🟡 MEDIUM CHURN RISK — Early Inactivity Warning"
    banner_desc = "The customer is in a cooling-off transition phase with expanding purchase intervals or elevated discount dependency. Targeted re-engagement will restore ordering habits before churn occurs."
    status_tag = "Elevated Inactivity"
    status_color = "#d97706"
else:
    risk_level = "Low"
    banner_class = "risk-banner-low"
    banner_title = "🟢 LOW CHURN RISK — Stable & Engaged Customer"
    banner_desc = "The customer maintains healthy purchasing rhythm, recent interactions, and negligible friction. Standard brand nurturing and loyalty benefits are recommended to preserve baseline loyalty."
    status_tag = "Healthy Account"
    status_color = "#16a34a"

# Customer Behavioral Segmentation
if churn_probability >= 0.66:
    if support_tickets >= 2 or num_returns >= 2:
        segment_name = "Friction-Heavy Dissatisfied Account"
        segment_desc = "Customer churn driven by support tickets or product dissatisfaction."
    elif total_spend >= 8000:
        segment_name = "High-Value Churn Risk (At-Risk VIP)"
        segment_desc = "Historically high-spend account showing sudden churn signals."
    elif tenure_days <= 150:
        segment_name = "Early-Stage Abandoner"
        segment_desc = "Failed initial onboarding with weak brand attachment."
    else:
        segment_name = "Dormant Disengaged Shopper"
        segment_desc = "Prolonged inactivity with decaying purchase frequency."
elif churn_probability >= 0.33:
    if discount_rate >= 0.40:
        segment_name = "Price-Sensitive Bargain Seeker"
        segment_desc = "Orders primarily during promotional discounts and sales."
    elif recency_days >= 60:
        segment_name = "Cooling-Off Periodic Buyer"
        segment_desc = "Extended gap since last order; requires reactivation."
    else:
        segment_name = "Moderate-Engagement Core Shopper"
        segment_desc = "Stable volume but showing slight engagement slowdown."
else:
    if total_spend >= 10000:
        segment_name = "Loyal Brand Champion (VIP)"
        segment_desc = "High frequency, high monetary value, and recent orders."
    elif tenure_days >= 500:
        segment_name = "Established Core Patron"
        segment_desc = "Long-standing regular customer with dependable engagement."
    else:
        segment_name = "Promising Active Newcomer"
        segment_desc = "Recent joiner demonstrating consistent repeat orders."

# ---------------------------------------------------------
# Three Result Cards
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Churn Prediction Results")

card1, card2, card3 = st.columns(3)

with card1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Churn Probability</div>
            <div class="kpi-value" style="color: {status_color};">{churn_probability * 100:.1f}%</div>
            <div class="kpi-sub">Estimated statistical likelihood of churn</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with card2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Customer Risk Score</div>
            <div class="kpi-value">{risk_score} <span style="font-size:1.1rem; color:#94a3b8;">/ 100</span></div>
            <div class="kpi-sub">Normalized churn index ({status_tag})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with card3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Assigned Customer Segment</div>
            <div class="kpi-value" style="font-size:1.35rem; line-height:1.2; margin-top:0.45rem; color:#1e293b;">{segment_name}</div>
            <div class="kpi-sub" style="margin-top:0.35rem;">{segment_desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Large Visual Low/Medium/High Risk Indicator
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="{banner_class}">
        <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.35rem;">{banner_title}</div>
        <div style="font-size: 0.92rem; color: #334155; line-height: 1.5;">{banner_desc}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Visual Progress Gauge
st.write(
    f"**Risk Spectrum Position:** `{churn_probability * 100:.1f}%` (Thresholds: Low < 33.0% | Medium 33.0%–65.9% | High ≥ 66.0%)"
)
st.progress(churn_probability)

# ---------------------------------------------------------
# Customer Insights Section (3–5 Generated Insights)
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📋 Customer Behavioral Insights")
st.caption(
    "Automated behavioral diagnostics generated from real-time customer feature analysis:"
)

insights = []

# Insight 1: Recency & Inactivity Check
if recency_days > 90:
    insights.append((
        "CRITICAL INACTIVITY",
        "#dc2626",
        "#fee2e2",
        f"**Prolonged Inactivity ({recency_days} days):** The customer has not placed an order in over 3 months, exceeding the healthy e-commerce re-order window (typically 30–45 days) by {recency_days - 45} days.",
    ))
elif recency_days > 45:
    insights.append((
        "EXPANDING GAP",
        "#d97706",
        "#fef3c7",
        f"**Cooling-Off Inactivity ({recency_days} days):** Order gap is beginning to widen beyond the standard 45-day cycle, indicating early disengagement.",
    ))
else:
    insights.append((
        "ACTIVE RECENCY",
        "#16a34a",
        "#dcfce7",
        f"**Fresh Engagement ({recency_days} days ago):** The account has made a recent purchase, reflecting current platform habituation.",
    ))

# Insight 2: Order Frequency & Velocity
months_tenure = max(tenure_days / 30.0, 1.0)
order_velocity = num_orders / months_tenure
if order_velocity < 0.5:
    insights.append((
        "LOW FREQUENCY",
        "#dc2626",
        "#fee2e2",
        f"**Sluggish Purchase Velocity:** Averaging only **{order_velocity:.2f} orders/month** ({num_orders} orders over {tenure_days} days active), signaling weak habit formation.",
    ))
elif order_velocity >= 1.5:
    insights.append((
        "HIGH CADENCE",
        "#16a34a",
        "#dcfce7",
        f"**Strong Purchase Velocity:** Consistently ordering at **{order_velocity:.2f} orders/month** ({num_orders} orders completed), establishing strong loyalty stickiness.",
    ))
else:
    insights.append((
        "STEADY CADENCE",
        "#2563eb",
        "#dbeafe",
        f"**Moderate Purchase Frequency:** Placing approximately **{order_velocity:.2f} orders/month**, indicating regular but periodic purchasing habits.",
    ))

# Insight 3: Friction & Customer Care
if support_tickets >= 2 or num_returns >= 2:
    insights.append((
        "FRICTION ALERT",
        "#dc2626",
        "#fee2e2",
        f"**Service Friction Indicator:** Recorded **{support_tickets} support tickets** and **{num_returns} product returns**, indicating service friction that directly elevates churn probability.",
    ))
elif support_tickets == 1 or num_returns == 1:
    insights.append((
        "MINOR FRICTION",
        "#d97706",
        "#fef3c7",
        f"**Minor Service Touchpoint:** Recorded {support_tickets} ticket and {num_returns} return. Experience requires monitoring to ensure satisfaction.",
    ))
else:
    insights.append((
        "FRICTION-FREE",
        "#16a34a",
        "#dcfce7",
        "**Clean Customer Journey:** Zero recorded support complaints or returned merchandise, representing friction-free satisfaction.",
    ))

# Insight 4: Price & Promotional Sensitivity
if discount_rate >= 0.50:
    insights.append((
        "PRICE SENSITIVE",
        "#d97706",
        "#fef3c7",
        f"**Heavy Deal Dependency:** **{discount_rate * 100:.0f}% of transactions** used promotional discounts, highlighting high sensitivity to pricing changes and discount withdrawal.",
    ))
elif discount_rate <= 0.15:
    insights.append((
        "ORGANIC BUYER",
        "#16a34a",
        "#dcfce7",
        f"**Organic Value Loyalty:** Only **{discount_rate * 100:.0f}% discount usage**, indicating the customer purchases based on product value rather than promotional incentives.",
    ))
else:
    insights.append((
        "BALANCED PRICING",
        "#2563eb",
        "#dbeafe",
        f"**Balanced Promotion Affinity:** Utilizes discounts on **{discount_rate * 100:.0f}% of purchases**, standard for regular retail consumers.",
    ))

# Insight 5: Monetary Value & Basket Quality
if total_spend >= 8000:
    insights.append((
        "HIGH-SPEND ACCOUNT",
        "#16a34a",
        "#dcfce7",
        f"**Substantial Financial Contribution:** Lifetime spend of **₹{total_spend:,.2f}** with an AOV of **₹{avg_order_value:,.2f}**, categorizing this customer in the top revenue deciles.",
    ))
else:
    insights.append((
        "STANDARD SPEND",
        "#64748b",
        "#f1f5f9",
        f"**Moderate Spending Base:** Lifetime spend of **₹{total_spend:,.2f}** across **{categories} product categories**, indicating room for cross-category growth.",
    ))

for tag, text_color, bg_color, text in insights:
    st.markdown(
        f"""
        <div class="insight-row">
            <span class="insight-badge" style="color: {text_color}; background: {bg_color};">{tag}</span>
            <div style="font-size: 0.92rem; color: #1e293b; line-height: 1.45;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Next Best Action Section (One Recommendation & Explanation)
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 🎯 Retention Recommendation")

if risk_level == "High":
    action_title = (
        "🚨 High-Priority Win-Back Campaign & Proactive Care Outreach"
    )
    action_strategy = "Deploy an assertive 20% reactivation voucher with free delivery, paired with a proactive customer success call to resolve previous friction points."
    action_channel = "Direct SMS + High-Priority Email + Assigned Care Specialist Follow-Up"
    action_timeframe = "Execute within 24 to 48 Hours"
    action_rationale = (
        f"The model estimates a **{churn_probability * 100:.1f}% churn probability** driven primarily by extended inactivity ({recency_days} days) and service friction ({support_tickets} tickets, {num_returns} returns). "
        "The recommendation prioritizes rapid re-engagement because the model indicates a high likelihood of churn."
    )
elif risk_level == "Medium":
    action_title = (
        "⚠️ Targeted Catalog Re-Engagement & Category Bundle Incentive"
    )
    action_strategy = f"Deliver an automated dynamic email digest showcasing trending items across their top explored categories ({categories} categories), supplemented by a 10% next-order reward boost."
    action_channel = "Personalized App Push Notification + Bi-Weekly Curated Email"
    action_timeframe = "Deploy within 5 to 7 Days"
    action_rationale = (
        f"With a **{churn_probability * 100:.1f}% churn probability**, this customer sits in the critical transition zone. "
        "Aggressive price discounting would needlessly compress product gross margins; instead, intelligent merchandising recommendations re-ignite purchase cadence while maintaining price integrity."
    )
else:
    action_title = "✅ Brand Advocacy Cultivation & VIP Early-Access Nurture"
    action_strategy = "Enroll into the Premium Loyalty Tier, offering zero-cost early access to seasonal product drops and an invite to the referral rewards program."
    action_channel = (
        "In-App Loyalty Hub + Standard Monthly Product Digest"
    )
    action_timeframe = "Maintain on Regular Bi-Weekly Schedule"
    action_rationale = (
        f"This customer maintains a low churn probability of **{churn_probability * 100:.1f}%**. "
        "Promotional discounts are counter-productive here because customer satisfaction is already strong. Protecting margins while reinforcing VIP recognition drives word-of-mouth referral acquisition."
    )

st.markdown(
    f"""
    <div class="nba-card">
        <h4 style="color: #1e3a8a; margin-top: 0; font-size: 1.18rem; font-weight: 700;">{action_title}</h4>
        <p style="font-size: 0.95rem; color: #1e293b; margin-bottom: 0.6rem;">
            <strong>Recommended Action:</strong> {action_strategy}
        </p>
        <div style="display: flex; gap: 24px; flex-wrap: wrap; font-size: 0.88rem; color: #475569; margin-bottom: 0.8rem;">
            <div><strong>Preferred Channel:</strong> {action_channel}</div>
            <div><strong>Execution Window:</strong> <span style="color:#2563eb; font-weight:600;">{action_timeframe}</span></div>
        </div>
        <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 14px; font-size: 0.88rem; color: #334155; line-height: 1.5;">
            <strong>Strategic Business Rationale:</strong><br>{action_rationale}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Separate Model Insights Section
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📈 Machine Learning Model Insights & Validation")
st.caption(
    "Architecture transparency, behavioral feature ranking, and hold-out test set performance."
)

col_mod_left, col_mod_right = st.columns([1.1, 1], gap="large")

with col_mod_left:
    st.markdown("#### 🎯 Relative Feature Importance")
    if fi_data is not None and not fi_data.empty:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        sns.barplot(
            data=fi_data,
            x="importance",
            y="feature",
            palette="Blues_r",
            hue="feature",
            legend=False,
            ax=ax,
        )
        ax.set_title(
            "Gini Impurity Feature Weights (Random Forest)",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Relative Predictive Weight", fontsize=9)
        ax.set_ylabel("Behavioral Feature", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Top drivers: **`total_spend` (21.6%)** and **`recency_days` (20.7%)** constitute over 42% of total predictive power."
        )
    else:
        st.info("Feature importance data not found.")

with col_mod_right:
    st.markdown("#### 📊 Evaluation Benchmarks & Confusion Matrix")
    if metrics_data:
        rf_m = metrics_data.get(
            "Random Forest",
            metrics_data.get("accuracy") and metrics_data or {},
        )
        gb_m = metrics_data.get("Gradient Boosting", None)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{rf_m.get('accuracy', 0.0) * 100:.1f}%")
        m2.metric("Recall", f"{rf_m.get('recall', 0.0) * 100:.1f}%")
        m3.metric("F1-Score", f"{rf_m.get('f1', 0.0):.3f}")
        m4.metric("ROC-AUC", f"{rf_m.get('roc_auc', 0.0):.3f}")

        cm_path = BASE_DIR / "outputs" / "confusion_matrix.png"
        if cm_path.exists():
            safe_image(
                str(cm_path),
                caption="Confusion Matrix on Hold-Out Test Set (375 instances)",
                stretch=True,
            )

        if gb_m:
            bench_df = pd.DataFrame([
                {
                    "Model": "Random Forest (Primary)",
                    "Accuracy": f"{rf_m['accuracy'] * 100:.1f}%",
                    "Recall": f"{rf_m['recall'] * 100:.1f}%",
                    "ROC-AUC": f"{rf_m['roc_auc']:.3f}",
                },
                {
                    "Model": "Gradient Boosting (Baseline)",
                    "Accuracy": f"{gb_m['accuracy'] * 100:.1f}%",
                    "Recall": f"{gb_m['recall'] * 100:.1f}%",
                    "ROC-AUC": f"{gb_m['roc_auc']:.3f}",
                },
            ])
            safe_dataframe(bench_df, stretch=True)
            st.caption(
                "Note: Random Forest with `class_weight='balanced'` was selected as the deployment model due to higher churn Recall (66.3% vs 42.7%)."
            )
    else:
        st.info("Metrics data not found. Please run `python train_model.py`.")

# ---------------------------------------------------------
# Optional Batch CSV Analysis Section
# ---------------------------------------------------------
st.markdown("---")
with st.expander("📁 Batch Customer Analysis & Dataset Diagnostics", expanded=False):
    st.markdown("#### Batch Account Scoring & Cohort Overview")
    st.write(
        "Upload a customer behavioral CSV file or inspect pre-calculated sample customer predictions."
    )

    batch_file = st.file_uploader(
        "Upload Custom Customer Batch CSV", type=["csv"]
    )
    sample_df = load_sample_predictions()

    if batch_file is not None:
        try:
            uploaded_df = pd.read_csv(batch_file)
            missing = [f for f in FEATURES if f not in uploaded_df.columns]
            if missing:
                st.error(f"Uploaded CSV is missing required columns: {missing}")
            else:
                probs = model.predict_proba(uploaded_df[FEATURES])[:, 1]
                uploaded_df["churn_probability"] = probs
                uploaded_df["risk_level"] = [
                    "High" if p >= 0.66 else ("Medium" if p >= 0.33 else "Low")
                    for p in probs
                ]
                st.success(
                    f"Successfully scored {len(uploaded_df)} accounts from uploaded file!"
                )
                safe_dataframe(uploaded_df, stretch=True)
                csv_bytes = uploaded_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Scored Batch Predictions CSV",
                    data=csv_bytes,
                    file_name="batch_scored_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
    elif sample_df is not None:
        st.markdown("##### Demonstration Cohort (50 Sample Accounts)")
        filter_tier = st.selectbox(
            "Filter Sample Accounts by Risk Tier",
            ["All Tiers", "High", "Medium", "Low"],
        )
        if filter_tier != "All Tiers":
            filtered_sample = sample_df[sample_df["risk_level"] == filter_tier]
        else:
            filtered_sample = sample_df

        safe_dataframe(filtered_sample, stretch=True)
        sample_csv = filtered_sample.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Demonstration Inferences CSV",
            data=sample_csv,
            file_name="sample_cohort_predictions.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "AI-Based E-Commerce Customer Churn Prediction and Retention Recommendation System | "
    "Demonstration prototype built for academic review. Bundled demonstration dataset is synthetic."
)
