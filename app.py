import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Page Setup
st.set_page_config(
    page_title="Customer Satisfaction Predictor - MLOps",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3A1C71, #D76D77, #FFAF7B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .badge-star-5 { background-color: #059669; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
    .badge-star-4 { background-color: #10B981; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
    .badge-star-3 { background-color: #F59E0B; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
    .badge-star-2 { background-color: #EF4444; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
    .badge-star-1 { background-color: #991B1B; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# App Title & Header
st.markdown('<div class="main-header">🛍️ Customer Satisfaction MLOps Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production Machine Learning Pipeline with ZenML, MLflow & Design Patterns</div>', unsafe_allow_html=True)

# Helper function to load model
@st.cache_resource
def load_deployed_model():
    model_dir = "deployed_model"
    model_path = os.path.join(model_dir, "satisfaction_model.pkl")
    feature_path = os.path.join(model_dir, "feature_names.pkl")
    
    if os.path.exists(model_path) and os.path.exists(feature_path):
        model = joblib.load(model_path)
        feature_names = joblib.load(feature_path)
        return model, feature_names
    else:
        # Fallback to locally trained models folder
        fallback_path = os.path.join("models", "lightgbm_model.pkl")
        if os.path.exists(fallback_path):
            model = joblib.load(fallback_path)
            feature_names = [
                "payment_sequential", "payment_installments", "payment_value", "price",
                "freight_value", "product_name_lenght", "product_description_lenght",
                "product_photos_qty", "product_weight_g", "product_length_cm",
                "product_height_cm", "product_width_cm", "delivery_days",
                "estimated_delivery_days", "delay_days", "product_volume_cm3"
            ]
            return model, feature_names
    return None, None

model, feature_names = load_deployed_model()

# Sidebar: E-commerce Order Input Parameters
st.sidebar.header("📦 Order & Logistics Parameters")

payment_value = st.sidebar.number_input("Payment Value ($)", min_value=10.0, max_value=2000.0, value=150.0, step=10.0)
freight_value = st.sidebar.number_input("Freight Value ($)", min_value=2.0, max_value=300.0, value=25.0, step=2.0)
price = max(5.0, payment_value - freight_value)

payment_installments = st.sidebar.slider("Payment Installments", 1, 24, 3)
payment_sequential = st.sidebar.slider("Payment Sequence Count", 1, 5, 1)

st.sidebar.subheader("🚚 Delivery Metrics")
delivery_days = st.sidebar.slider("Actual Delivery Time (Days)", 1, 30, 7)
estimated_delivery_days = st.sidebar.slider("Estimated Delivery (Days)", 1, 40, 12)
delay_days = max(0, delivery_days - estimated_delivery_days)

st.sidebar.subheader("📦 Product Details")
product_weight_g = st.sidebar.number_input("Product Weight (grams)", 50, 15000, 1200, 50)
product_length_cm = st.sidebar.slider("Length (cm)", 5, 100, 25)
product_height_cm = st.sidebar.slider("Height (cm)", 2, 50, 15)
product_width_cm = st.sidebar.slider("Width (cm)", 5, 60, 20)
product_volume_cm3 = product_length_cm * product_height_cm * product_width_cm

product_photos_qty = st.sidebar.slider("Product Photos Quantity", 1, 10, 3)
product_name_lenght = 45
product_description_lenght = 500

# Prepare Feature DataFrame
input_dict = {
    "payment_sequential": payment_sequential,
    "payment_installments": payment_installments,
    "payment_value": payment_value,
    "price": price,
    "freight_value": freight_value,
    "product_name_lenght": product_name_lenght,
    "product_description_lenght": product_description_lenght,
    "product_photos_qty": product_photos_qty,
    "product_weight_g": product_weight_g,
    "product_length_cm": product_length_cm,
    "product_height_cm": product_height_cm,
    "product_width_cm": product_width_cm,
    "delivery_days": delivery_days,
    "estimated_delivery_days": estimated_delivery_days,
    "delay_days": delay_days,
    "product_volume_cm3": product_volume_cm3
}

input_df = pd.DataFrame([input_dict])
if feature_names:
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📊 Prediction Dashboard", "📈 Pipeline Performance", "⚙️ Architecture & Design Patterns"])

with tab1:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Predicted Review Score")

        if model is not None:
            raw_prediction = float(model.predict(input_df)[0])
            pred_score = float(np.clip(raw_prediction, 1.0, 5.0))

            # Gauge Meter Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Customer Rating (1-5 ⭐)", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [1, 5], 'tickwidth': 1},
                    'bar': {'color': "#6366F1"},
                    'steps': [
                        {'range': [1, 2], 'color': '#EF4444'},
                        {'range': [2, 3.5], 'color': '#F59E0B'},
                        {'range': [3.5, 5], 'color': '#10B981'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': pred_score
                    }
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Rating Badge & Text
            if pred_score >= 4.2:
                badge_html = '<span class="badge-star-5">⭐⭐⭐⭐⭐ Highly Satisfied</span>'
                status_desc = "Customer is extremely satisfied! Fast delivery and good value."
            elif pred_score >= 3.5:
                badge_html = '<span class="badge-star-4">⭐⭐⭐⭐ Satisfied</span>'
                status_desc = "Customer is satisfied with the order experience."
            elif pred_score >= 2.5:
                badge_html = '<span class="badge-star-3">⭐⭐⭐ Neutral</span>'
                status_desc = "Customer experience was average. Delivery delays or freight costs could be improved."
            else:
                badge_html = '<span class="badge-star-1">⭐ Unsatisfied / At Risk</span>'
                status_desc = "High likelihood of negative review due to delivery delays or high costs!"

            st.markdown(f"### Status: {badge_html}", unsafe_allow_html=True)
            st.info(status_desc)

        else:
            st.warning("⚠️ No trained model found. Please run `python run_pipeline.py` or `python run_deployment.py` to train and deploy the model.")

    with col2:
        st.subheader("Order Metrics Overview")
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Total Cost", f"${payment_value:.2f}", f"Freight: ${freight_value:.2f}")
        mcol2.metric("Delivery Speed", f"{delivery_days} days", f"Delay: {delay_days} days", delta_color="inverse" if delay_days > 0 else "normal")

        st.divider()
        st.markdown("#### Key Drivers Impacting Satisfaction:")
        st.markdown(f"- **Logistics Delay**: {'⚠️ Delayed by ' + str(delay_days) + ' days' if delay_days > 0 else '✅ Delivered on time'}")
        st.markdown(f"- **Freight Ratio**: {freight_value/payment_value:.1%} of total order value")
        st.markdown(f"- **Product Volume**: {product_volume_cm3/1000:.2f} liters")

with tab2:
    st.subheader("📈 Model Evaluation & Feature Importance")
    
    # Feature Importance Plot
    if model is not None and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        features = feature_names if feature_names else [f"Feature_{i}" for i in range(len(importances))]
        fi_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=True)

        fig_fi = px.bar(
            fi_df.tail(10),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Feature Importances (Tree Model)",
            color="Importance",
            color_continuous_scale="Viridis"
        )
        fig_fi.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.markdown("""
        **Pipeline Evaluation Metrics:**
        - **Algorithm**: LightGBM Regressor (Strategy Pattern)
        - **Target Metric**: R2 Score / Root Mean Squared Error (RMSE)
        - **Experiment Tracker**: MLflow Server Logging
        """)

with tab3:
    st.subheader("⚙️ Architecture & Design Patterns")
    st.markdown("""
    ### 🏗️ Software Engineering Design Patterns Implemented

    1. **Factory Design Pattern (`src/data_ingestion.py`)**:
       - `DataIngestorFactory` dynamically creates data readers (`ZipDataIngestor`, `CSVDataIngestor`) based on file format.
    
    2. **Strategy Design Pattern (`src/data_cleaning.py`, `src/model_dev.py`, `src/evaluation.py`)**:
       - **Data Strategy**: `DataPreProcessStrategy` and `DataDivideStrategy` decouple preprocessing logic from execution.
       - **Model Strategy**: `LinearRegressionModel`, `RandomForestModel`, `XGBoostModel`, `LightGBMModel` inherit from abstract `Model` base class.
       - **Evaluation Strategy**: `MSE`, `RMSE`, and `R2Score` implement decoupled evaluation metrics.
    
    3. **ZenML & MLflow Orchestration**:
       - Pipeline steps (`ingest_data`, `clean_data`, `model_train`, `evaluation`) are orchestrated end-to-end.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("🚀 Built for End-to-End MLOps Customer Satisfaction Prediction")
