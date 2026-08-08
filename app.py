import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Credit Card Churn Prediction",
    page_icon="💳",
    layout="centered"
)

# 1. Load trained models and transformers (.pkl files)
@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    transformer = joblib.load('transformer.pkl')
    return model, scaler, transformer

try:
    model, scaler, transformer = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info("Ensure model.pkl, scaler.pkl, and transformer.pkl are present in your GitHub repo.")
    st.stop()

# Title
st.title("💳 Customer Churn Prediction")
st.write("Enter the customer details below to calculate the churn probability.")

# 2. Input Form
with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3)

    with col2:
        balance = st.number_input("Balance ($)", min_value=0.0, value=60000.0, step=1000.0)
        num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0)

    submit_button = st.form_submit_button("Predict Churn")

# 3. Processing and Prediction
if submit_button:
    # Convert binary dropdowns to 1/0
    has_card = 1 if has_cr_card == "Yes" else 0
    active_member = 1 if is_active_member == "Yes" else 0

    # Create initial raw DataFrame matching training column names
    raw_input = pd.DataFrame([{
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_card,
        'IsActiveMember': active_member,
        'EstimatedSalary': estimated_salary
    }])

    try:
        # Step A: Apply Transformer (OneHotEncoder / ColumnTransformer)
        if hasattr(transformer, "transform"):
            transformed_data = transformer.transform(raw_input)
        else:
            transformed_data = raw_input

        # Step B: Apply Scaler
        if hasattr(scaler, "transform"):
            processed_input = scaler.transform(transformed_data)
        else:
            processed_input = transformed_data

        # Step C: Predict Probability
        if hasattr(model, "predict_proba"):
            # For Scikit-Learn models like RandomForest or XGBoost
            prediction = model.predict_proba(processed_input)[0][1]
        else:
            # For Neural Networks or direct probability estimators
            prediction = model.predict(processed_input)[0]
            if isinstance(prediction, (np.ndarray, list)):
                prediction = prediction[0]

        churn_prob = float(prediction) * 100

        # Display Results
        st.markdown("---")
        st.subheader("Prediction Results")

        col_res1, col_res2 = st.columns([1, 1])

        with col_res1:
            if churn_prob > 50.0:
                st.error("⚠️ **Customer is Likely to Churn**")
            else:
                st.success("✅ **Customer is Likely to Stay**")

        with col_res2:
            st.metric(label="Calculated Churn Probability", value=f"{churn_prob:.1f}%")

    except Exception as err:
        st.error(f"Error during feature processing: {err}")
        st.warning("Ensure feature column names in app.py match the ones used in model training.")