import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration & Custom CSS
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Load Pre-trained Artifacts
@st.cache_resource
def load_artifacts():
    trf1 = joblib.load("transformer.pkl")
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
    return trf1, scaler, model

try:
    trf1, scaler, model = load_artifacts()
except Exception as e:
    st.error(f"Error loading saved model artifacts: {e}")
    st.info("Ensure 'transformer.pkl', 'scaler.pkl', and 'model.pkl' exist in your working directory.")
    st.stop()

# 3. Sample Data Logic
def fill_sample_data(sample_type="default"):
    if sample_type == "churn":
        st.session_state.form_data = {
            'CreditScore': 502,
            'Geography': 'Germany',
            'Gender': 'Female',
            'Age': 42,
            'Tenure': 8,
            'Balance': 159660.80,
            'NumOfProducts': 3,
            'HasCrCard': 1,
            'IsActiveMember': 0,
            'EstimatedSalary': 113931.57,
            'Complain': 1,
            'Satisfaction Score': 2,
            'Card Type': 'DIAMOND',
            'Point Earned': 320
        }
    else:
        st.session_state.form_data = {
            'CreditScore': 650,
            'Geography': 'France',
            'Gender': 'Female',
            'Age': 35,
            'Tenure': 5,
            'Balance': 50000.0,
            'NumOfProducts': 1,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 75000.0,
            'Complain': 0,
            'Satisfaction Score': 3,
            'Card Type': 'GOLD',
            'Point Earned': 500
        }

if 'form_data' not in st.session_state:
    fill_sample_data("default")

# 4. Interface Header
st.title("💳 Customer Churn Analytics & Prediction")
st.write("Predict churn risk using machine learning based on customer activity metrics.")

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("🎲 Auto-Fill Sample Data"):
        fill_sample_data("churn")

st.divider()

# 5. Input Form Layout
with st.form("churn_form"):
    st.subheader("Customer Demographic & Financial Details")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        credit_score = st.number_input("Credit Score", 300, 850, value=st.session_state.form_data['CreditScore'])
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"], index=["France", "Spain", "Germany"].index(st.session_state.form_data['Geography']))
        gender = st.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(st.session_state.form_data['Gender']))
        age = st.number_input("Age", 18, 100, value=st.session_state.form_data['Age'])
        tenure = st.slider("Tenure (Years)", 0, 10, value=st.session_state.form_data['Tenure'])

    with c2:
        balance = st.number_input("Account Balance ($)", 0.0, 300000.0, value=st.session_state.form_data['Balance'])
        num_products = st.slider("Number of Products", 1, 4, value=st.session_state.form_data['NumOfProducts'])
        has_card = st.selectbox("Has Credit Card?", [0, 1], index=st.session_state.form_data['HasCrCard'], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active = st.selectbox("Is Active Member?", [0, 1], index=st.session_state.form_data['IsActiveMember'], format_func=lambda x: "Yes" if x == 1 else "No")
        salary = st.number_input("Estimated Salary ($)", 0.0, 250000.0, value=st.session_state.form_data['EstimatedSalary'])

    with c3:
        complain = st.selectbox("Customer Complain Filed?", [0, 1], index=st.session_state.form_data['Complain'], format_func=lambda x: "Yes" if x == 1 else "No")
        satisfaction = st.slider("Satisfaction Score", 1, 5, value=st.session_state.form_data['Satisfaction Score'])
        card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"], index=["DIAMOND", "GOLD", "SILVER", "PLATINUM"].index(st.session_state.form_data['Card Type']))
        points = st.number_input("Points Earned", 0, 1000, value=st.session_state.form_data['Point Earned'])

    submit = st.form_submit_button("Predict Churn Status", use_container_width=True)

# 6. Processing & Prediction Logic
if submit:
    input_data = pd.DataFrame([{
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': salary,
        'Complain': complain,
        'Satisfaction Score': satisfaction,
        'Card Type': card_type,
        'Point Earned': points
    }])

    # Step A: One-Hot Transform
    input_trf = trf1.transform(input_data)

    # Step B: Standard Scaling
    input_scaled = scaler.transform(input_trf)

    # Step C: Re-align columns safely for model entry
    if hasattr(trf1, "get_feature_names_out"):
        cols = trf1.get_feature_names_out()
        input_scaled_df = pd.DataFrame(input_scaled, columns=cols)
    else:
        input_scaled_df = input_scaled

    # Step D: Safe Prediction & Probability Extraction
    prediction = int(model.predict(input_scaled_df)[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_scaled_df)[0][1])
    elif hasattr(model, "decision_function"):
        dec_score = float(model.decision_function(input_scaled_df)[0])
        probability = float(1 / (1 + np.exp(-dec_score)))
    else:
        probability = float(prediction)

    # Output Visuals
    st.divider()
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if prediction == 1:
            st.error("⚠️ **High Risk of Churn**")
            st.write("This customer is likely to leave the service.")
        else:
            st.success("✅ **Low Risk of Churn**")
            st.write("This customer is likely to stay retained.")

    with res_col2:
        st.metric(label="Churn Probability", value=f"{probability * 100:.2f}%")
        st.progress(max(0.0, min(probability, 1.0)))