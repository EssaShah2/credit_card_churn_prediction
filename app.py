import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Optional TensorFlow import check
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# 1. Page Setup
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
    .main { padding: 2rem 2rem; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# 2. Artifact Loader
@st.cache_resource
def load_artifacts():
    trf1 = joblib.load("transformer.pkl")
    scaler = joblib.load("scaler.pkl")
    
    # Check if model is saved via Keras (.keras/.h5) or joblib (.pkl)
    try:
        if TF_AVAILABLE:
            try:
                model = tf.keras.models.load_model("churn_model.keras")
            except Exception:
                model = tf.keras.models.load_model("churn_model.h5")
        else:
            model = joblib.load("model.pkl")
    except Exception:
        model = joblib.load("model.pkl")
        
    return trf1, scaler, model

try:
    trf1, scaler, model = load_artifacts()
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.info("Ensure preprocessor (.pkl) and model files exist in the project directory.")
    st.stop()

# 3. Session State & Form Defaults
defaults = {
    'credit_score': 650, 'geography': 'France', 'gender': 'Female',
    'age': 35, 'tenure': 5, 'balance': 50000.0, 'num_products': 1,
    'has_card': 1, 'is_active': 1, 'salary': 75000.0,
    'complain': 0, 'satisfaction': 3, 'card_type': 'GOLD', 'points': 500
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

def fill_high_risk_sample():
    st.session_state.credit_score = 502
    st.session_state.geography = 'Germany'
    st.session_state.gender = 'Female'
    st.session_state.age = 52
    st.session_state.tenure = 2
    st.session_state.balance = 125000.0
    st.session_state.num_products = 3
    st.session_state.has_card = 1
    st.session_state.is_active = 0
    st.session_state.salary = 113931.0
    st.session_state.complain = 1
    st.session_state.satisfaction = 1
    st.session_state.card_type = 'DIAMOND'
    st.session_state.points = 200

# 4. Interface Header
st.title("💳 Customer Churn Analytics & Prediction")
st.write("Predict churn risk using machine learning based on customer activity metrics.")

col_btn1, _ = st.columns([1, 4])
with col_btn1:
    st.button("🎲 Auto-Fill High Risk Sample", on_click=fill_high_risk_sample)

st.divider()

# 5. Form Input Layout
with st.form("churn_form"):
    st.subheader("Customer Demographic & Financial Details")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.number_input("Credit Score", 300, 850, key="credit_score")
        st.selectbox("Geography", ["France", "Spain", "Germany"], key="geography")
        st.selectbox("Gender", ["Female", "Male"], key="gender")
        st.number_input("Age", 18, 100, key="age")
        st.slider("Tenure (Years)", 0, 10, key="tenure")

    with c2:
        st.number_input("Account Balance ($)", 0.0, 300000.0, key="balance")
        st.slider("Number of Products", 1, 4, key="num_products")
        st.selectbox("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="has_card")
        st.selectbox("Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="is_active")
        st.number_input("Estimated Salary ($)", 0.0, 250000.0, key="salary")

    with c3:
        st.selectbox("Customer Complain Filed?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="complain")
        st.slider("Satisfaction Score", 1, 5, key="satisfaction")
        st.selectbox("Card Type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"], key="card_type")
        st.number_input("Points Earned", 0, 1000, key="points")

    submit = st.form_submit_button("Predict Churn Status", use_container_width=True)

# 6. Pipeline Processing & Inference
if submit:
    # Build raw input DataFrame matching training feature structure
    input_df = pd.DataFrame([{
        'CreditScore': st.session_state.credit_score,
        'Geography': st.session_state.geography,
        'Gender': st.session_state.gender,
        'Age': st.session_state.age,
        'Tenure': st.session_state.tenure,
        'Balance': st.session_state.balance,
        'NumOfProducts': st.session_state.num_products,
        'HasCrCard': st.session_state.has_card,
        'IsActiveMember': st.session_state.is_active,
        'EstimatedSalary': st.session_state.salary,
        'Complain': st.session_state.complain,
        'Satisfaction Score': st.session_state.satisfaction,
        'Card Type': st.session_state.card_type,
        'Point Earned': st.session_state.points
    }])

    # Preprocessing pipeline
    transformed_data = trf1.transform(input_df)
    scaled_data = scaler.transform(transformed_data)
    arr_data = np.asarray(scaled_data, dtype=np.float32)

    # Safe Keras/Scikit prediction handling
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(arr_data)[0][1])
    else:
        # Keras models output probability tensor via .predict()
        raw_pred = model.predict(arr_data, verbose=0)
        probability = float(np.ravel(raw_pred)[0])

    prediction = 1 if probability >= 0.5 else 0

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