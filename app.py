import numpy as np
import pandas as pd
import streamlit as st
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS STYLING ---
st.markdown(
    """
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


try:
    model, scaler = load_assets()
except Exception as e:
    st.error(
        f"⚠️ Could not load model/scaler files. Ensure 'model.pkl' and 'scaler.pkl' are in the directory. Error: {e}"
    )
    st.stop()


# --- PREDEFINED SAMPLE PROFILES FOR AUTO-FILL ---
SAMPLE_PROFILES = {
    "Custom Input": None,
    "High Risk Profile (Churn Likely)": {
        "CreditScore": 502,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 8,
        "Balance": 159660.80,
        "NumOfProducts": 3,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 113931.57,
        "Complain": 1,
        "Satisfaction Score": 3,
        "Card Type": "DIAMOND",
        "Point Earned": 377,
    },
    "Low Risk Profile (Loyal Customer)": {
        "CreditScore": 771,
        "Geography": "France",
        "Gender": "Male",
        "Age": 39,
        "Tenure": 5,
        "Balance": 0.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 96270.64,
        "Complain": 0,
        "Satisfaction Score": 4,
        "Card Type": "GOLD",
        "Point Earned": 750,
    },
}

# --- HEADER SECTION ---
st.title("💳 Customer Churn Prediction Dashboard")
st.caption(
    "Predict whether a banking customer is likely to churn using machine learning."
)
st.divider()

# --- AUTO-FILL SELECTION ---
st.sidebar.header("⚡ Quick Controls")
selected_profile = st.sidebar.selectbox(
    "Auto-Fill Test Samples:", options=list(SAMPLE_PROFILES.keys())
)

defaults = SAMPLE_PROFILES[selected_profile]

# --- MAIN FORM INPUTS ---
st.subheader("📋 Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=defaults["CreditScore"] if defaults else 650,
    )
    geography = st.selectbox(
        "Geography",
        options=["France", "Germany", "Spain"],
        index=(
            ["France", "Germany", "Spain"].index(defaults["Geography"])
            if defaults
            else 0
        ),
    )
    gender = st.selectbox(
        "Gender",
        options=["Female", "Male"],
        index=(
            ["Female", "Male"].index(defaults["Gender"]) if defaults else 0
        ),
    )
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=defaults["Age"] if defaults else 38,
    )
    tenure = st.slider(
        "Tenure (Years)",
        min_value=0,
        max_value=10,
        value=defaults["Tenure"] if defaults else 5,
    )

with col2:
    balance = st.number_input(
        "Account Balance ($)",
        min_value=0.0,
        value=defaults["Balance"] if defaults else 50000.0,
        step=1000.0,
    )
    num_products = st.selectbox(
        "Number of Products",
        options=[1, 2, 3, 4],
        index=(
            [1, 2, 3, 4].index(defaults["NumOfProducts"]) if defaults else 0
        ),
    )
    has_card = st.radio(
        "Has Credit Card?",
        options=["Yes", "No"],
        index=(
            0
            if (defaults["HasCrCard"] if defaults else 1) == 1
            else 1
        ),
        horizontal=True,
    )
    is_active = st.radio(
        "Is Active Member?",
        options=["Yes", "No"],
        index=(
            0
            if (defaults["IsActiveMember"] if defaults else 1) == 1
            else 1
        ),
        horizontal=True,
    )
    complain = st.radio(
        "Has Lodged Complain?",
        options=["Yes", "No"],
        index=(
            0
            if (defaults["Complain"] if defaults else 0) == 1
            else 1
        ),
        horizontal=True,
    )

with col3:
    salary = st.number_input(
        "Estimated Salary ($)",
        min_value=0.0,
        value=defaults["EstimatedSalary"] if defaults else 75000.0,
        step=1000.0,
    )
    satisfaction = st.slider(
        "Satisfaction Score",
        min_value=1,
        max_value=5,
        value=defaults["Satisfaction Score"] if defaults else 3,
    )
    card_type = st.selectbox(
        "Card Type",
        options=["DIAMOND", "GOLD", "PLATINUM", "SILVER"],
        index=(
            ["DIAMOND", "GOLD", "PLATINUM", "SILVER"].index(
                defaults["Card Type"]
            )
            if defaults
            else 0
        ),
    )
    points = st.number_input(
        "Points Earned",
        min_value=0,
        max_value=1000,
        value=defaults["Point Earned"] if defaults else 450,
    )

st.divider()

# --- PREPROCESSING FUNCTION ---
# Matches the encoded columns created in your notebook:
# ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
#  'IsActiveMember', 'EstimatedSalary', 'Complain', 'Satisfaction Score',
#  'Point Earned', 'Geography_Germany', 'Geography_Spain', 'Gender_Male',
#  'Card Type_GOLD', 'Card Type_PLATINUM', 'Card Type_SILVER']


def preprocess_inputs():
    input_dict = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
        "Complain": 1 if complain == "Yes" else 0,
        "Satisfaction Score": satisfaction,
        "Point Earned": points,
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Card Type_GOLD": 1 if card_type == "GOLD" else 0,
        "Card Type_PLATINUM": 1 if card_type == "PLATINUM" else 0,
        "Card Type_SILVER": 1 if card_type == "SILVER" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    scaled_input = scaler.transform(input_df)
    return scaled_input


# --- PREDICTION BUTTON ---
if st.button("🚀 Predict Churn Risk", use_container_width=True, type="primary"):
    X_input = preprocess_inputs()

    prediction = model.predict(X_input)[0]

    # Handle probabilistic prediction if model supports it
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_input)[0][1] * 100
    else:
        prob = 100.0 if prediction == 1 else 0.0

    st.subheader("📊 Prediction Results")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if prediction == 1:
            st.error("🚨 Customer is Likely to Churn!")
        else:
            st.success("✅ Customer is Likely to Stay (No Churn)")

    with res_col2:
        st.metric(label="Calculated Churn Probability", value=f"{prob:.1f}%")
        st.progress(int(prob))