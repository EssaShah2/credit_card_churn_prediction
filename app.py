import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration & Custom CSS (Responsive & Modern Design)
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
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# 2. Pipeline & Model Training Function
@st.cache_resource
def train_pipeline():
    # Load dataset match training pipeline
    df = pd.read_csv("Customer-Churn-Records.csv")
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
    
    x = df.drop(columns=['Exited'])
    y = df['Exited']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Column Transformer matching your notebook
    trf1 = ColumnTransformer([
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), ['Geography', 'Gender', 'Card Type'])
    ], remainder='passthrough')
    
    trf1.set_output(transform='pandas')
    x_train_trf = trf1.fit_transform(x_train)
    
    # Scaler transformation
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_trf)
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=trf1.get_feature_names_out())
    
    # Train classifier model
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train_scaled, y_train)
    
    return trf1, scaler, model, df

try:
    trf1, scaler, model, original_df = train_pipeline()
except Exception as e:
    st.error(f"Error loading dataset or model pipeline: {e}")
    st.stop()

# 3. Auto-fill Sample Feature Logic
if 'form_data' not in st.session_state:
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

def fill_sample_data():
    sample = original_df.sample(1).iloc[0]
    st.session_state.form_data = {
        'CreditScore': int(sample['CreditScore']),
        'Geography': str(sample['Geography']),
        'Gender': str(sample['Gender']),
        'Age': int(sample['Age']),
        'Tenure': int(sample['Tenure']),
        'Balance': float(sample['Balance']),
        'NumOfProducts': int(sample['NumOfProducts']),
        'HasCrCard': int(sample['HasCrCard']),
        'IsActiveMember': int(sample['IsActiveMember']),
        'EstimatedSalary': float(sample['EstimatedSalary']),
        'Complain': int(sample['Complain']),
        'Satisfaction Score': int(sample['Satisfaction Score']),
        'Card Type': str(sample['Card Type']),
        'Point Earned': int(sample['Point Earned'])
    }

# 4. Interface Header
st.title("💳 Customer Churn Analytics & Prediction")
st.write("Predict churn risk using machine learning based on customer activity metrics.")

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    st.button("🎲 Auto-Fill Random Customer Data", on_click=fill_sample_data)

st.divider()

# 5. Input Form Layout (Responsive Grid)
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

    # Transform input using pre-fitted transformers
    input_trf = trf1.transform(input_data)
    input_scaled = scaler.transform(input_trf)
    input_scaled_df = pd.DataFrame(input_scaled, columns=trf1.get_feature_names_out())

    # Generate Prediction
    prediction = model.predict(input_scaled_df)[0]
    probability = model.predict_proba(input_scaled_df)[0][1]

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
        st.progress(probability)