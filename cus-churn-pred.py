# app.py
import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load model, scaler & training columns
# -------------------------------
model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')
X_train_columns = joblib.load('models/X_train_columns.pkl')

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={'About': "This app predicts customer churn using a trained Random Forest model."}
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #FFFFFF; }
.big-font { font-size:25px !important; font-weight:bold; }
.medium-font { font-size:18px !important; }
.small-font { font-size:14px !important; }
label, .stMarkdown p { color: #00FF00 !important; font-weight: bold; }
div.stButton>button { color: #FFFFFF; background-color: #4CAF50; font-size:16px; font-weight:bold; }
div.stButton>button:hover { background-color: #45A049; color: #FFFFFF; }
div.stTextInput>div>input { color: #000000; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# App Title
# -------------------------------
st.markdown("<h1 class='big-font'>📊 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("Enter customer details below to predict whether they are likely to churn.", unsafe_allow_html=True)

# -------------------------------
# User Input
# -------------------------------
def user_input_features():
    gender = st.selectbox("Gender", ("Male", "Female"))
    SeniorCitizen = st.selectbox("Senior Citizen", ("No", "Yes"))
    Partner = st.selectbox("Has Partner?", ("No", "Yes"))
    Dependents = st.selectbox("Has Dependents?", ("No", "Yes"))
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    PhoneService = st.selectbox("Phone Service?", ("No", "Yes"))
    MultipleLines = st.selectbox("Multiple Lines?", ("No", "Yes", "No phone service"))
    InternetService = st.selectbox("Internet Service Type", ("DSL", "Fiber optic", "No"))
    OnlineSecurity = st.selectbox("Online Security?", ("No", "Yes", "No internet service"))
    OnlineBackup = st.selectbox("Online Backup?", ("No", "Yes", "No internet service"))
    DeviceProtection = st.selectbox("Device Protection?", ("No", "Yes", "No internet service"))
    TechSupport = st.selectbox("Tech Support?", ("No", "Yes", "No internet service"))
    StreamingTV = st.selectbox("Streaming TV?", ("No", "Yes", "No internet service"))
    StreamingMovies = st.selectbox("Streaming Movies?", ("No", "Yes", "No internet service"))
    Contract = st.selectbox("Contract Type", ("Month-to-month", "One year", "Two year"))
    PaperlessBilling = st.selectbox("Paperless Billing?", ("No", "Yes"))
    PaymentMethod = st.selectbox("Payment Method", ("Electronic check", "Mailed check", "Bank transfer", "Credit card"))
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0)

    # Convert Yes/No to 1/0 for SeniorCitizen and other binary fields
    binary_map = {'Yes': 1, 'No': 0}
    data = {
        'gender': gender,
        'SeniorCitizen': binary_map[SeniorCitizen],
        'Partner': binary_map[Partner],
        'Dependents': binary_map[Dependents],
        'tenure': tenure,
        'PhoneService': binary_map[PhoneService],
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': binary_map[PaperlessBilling],
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# -------------------------------
# Preprocessing Input
# -------------------------------
cat_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaymentMethod']

# Encode categorical columns
input_encoded = pd.get_dummies(input_df, columns=cat_cols)

# Align input with training data
input_encoded = input_encoded.reindex(columns=X_train_columns, fill_value=0)

# Scale numeric columns
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_encoded)
    prediction_prob = model.predict_proba(input_encoded)[0][1]

    if prediction[0] == 1:
        st.markdown(f"<h2 class='medium-font'>⚠️ Customer is likely to churn!</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 class='medium-font'>✅ Customer is likely to stay!</h2>", unsafe_allow_html=True)

    st.markdown(f"<p class='small-font'>Prediction probability of churn: {prediction_prob:.2f}</p>", unsafe_allow_html=True)
