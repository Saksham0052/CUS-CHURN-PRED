# Customer Churn Prediction

## Project Overview
Predicts whether a customer will churn (leave) based on their profile and usage.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train model: `python src/train_model.py` (creates churn_model.pkl & scaler.pkl)
3. Run Streamlit app: `streamlit run app.py`

## Dataset
Telco Customer Churn dataset from Kaggle.

## Tech Stack
Python, Pandas, Scikit-learn, Streamlit

## Features
- Data preprocessing
- Model training (Random Forest)
- Interactive Streamlit web app
