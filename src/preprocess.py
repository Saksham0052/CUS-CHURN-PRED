# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    """
    Load the Telco Customer Churn dataset from CSV
    """
    df = pd.read_csv(path)
    df.drop('customerID', axis=1, inplace=True)  # Drop ID column
    # Convert TotalCharges to numeric, fill missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    return df

# src/preprocess.py
def encode_features(df):
    """
    Encode categorical columns and target variable
    """
    cat_cols = df.select_dtypes(include='object').columns
    if 'Churn' in cat_cols:
        cat_cols = cat_cols.drop('Churn')

    for col in cat_cols:
        df[col] = df[col].astype(str)  # ensure no numbers
        df[col] = LabelEncoder().fit_transform(df[col])

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'No':0, 'Yes':1})
    return df


def scale_features(X, num_cols):
    """
    Scale numerical columns using StandardScaler
    """
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    return X, scaler
