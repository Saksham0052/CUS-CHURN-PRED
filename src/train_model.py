# src/train_model.py
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import load_data, encode_features, scale_features

# -------------------------------
# 1. Load & preprocess data
# -------------------------------
csv_path = os.path.join(os.path.dirname(__file__), 'Telco-Customer-Churn.csv')
df = load_data(csv_path)
df = encode_features(df)

# -------------------------------
# 2. Split features and target
# -------------------------------
X = df.drop('Churn', axis=1)
y = df['Churn']

# -------------------------------
# 3. Scale numerical features
# -------------------------------
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X, scaler = scale_features(X, num_cols)

# -------------------------------
# 4. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 5. Save training columns
# -------------------------------
models_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
if not os.path.exists(models_folder):
    os.makedirs(models_folder)

X_train_columns = X_train.columns
joblib.dump(X_train_columns, os.path.join(models_folder, 'X_train_columns.pkl'))
print("X_train columns saved!")

# -------------------------------
# 6. Train Random Forest model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 7. Evaluate model
# -------------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 8. Save model and scaler
# -------------------------------
joblib.dump(model, os.path.join(models_folder, 'churn_model.pkl'))
joblib.dump(scaler, os.path.join(models_folder, 'scaler.pkl'))
print(f"Model, scaler, and columns saved in {models_folder}")
