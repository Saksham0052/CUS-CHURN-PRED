# 📊 Customer Churn Prediction App

An interactive **web application** that predicts whether a customer is likely to churn (leave a service) based on telecom usage data.  
Built with **Python, Streamlit, and Machine Learning (Random Forest)**, the app provides quick, accurate predictions with an easy-to-use interface.

---

##Video link   

▶️ [Watch the demo video on Loom](https://www.loom.com/share/85a95fc138fd459da992b98c562c01eb?sid=4f72edd9-898a-4c4b-987b-ee492c8bd719)


## 🚀 Features

- 🔮 **Real-Time Churn Prediction**  
  Instantly predicts customer churn probability based on user-provided data.

- ⚙️ **Machine Learning Model**  
  Trained using a **Random Forest Classifier** with over 80% accuracy on test data.

- 🧩 **Feature Engineering & Preprocessing**  
  Automatically encodes categorical variables and scales numeric ones for consistency.

- 🖥️ **Interactive Streamlit UI**  
  Simple and elegant user interface for entering customer data and viewing predictions.

- 💾 **Pre-Trained Model & Scaler**  
  Uses saved `.joblib` files to avoid retraining every time the app is launched.

---

## 🧠 Machine Learning Workflow

1. **Data Preprocessing:**  
   - Handled missing values  
   - Encoded categorical columns  
   - Scaled numerical features  

2. **Model Training:**  
   - Used Random Forest Classifier  
   - Evaluated accuracy, precision, recall, and F1-score  

3. **Deployment:**  
   - Serialized trained model and scaler using `joblib`  
   - Integrated with Streamlit for live user interaction  

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Joblib, Streamlit  
- **Frontend/UI:** Streamlit  
- **Deployment:** Streamlit Cloud  

---

## ⚙️ Installation

To set up the project locally:

```bash
# Clone the repository
git clone https://github.com/Saksham0052/CUS-CHURN-PRED

# Navigate into the project directory
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
python -m streamlit run cus-churn-pred.py
in case some python modules does not work with any python version


