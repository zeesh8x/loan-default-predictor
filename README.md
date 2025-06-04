# 💳 Loan Default Predictor

This is a machine learning web app built using **Streamlit** that predicts whether a credit card user will default on their payment next month, based on various personal and financial features.

🔗 **Live Demo**:  
[👉 Loan Default Predictor Streamlit App](https://loan-default-predictor-2igt8q8qvwksb7vbc4xwnc.streamlit.app/)

---

## 🧠 Problem Statement

Financial institutions need to assess credit risk before issuing credit. This app uses historical data to predict the likelihood of a customer defaulting on their credit card payment the following month.

---

## 🚀 Features

- Interactive Streamlit UI
- Real-time predictions using a trained Random Forest model
- Scaled numeric features and properly encoded categorical inputs
- Clean, intuitive layout for non-technical users

---

## 📊 Model Info

- **Algorithm**: Random Forest Classifier  
- **Preprocessing**: StandardScaler for feature scaling  
- **Trained On**: `default_of_credit_card_clients.xls` dataset (UCI repository)

---

## 🖥️ Tech Stack

- Python
- Pandas, Scikit-learn, Joblib
- Streamlit for frontend
- Git/GitHub for version control and deployment

---

## ⚙️ How to Run Locally

### 1. Clone the repository

git clone https://github.com/your-username/loan-default-predictor.git
cd loan-default-predictor


### 2. Install dependencies
Make sure you have Python 3.8 or above:

pip install -r requirements.txt


### 3. Train the model (if not already trained)
python train_model.py

### 4. Run the Streamlit app
python -m streamlit run app.py


___________________________________________________________________________________________________________________________________________________________

🧪 Sample Inputs for Testing
Try out combinations like:

Education: University

Marriage: Single

Bill Payments: Positive numbers for debts, 0 for paid

Repayment Status: 0 means paid, 1–9 means delayed payments

Past Payments: Amounts paid in previous months


___________________________________________________________________________________________________________________________________________________________

📤 Deployment
The app is deployed using Streamlit Community Cloud. Push changes to GitHub and they’ll automatically reflect live.


🙌 Acknowledgements
Dataset from UCI Machine Learning Repository

Streamlit for rapid web UI development

___________________________________________________________________________________________________________________________________________________________

🧑‍💻 Author
Zeeshan
Feel free to connect on LinkedIn or raise issues in the repository.

