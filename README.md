# Bank Customer Churn Prediction

## 📌 Project Overview
This project predicts customer churn in a bank using machine learning. It assigns a churn probability score and categorizes customers into risk segments.

---

## 🎯 Objectives
- Predict customer churn
- Generate churn probability scores
- Identify key churn drivers
- Build a risk scoring system

---

## 📊 Dataset Features
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

---

## ⚙️ Models Used
- Logistic Regression
- Random Forest
- XGBoost (Final Model)

---

## 📈 Final Model Performance (XGBoost)
- Accuracy: 86.9%
- Precision: 79.2%
- Recall: 66.5% (after threshold tuning)
- F1 Score: 63.4%
- ROC-AUC: 86.6%

---

## 🧠 Key Insights
- Age is the strongest churn factor
- Low engagement increases churn risk
- Customers with fewer products are more likely to churn

---

## 🚀 Features
- Churn probability prediction
- Risk categorization (Low / Medium / High)
- Streamlit web application
- Real-time prediction system

---

## 🖥️ Run Locally
```bash
python eda.py
python -m streamlit run app/streamlit_app.py