# Predictive Modeling and Risk Scoring for Bank Customer Churn

## Abstract
Customer churn significantly impacts revenue and customer lifetime value in the banking sector. This project develops a predictive modeling system to identify customers at risk of churn using machine learning techniques. The system generates churn probability scores and classifies customers into risk categories, enabling proactive retention strategies.

---

## Introduction
Customer retention is a critical challenge for banks. Traditional analysis focuses on past churn behavior, but modern systems require predictive intelligence to identify potential churn before it occurs.

---

## Dataset Description
The dataset contains 10,000 customer records with features such as:
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

Target variable:
- Exited (1 = churn, 0 = retained)

---

## Methodology

### Data Preprocessing
- Removed non-informative features (CustomerId, Surname)
- Handled categorical variables using one-hot encoding
- No missing values found

### Feature Engineering
New features created:
- BalanceSalaryRatio
- ProductDensity
- EngagementProductInteraction
- AgeTenureInteraction

### Model Development
- Logistic Regression (baseline)
- Random Forest
- XGBoost (final model)

### Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## Results

### Logistic Regression
- Accuracy: 80%
- Recall: 19%

### Random Forest
- Accuracy: 86%
- Recall: 45%

### XGBoost (Final Model)
- Accuracy: 86.9%
- Precision: 79.2%
- Recall: 48.6%
- ROC-AUC: 86.6%

After threshold tuning (0.3):
- Recall improved to 66.5%

---

## Key Insights
- Age is the most important factor in churn prediction
- Customers with fewer products are more likely to churn
- Low engagement increases churn risk
- Balance-to-salary ratio influences customer behavior

---

## Conclusion
The project successfully builds a predictive churn intelligence system. By focusing on behavioral and engagement factors, the model provides actionable insights for retention strategies. Threshold tuning significantly improves churn detection performance.

---

## Future Work
- Deploy model using cloud platforms
- Improve recall using advanced hyperparameter tuning
- Integrate real-time data pipelines