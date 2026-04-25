import pandas as pd

df = pd.read_csv("data/churn.csv")

# 1. Shape of dataset
print("Shape of dataset:", df.shape)

# 2. Column information
print("\nDataset Info:")
print(df.info())

# 3. Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# 4. Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Target distribution
print("\nChurn Distribution:")
print(df["Exited"].value_counts())

print("\nChurn Percentage:")
print(df["Exited"].value_counts(normalize=True) * 100)

# 6. Drop non-informative columns
df = df.drop(["CustomerId", "Surname"], axis=1)

print("\nColumns after dropping unnecessary features:")
print(df.columns)

# 7. Feature Engineering

# Balance to Salary Ratio
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

# Product Density
df["ProductDensity"] = df["NumOfProducts"] / (df["Tenure"] + 1)

# Engagement-Product Interaction
df["EngagementProductInteraction"] = df["IsActiveMember"] * df["NumOfProducts"]

# Age-Tenure Interaction
df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]

print("\nNew Features Added:")
print(df.head())

# 8. Encoding categorical variables

df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

print("\nColumns after encoding:")
print(df.columns)

# 9. Split features and target

X = df.drop("Exited", axis=1)
y = df["Exited"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain set size:", X_train.shape)
print("Test set size:", X_test.shape)
# 10. Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaling completed")

# 11. Logistic Regression Model

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_scaled, y_train)

print("\nLogistic Regression model trained")

# 12. Predictions

y_pred = log_model.predict(X_test_scaled)
y_prob = log_model.predict_proba(X_test_scaled)[:, 1]

print("\nPredictions completed")

# Show first 10 probabilities
print("\nFirst 10 churn probabilities:")
print(y_prob[:10])

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("\nModel Evaluation:")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

print("\nRandom Forest trained")

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Evaluation:")


print("\n=== Random Forest Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")

# Feature Importance

import pandas as pd

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

print("\nXGBoost trained")

# Predictions
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\n=== XGBoost Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_xgb):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}")

# Custom threshold (try 0.3)

threshold = 0.3

y_pred_custom = (y_prob_xgb >= threshold).astype(int)

print("\n=== XGBoost with Custom Threshold (0.3) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_custom):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_custom):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_custom):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_custom):.4f}")

# Risk Scoring System

def risk_category(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

risk_df = pd.DataFrame({
    "Actual": y_test.values,
    "Probability": y_prob_xgb
})

risk_df["RiskCategory"] = risk_df["Probability"].apply(risk_category)

print("\nSample Risk Scores:")
print(risk_df.head(10))
import joblib

# Save model
joblib.dump(xgb_model, "models/churn_model.pkl")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Save feature names
joblib.dump(X.columns.tolist(), "models/features.pkl")

print("\nModel, scaler, and features saved successfully!")

