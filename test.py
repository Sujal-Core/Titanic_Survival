# test_from_csv.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# -------------------------
# Paths
# -------------------------
PROCESSED_DIR = r"D:\village\Titanic\archive"  # folder containing your X_test.csv and y_test.csv
MODEL_DIR = r"D:\village\Titanic\models"

# -------------------------
# Load test data
# -------------------------
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

# -------------------------
# Essential features (must match training)
# -------------------------
essential_features = ['Pclass', 'Sex', 'Age', 'IsAlone', 'Embarked']
X_test = X_test[essential_features]

# -------------------------
# Load trained models
# -------------------------
rf_model = joblib.load(os.path.join(MODEL_DIR, "RandomForest_minimal.pkl"))
gb_model = joblib.load(os.path.join(MODEL_DIR, "GradientBoosting_minimal.pkl"))
meta_model = joblib.load(os.path.join(MODEL_DIR, "XGB_Stacked_minimal.pkl"))

# -------------------------
# Base model predictions
# -------------------------
rf_preds = rf_model.predict(X_test)
gb_preds = gb_model.predict(X_test)

# -------------------------
# Stacked model predictions
# -------------------------
rf_probs = rf_model.predict_proba(X_test)
gb_probs = gb_model.predict_proba(X_test)
stacked_features_test = np.hstack([rf_probs, gb_probs])

stacked_preds = meta_model.predict(stacked_features_test)
stacked_probs = meta_model.predict_proba(stacked_features_test)[:, 1]

# -------------------------
# Evaluation
# -------------------------
print("RandomForest Accuracy:", accuracy_score(y_test, rf_preds))
print("GradientBoosting Accuracy:", accuracy_score(y_test, gb_preds))
print("Stacked XGB Accuracy:", accuracy_score(y_test, stacked_preds))
print("\nStacked Model Classification Report:\n")
print(classification_report(y_test, stacked_preds))

# -------------------------
# Optional: save predictions
# -------------------------
pred_df = X_test.copy()
pred_df['Actual'] = y_test
pred_df['Predicted'] = stacked_preds
pred_df['Survival_Prob'] = stacked_probs
pred_df.to_csv(os.path.join(PROCESSED_DIR, "predictions_minimal.csv"), index=False)
print("\nPredictions saved to 'predictions_minimal.csv'")
