# train_from_csv.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# -------------------------
# Paths
# -------------------------
PROCESSED_DIR = r"D:\village\Titanic\archive"  # folder containing your X_train.csv and y_train.csv
MODEL_DIR = r"D:\village\Titanic\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Load preprocessed data
# -------------------------
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()

# -------------------------
# Minimal features
# -------------------------
essential_features = ['Pclass', 'Sex', 'Age', 'IsAlone', 'Embarked']
X_train = X_train[essential_features]

# -------------------------
# Train base models
# -------------------------
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Save base models
joblib.dump(rf_model, os.path.join(MODEL_DIR, "RandomForest_minimal.pkl"))
joblib.dump(gb_model, os.path.join(MODEL_DIR, "GradientBoosting_minimal.pkl"))

# -------------------------
# Train stacked meta-model
# -------------------------
rf_preds_train = rf_model.predict_proba(X_train)
gb_preds_train = gb_model.predict_proba(X_train)
stacked_features_train = np.hstack([rf_preds_train, gb_preds_train])

meta_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
meta_model.fit(stacked_features_train, y_train)
joblib.dump(meta_model, os.path.join(MODEL_DIR, "XGB_Stacked_minimal.pkl"))

# -------------------------
# Training accuracy
# -------------------------
rf_acc = accuracy_score(y_train, rf_model.predict(X_train))
gb_acc = accuracy_score(y_train, gb_model.predict(X_train))
stacked_acc = accuracy_score(y_train, meta_model.predict(stacked_features_train))

print("RandomForest Accuracy:", rf_acc)
print("GradientBoosting Accuracy:", gb_acc)
print("Stacked XGB Accuracy:", stacked_acc)
