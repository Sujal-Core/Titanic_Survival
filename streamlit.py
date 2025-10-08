import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# -------------------------
# Paths
# -------------------------
MODEL_DIR = r"D:\village\Titanic\models"

# Load trained models
rf_model = joblib.load(f"{MODEL_DIR}/RandomForest_minimal.pkl")
gb_model = joblib.load(f"{MODEL_DIR}/GradientBoosting_minimal.pkl")
meta_model = joblib.load(f"{MODEL_DIR}/XGB_Stacked_minimal.pkl")

# -------------------------
# Essential features
# -------------------------
essential_features = ['Pclass', 'Sex', 'Age', 'IsAlone', 'Embarked']

# -------------------------
# Streamlit UI
# -------------------------
st.title("Titanic Survival Prediction")
st.write("Enter essential passenger details for prediction:")

# Input form
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.number_input("Age", min_value=0, max_value=100, value=30)
IsAlone = st.selectbox("Is Alone?", ["Yes", "No"])
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Button for prediction
if st.button("Predict"):

    # Encode categorical inputs
    Sex_enc = 0 if Sex == "male" else 1
    Embarked_enc = 0 if Embarked == "C" else 1 if Embarked == "Q" else 2
    IsAlone_enc = 1 if IsAlone == "Yes" else 0

    # Prepare input DataFrame
    X_input = pd.DataFrame([{
        'Pclass': Pclass,
        'Sex': Sex_enc,
        'Age': Age,
        'IsAlone': IsAlone_enc,
        'Embarked': Embarked_enc
    }])

    X_input = X_input[essential_features]

    # Base model predictions
    rf_probs = rf_model.predict_proba(X_input)
    gb_probs = gb_model.predict_proba(X_input)

    # Stacked model prediction
    stacked_features = np.hstack([rf_probs, gb_probs])
    survival_prob = meta_model.predict_proba(stacked_features)[0, 1]
    survival_pred = meta_model.predict(stacked_features)[0]

    # Show results
    st.subheader("Prediction Results")
    st.write(f"Survival Probability: **{survival_prob:.2f}**")
    st.write("Prediction:", "**Survived**" if survival_pred == 1 else "**Did Not Survive**")
    # -------------------------
    # SHAP explanation (CPU-friendly)
    # -------------------------
    st.subheader("Feature Contribution (SHAP)")

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_input)

    # Handle both binary and multiclass
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_array = shap_values[1].flatten()
    else:
        shap_array = np.array(shap_values).flatten()

    # Ensure length matches features
    if len(shap_array) != len(essential_features):
        shap_array = shap_array[:len(essential_features)]

    shap_df = pd.DataFrame({
        "Feature": essential_features,
        "SHAP Value": shap_array
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    st.table(shap_df)