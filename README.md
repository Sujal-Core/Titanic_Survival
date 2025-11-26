# Titanic Survival Prediction

Predict Titanic passenger survival using a **stacked ensemble of machine learning models** with **explainable AI** through SHAP. Built as a **Streamlit app**, this project demonstrates data preprocessing, feature engineering, model stacking, and interpretability.

---

##  Overview

This project predicts whether a Titanic passenger survived using a **stacked ensemble** of the following models:

* **Random Forest**
* **Gradient Boosting**
* **XGBoost**

**SHAP** is used to interpret feature contributions, providing insights into model predictions. A **Streamlit app** enables interactive passenger survival prediction.

---

## ✨ Features

The app uses the following features:

* **Pclass** – Passenger class (1st, 2nd, 3rd)
* **Sex** – Male or Female
* **Age** – Passenger age
* **IsAlone** – Derived feature: 1 if passenger is alone, 0 otherwise
* **Embarked** – Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
* **Deck** – Deck of cabin

---

## 📦 Project Structure

```
Titanic_Survival/
├─ models/                  # Pre-trained model files (Random Forest, Gradient Boosting, Stacked XGBoost)
├─ data/                    # Optional: dataset files (X_train.csv, y_train.csv, etc.)
├─ Streamlit.py   # Streamlit app for interactive predictions
├─ train.py                 # Training script
├─ test.py                  # Testing / evaluation script
├─ requirements.txt         # Python dependencies
└─ README.md                # Project documentation
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Sujal-Core/Titanic_Survival.git
cd Titanic_Survival
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run streamlit.py
```

---

## 🖥 Usage

1. Enter passenger details:

   * Age
   * Sex
   * Passenger Class
   * Family Info
   * Port of Embarkation
   * Deck

2. Click **Predict**.

3. View results:

   * **Survival probability**
   * **Prediction** (Survived / Did Not Survive)
   * **SHAP explanations** showing feature contributions

---

## 📊 Model Training & Testing

* **train.py** – Preprocess data, train Random Forest, Gradient Boosting, and Stacked XGBoost models, and save them to `models/`.
* **test.py** – Evaluate models on test data.

---

## 🔧 Dependencies

* Python >= 3.9
* pandas, numpy, scikit-learn
* xgboost, shap
* streamlit, matplotlib, seaborn


