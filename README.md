# TransplantCare – Liver Transplant Death Risk Predictor

This project is my Introduction to Data Science (IDS) term project. It uses real liver transplant waitlist data to predict the risk of death for patients based on their age, sex, blood group, listing year, and follow‑up time.

## 1. Project Overview

- **Goal:** Estimate death risk for liver transplant waitlist patients.
- **Dataset:** `transplant.csv` (age, sex, blood group, year, futime, event).
- **Target:** Binary label `death_flag`  
  - `1` = patient died on waitlist  
  - `0` = transplanted / censored / withdrawn
- **Model:** Random Forest classifier (scikit‑learn).
- **App:** Deployed using Streamlit for interactive, user‑input based prediction.

## 2. Data & Features

Original columns (from `transplant.csv`):

- `age` – Patient age (years)
- `sex` – `m` or `f`
- `abo` – Blood group (`A`, `B`, `AB`, `O`)
- `year` – Listing year
- `futime` – Follow‑up time (days)
- `event` – Outcome (`death`, `ltx`, `censored`, `withdraw`)

Engineered columns used for modelling:

- `death_flag` – 1 if `event == 'death'`, else 0
- `sex_enc` – Encoded sex (LabelEncoder)
- `abo_enc` – Encoded blood group (LabelEncoder)

Final feature vector:

- `['age', 'year', 'futime', 'sex_enc', 'abo_enc']`

## 3. Modelling Pipeline

1. Load and inspect `transplant.csv`.
2. Create `death_flag` as the binary target.
3. Encode `sex` and `abo` with `LabelEncoder`.
4. Handle missing ages by mean imputation.
5. Split into train/test (80/20).
6. Scale numeric features with `StandardScaler`.
7. Train a `RandomForestClassifier`.
8. Evaluate using accuracy, confusion matrix, and classification report.
9. Save model and preprocessing objects as:
   - `clf.pkl`
   - `scaler.pkl`
   - `le_sex.pkl`
   - `le_abo.pkl`
   - `feature_cols.pkl`

## 4. Streamlit App

The Streamlit app (`app.py`) loads the saved model and allows the user to enter:

- Age  
- Listing year  
- Follow‑up time (days)  
- Sex (`m` / `f`)  
- Blood group (`A` / `B` / `AB` / `O`)

On clicking **“Predict death risk”**, the app:

1. Encodes the categorical inputs using the saved encoders.
2. Scales the features using the saved scaler.
3. Uses the Random Forest model to predict:
   - A binary death flag (high‑risk or low‑risk).
   - A probability score for death.

The app displays a clear message:
- **High Risk (death)** or  
- **Low Risk (no death)**  
along with the estimated death probability.

## 5. How to Run Locally

