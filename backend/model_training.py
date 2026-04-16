import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_performance.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.joblib")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.joblib")

TARGET = "Final_Score"
CATEGORICAL_COLS = ["Class", "Subject", "Participation_Level"]
DROP_COLS = ["Student_ID", "Name"]
SCORE_COLS = ["Exam_Score_1", "Exam_Score_2", "Exam_Score_3"]


# -------------------------------
# Load Data
# -------------------------------
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset not found")
    return pd.read_csv(DATA_PATH)


# -------------------------------
# Preprocess
# -------------------------------
def preprocess_data(df):
    df = df.copy()

    df = df.dropna(subset=[TARGET])

    # Fill missing values
    for col in df.select_dtypes(include=["number"]).columns:
        df[col].fillna(df[col].mean(), inplace=True)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Feature Engineering
    df["Prev_Avg_Score"] = df[SCORE_COLS].mean(axis=1)
    df["Score_Trend"] = df["Exam_Score_3"] - df["Exam_Score_1"]

    # Drop IDs
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # Encode
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    feature_cols = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, encoders, feature_cols


# -------------------------------
# Train Models
# -------------------------------
def train_models(X_train, y_train):
    models = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoost": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


# -------------------------------
# Evaluate
# -------------------------------
def evaluate_models(models, X_test, y_test):
    best_model = None
    best_score = -1
    best_name = ""

    for name, model in models.items():
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)

        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name

    return best_model, best_name


# -------------------------------
# Save Model (FIXED)
# -------------------------------
def save_model(model, scaler, encoders, feature_cols):
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)


# -------------------------------
# Main function for API
# -------------------------------
def train_best_model():
    df = load_data()

    X_train, X_test, y_train, y_test, scaler, encoders, feature_cols = preprocess_data(df)

    models = train_models(X_train, y_train)

    best_model, best_name = evaluate_models(models, X_test, y_test)

    save_model(best_model, scaler, encoders, feature_cols)

    return best_model, {"model_name": best_name}


# -------------------------------
# Run manually
# -------------------------------
if __name__ == "__main__":
    train_best_model()
    print("Training complete ✅")