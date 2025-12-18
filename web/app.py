from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import json
import os
import numpy as np

app = Flask(__name__, static_folder=".")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_risk_model.pkl")
PT_PATH = os.path.join(MODEL_DIR, "pt_transform.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_scaler.pkl")
NORM_PATH = os.path.join(MODEL_DIR, "normalizer.pkl")
FINAL_COLS_PATH = os.path.join(MODEL_DIR, "final_columns.json")
TRANSFORM_COLS_PATH = os.path.join(MODEL_DIR, "transform_cols.json")

model = joblib.load(MODEL_PATH)
pt = joblib.load(PT_PATH)
scaler = joblib.load(SCALER_PATH)
normalizer = joblib.load(NORM_PATH)

with open(TRANSFORM_COLS_PATH, "r") as f:
    transform_cols = json.load(f)

with open(FINAL_COLS_PATH, "r") as f:
    final_columns = json.load(f)

def prepare_dataframe(payload):
    df = pd.DataFrame([payload])
    encoded = pd.get_dummies(df)
    pt_cols = list(pt.feature_names_in_) if hasattr(pt, "feature_names_in_") else list(transform_cols)
    scaler_cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else list(transform_cols)
    norm_cols = list(normalizer.feature_names_in_) if hasattr(normalizer, "feature_names_in_") else list(transform_cols)
    required_cols = list({*pt_cols, *scaler_cols, *norm_cols})
    defaults = {"SES_Quartile": 2, "GPA": 2.5}
    for col in required_cols:
        if col not in encoded:
            encoded[col] = defaults.get(col, 0)
    encoded[pt_cols] = pt.transform(encoded[pt_cols])
    encoded[scaler_cols] = scaler.transform(encoded[scaler_cols])
    encoded[norm_cols] = normalizer.transform(encoded[norm_cols])
    for col in final_columns:
        if col not in encoded:
            encoded[col] = 0
    encoded = encoded[final_columns]
    return encoded

def calculate_stats(df):
    stats = {}
    
    # Calculate means
    stats["avg_math"] = float(df["TestScore_Math"].mean())
    stats["avg_reading"] = float(df["TestScore_Reading"].mean())
    stats["avg_science"] = float(df["TestScore_Science"].mean())
    stats["avg_attendance"] = float(df["AttendanceRate"].mean() * 100)

    # Calculate histograms
    def get_hist(data):
        counts, edges = np.histogram(data, bins=10)
        labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges)-1)]
        return {"labels": labels, "counts": counts.tolist()}

    stats["hist_math"] = get_hist(df["TestScore_Math"].dropna())
    stats["hist_reading"] = get_hist(df["TestScore_Reading"].dropna())
    stats["hist_science"] = get_hist(df["TestScore_Science"].dropna())
    stats["hist_attendance"] = get_hist(df["AttendanceRate"].dropna() * 100)
    
    return stats

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")

@app.route("/test.csv")
def serve_test_csv():
    return send_from_directory(os.path.dirname(__file__), "test.csv")

@app.route("/dataset_stats")
def dataset_stats():
    try:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "test.csv"))
        stats = calculate_stats(df)
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    try:
        df = pd.read_csv(file)
        # Validate columns
        required = ["TestScore_Math", "TestScore_Reading", "TestScore_Science", "AttendanceRate"]
        if not all(col in df.columns for col in required):
             return jsonify({"error": "Missing required columns"}), 400
             
        stats = calculate_stats(df)
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    encoded = prepare_dataframe(data)
    pred = int(model.predict(encoded)[0])
    prob = float(model.predict_proba(encoded)[0][1])
    label = "High Risk" if pred == 1 else "Low Risk"
    return jsonify({"predicted_risk": pred, "probability": prob, "label": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
