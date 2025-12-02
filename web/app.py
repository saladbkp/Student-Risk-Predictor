from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import json
import os

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

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")

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
