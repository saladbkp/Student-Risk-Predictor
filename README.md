# Student Risk Predictor

A simple web app and API for predicting student academic risk based on current semester factors. It compares performance with historical-trained models and supports quick lecturer intervention.

## Features
- Landing page with Start flow
- Two input modes: Step-by-step wizard or All-at-once form
- Friendly inputs with word-based ratings and sliders
- Prediction result with risk level and probability, plus quick suggestions

## Directory
- `web/app.py` – Flask backend and `/predict` endpoint
- `web/index.html` – Frontend UI (home, mode selection, wizard, results)
- `web/model/` – Model and preprocess artifacts
  - `best_risk_model.pkl`
  - `pt_transform.pkl`
  - `scaler_scaler.pkl`
  - `normalizer.pkl`
  - `final_columns.json`
  - `transform_cols.json`

## Prerequisites
- Python 3.10+
- Install dependencies:
  ```bash
  pip install Flask joblib pandas scikit-learn
  ```

## Run
- Development server:
  ```bash
  python -m flask --app web/app.py run -h 0.0.0.0 -p 5000
  ```
- Open: `http://127.0.0.1:5000/`

## API
- `POST /predict`
- Request JSON example:
  ```json
  {
    "Age": 16,
    "Grade": 10,
    "Gender": "Female",
    "Race": "White",
    "ParentalEducation": "HS",
    "SchoolType": "Public",
    "Locale": "City",
    "TestScore_Math": 72.0,
    "TestScore_Reading": 65.0,
    "TestScore_Science": 70.0,
    "AttendanceRate": 0.88,
    "StudyHours": 1.0,
    "InternetAccess": 1,
    "Extracurricular": 0,
    "PartTimeJob": 0,
    "ParentSupport": 0.40,
    "Romantic": 0,
    "FreeTime": 3,
    "GoOut": 1
  }
  ```
- Response JSON example:
  ```json
  {
    "predicted_risk": 0,
    "probability": 0.17588399022140158,
    "label": "Low Risk"
  }
  ```

## Notes
- Preprocessing order for numeric transforms: PowerTransformer → StandardScaler → MinMaxScaler
- Missing features are safely defaulted:
  - `SES_Quartile = 2`
  - `GPA = 2.5` (not required in UI)
- One-hot encoded columns are aligned to `final_columns.json` to match training order.
- Sklearn version warnings may appear when loading pickled estimators from older versions; predictions still run normally.

## Suggestion Logic
- Simple heuristics generate tips based on inputs (e.g., attendance < 90%, study hours < 7).

## License
- For academic use and prototyping purposes.
