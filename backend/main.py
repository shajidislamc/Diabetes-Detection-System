from fastapi import FastAPI, HTTPException
from backend.schemas import DiabetesInput, DiabetesPredictionResponse, DiabetesMetricResponse
import joblib
import numpy as np
import pandas as pd
import json

# load model in joblib format
model = joblib.load("model/diabetes_model.joblib")

# create FastAPI instance
app = FastAPI(
    title="Diabetes Prediction API",
    description="An API to predict diabetes based on health metrics.",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "API is healthy"}

@app.get("/info")
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "features": [
            "pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree_function", "age"
        ]
    }

@app.post("/predict", response_model=DiabetesPredictionResponse)
def predict_diabetes(input_data: DiabetesInput):
    data = pd.DataFrame([input_data.model_dump()])

    # Determine expected feature names from the trained model if available
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    else:
        expected_features = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]

    # Normalization helper to compare names case/underscore-insensitively
    def _normalize(name: str) -> str:
        return name.replace("_", "").lower()

    # Build mapping from incoming column names to expected feature names
    col_map = {}
    for expected in expected_features:
        match = None
        for incoming in data.columns:
            if _normalize(incoming) == _normalize(expected):
                match = incoming
                break
        if match:
            col_map[match] = expected
        else:
            raise HTTPException(status_code=400, detail=f"Missing feature for model: {expected}")

    # Rename and reorder columns to match training
    data = data.rename(columns=col_map)
    data = data[expected_features]

    try:
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Compute result string and confidence
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    confidence = max(probability)

    return DiabetesPredictionResponse(
        prediction=prediction,
        result=result,
        confidence=confidence,
        probability=probability
    )

@app.get("/metrics", response_model=DiabetesMetricResponse)
def get_metrics():
    try:
        with open("model/diabetes_metrics.json", "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Metrics file not found")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid metrics JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Support both new metrics format (accuracy/precision/recall/f1_score)
    # and older training output which contained best_accuracy/accuracies.
    required_keys = {"accuracy", "precision", "recall", "f1_score"}
    if required_keys.issubset(metrics.keys()):
        return DiabetesMetricResponse(**{k: float(metrics[k]) for k in required_keys})

    # If JSON contains legacy fields, map best_accuracy -> accuracy and fill others with 0.0
    if "best_accuracy" in metrics:
        return DiabetesMetricResponse(
            accuracy=float(metrics.get("best_accuracy", 0.0)),
            precision=0.0,
            recall=0.0,
            f1_score=0.0
        )

    raise HTTPException(status_code=500, detail="Metrics JSON missing required keys")