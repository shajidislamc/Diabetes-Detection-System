# Diabetes Prediction API & Frontend

A FastAPI-based REST API for predicting diabetes from health metrics using a trained machine learning model. Includes endpoints for prediction, model info, health check, and metrics.

## Features
- Predict diabetes using health data
- View model info and metrics
- Built with FastAPI, scikit-learn, pandas, joblib

## Project Structure
```
app/
  main.py           # FastAPI app
  schemas.py        # Pydantic models
model/
  train_model.py    # Model training script
  diabetes_model.joblib  # Trained model
  diabetes_metrics.json  # Model metrics
requirements.txt    # Python dependencies
diabetes.csv        # Dataset
```

## Setup
1. Clone the repo and navigate to the project folder.
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/Scripts/activate  # Windows
   # or
   source env/bin/activate      # Linux/Mac
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Train the model (generates model and metrics):
   ```bash
   python model/train_model.py
   ```
5. Start the API server:
   ```bash
   uvicorn app.main:app --reload
   ```

## API Endpoints

### Health Check
- `GET /health`
- Response: `{ "status": "API is healthy" }`

### Model Info
- `GET /info`
- Response: `{ "model_type": "RandomForestClassifier", "features": [ ... ] }`

### Predict Diabetes
- `POST /predict`
- Request JSON:
  ```json
  {
    "Pregnancies": 3,
    "Glucose": 145,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 85,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.35,
    "Age": 29
  }
  ```
- Response JSON:
  ```json
  {
    "prediction": 0,
    "result": "Not Diabetic",
    "confidence": 0.87,
    "probability": [0.87, 0.13]
  }
  ```

### Model Metrics
- `GET /metrics`
- Response JSON:
  ```json
  {
    "accuracy": 0.76,
    "precision": 0.78,
    "recall": 0.68,
    "f1_score": 0.73
  }
  ```

## Notes
- Retrain the model with `python model/train_model.py` if you update the dataset.
- The API expects input keys to match the example (case-insensitive, underscores allowed).
