# Diabetes Prediction System

A full-stack project for diabetes prediction using a machine learning model, featuring:
- **Backend**: FastAPI REST API for predictions, model info, and metrics.
- **Frontend**: Streamlit web app for user-friendly predictions.

---

## Project Structure

```
backend/
  main.py           # FastAPI app
  schemas.py        # Pydantic models
  requirements.txt  # Backend dependencies
frontend/
  app.py            # Streamlit frontend
  requirements.txt  # Frontend dependencies
model/
  train_model.py    # Model training script
  diabetes_model.joblib  # Trained model (generated)
  diabetes_metrics.json  # Model metrics (generated)
diabetes.csv        # Dataset
Dockerfile
docker-compose.yml
README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repo-url>
cd Diabetes Prediction API & Frontend
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv env
# On Windows:
env\Scripts\activate
# On Linux/Mac:
source env/bin/activate
```

### 3. Install dependencies

#### Backend
```bash
pip install -r backend/requirements.txt
```

#### Frontend
```bash
pip install -r frontend/requirements.txt
```

### 4. Train the model

```bash
python model/train_model.py
```

### 5. Start the backend API

```bash
uvicorn backend.main:app --reload
```
The API will be available at: http://127.0.0.1:8000

### 6. Start the frontend (in a new terminal)

```bash
cd frontend
streamlit run app.py
```
The Streamlit app will open in your browser.

---

## API Endpoints

### Health Check
- `GET /health`
- Response: `{ "status": "API is healthy" }`

### Model Info
- `GET /info`
- Response:  
  ```json
  {
    "model_type": "RandomForestClassifier",
    "features": [
      "pregnancies", "glucose", "blood_pressure", "skin_thickness",
      "insulin", "bmi", "diabetes_pedigree_function", "age"
    ]
  }
  ```

### Predict Diabetes
- `POST /predict`
- Request JSON:
  ```json
  {
  "pregnancies": 3,
  "glucose": 145,
  "blood_pressure": 70,
  "skin_thickness": 20,
  "insulin": 85,
  "bmi": 33.6,
  "diabetes_pedigree_function": 0.35,
  "age": 29
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

---

## Frontend Usage

- Open the Streamlit app (`streamlit run app.py` in the `frontend` folder).
- Enter patient details and click "Predict" to get results from the backend API.


---

## Notes

- Retrain the model with `python model/train_model.py` if you update the dataset.
- The API expects input keys to match the example (case-insensitive, underscores allowed).
- You can deploy the backend and frontend separately or together using Docker.