from pydantic import BaseModel
from typing import List

class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int
    
class DiabetesPredictionResponse(BaseModel):
    prediction: int
    result: str
    confidence: float
    probability: List[float]

class DiabetesMetricResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float