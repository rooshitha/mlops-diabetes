from src.schema.predict_schema import PredictionRequest
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(
    title="Diabetes Prediction API",
    description="MLOps Project using FastAPI and MLflow",
    version="1.0"
)

# Load trained model
model = joblib.load("artifacts/model.pkl")


@app.get("/")
def home():
    return {
        "message": "Welcome to Diabetes Prediction API"
    }


@app.get("/health")
def health():
    return {
        "status": "Healthy"
    }


from fastapi import Body

@app.post("/predict")
def predict(request: PredictionRequest):

    data = np.array([
        [
            request.age,
            request.sex,
            request.bmi,
            request.bp,
            request.s1,
            request.s2,
            request.s3,
            request.s4,
            request.s5,
            request.s6,
        ]
    ])

    prediction = model.predict(data)

    return {
        "prediction": float(prediction[0])
    }