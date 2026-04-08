from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("model/model.joblib")

app = FastAPI(title="Mini ML API", version="1.0.0")


class PredictionRequest(BaseModel):
    age: float = Field(..., json_schema_extra={"example": 30})
    salary: float = Field(..., json_schema_extra={"example": 50000})
    score: float = Field(..., json_schema_extra={"example": 0.8})
    history: float = Field(..., json_schema_extra={"example": 1})


class PredictionResponse(BaseModel):
    prediction: int


model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "API ML opérationnelle"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}