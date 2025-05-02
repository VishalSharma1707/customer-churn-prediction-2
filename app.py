
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn

# Load model
model = joblib.load("xgb_churn_model.pkl")

# Define input model
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

# Create FastAPI app
app = FastAPI()

@app.post("/predict/")
def predict_churn(data: CustomerData):
    input_df = pd.DataFrame([data.model_dump()])
    pred_prob = model.predict_proba(input_df)[0][1]
    prediction = int(pred_prob > 0.5)
    return {
        "churn_probability": round(float(pred_prob), 3),
        "churn_prediction": prediction
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
