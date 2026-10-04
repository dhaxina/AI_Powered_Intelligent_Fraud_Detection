from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# 1. MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "xgboost_fraud_classifier.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Please check where your .pkl file is saved."
    )

model = joblib.load(MODEL_PATH)

print(f"Model loaded successfully from: {MODEL_PATH}")


# ============================================================
# 2. FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI-Powered Intelligent Fraud Detection API",
    description=(
        "API for detecting fraudulent credit card transactions "
        "using an XGBoost classifier."
    ),
    version="1.0.0",
)


# ============================================================
# 3. INPUT MODEL
# ============================================================

class Transaction(BaseModel):
    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float = Field(..., ge=0)


# ============================================================
# 4. OUTPUT MODEL
# ============================================================

class FraudPredictionResponse(BaseModel):
    prediction: int
    result: str
    fraud_probability: float
    fraud_percentage: str


# ============================================================
# 5. FEATURE ORDER EXPECTED BY MODEL
# ============================================================

FEATURE_ORDER = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
    "Hour",
]


# ============================================================
# 6. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():
    return {
        "message": "AI-Powered Intelligent Fraud Detection API",
        "status": "running",
        "model": "XGBoost",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
    }


# ============================================================
# 7. HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


# ============================================================
# 8. PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=FraudPredictionResponse
)
def predict_fraud(transaction: Transaction):

    try:
        # --------------------------------------------------------
        # Convert request to dictionary
        # --------------------------------------------------------

        transaction_data = transaction.model_dump()

        # --------------------------------------------------------
        # Calculate Hour from Time
        #
        # Time in the credit-card dataset is measured in seconds.
        #
        # 3600 seconds = 1 hour
        # --------------------------------------------------------

        transaction_data["Hour"] = (
            transaction_data["Time"] // 3600
        ) % 24

        # --------------------------------------------------------
        # Create DataFrame
        # --------------------------------------------------------

        input_data = pd.DataFrame(
            [transaction_data],
            columns=FEATURE_ORDER
        )

        # --------------------------------------------------------
        # Make prediction
        # --------------------------------------------------------

        prediction = model.predict(input_data)[0]

        # --------------------------------------------------------
        # Get probabilities
        # --------------------------------------------------------

        probabilities = model.predict_proba(input_data)[0]

        fraud_probability = float(probabilities[1])

        fraud_percentage = fraud_probability * 100

        # --------------------------------------------------------
        # Result
        # --------------------------------------------------------

        if int(prediction) == 1:
            result = "Fraudulent Transaction"
        else:
            result = "Legitimate Transaction"

        # --------------------------------------------------------
        # Return response
        # --------------------------------------------------------

        return {
            "prediction": int(prediction),
            "result": result,
            "fraud_probability": round(
                fraud_probability,
                6
            ),
            "fraud_percentage": f"{fraud_percentage:.2f}%"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# 9. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )