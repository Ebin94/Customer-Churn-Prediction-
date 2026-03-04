import pandas as pd
import numpy as np
import logging
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to hold the model pipeline
model_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML Model
    global model_pipeline
    try:
        model_path = 'models/churn_pipeline.pkl'
        logger.info(f"Loading model pipeline from {model_path}...")
        model_pipeline = joblib.load(model_path)
        logger.info("Model pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model pipeline: {e}")
        # Not raising an exception here so the app can start (though /predict will fail)
    
    yield
    # Clean up on shutdown
    model_pipeline = None
    logger.info("Model pipeline unloaded. Application shutting down.")


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability using an optimized ML Pipeline.",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic schemas with strict type validation
class CustomerData(BaseModel):
    gender: str = Field(..., description="Customer gender (Male/Female)")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether the customer is a senior citizen (1, 0)")
    Partner: str = Field(..., description="Whether the customer has a partner (Yes, No)")
    Dependents: str = Field(..., description="Whether the customer has dependents (Yes, No)")
    tenure: int = Field(..., ge=0, description="Number of months the customer has stayed with the company")
    PhoneService: str = Field(..., description="Whether the customer has a phone service (Yes, No)")
    MultipleLines: str = Field(..., description="Whether the customer has multiple lines (Yes, No, No phone service)")
    InternetService: str = Field(..., description="Customer's internet service provider (DSL, Fiber optic, No)")
    OnlineSecurity: str = Field(..., description="Whether the customer has online security (Yes, No, No internet service)")
    OnlineBackup: str = Field(..., description="Whether the customer has online backup (Yes, No, No internet service)")
    DeviceProtection: str = Field(..., description="Whether the customer has device protection (Yes, No, No internet service)")
    TechSupport: str = Field(..., description="Whether the customer has tech support (Yes, No, No internet service)")
    StreamingTV: str = Field(..., description="Whether the customer has streaming TV (Yes, No, No internet service)")
    StreamingMovies: str = Field(..., description="Whether the customer has streaming movies (Yes, No, No internet service)")
    Contract: str = Field(..., description="The contract term of the customer (Month-to-month, One year, Two year)")
    PaperlessBilling: str = Field(..., description="Whether the customer has paperless billing (Yes, No)")
    PaymentMethod: str = Field(..., description="The customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))")
    MonthlyCharges: float = Field(..., ge=0.0, description="The amount charged to the customer monthly")
    TotalCharges: float = Field(..., ge=0.0, description="The total amount charged to the customer")

class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of churn")
    churn_prediction: str = Field(..., description="'Yes' or 'No'")
    confidence: str = Field(..., description="Level of confidence in prediction (High, Medium, Low)")


@app.get("/")
def read_root():
    """Root endpoint to check API status."""
    return {"message": "Customer Churn Prediction API is running", "status": "active"}

@app.get("/health")
def health_check():
    """Health check endpoint. Useful for orchestrated deployments."""
    model_status = "loaded" if model_pipeline is not None else "failed"
    return {"status": "healthy", "model_pipeline": model_status}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """
    Predict customer churn based on input features.
    
    Data is passed directly to the loaded scikit-learn Pipeline, 
    so manual preprocessing inside the endpoint is no longer needed.
    """
    logger.info(f"Received prediction request")
    
    if model_pipeline is None:
        logger.error("Prediction attempted but model pipeline is not loaded.")
        raise HTTPException(status_code=503, detail="Model pipeline is currently unavailable.")

    try:
        # Convert Pydantic object to Pandas DataFrame 
        # (dict() handles nested dictionaries if any, standardizing for pandas construction)
        input_df = pd.DataFrame([customer.model_dump()])
        
        # Ensure category dtypes where appropriate if pipeline expects it 
        # (Usually pipeline handles base types, but categorizing here is safer)
        for col in input_df.select_dtypes(include=['object']).columns:
           input_df[col] = input_df[col].astype('category')

        logger.info("Executing model pipeline...")
        # Get probability and prediction
        churn_proba = float(model_pipeline.predict_proba(input_df)[0, 1])
        churn_pred = int(model_pipeline.predict(input_df)[0])

        # Logging output for monitoring
        logger.info(f"Prediction successful. Result: {churn_pred}, Proba: {churn_proba:.4f}")

        # Determine confidence
        if churn_proba > 0.7 or churn_proba < 0.3:
            confidence = "High"
        elif churn_proba > 0.6 or churn_proba < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"

        return PredictionResponse(
            churn_probability=churn_proba,
            churn_prediction="Yes" if churn_pred == 1 else "No",
            confidence=confidence
        )

    except Exception as e:
        logger.error(f"Prediction failed with error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
