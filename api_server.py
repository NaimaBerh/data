"""
FastAPI Deployment for FakeGuard AI
Web service for fake account detection across GitHub, LinkedIn, and Instagram
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pickle
import numpy as np

app = FastAPI(
    title="FakeGuard AI",
    description="Unified Fake Account Detection API for GitHub, LinkedIn, and Instagram",
    version="1.0.0"
)

# Load model at startup
MODEL_PATH = "fakeguard_model.pkl"

try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    MODEL = model_data['model']
    FEATURE_LIST = model_data['feature_list']
    PLATFORM_MAP = model_data['platform_map']
    PLATFORM_MAP_REV = {v: k for k, v in PLATFORM_MAP.items()}
except FileNotFoundError:
    MODEL = None
    FEATURE_LIST = None
    PLATFORM_MAP = None
    PLATFORM_MAP_REV = None


class AccountFeatures(BaseModel):
    """Account features for prediction"""
    features: Dict[str, float]
    platform: str = "unknown"
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "Followers": 500,
                    "Connections": 250,
                    "Photo": 1,
                    "Number of Skills": 10,
                    "Number of Experiences": 3
                },
                "platform": "LinkedIn"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response model"""
    label: str
    prediction: int
    confidence: float
    probabilities: Dict[str, float]
    platform: str
    message: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to FakeGuard AI - Fake Account Detection Service",
        "version": "1.0.0",
        "supported_platforms": list(PLATFORM_MAP.values()) if PLATFORM_MAP else []
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if MODEL is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "feature_count": len(FEATURE_LIST) if FEATURE_LIST else 0
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_fake_account(account: AccountFeatures):
    """
    Predict if an account is fake or real
    
    - **features**: Dictionary of feature names and values
    - **platform**: Platform name (LinkedIn, Instagram, GitHub)
    
    Returns prediction result with confidence scores
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate platform
    platform = account.platform
    if platform not in PLATFORM_MAP_REV:
        platform = "LinkedIn"  # Default to LinkedIn
    
    platform_id = PLATFORM_MAP_REV.get(platform, 0)
    
    # Create feature vector
    feature_vector = []
    for feat in FEATURE_LIST:
        feature_vector.append(account.features.get(feat, 0))
    feature_vector.append(platform_id)
    
    X = np.array([feature_vector])
    
    # Predict
    prediction = int(MODEL.predict(X)[0])
    probability = MODEL.predict_proba(X)[0]
    
    label = "Real/Human" if prediction == 1 else "Fake/Bot"
    confidence = float(probability[prediction])
    
    # Generate message based on prediction
    if prediction == 1:
        message = f"This account appears to be a genuine human account on {platform}."
    else:
        message = f"Warning: This account shows signs of being fake/bot on {platform}."
    
    return PredictionResponse(
        label=label,
        prediction=prediction,
        confidence=confidence,
        probabilities={
            "Fake/Bot": float(probability[0]),
            "Real/Human": float(probability[1])
        },
        platform=platform,
        message=message
    )


@app.post("/predict/batch")
async def predict_batch_accounts(accounts: list[AccountFeatures]):
    """
    Predict multiple accounts at once
    
    - **accounts**: List of account feature dictionaries with platform
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for account in accounts:
        platform = account.platform
        if platform not in PLATFORM_MAP_REV:
            platform = "LinkedIn"
        
        platform_id = PLATFORM_MAP_REV.get(platform, 0)
        
        # Create feature vector
        feature_vector = []
        for feat in FEATURE_LIST:
            feature_vector.append(account.features.get(feat, 0))
        feature_vector.append(platform_id)
        
        X = np.array([feature_vector])
        
        # Predict
        prediction = int(MODEL.predict(X)[0])
        probability = MODEL.predict_proba(X)[0]
        
        label = "Real/Human" if prediction == 1 else "Fake/Bot"
        
        results.append({
            "label": label,
            "prediction": prediction,
            "confidence": float(probability[prediction]),
            "probabilities": {
                "Fake/Bot": float(probability[0]),
                "Real/Human": float(probability[1])
            },
            "platform": platform
        })
    
    return {"predictions": results, "total": len(results)}


@app.get("/features")
async def get_feature_list():
    """Get the list of features required by the model"""
    if FEATURE_LIST is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "features": FEATURE_LIST,
        "feature_count": len(FEATURE_LIST),
        "platforms": PLATFORM_MAP
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
