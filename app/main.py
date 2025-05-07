from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Optional

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our prediction module
from prediction import predict_classification
from train_models import train_take_keep_classifier, train_toxic_gas_classifier

# Initialize FastAPI app
app = FastAPI(
    title="Sensor Data Classification API",
    description="API for classifying sensor data using machine learning models",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup for static files and templates
templates = Jinja2Templates(directory="templates")
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Pydantic models for request validation
class SensorData(BaseModel):
    values: List[float]
    
    class Config:
        schema_extra = {
            "example": {
                "values": [450, 5, 30]  # Example values: [gas_value, distance, weight]
            }
        }

class TrainingResult(BaseModel):
    message: str
    accuracy: float

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Return the home page with documentation"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(data: SensorData):
    """
    Endpoint to predict action (Take/Keep) and gas type (Normal/Toxic) based on sensor data
    
    The expected input is a list of 3 values:
    - For Take/Keep classification: [gas_value, distance, weight]
    - For Gas classification: These same values are reinterpreted as [fill_level, weight, gas_concentration]
    """
    try:
        # Validate input
        if len(data.values) != 3:
            raise HTTPException(
                status_code=400, 
                detail="Input must contain exactly 3 values: [gas_value/fill_level, distance/weight, weight/gas_concentration]"
            )
        
        # Make prediction
        result = predict_classification(data.values)
        return result
        
    except Exception as e:
        # Check if model files exist, if not suggest training
        model_files = Path(".").glob("*.pkl")
        if not list(model_files):
            raise HTTPException(
                status_code=500,
                detail="Model files not found. Please train models first by making a POST request to /train endpoint."
            )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=TrainingResult)
async def train_models():
    """
    Endpoint to train the machine learning models from scratch
    """
    try:
        # Train Take/Keep model
        take_keep_model = train_take_keep_classifier()
        
        # Train Toxic Gas model
        toxic_gas_model = train_toxic_gas_classifier()
        
        # Return success message
        return {
            "message": "Models trained successfully and saved to disk.",
            "accuracy": 0.95  # Example accuracy, this will be replaced with actual metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """
    Endpoint to get information about the trained models
    """
    # Check if model files exist
    take_keep_exists = os.path.exists("take_keep_classifier.pkl")
    toxic_gas_exists = os.path.exists("toxic_gas_model.pkl")
    
    if not take_keep_exists or not toxic_gas_exists:
        return {
            "status": "not_trained",
            "message": "Models not found. Please train models first."
        }
    
    return {
        "status": "trained",
        "models": [
            {
                "name": "Take/Keep Classifier",
                "description": "Binary SVM classifier to decide whether to take or keep an item based on gas value, distance, and weight.",
                "features": ["gas_value", "distance", "weight"],
                "file": "take_keep_classifier.pkl"
            },
            {
                "name": "Toxic Gas Classifier",
                "description": "Binary SVM classifier to detect toxic gas based on fill level, weight, and gas concentration.",
                "features": ["fill_level", "weight", "gas_concentration"],
                "file": "toxic_gas_model.pkl"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)