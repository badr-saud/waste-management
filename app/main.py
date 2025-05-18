from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
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
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("waste-management-api")

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our prediction module
from prediction import predict_classification, train_and_predict_once
from train_models import train_take_keep_classifier, train_toxic_gas_classifier

# Initialize FastAPI app
app = FastAPI(
    title="Sensor Data Classification API",
    description="API for classifying sensor data using machine learning models",
    version="1.0.0",
)

# Configure CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
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


# Add middleware to log all requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    # Log request body for POST requests
    if request.method == "POST":
        try:
            body = await request.body()
            if body:
                logger.info(f"Request body: {body.decode()}")
        except Exception as e:
            logger.error(f"Error reading request body: {e}")

    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


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
    logger.info(f"Received prediction request with data: {data.values}")
    try:
        # Validate input
        if len(data.values) != 3:
            logger.error("Invalid input: not exactly 3 values")
            raise HTTPException(
                status_code=400,
                detail="Input must contain exactly 3 values: [gas_value/fill_level, distance/weight, weight/gas_concentration]",
            )

        # Make prediction
        result = predict_classification(data.values)
        logger.info(f"Prediction result: {result}")
        return result

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Check if model files exist, if not suggest training
        model_files = Path(".").glob("*.pkl")
        if not list(model_files):
            raise HTTPException(
                status_code=500,
                detail="Model files not found. Please train models first by making a POST request to /train endpoint.",
            )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=TrainingResult)
async def train_models():
    """
    Endpoint to train the machine learning models from scratch
    """
    logger.info("Received training request")
    try:
        # Train Take/Keep model
        take_keep_model = train_take_keep_classifier()

        # Train Toxic Gas model
        toxic_gas_model = train_toxic_gas_classifier()

        # Return success message
        logger.info("Models trained successfully")
        return {
            "message": "Models trained successfully and saved to disk.",
            "accuracy": 0.95,  # Example accuracy, this will be replaced with actual metrics
        }
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def get_model_info():
    """
    Endpoint to get information about the trained models
    """
    logger.info("Received model info request")
    # Check if model files exist
    take_keep_exists = os.path.exists("take_keep_classifier.pkl")
    toxic_gas_exists = os.path.exists("toxic_gas_model.pkl")

    if not take_keep_exists or not toxic_gas_exists:
        logger.warning("Models not found")
        return {
            "status": "not_trained",
            "message": "Models not found. Please train models first.",
        }

    logger.info("Returning model info")
    return {
        "status": "trained",
        "models": [
            {
                "name": "Take/Keep Classifier",
                "description": "Binary SVM classifier to decide whether to take or keep an item based on gas value, distance, and weight.",
                "features": ["gas_value", "distance", "weight"],
                "file": "take_keep_classifier.pkl",
            },
            {
                "name": "Toxic Gas Classifier",
                "description": "Binary SVM classifier to detect toxic gas based on fill level, weight, and gas concentration.",
                "features": ["fill_level", "weight", "gas_concentration"],
                "file": "toxic_gas_model.pkl",
            },
        ],
    }


@app.post("/auto-predict")
async def auto_predict(request: Request):
    """
    Flexible endpoint that accepts sensor data in various formats.
    Designed to be compatible with Arduino clients and other IoT devices.
    Trains models and immediately returns prediction results.
    """
    logger.info("Received auto-predict request")
    try:
        # Get raw data
        body = await request.body()
        logger.info(f"Raw request body: {body}")

        # Try to parse as JSON
        try:
            import json

            data = json.loads(body)
            if (
                isinstance(data, dict)
                and "values" in data
                and isinstance(data["values"], list)
            ):
                values = data["values"]
            else:
                # If it's not in the expected format, try to extract values
                if isinstance(data, dict):
                    values = list(data.values())[:3]  # Take first 3 values
                elif isinstance(data, list):
                    values = data[:3]  # Take first 3 values
                else:
                    values = [
                        float(val)
                        for val in str(data)
                        .replace("[", "")
                        .replace("]", "")
                        .split(",")[:3]
                    ]
        except Exception as json_error:
            logger.error(f"JSON parsing error: {json_error}")
            # If JSON parsing fails, try to extract numbers from the string
            try:
                import re

                values = [
                    float(val)
                    for val in re.findall(r"[-+]?\d*\.\d+|\d+", body.decode())[:3]
                ]
            except Exception as regex_error:
                logger.error(f"Regex extraction error: {regex_error}")
                raise HTTPException(
                    status_code=400, detail="Could not parse input data"
                )

        # Ensure we have exactly 3 values
        if len(values) != 3:
            logger.warning(f"Wrong number of values: {len(values)}")
            # If we don't have 3 values, pad or truncate
            if len(values) < 3:
                values = values + [0] * (3 - len(values))
            else:
                values = values[:3]

        logger.info(f"Extracted values: {values}")

        # Process using our existing function
        result = train_and_predict_once(values)
        logger.info(f"Auto-predict result: {result}")

        return JSONResponse(
            content={
                "Action": result.get("Action", "unknown"),
                "Gas": result.get("Gas", "unknown"),
                "message": "Models trained and prediction completed.",
            },
            headers={"Content-Type": "application/json"},
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Auto-predict error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Auto prediction failed: {str(e)}")


# Add a simple health check endpoint
@app.get("/health")
async def health_check():
    """Endpoint to check if the API is running"""
    return {"status": "healthy", "message": "API is running"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info")
