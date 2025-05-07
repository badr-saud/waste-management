# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.predictor import predict_action_and_gas
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

class SensorData(BaseModel):
    fill_level: float
    weight: float
    gas_concentration: float
@app.get("/")
def read_index():
    return FileResponse("static/index.html")

@app.post("/predict")
def predict(data: SensorData):
    vector = [data.fill_level, data.weight, data.gas_concentration]
    return predict_action_and_gas(vector)
