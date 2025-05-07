# app/predictor.py

from app.ml_model import SensorModel

# Instantiate once (same as in main.py)
model = SensorModel()

def predict_action_and_gas(input_vector):
    return model.predict(input_vector)

