import io
import pickle
import random
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import model_from_json
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model architecture and weights
try:
    logger.info("Loading model architecture...")
    with open('model_architecture.pkl', 'rb') as f:
        model_json = pickle.load(f)
    loaded_model = model_from_json(model_json)
    logger.info("Model architecture loaded successfully.")

    logger.info("Loading model weights...")
    loaded_model.load_weights('model_weights.h5')
    logger.info("Model weights loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Model loading error")

# Define the data model
class WeatherData(BaseModel):
    Humidity: float
    AirPressure: float
    Temperature: float
    Year: int
    Month: int
    Day: int
    hour: int


temp_training_mean4 =17.527442944444445
temp_training_std4 =8.223213560661346

p_training_mean4=99.9095251111111
p_training_std4=0.5779492889446739

h_training_mean4=8.482944805555556
h_training_std4=34.169496459076676

def calculate_sin_cos(seconds):
    day_in_seconds = 24 * 60 * 60
    year_in_seconds = 365.25 * day_in_seconds
    day_sin = np.sin(2 * np.pi * (seconds % day_in_seconds) / day_in_seconds)
    day_cos = np.cos(2 * np.pi * (seconds % day_in_seconds) / day_in_seconds)
    year_sin = np.sin(2 * np.pi * (seconds % year_in_seconds) / year_in_seconds)
    year_cos = np.cos(2 * np.pi * (seconds % year_in_seconds) / year_in_seconds)
    return day_sin, day_cos, year_sin, year_cos

def preprocess_input(data):
    try:
        # Calculate the seconds and sinusoidal features
        ts = pd.Timestamp(year=data.Year, month=data.Month, day=data.Day, hour=data.hour)
        seconds = ts.timestamp()
        day_sin, day_cos, year_sin, year_cos = calculate_sin_cos(seconds)

        # Normalize input data
        normalized_data = [
            (data.Temperature - temp_training_mean4) / temp_training_std4,
            (data.AirPressure - p_training_mean4) / p_training_std4,
            (data.Humidity - h_training_mean4) / h_training_std4,
            day_sin, day_cos, year_sin, year_cos
        ]
        return np.array([normalized_data])
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise HTTPException(status_code=500, detail="Preprocessing error")

def postprocess_temp(pred):
    return pred * temp_training_std4 + temp_training_mean4

def postprocess_p(pred):
    return pred * p_training_std4 + p_training_mean4

def postprocess_h(pred):
    return pred * h_training_std4 + h_training_mean4

@app.post("/predict/")
async def predict(weather_data: WeatherData):
    logger.info(f"Received data: {weather_data}")
    try:
        input_data = preprocess_input(weather_data)
        prediction = loaded_model.predict(input_data)
        return {
            "prediction": {
                "temperature": postprocess_temp(prediction[0][0]),
                "air_pressure": postprocess_p(prediction[0][1]),
                "humidity": postprocess_h(prediction[0][2])
            }
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")
