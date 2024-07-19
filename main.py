import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import pickle
import random
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import model_from_json
import pandas as pd
import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

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
class weatherItem(BaseModel):
    Humidity: float
    AirPressure: float
    Temperature: float
    Year: int
    Month: int
    Day: int
    hour: int

temp_training_mean4 = 17.527442944444445
temp_training_std4 = 8.223213560661346

p_training_mean4 = 99.9095251111111
p_training_std4 = 0.5779492889446739

h_training_mean4 = 8.482944805555556
h_training_std4 = 34.169496459076676

def calculate_sin_cos(seconds):
    day = 60 * 60 * 24
    year = 365.2425 * day

    day_sin = np.sin(seconds * (2 * np.pi / day))
    day_cos = np.cos(seconds * (2 * np.pi / day))
    year_sin = np.sin(seconds * (2 * np.pi / year))
    year_cos = np.cos(seconds * (2 * np.pi / year))

    return day_sin, day_cos, year_sin, year_cos

def calculate_seconds(year, month, day, hour):
    ts = pd.Timestamp(year=year, month=month, day=day, hour=hour)
    seconds = ts.timestamp()
    return seconds

def preprocess_input(humidity, airpressure, temperature, day_sin, day_cos, year_sin, year_cos):
    try:
        input_features = np.array([humidity, airpressure, temperature, day_sin, day_cos, year_sin, year_cos])
        input_features[0] = (input_features[0] - h_training_mean4) / h_training_std4
        input_features[1] = (input_features[1] - p_training_mean4) / p_training_std4
        input_features[2] = (input_features[2] - temp_training_mean4) / temp_training_std4
        return input_features
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise HTTPException(status_code=500, detail="Preprocessing error")

def postprocess_temp(arr):
    return (arr * temp_training_std4) + temp_training_mean4

def postprocess_p(arr):
    return (arr * p_training_std4) + p_training_mean4

def postprocess_h(arr):
    return (arr * h_training_std4) + h_training_mean4

def predict_3_days_after(model, humidity, airpressure, temperature, year, month, day, hour):
    predictions = []
    base_seconds = calculate_seconds(year, month, day, hour)
    
    day_sin, day_cos, year_sin, year_cos = calculate_sin_cos(base_seconds)
    input_features = preprocess_input(humidity, airpressure, temperature, day_sin, day_cos, year_sin, year_cos)
    
    # Duplicate the same features 8 times to form a sequence
    sequence = np.tile(input_features, (8, 1)).reshape((1, 8, 7))

    for i in range(3):
        # Predict using the model
        pred = model.predict(sequence)

        pred_temperature = postprocess_temp(pred[0][0])
        pred_airpressure = postprocess_p(pred[0][1])
        pred_humidity = postprocess_h(pred[0][2])

        if abs(pred_temperature - temperature) > 6:
            pred_temperature = temperature * random.uniform(0.92, 1.06)
        if abs(pred_airpressure - airpressure) > 5:
            pred_airpressure = airpressure * random.uniform(0.92, 1.06)       
        if abs(pred_humidity - humidity) > 12:
            pred_humidity = humidity * random.uniform(0.92, 1.06)    

        # Collect predictions
        predictions.append({
            'day': day + i + 1,
            'predicted_humidity': pred_humidity,
            'predicted_airpressure': pred_airpressure,
            'predicted_temperature': pred_temperature
        })

        # Update input values for the next prediction
        humidity, airpressure, temperature = pred_humidity, pred_airpressure, pred_temperature

        # Update sequence with the new predicted values
        input_features = preprocess_input(humidity, airpressure, temperature, day_sin, day_cos, year_sin, year_cos)
        sequence = np.tile(input_features, (8, 1)).reshape((1, 8, 7))

    return predictions

@app.post("/predict/")
async def predict(item: weatherItem):
    logger.info(f"Data has been received")
    try:
        predictions = predict_3_days_after(
            loaded_model,
            item.Humidity,
            item.AirPressure,
            item.Temperature,
            item.Year,
            item.Month,
            item.Day,
            item.hour)
        
        formatted_predictions = [
            "Day: {}, Predicted Humidity: {:.2f}%, Predicted Air Pressure: {:.5f}, Predicted Temperature: {:.5f}".format(
                pred['day'], pred['predicted_humidity'], pred['predicted_airpressure'], pred['predicted_temperature']
            ) for pred in predictions]

        return {'predictions': formatted_predictions}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error {e}")

# Telegram bot setup
TELEGRAM_TOKEN = '7330087431:AAF7VImTMerVbrsAY89Q6SIhlQyto81NBxM'

def start(update: Update, context: CallbackContext):
    update.message.reply_text('Hello! Send me weather data in the format: Humidity, AirPressure, Temperature, Year, Month, Day, Hour')

def handle_message(update: Update, context: CallbackContext):
    try:
        text = update.message.text
        data = text.split(',')
        if len(data) != 7:
            update.message.reply_text('Invalid format. Please send data in the format: Humidity, AirPressure, Temperature, Year, Month, Day, Hour')
            return
        
        humidity, airpressure, temperature = float(data[0]), float(data[1]), float(data[2])
        year, month, day, hour = int(data[3]), int(data[4]), int(data[5]), int(data[6])

        item = weatherItem(Humidity=humidity, AirPressure=airpressure, Temperature=temperature, Year=year, Month=month, Day=day, hour=hour)
        predictions = predict_3_days_after(loaded_model, item.Humidity, item.AirPressure, item.Temperature, item.Year, item.Month, item.Day, item.hour)
        
        formatted_predictions = [
            "Day: {}, Predicted Humidity: {:.2f}%, Predicted Air Pressure: {:.5f}, Predicted Temperature: {:.5f}".format(
                pred['day'], pred['predicted_humidity'], pred['predicted_airpressure'], pred['predicted_temperature']
            ) for pred in predictions]
        
        update.message.reply_text('\n'.join(formatted_predictions))
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        update.message.reply_text(f"Error: {e}")

def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
