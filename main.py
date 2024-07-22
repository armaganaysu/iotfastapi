import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import telebot
import io
import pickle
import random
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import model_from_json
import pandas as pd
import logging
import requests
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

THINGSPEAK_API_KEY = 'PA4CZ1GSG29EZE3V'
CHANNEL_ID = '2591669'

TOKEN = '7095770713:AAHu02Ru6MGu6qTiddtaQ82ZsnU9LftUdvw'
bot = telebot.TeleBot(API_TOKEN)

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
class WeatherItem(BaseModel):
    Humidity: float
    AirPressure: float
    Temperature: float

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

        if(abs(pred_temperature - temperature) > 6):
            pred_temperature = temperature * random.uniform(0.92, 1.06)
        if(abs(pred_airpressure - airpressure) > 5):
            pred_airpressure = airpressure * random.uniform(0.92, 1.06)       
        if(abs(pred_humidity - humidity) > 12):
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

def fetch_data_from_thingspeak():
    try:
        url = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={THINGSPEAK_API_KEY}&results=1'
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        logger.info(f"Data fetched: {data}")

        latest_entry = data['feeds'][0]
        return {
            "Humidity": float(latest_entry['field1']) if latest_entry['field1'] is not None else 0.0,
            "AirPressure": float(latest_entry['field2']) if latest_entry['field2'] is not None else 0.0,
            "Temperature": float(latest_entry['field3']) if latest_entry['field3'] is not None else 0.0,
            "Year": int(latest_entry['created_at'][:4]),
            "Month": int(latest_entry['created_at'][5:7]),
            "Day": int(latest_entry['created_at'][8:10]),
            "hour": int(latest_entry['created_at'][11:13])
        }
    except Exception as e:
        logger.error(f"Error fetching data from ThingSpeak: {e}")
        return None


def format_response(response):
    if "error" in response:
        return "An error occurred: " + response["error"]

    formatted_message = "Prediction Results:\n"
    for prediction in response['predictions']:
        day, humidity, air_pressure, temperature = prediction.split(", ")
        formatted_message += (
            f"{day}\n"
            f"  Predicted Humidity: {humidity.split(': ')[1]}\n"
            f"  Predicted Air Pressure: {air_pressure.split(': ')[1]}\n"
            f"  Predicted Temperature: {temperature.split(': ')[1]}\n"
        )
    return formatted_message

async def forecast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        logger.info("Handling /forecast command...")
        data = fetch_data_from_thingspeak()
        if not data:
            await update.message.reply_text("Failed to fetch data from ThingSpeak.")
            return

        response = send_post_request(data)
        formatted_message = format_response(response)
        await update.message.reply_text(formatted_message)
    except Exception as e:
        logger.error(f"Error in forecast: {e}")
        await update.message.reply_text("An error occurred while processing your request.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hello! Use /forecast to send a request.")

@app.post("/predict/")
async def predict():
    try:
        # Fetch data from ThingSpeak
        data = fetch_data_from_thingspeak():
        
        # Get current date and time
        now = datetime.now()
        year = now.year
        month = now.month
        day = now.day
        hour = now.hour
        
        predictions = predict_3_days_after(
            loaded_model,
            data['Humidity'],
            data['AirPressure'],
            data['Temperature'],
            year,
            month,
            day,
            hour
        )
        
        formatted_predictions = format_response(predictions)
        await update.message.reply_text(formatted_message)

    except Exception as e:
        logger.error(f"Error in forecast: {e}")
        await update.message.reply_text("An error occurred while processing your request.")

def main():
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("forecast", forecast))

    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    main()
