from fastapi import FastAPI
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.post("/predict/")
async def predict(data: dict):
    logger.info(f"Received data: {data}")
    return {"prediction": "dummy prediction"}
