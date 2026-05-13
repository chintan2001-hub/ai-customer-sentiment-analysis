from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from src.logger import logger

app = FastAPI(title="AI Customer Sentiment Analysis API")

# Load model one time when API starts
model = joblib.load("models/sentiment_model.pkl")

class ReviewInput(BaseModel):
    text: str

# Clean input text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.get("/")
def home():
    return {"message": "AI Customer Sentiment Analysis API is running"}

@app.post("/predict")
def predict_sentiment(review: ReviewInput):
    logger.info(f"Prediction request received: {review.text}")
    cleaned_text = clean_text(review.text)
    prediction = model.predict([cleaned_text])[0]
    logger.info(f"Predicted sentiment: {prediction}")

    return {
        "input_text": review.text,
        "cleaned_text": cleaned_text,
        "predicted_sentiment": prediction
    }