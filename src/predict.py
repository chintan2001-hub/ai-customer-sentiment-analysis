import joblib
import re

# Load saved model
model = joblib.load("models/sentiment_model.pkl")

# Same cleaning function used during training
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# This function predicts sentiment for new text
def predict_sentiment(text):
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    return prediction