import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import re
import os
from src.logger import logger

logger.info("Streamlit dashboard started")
st.set_page_config(page_title="AI Customer Sentiment Analysis", layout="wide")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("models/sentiment_model.pkl")

# Clean text before prediction
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("AI-Based Customer Sentiment Analysis Platform")
st.write("This project analyzes Amazon customer reviews and predicts customer sentiment.")

if not os.path.exists("models/sentiment_model.pkl"):
    st.error("Model not found. Please run data cleaning and model training first.")
    st.stop()

model = load_model()

data_path = "data/processed/cleaned_amazon_reviews.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)

    st.subheader("Project Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", len(df))
    col2.metric("Positive Reviews", len(df[df["sentiment"] == "Positive"]))
    col3.metric("Neutral Reviews", len(df[df["sentiment"] == "Neutral"]))
    col4.metric("Negative Reviews", len(df[df["sentiment"] == "Negative"]))

    st.subheader("Sentiment Distribution")

    sentiment_count = df["sentiment"].value_counts().reset_index()
    sentiment_count.columns = ["Sentiment", "Count"]

    fig = px.bar(
        sentiment_count,
        x="Sentiment",
        y="Count",
        title="Overall Sentiment Count"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.pie(
        sentiment_count,
        names="Sentiment",
        values="Count",
        title="Overall Sentiment Percentage"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sentiment by Product Category")

    category_data = df.groupby(["category", "sentiment"]).size().reset_index(name="count")
    top_categories = df["category"].value_counts().head(10).index
    category_data = category_data[category_data["category"].isin(top_categories)]

    fig3 = px.bar(
        category_data,
        x="category",
        y="count",
        color="sentiment",
        title="Sentiment by Top Product Categories"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Sample Review Data")
    st.dataframe(df[["product_name", "rating", "sentiment", "review_text"]].head(50))

else:
    st.warning("Cleaned dataset not found. Please run src/data_cleaning.py first.")

st.subheader("Try Your Own Review")

user_review = st.text_area("Enter a customer review:")
logger.info(f"User entered review: {user_review}")
if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter some review text.")
    else:
        cleaned_review = clean_text(user_review)
        prediction = model.predict([cleaned_review])[0]
        logger.info(f"Dashboard prediction result: {prediction}")
        st.success(f"Predicted Sentiment: {prediction}")