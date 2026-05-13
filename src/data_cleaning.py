import pandas as pd
import re
import os
from src.logger import logger
# This function cleans review text
def clean_text(text):
    text = str(text).lower()  # convert text to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # remove links
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove numbers and special characters
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

# This function converts rating into sentiment
def rating_to_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

def main():
    raw_path = "data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
    output_path = "data/processed/cleaned_amazon_reviews.csv"

    # Load dataset
    logger.info("Loading raw dataset")
    df = pd.read_csv(raw_path)

    # Keep only useful columns for this project
    selected_columns = [
        "reviews.text",
        "reviews.rating",
        "name",
        "categories",
        "reviews.title",
        "reviews.date"
    ]

    df = df[selected_columns]

    # Rename columns to simple names
    df = df.rename(columns={
        "reviews.text": "review_text",
        "reviews.rating": "rating",
        "name": "product_name",
        "categories": "category",
        "reviews.title": "review_title",
        "reviews.date": "review_date"
    })

    # Remove missing values
    logger.info("Removing missing values")
    df = df.dropna(subset=["review_text", "rating"])

    # Remove duplicate reviews
    logger.info("Removing duplicate reviews")
    df = df.drop_duplicates(subset=["review_text"])

    # Clean review text
    df["clean_text"] = df["review_text"].apply(clean_text)

    # Create sentiment column from rating
    df["sentiment"] = df["rating"].apply(rating_to_sentiment)

    # Remove rows where cleaned text is empty
    df = df[df["clean_text"].str.len() > 0]

    # Create output folder if it does not exist
    os.makedirs("data/processed", exist_ok=True)

    # Save cleaned dataset
    logger.info("Saving cleaned dataset")
    df.to_csv(output_path, index=False)

    print("Data cleaning completed.")
    print("Cleaned file saved at:", output_path)
    print("Final rows:", len(df))
    print(df["sentiment"].value_counts())
    logger.info("Data cleaning completed successfully")
if __name__ == "__main__":
    main()