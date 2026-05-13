import pandas as pd
import os
import joblib
from src.logger import logger
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def main():
    data_path = "data/processed/cleaned_amazon_reviews.csv"
    logger.info("Loading cleaned dataset")
    df = pd.read_csv(data_path)

    X = df["clean_text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Better model for imbalanced sentiment data
    logger.info("Training Logistic Regression model")
    logistic_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2)
        )),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    # Simple comparison model
    nb_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2)
        )),
        ("model", MultinomialNB())
    ])

    logistic_model.fit(X_train, y_train)
    logistic_pred = logistic_model.predict(X_test)
    logistic_accuracy = accuracy_score(y_test, logistic_pred)
    logger.info("Training Naive Bayes model")
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_pred)

    print("Logistic Regression Accuracy:", round(logistic_accuracy, 4))
    print("Naive Bayes Accuracy:", round(nb_accuracy, 4))

    # Force Logistic Regression as final model because it handles balance better
    best_model = logistic_model
    best_model_name = "Balanced Logistic Regression"
    best_accuracy = logistic_accuracy
    best_pred = logistic_pred

    print("\nBest Model:", best_model_name)
    print("Best Accuracy:", round(best_accuracy, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, best_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/sentiment_model.pkl")
    logger.info("Saving trained model")
    with open("models/model_results.txt", "w") as file:
        file.write(f"Best Model: {best_model_name}\n")
        file.write(f"Best Accuracy: {round(best_accuracy, 4)}\n\n")
        file.write("Classification Report:\n")
        file.write(classification_report(y_test, best_pred))

    print("\nImproved model saved at: models/sentiment_model.pkl")
    logger.info(f"Best model selected: {best_model_name}")
    logger.info("Model training completed successfully")

if __name__ == "__main__":
    main()