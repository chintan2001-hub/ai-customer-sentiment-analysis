# AI-Based Customer Sentiment Analysis Platform

## Project Overview

This project is a school-level machine learning and business intelligence application developed using real Amazon customer review data.

The platform analyzes customer reviews and predicts sentiment as:
- Positive
- Neutral
- Negative

The project demonstrates:
- Data cleaning
- Natural language processing (NLP)
- Machine learning model training
- API development
- Dashboard visualization

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- FastAPI
- Plotly
- Joblib

---

## Dataset

Dataset used:
Amazon Consumer Reviews Dataset from Kaggle

Main columns used:
- reviews.text
- reviews.rating
- name
- categories
- reviews.title

---

## Machine Learning Workflow

1. Load raw Amazon review dataset.
2. Clean review text.
3. Remove duplicates and missing values.
4. Convert ratings into sentiment labels.
5. Apply TF-IDF vectorization.
6. Train Logistic Regression and Naive Bayes models.
7. Compare model performance.
8. Save best model for prediction.

---

## Sentiment Label Logic

| Rating | Sentiment |
|--------|-----------|
| 1-2 | Negative |
| 3 | Neutral |
| 4-5 | Positive |

---

## Dashboard Features

- Sentiment distribution visualization
- Product category analysis
- Real-time review prediction
- Interactive charts
- Customer review exploration

---

## API Features

FastAPI backend provides:
- Real-time sentiment prediction
- REST API endpoint
- Swagger API testing

API Endpoint:

POST `/predict`

Example Input:

```json
{
  "text": "This product is amazing"
}
