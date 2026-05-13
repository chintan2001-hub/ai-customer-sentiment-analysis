# AI-Based Customer Sentiment Analysis Platform

## Project Overview

This project is a school-level machine learning and business intelligence application developed using real Amazon customer review data.

The platform analyzes customer reviews and predicts customer sentiment as:

- Positive
- Neutral
- Negative

The project demonstrates:
- Data cleaning and preprocessing
- Natural Language Processing (NLP)
- Machine learning model training
- API development using FastAPI
- Dashboard visualization using Streamlit
- Logging and project structure organization

---

## Project Objective

The main goal of this project is to analyze customer review data and understand customer opinions about products using machine learning techniques.

The system processes customer reviews, cleans the text data, trains sentiment classification models, and predicts sentiment dynamically through a dashboard and API.

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Main programming language |
| Pandas | Data cleaning and manipulation |
| Scikit-learn | Machine learning model training |
| TF-IDF Vectorizer | Text feature extraction |
| Logistic Regression | Sentiment classification |
| Naive Bayes | Model comparison |
| Streamlit | Interactive dashboard |
| FastAPI | Backend API |
| Plotly | Data visualization |
| Joblib | Model saving/loading |
| Logging | Application monitoring |

---

## Dataset Information

Dataset used:
Amazon Consumer Reviews Dataset from Kaggle

Dataset Link:
https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products

Important columns used:
- reviews.text
- reviews.rating
- name
- categories
- reviews.title
- reviews.date

Note:
The original raw dataset was not uploaded to GitHub because the file size exceeded GitHub upload limitations.

---

## Sentiment Label Logic

Customer ratings were converted into sentiment labels using the following logic:

| Rating | Sentiment |
|--------|-----------|
| 1 - 2 | Negative |
| 3 | Neutral |
| 4 - 5 | Positive |

---

## Machine Learning Workflow

### Step 1 — Load Dataset
The Amazon customer review dataset is loaded using Pandas.

### Step 2 — Data Cleaning
The following preprocessing steps were applied:
- Remove missing values
- Remove duplicate reviews
- Convert text to lowercase
- Remove special characters
- Remove URLs
- Remove extra spaces

### Step 3 — Feature Engineering
TF-IDF Vectorizer was used to convert review text into numerical features for machine learning.

### Step 4 — Model Training
Two machine learning models were trained and compared:
- Logistic Regression
- Multinomial Naive Bayes

### Step 5 — Model Evaluation
Models were evaluated using:
- Accuracy Score
- Classification Report

### Step 6 — Model Deployment
The best model was saved using Joblib and integrated into:
- Streamlit dashboard
- FastAPI backend

---

## Dashboard Features

The Streamlit dashboard includes:
- Total review statistics
- Sentiment distribution charts
- Product category sentiment analysis
- Real-time customer review prediction
- Interactive visualizations
- Sample customer review display

---

## API Features

FastAPI backend provides:
- Real-time sentiment prediction
- REST API endpoint
- Swagger API testing interface

### API Endpoint

POST `/predict`

### Example Input

```json
{
  "text": "This product is amazing and very useful"
}
```

### Example Output

```json
{
  "predicted_sentiment": "Positive"
}
```

---

## Logging System

The project includes a logging system for tracking:
- Dataset loading
- Data cleaning process
- Model training
- API requests
- Prediction results
- Error tracking

Log files are stored inside:

```text
logs/app.log
```

---

## Project Structure

```text
AI_Public_Sentiment_Analysis/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── logs/
│
├── models/
│
├── src/
│   ├── logger.py
│   ├── data_cleaning.py
│   ├── train_model.py
│   └── predict.py
│
├── app.py
├── api.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run the Project

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Run Data Cleaning

```bash
python src/data_cleaning.py
```

### Step 3 — Train Machine Learning Model

```bash
python src/train_model.py
```

### Step 4 — Run Streamlit Dashboard

```bash
streamlit run app.py
```

### Step 5 — Run FastAPI Backend

```bash
uvicorn api:app --reload
```

---

## API Documentation

After running FastAPI, open:

```text
http://127.0.0.1:8000/docs
```

---

## Sample Prediction Examples

| Customer Review | Predicted Sentiment |
|----------------|-------------------|
| This product is amazing | Positive |
| I hate this product | Negative |
| The product is okay | Neutral |

---

## Future Improvements

Possible future improvements:
- Deep learning models
- Real-time social media sentiment analysis
- Cloud deployment
- User authentication
- Database integration
- Advanced NLP preprocessing
- Real-time streaming analytics

---

## Project Type

Academic / School-Level Machine Learning Project

---

## Author

Chintan Domadiya

Graduate Certificate in Business Intelligence & Systems Infrastructure  
Algonquin College
