import pickle
import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

import nltk
nltk.download('stopwords')

# Load artifacts
cv = pickle.load(open('countVectorizer.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
model_xgb = pickle.load(open('model_xgb.pkl', 'rb'))

def preprocess_input(review):
    stemmer = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)

    # Transform using Count Vectorizer
    features = cv.transform([review]).toarray()

    # Scale the features
    scaled_features = scaler.transform(features)

    return scaled_features

@st.cache
def predict_sentiment(review):
    processed_review = preprocess_input(review)
    prediction = model_xgb.predict(processed_review)
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment

st.title("Sentiment Analysis for Amazon Alexa Reviews")

user_input = st.text_area("Enter your review:")

if st.button("Analyze"):
    sentiment = predict_sentiment(user_input)
    st.write(f"Predicted Sentiment: {sentiment}")

