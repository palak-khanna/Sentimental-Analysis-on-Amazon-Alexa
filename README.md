# **Sentiment Analysis of Amazon Alexa Reviews**

This project performs sentiment analysis on Amazon Alexa product reviews using machine learning models. The application processes customer feedback and predicts whether a review is positive or negative. It includes data preprocessing, feature extraction, model training, and a user-friendly interface built with Streamlit.

---

## **Features**

- **Preprocessing**: Text cleaning, stemming, and stopwords removal.
- **Feature Extraction**: Bag of Words representation using `CountVectorizer`.
- **Machine Learning Model**: Sentiment prediction using XGBoost Classifier.
- **Web Interface**: User-friendly interface for inputting reviews and getting sentiment predictions.
- **Scalable Pipeline**: Easily adaptable to other datasets and sentiment analysis tasks.

---

## **Technologies Used**

- **Programming Language**: Python
- **Libraries**:
  - Natural Language Toolkit (NLTK)
  - Scikit-learn
  - XGBoost
  - Matplotlib
  - Streamlit
- **Pickle**: To save and load the model, vectorizer, and scaler.

---

## **Setup Instructions**

### **1. Clone the Repository**

- git clone https://github.com/your-username/repository-name.git
  cd repository-name

### **2. Clone the Repository**
Before proceeding, ensure that Python is installed on your system. Use the following commands to create and activate a virtual environment:

On Windows:

python -m venv venv
venv\Scripts\activate

On macOS/Linux:

python3 -m venv venv
source venv/bin/activate

### **3. Install Required Packages**

pip install -r requirements.txt

or

pip install numpy pandas nltk scikit-learn xgboost matplotlib streamlit


### **4.Demo**
<img width="832" alt="image" src="https://github.com/user-attachments/assets/f4b69584-e522-4ab2-8d5d-7603c6b38e7a">


### **Results**
1. Training Accuracy: Achieved high accuracy using the XGBoost classifier.
2. Confusion Matrix: Displays performance metrics.

### **Future Improvements**
- Support for multi-class sentiment analysis.
- Deployment to a cloud platform.
- Integration with more advanced NLP models (e.g., BERT).

### **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request.


