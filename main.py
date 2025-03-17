import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from io import BytesIO

# Load pre-trained models
xgb_model = pickle.load(open("./Models/model_xgb.pkl", "rb"))
scaler = pickle.load(open("./Models/scaler.pkl", "rb"))

# Text Preprocessing
def preprocess_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text).lower().split()
    return " ".join([stemmer.stem(word) for word in text])

# Single Prediction Function
def single_prediction(text_input):
    processed_text = [preprocess_text(text_input)]  # Preprocess input text
    
    # Load the fitted TF-IDF vectorizer
    with open("./Models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    # Debug: Check if vectorizer is fitted
    if not hasattr(tfidf_vectorizer, "idf_"):
        return "❌ ERROR: TF-IDF vectorizer is NOT fitted. Retrain and save it again."

    # Transform the input text using the trained vectorizer
    X_tfidf = tfidf_vectorizer.transform(processed_text).toarray()

    # Apply MinMax Scaling
    X_scaled = scaler.transform(X_tfidf)

    # Predict sentiment
    y_pred = xgb_model.predict_proba(X_scaled).argmax(axis=1)[0]

    return "Positive" if y_pred == 1 else "Negative"


# Bulk Prediction Function
def bulk_prediction(data):
    data["Processed_Text"] = data["Sentence"].apply(preprocess_text)
    X_scaled = scaler.transform(pd.DataFrame(data["Processed_Text"]))
    data["Predicted sentiment"] = xgb_model.predict_proba(X_scaled).argmax(axis=1).map({1: "Positive", 0: "Negative"})
    return data

# Visualization Function
def plot_distribution(data):
    fig, ax = plt.subplots()
    data["Predicted sentiment"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["green", "red"], startangle=90, shadow=True, ax=ax)
    st.pyplot(fig)

# Streamlit App UI
st.title("Sentiment Analysis App")
st.write("Analyze sentiments of customer reviews using a trained XGBoost model.")

# Single Text Prediction
st.subheader("Single Review Prediction")
text_input = st.text_area("Enter a review:")
if st.button("Predict Sentiment"):
    if text_input:
        sentiment = single_prediction(text_input)
        st.success(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review.")

# Bulk Prediction
st.subheader("Bulk Review Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV file with a 'Sentence' column", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    if "Sentence" in data.columns:
        predictions = bulk_prediction(data)
        st.write("Predictions:", predictions)
        plot_distribution(predictions)
        
        # Download link for results
        csv = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download Predictions", data=csv, file_name="Predictions.csv", mime="text/csv")
    else:
        st.error("CSV file must contain a 'Sentence' column.")