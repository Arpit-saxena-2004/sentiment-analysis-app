import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define your text cleaning function (same as used during training)
def clean_text(text):
    import re
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special characters
    text = text.lower()
    return text.strip()

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter a product review and the model will predict whether it's **Positive** or **Negative**.")

user_input = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if user_input:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        sentiment = "Positive 😊" if prediction == 2 else "Negative 😞"
        st.success(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter some text.")
