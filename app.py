import streamlit as st
import pickle
import re
import string

# Load the vectorizer and model
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def predict_news(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    processed_text = vectorizer.transform([text])
    prediction = model.predict(processed_text)[0]
    return "Fake News" if prediction == 0 else "Not Fake News"

# Streamlit interface
st.title("Fake News Detector")
user_input = st.text_area("Enter the news text:")

if st.button("Predict"):
    result = predict_news(user_input)
    st.write(f"Prediction: {result}")