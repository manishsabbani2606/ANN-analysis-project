# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

#Load the model
model = load_model('simplernn_imdb_model.h5')


# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500, padding='post')
    return padded_review


def predict_sentiment(review):
    preprocessed_review = preprocess_text(review)
    prediction = model.predict(preprocessed_review)
    sentiment = "Positive" if prediction [0][0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]

import streamlit as st
import numpy as np

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ------------------ Header ------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🎬 IMDB Movie Review Sentiment Analysis</h1>
    <p style='text-align: center; color: gray;'>
        Analyze movie reviews using Deep Learning (RNN-based model)
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ------------------ Instructions ------------------
st.markdown("### ✍️ Enter a Movie Review")
st.write("The model will classify the review as **Positive** or **Negative**.")

# ------------------ User Input ------------------
user_input = st.text_area(
    label="Movie Review",
    height=180,
    placeholder="Type or paste a movie review here..."
)

# ------------------ Predict Button ------------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

# ------------------ Prediction Logic ------------------
if predict_btn:

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a movie review before analyzing.")
    else:
        with st.spinner("🧠 Analyzing sentiment..."):
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)

            score = float(prediction[0][0])
            sentiment = "Positive 😊" if score > 0.5 else "Negative 😞"

        st.divider()

        # ------------------ Result Display ------------------
        if score > 0.5:
            st.success(f"### 🎉 Sentiment: {sentiment}")
        else:
            st.error(f"### 💔 Sentiment: {sentiment}")

        # ------------------ Confidence Score ------------------
        st.markdown("### 📊 Prediction Confidence")
        st.progress(score if score > 0.5 else 1 - score)

        st.metric(
            label="Confidence Score",
            value=f"{score:.2f}" if score > 0.5 else f"{1 - score:.2f}"
        )

        # ------------------ Interpretation ------------------
        st.info(
            "💡 **How to read this:**\n\n"
            "- Score close to **1.0** → Very Positive\n"
            "- Score close to **0.0** → Very Negative\n"
            "- Around **0.5** → Neutral / Uncertain"
        )

# ------------------ Footer ------------------
st.divider()
st.markdown(
    """
    <p style='text-align: center; font-size: 0.85em; color: gray;'>
        Built with ❤️ using Streamlit & Deep Learning<br>
        IMDB Sentiment Analysis Project
    </p>
    """,
    unsafe_allow_html=True
)
