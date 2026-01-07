import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Next Word Prediction using LSTM")
st.caption("Trained LSTM language model with tokenizer integration")

# -------------------------------
# Load Model & Tokenizer (Cached)
# -------------------------------
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_text_generator.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pickle", "rb") as f:
        return pickle.load(f)

model = load_lstm_model()
tokenizer = load_tokenizer()

# -------------------------------
# Helper Function
# -------------------------------
def predict_next_word(model, tokenizer, text, top_k=1):
    if not text.strip():
        return []

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return []

    max_sequence_len = model.input_shape[1]
    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len,
        padding="pre"
    )

    predictions = model.predict(token_list, verbose=0)[0]

    top_indices = predictions.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        word = tokenizer.index_word.get(idx)
        confidence = float(predictions[idx])
        if word:
            results.append((word, confidence))

    return results

# -------------------------------
# UI Inputs
# -------------------------------
input_text = st.text_input(
    "Enter text sequence",
    value="To be or not to",
    help="Enter a sequence of words to predict the next word."
)

top_k = st.slider(
    "Number of predictions",
    min_value=1,
    max_value=5,
    value=3,
    help="Select how many possible next words to display."
)

# -------------------------------
# UI Output
# -------------------------------
if st.button("🔮 Predict Next Word"):
    predictions = predict_next_word(model, tokenizer, input_text, top_k)

    if not predictions:
        st.warning("⚠️ Unable to generate prediction. Try a different input.")
    else:
        st.subheader("📌 Predictions")
        for i, (word, confidence) in enumerate(predictions, start=1):
            st.markdown(
                f"**{i}. {word}**  \n"
                f"Confidence: `{confidence:.4f}`"
            )

        # Best prediction highlighted
        best_word, best_conf = predictions[0]
        st.success(
            f"✅ **Most likely next word:** `{best_word}` "
            f"(confidence: {best_conf:.4f})"
        )

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit · TensorFlow · LSTM Language Model")
