import streamlit as st
import pandas as pd
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Set title
st.title("Text Summarization & Sentiment Analysis App")

# Load model and tokenizer
@st.cache_resource
def load_sentiment_model():
    model = load_model("sentiment_cnn.h5")
    return model

@st.cache_resource
def load_tokenizer():
    import pickle
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

model = load_sentiment_model()
tokenizer = load_tokenizer()
maxlen = 200

# Preprocessing
def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Sentiment Prediction
def predict_sentiment(text):
    clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(padded)
    sentiment = "Positive" if pred >= 0.5 else "Negative"
    return sentiment, float(pred)

# Summarization
def summarize_text(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

# Input
input_text = st.text_area("Enter your review text:", height=300)

if input_text:
    st.subheader("Summary")
    summary = summarize_text(input_text)
    st.write(summary)

    st.subheader("Sentiment")
    sentiment, score = predict_sentiment(input_text)
    st.write(f"**Sentiment**: {sentiment} ({score:.2f})")
