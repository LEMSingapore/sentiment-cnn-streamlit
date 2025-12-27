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
import nltk
import ssl

# -----------------------------
# Page config & global styling
# -----------------------------
st.set_page_config(
    page_title="CNN Sentiment & Text Summarizer",
    page_icon="💬",
    layout="wide",
)

# Custom CSS to mimic the clean, card-based look of changcheeyoung.github.io
CUSTOM_CSS = """
<style>
:root {
  --bg-page: #f3f4f6;
  --bg-card: #ffffff;
  --text-main: #111827;
  --text-muted: #6b7280;
  --accent: #2563eb;
}

/* Overall page */
html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg-page) !important;
  color: var(--text-main) !important;
}

/* Make all standard text dark for contrast */
h1, h2, h3, h4, h5, h6,
p, label, span, div,
.stMarkdown, .stMarkdown p {
  color: var(--text-main);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen,
               Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
}

/* Main content padding */
.block-container {
  padding-top: 2.5rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background-color: #111827;
}

[data-testid="stSidebar"] * {
  color: #e5e7eb !important;
}

/* Links / accents */
a, .stMarkdown a {
  color: var(--accent);
}

/* Cards */
.app-card {
  background-color: var(--bg-card);
  padding: 1.5rem 1.75rem;
  border-radius: 1rem;
  box-shadow: 0 16px 40px rgba(15, 23, 42, 0.12);
  border: 1px solid #e5e7eb;
}

/* Section titles inside cards */
.app-section-title {
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

/* Muted text */
.app-muted {
  color: var(--text-muted) !important;
  font-size: 0.9rem;
}

/* Result labels */
.app-badge {
  display: inline-block;
  padding: 0.35rem 0.7rem;
  border-radius: 999px;
  background-color: #e5f3ff;
  color: var(--accent);
  font-size: 0.8rem;
  font-weight: 600;
}

/* Sentiment pills */
.sentiment-positive {
  background-color: #dcfce7;
  color: #166534;
  padding: 0.4rem 0.8rem;
  border-radius: 999px;
  font-weight: 600;
  display: inline-block;
}

.sentiment-negative {
  background-color: #fee2e2;
  color: #b91c1c;
  padding: 0.4rem 0.8rem;
  border-radius: 999px;
  font-weight: 600;
  display: inline-block;
}

/* Small score text */
.sentiment-score {
  font-size: 0.85rem;
  color: var(--text-muted);
  margin-left: 0.5rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# NLTK setup
# -----------------------------
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("### Chang Chee Young | AI & ML Developer")
st.title("Text Summarization & Sentiment Analysis")
st.markdown(
    "<p class='app-muted'>Deep learning NLP sentiment classifier with automatic "
    "text summarization – built with TensorFlow/Keras & Streamlit.</p>",
    unsafe_allow_html=True,
)

# -----------------------------
# Load model and tokenizer
# -----------------------------
@st.cache_resource
def load_sentiment_model():
    model = load_model("models/sentiment_cnn.h5")
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

# -----------------------------
# Preprocessing
# -----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)  # remove HTML tags
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Sentiment Prediction
# -----------------------------
def predict_sentiment(text: str):
    clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(padded)
    pred_value = float(pred[0][0])
    sentiment = "Positive" if pred_value >= 0.5 else "Negative"
    return sentiment, pred_value


# -----------------------------
# Summarization
# -----------------------------
def summarize_text(text: str, sentence_count: int = 3) -> str:
    parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)


# -----------------------------
# Layout
# -----------------------------
with st.sidebar:
    st.markdown("#### App Info")
    st.markdown(
        "- Built with **TensorFlow/Keras**\n"
        "- Deployed via **Streamlit**\n"
        "- Uses **LexRank** summarization\n"
        "- Part of my portfolio on `changcheeyoung.github.io`"
    )
    st.markdown("---")
    st.markdown("#### How to use")
    st.markdown(
        "1. Paste or type a review or paragraph.\n"
        "2. View the auto-generated **summary**.\n"
        "3. Inspect the **sentiment score**."
    )

# Two-column main layout: left input, right results
col_input, col_output = st.columns([1.1, 1.1])

with col_input:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("<div class='app-section-title'>📝 Input Text</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='app-muted'>Enter a review, comment or paragraph to summarize and analyse.</p>",
        unsafe_allow_html=True,
    )
    input_text = st.text_area(
        label="",
        height=280,
        placeholder="Paste or type your review text here...",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_output:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("<div class='app-section-title'>📊 Results</div>", unsafe_allow_html=True)

    if input_text:
        # Summary
        st.markdown("<span class='app-badge'>Summary</span>", unsafe_allow_html=True)
        summary = summarize_text(input_text)
        st.write(summary)

        st.markdown("---")

        # Sentiment
        sentiment, score = predict_sentiment(input_text)
        if sentiment == "Positive":
            pill_class = "sentiment-positive"
        else:
            pill_class = "sentiment-negative"

        sentiment_html = (
            f"<span class='{pill_class}'>{sentiment}</span>"
            f"<span class='sentiment-score'>Model score: {score:.2f}</span>"
        )
        st.markdown("<span class='app-badge'>Sentiment</span>", unsafe_allow_html=True)
        st.markdown(sentiment_html, unsafe_allow_html=True)

    else:
        st.markdown(
            "<p class='app-muted'>Results will appear here once you enter some text on the left.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

