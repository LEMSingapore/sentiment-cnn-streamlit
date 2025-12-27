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
/* Overall page */
body {
    background-color: #f5f5f5;
}

/* Main app background */
[data-testid="stAppViewContainer"] {
    background: #f5f5f5;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1f2933;
    color: #f9fafb;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #f9fafb !important;
}

/* Title */
h1, h2, h3 {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen,
                 Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
}

/* Accent color similar to Minimal Mistakes links */
a, .stMarkdown a {
    color: #1d72b8;
}

/* Cards */
.app-card {
    background-color: #ffffff;
    padding: 1.5rem 1.75rem;
    border-radius: 0.75rem;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
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
    color: #6b7280;
    font-size: 0.9rem;
}

/* Result labels */
.app-badge {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background-color: #e5f3ff;
    color: #1d72b8;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Sentiment pills */
.sentiment-positive {
    background-color: #dcfce7;
    color: #15803d;
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
    color: #6b7280;
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
st.markd
