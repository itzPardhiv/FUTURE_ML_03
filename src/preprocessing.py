import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+#]", " ", text)
    text = re.sub(r"\s+", " ", text)

    words = text.split()

    filtered_words = [
        word for word in words
        if word not in STOPWORDS and len(word) > 2
    ]

    return " ".join(filtered_words)
