from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


def preprocess_text(text: str, stopwords: list[str]) -> str:
    # Getting rid of html tags
    text = re.sub(re.compile("<.*?>"), "", text)

    # We need only words
    text = re.sub(re.compile("[^A-Za-z0-9]+"), " ", text)

    text = text.lower()

    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()

    result = " ".join(
        [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word not in stopwords
        ]
    )

    return result


try:
    EXCLUDED_WORDS = ["not"]

    STOP_WORDS = [
        word
        for word in stopwords.words("english")
        if word not in EXCLUDED_WORDS
    ]
except:
    print("Run download_nltk_data.py first")
