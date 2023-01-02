from keras.models import load_model, Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
from transform_dataset import preprocess_text, STOP_WORDS
from joblib import load
import os

# Needed to remove the TensorFLow CPU performance annoying message :)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    print("Loading models...")

    model: Sequential = load_model("model/model")

    vectorizer: TfidfVectorizer = load("model/vectorizer.joblib")
except IOError:
    raise RuntimeError("You have to run train_model.py first")

text = preprocess_text(
    input("Enter a review: "),
    STOP_WORDS,
)

print("The neural network's prediction: ")
print(
    "It's negative"
    if (model.predict(vectorizer.transform([text]).astype("float16"))) < 0.5
    else "It's positive"
)
