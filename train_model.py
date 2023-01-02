from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model import get_model
from keras.utils import to_categorical
from keras.callbacks import History
from pandas import DataFrame, read_csv
from joblib import dump
import matplotlib.pyplot as plt

try:
    dataset = read_csv("dataset/transformed_dataset.csv")
except FileNotFoundError:
    raise RuntimeError("You have to run transform_dataset.py first")

vectorizer = TfidfVectorizer(min_df=10)

X = dataset["review"]
y = LabelEncoder().fit_transform(dataset["sentiment"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

vectorizer = TfidfVectorizer(min_df=10, binary=True)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
X_val = vectorizer.transform(X_val)


model = get_model(len(vectorizer.vocabulary_))


train_history: History = model.fit(
    X_train.astype("float16"),
    y_train,
    validation_data=(X_val.sorted_indices(), y_val),
)

model.save("model/model")

dump(vectorizer, "model/vectorizer.joblib")

test_history = model.evaluate(X_test.astype("float16"), y_test)

# Visualization

figure, axis = plt.subplots(ncols=2)

axis[0].plot(train_history.history["binary_accuracy"])
axis[0].plot(train_history.history["val_binary_accuracy"])
axis[0].set_title("model accuracy")
axis[0].set_ylabel("accuracy")
axis[0].set_xlabel("epoch")
axis[0].legend(["train", "test"], loc="upper left")

# summarize train_history for loss
axis[1].plot(train_history.history["loss"])
axis[1].plot(train_history.history["val_loss"])
axis[1].set_title("model loss")
axis[1].set_ylabel("loss")
axis[1].set_xlabel("epoch")
axis[1].legend(["train", "test"], loc="upper left")

plt.show()
