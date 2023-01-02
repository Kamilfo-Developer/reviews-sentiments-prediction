from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.metrics import Accuracy, BinaryAccuracy
from keras import Sequential


def get_model(input_size: int) -> Sequential:

    model = Sequential()

    model.add(InputLayer(input_size))

    model.add(Dense(1500, activation="relu"))

    model.add(Dense(1000, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))

    loss = BinaryCrossentropy()

    optimizer = Adam(3e-4)

    model.compile(optimizer, loss, metrics=[BinaryAccuracy()])

    return model
