"""
model.py
--------
Defines the small neural network (a compact CNN) used to recognize digits.

The network is intentionally small so it trains in a couple of minutes on a
normal laptop CPU -- no GPU required.
"""

from tensorflow import keras
from tensorflow.keras import layers

# Where the trained model is saved/loaded from.
MODEL_PATH = "saved_model/digit_model.keras"

# How many training passes over the full dataset to run.
EPOCHS = 8

# How many training examples to look at before updating the network's
# weights once. Smaller = slower but sometimes more accurate; 128 is a good
# balance for a laptop CPU.
BATCH_SIZE = 128


def build_model():
    """
    Builds a small Convolutional Neural Network (CNN).

    Why a CNN? Convolutional layers slide small filters across the image
    looking for simple patterns (edges, curves, corners). Because the same
    filter is reused across the whole image, the network can still recognize
    a digit even if it's drawn slightly off-center, larger, or smaller than
    the training examples -- exactly the kind of variation we expect from a
    real phone photo.

    Layers, in plain terms:
      Conv2D(8)   -> learns 8 simple stroke/edge patterns
      MaxPool     -> shrinks the image, keeping only the strongest signals
      Conv2D(16)  -> combines simple patterns into more complex shapes
      MaxPool     -> shrinks again
      Flatten     -> turns the 2D feature maps into a single list of numbers
      Dense(64)   -> combines all the evidence together
      Dropout     -> randomly ignores some neurons during training, which
                     helps prevent the network from "memorizing" the
                     training images instead of learning general patterns
      Dense(10)   -> one output per digit (0-9), turned into probabilities
                     by the softmax activation
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(8, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(16, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_saved_model(path: str = MODEL_PATH):
    """
    Loads a previously trained model from disk.

    Raises FileNotFoundError with a friendly message if no saved model
    exists yet.
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found at '{path}'.\n"
            "Click 'Train Model' first -- it only needs to be done once, "
            "and the result is saved for next time."
        )
    return keras.models.load_model(path)
