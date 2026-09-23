"""
data.py
-------
Loads the MNIST handwritten-digit dataset and prepares it for training.

MNIST contains 70,000 small grayscale images (28x28 pixels) of handwritten
digits 0-9, split into 60,000 training images and 10,000 test images.
"""

import numpy as np
import tensorflow as tf


def load_mnist_data():
    """
    Downloads (or loads from local cache) the MNIST dataset and prepares it
    for a Keras CNN.

    Returns:
        x_train, y_train, x_test, y_test
        - x_train/x_test: images, shape (N, 28, 28, 1), pixel values in [0, 1]
        - y_train/y_test: integer labels 0-9, shape (N,)
    """
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    except Exception as exc:
        raise RuntimeError(
            "Could not download the MNIST dataset. The first time you train, "
            "this program needs an internet connection to fetch it "
            "(about 11 MB, cached afterwards in ~/.keras/datasets).\n"
            f"Original error: {exc}"
        ) from exc

    # Pixel values start as integers 0-255. Neural networks train much better
    # on small numbers, so we scale them down to the range [0, 1].
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Keras' Conv2D layers expect a "channels" dimension even for grayscale
    # images, so we reshape (N, 28, 28) -> (N, 28, 28, 1).
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, y_train, x_test, y_test
