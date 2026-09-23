"""
preprocess.py
-------------
Turns a real photograph of a handwritten digit into the same kind of image
the neural network was trained on (a centered 28x28 grayscale digit, white
strokes on a black background, like MNIST).

A raw phone photo is very different from an MNIST image: it has a paper
background, shadows, the digit is somewhere off-center, the stroke may be
thick or thin, and the photo could be any size. This file fixes all of that
step by step, using only Pillow and NumPy so the code stays easy to read.

Optionally uses SciPy (if installed) to pick out the single largest shape in
the photo, in case the picture accidentally contains more than one object.
"""

import numpy as np
from PIL import Image, ImageFilter

try:
    from scipy import ndimage

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class NoDigitFoundError(ValueError):
    """Raised when no handwritten digit could be located in the photo."""


def preprocess_for_model(pil_image: Image.Image):
    """
    Runs the full preprocessing pipeline on a PIL image.

    Returns:
        model_input: numpy array, shape (1, 28, 28, 1), values in [0, 1] --
                     ready to feed into the trained model.
        display_image: a 28x28 PIL image (scaled up for viewing) showing
                        exactly what the network "sees" -- this is shown in
                        the GUI next to the original photo.
    """
    # Step 1: grayscale. Color doesn't matter for recognizing a digit shape.
    gray = pil_image.convert("L")

    # Shrink very large photos so the rest of the steps run quickly. This
    # keeps the blur/threshold steps fast without hurting quality.
    gray.thumbnail((800, 800))

    gray_array = np.array(gray, dtype=np.float32)

    # Step 2: correct for shadows / uneven lighting. We estimate the
    # background brightness with a heavy blur, then subtract it out. This
    # flattens dim corners and shadows so the next steps see a roughly even
    # background instead of a gradient.
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=25))
    blurred_array = np.array(blurred, dtype=np.float32)
    corrected = gray_array - blurred_array + 128.0
    corrected = np.clip(corrected, 0, 255)

    # Step 3: figure out polarity. MNIST digits are bright strokes on a dark
    # background. A phone photo is usually the opposite (dark pen on light
    # paper), so we sample the four corners -- assumed to be background --
    # and invert the image if the background turns out to be light.
    corner_size = max(2, min(corrected.shape) // 10)
    corners = np.concatenate(
        [
            corrected[:corner_size, :corner_size].ravel(),
            corrected[:corner_size, -corner_size:].ravel(),
            corrected[-corner_size:, :corner_size].ravel(),
            corrected[-corner_size:, -corner_size:].ravel(),
        ]
    )
    background_is_light = np.mean(corners) > 127
    if background_is_light:
        corrected = 255.0 - corrected

    # Step 4: turn the grayscale image into a clean black/white (binary)
    # image, separating "digit" pixels from "background" pixels. Otsu's
    # method automatically finds the best brightness cutoff for this photo,
    # instead of us having to guess a fixed threshold.
    threshold = _otsu_threshold(corrected)
    binary = (corrected > threshold).astype(np.uint8) * 255

    # Step 5: if the photo accidentally has more than one blob (e.g. a
    # smudge or a second digit), keep only the largest one.
    binary = _keep_largest_blob(binary)

    # Step 6: crop tightly around the digit, then center it on a 28x28
    # canvas the same way the MNIST images were built.
    cropped = _crop_to_bounding_box(binary)
    canvas = _resize_and_center(cropped)

    # Step 7: normalize pixel values to [0, 1] and add the batch/channel
    # dimensions Keras expects: (1, 28, 28, 1).
    model_input = canvas.astype("float32") / 255.0
    model_input = model_input.reshape(1, 28, 28, 1)

    display_image = Image.fromarray(canvas).resize((140, 140), Image.NEAREST)
    return model_input, display_image


def _otsu_threshold(gray_array: np.ndarray) -> float:
    """
    Finds the brightness cutoff that best separates a bright group of
    pixels (the digit) from a dark group (the background), by trying every
    possible cutoff and keeping the one that maximizes the separation
    between the two groups.
    """
    histogram, _ = np.histogram(gray_array, bins=256, range=(0, 255))
    histogram = histogram.astype(np.float64)
    total_pixels = gray_array.size
    overall_sum = np.dot(np.arange(256), histogram)

    best_threshold = 127
    best_separation = 0.0
    weight_background = 0.0
    sum_background = 0.0

    for level in range(256):
        weight_background += histogram[level]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += level * histogram[level]
        mean_background = sum_background / weight_background
        mean_foreground = (overall_sum - sum_background) / weight_foreground

        separation = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )
        if separation > best_separation:
            best_separation = separation
            best_threshold = level

    return best_threshold


def _keep_largest_blob(binary: np.ndarray) -> np.ndarray:
    """Keeps only the largest connected white shape (requires SciPy)."""
    if not _HAS_SCIPY:
        return binary

    labeled, num_shapes = ndimage.label(binary > 0)
    if num_shapes <= 1:
        return binary

    sizes = ndimage.sum(binary > 0, labeled, index=range(1, num_shapes + 1))
    largest_label = np.argmax(sizes) + 1
    return np.where(labeled == largest_label, 255, 0).astype(np.uint8)


def _crop_to_bounding_box(binary: np.ndarray) -> np.ndarray:
    """Crops the image tightly around the white (digit) pixels."""
    rows = np.any(binary > 0, axis=1)
    cols = np.any(binary > 0, axis=0)

    if not rows.any() or not cols.any():
        raise NoDigitFoundError(
            "No digit could be detected in this image. Try a photo with a "
            "single digit, clearly darker (or brighter) than its "
            "background, filling a good part of the frame."
        )

    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    # Small margin so we don't crop right up against the stroke edge.
    margin = 4
    top = max(0, top - margin)
    left = max(0, left - margin)
    bottom = min(binary.shape[0] - 1, bottom + margin)
    right = min(binary.shape[1] - 1, right + margin)

    return binary[top : bottom + 1, left : right + 1]


def _resize_and_center(cropped: np.ndarray) -> np.ndarray:
    """
    Resizes the cropped digit to fit inside a 20x20 box (preserving its
    aspect ratio, like the original MNIST preparation), then pastes it onto
    the middle of a 28x28 black canvas -- the exact size the model expects.
    """
    digit_image = Image.fromarray(cropped)

    height, width = cropped.shape
    if height > width:
        new_height = 20
        new_width = max(1, round(width * (20 / height)))
    else:
        new_width = 20
        new_height = max(1, round(height * (20 / width)))

    digit_image = digit_image.resize((new_width, new_height), Image.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    paste_x = (28 - new_width) // 2
    paste_y = (28 - new_height) // 2
    canvas.paste(digit_image, (paste_x, paste_y))

    return np.array(canvas, dtype=np.uint8)
