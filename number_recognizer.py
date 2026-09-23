"""
number_recognizer.py
---------------------
ONE FILE. Just run it.

Teaches a small neural network to recognize handwritten digits (0-9) using
the MNIST dataset, shows it learning live in a window with graphs, saves
the trained model so you never have to retrain, and lets you test it on a
photo of your own handwriting.

HOW TO RUN
  1. pip install -r requirements.txt   (see requirements.txt / README.md)
  2. python number_recognizer.py
  3. Click "Train Model" (first time only). Training runs until you click
     "Stop Training" -- there's no fixed length, so let it run as long as
     you'd like before stopping it.
  4. Click "Test My Handwriting" and choose a photo of a digit you wrote.

Everything is in this one file so it's easy to copy/paste into PyCharm.
It's organized top to bottom in the order things happen:
  1. Setup / constants
  2. Loading the MNIST dataset
  3. Building the neural network
  4. Turning a phone photo into an MNIST-like image
  5. The Tkinter window (GUI) that ties it all together
"""

import os
import sys

# ----------------------------------------------------------------------
# 1. SETUP
# ----------------------------------------------------------------------
# Check for required libraries up front and give a clear message instead
# of a confusing crash if one is missing.
_MISSING = []
for _module_name in ("tensorflow", "numpy", "PIL", "matplotlib"):
    try:
        __import__(_module_name)
    except ImportError:
        _MISSING.append(_module_name)
if _MISSING:
    print(
        "Missing required libraries: "
        + ", ".join(_MISSING)
        + "\nRun this first:  pip install -r requirements.txt"
    )
    sys.exit(1)

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageFilter, ImageTk

try:
    from scipy import ndimage

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# Where the trained model is saved/loaded from.
MODEL_PATH = "saved_model/digit_model.keras"

# Training runs indefinitely -- click "Stop Training" in the app whenever
# you want it to stop, rather than after a fixed number of passes. This is
# just a hard safety ceiling in case it's left running unattended.
MAX_EPOCHS = 100_000

# Each image is processed in small batches (not all 60,000 at once -- that
# would need far more memory than a laptop has). This many images per
# batch:
BATCH_SIZE = 128

# ...and this many batches make up one "epoch" shown on the dashboard. The
# full training set is 60,000 images / 128 per batch = ~469 batches; using
# fewer than that per epoch means each epoch covers less data, so it
# finishes faster and the dashboard updates more often -- the network
# still sees the rest of the data on the epochs that follow.
STEPS_PER_EPOCH = 150

# How many test images to show live in the training dashboard.
NUM_SAMPLE_PREDICTIONS = 8

# Validated, colorblind-safe chart/UI colors, in a light and a dark
# variant (toggled from the "Dark Mode" / "Light Mode" button in the app).
# Used consistently across the window and both graphs so "Train" and
# "Test" always mean the same color everywhere on screen.
LIGHT_PALETTE = {
    "surface": "#fcfcfb",        # chart / panel background
    "page": "#f9f9f7",           # window background
    "ink_primary": "#0b0b0b",    # main text
    "ink_secondary": "#52514e",  # secondary text
    "ink_muted": "#898781",      # axis labels, muted text
    "gridline": "#e1e0d9",       # chart gridlines
    "axis": "#c3c2b7",           # chart axis lines / borders
    "train": "#2a78d6",          # blue -- always means "training data"
    "test": "#eb6834",           # orange -- always means "test data"
    "accent": "#2a78d6",         # primary button / highlight color
    "good": "#0ca30c",           # correct prediction
    "critical": "#d03b3b",       # incorrect prediction
}
DARK_PALETTE = {
    "surface": "#1a1a19",
    "page": "#0d0d0d",
    "ink_primary": "#ffffff",
    "ink_secondary": "#c3c2b7",
    "ink_muted": "#898781",
    "gridline": "#2c2c2a",
    "axis": "#383835",
    "train": "#3987e5",
    "test": "#d95926",
    "accent": "#3987e5",
    "good": "#0ca30c",
    "critical": "#e66767",
}

# These start out pointing at the light palette; apply_palette() below
# reassigns them (and every function that reads them looks the value up
# fresh each time it's called, so a reassignment here is picked up
# everywhere immediately -- no need to pass colors around as arguments).
COLOR_SURFACE = LIGHT_PALETTE["surface"]
COLOR_PAGE = LIGHT_PALETTE["page"]
COLOR_INK_PRIMARY = LIGHT_PALETTE["ink_primary"]
COLOR_INK_SECONDARY = LIGHT_PALETTE["ink_secondary"]
COLOR_INK_MUTED = LIGHT_PALETTE["ink_muted"]
COLOR_GRIDLINE = LIGHT_PALETTE["gridline"]
COLOR_AXIS = LIGHT_PALETTE["axis"]
COLOR_TRAIN = LIGHT_PALETTE["train"]
COLOR_TEST = LIGHT_PALETTE["test"]
COLOR_ACCENT = LIGHT_PALETTE["accent"]
COLOR_GOOD = LIGHT_PALETTE["good"]
COLOR_CRITICAL = LIGHT_PALETTE["critical"]


def apply_palette(mode):
    """Switches every COLOR_* constant above to the light or dark palette."""
    global COLOR_SURFACE, COLOR_PAGE, COLOR_INK_PRIMARY, COLOR_INK_SECONDARY
    global COLOR_INK_MUTED, COLOR_GRIDLINE, COLOR_AXIS, COLOR_TRAIN
    global COLOR_TEST, COLOR_ACCENT, COLOR_GOOD, COLOR_CRITICAL

    palette = DARK_PALETTE if mode == "dark" else LIGHT_PALETTE
    COLOR_SURFACE = palette["surface"]
    COLOR_PAGE = palette["page"]
    COLOR_INK_PRIMARY = palette["ink_primary"]
    COLOR_INK_SECONDARY = palette["ink_secondary"]
    COLOR_INK_MUTED = palette["ink_muted"]
    COLOR_GRIDLINE = palette["gridline"]
    COLOR_AXIS = palette["axis"]
    COLOR_TRAIN = palette["train"]
    COLOR_TEST = palette["test"]
    COLOR_ACCENT = palette["accent"]
    COLOR_GOOD = palette["good"]
    COLOR_CRITICAL = palette["critical"]
    apply_chart_style()


def apply_chart_style():
    """
    Applies one consistent look to every *new* matplotlib chart in the
    app, so graphs match the window's colors instead of matplotlib's
    blue-gray defaults with a mismatched white background. This only
    affects charts created from here on -- an already-built chart needs
    _style_axes() below to actually change color (matplotlib doesn't
    retroactively restyle existing ones from rcParams alone).
    """
    plt_rc = {
        "figure.facecolor": COLOR_SURFACE,
        "axes.facecolor": COLOR_SURFACE,
        "axes.edgecolor": COLOR_AXIS,
        "axes.labelcolor": COLOR_INK_SECONDARY,
        "axes.titlecolor": COLOR_INK_PRIMARY,
        "axes.grid": True,
        "grid.color": COLOR_GRIDLINE,
        "grid.linewidth": 0.8,
        "xtick.color": COLOR_INK_MUTED,
        "ytick.color": COLOR_INK_MUTED,
        "text.color": COLOR_INK_PRIMARY,
        "legend.labelcolor": COLOR_INK_PRIMARY,
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
    matplotlib.rcParams.update(plt_rc)


def _style_axes(ax):
    """
    Explicitly recolors one already-existing chart axes to match the
    current palette. Called after every ax.clear() + replot, including
    when the theme is toggled, since matplotlib only applies rcParams to
    axes at the moment they're first created.
    """
    ax.set_facecolor(COLOR_SURFACE)
    ax.tick_params(colors=COLOR_INK_MUTED)
    for side, spine in ax.spines.items():
        spine.set_visible(side not in ("top", "right"))
        spine.set_color(COLOR_AXIS)
    ax.title.set_color(COLOR_INK_PRIMARY)
    ax.xaxis.label.set_color(COLOR_INK_SECONDARY)
    ax.yaxis.label.set_color(COLOR_INK_SECONDARY)
    ax.grid(True, color=COLOR_GRIDLINE, linewidth=0.8)


# ----------------------------------------------------------------------
# 2. LOADING THE MNIST DATASET
# ----------------------------------------------------------------------
def load_mnist_data():
    """
    Downloads (or loads from local cache) the MNIST dataset: 70,000 small
    grayscale images (28x28 pixels) of handwritten digits 0-9.

    Returns x_train, y_train, x_test, y_test where:
      - x_train/x_test are images, shape (N, 28, 28, 1), pixels in [0, 1]
      - y_train/y_test are the correct digit labels (0-9)
    """
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    except Exception as exc:
        raise RuntimeError(
            "Could not download the MNIST dataset. This needs an internet "
            "connection the first time you train (about 11 MB, then it's "
            f"cached). Original error: {exc}"
        ) from exc

    # Scale pixel values from 0-255 down to 0-1 -- neural networks train
    # much better on small numbers.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Conv2D layers expect a "channels" dimension even for grayscale
    # images: (N, 28, 28) -> (N, 28, 28, 1).
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, y_train, x_test, y_test


# ----------------------------------------------------------------------
# 3. BUILDING THE NEURAL NETWORK
# ----------------------------------------------------------------------
def build_model():
    """
    Builds a small Convolutional Neural Network (CNN) -- bigger than the
    bare minimum needed for MNIST, so it has enough capacity to squeeze
    out the last fraction of a percent of accuracy over a longer training
    run, and to generalize better to real handwriting photos.

    In plain terms, top to bottom:
      RandomRotation / RandomTranslation
                  -> "data augmentation": during training only, each image
                     is randomly nudged/rotated a little. This teaches the
                     network that a digit is still the same digit even if
                     it's slightly rotated or off-center -- exactly the
                     kind of variation a real phone photo has.
      Conv2D(32) x2 + BatchNorm
                  -> learns 32 simple stroke/edge patterns, refines them,
                     and BatchNorm keeps the numbers flowing through the
                     network well-scaled so training is faster and more
                     stable
      MaxPool     -> shrinks the image, keeping the strongest signals
      Dropout     -> randomly ignores some neurons while training, which
                     helps the network generalize instead of memorizing
      Conv2D(64) + BatchNorm
                  -> combines simple patterns into more complex shapes
      MaxPool + Dropout again
      Flatten     -> turns the 2D feature maps into a single list of numbers
      Dense(128) + BatchNorm + Dropout
                  -> combines all the evidence together
      Dense(10)   -> one output per digit 0-9, turned into probabilities
                     that add up to 100% (softmax)
    """
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_saved_model(path: str = MODEL_PATH):
    """Loads a previously trained model, or raises a friendly error."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found at '{path}'.\n"
            "Click 'Train Model' first -- it only needs to be done once."
        )
    return tf.keras.models.load_model(path)


# ----------------------------------------------------------------------
# 4. TURNING A PHONE PHOTO INTO AN MNIST-LIKE IMAGE
# ----------------------------------------------------------------------
class NoDigitFoundError(ValueError):
    """Raised when no handwritten digit could be located in the photo."""


def preprocess_for_model(pil_image: Image.Image):
    """
    Runs a real phone photo through the same kind of pipeline that made the
    original MNIST images: grayscale, shadow correction, thresholding,
    cropping, and centering on a 28x28 canvas.

    Returns:
        model_input:  numpy array, shape (1, 28, 28, 1), values in [0, 1]
        display_image: a 28x28 image (scaled up for viewing) showing
                        exactly what the network "sees"
    """
    # Step 1: grayscale -- color doesn't matter for digit shape.
    gray = pil_image.convert("L")
    gray.thumbnail((800, 800))  # keep large photos fast to process
    gray_array = np.array(gray, dtype=np.float32)

    # Step 2: correct for shadows / uneven lighting by estimating the
    # background with a heavy blur and subtracting it out.
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=25))
    blurred_array = np.array(blurred, dtype=np.float32)
    corrected = np.clip(gray_array - blurred_array + 128.0, 0, 255)

    # Step 3: figure out polarity. MNIST digits are bright strokes on a
    # dark background; a phone photo is usually the opposite (dark pen on
    # light paper). We sample the four corners (assumed background) and
    # invert if they're light.
    corner_size = max(2, min(corrected.shape) // 10)
    corners = np.concatenate(
        [
            corrected[:corner_size, :corner_size].ravel(),
            corrected[:corner_size, -corner_size:].ravel(),
            corrected[-corner_size:, :corner_size].ravel(),
            corrected[-corner_size:, -corner_size:].ravel(),
        ]
    )
    if np.mean(corners) > 127:
        corrected = 255.0 - corrected

    # Step 4: Otsu's method automatically finds the best brightness cutoff
    # to separate the digit from the background.
    threshold = _otsu_threshold(corrected)
    binary = (corrected > threshold).astype(np.uint8) * 255

    # Step 5: if the photo has more than one blob, keep only the largest.
    binary = _keep_largest_blob(binary)

    # Step 6: crop tightly around the digit, then center it on a 28x28
    # canvas the same way MNIST images were built.
    cropped = _crop_to_bounding_box(binary)
    canvas = _resize_and_center(cropped)

    # Step 7: normalize to [0, 1] and add the batch/channel dimensions
    # Keras expects: (1, 28, 28, 1).
    model_input = canvas.astype("float32") / 255.0
    model_input = model_input.reshape(1, 28, 28, 1)

    display_image = Image.fromarray(canvas).resize((140, 140), Image.Resampling.NEAREST)
    return model_input, display_image


def _otsu_threshold(gray_array: np.ndarray) -> float:
    """Finds the brightness cutoff that best separates digit from background."""
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
            weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
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
    """Crops tightly around the white (digit) pixels."""
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
    margin = 4
    top = max(0, top - margin)
    left = max(0, left - margin)
    bottom = min(binary.shape[0] - 1, bottom + margin)
    right = min(binary.shape[1] - 1, right + margin)
    return binary[top : bottom + 1, left : right + 1]


def _resize_and_center(cropped: np.ndarray) -> np.ndarray:
    """Fits the digit into a 20x20 box (preserving proportions), centered on 28x28."""
    digit_image = Image.fromarray(cropped)
    height, width = cropped.shape
    if height > width:
        new_height = 20
        new_width = max(1, round(width * (20 / height)))
    else:
        new_width = 20
        new_height = max(1, round(height * (20 / width)))
    digit_image = digit_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    paste_x = (28 - new_width) // 2
    paste_y = (28 - new_height) // 2
    canvas.paste(digit_image, (paste_x, paste_y))
    return np.array(canvas, dtype=np.uint8)


# ----------------------------------------------------------------------
# 5. THE WINDOW (GUI)
# ----------------------------------------------------------------------
class DashboardCallback(tf.keras.callbacks.Callback):
    """Updates the live training dashboard after every epoch."""

    def __init__(self, app, sample_images, sample_labels):
        super().__init__()
        self.app = app
        self.sample_images = sample_images
        self.sample_labels = sample_labels

    def on_epoch_begin(self, epoch, logs=None):
        self.app.set_status(f"Training epoch {epoch + 1}... (click 'Stop Training' any time)")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        predictions = self.model.predict(self.sample_images, verbose=0)
        self.app.update_dashboard(
            epoch=epoch,
            logs=logs,
            sample_images=self.sample_images,
            sample_labels=self.sample_labels,
            sample_predictions=predictions,
        )
        # Checked once per epoch (not continuously) -- update_dashboard()
        # just processed pending GUI events, so a "Stop Training" click is
        # already reflected in self.app.stop_requested by this point.
        if self.app.stop_requested:
            self.model.stop_training = True


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Teaching a Neural Network to Read Handwriting")
        self.root.geometry("1200x780")
        self.root.minsize(1000, 680)
        self.root.configure(bg=COLOR_PAGE)

        self.model = None
        self.x_train = self.y_train = self.x_test = self.y_test = None
        self.train_losses, self.test_losses = [], []
        self.train_accs, self.test_accs = [], []
        self.displayed_photo = None
        self.displayed_processed_photo = None
        self.stop_requested = False
        self.theme_mode = "light"

        # Cached so charts can be redrawn with the same content (just in
        # new colors) when the theme is toggled, instead of going blank.
        self.last_sample_images = None
        self.last_sample_labels = None
        self.last_sample_predictions = None
        self.last_probabilities = None
        self.last_predicted_digit = None

        self._configure_style()
        self._build_layout()

    # ---------------- Style ----------------
    def _configure_style(self):
        """Configures one consistent color/spacing theme for every widget."""
        style = ttk.Style(self.root)
        # "clam" is the only built-in ttk theme that reliably honors custom
        # colors on every platform (the native themes mostly ignore them).
        style.theme_use("clam")

        style.configure("TFrame", background=COLOR_PAGE)
        style.configure("Surface.TFrame", background=COLOR_SURFACE)
        style.configure("TLabel", background=COLOR_PAGE, foreground=COLOR_INK_PRIMARY, font=("TkDefaultFont", 11))
        style.configure(
            "Header.TLabel",
            background=COLOR_PAGE,
            foreground=COLOR_INK_PRIMARY,
            font=("TkDefaultFont", 18, "bold"),
        )
        style.configure(
            "Subheader.TLabel",
            background=COLOR_PAGE,
            foreground=COLOR_INK_SECONDARY,
            font=("TkDefaultFont", 11),
        )
        style.configure(
            "Status.TLabel",
            background=COLOR_SURFACE,
            foreground=COLOR_INK_SECONDARY,
            font=("TkDefaultFont", 10),
            padding=10,
        )
        style.configure(
            "Surface.TLabel",
            background=COLOR_SURFACE,
            foreground=COLOR_INK_PRIMARY,
        )
        style.configure(
            "StatTileLabel.TLabel",
            background=COLOR_SURFACE,
            foreground=COLOR_INK_MUTED,
            font=("TkDefaultFont", 9, "bold"),
        )
        style.configure(
            "StatTileValue.TLabel",
            background=COLOR_SURFACE,
            foreground=COLOR_INK_PRIMARY,
            font=("TkDefaultFont", 15, "bold"),
        )
        style.configure(
            "StatTile.TFrame",
            background=COLOR_SURFACE,
            relief="solid",
            borderwidth=1,
        )
        style.map("StatTile.TFrame", bordercolor=[("!disabled", COLOR_GRIDLINE)])

        # Primary (accent-colored) buttons for the main actions.
        style.configure(
            "Accent.TButton",
            background=COLOR_ACCENT,
            foreground="#ffffff",
            font=("TkDefaultFont", 10, "bold"),
            padding=(14, 8),
            borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#1c5cab"), ("disabled", COLOR_GRIDLINE)],
            foreground=[("disabled", COLOR_INK_MUTED)],
        )

        # Secondary (outline-style) buttons for less prominent actions.
        style.configure(
            "Secondary.TButton",
            background=COLOR_SURFACE,
            foreground=COLOR_INK_PRIMARY,
            font=("TkDefaultFont", 10),
            padding=(14, 8),
            borderwidth=1,
        )
        style.map(
            "Secondary.TButton",
            background=[("active", COLOR_GRIDLINE), ("disabled", COLOR_SURFACE)],
            foreground=[("disabled", COLOR_INK_MUTED)],
        )

        # "Stop Training" -- status-red, only enabled while training runs.
        style.configure(
            "Danger.TButton",
            background=COLOR_CRITICAL,
            foreground="#ffffff",
            font=("TkDefaultFont", 10, "bold"),
            padding=(14, 8),
            borderwidth=0,
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#a82f2f"), ("disabled", COLOR_GRIDLINE)],
            foreground=[("disabled", COLOR_INK_MUTED)],
        )

        style.configure("TLabelframe", background=COLOR_SURFACE, borderwidth=1, relief="solid")
        style.configure(
            "TLabelframe.Label",
            background=COLOR_SURFACE,
            foreground=COLOR_INK_SECONDARY,
            font=("TkDefaultFont", 10, "bold"),
        )

    def _make_stat_tile(self, parent, label_text, initial_value):
        """
        Builds one small bordered "tile" showing a label (e.g. "Epoch") on
        top and a bold value below -- used for the live training stats, so
        they read as distinct data points instead of a run-on line of text.
        """
        tile = ttk.Frame(parent, style="StatTile.TFrame", padding=(14, 8))
        ttk.Label(tile, text=label_text, style="StatTileLabel.TLabel").pack(anchor="w")
        value_var = tk.StringVar(value=initial_value)
        ttk.Label(tile, textvariable=value_var, style="StatTileValue.TLabel").pack(anchor="w")
        return tile, value_var

    # ---------------- Theme ----------------
    def _theme_button_label(self):
        # The label names the mode you'd SWITCH TO, not the current one.
        return "Light Mode" if self.theme_mode == "dark" else "Dark Mode"

    def on_toggle_theme_clicked(self):
        self.set_theme("dark" if self.theme_mode == "light" else "light")

    def set_theme(self, mode):
        if mode == self.theme_mode:
            return
        self.theme_mode = mode
        apply_palette(mode)  # updates the COLOR_* constants + matplotlib rcParams

        # ttk widgets (buttons, frames, labels, tiles) redraw themselves
        # automatically once their named style is reconfigured.
        self._configure_style()
        self.root.configure(bg=COLOR_PAGE)
        self.theme_button.configure(text=self._theme_button_label())

        # tk.Text isn't a ttk widget, so its colors need to be set directly.
        self.confidence_text.configure(background=COLOR_SURFACE, foreground=COLOR_INK_PRIMARY)

        # matplotlib charts already on screen need to be explicitly
        # restyled and redrawn -- rcParams only affects charts created
        # from this point on, not these existing ones.
        self.dashboard_figure.set_facecolor(COLOR_SURFACE)
        self.probability_figure.set_facecolor(COLOR_SURFACE)
        if self.train_losses:
            self._redraw_dashboard_charts()
        else:
            self._reset_dashboard_plots()
        self.dashboard_canvas.draw()

        self._update_confidence_text()
        self._update_probability_chart()

    # ---------------- Layout ----------------
    def _build_layout(self):
        header = ttk.Frame(self.root, padding=(16, 16, 16, 8))
        header.pack(side=tk.TOP, fill=tk.X)

        title_row = ttk.Frame(header)
        title_row.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(title_row, text="Teaching a Neural Network to Read Handwriting", style="Header.TLabel").pack(
            side=tk.LEFT, anchor="w"
        )
        self.theme_button = ttk.Button(
            title_row, text=self._theme_button_label(), command=self.on_toggle_theme_clicked, style="Secondary.TButton"
        )
        self.theme_button.pack(side=tk.RIGHT)

        ttk.Label(
            header,
            text="Train a small neural network on MNIST, watch it learn live, then test it on your own handwriting.",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        button_bar = ttk.Frame(self.root, padding=(16, 4, 16, 12))
        button_bar.pack(side=tk.TOP, fill=tk.X)

        self.main_buttons = []

        train_button = ttk.Button(
            button_bar, text="Train Model", command=self.on_train_clicked, style="Accent.TButton"
        )
        train_button.pack(side=tk.LEFT, padx=(0, 8))
        self.main_buttons.append(train_button)

        self.stop_button = ttk.Button(
            button_bar,
            text="Stop Training",
            command=self.on_stop_clicked,
            state=tk.DISABLED,
            style="Danger.TButton",
        )
        self.stop_button.pack(side=tk.LEFT, padx=8)

        load_button = ttk.Button(
            button_bar, text="Load Saved Model", command=self.on_load_clicked, style="Secondary.TButton"
        )
        load_button.pack(side=tk.LEFT, padx=8)
        self.main_buttons.append(load_button)

        retrain_button = ttk.Button(
            button_bar, text="Retrain Model", command=self.on_retrain_clicked, style="Secondary.TButton"
        )
        retrain_button.pack(side=tk.LEFT, padx=8)
        self.main_buttons.append(retrain_button)

        self.test_button = ttk.Button(
            button_bar,
            text="Test My Handwriting",
            command=self.show_test_screen,
            state=tk.DISABLED,
            style="Accent.TButton",
        )
        self.test_button.pack(side=tk.LEFT, padx=8)

        back_button = ttk.Button(
            button_bar,
            text="Back to Training Dashboard",
            command=self.show_dashboard_screen,
            style="Secondary.TButton",
        )
        back_button.pack(side=tk.LEFT, padx=8)
        self.main_buttons.append(back_button)

        self.status_var = tk.StringVar(value="Click 'Train Model' or 'Load Saved Model' to begin.")
        status_bar = ttk.Frame(self.root, style="Surface.TFrame")
        status_bar.pack(side=tk.TOP, fill=tk.X, padx=16)
        ttk.Label(status_bar, textvariable=self.status_var, style="Status.TLabel").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        self.screen_container = ttk.Frame(self.root, padding=16)
        self.screen_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.dashboard_frame = ttk.Frame(self.screen_container)
        self.test_frame = ttk.Frame(self.screen_container)
        for frame in (self.dashboard_frame, self.test_frame):
            frame.grid(row=0, column=0, sticky="nsew")
        self.screen_container.rowconfigure(0, weight=1)
        self.screen_container.columnconfigure(0, weight=1)

        self._build_dashboard_screen()
        self._build_test_screen()
        self.show_dashboard_screen()

    def _build_dashboard_screen(self):
        stats = ttk.Frame(self.dashboard_frame)
        stats.pack(side=tk.TOP, fill=tk.X, pady=(0, 12))

        epoch_tile, self.epoch_var = self._make_stat_tile(stats, "EPOCH", "-")
        train_loss_tile, self.train_loss_var = self._make_stat_tile(stats, "TRAIN LOSS", "-")
        test_loss_tile, self.test_loss_var = self._make_stat_tile(stats, "TEST LOSS", "-")
        train_acc_tile, self.train_acc_var = self._make_stat_tile(stats, "TRAIN ACCURACY", "-")
        test_acc_tile, self.test_acc_var = self._make_stat_tile(stats, "TEST ACCURACY", "-")

        for tile in (epoch_tile, train_loss_tile, test_loss_tile, train_acc_tile, test_acc_tile):
            tile.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)

        chart_panel = ttk.Frame(self.dashboard_frame, style="Surface.TFrame", padding=8)
        chart_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.dashboard_figure = Figure(figsize=(10, 5.5), dpi=100, constrained_layout=True)
        grid = self.dashboard_figure.add_gridspec(2, 8)
        self.loss_ax = self.dashboard_figure.add_subplot(grid[0, 0:4])
        self.acc_ax = self.dashboard_figure.add_subplot(grid[1, 0:4])
        self.sample_axes = [
            self.dashboard_figure.add_subplot(grid[r, 4 + c]) for r in range(2) for c in range(4)
        ]
        self._reset_dashboard_plots()

        self.dashboard_canvas = FigureCanvasTkAgg(self.dashboard_figure, master=chart_panel)
        self.dashboard_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _build_test_screen(self):
        top = ttk.Frame(self.test_frame)
        top.pack(side=tk.TOP, fill=tk.X, pady=(0, 12))
        ttk.Button(
            top, text="Choose Image...", command=self.on_choose_image_clicked, style="Accent.TButton"
        ).pack(side=tk.LEFT)
        self.prediction_var = tk.StringVar(value="Prediction: -    Confidence: -")
        ttk.Label(top, textvariable=self.prediction_var, style="Header.TLabel").pack(side=tk.LEFT, padx=20)

        images_row = ttk.Frame(self.test_frame)
        images_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 12))

        original_box = ttk.LabelFrame(images_row, text="Original Photo", padding=8)
        original_box.pack(side=tk.LEFT, padx=(0, 12))
        self.original_image_label = ttk.Label(original_box, style="Surface.TLabel")
        self.original_image_label.pack(padx=6, pady=6)

        processed_box = ttk.LabelFrame(images_row, text="What the Network Sees (28x28)", padding=8)
        processed_box.pack(side=tk.LEFT, padx=12)
        self.processed_image_label = ttk.Label(processed_box, style="Surface.TLabel")
        self.processed_image_label.pack(padx=6, pady=6)

        readout_box = ttk.LabelFrame(images_row, text="Confidence for each digit", padding=8)
        readout_box.pack(side=tk.LEFT, padx=12, fill=tk.BOTH, expand=True)
        self.confidence_text = tk.Text(
            readout_box,
            width=16,
            height=12,
            font=("Courier", 11),
            background=COLOR_SURFACE,
            foreground=COLOR_INK_PRIMARY,
            relief="flat",
            highlightthickness=0,
        )
        self.confidence_text.pack(padx=6, pady=6)
        self.confidence_text.configure(state=tk.DISABLED)

        chart_panel = ttk.Frame(self.test_frame, style="Surface.TFrame", padding=8)
        chart_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.probability_figure = Figure(figsize=(9, 3), dpi=100, constrained_layout=True)
        self.probability_ax = self.probability_figure.add_subplot(111)
        self.probability_ax.set_title("Prediction confidence by digit")
        self.probability_ax.set_xticks(range(10))
        self.probability_ax.set_ylim(0, 100)
        self.probability_canvas = FigureCanvasTkAgg(self.probability_figure, master=chart_panel)
        self.probability_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # ---------------- Screen switching ----------------
    def show_dashboard_screen(self):
        self.dashboard_frame.tkraise()

    def show_test_screen(self):
        self.test_frame.tkraise()

    # ---------------- Helpers ----------------
    def set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def set_buttons_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in self.main_buttons:
            button.configure(state=state)

    def _reset_dashboard_plots(self):
        self.loss_ax.clear()
        self.loss_ax.set_title("Loss (lower is better)")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.plot([], [], label="Train", color=COLOR_TRAIN)
        self.loss_ax.plot([], [], label="Test", color=COLOR_TEST)
        self.loss_ax.legend(loc="upper right")
        _style_axes(self.loss_ax)

        self.acc_ax.clear()
        self.acc_ax.set_title("Accuracy (higher is better)")
        self.acc_ax.set_xlabel("Epoch")
        self.acc_ax.set_ylim(0, 1)
        self.acc_ax.plot([], [], label="Train", color=COLOR_TRAIN)
        self.acc_ax.plot([], [], label="Test", color=COLOR_TEST)
        self.acc_ax.legend(loc="lower right")
        _style_axes(self.acc_ax)

        for ax in self.sample_axes:
            ax.clear()
            ax.axis("off")
            ax.set_facecolor(COLOR_SURFACE)

    def _redraw_dashboard_charts(self):
        """
        Redraws the loss/accuracy graphs and sample-prediction thumbnails
        from whatever data is currently known (self.train_losses etc. and
        the last epoch's sample predictions) -- used both for a normal
        per-epoch update and to reapply colors after a theme change.
        """
        epochs_so_far = range(1, len(self.train_losses) + 1)

        self.loss_ax.clear()
        self.loss_ax.set_title("Loss (lower is better)")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.plot(
            epochs_so_far, self.train_losses, label="Train", color=COLOR_TRAIN, linewidth=2, marker="o", markersize=4
        )
        self.loss_ax.plot(
            epochs_so_far, self.test_losses, label="Test", color=COLOR_TEST, linewidth=2, marker="o", markersize=4
        )
        self.loss_ax.legend(loc="upper right")
        _style_axes(self.loss_ax)

        self.acc_ax.clear()
        self.acc_ax.set_title("Accuracy (higher is better)")
        self.acc_ax.set_xlabel("Epoch")
        self.acc_ax.set_ylim(0, 1)
        self.acc_ax.plot(
            epochs_so_far, self.train_accs, label="Train", color=COLOR_TRAIN, linewidth=2, marker="o", markersize=4
        )
        self.acc_ax.plot(
            epochs_so_far, self.test_accs, label="Test", color=COLOR_TEST, linewidth=2, marker="o", markersize=4
        )
        self.acc_ax.legend(loc="lower right")
        _style_axes(self.acc_ax)

        if self.last_sample_images is not None:
            predicted_labels = np.argmax(self.last_sample_predictions, axis=1)
            for i, ax in enumerate(self.sample_axes):
                ax.clear()
                ax.axis("off")
                ax.set_facecolor(COLOR_SURFACE)
                ax.imshow(self.last_sample_images[i].squeeze(), cmap="gray")
                true_label = int(self.last_sample_labels[i])
                predicted_label = int(predicted_labels[i])
                color = COLOR_GOOD if predicted_label == true_label else COLOR_CRITICAL
                ax.set_title(f"true {true_label} / guess {predicted_label}", color=color, fontsize=8, fontweight="bold")
        else:
            for ax in self.sample_axes:
                ax.clear()
                ax.axis("off")
                ax.set_facecolor(COLOR_SURFACE)

    # ---------------- Button handlers ----------------
    def on_train_clicked(self):
        if self.model is not None and not messagebox.askyesno(
            "Train Model", "A trained model is already loaded. Training a new one will overwrite it. Continue?"
        ):
            return
        self._train_model()

    def on_retrain_clicked(self):
        if messagebox.askyesno(
            "Retrain Model", "This will train a brand new model and overwrite the saved one. Continue?"
        ):
            self._train_model()

    def on_load_clicked(self):
        try:
            self.set_status("Loading saved model...")
            self.model = load_saved_model()
            self.set_status("Saved model loaded and ready.")
            self.test_button.configure(state=tk.NORMAL)
        except FileNotFoundError as exc:
            messagebox.showerror("No Saved Model", str(exc))
            self.set_status("No saved model found. Click 'Train Model' first.")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error Loading Model", str(exc))
            self.set_status("Failed to load saved model.")

    def on_choose_image_clicked(self):
        if self.model is None:
            messagebox.showwarning(
                "No Model Loaded", "Train or load a model first, then come back and choose an image."
            )
            return
        file_path = filedialog.askopenfilename(
            title="Choose a photo of a handwritten digit",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"), ("All files", "*.*")],
        )
        if not file_path:
            return
        self._run_prediction_on_image(file_path)

    # ---------------- Training ----------------
    def on_stop_clicked(self):
        self.stop_requested = True
        self.set_status("Stopping after this epoch...")
        self.stop_button.configure(state=tk.DISABLED)

    def _train_model(self):
        self.set_buttons_enabled(False)
        self.test_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.stop_requested = False
        try:
            if self.x_train is None:
                self.set_status("Downloading/loading MNIST dataset...")
                self.x_train, self.y_train, self.x_test, self.y_test = load_mnist_data()

            self.set_status("Building the neural network...")
            self.model = build_model()

            self.train_losses, self.test_losses = [], []
            self.train_accs, self.test_accs = [], []
            self._reset_dashboard_plots()
            self.dashboard_canvas.draw()

            rng = np.random.default_rng(seed=42)
            sample_indices = rng.choice(len(self.x_test), size=NUM_SAMPLE_PREDICTIONS, replace=False)
            sample_images = self.x_test[sample_indices]
            sample_labels = self.y_test[sample_indices]

            callback = DashboardCallback(self, sample_images, sample_labels)

            # If progress stalls, shrink the learning rate so training can
            # keep inching accuracy up instead of plateauing.
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
            )

            # epochs is a very high ceiling, not a target -- training keeps
            # going, checking after every epoch whether "Stop Training" was
            # clicked (see DashboardCallback.on_epoch_end), until either you
            # stop it or that ceiling is reached.
            self.model.fit(
                self.x_train,
                self.y_train,
                validation_data=(self.x_test, self.y_test),
                epochs=MAX_EPOCHS,
                steps_per_epoch=STEPS_PER_EPOCH,
                batch_size=BATCH_SIZE,
                callbacks=[callback, reduce_lr],
                verbose=0,
            )

            final_loss, final_acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            self.model.save(MODEL_PATH)

            self.set_status(
                f"Training stopped. Final test accuracy: {final_acc * 100:.2f}% (model saved to {MODEL_PATH})"
            )
            self.test_button.configure(state=tk.NORMAL)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Training Error", str(exc))
            self.set_status("Training failed. See the error message above.")
        finally:
            self.set_buttons_enabled(True)
            self.stop_button.configure(state=tk.DISABLED)

    def update_dashboard(self, epoch, logs, sample_images, sample_labels, sample_predictions):
        self.train_losses.append(logs.get("loss"))
        self.test_losses.append(logs.get("val_loss"))
        self.train_accs.append(logs.get("accuracy"))
        self.test_accs.append(logs.get("val_accuracy"))

        self.epoch_var.set(f"Epoch: {epoch + 1}")
        self.train_loss_var.set(f"Train Loss: {logs.get('loss'):.3f}")
        self.test_loss_var.set(f"Test Loss: {logs.get('val_loss'):.3f}")
        self.train_acc_var.set(f"Train Accuracy: {logs.get('accuracy') * 100:.2f}%")
        self.test_acc_var.set(f"Test Accuracy: {logs.get('val_accuracy') * 100:.2f}%")

        self.last_sample_images = sample_images
        self.last_sample_labels = sample_labels
        self.last_sample_predictions = sample_predictions

        self._redraw_dashboard_charts()
        self.dashboard_canvas.draw()
        self.root.update_idletasks()
        self.root.update()

    # ---------------- Handwriting testing ----------------
    def _run_prediction_on_image(self, file_path):
        try:
            original_image = Image.open(file_path)
            original_image.load()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Could Not Open Image", f"This file could not be opened as an image.\n\n{exc}")
            return

        try:
            model_input, processed_display = preprocess_for_model(original_image)
        except NoDigitFoundError as exc:
            messagebox.showerror("No Digit Found", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Preprocessing Error", str(exc))
            return

        probabilities = self.model.predict(model_input, verbose=0)[0]
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit]) * 100

        self.last_probabilities = probabilities
        self.last_predicted_digit = predicted_digit

        self.prediction_var.set(f"Prediction: {predicted_digit}    Confidence: {confidence:.1f}%")

        self._show_image_in_label(original_image, self.original_image_label, max_size=(220, 220))
        self._show_image_in_label(processed_display, self.processed_image_label, max_size=(220, 220))

        self._update_confidence_text()
        self._update_probability_chart()

    def _show_image_in_label(self, pil_image, label_widget, max_size):
        display_copy = pil_image.copy()
        display_copy.thumbnail(max_size)
        photo = ImageTk.PhotoImage(display_copy)
        label_widget.configure(image=photo)
        if label_widget is self.original_image_label:
            self.displayed_photo = photo
        else:
            self.displayed_processed_photo = photo

    def _update_confidence_text(self):
        """Redraws the digit-by-digit confidence text from self.last_probabilities."""
        self.confidence_text.configure(state=tk.NORMAL)
        self.confidence_text.delete("1.0", tk.END)
        self.confidence_text.tag_configure("predicted", foreground=COLOR_ACCENT, font=("Courier", 11, "bold"))
        if self.last_probabilities is not None:
            for digit in range(10):
                marker = " <--" if digit == self.last_predicted_digit else ""
                line = f"{digit}  {self.last_probabilities[digit] * 100:5.1f}%{marker}\n"
                tag = "predicted" if digit == self.last_predicted_digit else ()
                self.confidence_text.insert(tk.END, line, tag)
        self.confidence_text.configure(state=tk.DISABLED)

    def _update_probability_chart(self):
        """Redraws the confidence bar chart from self.last_probabilities."""
        self.probability_ax.clear()
        self.probability_ax.set_title("Prediction confidence by digit")
        self.probability_ax.set_ylim(0, 100)
        self.probability_ax.set_xticks(range(10))
        if self.last_probabilities is not None:
            colors = [COLOR_ACCENT if d == self.last_predicted_digit else COLOR_AXIS for d in range(10)]
            self.probability_ax.bar(range(10), self.last_probabilities * 100, color=colors)
        _style_axes(self.probability_ax)
        self.probability_canvas.draw()


def main():
    apply_chart_style()
    root = tk.Tk()
    DigitRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
