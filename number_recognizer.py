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
  3. Click "Train Model" (first time only -- takes 1-3 minutes on a CPU).
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

# How many passes over the full training set to run. Small enough to
# finish in a few minutes on a normal laptop CPU.
EPOCHS = 8
BATCH_SIZE = 128

# How many test images to show live in the training dashboard.
NUM_SAMPLE_PREDICTIONS = 8


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
    Builds a small Convolutional Neural Network (CNN).

    In plain terms:
      Conv2D(8)   -> learns 8 simple stroke/edge patterns
      MaxPool     -> shrinks the image, keeping the strongest signals
      Conv2D(16)  -> combines simple patterns into more complex shapes
      MaxPool     -> shrinks again
      Flatten     -> turns the 2D feature maps into a single list of numbers
      Dense(64)   -> combines all the evidence together
      Dropout     -> randomly ignores some neurons while training, which
                     helps the network generalize instead of memorizing
      Dense(10)   -> one output per digit 0-9, turned into probabilities
                     that add up to 100% (softmax)
    """
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(8, kernel_size=3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(16, kernel_size=3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
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

    display_image = Image.fromarray(canvas).resize((140, 140), Image.NEAREST)
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
    digit_image = digit_image.resize((new_width, new_height), Image.LANCZOS)

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
        self.app.set_status(f"Training epoch {epoch + 1} of {EPOCHS}...")

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


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Teaching a Neural Network to Read Handwriting")
        self.root.geometry("1050x720")

        self.model = None
        self.x_train = self.y_train = self.x_test = self.y_test = None
        self.train_losses, self.test_losses = [], []
        self.train_accs, self.test_accs = [], []
        self.displayed_photo = None
        self.displayed_processed_photo = None

        self._build_layout()

    # ---------------- Layout ----------------
    def _build_layout(self):
        button_bar = ttk.Frame(self.root, padding=8)
        button_bar.pack(side=tk.TOP, fill=tk.X)

        self.main_buttons = []

        train_button = ttk.Button(button_bar, text="Train Model", command=self.on_train_clicked)
        train_button.pack(side=tk.LEFT, padx=4)
        self.main_buttons.append(train_button)

        load_button = ttk.Button(button_bar, text="Load Saved Model", command=self.on_load_clicked)
        load_button.pack(side=tk.LEFT, padx=4)
        self.main_buttons.append(load_button)

        retrain_button = ttk.Button(button_bar, text="Retrain Model", command=self.on_retrain_clicked)
        retrain_button.pack(side=tk.LEFT, padx=4)
        self.main_buttons.append(retrain_button)

        self.test_button = ttk.Button(
            button_bar, text="Test My Handwriting", command=self.show_test_screen, state=tk.DISABLED
        )
        self.test_button.pack(side=tk.LEFT, padx=4)

        back_button = ttk.Button(
            button_bar, text="Back to Training Dashboard", command=self.show_dashboard_screen
        )
        back_button.pack(side=tk.LEFT, padx=4)
        self.main_buttons.append(back_button)

        self.status_var = tk.StringVar(value="Click 'Train Model' or 'Load Saved Model' to begin.")
        ttk.Label(self.root, textvariable=self.status_var, padding=6).pack(side=tk.TOP, fill=tk.X)

        self.screen_container = ttk.Frame(self.root)
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
        stats = ttk.Frame(self.dashboard_frame, padding=8)
        stats.pack(side=tk.TOP, fill=tk.X)

        self.epoch_var = tk.StringVar(value="Epoch: -")
        self.train_loss_var = tk.StringVar(value="Train Loss: -")
        self.test_loss_var = tk.StringVar(value="Test Loss: -")
        self.train_acc_var = tk.StringVar(value="Train Accuracy: -")
        self.test_acc_var = tk.StringVar(value="Test Accuracy: -")

        for var in (
            self.epoch_var,
            self.train_loss_var,
            self.test_loss_var,
            self.train_acc_var,
            self.test_acc_var,
        ):
            ttk.Label(stats, textvariable=var, font=("TkDefaultFont", 11, "bold")).pack(side=tk.LEFT, padx=12)

        self.dashboard_figure = Figure(figsize=(10, 5.5), dpi=100)
        grid = self.dashboard_figure.add_gridspec(2, 8)
        self.loss_ax = self.dashboard_figure.add_subplot(grid[0, 0:4])
        self.acc_ax = self.dashboard_figure.add_subplot(grid[1, 0:4])
        self.sample_axes = [
            self.dashboard_figure.add_subplot(grid[r, 4 + c]) for r in range(2) for c in range(4)
        ]
        self._reset_dashboard_plots()

        self.dashboard_canvas = FigureCanvasTkAgg(self.dashboard_figure, master=self.dashboard_frame)
        self.dashboard_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _build_test_screen(self):
        top = ttk.Frame(self.test_frame, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(top, text="Choose Image...", command=self.on_choose_image_clicked).pack(side=tk.LEFT)
        self.prediction_var = tk.StringVar(value="Prediction: -    Confidence: -")
        ttk.Label(top, textvariable=self.prediction_var, font=("TkDefaultFont", 14, "bold")).pack(
            side=tk.LEFT, padx=20
        )

        images_row = ttk.Frame(self.test_frame, padding=8)
        images_row.pack(side=tk.TOP, fill=tk.X)

        original_box = ttk.LabelFrame(images_row, text="Original Photo")
        original_box.pack(side=tk.LEFT, padx=10)
        self.original_image_label = ttk.Label(original_box)
        self.original_image_label.pack(padx=6, pady=6)

        processed_box = ttk.LabelFrame(images_row, text="What the Network Sees (28x28)")
        processed_box.pack(side=tk.LEFT, padx=10)
        self.processed_image_label = ttk.Label(processed_box)
        self.processed_image_label.pack(padx=6, pady=6)

        readout_box = ttk.LabelFrame(images_row, text="Confidence for each digit")
        readout_box.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        self.confidence_text = tk.Text(readout_box, width=16, height=12, font=("Courier", 11))
        self.confidence_text.pack(padx=6, pady=6)
        self.confidence_text.configure(state=tk.DISABLED)

        self.probability_figure = Figure(figsize=(9, 3), dpi=100)
        self.probability_ax = self.probability_figure.add_subplot(111)
        self.probability_ax.set_title("Prediction confidence by digit")
        self.probability_ax.set_xticks(range(10))
        self.probability_ax.set_ylim(0, 100)
        self.probability_canvas = FigureCanvasTkAgg(self.probability_figure, master=self.test_frame)
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
        self.loss_ax.plot([], [], label="Train")
        self.loss_ax.plot([], [], label="Test")
        self.loss_ax.legend(loc="upper right")

        self.acc_ax.clear()
        self.acc_ax.set_title("Accuracy (higher is better)")
        self.acc_ax.set_xlabel("Epoch")
        self.acc_ax.set_ylim(0, 1)
        self.acc_ax.plot([], [], label="Train")
        self.acc_ax.plot([], [], label="Test")
        self.acc_ax.legend(loc="lower right")

        for ax in self.sample_axes:
            ax.clear()
            ax.axis("off")

    # ---------------- Button handlers ----------------
    def on_train_clicked(self):
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
    def _train_model(self):
        self.set_buttons_enabled(False)
        self.test_button.configure(state=tk.DISABLED)
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

            self.model.fit(
                self.x_train,
                self.y_train,
                validation_data=(self.x_test, self.y_test),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[callback],
                verbose=0,
            )

            final_loss, final_acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            self.model.save(MODEL_PATH)

            self.set_status(
                f"Training complete! Final test accuracy: {final_acc * 100:.1f}% (model saved to {MODEL_PATH})"
            )
            self.test_button.configure(state=tk.NORMAL)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Training Error", str(exc))
            self.set_status("Training failed. See the error message above.")
        finally:
            self.set_buttons_enabled(True)

    def update_dashboard(self, epoch, logs, sample_images, sample_labels, sample_predictions):
        self.train_losses.append(logs.get("loss"))
        self.test_losses.append(logs.get("val_loss"))
        self.train_accs.append(logs.get("accuracy"))
        self.test_accs.append(logs.get("val_accuracy"))

        self.epoch_var.set(f"Epoch: {epoch + 1}/{EPOCHS}")
        self.train_loss_var.set(f"Train Loss: {logs.get('loss'):.3f}")
        self.test_loss_var.set(f"Test Loss: {logs.get('val_loss'):.3f}")
        self.train_acc_var.set(f"Train Accuracy: {logs.get('accuracy') * 100:.1f}%")
        self.test_acc_var.set(f"Test Accuracy: {logs.get('val_accuracy') * 100:.1f}%")

        epochs_so_far = range(1, len(self.train_losses) + 1)

        self.loss_ax.clear()
        self.loss_ax.set_title("Loss (lower is better)")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.plot(epochs_so_far, self.train_losses, label="Train", marker="o")
        self.loss_ax.plot(epochs_so_far, self.test_losses, label="Test", marker="o")
        self.loss_ax.legend(loc="upper right")

        self.acc_ax.clear()
        self.acc_ax.set_title("Accuracy (higher is better)")
        self.acc_ax.set_xlabel("Epoch")
        self.acc_ax.set_ylim(0, 1)
        self.acc_ax.plot(epochs_so_far, self.train_accs, label="Train", marker="o")
        self.acc_ax.plot(epochs_so_far, self.test_accs, label="Test", marker="o")
        self.acc_ax.legend(loc="lower right")

        predicted_labels = np.argmax(sample_predictions, axis=1)
        for i, ax in enumerate(self.sample_axes):
            ax.clear()
            ax.axis("off")
            ax.imshow(sample_images[i].squeeze(), cmap="gray")
            true_label = int(sample_labels[i])
            predicted_label = int(predicted_labels[i])
            color = "green" if predicted_label == true_label else "red"
            ax.set_title(f"true {true_label} / guess {predicted_label}", color=color, fontsize=8)

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

        self.prediction_var.set(f"Prediction: {predicted_digit}    Confidence: {confidence:.1f}%")

        self._show_image_in_label(original_image, self.original_image_label, max_size=(220, 220))
        self._show_image_in_label(processed_display, self.processed_image_label, max_size=(220, 220))

        self._update_confidence_text(probabilities, predicted_digit)
        self._update_probability_chart(probabilities, predicted_digit)

    def _show_image_in_label(self, pil_image, label_widget, max_size):
        display_copy = pil_image.copy()
        display_copy.thumbnail(max_size)
        photo = ImageTk.PhotoImage(display_copy)
        label_widget.configure(image=photo)
        if label_widget is self.original_image_label:
            self.displayed_photo = photo
        else:
            self.displayed_processed_photo = photo

    def _update_confidence_text(self, probabilities, predicted_digit):
        self.confidence_text.configure(state=tk.NORMAL)
        self.confidence_text.delete("1.0", tk.END)
        for digit in range(10):
            marker = " <--" if digit == predicted_digit else ""
            self.confidence_text.insert(tk.END, f"{digit}  {probabilities[digit] * 100:5.1f}%{marker}\n")
        self.confidence_text.configure(state=tk.DISABLED)

    def _update_probability_chart(self, probabilities, predicted_digit):
        self.probability_ax.clear()
        self.probability_ax.set_title("Prediction confidence by digit")
        self.probability_ax.set_ylim(0, 100)
        self.probability_ax.set_xticks(range(10))
        colors = ["green" if d == predicted_digit else "steelblue" for d in range(10)]
        self.probability_ax.bar(range(10), probabilities * 100, color=colors)
        self.probability_canvas.draw()


def main():
    root = tk.Tk()
    DigitRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
