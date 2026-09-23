"""
app.py
------
RUN THIS FILE. Main entry point for the "Teaching a Neural Network to Read
Handwriting" demo.

This builds a simple Tkinter window with buttons:
  - Train Model            : trains the network from scratch, showing a
                              live dashboard of it learning
  - Load Saved Model       : loads a model that was already trained before
  - Retrain Model          : same as Train Model, but confirms first since
                              it overwrites the saved model
  - Test My Handwriting    : switches to the screen where you can pick a
                              photo of a digit you wrote and see the
                              network's prediction

For an ENGR-126 audience: everything happening "under the hood" (loading
data, building the network, training, preprocessing a photo) lives in the
other files in this folder (data.py, model.py, preprocess.py) with comments
explaining each step. This file is just the window and the buttons that
tie it all together.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

import data
import model as model_module
import preprocess

NUM_SAMPLE_PREDICTIONS = 8


class DashboardCallback(tf.keras.callbacks.Callback):
    """
    A Keras callback that updates the live training dashboard after every
    epoch. Keras calls on_epoch_begin/on_epoch_end automatically during
    model.fit() -- this subclasses keras.callbacks.Callback so Keras'
    internal machinery (which relies on other hooks defined on the base
    class) works correctly.
    """

    def __init__(self, app, sample_images, sample_labels):
        super().__init__()
        self.app = app
        self.sample_images = sample_images
        self.sample_labels = sample_labels

    def on_epoch_begin(self, epoch, logs=None):
        self.app.set_status(f"Training epoch {epoch + 1} of {model_module.EPOCHS}...")

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
        self.displayed_photo = None  # keeps a reference so Tkinter doesn't garbage-collect it
        self.displayed_processed_photo = None

        self._build_layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        button_bar = ttk.Frame(self.root, padding=8)
        button_bar.pack(side=tk.TOP, fill=tk.X)

        # Buttons that should be disabled while training is in progress
        # (tracked explicitly, rather than by widget position, so this
        # keeps working correctly even if the layout above changes).
        self.main_buttons = []

        train_button = ttk.Button(button_bar, text="Train Model", command=self.on_train_clicked)
        train_button.pack(side=tk.LEFT, padx=4)
        self.main_buttons.append(train_button)

        load_button = ttk.Button(
            button_bar, text="Load Saved Model", command=self.on_load_clicked
        )
        load_button.pack(side=tk.LEFT, padx=4)
        self.main_buttons.append(load_button)

        retrain_button = ttk.Button(
            button_bar, text="Retrain Model", command=self.on_retrain_clicked
        )
        retrain_button.pack(side=tk.LEFT, padx=4)
        self.main_buttons.append(retrain_button)

        self.test_button = ttk.Button(
            button_bar,
            text="Test My Handwriting",
            command=self.show_test_screen,
            state=tk.DISABLED,
        )
        self.test_button.pack(side=tk.LEFT, padx=4)

        back_button = ttk.Button(
            button_bar, text="Back to Training Dashboard", command=self.show_dashboard_screen
        )
        back_button.pack(side=tk.LEFT, padx=4)
        self.main_buttons.append(back_button)

        self.status_var = tk.StringVar(value="Click 'Train Model' or 'Load Saved Model' to begin.")
        ttk.Label(self.root, textvariable=self.status_var, padding=6).pack(
            side=tk.TOP, fill=tk.X
        )

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
            ttk.Label(stats, textvariable=var, font=("TkDefaultFont", 11, "bold")).pack(
                side=tk.LEFT, padx=12
            )

        self.dashboard_figure = Figure(figsize=(10, 5.5), dpi=100)
        grid = self.dashboard_figure.add_gridspec(2, 8)
        self.loss_ax = self.dashboard_figure.add_subplot(grid[0, 0:4])
        self.acc_ax = self.dashboard_figure.add_subplot(grid[1, 0:4])
        self.sample_axes = [
            self.dashboard_figure.add_subplot(grid[r, 4 + c])
            for r in range(2)
            for c in range(4)
        ]
        self._reset_dashboard_plots()

        self.dashboard_canvas = FigureCanvasTkAgg(self.dashboard_figure, master=self.dashboard_frame)
        self.dashboard_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _build_test_screen(self):
        top = ttk.Frame(self.test_frame, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(top, text="Choose Image...", command=self.on_choose_image_clicked).pack(
            side=tk.LEFT
        )
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

    # ------------------------------------------------------------------
    # Screen switching
    # ------------------------------------------------------------------
    def show_dashboard_screen(self):
        self.dashboard_frame.tkraise()

    def show_test_screen(self):
        self.test_frame.tkraise()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def on_train_clicked(self):
        self._train_model()

    def on_retrain_clicked(self):
        if messagebox.askyesno(
            "Retrain Model",
            "This will train a brand new model and overwrite the saved one. Continue?",
        ):
            self._train_model()

    def on_load_clicked(self):
        try:
            self.set_status("Loading saved model...")
            self.model = model_module.load_saved_model()
            self.set_status("Saved model loaded and ready.")
            self.test_button.configure(state=tk.NORMAL)
        except FileNotFoundError as exc:
            messagebox.showerror("No Saved Model", str(exc))
            self.set_status("No saved model found. Click 'Train Model' first.")
        except Exception as exc:  # noqa: BLE001 - show any load error to the user
            messagebox.showerror("Error Loading Model", str(exc))
            self.set_status("Failed to load saved model.")

    def on_choose_image_clicked(self):
        if self.model is None:
            messagebox.showwarning(
                "No Model Loaded",
                "Train or load a model first, then come back and choose an image.",
            )
            return

        file_path = filedialog.askopenfilename(
            title="Choose a photo of a handwritten digit",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        self._run_prediction_on_image(file_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _train_model(self):
        self.set_buttons_enabled(False)
        self.test_button.configure(state=tk.DISABLED)
        try:
            if self.x_train is None:
                self.set_status("Downloading/loading MNIST dataset...")
                self.x_train, self.y_train, self.x_test, self.y_test = data.load_mnist_data()

            self.set_status("Building the neural network...")
            self.model = model_module.build_model()

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
                epochs=model_module.EPOCHS,
                batch_size=model_module.BATCH_SIZE,
                callbacks=[callback],
                verbose=0,
            )

            final_loss, final_acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            os.makedirs(os.path.dirname(model_module.MODEL_PATH), exist_ok=True)
            self.model.save(model_module.MODEL_PATH)

            self.set_status(
                f"Training complete! Final test accuracy: {final_acc * 100:.1f}% "
                f"(model saved to {model_module.MODEL_PATH})"
            )
            self.test_button.configure(state=tk.NORMAL)
        except Exception as exc:  # noqa: BLE001 - surface any training error to the user
            messagebox.showerror("Training Error", str(exc))
            self.set_status("Training failed. See the error message above.")
        finally:
            self.set_buttons_enabled(True)

    def update_dashboard(self, epoch, logs, sample_images, sample_labels, sample_predictions):
        self.train_losses.append(logs.get("loss"))
        self.test_losses.append(logs.get("val_loss"))
        self.train_accs.append(logs.get("accuracy"))
        self.test_accs.append(logs.get("val_accuracy"))

        self.epoch_var.set(f"Epoch: {epoch + 1}/{model_module.EPOCHS}")
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

    # ------------------------------------------------------------------
    # Handwriting testing
    # ------------------------------------------------------------------
    def _run_prediction_on_image(self, file_path):
        try:
            original_image = Image.open(file_path)
            original_image.load()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(
                "Could Not Open Image",
                f"This file could not be opened as an image.\n\n{exc}",
            )
            return

        try:
            model_input, processed_display = preprocess.preprocess_for_model(original_image)
        except preprocess.NoDigitFoundError as exc:
            messagebox.showerror("No Digit Found", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Preprocessing Error", str(exc))
            return

        probabilities = self.model.predict(model_input, verbose=0)[0]
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit]) * 100

        self.prediction_var.set(
            f"Prediction: {predicted_digit}    Confidence: {confidence:.1f}%"
        )

        self._show_image_in_label(original_image, self.original_image_label, max_size=(220, 220))
        self._show_image_in_label(processed_display, self.processed_image_label, max_size=(220, 220))

        self._update_confidence_text(probabilities, predicted_digit)
        self._update_probability_chart(probabilities, predicted_digit)

    def _show_image_in_label(self, pil_image, label_widget, max_size):
        display_copy = pil_image.copy()
        display_copy.thumbnail(max_size)
        photo = ImageTk.PhotoImage(display_copy)
        label_widget.configure(image=photo)
        # Keep a reference so Tkinter doesn't garbage-collect the image.
        if label_widget is self.original_image_label:
            self.displayed_photo = photo
        else:
            self.displayed_processed_photo = photo

    def _update_confidence_text(self, probabilities, predicted_digit):
        self.confidence_text.configure(state=tk.NORMAL)
        self.confidence_text.delete("1.0", tk.END)
        for digit in range(10):
            marker = " <--" if digit == predicted_digit else ""
            self.confidence_text.insert(
                tk.END, f"{digit}  {probabilities[digit] * 100:5.1f}%{marker}\n"
            )
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
    missing = []
    for module_name in ("tensorflow", "numpy", "PIL", "matplotlib"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        print(
            "Missing required libraries: "
            + ", ".join(missing)
            + "\nRun: pip install -r requirements.txt"
        )
        sys.exit(1)

    root = tk.Tk()
    DigitRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
