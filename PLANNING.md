# Project Plan: Teaching a Neural Network to Read Handwriting

**Course:** ENGR-126
**Deliverable:** A Python program that trains a neural network to recognize
handwritten digits, demonstrates the training process live, and then
classifies a photograph of the user's own handwriting.

## 1. Objective

The goal of this project is to build a complete, self-contained
demonstration of supervised machine learning: training a small neural
network on a labeled dataset, visualizing the learning process as it
happens, and applying the trained model to a new, real-world input. The
project is designed to make an abstract concept — "a computer learning
from examples" — visible and understandable to an audience with no machine
learning background.

## 2. Technical Approach

| Decision | Choice | Rationale |
|---|---|---|
| Dataset | MNIST (60,000 training / 10,000 test images of handwritten digits 0-9) | Standard, well-documented benchmark dataset; small enough to train quickly on a laptop CPU |
| Framework | TensorFlow / Keras | Widely used, well-documented, and provides a straightforward API for both building the network and hooking into the training loop for live visualization |
| Model architecture | A compact Convolutional Neural Network (CNN) | CNNs are well-suited to image recognition and generalize better than simpler architectures to the kind of variation (rotation, off-center placement) present in a real handwriting photo |
| Interface | A Tkinter desktop application with embedded matplotlib charts | Provides a self-contained, presentable GUI with live graphs, without requiring a web server or browser |
| Image input | A file picker allowing the user to select a saved photo | Keeps the workflow simple: photograph a digit on a phone, transfer the file to the laptop, and select it in the application |
| Code structure | A single Python file, `number_recognizer.py` | Keeps the project easy to review, submit, and run without managing a multi-file project structure |

### Python version compatibility

The development machine runs Python 3.14, which is newer than what
TensorFlow currently supports (TensorFlow generally supports up through
Python 3.12/3.13). Rather than downgrading the system's Python
installation, the project runs inside a dedicated Python 3.11 virtual
environment, installed alongside the system Python and used only for this
project. This is standard practice for machine learning projects, since
ML libraries typically lag behind the newest Python releases.

## 3. System Design

The program is organized into four stages, all within `number_recognizer.py`:

1. **Data loading** — Downloads and normalizes the MNIST dataset (pixel
   values scaled to the 0-1 range, images reshaped for the network's
   input format).
2. **Model** — Defines a CNN with two convolutional blocks, batch
   normalization, dropout for regularization, and light random
   rotation/translation applied only during training (data augmentation),
   which improves the model's ability to generalize to real, imperfect
   handwriting photos.
3. **Training and live visualization** — Trains the network using an
   open-ended training loop: rather than a fixed number of epochs, the
   user controls training length directly via a "Stop Training" control
   in the interface. After every epoch, the interface updates in real
   time with the current loss and accuracy (both on the training set and
   a held-out test set), line graphs of these metrics over time, and a
   set of sample test images with the model's current prediction shown
   alongside the correct answer.
4. **Handwriting recognition** — Once trained, the model is saved to disk
   so it does not need to be retrained on subsequent runs. The user can
   then select a photograph of their own handwritten digit; the program
   preprocesses it (grayscale conversion, shadow/lighting correction,
   automatic thresholding, cropping, and centering) to match the format
   of the training data, displays both the original and processed image,
   and reports the model's predicted digit along with a confidence score
   for all ten possible digits.

## 4. Implementation Status

| Phase | Description | Status |
|---|---|---|
| 1 | Environment setup and dependency compatibility | Complete |
| 2 | Dataset loading and preprocessing | Complete |
| 3 | Model architecture, training loop, save/load | Complete |
| 4 | Live training dashboard | Complete |
| 5 | Model persistence (train once, reuse across runs) | Complete |
| 6 | Handwriting photo preprocessing and prediction interface | Complete |
| 7 | Light/dark interface theme | Complete |
| 8 | Full end-to-end run-through and demo rehearsal | In progress |

## 5. Expected Results and Known Limitations

- **Accuracy:** A well-trained run typically reaches approximately
  99.2-99.5% accuracy on the MNIST test set. Note that accuracy in the
  99.9% range is not a realistic target for a model of this size — even
  research models specifically tuned for MNIST rarely exceed
  approximately 99.7-99.8%, and doing so generally requires large model
  ensembles beyond the scope of this project.
- **Handwriting recognition accuracy** is expected to be somewhat lower
  than the MNIST test accuracy, since a real photograph differs from the
  clean, pre-processed MNIST images in lighting, stroke thickness, and
  framing. The preprocessing pipeline is designed to close as much of
  this gap as possible, and the transformation is shown step-by-step in
  the interface as part of the demonstration.
- **Preprocessing assumptions:** The handwriting recognition step assumes
  a single digit that occupies a reasonable portion of the frame, against
  a relatively plain background. Images where no digit can be confidently
  located produce a clear error message rather than an incorrect
  prediction.
- **Training duration:** Because training length is user-controlled
  rather than fixed, the demonstration can be shortened or extended as
  needed — for example, running only a few epochs to show early,
  imperfect predictions, or training longer beforehand to demonstrate a
  higher-accuracy result.
