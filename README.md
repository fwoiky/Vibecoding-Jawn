# Teaching a Neural Network to Read Handwriting

A small, beginner-friendly Python project for an ENGR-126 demo. It:

1. Trains a small neural network on the MNIST handwritten-digit dataset,
   showing the learning process live (loss/accuracy graphs, sample
   predictions changing from wrong to right) in a simple window with
   buttons.
2. Saves the trained model to disk so it never has to retrain automatically.
3. Lets you pick a **photo of a digit you wrote by hand** (e.g. a picture
   taken on your phone and saved to the laptop) and shows the network's
   prediction, its confidence, and a breakdown of confidence for every
   digit 0-9.

Everything is **one file**: `number_recognizer.py`. Just open it in
PyCharm and run it — no other files to copy, no folder structure to set
up.

```
number_recognizer.py  <- RUN THIS. The whole program.
requirements.txt      <- Exact libraries needed.
PLANNING.md            <- One-page project plan/status (for your own reference).
saved_model/            <- Where the trained model gets saved (created for you).
```

---

## 1. IMPORTANT: Python version compatibility

Your laptop currently has **Python 3.14** installed. As of now, TensorFlow
(the library this project uses for the neural network) does **not** yet
publish installable packages for Python 3.14 — it typically supports up to
around 3.12/3.13. Running `pip install tensorflow` directly on 3.14 will
fail with an error like "no matching distribution found."

**The fix is not to remove or downgrade your main Python install.** Instead,
install an *additional* Python version (3.11 is a safe, well-supported
choice) just for this project, and point PyCharm at it. Multiple Python
versions can happily live on the same laptop side by side — this is normal
practice for machine-learning projects, since ML libraries usually lag
behind the newest Python release.

### Step-by-step (do this once)

1. **Install Python 3.11** alongside your existing 3.14:
   - Go to https://www.python.org/downloads/ and download the Python 3.11
     installer for your OS (Windows/macOS). Run it normally — this does
     **not** replace or remove your 3.14 install.
   - (Mac/Linux alternative: if you use `pyenv`, run `pyenv install 3.11.9`.)

2. **Open this project folder in PyCharm** (`File > Open`, select the
   folder containing this README).

3. **Create a project virtual environment using Python 3.11:**
   - `File > Settings` (Windows/Linux) or `PyCharm > Settings` (macOS)
   - `Project: <name> > Python Interpreter`
   - Click **Add Interpreter > Add Local Interpreter**
   - Choose **Virtualenv Environment > New**
   - For "Base interpreter", browse to the Python **3.11** you just
     installed (on Windows this is typically something like
     `C:\Users\<you>\AppData\Local\Programs\Python\Python311\python.exe`; on
     macOS/Linux, try `/usr/local/bin/python3.11` or wherever the installer
     put it — PyCharm usually finds it automatically in the dropdown).
   - Click OK. PyCharm creates a `.venv` folder using Python 3.11, isolated
     from your system Python 3.14.

4. **Install the dependencies** — open the PyCharm terminal (bottom of the
   window; it should already be using your new 3.11 virtual environment)
   and run:
   ```
   pip install -r requirements.txt
   ```

5. **Verify:** in the same terminal, run `python --version` — it should now
   print `Python 3.11.x` (not 3.14), confirming PyCharm is using the right
   environment for this project.

If a future project needs a *different* Python version, just repeat step 3
for that project — your 3.11 and 3.14 installs stay untouched.

### Optional: use your Mac's GPU instead of just the CPU

If you're on an Apple Silicon Mac (M1/M2/M3/M4/M5), TensorFlow runs on CPU
only by default -- one extra package lets it use the GPU instead, which can
noticeably cut training time. In the same terminal:
```
pip install tensorflow-metal
```
No code changes needed; TensorFlow detects it automatically. To confirm it
worked:
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
You should see a GPU device listed (not an empty list).

---

## 2. Running the program

With the virtual environment set up and dependencies installed:

- In PyCharm, right-click `number_recognizer.py` and choose **Run**.
- Or from a terminal (with the virtual environment active):
  ```
  python number_recognizer.py
  ```

A window titled "Teaching a Neural Network to Read Handwriting" opens with
buttons across the top.

---

## 3. Using the program

### First time: Train the model

Click **Train Model**.

- The first click downloads the MNIST dataset (~11 MB, needs internet,
  cached afterward so this only happens once).
- The network then **trains indefinitely** — there's no fixed number of
  epochs. Click the red **Stop Training** button whenever you want it to
  stop (it finishes the current epoch first, then saves). Each epoch only
  covers part of the training data (not the full 60,000 images), so it
  finishes fast and the dashboard updates often; the network still works
  through the rest of the data on the epochs that follow. On a normal
  laptop CPU, expect roughly 10-20 seconds per epoch (faster with a GPU —
  see the optional Apple Silicon step above). Longer training generally
  means higher accuracy, with diminishing returns after a while.
- While it trains, the dashboard updates **after every epoch** showing:
  - Current epoch number
  - Training loss / test loss (how wrong the model is — lower is better)
  - Training accuracy / test accuracy (percent correct — higher is better)
  - Live loss and accuracy graphs
  - 8 sample digits from the test set with the model's current guess next
    to the correct answer — printed in **red** if wrong, **green** if
    correct. Watch these flip from red to green as training progresses;
    that's the network visibly learning.
- When you click **Stop Training** (or close in on a very high safety
  ceiling of epochs, which you're very unlikely to ever reach), the status
  bar shows the final test accuracy, and the trained model is
  automatically saved to `saved_model/digit_model.keras`. A well-trained
  run typically lands around 99.2-99.5% -- note that **99.9% is not a
  realistic target** for a network this size; even research models tuned
  specifically for MNIST rarely clear ~99.7-99.8%.

### Later runs: skip retraining

Once a model has been trained and saved, you don't need to train again.
Click **Load Saved Model** to instantly load it from disk.

If you ever want a fresh model (e.g. after changing the code), click
**Retrain Model** — it asks for confirmation first, since it overwrites the
saved model.

### Testing your own handwriting

1. Write a single digit (0-9) on plain paper with a dark pen/pencil,
   fairly large and clear, and take a photo with your phone.
2. Transfer/save the photo to your laptop (AirDrop, email, USB, cloud
   drive — whatever's easiest).
3. In the app, click **Test My Handwriting**, then **Choose Image...** and
   select the photo file.
4. The screen shows:
   - The original photo
   - The exact 28x28 image that was actually fed into the neural network
     (this is what "preprocessing" produced — see below)
   - The predicted digit and confidence, e.g. `Prediction: 7  Confidence: 96.3%`
   - A text list and a bar chart showing the confidence for every digit
     0-9, so the audience can see the model wasn't just guessing — it
     considered all ten possibilities and one clearly won out.

Click **Back to Training Dashboard** any time to return to the training
screen.

---

## 4. How the handwriting photo gets processed

A phone photo looks nothing like an MNIST training image (it has a paper
background, shadows, and the digit could be anywhere in the frame, any
size, any thickness). The `preprocess_for_model()` function in
`number_recognizer.py` fixes this step by step, and the processed result
is shown right in the app so the transformation is visible to the
audience:

1. **Grayscale** — color doesn't matter for digit shape.
2. **Shadow/lighting correction** — a blurred copy of the photo estimates
   the background brightness, which is subtracted out. This flattens
   shadows and uneven lighting.
3. **Polarity detection** — the four corners of the photo are assumed to be
   background. If they're light (typical: dark pen on white paper), the
   image is inverted so the digit ends up bright-on-dark, matching MNIST's
   convention.
4. **Thresholding (Otsu's method)** — automatically picks the best
   brightness cutoff to separate the digit from the background, turning
   the photo into a clean black-and-white image.
5. **Largest-shape selection** — if the photo has more than one blob
   (smudges, a second digit, etc.), only the largest one is kept.
6. **Crop + resize + center** — the image is cropped tightly around the
   digit, resized to fit a 20x20 box while keeping its proportions, and
   placed in the middle of a 28x28 canvas — the same layout MNIST images
   use.
7. **Normalize** — pixel values are scaled to 0-1 for the network.

If no digit can be found at all (e.g. a blank or unreadable photo), the app
shows a clear error message instead of crashing or guessing.

---

## 5. How the neural network works (for your presentation)

In plain terms, without heavy math:

- The network is a **Convolutional Neural Network (CNN)** — a design
  especially good at recognizing shapes in images.
- **Convolutional layers** slide small filters across the image, each
  learning to detect a simple pattern (an edge, a curve, a loop). Early
  layers detect simple strokes; combining many of them lets later layers
  recognize more complex shapes — eventually, whole digits.
- **Pooling layers** shrink the image between convolutions, keeping the
  strongest signals and discarding unnecessary detail. This also makes the
  network more tolerant of the digit being a bit off-center or a different
  size, which matters a lot for a real phone photo.
- **Dense (fully connected) layers** at the end combine everything the
  convolutional layers found into a final decision.
- The last layer outputs **10 numbers that add up to 100%** — one
  probability per digit 0-9 (via a "softmax" activation). The digit with
  the highest number is the prediction.
- **Training** works by showing the network thousands of labeled examples.
  For each one, it makes a guess, compares the guess to the correct answer
  (this difference is the "loss"), and slightly adjusts its internal
  numbers (weights) to make a better guess next time — a process called
  **backpropagation** combined with **gradient descent**. Repeating this
  over the whole dataset multiple times (epochs) is what the loss/accuracy
  graphs in the dashboard are tracking.
- The network never sees your handwriting during training — the "Test My
  Handwriting" step shows it generalizing to something new, which is the
  whole point of machine learning.

---

## 6. Troubleshooting

| Problem | What to do |
|---|---|
| `pip install` fails for tensorflow | You're probably still on Python 3.14 — see Section 1 and make sure PyCharm's interpreter is 3.11. |
| "No saved model found" | Click **Train Model** first — this only needs to be done once. |
| "This file could not be opened as an image" | The file may be corrupted or not actually an image — try re-saving/re-exporting the photo. |
| "No digit could be detected in this image" | Use a photo with a single, clearly dark (or bright) digit against a plain, evenly lit background, filling a good portion of the frame. |
| MNIST download fails (first Train click) | Check your internet connection — the first training run needs to download the dataset once (~11 MB). |
| "Missing required libraries" message on startup | Run `pip install -r requirements.txt` inside your 3.11 virtual environment. |
| Training seems to freeze the window | This is expected — the dashboard updates once per epoch (not continuously). Training runs until you click **Stop Training**, so the window stays busy the whole time you let it run. |
| Training feels slow / "less than 1 epoch a second" | Normal — each epoch still processes thousands of images in small batches, which takes real time even on fast hardware. If it feels too slow, make sure you're not accidentally on CPU when you meant to use a GPU (see the Apple Silicon step above), and double-check you're running the up-to-date file, not a stale copy. |

---

## 7. Retraining / resetting

To force a completely fresh model, either click **Retrain Model** in the
app, or manually delete the file `saved_model/digit_model.keras` and click
**Train Model** again.
