# Project Plan: Teaching a Neural Network to Read Handwriting

**Goal:** Train a small neural network to recognize handwritten digits
(0-9), show it learning live on screen, then use it to recognize a digit
from a photo of your own handwriting. ENGR-126 midterm demo.

## Decisions made

| Question | Decision |
|---|---|
| Framework | TensorFlow/Keras |
| Interface | Tkinter GUI (buttons, live matplotlib graphs) |
| Image input | File picker (choose a saved photo) |
| Python version | Laptop has 3.14 (too new for TensorFlow); project runs in a dedicated **Python 3.11** virtual environment instead — see README |
| Code structure | **One file** — `number_recognizer.py` |

## What the program does

1. **Train Model** — loads MNIST, builds a small CNN (with light data
   augmentation and batch normalization for extra accuracy), and **trains
   indefinitely** until you click **Stop Training**. Each epoch only
   covers part of the data (not the full 60,000 images), so epochs finish
   in roughly 10-20 seconds on CPU and the dashboard updates often; the
   network still works through the rest of the data over subsequent
   epochs. After every epoch, a dashboard updates live: loss graph,
   accuracy graph, and 8 sample digits whose guesses flip from red (wrong)
   to green (correct) as the network improves.
2. Saves the trained model to `saved_model/digit_model.keras` automatically.
3. **Load Saved Model** — skips training on future runs.
4. **Test My Handwriting** — pick a photo of a digit you wrote. The app:
   - shows the original photo
   - preprocesses it (grayscale → shadow correction → auto-invert →
     threshold → crop → center on 28x28) and shows that result too
   - runs it through the network
   - shows the predicted digit, confidence %, and a 0-9 confidence
     breakdown (text + bar chart)

## Status

- [x] Phase 1 — Setup (Python version issue identified, requirements pinned)
- [x] Phase 2 — MNIST loading/normalization
- [x] Phase 3 — Model build/train/evaluate/save/load
- [x] Phase 4 — Live training dashboard (loss/accuracy graphs, sample predictions)
- [x] Phase 5 — Save/load so retraining isn't required on every launch
- [x] Phase 6 — Handwriting photo preprocessing + prediction UI
- [ ] Phase 7 — Full run-through on your laptop (do this next — see README "Running the program")

## Known limits / things to know for the presentation

- Preprocessing assumes a **single digit**, reasonably large in frame, with
  a plain background — very cluttered photos may fail with a clear
  "No digit could be detected" error rather than a wrong guess.
- Training happens on the main thread, so the window is "busy" (not
  frozen — the dashboard updates once per epoch) for as long as you let
  it run. There's no fixed end point; you decide when it's trained enough
  and click **Stop Training**.
- Typical final test accuracy on MNIST: ~99.2-99.5%. **99.9% is not a
  realistic target** for a network this size — even research models
  tuned specifically for MNIST rarely clear ~99.7-99.8%, and that's with
  heavy ensembling. Say the actual final number live in the demo rather
  than promising a specific target in advance. Accuracy on your own
  handwriting will usually be a bit lower — real photos are messier than
  MNIST, which is exactly what the preprocessing step is trying to close
  the gap on, and is worth explaining live in the demo.
