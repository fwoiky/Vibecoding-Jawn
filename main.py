"""
============================================================
                    THE 81ST OBJECT
============================================================

Teach a pretrained YOLO11 detector (80 COCO classes) to recognise a
Rubik's cube as its 81st class, WITHOUT forgetting the original 80.

    Stock YOLO11 ........ 80 known classes   (ids 0-79: person, chair, laptop, cup, ...)
    New training ........ 80 original classes + Rubik's cube
    Final detector ...... 81 classes         (ids 0-79 unchanged, id 80 = rubiks_cube)

This is a small demonstration of CONTINUAL (incremental) LEARNING:

  * If we fine-tuned only on cube photos, the network would slowly
    "forget" people, chairs, laptops ... (catastrophic forgetting),
    because nothing in the training data would remind it of them.

  * So we mix in REPLAY data: ordinary images of everyday scenes whose
    boxes were produced by the ORIGINAL stock YOLO11 model. These are
    PSEUDO-LABELS - the teacher model's best guesses, NOT perfect human
    ground truth - but they keep the 80 original classes represented in
    every training batch.

  * The cube boxes are drawn by YOU (menu option 2). Those are the only
    manual labels in the project.

Run it with:   python main.py

Everything runs locally. The only internet access is the one-time automatic
download of the official pretrained weights file (yolo11n.pt) the very first
time you use them (you can also copy that file into weights/ by hand).
"""

import copy
import datetime
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

# ----------------------------------------------------------------------------
# OpenCV and NumPy are needed for every menu option, so check them right away.
# ----------------------------------------------------------------------------
try:
    import cv2
    import numpy as np
except ImportError as error:
    print("\nERROR: A required package is missing:", error)
    print("Open the PyCharm terminal and run:\n")
    print("    pip install ultralytics\n")
    print("(Ultralytics also installs OpenCV and NumPy.) Then run main.py again.")
    sys.exit(1)


# ============================================================================
# SETTINGS - change these, not the code further down
# ============================================================================

MODEL_NAME = "yolo11n.pt"      # pretrained YOLO11 weights (n=nano is fastest; s/m/l/x are bigger & slower)
EPOCHS = 30                    # passes over the training data (more = better but slower)
IMAGE_SIZE = 640               # training / detection image size in pixels (multiple of 32)
BATCH_SIZE = 8                 # images per training step (lower it if you run out of memory)
CONFIDENCE_THRESHOLD = 0.35    # webcam: only show detections at least this confident

PSEUDO_LABEL_CONFIDENCE = 0.50  # replay: keep stock-YOLO boxes at least this confident as pseudo-labels
VALIDATION_FRACTION = 0.20      # 20 % of images go to validation, 80 % to training
FREEZE_LAYERS = 10              # freeze the first 10 layers (the YOLO11 backbone) during training; 0 = train all
PATIENCE = 0                    # early stopping; 0 = off (the cube score starts near zero, so early
                                #   epochs can look like "no improvement" even though training is working)
OPTIMIZER = "SGD"               # Ultralytics' standard optimizer for fine-tuning ("auto" is too cautious here)
LEARNING_RATE = 0.01            # SGD starting learning rate (Ultralytics default lr0)
WORKERS = 2                     # data-loading processes (0 or 2 is safest on Windows laptops)
DEVICE = None                   # None = automatic (NVIDIA GPU if available, else CPU). Or "cpu", "0", "mps"
RANDOM_SEED = 42                # makes training repeatable

CAMERA_INDEX = 0                # 0 = built-in laptop webcam. Try 1 or 2 if the wrong camera opens
MAX_DATASET_IMAGE_SIDE = 1280   # big phone photos are shrunk to this size in dataset/ (originals untouched)
MAX_REPLAY_IMAGES = 1000        # use at most this many replay images
KEEP_EMPTY_REPLAY_IMAGES = False  # keep replay images where stock YOLO found nothing (as background examples)

CUBE_CLASS_ID = 80              # the new class goes AFTER the 80 COCO classes (0-79)
CUBE_CLASS_NAME = "rubiks_cube"
TRAIN_RUN_NAME = "rubiks_cube_81"


# ============================================================================
# FOLDERS AND FILES (all inside the project folder)
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
PHOTOS_DIR = PROJECT_DIR / "photos"              # YOUR cube photos (never modified)
CUBE_LABELS_DIR = PROJECT_DIR / "cube_labels"    # YOUR manual cube boxes, one .txt per photo (option 2)
REPLAY_DIR = PROJECT_DIR / "replay_images"       # everyday scenes WITHOUT a cube (option 3 pseudo-labels them)
DATASET_DIR = PROJECT_DIR / "dataset"            # generated by option 3 (safe to delete & rebuild)
WEIGHTS_DIR = PROJECT_DIR / "weights"
RUNS_DIR = PROJECT_DIR / "runs"

DATA_YAML = DATASET_DIR / "data.yaml"
LABEL_INFO_JSON = DATASET_DIR / "label_info.json"   # remembers which boxes are manual vs pseudo
STOCK_WEIGHTS = WEIGHTS_DIR / MODEL_NAME
START_WEIGHTS = WEIGHTS_DIR / (Path(MODEL_NAME).stem + "_81class_start.pt")
FINAL_WEIGHTS = WEIGHTS_DIR / "rubiks_cube_81.pt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val")

# OpenCV colours are (Blue, Green, Red)
CUBE_COLOR = (255, 0, 255)      # magenta for the Rubik's cube
DRAG_COLOR = (0, 255, 255)      # yellow while dragging
TEXT_BG = (30, 30, 30)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 200, 0)

MAX_WINDOW_WIDTH = 1200         # large photos are scaled down on screen to fit
MAX_WINDOW_HEIGHT = 750
MIN_WINDOW_WIDTH = 760          # narrow photos get a grey margin so the instructions fit
HEADER_HEIGHT = 58              # instruction bar drawn above images


# ============================================================================
# SMALL HELPER FUNCTIONS
# ============================================================================

def create_project_folders():
    """Create every folder the program uses (existing files are never touched)."""
    folders = [PHOTOS_DIR, CUBE_LABELS_DIR, REPLAY_DIR, WEIGHTS_DIR, RUNS_DIR, DATASET_DIR]
    for split in SPLITS:
        folders.append(DATASET_DIR / "images" / split)
        folders.append(DATASET_DIR / "labels" / split)
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def import_yolo():
    """Import the Ultralytics YOLO class, or explain how to install it. Returns None on failure."""
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        print("\nERROR: Ultralytics (YOLO11) is not installed.")
        print("Open the PyCharm terminal and run:    pip install ultralytics")
    except Exception as error:  # e.g. a broken PyTorch installation
        print("\nERROR: Ultralytics is installed but could not be loaded:")
        print("   ", error)
        print("Try reinstalling:    pip install --upgrade --force-reinstall ultralytics")
    return None


def device_arguments():
    """Extra keyword arguments for YOLO calls: only pass 'device' if the user chose one."""
    return {} if DEVICE is None else {"device": DEVICE}


def load_stock_model():
    """Load the original pretrained 80-class YOLO11 model (downloads it once if needed)."""
    YOLO = import_yolo()
    if YOLO is None:
        return None
    if not STOCK_WEIGHTS.exists():
        print(f"    {MODEL_NAME} not found in weights/ - downloading the official file once (needs internet)...")
    try:
        model = YOLO(str(STOCK_WEIGHTS))
    except Exception as error:
        print(f"\nERROR: Could not load {MODEL_NAME}: {error}")
        print("If you are offline, download the file on another computer from")
        print("    https://github.com/ultralytics/assets/releases")
        print(f"and copy it to:  {STOCK_WEIGHTS}")
        return None
    if len(model.names) != 80:
        print(f"\nERROR: {MODEL_NAME} has {len(model.names)} classes, but this project needs the")
        print("standard 80-class COCO model. Set MODEL_NAME to e.g. 'yolo11n.pt' at the top of main.py.")
        return None
    return model


def list_images(folder):
    """Return the image files directly inside a folder, sorted by name."""
    if not folder.exists():
        return []
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    unsupported = [p.name for p in folder.iterdir() if p.suffix.lower() in {".heic", ".heif"}]
    if unsupported:
        print(f"    NOTE: {len(unsupported)} .HEIC photo(s) in {folder.name}/ can't be read by OpenCV.")
        print("          Convert them to .jpg first (e.g. export as JPEG from your phone/Photos app).")
    return sorted(images, key=lambda p: p.name.lower())


def read_image(path):
    """Read an image as a BGR NumPy array (works with non-English folder names too). None if unreadable."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def write_image(path, image):
    """Save a BGR image as a JPEG file."""
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        encoded.tofile(str(path))
    return ok


def cube_label_path(photo_path):
    """cube_labels/<photo name>.txt  - where the manual box for a photo is stored."""
    return CUBE_LABELS_DIR / (photo_path.stem + ".txt")


def read_yolo_label_file(label_path):
    """Read a YOLO label file -> list of (class_id, x_center, y_center, width, height), plus a list of problems."""
    boxes, problems = [], []
    if not label_path.exists():
        return boxes, ["label file missing"]
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            problems.append(f"line {line_number}: expected 5 numbers, found {len(parts)}")
            continue
        try:
            class_id = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:])
        except ValueError:
            problems.append(f"line {line_number}: not a number")
            continue
        if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)) or w <= 0 or h <= 0:
            problems.append(f"line {line_number}: coordinates must be between 0 and 1")
        boxes.append((class_id, xc, yc, w, h))
    return boxes, problems


def yolo_line(class_id, xc, yc, w, h):
    """Format one box as a YOLO label line: class_id x_center y_center width height (normalised 0-1)."""
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def pixel_box_to_yolo(x1, y1, x2, y2, image_width, image_height):
    """Convert a pixel box (corners) into normalised YOLO numbers (centre + size)."""
    x1, x2 = sorted((max(0, min(x1, image_width)), max(0, min(x2, image_width))))
    y1, y2 = sorted((max(0, min(y1, image_height)), max(0, min(y2, image_height))))
    return ((x1 + x2) / 2 / image_width, (y1 + y2) / 2 / image_height,
            (x2 - x1) / image_width, (y2 - y1) / image_height)


def yolo_to_pixel_box(xc, yc, w, h, image_width, image_height):
    """Convert normalised YOLO numbers back into integer pixel corners (x1, y1, x2, y2)."""
    return (int(round((xc - w / 2) * image_width)), int(round((yc - h / 2) * image_height)),
            int(round((xc + w / 2) * image_width)), int(round((yc + h / 2) * image_height)))


def fit_to_window(image, max_width=MAX_WINDOW_WIDTH, max_height=MAX_WINDOW_HEIGHT):
    """Shrink an image to fit on screen. Returns (display_image, scale)."""
    height, width = image.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return image, scale


def add_header(image, lines, color=WHITE):
    """Put a dark instruction bar with up to 2 lines of text above an image."""
    if image.shape[1] < MIN_WINDOW_WIDTH:  # pad narrow (portrait) images so the help text fits
        padding = np.full((image.shape[0], MIN_WINDOW_WIDTH - image.shape[1], 3), 60, dtype=np.uint8)
        image = np.hstack([image, padding])
    header = np.full((HEADER_HEIGHT, image.shape[1], 3), TEXT_BG, dtype=np.uint8)
    for i, text in enumerate(lines[:2]):
        cv2.putText(header, text, (8, 22 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color if i == 0 else (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([header, image])


def draw_labeled_box(image, x1, y1, x2, y2, text, color, thickness=2):
    """Draw a rectangle with a filled caption above it."""
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    top = max(0, y1 - text_h - 8)
    cv2.rectangle(image, (x1, top), (x1 + text_w + 6, top + text_h + 8), color, -1)
    cv2.putText(image, text, (x1 + 3, top + text_h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA)


def class_color(class_id):
    """A stable, distinct colour for each class id."""
    rng = np.random.default_rng(class_id * 7 + 3)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def window_was_closed(window_name, seen_open):
    """True if the user clicked the window's X button. seen_open is a 1-item list used as memory."""
    try:
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
    except cv2.error:
        return seen_open[0]
    if visible >= 1:
        seen_open[0] = True
        return False
    return seen_open[0]  # only trust "not visible" after we've seen it open (some systems report -1)


def close_windows():
    """Destroy all OpenCV windows (the extra waitKey lets macOS/Linux actually close them)."""
    cv2.destroyAllWindows()
    for _ in range(3):
        cv2.waitKey(1)


def is_enter(key):
    return key in (10, 13)


def ask_yes_no(question, default=False):
    """Ask a y/n question in the console."""
    suffix = " (Y/n): " if default else " (y/N): "
    answer = input(question + suffix).strip().lower()
    if not answer:
        return default
    return answer.startswith("y")


def load_dataset_names():
    """Read the 81 class names from dataset/data.yaml. Returns a dict {id: name} or None."""
    if not DATA_YAML.exists():
        return None
    import yaml  # PyYAML is installed together with Ultralytics
    with open(DATA_YAML, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    return {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else None


# ============================================================================
# WEBCAM HELPERS
# ============================================================================

def open_webcam():
    """Open the laptop webcam. Returns a cv2.VideoCapture or None with a helpful message."""
    capture = None
    if os.name == "nt":  # Windows: DirectShow usually opens the built-in camera faster
        capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = None
    if capture is None:
        capture = cv2.VideoCapture(CAMERA_INDEX)
    ok = capture.isOpened() and capture.read()[0]
    if not ok:
        capture.release()
        print(f"\nERROR: Could not open the webcam (CAMERA_INDEX = {CAMERA_INDEX}).")
        print(" - Close other apps that use the camera (Zoom, Teams, browser tabs).")
        print(" - Windows: Settings > Privacy > Camera > allow desktop apps.")
        print(" - macOS: System Settings > Privacy & Security > Camera > allow PyCharm (then restart PyCharm).")
        print(" - Try CAMERA_INDEX = 1 at the top of main.py.")
        return None
    return capture


def run_webcam_detection(model, window_name, banner):
    """Live detection loop shared by option 1 and option 6. Press Q (or ESC) to return to the menu."""
    capture = open_webcam()
    if capture is None:
        return
    print("    Webcam running. Click the video window and press Q to return to the menu.")
    seen_open = [False]
    last_time = time.time()
    fps = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("    The webcam stopped sending frames.")
                break

            # Run YOLO on this frame. plot() draws the boxes, class names and confidence scores.
            result = model.predict(frame, conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE,
                                   verbose=False, **device_arguments())[0]
            annotated = result.plot()

            # A small list in the corner of what is detected right now, e.g. "laptop 0.87".
            detections = sorted(((float(conf), result.names[int(cls)])
                                 for cls, conf in zip(result.boxes.cls, result.boxes.conf)), reverse=True)
            if detections:
                cv2.rectangle(annotated, (5, 5), (215, 14 + 24 * min(len(detections), 8)), TEXT_BG, -1)
            for i, (conf, name) in enumerate(detections[:8]):
                color = CUBE_COLOR if name == CUBE_CLASS_NAME else WHITE
                cv2.putText(annotated, name, (12, 27 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 1, cv2.LINE_AA)
                cv2.putText(annotated, f"{conf:.2f}", (160, 27 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 1, cv2.LINE_AA)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
            last_time = now
            screen = add_header(annotated, [banner, f"Q = back to menu      {fps:4.1f} FPS"])
            cv2.imshow(window_name, screen)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or window_was_closed(window_name, seen_open):
                break
    finally:
        capture.release()
        close_windows()


def capture_images_from_webcam(destination, prefix, stock_model=None, advice=""):
    """
    Save webcam snapshots into a folder (used to collect replay images or extra cube photos).
    SPACE = save one frame, A = auto-save every second on/off, Q = finish.
    If a stock model is given, its detections are shown on screen (but the SAVED image is the clean frame).
    """
    capture = open_webcam()
    if capture is None:
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    window_name = "Capture images"
    print("    SPACE = save a snapshot    A = auto-save every second (on/off)    Q = finish")
    if advice:
        print("   ", advice)
    saved, auto_save, last_save = 0, False, 0.0
    seen_open = [False]
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            preview = frame
            if stock_model is not None:
                preview = stock_model.predict(frame, conf=PSEUDO_LABEL_CONFIDENCE, imgsz=IMAGE_SIZE,
                                              verbose=False, **device_arguments())[0].plot()
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("a"), ord("A")):
                auto_save = not auto_save
            want_save = key == 32 or (auto_save and time.time() - last_save >= 1.0)
            if want_save:
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                write_image(destination / f"{prefix}_{stamp}.jpg", frame)
                saved += 1
                last_save = time.time()
            status = f"saved: {saved}   auto-save: {'ON' if auto_save else 'off'}"
            screen = add_header(preview, ["SPACE = save   A = auto-save on/off   Q = finish", status])
            if want_save:
                cv2.rectangle(screen, (0, 0), (screen.shape[1] - 1, screen.shape[0] - 1), GREEN, 6)
            cv2.imshow(window_name, screen)
            if key in (ord("q"), ord("Q"), 27) or window_was_closed(window_name, seen_open):
                break
    finally:
        capture.release()
        close_windows()
    print(f"    Saved {saved} new image(s) into {destination.name}/")
    return saved


# ============================================================================
# MENU OPTION 1 - STOCK YOLO WEBCAM TEST  (the "BEFORE" demonstration)
# ============================================================================

def test_stock_yolo():
    """Run the untouched, pretrained 80-class YOLO11 on the webcam."""
    print("\n[1/2] Loading the stock pretrained YOLO11 model...")
    model = load_stock_model()
    if model is None:
        return
    names = list(model.names.values())
    print(f"      Loaded {MODEL_NAME}: {len(names)} classes (e.g. {', '.join(names[:6])}, ...)")
    has_cube = any("rubik" in name.lower() or "cube" in name.lower() for name in names)
    print(f"      Does it have a Rubik's cube class?  {'YES' if has_cube else 'NO - that is our mission!'}")
    print("\n[2/2] Starting webcam... hold up a Rubik's cube: stock YOLO has no name for it.")
    run_webcam_detection(model, "Stock YOLO11 (80 classes)",
                         f"BEFORE: stock {MODEL_NAME} - 80 COCO classes, no '{CUBE_CLASS_NAME}'")
    # The model was only used for prediction; the weights file on disk is never changed.


# ============================================================================
# MENU OPTION 2 - LABEL YOUR PHOTOS  (manual Rubik's cube boxes, class 80)
# ============================================================================

def on_label_mouse(event, x, y, flags, state):
    """Mouse callback for the labeling window: click-and-drag draws a box."""
    y -= HEADER_HEIGHT  # the instruction bar sits above the photo
    x = max(0, min(x, state["display_w"] - 1))
    y = max(0, min(y, state["display_h"] - 1))
    if event == cv2.EVENT_LBUTTONDOWN:
        state["drag_start"] = (x, y)
        state["drag_now"] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state["drag_start"] is not None:
        state["drag_now"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and state["drag_start"] is not None:
        (x1, y1), (x2, y2) = state["drag_start"], (x, y)
        state["drag_start"] = None
        if abs(x2 - x1) >= 5 and abs(y2 - y1) >= 5:  # ignore accidental clicks
            s = state["scale"]  # convert screen pixels back to original photo pixels
            state["boxes"].append((min(x1, x2) / s, min(y1, y2) / s, max(x1, x2) / s, max(y1, y2) / s))
            state["dirty"] = True


def label_cube_images():
    """Show each photo in photos/, let the user drag a box around the cube, save it as YOLO class 80."""
    print(f"\nPhotos folder: {PHOTOS_DIR}")
    photos = list_images(PHOTOS_DIR)

    if not photos:
        print("No photos found. Copy your Rubik's cube photos (.jpg/.png) into the photos/ folder,")
        print("or take some now with the webcam.")
        want_capture = ask_yes_no("Capture cube photos with the webcam now?")
        if not want_capture:
            return
    else:
        want_capture = ask_yes_no(f"Found {len(photos)} photo(s). Capture EXTRA cube photos with the webcam first?")
    if want_capture:
        capture_images_from_webcam(PHOTOS_DIR, "webcam_cube",
                                   advice="Show the cube at different distances, angles and backgrounds.")
        photos = list_images(PHOTOS_DIR)
    if not photos:
        print("Still no photos - nothing to label.")
        return

    # Skip files OpenCV can't open, and photos whose names clash (cube1.jpg + cube1.png would
    # share the label file cube1.txt).
    print("Checking photos...")
    seen, usable = set(), []
    for photo in photos:
        if photo.stem.lower() in seen:
            print(f"    Skipping {photo.name}: another photo already uses the name '{photo.stem}'. Rename it.")
        elif read_image(photo) is None:
            print(f"    Skipping {photo.name}: the file can't be opened as an image.")
        else:
            seen.add(photo.stem.lower())
            usable.append(photo)
    photos = usable
    if not photos:
        print("No readable photos - nothing to label.")
        return

    labeled = sum(cube_label_path(p).exists() for p in photos)
    print(f"{len(photos)} photo(s), {labeled} already labeled. Boxes are saved in cube_labels/ "
          f"as class {CUBE_CLASS_ID} ({CUBE_CLASS_NAME}).")
    print("""
    CONTROLS (click the image window first)
      mouse drag = draw a box around the Rubik's cube (draw more if there are several cubes)
      ENTER      = accept & save the box(es), go to the next photo
      U          = undo the last box
      R          = reset (remove all boxes on this photo; ENTER afterwards deletes the saved label)
      N          = next photo without saving       B = back to previous photo
      Q / ESC    = quit labeling
    Tip: draw the box tightly around the whole cube, edges included.""")

    # Start at the first photo that has no label yet.
    index = next((i for i, p in enumerate(photos) if not cube_label_path(p).exists()), 0)
    window_name = "Label your photos"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    state = {"boxes": [], "drag_start": None, "drag_now": None, "scale": 1.0,
             "display_w": 1, "display_h": 1, "dirty": False}
    cv2.setMouseCallback(window_name, on_label_mouse, state)
    seen_open = [False]
    quit_requested = False

    try:
        while 0 <= index < len(photos) and not quit_requested:
            photo = photos[index]
            image = read_image(photo)
            if image is None:
                print(f"    Could not read {photo.name} - skipping it.")
                index += 1
                continue
            height, width = image.shape[:2]
            display, scale = fit_to_window(image, MAX_WINDOW_WIDTH, MAX_WINDOW_HEIGHT - HEADER_HEIGHT)
            label_path = cube_label_path(photo)

            # Load any previously saved boxes so they can be checked or edited.
            state["boxes"] = []
            saved_boxes, _ = read_yolo_label_file(label_path)
            for class_id, xc, yc, w, h in saved_boxes:
                x1, y1, x2, y2 = yolo_to_pixel_box(xc, yc, w, h, width, height)
                state["boxes"].append((x1, y1, x2, y2))
            state.update(scale=scale, display_w=display.shape[1], display_h=display.shape[0],
                         drag_start=None, dirty=False)

            while True:  # redraw loop for the current photo
                canvas = display.copy()
                for i, (x1, y1, x2, y2) in enumerate(state["boxes"]):
                    draw_labeled_box(canvas, int(x1 * scale), int(y1 * scale), int(x2 * scale),
                                     int(y2 * scale), f"{CUBE_CLASS_NAME} #{i + 1}", CUBE_COLOR)
                if state["drag_start"] is not None:
                    cv2.rectangle(canvas, state["drag_start"], state["drag_now"], DRAG_COLOR, 2)
                status = "saved" if label_path.exists() and not state["dirty"] else (
                    "UNSAVED - press ENTER" if state["dirty"] else "not labeled yet")
                screen = add_header(canvas, [
                    f"Photo {index + 1}/{len(photos)}: {photo.name}   boxes: {len(state['boxes'])}   ({status})",
                    "drag = box   ENTER = save+next   U = undo   R = reset   N = next   B = back   Q = quit"])
                cv2.imshow(window_name, screen)

                key = cv2.waitKey(20) & 0xFF
                if window_was_closed(window_name, seen_open):
                    quit_requested = True
                    break
                if key == 255:
                    continue
                if is_enter(key) or key == 32:
                    if state["boxes"]:
                        lines = [yolo_line(CUBE_CLASS_ID, *pixel_box_to_yolo(*box, width, height))
                                 for box in state["boxes"]]
                        label_path.write_text("\n".join(lines) + "\n")
                        print(f"    Saved {label_path.name}: {len(lines)} box(es), class {CUBE_CLASS_ID}")
                        index += 1
                        break
                    if label_path.exists():
                        label_path.unlink()
                        print(f"    Removed label for {photo.name} (it will not be used for training).")
                        index += 1
                        break
                    print("    No box yet: drag a box around the cube, or press N to skip this photo.")
                elif key in (ord("u"), ord("U")) and state["boxes"]:
                    state["boxes"].pop()
                    state["dirty"] = True
                elif key in (ord("r"), ord("R")):
                    state["boxes"] = []
                    state["dirty"] = True
                elif key in (ord("n"), ord("N")):
                    index += 1
                    break
                elif key in (ord("b"), ord("B")):
                    index = max(0, index - 1)
                    break
                elif key in (ord("q"), ord("Q"), 27):
                    quit_requested = True
                    break
    finally:
        close_windows()

    labeled = sum(cube_label_path(p).exists() for p in photos)
    print(f"\nLabeling finished: {labeled}/{len(photos)} photo(s) have a cube label in cube_labels/.")
    if labeled:
        print("Next step: option 3 (Prepare cube + replay dataset).")


# ============================================================================
# MENU OPTION 3 - PREPARE CUBE + REPLAY DATASET
# ============================================================================

def box_overlap(a, b):
    """For normalised boxes (xc, yc, w, h): return (IoU, fraction of box b that lies inside box a)."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a, area_b = a[2] * a[3], b[2] * b[3]
    union = area_a + area_b - inter
    return (inter / union if union > 0 else 0.0), (inter / area_b if area_b > 0 else 0.0)


def pseudo_label(stock_model, image):
    """
    Ask the ORIGINAL stock YOLO11 what it sees. Its confident boxes become PSEUDO-LABELS for
    classes 0-79. They are the teacher model's guesses - usually right, but NOT perfect ground truth.
    Returns a list of (class_id, xc, yc, w, h, confidence).
    """
    result = stock_model.predict(image, conf=PSEUDO_LABEL_CONFIDENCE, imgsz=IMAGE_SIZE,
                                 verbose=False, **device_arguments())[0]
    boxes = []
    for cls, conf, xywhn in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                result.boxes.xywhn.tolist()):
        boxes.append((int(cls), *xywhn, float(conf)))
    return boxes


def split_for_validation(paths):
    """
    Deterministic 80/20 split: every file gets a fixed pseudo-random number from its NAME
    (an MD5 hash), so re-running gives the same split and no image is ever in both sets.
    """
    ordered = sorted(paths, key=lambda p: hashlib.md5(p.name.encode("utf-8")).hexdigest())
    n_val = int(round(len(ordered) * VALIDATION_FRACTION))
    if len(ordered) >= 2:
        n_val = max(1, n_val)  # keep at least one image for validation
    return {"val": set(ordered[:n_val]), "train": set(ordered[n_val:])}


def unique_name(prefix, stem, used):
    """Make an output file name like 'cube_IMG_001' that hasn't been used yet."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    name, n = f"{prefix}_{safe}", 2
    while name.lower() in used:
        name, n = f"{prefix}_{safe}_{n}", n + 1
    used.add(name.lower())
    return name


def shrink_for_dataset(image):
    """Resize very large photos so training is faster. Normalised YOLO labels stay valid."""
    height, width = image.shape[:2]
    scale = MAX_DATASET_IMAGE_SIDE / max(height, width)
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return image


def clear_generated_dataset():
    """Empty dataset/images/* and dataset/labels/* (generated files only - photos/ is never touched)."""
    for kind in ("images", "labels"):
        folder = DATASET_DIR / kind
        if folder.exists():
            shutil.rmtree(folder)
    for split in SPLITS:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(names81):
    """Write dataset/data.yaml describing the 81-class dataset for Ultralytics."""
    import yaml  # installed together with Ultralytics
    data = {
        "path": DATASET_DIR.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names81),
        "names": names81,
    }
    header = ("# THE 81ST OBJECT - generated by main.py (option 3)\n"
              "# classes 0-79 = original COCO classes (replay data, pseudo-labeled by stock YOLO11)\n"
              f"# class {CUBE_CLASS_ID}   = {CUBE_CLASS_NAME} (labeled by hand in option 2)\n")
    with open(DATA_YAML, "w", encoding="utf-8") as file:
        file.write(header)
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)


def prepare_dataset():
    """Build dataset/ = manual cube labels (class 80) + pseudo-labeled replay images (classes 0-79)."""
    print("\n[1/6] Loading the ORIGINAL stock YOLO11 model (the 'teacher' for replay pseudo-labels)...")
    stock_model = load_stock_model()
    if stock_model is None:
        return
    coco_names = {int(k): str(v) for k, v in stock_model.names.items()}
    names81 = {**coco_names, CUBE_CLASS_ID: CUBE_CLASS_NAME}
    print(f"      {len(coco_names)} original classes + '{CUBE_CLASS_NAME}' as class {CUBE_CLASS_ID} "
          f"= {len(names81)} classes")

    print("\n[2/6] Finding your labeled cube photos...")
    cube_photos = [p for p in list_images(PHOTOS_DIR) if cube_label_path(p).exists()]
    if not cube_photos:
        print("ERROR: No labeled cube photos were found (photos/ + cube_labels/).")
        print("Run option 2 first to draw boxes around the cube.")
        return
    print(f"      {len(cube_photos)} labeled cube photo(s).")

    print("\n[3/6] Finding replay images (everyday scenes for the 80 original classes)...")
    replay_images = list_images(REPLAY_DIR)
    print(f"      {len(replay_images)} image(s) in replay_images/")
    recommended = max(len(cube_photos), 50)
    if len(replay_images) < recommended:
        print(f"      Recommended: at least {recommended} replay images (more is better).")
        print("      Good replay images show people, chairs, laptops, cups, bottles, phones, books...")
        print("      and must NOT contain a Rubik's cube.")
        if ask_yes_no("      Capture replay images with the webcam now?", default=not replay_images):
            capture_images_from_webcam(
                REPLAY_DIR, "webcam_replay", stock_model,
                advice="Point the camera at everyday objects. Keep the Rubik's cube OUT of view!")
            replay_images = list_images(REPLAY_DIR)
    if not replay_images:
        print("\nERROR: The replay dataset is empty, so training would make YOLO forget its 80 classes.")
        print(f"Put everyday photos (without a cube) into {REPLAY_DIR}")
        print("or capture some with the webcam, then run option 3 again.")
        return
    if len(replay_images) > MAX_REPLAY_IMAGES:
        replay_images = sorted(replay_images, key=lambda p: hashlib.md5(p.name.encode()).hexdigest())
        replay_images = sorted(replay_images[:MAX_REPLAY_IMAGES], key=lambda p: p.name.lower())
        print(f"      Using {MAX_REPLAY_IMAGES} of them (MAX_REPLAY_IMAGES).")

    print("\n[4/6] Clearing the old generated dataset/ (your photos and cube_labels/ are NOT touched)...")
    clear_generated_dataset()

    print(f"\n[5/6] Writing images + labels  (pseudo-label confidence >= {PSEUDO_LABEL_CONFIDENCE})")
    label_info = {}          # remembers where every box came from (manual / pseudo + confidence)
    used_names = set()
    stats = {"manual": 0, "pseudo": 0, "dropped_overlap": 0, "skipped_empty": 0, "unreadable": 0}
    class_counts = {}
    image_counts = {(group, split): 0 for group in ("cube", "replay") for split in SPLITS}

    groups = [("cube", cube_photos), ("replay", replay_images)]
    for group, paths in groups:
        split_of = split_for_validation(paths)
        for number, path in enumerate(paths, start=1):
            split = "val" if path in split_of["val"] else "train"
            image = read_image(path)
            if image is None:
                print(f"      could not read {path.name} - skipped")
                stats["unreadable"] += 1
                continue
            image = shrink_for_dataset(image)  # re-saved as a clean JPEG (the original file is untouched)

            boxes = []  # (class_id, xc, yc, w, h, source, confidence)
            if group == "cube":
                # MANUAL labels: the boxes you drew in option 2 (class 80).
                manual, problems = read_yolo_label_file(cube_label_path(path))
                for _, xc, yc, w, h in manual:
                    boxes.append((CUBE_CLASS_ID, xc, yc, w, h, "manual", None))
                if problems:
                    print(f"      WARNING {cube_label_path(path).name}: {'; '.join(problems)}")

            # PSEUDO labels from the stock model for classes 0-79. We also do this on the cube photos:
            # otherwise the person/table/laptop in a cube photo would be taught as "background".
            for class_id, xc, yc, w, h, conf in pseudo_label(stock_model, image):
                # If stock YOLO calls the cube something else (e.g. "clock"), that box is wrong - drop it.
                overlaps_cube = False
                for cube in boxes:
                    if cube[0] == CUBE_CLASS_ID:
                        iou, inside = box_overlap(cube[1:5], (xc, yc, w, h))
                        if iou > 0.5 or inside > 0.8:
                            overlaps_cube = True
                if overlaps_cube:
                    stats["dropped_overlap"] += 1
                    continue
                boxes.append((class_id, xc, yc, w, h, "pseudo", round(conf, 3)))

            if group == "replay" and not boxes and not KEEP_EMPTY_REPLAY_IMAGES:
                stats["skipped_empty"] += 1
                print(f"      [replay {number}/{len(paths)}] {path.name}: stock YOLO found nothing - skipped")
                continue

            out_name = unique_name(group, path.stem, used_names)
            write_image(DATASET_DIR / "images" / split / f"{out_name}.jpg", image)
            lines = [yolo_line(b[0], *b[1:5]) for b in boxes]
            (DATASET_DIR / "labels" / split / f"{out_name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))

            label_info[f"{split}/{out_name}.jpg"] = {
                "source_file": str(path.relative_to(PROJECT_DIR)),
                "boxes": [{"class_id": b[0], "label_source": b[5], "confidence": b[6]} for b in boxes],
            }
            for b in boxes:
                stats[b[5]] += 1
                class_counts[b[0]] = class_counts.get(b[0], 0) + 1
            image_counts[(group, split)] += 1
            n_manual = sum(b[5] == "manual" for b in boxes)
            cube_text = f"{n_manual} manual cube box(es), " if group == "cube" else ""
            print(f"      [{group} {number}/{len(paths)}] {split:<5} {path.name}: "
                  f"{cube_text}{len(boxes) - n_manual} pseudo box(es)")

    print("\n[6/6] Writing dataset/data.yaml (81 classes) and dataset/label_info.json ...")
    write_data_yaml(names81)
    LABEL_INFO_JSON.write_text(json.dumps({
        "note": "label_source 'manual' = drawn by you in option 2 (class 80). "
                "label_source 'pseudo' = predicted by the ORIGINAL stock YOLO11 (classes 0-79); "
                "pseudo-labels are the teacher model's guesses, not verified ground truth.",
        "pseudo_label_confidence_threshold": PSEUDO_LABEL_CONFIDENCE,
        "images": label_info,
    }, indent=1))

    # ---------------- summary ----------------
    print("\n" + "-" * 60)
    print("DATASET READY")
    print("-" * 60)
    for split in SPLITS:
        print(f"  {split:<5}: {image_counts[('cube', split)]:4d} cube images + "
              f"{image_counts[('replay', split)]:4d} replay images")
    print(f"  Manual cube boxes (class {CUBE_CLASS_ID}, drawn by you):          {stats['manual']}")
    print(f"  Pseudo-label boxes (classes 0-79, from stock YOLO11):  {stats['pseudo']}")
    if stats["dropped_overlap"]:
        print(f"  Stock-YOLO boxes dropped because they sat on the cube:  {stats['dropped_overlap']}")
    if stats["skipped_empty"]:
        print(f"  Replay images skipped (stock YOLO found nothing):      {stats['skipped_empty']}")
    represented = sorted((c for c in class_counts if c != CUBE_CLASS_ID), key=lambda c: -class_counts[c])
    print(f"  Original classes represented in replay data: {len(represented)}/80")
    if represented:
        print("  Most common: " + ", ".join(f"{coco_names[c]} ({class_counts[c]})" for c in represented[:8]))
    print("  (Classes with 0 examples get no replay protection - add photos containing them if you care.)")
    if image_counts[("cube", "val")] == 0:
        print("  WARNING: no cube photo in the validation set - label at least 2 photos (30+ is better).")
    print(f"\nFiles: {DATASET_DIR}")
    print("Next step: option 4 (Verify Labels), then option 5 (Train).")


# ============================================================================
# MENU OPTION 4 - VERIFY LABELS
# ============================================================================

def verify_labels():
    """Draw the saved YOLO labels back onto the dataset images so mistakes can be spotted."""
    names = load_dataset_names()
    if names is None:
        print("\nNo dataset/data.yaml found. Run option 3 (Prepare cube + replay dataset) first.")
        return
    images = []
    for split in SPLITS:
        images += [(split, p) for p in list_images(DATASET_DIR / "images" / split)]
    if not images:
        print("\nThe dataset is empty. Run option 3 first.")
        return

    info = {}
    if LABEL_INFO_JSON.exists():
        try:
            info = json.loads(LABEL_INFO_JSON.read_text()).get("images", {})
        except (ValueError, OSError):
            info = {}

    # ---- quick text check of every label file ----
    print(f"\n[1/2] Checking {len(images)} label files...")
    bad_files = 0
    for split, image_path in images:
        label_path = DATASET_DIR / "labels" / split / (image_path.stem + ".txt")
        boxes, problems = read_yolo_label_file(label_path)
        problems += [f"class {b[0]} is not between 0 and {len(names) - 1}" for b in boxes
                     if not 0 <= b[0] < len(names)]
        if problems:
            bad_files += 1
            print(f"      PROBLEM {split}/{label_path.name}: {'; '.join(problems)}")
    print(f"      {bad_files} file(s) with problems." if bad_files else "      All label files look valid.")

    print("\n[2/2] Opening the viewer.")
    print("    N / SPACE = next    B = previous    C = jump to next cube image    Q / ESC = quit")
    print("    MAGENTA = rubiks_cube (manual label)     other colours = pseudo-labels from stock YOLO11")
    print("    Bad cube box?    Fix it in option 2, then re-run option 3.")
    print("    Bad pseudo-label? Remove that image from replay_images/ or raise PSEUDO_LABEL_CONFIDENCE.")

    window_name = "Verify Labels"
    index = 0
    seen_open = [False]
    try:
        while True:
            split, image_path = images[index]
            image = read_image(image_path)
            if image is None:
                image = np.zeros((480, 640, 3), dtype=np.uint8)
            height, width = image.shape[:2]
            display, scale = fit_to_window(image, MAX_WINDOW_WIDTH, MAX_WINDOW_HEIGHT - HEADER_HEIGHT)
            label_path = DATASET_DIR / "labels" / split / (image_path.stem + ".txt")
            boxes, problems = read_yolo_label_file(label_path)
            box_info = info.get(f"{split}/{image_path.name}", {}).get("boxes", [])
            if len(box_info) != len(boxes):
                box_info = [{}] * len(boxes)  # labels were edited by hand - no source info

            n_cube = 0
            for (class_id, xc, yc, w, h), extra in zip(boxes, box_info):
                x1, y1, x2, y2 = yolo_to_pixel_box(xc, yc, w, h, display.shape[1], display.shape[0])
                name = names.get(class_id, f"UNKNOWN CLASS {class_id}")
                if class_id == CUBE_CLASS_ID:
                    n_cube += 1
                    draw_labeled_box(display, x1, y1, x2, y2, f"{name} (manual)", CUBE_COLOR, 3)
                else:
                    conf = extra.get("confidence")
                    text = f"{name} {conf:.2f} (pseudo)" if conf is not None else name
                    draw_labeled_box(display, x1, y1, x2, y2, text, class_color(class_id))

            line1 = (f"{split} | {index + 1}/{len(images)} | {image_path.name} | "
                     f"{n_cube} cube box(es), {len(boxes) - n_cube} pseudo box(es)")
            if problems:
                line1 = "PROBLEM: " + "; ".join(problems)
            screen = add_header(display, [line1, "N = next   B = previous   C = next cube image   Q = quit"],
                                RED if problems else WHITE)
            cv2.imshow(window_name, screen)

            key = 255
            while key == 255 and not window_was_closed(window_name, seen_open):
                key = cv2.waitKey(50) & 0xFF  # wait for a key press (255 = no key yet)
            if key == 255 or key in (ord("q"), ord("Q"), 27):
                break  # window closed or Q pressed
            if key in (ord("n"), ord("N"), 32):
                index = (index + 1) % len(images)
            elif key in (ord("b"), ord("B")):
                index = (index - 1) % len(images)
            elif key in (ord("c"), ord("C")):
                for step in range(1, len(images) + 1):
                    j = (index + step) % len(images)
                    if images[j][1].name.startswith("cube_"):
                        index = j
                        break
    finally:
        close_windows()


# ============================================================================
# MENU OPTION 5 - TRAIN THE NEW 81-CLASS MODEL
# ============================================================================
#
# HOW THE 80 -> 81 CLASS CHANGE WORKS (and why we do it ourselves)
#
# A YOLO11 detector = backbone + neck (general visual features, independent of the number of classes)
#                     + detection head. Inside the head:
#       * the BOX branch (cv2) predicts box coordinates  -> does not depend on the number of classes
#       * the CLASS branch (cv3) predicts one score per class -> its output has nc channels
#   In YOLO11 the class branch's hidden width is also max(channels, min(nc, 100)), so for yolo11n
#   going from 80 to 81 classes changes the shape of EVERY layer of the class branch, not just the last.
#
# If you simply train "yolo11n.pt" on an 81-class data.yaml, Ultralytics builds a new 81-class model
# and copies only the weights whose shapes still match. Tensors whose shape changed are left randomly
# initialised - for yolo11n that is the whole class branch (newer Ultralytics versions re-map the final
# layer's rows by class name, but the hidden layers in front of it are still random). So the model would
# have to re-learn "what is a person" from our small, pseudo-labeled replay set.
#
# Instead, build_81_class_start_model() WIDENS the stock network carefully:
#   * every tensor with an unchanged shape is copied exactly (backbone, neck, box branch, DFL)
#   * for every widened tensor the old 80-channel weights are copied into the top-left corner,
#     and weights that connect the NEW channel into OLD channels are set to 0
#   * the new class-80 output row starts at zero weights + the standard low starting bias
# Result: at step 0 the 81-class model gives EXACTLY the same scores for the 80 original classes as the
# stock model (we check this), and the cube score starts near 0. Training then only has to learn the cube,
# while the replay data keeps the 80 original classes from drifting.
#
# The saved start model is a normal Ultralytics checkpoint, so model.train() loads 100 % of its weights.
# ============================================================================

def build_81_class_start_model():
    """Create weights/<model>_81class_start.pt: an 81-class YOLO11 that behaves like the stock model."""
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect
    from ultralytics.nn.tasks import DetectionModel

    # Load a fresh copy of the pretrained file (a model that already ran predict() has merged layers).
    stock_model = YOLO(str(STOCK_WEIGHTS))
    old_net = copy.deepcopy(stock_model.model).float().eval()
    config = copy.deepcopy(old_net.yaml)                 # the YOLO11 architecture description
    new_net = DetectionModel(config, nc=81, verbose=False)  # same architecture, 81-class head

    # Find the LAST layer of each class branch (the one that outputs one score per class).
    final_class_layers = set()
    for module_name, module in new_net.named_modules():
        if isinstance(module, Detect):
            for attribute in ("cv3", "one2one_cv3"):
                for i, branch in enumerate(getattr(module, attribute, None) or []):
                    final_class_layers.add(f"{module_name}.{attribute}.{i}.{len(branch) - 1}.weight")

    old_state, new_state = old_net.state_dict(), new_net.state_dict()
    copied, widened = 0, []
    with torch.no_grad():
        for key, old_tensor in old_state.items():
            new_tensor = new_state[key]
            if new_tensor.shape == old_tensor.shape:
                new_tensor.copy_(old_tensor)                          # identical layer: copy as-is
                copied += 1
                continue
            if new_tensor.ndim != old_tensor.ndim or any(n < o for n, o in zip(new_tensor.shape,
                                                                                old_tensor.shape)):
                raise RuntimeError(f"Unexpected layer shape change in {key}")
            corner = tuple(slice(0, size) for size in old_tensor.shape)
            new_tensor[corner] = old_tensor                           # keep all 80-class knowledge
            if new_tensor.ndim >= 2 and new_tensor.shape[1] > old_tensor.shape[1]:
                new_tensor[:old_tensor.shape[0], old_tensor.shape[1]:] = 0  # new channel can't disturb old ones
            if key in final_class_layers:
                new_tensor[old_tensor.shape[0]:] = 0                  # cube score starts at the bias prior
            widened.append(key)
    new_net.load_state_dict(new_state)

    unexpected = [k for k in widened if ".cv3." not in k and "one2one_cv3" not in k]
    if unexpected:
        raise RuntimeError(f"Layers outside the class branch changed shape: {unexpected[:3]}")

    names81 = {**{int(k): str(v) for k, v in stock_model.names.items()}, CUBE_CLASS_ID: CUBE_CLASS_NAME}
    new_net.names = names81
    checkpoint = {
        "date": datetime.datetime.now().isoformat(),
        "version": ultralytics.__version__,
        "license": "AGPL-3.0 (Ultralytics)",
        "docs": "81-class start model built by THE 81ST OBJECT main.py from " + MODEL_NAME,
        "model": new_net.half(),   # official Ultralytics checkpoints also store FP16 weights
        "train_args": {},
    }
    torch.save(checkpoint, START_WEIGHTS)
    return copied, len(widened)


def outputs_match_stock(stock_model, start_model, image):
    """Check that the new 81-class start model detects the 80 original classes exactly like the stock one."""
    kwargs = dict(conf=0.25, imgsz=IMAGE_SIZE, verbose=False, **device_arguments())
    old = stock_model.predict(image, **kwargs)[0].boxes
    new = start_model.predict(image, **kwargs)[0].boxes
    old_list = sorted((int(c), round(float(s), 2)) for c, s in zip(old.cls, old.conf))
    new_list = sorted((int(c), round(float(s), 2)) for c, s in zip(new.cls, new.conf))
    # FP16 storage can move a score by ~0.01, so compare classes exactly and scores approximately.
    same = [c for c, _ in old_list] == [c for c, _ in new_list] and all(
        abs(a[1] - b[1]) <= 0.02 for a, b in zip(old_list, new_list))
    return same, old_list, new_list


def train_model():
    """Fine-tune the 81-class YOLO11 on cube (manual) + replay (pseudo-labeled) data."""
    print("\n[1/5] Checking Ultralytics and the dataset...")
    YOLO = import_yolo()
    if YOLO is None:
        return
    names = load_dataset_names()
    if names is None:
        print("ERROR: dataset/data.yaml not found. Run option 3 (Prepare cube + replay dataset) first.")
        return
    counts = {split: len(list_images(DATASET_DIR / "images" / split)) for split in SPLITS}
    class_ids = set()
    for label_file in (DATASET_DIR / "labels" / "train").glob("*.txt"):
        class_ids.update(b[0] for b in read_yolo_label_file(label_file)[0])
    if counts["train"] == 0:
        print("ERROR: The training dataset is empty. Run option 3 first.")
        return
    if CUBE_CLASS_ID not in class_ids:
        print("ERROR: Training cannot start because no labeled cube images were found in the training set.")
        print("Run option 2 (label photos) and then option 3 again.")
        return
    if not any(c < CUBE_CLASS_ID for c in class_ids):
        print("ERROR: No replay labels (classes 0-79) in the training set - the model would forget them.")
        print("Add images to replay_images/ and run option 3 again.")
        return
    if counts["val"] == 0:
        print("ERROR: The validation set is empty. Label at least 2 cube photos and re-run option 3.")
        return
    print(f"      data.yaml: {len(names)} classes, class {CUBE_CLASS_ID} = '{names.get(CUBE_CLASS_ID)}'")
    print(f"      {counts['train']} training images, {counts['val']} validation images")

    print("\n[2/5] Building the 81-class starting model from the pretrained 80-class weights...")
    print("      (Ultralytics may print 'Overriding model.yaml nc=80 with nc=81' - that is expected.)")
    stock_model = load_stock_model()
    if stock_model is None:
        return
    expected = {**{int(k): str(v) for k, v in stock_model.names.items()}, CUBE_CLASS_ID: CUBE_CLASS_NAME}
    if names != expected:
        print("ERROR: data.yaml class names don't match 'stock COCO names + rubiks_cube'.")
        print("Re-run option 3 to regenerate the dataset.")
        return
    try:
        copied, widened = build_81_class_start_model()
        start_model = YOLO(str(START_WEIGHTS))
    except Exception as error:
        print(f"ERROR: Could not build the 81-class start model: {error}")
        return
    print(f"      Copied {copied} unchanged tensors, widened {widened} class-branch tensors 80 -> 81.")
    print(f"      Start model: {len(start_model.names)} classes, class 80 = '{start_model.names[80]}'")
    sample = read_image(list_images(DATASET_DIR / "images" / "train")[0])
    same, old_list, new_list = outputs_match_stock(stock_model, start_model, sample)
    print(f"      Sanity check - 80 original classes behave exactly like stock YOLO11: {'YES' if same else 'NO'}")
    if not same:
        print(f"      stock: {old_list}\n      start: {new_list}")
    print(f"      Saved: {START_WEIGHTS}")

    print("\n[3/5] Training settings")
    print(f"      MODEL_NAME={MODEL_NAME}  EPOCHS={EPOCHS}  IMAGE_SIZE={IMAGE_SIZE}  BATCH_SIZE={BATCH_SIZE}")
    print(f"      OPTIMIZER={OPTIMIZER}  LEARNING_RATE={LEARNING_RATE}  FREEZE_LAYERS={FREEZE_LAYERS}  "
          f"PATIENCE={PATIENCE}  DEVICE={DEVICE or 'auto'}")
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False
    if not gpu and DEVICE in (None, "cpu"):
        print("      No NVIDIA GPU detected -> training on the CPU. Expect roughly 1-3 minutes per epoch")
        print("      for ~200 images with yolo11n (lower EPOCHS or IMAGE_SIZE at the top of main.py to test).")
    if not ask_yes_no("      Start training now?", default=True):
        return

    print("\n[4/5] Training... (Ultralytics prints a progress table for every epoch; Ctrl+C / Stop to abort)")
    model = YOLO(str(START_WEIGHTS))
    try:
        metrics = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            optimizer=OPTIMIZER,
            lr0=LEARNING_RATE,
            freeze=FREEZE_LAYERS or None,
            patience=PATIENCE,
            workers=WORKERS,
            seed=RANDOM_SEED,
            project=str(RUNS_DIR / "detect"),
            name=TRAIN_RUN_NAME,
            exist_ok=True,
            **device_arguments(),
        )
    except KeyboardInterrupt:
        print("\nTraining stopped by you. Partial results (if any) are in runs/detect/" + TRAIN_RUN_NAME)
        return
    except Exception as error:
        print(f"\nERROR during training: {error}")
        if "out of memory" in str(error).lower():
            print("Your GPU ran out of memory: lower BATCH_SIZE (e.g. 4) or IMAGE_SIZE at the top of main.py.")
        elif "worker" in str(error).lower() or "pickl" in str(error).lower():
            print("Data loader problem: set WORKERS = 0 at the top of main.py and try again.")
        return

    print("\n[5/5] Saving the trained model...")
    best = Path(model.trainer.best) if getattr(model, "trainer", None) else None
    if best is None or not best.exists():
        best = RUNS_DIR / "detect" / TRAIN_RUN_NAME / "weights" / "best.pt"
    if not best.exists():
        print("ERROR: Training finished but best.pt was not found in runs/detect/.")
        return
    shutil.copy2(best, FINAL_WEIGHTS)
    final = YOLO(str(FINAL_WEIGHTS))
    ok = len(final.names) == 81 and final.names[CUBE_CLASS_ID] == CUBE_CLASS_NAME
    print(f"      Final model classes: {len(final.names)}  (0 = '{final.names[0]}', ..., "
          f"80 = '{final.names[CUBE_CLASS_ID]}')  {'OK' if ok else 'UNEXPECTED!'}")
    try:
        ap_classes = [int(c) for c in metrics.box.ap_class_index]
        print(f"      Validation mAP50 (all classes): {metrics.box.map50:.3f}")
        if CUBE_CLASS_ID in ap_classes:
            print(f"      Validation AP50 for {CUBE_CLASS_NAME}: "
                  f"{metrics.box.ap50[ap_classes.index(CUBE_CLASS_ID)]:.3f}")
        print("      (replay classes are scored against pseudo-labels, so treat those numbers as rough.)")
    except Exception:
        pass
    print(f"\nTrained 81-class model saved to:\n    {FINAL_WEIGHTS}")
    print(f"Training charts and logs: {best.parent.parent}")
    print("Next step: option 6 (Test the trained model with webcam).")


# ============================================================================
# MENU OPTION 6 - TEST THE TRAINED MODEL WITH WEBCAM  (the "AFTER" demonstration)
# ============================================================================

def test_trained_model():
    """Run the new 81-class model on the webcam: the cube AND the original objects should be detected."""
    print("\n[1/2] Loading the trained 81-class model...")
    YOLO = import_yolo()
    if YOLO is None:
        return
    weights = FINAL_WEIGHTS
    if not weights.exists():
        fallback = RUNS_DIR / "detect" / TRAIN_RUN_NAME / "weights" / "best.pt"
        if not fallback.exists():
            print(f"No trained model found at {FINAL_WEIGHTS}.")
            print("Run option 5 (Train the new 81-class model) first.")
            return
        weights = fallback
    try:
        model = YOLO(str(weights))
    except Exception as error:
        print(f"ERROR: Could not load {weights}: {error}")
        return
    print(f"      {weights.name}: {len(model.names)} classes, class 80 = '{model.names.get(CUBE_CLASS_ID)}'")
    if len(model.names) != 81:
        print("      WARNING: this is not an 81-class model - re-run option 5.")
    print("\n[2/2] Starting webcam... show the Rubik's cube together with a laptop, cup, phone, person...")
    run_webcam_detection(model, "Trained YOLO11 (81 classes)",
                         f"AFTER: 80 COCO classes + '{CUBE_CLASS_NAME}' (class {CUBE_CLASS_ID})")


# ============================================================================
# MAIN MENU
# ============================================================================

def print_menu():
    print("\n" + "=" * 60)
    print("THE 81ST OBJECT".center(60))
    print("=" * 60)
    print()
    print("1 - Test stock YOLO with webcam")
    print("2 - Label your photos")
    print("3 - Prepare cube + replay dataset")
    print("4 - Verify Labels")
    print("5 - Train the new 81-class model")
    print("6 - Test the trained model with webcam")
    print("7 - Exit")
    print()


def main_menu():
    """Show the menu again and again until the user chooses 7."""
    create_project_folders()
    os.chdir(PROJECT_DIR)  # keep any files Ultralytics creates inside the project folder
    actions = {
        "1": test_stock_yolo,
        "2": label_cube_images,
        "3": prepare_dataset,
        "4": verify_labels,
        "5": train_model,
        "6": test_trained_model,
    }
    while True:
        print_menu()
        try:
            choice = input("Choose 1-7: ").strip()
        except (KeyboardInterrupt, EOFError):
            choice = "7"
        if choice == "7":
            close_windows()
            print("\nGoodbye!")
            break
        action = actions.get(choice)
        if action is None:
            print("Please type a number from 1 to 7.")
            continue
        try:
            action()
        except KeyboardInterrupt:
            print("\nStopped - back to the menu.")
        except Exception as error:
            # Last safety net: show what went wrong but keep the program running.
            print(f"\nUnexpected error: {type(error).__name__}: {error}")
            traceback.print_exc(limit=2)
            print("The program is still running - back to the menu.")
        finally:
            close_windows()


if __name__ == "__main__":
    main_menu()
