#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J.A.R.V.I.S. V1 - single-file desktop AI assistant (Windows 11 target, runs elsewhere too)

    python jarvis.py            launch the dashboard
    python jarvis.py --check    print a hardware / dependency / model diagnostic and exit

Sections of this file
    1. Config, settings, logging
    2. Central state machine
    3. Local storage (encrypted face profiles)
    4. Vision: person detector, face engine, camera worker (+ consent-based enrollment)
    5. Greetings
    6. Audio in: microphone worker (wake word + endpointing) and speech-to-text
    7. Audio out: pluggable TTS providers + interruptible speaker
    8. AI: pluggable provider + conversation memory
    9. Core controller (wires everything together)
   10. PySide6 dashboard
   11. Entry point

Privacy: camera frames are processed in memory and never saved or uploaded. Only face
embeddings (not photos) of people who explicitly enroll are stored, encrypted, on this PC.
"""
from __future__ import annotations

import os
import sys

# =======================================================================================
# 0. SELF-SETUP  (runs before anything else, with only the standard library)
# =======================================================================================
# Run this file from PyCharm (or `python jarvis.py`) and it will, on first launch only:
#   1. create a private virtual environment in ./.venv
#   2. install every dependency (CUDA build of PyTorch first if an NVIDIA GPU is present)
#   3. restart itself inside that environment
# Later launches skip all of this and start in a couple of seconds.
_REQS = [
    "PySide6-Essentials>=6.6", "numpy>=1.26", "opencv-python>=4.9,<5", "ultralytics>=8.3",
    "python-dotenv>=1.0", "cryptography>=42", "keyring>=24", "requests>=2.31", "psutil>=5.9",
    "sounddevice>=0.4.6", "faster-whisper>=1.0", "onnxruntime>=1.17", "tqdm", "scipy", "scikit-learn",
    "edge-tts>=6.1", "miniaudio>=1.59", "openai>=1.40",
]


def _py_version(exe) -> tuple:
    import subprocess

    try:
        out = subprocess.check_output([str(exe), "-c", "import sys;print(sys.version_info[0],sys.version_info[1])"],
                                      text=True, timeout=20)
        return tuple(int(x) for x in out.split())
    except Exception:
        return (0, 0)


def _find_python(force_uv: bool = False):
    """Return a Python >= 3.10 executable (preferring 3.12/3.11), or None."""
    import shutil
    from pathlib import Path

    if force_uv:
        return None
    if (3, 10) <= sys.version_info[:2] <= (3, 13):
        return sys.executable
    cands = []
    for v in ("3.12", "3.11", "3.13", "3.10"):
        cands += [shutil.which(f"python{v}"), f"/opt/homebrew/bin/python{v}", f"/usr/local/bin/python{v}",
                  f"/Library/Frameworks/Python.framework/Versions/{v}/bin/python3",
                  str(Path.home() / f"AppData/Local/Programs/Python/Python{v.replace('.', '')}/python.exe")]
    for c in cands:
        if c and Path(c).exists() and (3, 10) <= _py_version(c) <= (3, 13):
            return c
    return None


def _ensure_venv(venv, say, force_uv: bool = False):
    """Create ./.jarvis_venv with Python 3.10-3.13 (downloading Python via `uv` if none is installed)."""
    import subprocess
    from pathlib import Path

    venv = Path(venv)
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if py.exists() and (3, 10) <= _py_version(py) <= (3, 13):
        return py
    if venv.exists():
        import shutil

        shutil.rmtree(venv, ignore_errors=True)
    base = _find_python(force_uv)
    if base:
        say(f"Creating private environment with {base} ...")
        if subprocess.call([base, "-m", "venv", str(venv)]) == 0 and py.exists():
            return py
    say("No suitable Python (3.10-3.13) found. Downloading one automatically (one time, ~30 MB)...")
    tool = venv.parent / ".jarvis_uv"
    subprocess.call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", "--target",
                     str(tool), "uv"])
    uv = next((c for c in (tool / "bin/uv", tool / "bin/uv.exe", tool / "Scripts/uv.exe") if c.exists()), None)
    if uv is None:
        sys.exit("Could not set up Python automatically. Install Python 3.12 from https://www.python.org/downloads/ "
                 "and run again.")
    if subprocess.call([str(uv), "venv", "--seed", "--python", "3.12", str(venv)]) != 0 or not py.exists():
        sys.exit("Automatic Python download failed. Install Python 3.12 from https://www.python.org/downloads/ "
                 "and run again.")
    return py


def _bootstrap() -> None:
    if os.environ.get("JARVIS_IN_VENV") == "1" or "--no-bootstrap" in sys.argv:
        if "--no-bootstrap" in sys.argv:
            sys.argv.remove("--no-bootstrap")
        return
    import hashlib
    import shutil
    import subprocess
    from pathlib import Path

    me = Path(__file__).resolve()
    venv = me.parent / ".jarvis_venv"  # deliberately not ".venv": PyCharm owns that one
    say = lambda m: print(f"[JARVIS setup] {m}", flush=True)
    py = _ensure_venv(venv, say)
    pip = lambda *a: subprocess.call([str(py), "-m", "pip", "install", "--disable-pip-version-check", *a])
    has_gpu = bool(shutil.which("nvidia-smi")) or os.path.exists(r"C:\Windows\System32\nvidia-smi.exe")
    sig = hashlib.sha256((repr(_REQS) + str(has_gpu)).encode()).hexdigest()
    marker = venv / ".jarvis_deps"
    if not marker.exists() or marker.read_text().strip() != sig:
        say("Installing dependencies (one time, ~5-15 minutes). Please wait...")
        pip("--upgrade", "pip")
        if has_gpu and os.name == "nt":
            say("NVIDIA GPU found: installing the CUDA build of PyTorch...")
            for tag in ("cu126", "cu124", "cu121"):
                if pip("torch", "torchvision", "--index-url", f"https://download.pytorch.org/whl/{tag}") == 0:
                    break
            else:
                say("CUDA PyTorch not available; JARVIS will use the CPU for person detection.")
        if pip(*_REQS) != 0 or pip("--no-deps", "openwakeword>=0.6.0") != 0:  # no-deps: skips its Linux-only tflite pin
            sys.exit("Dependency installation failed - see the messages above.")
        if has_gpu:
            pip("nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*")  # GPU speech recognition; optional
        marker.write_text(sig)
        say("Setup complete. Starting JARVIS...")
    env = dict(os.environ, JARVIS_IN_VENV="1", JARVIS_PARENT_PID=str(os.getpid()))
    child = subprocess.Popen([str(py), str(me), *sys.argv[1:]], env=env)
    try:
        sys.exit(child.wait())
    except KeyboardInterrupt:
        child.terminate()
        sys.exit(0)


_bootstrap()

# Must be set before cv2 is imported (much faster webcam start-up on Windows).
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import argparse
import base64
import datetime as dt
import hashlib
import json
import logging
import math
import queue
import random
import re
import shutil
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator, Optional

APP_DIR = Path(__file__).resolve().parent
USERS_DIR = APP_DIR / "users"
PROFILES_DIR = USERS_DIR / "enrolled_users"
MODELS_DIR = APP_DIR / "models"
SETTINGS_PATH = USERS_DIR / "settings.json"
for _d in (PROFILES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- .env ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv(APP_DIR / ".env")
except ImportError:  # minimal fallback parser so the app still starts
    _envp = APP_DIR / ".env"
    if _envp.exists():
        for _line in _envp.read_text(encoding="utf-8", errors="ignore").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

def env_file_set(key: str, value: str) -> None:
    """Save KEY=value into ./.env (created if missing) and the running process environment."""
    p = APP_DIR / ".env"
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    out, done = [], False
    for ln in lines:
        if ln.strip().startswith(key + "="):
            out.append(f"{key}={value}")
            done = True
        else:
            out.append(ln)
    if not done:
        out.append(f"{key}={value}")
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    os.environ[key] = value


# ---- hard requirements ----------------------------------------------------------------
try:
    import numpy as np
    import cv2
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
except ImportError as _e:  # pragma: no cover
    print(f"[JARVIS] Missing required package: {_e.name}\n"
          "Run:  pip install -r requirements.txt   (inside the virtual environment)")
    sys.exit(1)

log = logging.getLogger("jarvis")


def setup_logging(verbose: bool = False) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    try:  # log file never contains transcripts, API keys or camera data
        fh = RotatingFileHandler(APP_DIR / "jarvis.log", maxBytes=512_000, backupCount=2, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass
    for noisy in ("httpx", "httpcore", "urllib3", "numba", "matplotlib", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# =======================================================================================
# 1. SETTINGS
# =======================================================================================
# Secrets (OPENAI_API_KEY, ELEVENLABS_API_KEY) are read ONLY from the environment / .env and
# are never written to settings.json. Every key below can be overridden with JARVIS_<NAME>
# in .env, and from the Settings dialog (saved in users/settings.json).
DEFAULTS: dict = {
    # camera / vision
    "camera_source": "0",          # webcam index, or a video file path (handy for testing)
    "camera_fps_idle": 3,          # detection rate when nobody is in the room
    "camera_fps_active": 8,        # detection rate when someone is present
    "person_conf": 0.45,
    "enter_confirm_frames": 3,     # detections needed before "person entered" fires
    "leave_seconds": 4.0,          # seconds without detection before "person left" fires
    "preview": True,
    "face_threshold": 0.363,       # SFace cosine threshold recommended by OpenCV
    "face_confirm_votes": 3,       # consecutive matching frames needed to confirm identity
    "unknown_after_seconds": 6.0,  # give up identifying a person after this long
    "face_min_size": 60,
    # greetings
    "greeting_cooldown_min": 30,
    "unknown_greeting": "none",    # "none" or "neutral"
    "unknown_cooldown_min": 15,
    # audio in
    "mic_device": "",
    "wake_threshold": 0.5,
    "wake_cooldown": 1.5,
    "silence_end_seconds": 0.9,
    "followup_seconds": 8.0,       # how long JARVIS waits for a follow-up question
    "first_listen_seconds": 6.0,
    "whisper_model": "small.en",   # used on GPU
    "whisper_model_cpu": "base.en",
    "whisper_language": "en",
    # audio out
    "tts_provider": "edge",        # edge | elevenlabs | piper
    "edge_voice": "en-GB-RyanNeural",
    "edge_rate": "+5%",
    "elevenlabs_voice_id": "JBFqnCBsd6RMkjVDRZzb",
    "elevenlabs_model": "eleven_flash_v2_5",
    "piper_voice": "en_GB-alan-medium",
    "speaker_device": "",
    "volume": 0.8,
    # ai
    "ai_provider": "openai",
    "openai_model": "gpt-4o-mini",
}


class Settings:
    """Thread-safe settings: defaults < JARVIS_* env vars < users/settings.json."""

    def __init__(self, path: Path = SETTINGS_PATH):
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(self, "_d", dict(DEFAULTS))
        for k in DEFAULTS:
            env = os.getenv("JARVIS_" + k.upper())
            if env is not None and env != "":
                self._d[k] = self._coerce(k, env)
        try:
            if path.exists():
                for k, v in json.loads(path.read_text(encoding="utf-8")).items():
                    if k in DEFAULTS:
                        self._d[k] = self._coerce(k, v)
        except (OSError, ValueError) as e:
            log.warning("Could not read settings: %s", e)

    @staticmethod
    def _coerce(key, value):
        t = type(DEFAULTS[key])
        try:
            if t is bool:
                return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
            return t(value)
        except (TypeError, ValueError):
            return DEFAULTS[key]

    def __getattr__(self, k):
        if k.startswith("_") or k not in DEFAULTS:
            raise AttributeError(k)
        with self._lock:
            return self._d[k]

    def update(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                if k in DEFAULTS:
                    self._d[k] = self._coerce(k, v)
            self.save()

    def save(self) -> None:
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(json.dumps(self._d, indent=2), encoding="utf-8")
            except OSError as e:
                log.warning("Could not save settings: %s", e)


# =======================================================================================
# 2. CENTRAL STATE MACHINE
# =======================================================================================
class State(str, Enum):
    OFFLINE = "OFFLINE"
    INITIALIZING = "INITIALIZING"
    IDLE = "IDLE"
    DETECTING_PERSON = "DETECTING_PERSON"
    RECOGNIZING_USER = "RECOGNIZING_USER"
    GREETING = "GREETING"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"


S = State
ALLOWED: dict[State, set[State]] = {
    S.OFFLINE: {S.INITIALIZING},
    S.INITIALIZING: {S.IDLE, S.ERROR, S.OFFLINE},
    S.IDLE: {S.DETECTING_PERSON, S.RECOGNIZING_USER, S.GREETING, S.LISTENING, S.THINKING, S.OFFLINE, S.ERROR},
    S.DETECTING_PERSON: {S.IDLE, S.RECOGNIZING_USER, S.GREETING, S.LISTENING, S.THINKING, S.OFFLINE, S.ERROR},
    S.RECOGNIZING_USER: {S.IDLE, S.DETECTING_PERSON, S.GREETING, S.LISTENING, S.THINKING, S.OFFLINE, S.ERROR},
    S.GREETING: {S.IDLE, S.DETECTING_PERSON, S.LISTENING, S.OFFLINE, S.ERROR},
    S.LISTENING: {S.THINKING, S.IDLE, S.DETECTING_PERSON, S.OFFLINE, S.ERROR},
    S.THINKING: {S.SPEAKING, S.LISTENING, S.IDLE, S.DETECTING_PERSON, S.OFFLINE, S.ERROR},
    S.SPEAKING: {S.LISTENING, S.IDLE, S.DETECTING_PERSON, S.THINKING, S.OFFLINE, S.ERROR},
    S.ERROR: {S.IDLE, S.OFFLINE, S.INITIALIZING},
}
PASSIVE_STATES = {S.IDLE, S.DETECTING_PERSON, S.RECOGNIZING_USER}


class StateMachine(QObject):
    """Single source of truth for the assistant state. Main-thread only."""

    changed = Signal(str, str, str)  # new, old, detail

    def __init__(self):
        super().__init__()
        self.state = S.OFFLINE
        self.detail = ""

    def go(self, new: State, detail: str = "") -> bool:
        if new == self.state:
            if detail != self.detail:
                self.detail = detail
                self.changed.emit(new.value, new.value, detail)
            return True
        if new not in ALLOWED[self.state]:
            log.debug("Rejected transition %s -> %s", self.state.value, new.value)
            return False
        old, self.state, self.detail = self.state, new, detail
        log.info("State: %s -> %s %s", old.value, new.value, f"({detail})" if detail else "")
        self.changed.emit(new.value, old.value, detail)
        return True


# =======================================================================================
# 3. LOCAL STORAGE - encrypted face profiles (embeddings only, never photos)
# =======================================================================================
class ProfileStore:
    """Each enrolled user is one encrypted file. File names are hashes, so they don't leak names.

    The Fernet key lives in the Windows Credential Manager (via `keyring`) when available,
    otherwise in users/.profile.key (owner-only permissions where the OS supports it).
    This protects against casual copying of the folder; it is not a defence against malware
    running as your user.
    """

    SERVICE = "JARVIS-V1"

    def __init__(self, directory: Path = PROFILES_DIR):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fernet = None

    def _get_fernet(self):
        if self._fernet is not None:
            return self._fernet
        from cryptography.fernet import Fernet

        key = None
        try:
            import keyring

            key = keyring.get_password(self.SERVICE, "profile-key")
            if not key:
                key = Fernet.generate_key().decode()
                keyring.set_password(self.SERVICE, "profile-key", key)
        except Exception:  # no usable keyring backend
            key = None
        if not key:
            kp = USERS_DIR / ".profile.key"
            if kp.exists():
                key = kp.read_text().strip()
            else:
                key = Fernet.generate_key().decode()
                kp.write_text(key)
                try:
                    os.chmod(kp, 0o600)
                except OSError:
                    pass
        self._fernet = Fernet(key.encode())
        return self._fernet

    def _path(self, name: str) -> Path:
        return self.dir / (hashlib.sha256(name.strip().lower().encode()).hexdigest()[:16] + ".profile")

    def save(self, name: str, embeddings: "np.ndarray") -> None:
        payload = {
            "name": name.strip(),
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "embeddings": np.asarray(embeddings, dtype=np.float32).tolist(),
        }
        token = self._get_fernet().encrypt(json.dumps(payload).encode())
        p = self._path(name)
        p.write_bytes(token)
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass

    def load_all(self) -> dict[str, "np.ndarray"]:
        out = {}
        for rec in self.list_users(with_embeddings=True):
            out[rec["name"]] = rec["embeddings"]
        return out

    def list_users(self, with_embeddings: bool = False) -> list[dict]:
        users = []
        for p in sorted(self.dir.glob("*.profile")):
            try:
                d = json.loads(self._get_fernet().decrypt(p.read_bytes()))
                rec = {"name": d["name"], "created": d.get("created", "")}
                if with_embeddings:
                    emb = np.asarray(d["embeddings"], dtype=np.float32)
                    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
                    rec["embeddings"] = emb
                users.append(rec)
            except Exception as e:  # corrupt or encrypted with another key
                log.warning("Skipping unreadable profile %s (%s)", p.name, type(e).__name__)
        return users

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def delete(self, name: str) -> bool:
        p = self._path(name)
        if p.exists():
            p.unlink()
            return True
        return False


# =======================================================================================
# 4. VISION
# =======================================================================================
def download_file(urls, dest: Path, min_bytes: int = 50_000) -> Path:
    """Download to `dest` (only if missing), trying each URL in turn. Guards against Git-LFS pointer
    files and HTML error pages by enforcing a minimum size."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    import requests

    urls = [urls] if isinstance(urls, str) else list(urls)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for url in urls:
        try:
            log.info("Downloading %s", dest.name)
            with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
            if tmp.stat().st_size < min_bytes:
                raise RuntimeError("file too small (probably a Git-LFS pointer)")
            tmp.replace(dest)
            return dest
        except Exception as e:
            last = e
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {dest.name} ({last}). Download it manually from {urls[0]} "
                       f"and place it in the models/ folder.")


_ZOO = "opencv/opencv_zoo/main/models"
YUNET_URLS = [f"https://media.githubusercontent.com/media/{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
              f"https://github.com/{_ZOO.replace('/main', '/raw/main')}/face_detection_yunet/face_detection_yunet_2023mar.onnx"]
SFACE_URLS = [f"https://media.githubusercontent.com/media/{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
              f"https://github.com/{_ZOO.replace('/main', '/raw/main')}/face_recognition_sface/face_recognition_sface_2021dec.onnx"]


class PersonDetector:
    """YOLOv8n person detector (GPU when torch sees CUDA) with an OpenCV HOG CPU fallback."""

    def __init__(self, conf: float = 0.45):
        self.conf = conf
        self.backend = "none"
        self.device = "cpu"
        self._yolo = None
        self._hog = None
        self._lock = threading.Lock()

    def load(self) -> str:
        try:
            import torch
            from ultralytics import YOLO

            weights = MODELS_DIR / "yolov8n.pt"
            self._yolo = YOLO(str(weights))  # downloads into models/ on first run
            if not weights.exists():  # older ultralytics saved to CWD
                for cand in (Path.cwd() / "yolov8n.pt",):
                    if cand.exists():
                        shutil.move(str(cand), str(weights))
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._yolo.predict(np.zeros((480, 640, 3), np.uint8), classes=[0], verbose=False,
                               device=self.device, imgsz=480)  # warm-up (loads CUDA kernels once)
            self.backend = "yolov8n"
            return f"YOLOv8n on {self.device}" + (f" ({torch.cuda.get_device_name(0)})" if self.device != "cpu" else "")
        except Exception as e:
            log.warning("YOLO unavailable (%s: %s) - falling back to OpenCV HOG on CPU", type(e).__name__, e)
            self._yolo = None
        try:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        except AttributeError:  # OpenCV 5.x removed HOG
            self._hog = None
            raise RuntimeError("No person detector available: install ultralytics (pip install -r requirements.txt)")
        self.backend, self.device = "hog", "cpu"
        return "OpenCV HOG on CPU (lower accuracy; install ultralytics for YOLO)"

    def detect(self, frame) -> list[tuple[int, int, int, int, float]]:
        with self._lock:
            if self._yolo is not None:
                res = self._yolo.predict(frame, classes=[0], conf=self.conf, verbose=False,
                                         device=self.device, imgsz=480)[0]
                out = []
                for b in res.boxes:
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                    out.append((x1, y1, x2, y2, float(b.conf[0])))
                return out
            if self._hog is not None:
                h, w = frame.shape[:2]
                scale = 400.0 / w if w > 400 else 1.0
                small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
                rects, weights = self._hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.07)
                return [(int(x / scale), int(y / scale), int((x + rw) / scale), int((y + rh) / scale), float(wt))
                        for (x, y, rw, rh), wt in zip(rects, np.ravel(weights)) if float(wt) >= 0.6]
        return []


class FaceEngine:
    """OpenCV YuNet (detection) + SFace (128-d embeddings). Pure ONNX, no dlib/CMake."""

    def __init__(self):
        self.ready = False
        self._det = None
        self._rec = None
        self._lock = threading.Lock()

    def load(self) -> None:
        det = download_file(YUNET_URLS, MODELS_DIR / "face_detection_yunet_2023mar.onnx")
        rec = download_file(SFACE_URLS, MODELS_DIR / "face_recognition_sface_2021dec.onnx")
        self._det = cv2.FaceDetectorYN.create(str(det), "", (320, 320), 0.8, 0.3, 200)
        self._rec = cv2.FaceRecognizerSF.create(str(rec), "")
        self.ready = True

    def detect(self, img) -> "np.ndarray":
        with self._lock:
            h, w = img.shape[:2]
            self._det.setInputSize((w, h))
            _, faces = self._det.detect(img)
        return faces if faces is not None else np.empty((0, 15), np.float32)

    def feature(self, img, face_row) -> "np.ndarray":
        with self._lock:
            aligned = self._rec.alignCrop(img, face_row)
            f = self._rec.feature(aligned).reshape(-1).astype(np.float32)
        return f / (np.linalg.norm(f) + 1e-9)

    @staticmethod
    def match(feat, profiles: dict, threshold: float) -> tuple[Optional[str], float]:
        best_name, best = None, -1.0
        for name, embs in profiles.items():
            s = float(np.max(embs @ feat)) if len(embs) else -1.0
            if s > best:
                best_name, best = name, s
        return (best_name if best >= threshold else None), best


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    id: int
    box: tuple
    first_seen: float
    last_seen: float
    hits: int = 1
    confirmed: bool = False
    confirmed_at: float = 0.0
    status: str = "pending"      # pending | recognized | unknown
    name: str = ""
    votes: list = field(default_factory=list)


class CameraWorker(QThread):
    """Owns the webcam. Detects/tracks people, confirms identities across frames, runs enrollment.

    Frames are processed in memory only. Pausing releases the device (camera LED off).
    """

    frame_ready = Signal(object)          # QImage preview (only when preview is on or enrolling)
    status = Signal(str, str)             # "active" | "off" | "error", detail
    count_changed = Signal(int)           # number of confirmed people in view
    person_entered = Signal(int)          # track id
    person_left = Signal(int)
    identity = Signal(int, str)           # track id, name ("" = unknown visitor)
    enroll_progress = Signal(int, int)    # have, need
    enroll_hint = Signal(str)
    enroll_done = Signal(str, object)     # name, embeddings (n,128)
    enroll_failed = Signal(str)

    def __init__(self, settings: Settings, detector: PersonDetector, faces: FaceEngine):
        super().__init__()
        self.s, self.detector, self.faces = settings, detector, faces
        self._enabled = False
        self._quit = False
        self._profiles: dict = {}
        self._enroll: Optional[dict] = None
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._cap = None
        self._cap_src = None
        self._is_file = False
        self._fail = 0
        self._count = 0
        self._last_face = 0.0
        self._last_status: tuple = ()

    # -- control (any thread) ---------------------------------------------------------
    def _emit_status(self, kind: str, detail: str = "") -> None:
        if (kind, detail) != self._last_status:  # don't spam identical errors while retrying
            self._last_status = (kind, detail)
            self.status.emit(kind, detail)

    def set_enabled(self, on: bool) -> None:
        self._enabled = bool(on)

    def set_profiles(self, profiles: dict) -> None:
        self._profiles = dict(profiles)

    def start_enroll(self, name: str, samples: int = 8, timeout: float = 45.0) -> None:
        self._enroll = {"name": name, "target": samples, "samples": [], "last": 0.0,
                        "deadline": time.monotonic() + timeout}
        self.enroll_progress.emit(0, samples)

    def cancel_enroll(self) -> None:
        self._enroll = None

    def stop(self) -> None:
        self._quit = True
        self.wait(4000)
        self._release()

    # -- thread body ------------------------------------------------------------------
    def run(self) -> None:
        next_proc = 0.0
        while not self._quit:
            if not self._enabled:
                if self._cap is not None:
                    self._release()
                    self._reset_tracks()
                    self._emit_status("off", "")
                self.msleep(120)
                continue
            src = str(self.s.camera_source)
            if self._cap is None or src != self._cap_src:
                if not self._open(src):
                    self.msleep(2500)
                    continue
            if not self._cap.grab():
                if self._is_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._fail += 1
                if self._fail > 30:
                    self._release()
                    self._emit_status("error", "Camera stopped delivering frames")
                    self.msleep(1500)
                else:
                    self.msleep(50)
                continue
            self._fail = 0
            if self._is_file:
                self.msleep(30)
            now = time.monotonic()
            if now < next_proc:
                continue  # skipped frame: grab() only, very cheap
            ok, frame = self._cap.retrieve()
            if not ok or frame is None:
                continue
            busy = self._enroll is not None or bool(self._tracks)
            next_proc = now + 1.0 / max(1, self.s.camera_fps_active if busy else self.s.camera_fps_idle)
            try:
                self._process(frame, now)
            except Exception:
                log.exception("Camera processing error")
                self.msleep(300)
        self._release()

    def _open(self, src: str) -> bool:
        self._release()
        is_idx = src.strip().isdigit()
        cap = None
        try:
            if is_idx:
                idx = int(src)
                if os.name == "nt":
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap.release()
                        cap = cv2.VideoCapture(idx)
                else:
                    cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(src)
        except Exception as e:
            log.warning("Camera open failed: %s", e)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            self._emit_status("error", f"Cannot open camera '{src}'")
            return False
        if is_idx:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap, self._cap_src, self._is_file = cap, src, not is_idx
        self._fail = 0
        self._emit_status("active", f"source {src}")
        return True

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
        self._cap_src = None

    def _reset_tracks(self) -> None:
        for t in list(self._tracks.values()):
            if t.confirmed:
                self.person_left.emit(t.id)
        self._tracks.clear()
        if self._count != 0:
            self._count = 0
            self.count_changed.emit(0)

    # -- per-frame --------------------------------------------------------------------
    def _process(self, frame, now: float) -> None:
        if self._enroll is not None:
            self._process_enroll(frame, now)
            return
        boxes = self.detector.detect(frame)
        self._update_tracks(boxes, now)
        self._recognize(frame, now)
        if self.s.preview:
            self._emit_preview(frame)

    def _update_tracks(self, boxes, now: float) -> None:
        tracks = list(self._tracks.values())
        pairs = sorted(((_iou(t.box, b[:4]), ti, di) for ti, t in enumerate(tracks) for di, b in enumerate(boxes)),
                       reverse=True)
        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if score < 0.2:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = tracks[ti]
            t.box, t.last_seen, t.hits = tuple(boxes[di][:4]), now, t.hits + 1
        for di, b in enumerate(boxes):
            if di not in used_d:
                t = Track(self._next_id, tuple(b[:4]), now, now)
                self._tracks[t.id] = t
                self._next_id += 1
        confirm = self.s.enter_confirm_frames
        for t in list(self._tracks.values()):
            if not t.confirmed and t.hits >= confirm:
                t.confirmed, t.confirmed_at = True, now
                self.person_entered.emit(t.id)
            gone = now - t.last_seen
            if gone > (self.s.leave_seconds if t.confirmed else 1.5):
                del self._tracks[t.id]
                if t.confirmed:
                    self.person_left.emit(t.id)
        n = sum(1 for t in self._tracks.values() if t.confirmed)
        if n != self._count:
            self._count = n
            self.count_changed.emit(n)

    def _recognize(self, frame, now: float) -> None:
        window = 25.0
        pending = [t for t in self._tracks.values()
                   if t.confirmed and t.status != "recognized" and now - t.confirmed_at < window]
        if not pending:
            return
        if not self._profiles or not self.faces.ready:
            for t in pending:  # nobody enrolled: everyone is simply a visitor
                if t.status == "pending":
                    t.status = "unknown"
                    self.identity.emit(t.id, "")
            return
        if now - self._last_face >= 0.25:
            self._last_face = now
            faces = self.faces.detect(frame)
            confirmed = [t for t in self._tracks.values() if t.confirmed]
            k = max(1, self.s.face_confirm_votes)
            for f in faces:
                fx, fy, fw, fh = [float(v) for v in f[:4]]
                if fw < self.s.face_min_size:
                    continue
                cx, cy = fx + fw / 2, fy + fh / 2
                inside = [t for t in confirmed if t.box[0] <= cx <= t.box[2] and t.box[1] <= cy <= t.box[3]]
                if not inside:
                    continue
                t = min(inside, key=lambda tr: (tr.box[2] - tr.box[0]) * (tr.box[3] - tr.box[1]))
                if t.status == "recognized":
                    continue
                name, _score = self.faces.match(self.faces.feature(frame, f), self._profiles, self.s.face_threshold)
                t.votes = (t.votes + [name])[-8:]
                last = t.votes[-k:]
                if len(last) == k and last[0] and all(v == last[0] for v in last):
                    t.status, t.name = "recognized", last[0]
                    self.identity.emit(t.id, t.name)
        for t in pending:  # couldn't confirm anyone in time -> unknown visitor (never guess)
            if t.status == "pending" and now - t.confirmed_at >= self.s.unknown_after_seconds:
                t.status = "unknown"
                self.identity.emit(t.id, "")

    def _emit_preview(self, frame, face_box=None, caption: str = "") -> None:
        vis = cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1])))
        sx = vis.shape[1] / frame.shape[1]
        for t in self._tracks.values():
            if not t.confirmed:
                continue
            x1, y1, x2, y2 = [int(v * sx) for v in t.box]
            col = (255, 211, 25) if t.status == "recognized" else (140, 160, 255) if t.status == "pending" else (120, 120, 120)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            label = t.name if t.status == "recognized" else ("visitor" if t.status == "unknown" else "person")
            cv2.putText(vis, label, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        if face_box is not None:
            x, y, w, h = [int(v * sx) for v in face_box]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 211, 25), 2)
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.frame_ready.emit(QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy())

    # -- consent-based enrollment -----------------------------------------------------
    def _process_enroll(self, frame, now: float) -> None:
        e = self._enroll
        faces = self.faces.detect(frame)
        box = None
        if len(faces) == 0:
            self.enroll_hint.emit("No face detected - look at the camera.")
        elif len(faces) > 1:
            self.enroll_hint.emit("More than one face in view - only the person enrolling should be visible.")
        else:
            f = faces[0]
            box = tuple(float(v) for v in f[:4])
            if f[-1] < 0.85 or f[2] < 90:
                self.enroll_hint.emit("Move closer / face the camera in good light.")
            elif now - e["last"] >= 0.7:
                feat = self.faces.feature(frame, f)
                if e["samples"] and float(np.mean([s @ feat for s in e["samples"]])) < 0.25:
                    self.enroll_hint.emit("Face changed during capture - restart the enrollment.")
                else:
                    e["samples"].append(feat)
                    e["last"] = now
                    self.enroll_progress.emit(len(e["samples"]), e["target"])
                    self.enroll_hint.emit("Good. Slowly turn your head a little between captures.")
        self._emit_preview(frame, box)
        if len(e["samples"]) >= e["target"]:
            self._enroll = None
            self.enroll_done.emit(e["name"], np.stack(e["samples"]))
        elif now > e["deadline"]:
            self._enroll = None
            self.enroll_failed.emit("Could not capture enough clear samples. Check lighting and try again.")


# =======================================================================================
# 5. GREETINGS
# =======================================================================================
GREETINGS = {
    "morning": ["Good morning, {name}.", "Good morning, {name}. JARVIS is online.",
                "Morning, {name}. Ready when you are.", "Good morning, {name}. What's on the agenda today?"],
    "afternoon": ["Good afternoon, {name}.", "Good afternoon, {name}. JARVIS is online.",
                  "Welcome back, {name}. What are we working on today?", "Afternoon, {name}. How can I help?"],
    "evening": ["Good evening, {name}.", "Good evening, {name}. JARVIS is online.",
                "Welcome back, {name}. Anything I can help with this evening?", "Evening, {name}. At your service."],
    "night": ["Working late, {name}? JARVIS is online.", "Good evening, {name}. I'm here if you need me.",
              "Welcome back, {name}. Burning the midnight oil?"],
    "any": ["Hello, {name}. JARVIS is online.", "Welcome back, {name}.", "Hello, {name}. How can I help?"],
}
NEUTRAL_GREETINGS = {
    "morning": ["Good morning. JARVIS is online if you need anything."],
    "afternoon": ["Good afternoon. JARVIS is online if you need anything."],
    "evening": ["Good evening. JARVIS is online if you need anything."],
    "night": ["Hello. JARVIS is online if you need anything."],
}


def daypart(now: Optional[dt.datetime] = None) -> str:
    h = (now or dt.datetime.now()).hour
    return "morning" if 5 <= h < 12 else "afternoon" if 12 <= h < 17 else "evening" if 17 <= h < 22 else "night"


class GreetingManager:
    """Chooses greetings. Only first names are used - never schedules, files or other private data."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._last_greet: dict[str, float] = {}
        self._last_unknown = float("-inf")
        self._last_text: dict[str, str] = {}

    def compose(self, name: str, now: Optional[float] = None, when: Optional[dt.datetime] = None) -> Optional[str]:
        now = time.time() if now is None else now
        dp = daypart(when)
        if name:
            if now - self._last_greet.get(name, float("-inf")) < self.s.greeting_cooldown_min * 60:
                return None
            pool = GREETINGS[dp] + GREETINGS["any"]
            key = name
        else:
            if self.s.unknown_greeting != "neutral":
                return None
            if now - self._last_unknown < self.s.unknown_cooldown_min * 60:
                return None
            pool, key = NEUTRAL_GREETINGS[dp], "*unknown*"
        choices = [g for g in pool if g != self._last_text.get(key)] or pool
        text = random.choice(choices)
        self._last_text[key] = text
        return text.format(name=name)

    def commit(self, name: str, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        if name:
            self._last_greet[name] = now
        else:
            self._last_unknown = now


# =======================================================================================
# 6. AUDIO IN - microphone, wake word, endpointing, speech-to-text
# =======================================================================================
class MicWorker(QThread):
    """Owns the microphone stream. Modes: off | wake | capture | hold.

    wake    -> run the local openWakeWord model ("hey jarvis") on 80 ms chunks
    capture -> record one utterance; stop automatically when the speaker goes quiet
    hold    -> stream stays open but audio is discarded (JARVIS is thinking / speaking)
    The stream is closed (OS mic indicator off) whenever the mic is disabled.
    """

    RATE = 16000
    CHUNK = 1280  # 80 ms, required granularity for openWakeWord

    wake = Signal()
    utterance = Signal(object)        # float32 mono 16 kHz
    capture_timeout = Signal()
    level = Signal(float)
    status = Signal(str, str)         # "active" | "off" | "error", detail

    def __init__(self, settings: Settings):
        super().__init__()
        self.s = settings
        self._enabled = False
        self._quit = False
        self._mode = "off"
        self._mode_args: dict = {}
        self._mode_dirty = True
        self._lock = threading.Lock()
        self._q: "queue.Queue[bytes]" = queue.Queue(maxsize=300)
        self._ow = None
        self.wake_ready = False
        self.wake_error = ""
        self._noise = 0.004
        self._cap: dict = {}
        self._last_wake = 0.0
        self._stream = None
        self._last_status: tuple = ()

    # -- control (any thread) ---------------------------------------------------------
    def _emit_status(self, kind: str, detail: str = "") -> None:
        if (kind, detail) != self._last_status:
            self._last_status = (kind, detail)
            self.status.emit(kind, detail)

    def load_wake_model(self) -> str:
        try:
            import openwakeword
            from openwakeword.model import Model

            try:
                self._ow = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
            except Exception:
                log.info("Downloading openWakeWord models (first run only)...")
                openwakeword.utils.download_models(["hey_jarvis"])
                self._ow = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
            self.wake_ready = True
            return "openWakeWord 'hey jarvis' loaded"
        except Exception as e:
            self.wake_ready, self.wake_error = False, f"{type(e).__name__}: {e}"
            log.warning("Wake word unavailable: %s", self.wake_error)
            return "wake word unavailable - use the LISTEN button"

    def set_enabled(self, on: bool) -> None:
        self._enabled = bool(on)

    def set_mode(self, mode: str, **kw) -> None:
        with self._lock:
            self._mode, self._mode_args, self._mode_dirty = mode, kw, True

    def stop(self) -> None:
        self._quit = True
        self.wait(4000)
        self._close()

    # -- thread body ------------------------------------------------------------------
    def run(self) -> None:
        while not self._quit:
            if not self._enabled:
                if self._stream is not None:
                    self._close()
                    self._emit_status("off", "")
                self.msleep(120)
                continue
            if self._stream is None and not self._open():
                self.msleep(2000)
                continue
            with self._lock:
                mode, args, dirty = self._mode, self._mode_args, self._mode_dirty
                self._mode_dirty = False
            if dirty:
                self._drain()
                if mode == "capture":
                    self._begin_capture(**args)
                elif mode == "wake" and self._ow is not None:
                    try:
                        self._ow.reset()
                    except Exception:
                        pass
            try:
                data = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            pcm = np.frombuffer(data, dtype=np.int16)
            f32 = pcm.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(f32 * f32)) + 1e-9)
            self.level.emit(min(1.0, (rms * 9.0) ** 0.7))
            if mode == "wake":
                self._track_noise(rms)
                self._wake_step(pcm)
            elif mode == "capture":
                self._capture_step(f32, rms)
        self._close()

    def _open(self) -> bool:
        try:
            import sounddevice as sd

            dev = self.s.mic_device or None
            if dev is not None and str(dev).isdigit():
                dev = int(dev)

            def cb(indata, frames, t, st):
                try:
                    self._q.put_nowait(bytes(indata))
                except queue.Full:
                    pass

            self._stream = sd.RawInputStream(samplerate=self.RATE, channels=1, dtype="int16",
                                             blocksize=self.CHUNK, device=dev, callback=cb)
            self._stream.start()
            self._drain()
            self._emit_status("active", "")
            return True
        except Exception as e:
            self._stream = None
            self._emit_status("error", f"{type(e).__name__}: {e}")
            return False

    def _close(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass

    def _drain(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    # -- wake word --------------------------------------------------------------------
    def _wake_step(self, pcm) -> None:
        if self._ow is None:
            return
        try:
            scores = self._ow.predict(pcm)
        except Exception:
            log.exception("wake model error")
            self._ow = None
            return
        if scores and max(scores.values()) >= self.s.wake_threshold:
            now = time.monotonic()
            if now - self._last_wake >= self.s.wake_cooldown:
                self._last_wake = now
                log.info("Wake word detected")
                self.wake.emit()

    # -- noise floor + endpointing ------------------------------------------------------
    def _track_noise(self, rms: float) -> None:
        if rms < max(self._noise * 2.5, 0.004):
            self._noise = 0.97 * self._noise + 0.03 * rms
        else:
            self._noise = min(self._noise * 1.001 + 1e-6, 0.05)  # slowly follow rising ambience

    def _begin_capture(self, timeout: float = 6.0, grace: float = 0.0, **_) -> None:
        self._cap = {"t0": time.monotonic(), "timeout": timeout, "grace": grace, "buf": [], "speech": False,
                     "voiced": 0, "last_voice": 0.0, "t_speech": 0.0, "pre": deque(maxlen=5)}

    def _capture_step(self, f32, rms: float) -> None:
        c = self._cap
        if not c:
            return
        now = time.monotonic()
        if now - c["t0"] < c["grace"]:
            return
        voiced = rms > max(self._noise * 3.0, 0.008)
        if not c["speech"]:
            c["pre"].append(f32)
            c["voiced"] = c["voiced"] + 1 if voiced else 0
            if c["voiced"] >= 2:
                c.update(speech=True, buf=list(c["pre"]), last_voice=now, t_speech=now)
            else:
                self._track_noise(rms)
                if now - c["t0"] - c["grace"] > c["timeout"]:
                    self._cap = {}
                    self.set_mode("hold")
                    self.capture_timeout.emit()
            return
        c["buf"].append(f32)
        if voiced:
            c["last_voice"] = now
        if now - c["last_voice"] >= self.s.silence_end_seconds or now - c["t_speech"] > 20.0:
            audio = np.concatenate(c["buf"])
            self._cap = {}
            if len(audio) < 0.35 * self.RATE:  # a cough or click, not speech - keep listening
                self._begin_capture(timeout=max(1.0, c["timeout"] - (now - c["t0"])), grace=0.0)
                return
            self.set_mode("hold")
            self.utterance.emit(audio)


def prepare_cuda_dlls() -> None:
    """Windows: make pip-installed NVIDIA runtime DLLs (cuBLAS/cuDNN) visible to CTranslate2 (faster-whisper).
    Importing torch first also works because it loads its bundled CUDA DLLs into the process."""
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    if os.name != "nt":
        return
    for base in map(Path, sys.path):
        nv = base / "nvidia"
        if nv.is_dir():
            for b in nv.glob("*/bin"):
                try:
                    os.add_dll_directory(str(b))
                    os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
                except OSError:
                    pass


class Transcriber:
    """faster-whisper wrapper. Loads once; uses the GPU when CTranslate2 can, else CPU int8."""

    HALLUCINATIONS = {"", "you", "thank you.", "thanks for watching.", "thanks for watching!", "bye."}

    def __init__(self, settings: Settings):
        self.s = settings
        self.model = None
        self.device = "cpu"
        self.model_name = ""
        self._lock = threading.Lock()

    def load(self) -> str:
        prepare_cuda_dlls()
        from faster_whisper import WhisperModel

        cuda = False
        try:
            import ctranslate2

            cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            pass
        if cuda:
            try:
                self.model = WhisperModel(self.s.whisper_model, device="cuda", compute_type="float16")
                self.device, self.model_name = "cuda", self.s.whisper_model
                self._warm()  # surfaces missing cuBLAS/cuDNN DLLs immediately
            except Exception as e:
                log.warning("Whisper GPU unavailable (%s: %s) - using CPU", type(e).__name__, e)
                self.model = None
        if self.model is None:
            self._load_cpu()
        return f"Whisper '{self.model_name}' on {self.device}"

    def _load_cpu(self) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(self.s.whisper_model_cpu, device="cpu", compute_type="int8", cpu_threads=4)
        self.device, self.model_name = "cpu", self.s.whisper_model_cpu

    def _warm(self) -> None:
        segs, _ = self.model.transcribe(np.zeros(16000, np.float32), language=self.s.whisper_language, beam_size=1)
        list(segs)

    def transcribe(self, audio: "np.ndarray") -> str:
        if self.model is None:
            self.load()  # first-run download may have failed earlier (e.g. offline); try again
        with self._lock:
            try:
                return self._run(audio)
            except RuntimeError as e:  # typical for broken CUDA runtime: retry on CPU once
                if self.device != "cuda":
                    raise
                log.warning("Whisper CUDA failure (%s) - reloading on CPU", e)
                self._load_cpu()
                return self._run(audio)

    def _run(self, audio) -> str:
        segments, _ = self.model.transcribe(
            audio, language=self.s.whisper_language, beam_size=1, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}, condition_on_previous_text=False, temperature=0.0)
        parts = []
        for seg in segments:
            if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
                continue
            parts.append(seg.text.strip())
        text = " ".join(p for p in parts if p).strip()
        return "" if text.lower() in self.HALLUCINATIONS else text


# =======================================================================================
# 7. AUDIO OUT - pluggable TTS providers + interruptible, pipelined speaker
# =======================================================================================
def clean_for_speech(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"https?://\S+", " link ", text)
    text = re.sub(r"[*_`#>~|]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


class TTSProvider:
    """Base class. `synth()` yields int16 mono numpy arrays at `sample_rate`.

    To add a voice engine: subclass this, implement available() and synth(), and register it in
    TTS_PROVIDERS. Nothing else in the app needs to change.
    """

    name = "base"
    sample_rate = 24000

    def __init__(self, settings: Settings):
        self.s = settings

    def available(self) -> tuple[bool, str]:
        return True, ""

    def synth(self, text: str) -> Iterator["np.ndarray"]:
        raise NotImplementedError


class EdgeTTS(TTSProvider):
    """Microsoft Edge neural voices via the `edge-tts` package: free, no key, natural prosody.
    Needs internet. Audio is MP3, decoded in-process with `miniaudio` (no ffmpeg needed)."""

    name = "edge"
    sample_rate = 24000

    def available(self):
        try:
            import edge_tts  # noqa: F401
            import miniaudio  # noqa: F401
            return True, ""
        except ImportError as e:
            return False, f"missing package {e.name}"

    def synth(self, text):
        import asyncio

        import edge_tts
        import miniaudio

        async def run() -> bytes:
            comm = edge_tts.Communicate(text, self.s.edge_voice, rate=self.s.edge_rate)
            buf = bytearray()
            async for ch in comm.stream():
                if ch["type"] == "audio":
                    buf += ch["data"]
            return bytes(buf)

        data = asyncio.run(run())
        if not data:
            raise RuntimeError("Edge TTS returned no audio (offline?)")
        dec = miniaudio.decode(data, output_format=miniaudio.SampleFormat.SIGNED16,
                               nchannels=1, sample_rate=self.sample_rate)
        yield np.frombuffer(dec.samples, dtype=np.int16).copy()


class ElevenLabsTTS(TTSProvider):
    """ElevenLabs streaming TTS (most natural; paid/free-tier quota). Key: ELEVENLABS_API_KEY."""

    name = "elevenlabs"
    sample_rate = 24000

    def available(self):
        return (True, "") if os.getenv("ELEVENLABS_API_KEY") else (False, "ELEVENLABS_API_KEY not set")

    def synth(self, text):
        import requests

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.s.elevenlabs_voice_id}/stream"
        r = requests.post(url, params={"output_format": "pcm_24000", "optimize_streaming_latency": 3},
                          headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"},
                          json={"text": text, "model_id": self.s.elevenlabs_model}, stream=True, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"ElevenLabs HTTP {r.status_code}: {r.text[:160]}")
        carry = b""
        for chunk in r.iter_content(chunk_size=4800):
            if not chunk:
                continue
            chunk = carry + chunk
            n = len(chunk) - (len(chunk) % 2)
            carry = chunk[n:]
            if n:
                yield np.frombuffer(chunk[:n], dtype=np.int16).copy()


class PiperTTS(TTSProvider):
    """Fully local neural TTS (Piper). Voice model downloads once into models/piper/."""

    name = "piper"
    sample_rate = 22050
    _voice = None

    def available(self):
        try:
            import piper  # noqa: F401
            return True, ""
        except ImportError:
            return False, "pip install piper-tts to enable"

    def _load(self):
        if PiperTTS._voice is not None:
            return PiperTTS._voice
        from piper import PiperVoice

        v = self.s.piper_voice                       # e.g. en_GB-alan-medium
        lang, rest = v.split("-", 1)                 # en_GB , alan-medium
        speaker, quality = rest.rsplit("-", 1)
        base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang.split('_')[0]}/{lang}/{speaker}/{quality}/{v}"
        onnx = download_file(base + ".onnx", MODELS_DIR / "piper" / f"{v}.onnx", min_bytes=1_000_000)
        download_file(base + ".onnx.json", MODELS_DIR / "piper" / f"{v}.onnx.json", min_bytes=100)
        PiperTTS._voice = PiperVoice.load(str(onnx))
        PiperTTS.sample_rate = PiperTTS._voice.config.sample_rate
        return PiperTTS._voice

    def synth(self, text):
        voice = self._load()
        for chunk in voice.synthesize(text):
            yield np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).copy()


TTS_PROVIDERS = {"edge": EdgeTTS, "elevenlabs": ElevenLabsTTS, "piper": PiperTTS}


class Speaker(QObject):
    """Sentence-pipelined speech output.

    One thread synthesizes the next sentence while another plays the current one, so the first
    words start as soon as the first sentence is ready. stop() halts audio within ~one block
    (<50 ms). A generation counter guarantees old audio can never overlap new audio.
    """

    started = Signal()
    finished = Signal()
    level = Signal(float)
    error = Signal(str)

    BLOCK = 1024

    def __init__(self, settings: Settings):
        super().__init__()
        self.s = settings
        self.volume = float(settings.volume)
        self._gen = 0
        self._open_gen: Optional[int] = None
        self._lock = threading.Lock()
        self._text_q: "queue.Queue" = queue.Queue()
        self._audio_q: "queue.Queue" = queue.Queue(maxsize=4)
        self._providers: dict[str, TTSProvider] = {}
        self._bad_until: dict[str, float] = {}
        self._warned: set[str] = set()
        self._quit = False
        self.last_provider = ""
        self._threads = [threading.Thread(target=self._synth_loop, name="tts-synth", daemon=True),
                         threading.Thread(target=self._play_loop, name="tts-play", daemon=True)]
        for t in self._threads:
            t.start()

    # -- public API (main thread) -------------------------------------------------------
    def say(self, text: str) -> None:
        text = clean_for_speech(text)
        if not text:
            return
        with self._lock:
            self._open_gen = self._gen
            gen = self._gen
        self._text_q.put((gen, text))

    def end(self) -> None:
        with self._lock:
            gen, is_open = self._gen, self._open_gen == self._gen
        if is_open:
            self._text_q.put((gen, None))
        else:
            self.finished.emit()  # nothing was queued

    def stop(self) -> None:
        """Silence immediately. Does not emit `finished` - the caller already knows."""
        with self._lock:
            self._gen += 1
            self._open_gen = None
        for q in (self._text_q, self._audio_q):
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

    def set_volume(self, v: float) -> None:
        self.volume = max(0.0, min(1.0, float(v)))

    @property
    def busy(self) -> bool:
        return self._open_gen is not None

    def shutdown(self) -> None:
        self._quit = True
        self.stop()
        for t in self._threads:
            t.join(timeout=2.0)

    def chime(self) -> None:
        """Short, quiet 'I'm listening' cue."""
        try:
            import sounddevice as sd

            sr = 24000
            t = np.linspace(0, 0.16, int(sr * 0.16), False)
            tone = (np.sin(2 * np.pi * 880 * t) + 0.6 * np.sin(2 * np.pi * 1320 * t)) * np.exp(-t * 18)
            sd.play((tone * 0.25 * self.volume * 32767).astype(np.int16), sr, device=self._device())
        except Exception:
            pass

    # -- internals ----------------------------------------------------------------------
    def _device(self):
        d = self.s.speaker_device or None
        return int(d) if d is not None and str(d).isdigit() else d

    def _order(self) -> list[str]:
        first = self.s.tts_provider
        return [first] + [p for p in ("edge", "piper") if p != first]

    def _provider(self, name: str) -> Optional[TTSProvider]:
        if name not in TTS_PROVIDERS:
            return None
        if name not in self._providers:
            self._providers[name] = TTS_PROVIDERS[name](self.s)
        return self._providers[name]

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self.error.emit(msg)

    def _put_audio(self, item) -> bool:
        while not self._quit:
            if item[0] != self._gen:
                return False
            try:
                self._audio_q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _synth_loop(self) -> None:
        while not self._quit:
            try:
                gen, text = self._text_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen != self._gen:
                continue
            if text is None:
                self._put_audio((gen, "end", None))
                continue
            try:
                self._synthesize(gen, text)
            except Exception:
                log.exception("synth loop error")

    def _synthesize(self, gen: int, text: str) -> None:
        for name in self._order():
            if time.monotonic() < self._bad_until.get(name, 0.0):
                continue
            prov = self._provider(name)
            if prov is None:
                continue
            ok, why = prov.available()
            if not ok:
                self._bad_until[name] = time.monotonic() + 300
                if name == self.s.tts_provider:
                    self._warn_once(name + why, f"Voice '{name}' unavailable ({why}); using a fallback voice.")
                continue
            got_audio = False
            try:
                for arr in prov.synth(text):
                    if gen != self._gen:
                        return
                    got_audio = True
                    if not self._put_audio((gen, "audio", (prov.sample_rate, arr))):
                        return
                if got_audio:
                    self.last_provider = name
                    return
            except Exception as e:
                log.warning("TTS provider %s failed: %s", name, e)
                self._bad_until[name] = time.monotonic() + 60
                self._warn_once(name + "fail", f"Voice '{name}' failed ({type(e).__name__}); trying a fallback.")
                if got_audio:
                    return  # don't repeat a half-spoken sentence
        self._warn_once("none", "No voice engine is working - showing text only.")

    def _play_loop(self) -> None:
        stream, stream_sr, stream_gen, started_gen = None, None, -1, -1
        try:
            import sounddevice as sd
        except Exception as e:  # PortAudio missing: keep the loop alive so state never gets stuck
            sd = None
            self._warn_once("sd", f"Audio output unavailable ({type(e).__name__}: {e}) - text only.")

        def close(abort: bool):
            nonlocal stream
            if stream is not None:
                try:
                    stream.abort() if abort else stream.stop()
                    stream.close()
                except Exception:
                    pass
            stream = None

        while not self._quit:
            if stream is not None and stream_gen != self._gen:
                close(abort=True)
            try:
                gen, kind, payload = self._audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen != self._gen:
                continue
            if kind == "end":
                close(abort=False)
                started_gen = -1  # next utterance in the same generation must announce itself again
                with self._lock:
                    done = self._open_gen == gen
                    if done:
                        self._open_gen = None
                if done:
                    self.finished.emit()
                continue
            sr, arr = payload
            try:
                if sd is None:
                    raise RuntimeError("sounddevice/PortAudio is not available")
                if stream is None or stream_sr != sr:
                    close(abort=False)
                    stream = sd.OutputStream(samplerate=sr, channels=1, dtype="int16", latency="low",
                                             device=self._device())
                    stream.start()
                    stream_sr, stream_gen = sr, gen
                if started_gen != gen:
                    started_gen = gen
                    self.started.emit()
                for i in range(0, len(arr), self.BLOCK):
                    if gen != self._gen:
                        break
                    blk = arr[i:i + self.BLOCK].astype(np.float32) * self.volume
                    out = np.clip(blk, -32768, 32767).astype(np.int16)
                    stream.write(out.reshape(-1, 1))
                    self.level.emit(min(1.0, float(np.sqrt(np.mean(blk * blk))) / 9000.0))
                if gen != self._gen:
                    close(abort=True)
            except Exception as e:
                log.warning("Audio output error: %s", e)
                close(abort=True)
                self._warn_once("out" + type(e).__name__, f"Audio output error: {e}")
                with self._lock:
                    if self._open_gen == gen:
                        self._open_gen = None
                self.finished.emit()


# =======================================================================================
# 8. AI - pluggable provider + conversation memory
# =======================================================================================
SYSTEM_PROMPT = """You are JARVIS, a personal AI assistant living on the user's desktop computer.

Personality: intelligent, professional, calm and conversational, with an occasional dry wit. You
sound like a capable human assistant, never like a customer-service chatbot. No filler such as
"Certainly!" or "As an AI". Don't repeat the question back.

Your replies are SPOKEN ALOUD, so: usually one to three short sentences; plain spoken prose; no
markdown, bullet lists, headings, emojis or URLs. Say numbers and units the way a person would say
them. Only give long answers when the user clearly asks for detail. Use the conversation context so
follow-up questions ("and tomorrow?", "why?") make sense.

You are honest about your limits: in this version you cannot control the computer, open apps, browse
the web, or access files, and you have no live data. If you don't know something, say so.

Current local date and time: {now}.
The person speaking: {user}."""


class SentenceSplitter:
    """Cuts a token stream into speakable sentences (short first chunk for low latency)."""

    _BOUND = re.compile(r"(?<=[.!?…])[\"')\]]*\s+|\n+")
    _ABBR = {"mr.", "mrs.", "ms.", "dr.", "vs.", "e.g.", "i.e.", "st.", "no.", "prof."}

    def __init__(self, first_min: int = 14, min_len: int = 30):
        self.buf, self.first_min, self.min_len, self.emitted = "", first_min, min_len, 0

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out: list[str] = []
        pos = 0
        for m in self._BOUND.finditer(self.buf):
            cand = self.buf[pos:m.end()].strip()
            if not cand or cand.split()[-1].lower() in self._ABBR:
                continue
            if len(cand) >= (self.first_min if self.emitted == 0 else self.min_len):
                out.append(cand)
                self.emitted += 1
                pos = m.end()
        self.buf = self.buf[pos:]
        return out

    def flush(self) -> Optional[str]:
        tail, self.buf = self.buf.strip(), ""
        return tail or None


class AIProvider:
    """Interface for chat engines. A Claude provider can be added later by subclassing this and
    registering it in AI_PROVIDERS; the rest of the app only calls the methods below."""

    name = "base"

    def configured(self) -> bool:
        raise NotImplementedError

    def setup_help(self) -> str:
        raise NotImplementedError

    def check(self) -> tuple[bool, str]:
        return True, ""

    def stream_chat(self, messages: list[dict], cancel: threading.Event) -> Iterator[str]:
        raise NotImplementedError


class OpenAIProvider(AIProvider):
    """OpenAI Chat Completions (also works with any OpenAI-compatible server via OPENAI_BASE_URL)."""

    name = "openai"

    def __init__(self, settings: Settings):
        self.s = settings
        self._client = None

    def configured(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def setup_help(self) -> str:
        return ("No AI key is configured. To enable conversation: create an API key at "
                "https://platform.openai.com/api-keys, then copy .env.example to .env in the JARVIS "
                "folder and set OPENAI_API_KEY=your_key. Restart JARVIS afterwards.")

    def _get(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                                  base_url=os.getenv("OPENAI_BASE_URL") or None, timeout=30.0, max_retries=1)
        return self._client

    def check(self):
        try:
            self._get().models.list()
            return True, ""
        except Exception as e:
            return False, friendly_ai_error(e)

    def stream_chat(self, messages, cancel):
        stream = self._get().chat.completions.create(model=self.s.openai_model, messages=messages, stream=True)
        try:
            for chunk in stream:
                if cancel.is_set():
                    break
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            try:
                stream.close()
            except Exception:
                pass


AI_PROVIDERS = {"openai": OpenAIProvider}


def friendly_ai_error(e: Exception) -> str:
    n = type(e).__name__
    if n == "AuthenticationError":
        return "the API key was rejected (check OPENAI_API_KEY in .env)"
    if n == "RateLimitError":
        return "the AI service reports a rate limit or exhausted quota"
    if n in ("APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout"):
        return "I can't reach the AI service - check the internet connection"
    if n == "NotFoundError":
        return "that model name isn't available on this account (see OPENAI_MODEL / settings)"
    if n == "ImportError":
        return "the 'openai' package is not installed"
    return f"{n}: {str(e)[:140]}"


class Conversation:
    """Rolling chat memory. Forgotten after 15 idle minutes."""

    MAX_MESSAGES = 16
    EXPIRE_SECONDS = 900

    def __init__(self):
        self.history: list[dict] = []
        self.last = 0.0

    def messages(self, user_text: str, user_name: str = "") -> list[dict]:
        if self.history and time.time() - self.last > self.EXPIRE_SECONDS:
            self.history.clear()
        now = dt.datetime.now().strftime("%A, %B %d %Y, %I:%M %p").replace(" 0", " ")
        who = f"{user_name} (recognized by the camera)" if user_name else "unknown / not identified"
        return ([{"role": "system", "content": SYSTEM_PROMPT.format(now=now, user=who)}]
                + self.history[-self.MAX_MESSAGES:] + [{"role": "user", "content": user_text}])

    def commit(self, user_text: str, reply: str) -> None:
        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
        self.history = self.history[-self.MAX_MESSAGES:]
        self.last = time.time()

    def clear(self) -> None:
        self.history.clear()


END_PHRASES = re.compile(
    r"^\s*(?:ok(?:ay)?[, ]*)?(?:that'?s all|that is all|thanks?(?: you)?(?:,? jarvis)?|thank you(?:,? jarvis)?|"
    r"goodbye|good bye|bye(?: jarvis)?|never ?mind|stop listening|go to sleep|that will be all|i'?m done|"
    r"no(?:,? thanks?)?|cancel)[.!, ]*$", re.I)


# =======================================================================================
# 9. CORE CONTROLLER
# =======================================================================================
class JarvisCore(QObject):
    """Owns every component and is the only place that changes the assistant state.

    Worker threads (camera, microphone, STT, AI, TTS) never touch the UI or the state machine:
    they emit Qt signals that are delivered to this object on the main thread.
    """

    # -> UI
    state_changed = Signal(str, str)         # state, detail
    transcript = Signal(str, str, bool)      # role, text, starts_new_message
    camera_status = Signal(str, str)
    mic_status = Signal(str, str)
    ai_status = Signal(str, str)             # ready | online | busy | nokey | error
    user_changed = Signal(str)
    people_changed = Signal(int)
    level = Signal(str, float)               # "mic" | "speaker", 0..1
    frame = Signal(object)
    running_changed = Signal(bool)
    users_changed = Signal()
    enroll_progress = Signal(int, int)
    enroll_hint = Signal(str)
    enroll_finished = Signal(bool, str)
    # worker threads -> main thread
    _init_msg = Signal(str)
    _init_done = Signal(bool, str)
    _stt_ok = Signal(int, str)
    _stt_err = Signal(int, str)
    _ai_delta = Signal(int, str)
    _ai_sentence = Signal(int, str)
    _ai_done = Signal(int, str)
    _ai_err = Signal(int, str)
    _ai_checked = Signal(bool, str)

    def __init__(self, settings: Settings):
        super().__init__()
        self.s = settings
        self.sm = StateMachine()
        self.sm.changed.connect(self._on_state)
        self.store = ProfileStore()
        self.detector = PersonDetector(settings.person_conf)
        self.faces = FaceEngine()
        self.stt = Transcriber(settings)
        self.greeter = GreetingManager(settings)
        self.conv = Conversation()
        self.ai: AIProvider = AI_PROVIDERS.get(settings.ai_provider, OpenAIProvider)(settings)
        self.speaker = Speaker(settings)
        self.camera = CameraWorker(settings, self.detector, self.faces)
        self.mic = MicWorker(settings)

        self.running = False
        self.camera_on = True
        self.mic_on = True
        self._models_ready = False
        self._enrolling = False
        self._people = 0
        self._present: dict[int, str] = {}
        self._profiles: dict = {}
        self._turn = 0
        self._conv_active = False
        self._empty = 0
        self._listen_timeout = settings.first_listen_seconds
        self._mic_mode = ""
        self._ai_cancel = threading.Event()
        self._ai_first = True
        self._ai_spoke = False
        self._user_text = ""

        sp = self.speaker
        sp.started.connect(self._on_speech_started)
        sp.finished.connect(self._on_speech_finished)
        sp.error.connect(lambda m: self.transcript.emit("system", m, True))
        sp.level.connect(lambda v: self.level.emit("speaker", v))
        cam = self.camera
        cam.frame_ready.connect(self.frame)
        cam.status.connect(self._on_camera_status)
        cam.count_changed.connect(self._on_count)
        cam.person_entered.connect(self._on_entered)
        cam.person_left.connect(self._on_left)
        cam.identity.connect(self._on_identity)
        cam.enroll_progress.connect(self.enroll_progress)
        cam.enroll_hint.connect(self.enroll_hint)
        cam.enroll_done.connect(self._on_enroll_done)
        cam.enroll_failed.connect(self._on_enroll_failed)
        mic = self.mic
        mic.status.connect(self.mic_status)
        mic.level.connect(lambda v: self.level.emit("mic", v))
        mic.wake.connect(self._on_wake)
        mic.utterance.connect(self._on_utterance)
        mic.capture_timeout.connect(self._on_capture_timeout)
        self._init_msg.connect(lambda m: self.transcript.emit("system", m, True))
        self._init_done.connect(self._on_init_done)
        self._stt_ok.connect(self._on_stt_ok)
        self._stt_err.connect(self._on_stt_err)
        self._ai_delta.connect(self._on_ai_delta)
        self._ai_sentence.connect(self._on_ai_sentence)
        self._ai_done.connect(self._on_ai_done)
        self._ai_err.connect(self._on_ai_err)
        self._ai_checked.connect(self._on_ai_checked)
        self.camera.start()
        self.mic.start()

    # ---- helpers ------------------------------------------------------------------------
    def say_system(self, text: str) -> None:
        self.transcript.emit("system", text, True)

    def _rest(self) -> State:
        return S.DETECTING_PERSON if self._people > 0 else S.IDLE

    def current_name(self) -> str:
        return next((n for n in self._present.values() if n), "")

    def ai_initial_status(self) -> tuple[str, str]:
        return ("ready", self.ai.name) if self.ai.configured() else ("nokey", "")

    def _apply_devices(self) -> None:
        self.camera.set_enabled((self.running and self.camera_on) or self._enrolling)
        self.mic.set_enabled(self.running and self.mic_on)

    def _apply_mic_mode(self, st: State) -> None:
        if st == S.LISTENING:
            self._mic_mode = "capture"
            self.mic.set_mode("capture", timeout=self._listen_timeout, grace=0.35)
            return
        mode = ("wake" if st in PASSIVE_STATES else "hold" if st in (S.GREETING, S.THINKING, S.SPEAKING) else "off")
        if mode != self._mic_mode:
            self._mic_mode = mode
            self.mic.set_mode(mode)

    def _on_state(self, new: str, old: str, detail: str) -> None:
        self.state_changed.emit(new, detail)
        if new != old:
            self._apply_mic_mode(State(new))

    # ---- start / pause / shutdown -------------------------------------------------------
    def start(self) -> None:
        if self.sm.state not in (S.OFFLINE, S.ERROR):
            return
        self.running = True
        self.sm.go(S.INITIALIZING, "loading models")
        self.running_changed.emit(True)
        if self._models_ready:
            self._finish_start()
        else:
            threading.Thread(target=self._load_models, name="init", daemon=True).start()

    def _load_models(self) -> None:
        def step(msg, fn):
            self._init_msg.emit(msg)
            try:
                r = fn()
                if r:
                    self._init_msg.emit("  " + str(r))
                return True
            except Exception as e:
                log.warning("%s failed: %s", msg, e)
                self._init_msg.emit(f"  ! {msg} failed: {type(e).__name__}: {e}")
                return False

        try:
            step("Loading person detector...", self.detector.load)
            step("Loading face recognition...", self.faces.load)
            step("Loading wake-word model...", self.mic.load_wake_model)
            step("Loading speech recognition (first run downloads the model)...", self.stt.load)
            self._profiles = self.store.load_all()
            if self.ai.configured():
                self._init_msg.emit("Checking AI connection...")
                ok, why = self.ai.check()
                self._ai_checked.emit(ok, why)
            else:
                self._ai_checked.emit(False, "nokey")
            self._init_done.emit(True, "")
        except Exception as e:  # unexpected
            log.exception("Initialization failed")
            self._init_done.emit(False, f"{type(e).__name__}: {e}")

    def _on_init_done(self, ok: bool, err: str) -> None:
        if not ok:
            self.say_system(f"Initialization failed: {err}")
            if self.sm.state == S.INITIALIZING:
                self.sm.go(S.ERROR, err)
            return
        self._models_ready = True
        self.camera.set_profiles(self._profiles)
        self.users_changed.emit()
        if self.running and self.sm.state == S.INITIALIZING:
            self._finish_start()

    def _finish_start(self) -> None:
        self.sm.go(S.IDLE)
        self._apply_devices()
        self._apply_mic_mode(S.IDLE)
        self.say_system("JARVIS online. " + ("Say 'Hey JARVIS' or press LISTEN." if self.mic.wake_ready
                                             else "Wake word unavailable - press LISTEN or type below."))

    def pause(self) -> None:
        self.running = False
        self._turn += 1
        self._ai_cancel.set()
        self.speaker.stop()
        self._conv_active = False
        self._present.clear()
        self.user_changed.emit("")
        self.sm.go(S.OFFLINE, "paused")
        self._apply_devices()
        self.running_changed.emit(False)

    def shutdown(self) -> None:
        log.info("Shutting down")
        self.running = False
        self._enrolling = False
        self._ai_cancel.set()
        self.speaker.shutdown()
        self.camera.set_enabled(False)
        self.mic.set_enabled(False)
        self.camera.stop()
        self.mic.stop()

    # ---- user toggles -------------------------------------------------------------------
    def set_camera(self, on: bool) -> None:
        self.camera_on = on
        self._apply_devices()

    def set_mic(self, on: bool) -> None:
        self.mic_on = on
        self._apply_devices()
        if not on and self.sm.state == S.LISTENING:
            self._end_conversation()

    def set_volume(self, v: float) -> None:
        self.speaker.set_volume(v)
        self.s.update(volume=v)

    def settings_changed(self) -> None:
        self.detector.conf = self.s.person_conf
        if hasattr(self.ai, "_client"):
            self.ai._client = None  # pick up a newly saved API key
        self.ai_status.emit(*self.ai_initial_status())
        if self.ai.name != self.s.ai_provider and self.s.ai_provider in AI_PROVIDERS:
            self.ai = AI_PROVIDERS[self.s.ai_provider](self.s)
            self.ai_status.emit(*self.ai_initial_status())

    # ---- camera events ------------------------------------------------------------------
    def _on_camera_status(self, kind: str, detail: str) -> None:
        self.camera_status.emit(kind, detail)
        if kind == "error":
            self.say_system(f"Camera: {detail}")

    def _on_count(self, n: int) -> None:
        self._people = n
        self.people_changed.emit(n)
        if n == 0:
            self._present.clear()
            self._update_user()
        st = self.sm.state
        if st in PASSIVE_STATES:
            if n == 0:
                self.sm.go(S.IDLE)
            elif st == S.IDLE:
                self.sm.go(S.DETECTING_PERSON, f"{n} in view")

    def _on_entered(self, tid: int) -> None:
        log.info("Person entered (track %d)", tid)
        if self.sm.state in (S.IDLE, S.DETECTING_PERSON) and self._profiles and self.faces.ready:
            self.sm.go(S.RECOGNIZING_USER, "matching face")

    def _on_left(self, tid: int) -> None:
        log.info("Person left (track %d)", tid)
        self._present.pop(tid, None)
        self._update_user()

    def _on_identity(self, tid: int, name: str) -> None:
        self._present[tid] = name
        self._update_user()
        if self.sm.state == S.RECOGNIZING_USER:
            self.sm.go(self._rest())
        self._maybe_greet(name)

    def _update_user(self) -> None:
        names = [n for n in self._present.values() if n]
        if names:
            self.user_changed.emit(names[0] + (f" (+{len(names) - 1})" if len(names) > 1 else ""))
        elif self._present:
            self.user_changed.emit("Unknown visitor")
        else:
            self.user_changed.emit("")

    def _maybe_greet(self, name: str) -> None:
        if not self.running or self.sm.state not in PASSIVE_STATES:
            return  # never talk over a conversation or another greeting
        text = self.greeter.compose(name)
        if not text:
            return
        self.greeter.commit(name)
        self.sm.go(S.GREETING, name or "visitor")
        self.transcript.emit("assistant", text, True)
        self.speaker.say(text)
        self.speaker.end()

    # ---- wake / listening ---------------------------------------------------------------
    def _on_wake(self) -> None:
        if self.running and self.sm.state in PASSIVE_STATES:
            self.speaker.chime()
            self._empty = 0
            self._begin_listening(first=True)

    def listen_now(self) -> None:
        """Push-to-talk equivalent of the wake word."""
        if not self.running:
            self.say_system("Start the assistant first.")
        elif not self.mic_on:
            self.say_system("The microphone is switched off.")
        elif self.sm.state in (S.THINKING, S.SPEAKING, S.GREETING):
            self.stop_speaking()
            self._on_wake()
        else:
            self._on_wake()

    def _begin_listening(self, first: bool) -> None:
        if not (self.running and self.mic_on):
            self._end_conversation()
            return
        self._listen_timeout = self.s.first_listen_seconds if first else self.s.followup_seconds
        self._conv_active = True
        if self.sm.state == S.LISTENING:
            self._apply_mic_mode(S.LISTENING)
        else:
            self.sm.go(S.LISTENING, "follow-up" if not first else "")

    def _end_conversation(self) -> None:
        self._conv_active = False
        if self.sm.state not in (S.OFFLINE, S.INITIALIZING, S.ERROR):
            self.sm.go(self._rest())

    def _on_capture_timeout(self) -> None:
        if self.sm.state == S.LISTENING:
            self._end_conversation()

    def stop_speaking(self) -> None:
        self._turn += 1
        self._ai_cancel.set()
        self.speaker.stop()
        self._conv_active = False
        if self.sm.state in (S.SPEAKING, S.THINKING, S.GREETING, S.LISTENING):
            self.sm.go(self._rest())

    # ---- speech to text -----------------------------------------------------------------
    def _on_utterance(self, audio) -> None:
        if self.sm.state != S.LISTENING:
            return
        self._turn += 1
        gen = self._turn
        self.sm.go(S.THINKING, "transcribing")

        def work():
            try:
                self._stt_ok.emit(gen, self.stt.transcribe(audio))
            except Exception as e:
                log.exception("Transcription failed")
                self._stt_err.emit(gen, f"{type(e).__name__}: {e}")

        threading.Thread(target=work, name="stt", daemon=True).start()

    def _on_stt_err(self, gen: int, msg: str) -> None:
        if gen != self._turn:
            return
        self.say_system(f"Speech recognition error: {msg}. You can still type to me.")
        self._end_conversation()

    def _on_stt_ok(self, gen: int, text: str) -> None:
        if gen != self._turn or self.sm.state != S.THINKING:
            return
        if not text:
            self._empty += 1
            if self._empty < 2:
                self._begin_listening(first=True)
            else:
                self._end_conversation()
            return
        self._empty = 0
        self.transcript.emit("user", text, True)
        if END_PHRASES.match(text):
            self._conv_active = False
            self.speaker.say(random.choice(["Of course.", "Standing by.", "Just say the word."]))
            self.speaker.end()
            return
        self._ask_ai(text)

    def submit_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if not self.running:
            self.say_system("Start the assistant first.")
            return
        if self.sm.state in (S.THINKING, S.SPEAKING, S.GREETING):
            self.stop_speaking()
        self._turn += 1
        self._conv_active = True
        if not self.sm.go(S.THINKING, "typed"):
            return
        self.transcript.emit("user", text, True)
        self._ask_ai(text)

    # ---- AI -----------------------------------------------------------------------------
    def _ask_ai(self, text: str) -> None:
        gen = self._turn
        if not self.ai.configured():
            self.say_system(self.ai.setup_help())
            self.ai_status.emit("nokey", "")
            self._canned("I don't have an AI provider configured yet. I've put the setup steps in the transcript.")
            return
        self._user_text = text
        self._ai_cancel = threading.Event()
        cancel = self._ai_cancel
        self._ai_first, self._ai_spoke = True, False
        msgs = self.conv.messages(text, self.current_name())
        self.ai_status.emit("busy", "")

        def work():
            splitter, parts = SentenceSplitter(), []
            try:
                for delta in self.ai.stream_chat(msgs, cancel):
                    if cancel.is_set():
                        return
                    parts.append(delta)
                    self._ai_delta.emit(gen, delta)
                    for sent in splitter.feed(delta):
                        self._ai_sentence.emit(gen, sent)
                if cancel.is_set():
                    return
                tail = splitter.flush()
                if tail:
                    self._ai_sentence.emit(gen, tail)
                self._ai_done.emit(gen, "".join(parts))
            except Exception as e:
                log.warning("AI request failed: %s: %s", type(e).__name__, e)
                self._ai_err.emit(gen, friendly_ai_error(e))

        threading.Thread(target=work, name="ai", daemon=True).start()

    def _canned(self, text: str) -> None:
        self._conv_active = False
        self.speaker.say(text)
        self.speaker.end()

    def _on_ai_delta(self, gen: int, d: str) -> None:
        if gen == self._turn:
            self.transcript.emit("assistant", d, self._ai_first)
            self._ai_first = False

    def _on_ai_sentence(self, gen: int, s: str) -> None:
        if gen == self._turn and self.sm.state in (S.THINKING, S.SPEAKING):
            self._ai_spoke = True
            self.speaker.say(s)

    def _on_ai_done(self, gen: int, full: str) -> None:
        if gen != self._turn:
            return
        self.ai_status.emit("online", "")
        if not full.strip():
            self._canned("I'm afraid I have nothing to add to that.")
            return
        self.conv.commit(self._user_text, full)
        self.speaker.end()

    def _on_ai_err(self, gen: int, msg: str) -> None:
        if gen != self._turn:
            return
        self.ai_status.emit("error", msg)
        self.say_system(f"AI error: {msg}")
        if self._ai_spoke:
            self._conv_active = False
            self.speaker.end()
        else:
            self._canned("I couldn't get an answer from my AI service. The details are in the transcript.")

    def _on_ai_checked(self, ok: bool, why: str) -> None:
        if why == "nokey":
            self.ai_status.emit("nokey", "")
            self.say_system(self.ai.setup_help())
        elif ok:
            self.ai_status.emit("online", "")
        else:
            self.ai_status.emit("error", why)
            self.say_system(f"AI check failed: {why}")

    # ---- speaker events -----------------------------------------------------------------
    def _on_speech_started(self) -> None:
        if self.sm.state == S.THINKING:
            self.sm.go(S.SPEAKING)

    def _on_speech_finished(self) -> None:
        st = self.sm.state
        if st == S.GREETING:
            self.sm.go(self._rest())
        elif st in (S.SPEAKING, S.THINKING):
            if self._conv_active:
                self._begin_listening(first=False)
            else:
                self._end_conversation()

    # ---- users / enrollment -------------------------------------------------------------
    def list_users(self) -> list[dict]:
        return self.store.list_users()

    def remove_user(self, name: str) -> bool:
        ok = self.store.delete(name)
        self._profiles = self.store.load_all()
        self.camera.set_profiles(self._profiles)
        self.users_changed.emit()
        if ok:
            self.say_system(f"Removed enrolled user '{name}'.")
        return ok

    def start_enrollment(self, name: str) -> Optional[str]:
        """Returns an error string, or None when capture has begun. Caller must have obtained consent."""
        name = name.strip()
        if not self._models_ready or not self.faces.ready:
            return "Start the assistant first (face models must be loaded)."
        if not (1 <= len(name) <= 40):
            return "Please enter a name (1-40 characters)."
        self._enrolling = True
        self._apply_devices()
        self.camera.start_enroll(name)
        return None

    def cancel_enrollment(self) -> None:
        self.camera.cancel_enroll()
        self._enrolling = False
        self._apply_devices()

    def _on_enroll_done(self, name: str, embeddings) -> None:
        try:
            self.store.save(name, embeddings)
            self._profiles = self.store.load_all()
            self.camera.set_profiles(self._profiles)
            self.say_system(f"Enrolled '{name}'. Face template stored locally (encrypted).")
            self.enroll_finished.emit(True, f"{name} enrolled.")
        except Exception as e:
            log.exception("Saving profile failed")
            self.enroll_finished.emit(False, f"Could not save profile: {e}")
        self._enrolling = False
        self._apply_devices()
        self.users_changed.emit()

    def _on_enroll_failed(self, msg: str) -> None:
        self._enrolling = False
        self._apply_devices()
        self.enroll_finished.emit(False, msg)


# =======================================================================================
# 10. PYSIDE6 DASHBOARD
# =======================================================================================
C_BG, C_PANEL, C_EDGE = "#04060c", "#0a1124", "#12305a"
C_ACCENT, C_TEXT, C_DIM = "#19d3ff", "#cfe9ff", "#5d7a99"
C_OK, C_WARN, C_ERR = "#2dffb0", "#ffb02e", "#ff4d5e"

STATE_STYLE = {  # colour, animation speed
    S.OFFLINE: ("#3b4b63", 0.15), S.INITIALIZING: ("#4aa8ff", 1.6), S.IDLE: ("#19d3ff", 0.35),
    S.DETECTING_PERSON: ("#19d3ff", 0.6), S.RECOGNIZING_USER: ("#27e6c8", 1.4), S.GREETING: ("#7fe8ff", 0.8),
    S.LISTENING: ("#2d8cff", 1.0), S.THINKING: ("#9a7bff", 2.4), S.SPEAKING: ("#aef0ff", 0.9),
    S.ERROR: ("#ff4d5e", 0.5),
}

QSS = f"""
QWidget {{ background: {C_BG}; color: {C_TEXT}; font-family: 'Bahnschrift','Segoe UI','Inter','DejaVu Sans'; font-size: 13px; }}
QFrame#panel {{ background: {C_PANEL}; border: 1px solid {C_EDGE}; border-radius: 10px; }}
QLabel {{ background: transparent; }}
QLabel#h {{ color: {C_DIM}; font-size: 11px; letter-spacing: 3px; }}
QPushButton {{ background: #0b1830; border: 1px solid #1b4a8a; border-radius: 6px; padding: 8px 14px; color: #bfe6ff; letter-spacing: 1px; }}
QPushButton:hover {{ border-color: {C_ACCENT}; background: #0d2342; }}
QPushButton:checked {{ background: #0f3a6e; border-color: {C_ACCENT}; color: white; }}
QPushButton:disabled {{ color: #3d5575; border-color: {C_EDGE}; }}
QPushButton#primary {{ background: #0c4a8a; border-color: {C_ACCENT}; color: white; font-weight: bold; }}
QPushButton#danger {{ border-color: #8a2b3a; color: #ffb3bd; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: #08101f; border: 1px solid #1b4a8a; border-radius: 5px; padding: 6px; color: {C_TEXT}; }}
QComboBox QAbstractItemView {{ background: #08101f; selection-background-color: #0f3a6e; }}
QTextBrowser, QListWidget {{ background: transparent; border: none; }}
QListWidget::item {{ padding: 6px; }} QListWidget::item:selected {{ background: #0f3a6e; }}
QSlider::groove:horizontal {{ height: 4px; background: {C_EDGE}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {C_ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {C_ACCENT}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QProgressBar {{ border: 1px solid #1b4a8a; border-radius: 5px; text-align: center; background: #08101f; }}
QProgressBar::chunk {{ background: {C_ACCENT}; border-radius: 4px; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: #1b4a8a; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QCheckBox {{ spacing: 8px; }}
QToolTip {{ background: #08101f; color: {C_TEXT}; border: 1px solid {C_ACCENT}; }}
"""


def _qc(hex_or_color, alpha: Optional[int] = None) -> QtGui.QColor:
    c = QtGui.QColor(hex_or_color)
    if alpha is not None:
        c.setAlpha(max(0, min(255, int(alpha))))
    return c


def _lerp_color(a: QtGui.QColor, b: QtGui.QColor, k: float) -> QtGui.QColor:
    return QtGui.QColor(int(a.red() + (b.red() - a.red()) * k), int(a.green() + (b.green() - a.green()) * k),
                        int(a.blue() + (b.blue() - a.blue()) * k))


class ReactorWidget(QtWidgets.QWidget):
    """Animated arc-reactor style centrepiece; colour and speed follow the assistant state."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 320)
        self._t = 0.0
        self._speed = 0.35
        self._col = _qc(STATE_STYLE[S.OFFLINE][0])
        self._target = QtGui.QColor(self._col)
        self._lvl = 0.0
        self._lvl_in = 0.0
        self._label = "OFFLINE"
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def set_state(self, st: State, label: str) -> None:
        col, spd = STATE_STYLE[st]
        self._target, self._speed, self._label = _qc(col), spd, label

    def set_level(self, v: float) -> None:
        self._lvl_in = max(self._lvl_in, v)

    def _tick(self) -> None:
        if not self.isVisible() or self.window().isMinimized():
            return
        self._t += 0.033 * self._speed
        self._col = _lerp_color(self._col, self._target, 0.12)
        self._lvl += (self._lvl_in - self._lvl) * 0.35
        self._lvl_in *= 0.82
        self.update()

    def paintEvent(self, _e) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        c = QPointF(w / 2, h / 2)
        R = min(w, h) / 2 - 14
        col, t, lvl = self._col, self._t, self._lvl
        breathe = 0.5 + 0.5 * math.sin(t * 2.2)

        g = QtGui.QRadialGradient(c, R * 1.15)
        g.setColorAt(0.0, _qc(col, 55 + 90 * lvl + 20 * breathe))
        g.setColorAt(0.55, _qc(col, 16))
        g.setColorAt(1.0, _qc(col, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawEllipse(c, R * 1.15, R * 1.15)

        # tick ring
        p.save()
        p.translate(c)
        p.rotate(t * 18)
        for i in range(90):
            a = i * 4
            long = i % 6 == 0
            p.setPen(QtGui.QPen(_qc(col, 190 if long else 90), 2 if long else 1))
            p.save()
            p.rotate(a)
            p.drawLine(QPointF(R, 0), QPointF(R - (12 if long else 6), 0))
            p.restore()
        p.restore()

        def arc(radius, start, span, width, alpha):
            pen = QtGui.QPen(_qc(col, alpha), width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawArc(QRectF(c.x() - radius, c.y() - radius, 2 * radius, 2 * radius), int(start * 16), int(span * 16))

        for k, (rad, sp, n, span, wd) in enumerate([(0.84, -40, 3, 70, 4), (0.72, 65, 2, 120, 3), (0.60, -110, 4, 50, 5)]):
            for j in range(n):
                arc(R * rad, t * sp + j * 360 / n, span * (0.9 + 0.1 * breathe), wd, 200 - k * 30)
        p.setPen(QtGui.QPen(_qc(col, 70), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, R * 0.92, R * 0.92)
        p.drawEllipse(c, R * 0.50, R * 0.50)

        # pulsing core
        cr = R * (0.30 + 0.10 * lvl + 0.015 * breathe)
        cg = QtGui.QRadialGradient(c, cr * 1.6)
        cg.setColorAt(0.0, _qc("#ffffff", 235))
        cg.setColorAt(0.35, _qc(col, 230))
        cg.setColorAt(1.0, _qc(col, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(cg)
        p.drawEllipse(c, cr * 1.6, cr * 1.6)
        p.setPen(QtGui.QPen(_qc(col, 255), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, cr, cr)

        f = QtGui.QFont(self.font())
        f.setPointSizeF(max(8.0, R / 24))
        f.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, 3)
        p.setFont(f)
        p.setPen(_qc(C_TEXT, 230))
        p.drawText(QRectF(c.x() - R, c.y() + R * 0.34, 2 * R, R * 0.2), Qt.AlignmentFlag.AlignCenter, self._label)


class WaveformWidget(QtWidgets.QWidget):
    N = 72

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(86)
        self._vals: deque = deque([0.0] * self.N, maxlen=self.N)
        self._cur = 0.0
        self._col = _qc(C_ACCENT)
        self._phase = 0.0
        t = QTimer(self)
        t.timeout.connect(self._tick)
        t.start(40)

    def set_color(self, col: str) -> None:
        self._col = _qc(col)

    def push(self, v: float) -> None:
        self._cur = max(self._cur, v)

    def _tick(self) -> None:
        if not self.isVisible():
            return
        self._phase += 0.12
        self._vals.append(self._cur)
        self._cur *= 0.78
        self.update()

    def paintEvent(self, _e) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        bw = w / self.N
        mid = h / 2
        for i, v in enumerate(self._vals):
            idle = 0.04 + 0.02 * math.sin(self._phase + i * 0.35)
            amp = max(v, idle)
            bh = amp * (h - 8)
            x = i * bw
            fade = 0.35 + 0.65 * (i / self.N)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_qc(self._col, 70 * fade))
            p.drawRoundedRect(QRectF(x + 0.5, mid - bh / 2 - 2, bw - 2, bh + 4), 2, 2)
            p.setBrush(_qc(self._col, 255 * fade))
            p.drawRoundedRect(QRectF(x + 1, mid - bh / 2, bw - 3, bh), 2, 2)


class Led(QtWidgets.QWidget):
    COLORS = {"ok": C_OK, "on": C_ACCENT, "warn": C_WARN, "err": C_ERR, "off": "#2a3a52"}

    def __init__(self):
        super().__init__()
        self.setFixedSize(18, 18)
        self._kind, self._pulse, self._ph = "off", False, 0.0

    def set(self, kind: str, pulse: bool = False) -> None:
        self._kind, self._pulse = kind, pulse
        self.update()

    def tick(self) -> None:
        if self._pulse:
            self._ph += 0.25
            self.update()

    def paintEvent(self, _e) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        col = _qc(self.COLORS[self._kind])
        k = 0.65 + 0.35 * math.sin(self._ph) if self._pulse else 1.0
        g = QtGui.QRadialGradient(QPointF(9, 9), 9)
        g.setColorAt(0, _qc(col, 200 * k))
        g.setColorAt(1, _qc(col, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawEllipse(QPointF(9, 9), 9, 9)
        p.setBrush(_qc(col, 255 if self._kind != "off" else 160))
        p.drawEllipse(QPointF(9, 9), 4, 4)


class StatusRow(QtWidgets.QWidget):
    def __init__(self, title: str):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        self.led = Led()
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(0)
        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("h")
        self.value = QtWidgets.QLabel("-")
        self.value.setStyleSheet("font-size: 15px;")
        col.addWidget(self.title)
        col.addWidget(self.value)
        lay.addWidget(self.led)
        lay.addLayout(col, 1)

    def set(self, text: str, kind: str, pulse: bool = False, tip: str = "") -> None:
        self.value.setText(text)
        self.led.set(kind, pulse)
        self.setToolTip(tip)


class Chip(QtWidgets.QLabel):
    def __init__(self, text: str, color: str):
        super().__init__(text)
        self._c = color
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(120)
        self.set_active(False)

    def set_active(self, on: bool) -> None:
        if on:
            self.setStyleSheet(f"border:1px solid {self._c}; border-radius:14px; padding:6px 14px; color:white;"
                               f"background: rgba(25,150,255,40); letter-spacing:3px; font-size:11px;")
        else:
            self.setStyleSheet(f"border:1px solid {C_EDGE}; border-radius:14px; padding:6px 14px; color:{C_DIM};"
                               f"background: transparent; letter-spacing:3px; font-size:11px;")


def panel() -> QtWidgets.QFrame:
    f = QtWidgets.QFrame()
    f.setObjectName("panel")
    return f


def heading(text: str) -> QtWidgets.QLabel:
    l = QtWidgets.QLabel(text)
    l.setObjectName("h")
    return l


def list_audio_devices() -> tuple[list, list]:
    ins, outs = [], []
    try:
        import sounddevice as sd

        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                ins.append((i, d["name"]))
            if d["max_output_channels"] > 0:
                outs.append((i, d["name"]))
    except Exception:
        pass
    return ins, outs


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, settings: Settings, core: JarvisCore, parent=None):
        super().__init__(parent)
        self.setWindowTitle("JARVIS - Settings")
        self.s, self.core = settings, core
        self.setMinimumWidth(520)
        form = QtWidgets.QFormLayout()
        s = settings

        def dspin(v, lo, hi, step=0.05, dec=2):
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setDecimals(dec)
            w.setValue(v)
            return w

        def ispin(v, lo, hi):
            w = QtWidgets.QSpinBox()
            w.setRange(lo, hi)
            w.setValue(v)
            return w

        ins, outs = list_audio_devices()
        self.mic_dev = QtWidgets.QComboBox()
        self.mic_dev.addItem("System default", "")
        for i, n in ins:
            self.mic_dev.addItem(f"{i}: {n}", str(i))
        self.mic_dev.setCurrentIndex(max(0, self.mic_dev.findData(s.mic_device)))
        self.spk_dev = QtWidgets.QComboBox()
        self.spk_dev.addItem("System default", "")
        for i, n in outs:
            self.spk_dev.addItem(f"{i}: {n}", str(i))
        self.spk_dev.setCurrentIndex(max(0, self.spk_dev.findData(s.speaker_device)))
        self.cam_src = QtWidgets.QLineEdit(str(s.camera_source))
        self.cam_src.setToolTip("Webcam index (0, 1, ...) or a video file path for testing")
        self.preview = QtWidgets.QCheckBox("Show camera preview on the dashboard")
        self.preview.setChecked(s.preview)
        self.person_conf = dspin(s.person_conf, 0.2, 0.9)
        self.cooldown = ispin(s.greeting_cooldown_min, 0, 1440)
        self.unknown = QtWidgets.QComboBox()
        self.unknown.addItem("Stay silent", "none")
        self.unknown.addItem("Neutral greeting", "neutral")
        self.unknown.setCurrentIndex(0 if s.unknown_greeting == "none" else 1)
        self.unknown_cd = ispin(s.unknown_cooldown_min, 0, 1440)
        self.wake = dspin(s.wake_threshold, 0.1, 0.95)
        self.follow = dspin(s.followup_seconds, 2, 30, 0.5, 1)
        self.silence = dspin(s.silence_end_seconds, 0.4, 2.5, 0.1, 1)
        self.tts = QtWidgets.QComboBox()
        for k in TTS_PROVIDERS:
            self.tts.addItem(k)
        self.tts.setCurrentText(s.tts_provider)
        self.edge_voice = QtWidgets.QLineEdit(s.edge_voice)
        self.eleven_voice = QtWidgets.QLineEdit(s.elevenlabs_voice_id)
        self.ai = QtWidgets.QComboBox()
        for k in AI_PROVIDERS:
            self.ai.addItem(k)
        self.ai.setCurrentText(s.ai_provider)
        self.model = QtWidgets.QLineEdit(s.openai_model)
        self.api_key = QtWidgets.QLineEdit()
        self.api_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("saved" if os.getenv("OPENAI_API_KEY") else "paste key (saved to .env)")
        self.whisper = QtWidgets.QLineEdit(s.whisper_model)
        for label, w in [("Camera source", self.cam_src), ("", self.preview), ("Person detection confidence", self.person_conf),
                         ("Greeting cooldown (minutes)", self.cooldown), ("Unknown visitors", self.unknown),
                         ("Unknown-visitor cooldown (min)", self.unknown_cd), ("Microphone", self.mic_dev),
                         ("Speaker", self.spk_dev), ("Wake-word threshold", self.wake),
                         ("Follow-up window (s)", self.follow), ("End-of-speech silence (s)", self.silence),
                         ("Voice provider", self.tts), ("Edge voice", self.edge_voice),
                         ("ElevenLabs voice ID", self.eleven_voice), ("AI provider", self.ai),
                         ("OpenAI API key", self.api_key), ("AI model", self.model), ("Whisper model (restart)", self.whisper)]:
            form.addRow(label, w)
        note = QtWidgets.QLabel("API keys are stored only in the local .env file (never in settings or logs). Most changes apply immediately.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{C_DIM};")
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save |
                                        QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(note)
        lay.addWidget(bb)

    def _save(self) -> None:
        self.s.update(camera_source=self.cam_src.text().strip() or "0", preview=self.preview.isChecked(),
                      person_conf=self.person_conf.value(), greeting_cooldown_min=self.cooldown.value(),
                      unknown_greeting=self.unknown.currentData(), unknown_cooldown_min=self.unknown_cd.value(),
                      mic_device=self.mic_dev.currentData(), speaker_device=self.spk_dev.currentData(),
                      wake_threshold=self.wake.value(), followup_seconds=self.follow.value(),
                      silence_end_seconds=self.silence.value(), tts_provider=self.tts.currentText(),
                      edge_voice=self.edge_voice.text().strip(), elevenlabs_voice_id=self.eleven_voice.text().strip(),
                      ai_provider=self.ai.currentText(), openai_model=self.model.text().strip(),
                      whisper_model=self.whisper.text().strip())
        if self.api_key.text().strip():
            env_file_set("OPENAI_API_KEY", self.api_key.text().strip())
        self.core.settings_changed()
        self.accept()


class EnrollDialog(QtWidgets.QDialog):
    CONSENT = ("I am the person in front of the camera, and I consent to JARVIS storing an encrypted facial "
               "template of me on this computer so it can recognize and greet me. I can delete it at any time.")

    def __init__(self, core: JarvisCore, parent=None):
        super().__init__(parent)
        self.setWindowTitle("JARVIS - Enroll a user")
        self.core = core
        self.setMinimumWidth(560)
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("Name JARVIS should use, e.g. Lorenzo")
        self.consent = QtWidgets.QCheckBox(self.CONSENT)
        self.consent.setStyleSheet("QCheckBox { spacing: 8px; }")
        self.consent.setChecked(False)
        self.preview = QtWidgets.QLabel("Camera preview appears here once capture starts.")
        self.preview.setMinimumSize(480, 300)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet(f"border:1px solid {C_EDGE}; border-radius:8px; color:{C_DIM};")
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 8)
        self.hint = QtWidgets.QLabel("Only enroll yourself, or someone who is present and has agreed.")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f"color:{C_DIM};")
        self.start_btn = QtWidgets.QPushButton("START CAPTURE")
        self.start_btn.setObjectName("primary")
        self.start_btn.setEnabled(False)
        self.close_btn = QtWidgets.QPushButton("CLOSE")
        self.consent.toggled.connect(self._refresh)
        self.name.textChanged.connect(self._refresh)
        self.start_btn.clicked.connect(self._start)
        self.close_btn.clicked.connect(self.reject)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.close_btn)
        row.addWidget(self.start_btn)
        lay = QtWidgets.QVBoxLayout(self)
        for w in (self.name, self.consent, self.preview, self.bar, self.hint):
            lay.addWidget(w)
        lay.addLayout(row)
        core.frame.connect(self._on_frame)
        core.enroll_progress.connect(self._on_progress)
        core.enroll_hint.connect(self.hint.setText)
        core.enroll_finished.connect(self._on_finished)
        self.finished.connect(self._cleanup)
        self._active = False

    def _refresh(self) -> None:
        self.start_btn.setEnabled(self.consent.isChecked() and bool(self.name.text().strip()) and not self._active)

    def _start(self) -> None:
        name = self.name.text().strip()
        if self.core.store.exists(name):
            if QtWidgets.QMessageBox.question(self, "Replace profile", f"'{name}' is already enrolled. Replace it?") \
                    != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        err = self.core.start_enrollment(name)
        if err:
            self.hint.setText(err)
            return
        self._active = True
        self.start_btn.setEnabled(False)
        self.name.setEnabled(False)
        self.consent.setEnabled(False)
        self.hint.setText("Look at the camera. Hold still, then slowly turn your head a little between captures.")

    def _on_frame(self, img) -> None:
        if self._active:
            self.preview.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
                self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _on_progress(self, have: int, need: int) -> None:
        self.bar.setRange(0, need)
        self.bar.setValue(have)

    def _on_finished(self, ok: bool, msg: str) -> None:
        self._active = False
        self.hint.setText(msg)
        self.name.setEnabled(True)
        self.consent.setEnabled(True)
        self._refresh()
        if ok:
            QtCore.QTimer.singleShot(900, self.accept)

    def _cleanup(self) -> None:
        if self._active:
            self.core.cancel_enrollment()
        for sig, slot in ((self.core.frame, self._on_frame), (self.core.enroll_progress, self._on_progress),
                          (self.core.enroll_hint, self.hint.setText), (self.core.enroll_finished, self._on_finished)):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass


class UsersDialog(QtWidgets.QDialog):
    def __init__(self, core: JarvisCore, parent=None):
        super().__init__(parent)
        self.setWindowTitle("JARVIS - Enrolled users")
        self.core = core
        self.setMinimumSize(460, 360)
        self.list = QtWidgets.QListWidget()
        info = QtWidgets.QLabel("Only face templates are stored (encrypted, on this PC) - no photos. "
                                "Removing a user deletes their template immediately.")
        info.setWordWrap(True)
        info.setStyleSheet(f"color:{C_DIM};")
        add = QtWidgets.QPushButton("ENROLL NEW")
        rem = QtWidgets.QPushButton("REMOVE SELECTED")
        rem.setObjectName("danger")
        close = QtWidgets.QPushButton("CLOSE")
        add.clicked.connect(self._add)
        rem.clicked.connect(self._remove)
        close.clicked.connect(self.accept)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        row.addWidget(close)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(info)
        lay.addWidget(self.list, 1)
        lay.addLayout(row)
        self._reload()

    def _reload(self) -> None:
        self.list.clear()
        for u in self.core.list_users():
            it = QtWidgets.QListWidgetItem(f"{u['name']}    (enrolled {u['created'].replace('T', ' ')})")
            it.setData(Qt.ItemDataRole.UserRole, u["name"])
            self.list.addItem(it)
        if self.list.count() == 0:
            self.list.addItem("No users enrolled yet.")
            self.list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)

    def _add(self) -> None:
        EnrollDialog(self.core, self).exec()
        self._reload()

    def _remove(self) -> None:
        it = self.list.currentItem()
        name = it.data(Qt.ItemDataRole.UserRole) if it else None
        if not name:
            return
        if QtWidgets.QMessageBox.question(self, "Remove user", f"Permanently delete the face profile for '{name}'?") \
                == QtWidgets.QMessageBox.StandardButton.Yes:
            self.core.remove_user(name)
            self._reload()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, core: JarvisCore, settings: Settings):
        super().__init__()
        self.core, self.s = core, settings
        self.setWindowTitle("J.A.R.V.I.S")
        self.resize(1320, 800)
        self.setMinimumSize(1100, 700)
        self._lines: list[list[str]] = []
        self._state = S.OFFLINE
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(12)

        # header ---------------------------------------------------------------------
        head = QtWidgets.QHBoxLayout()
        logo = QtWidgets.QLabel("J.A.R.V.I.S")
        logo.setStyleSheet(f"font-size: 30px; letter-spacing: 10px; color: {C_ACCENT}; font-weight: bold;")
        glow = QtWidgets.QGraphicsDropShadowEffect(blurRadius=28, color=_qc(C_ACCENT, 200), offset=QPointF(0, 0))
        logo.setGraphicsEffect(glow)
        sub = QtWidgets.QLabel("JUST A RATHER VERY INTELLIGENT SYSTEM  -  V1")
        sub.setObjectName("h")
        lc = QtWidgets.QVBoxLayout()
        lc.setSpacing(0)
        lc.addWidget(logo)
        lc.addWidget(sub)
        self.state_pill = QtWidgets.QLabel("OFFLINE")
        self.state_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clock = QtWidgets.QLabel()
        self.clock.setStyleSheet(f"color:{C_DIM}; font-size: 15px;")
        head.addLayout(lc)
        head.addStretch(1)
        head.addWidget(self.state_pill)
        head.addSpacing(18)
        head.addWidget(self.clock)
        outer.addLayout(head)

        # body -----------------------------------------------------------------------
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(14)
        outer.addLayout(body, 1)

        left = panel()
        left.setFixedWidth(300)
        ll = QtWidgets.QVBoxLayout(left)
        ll.addWidget(heading("SYSTEM STATUS"))
        self.r_state = StatusRow("ASSISTANT")
        self.r_cam = StatusRow("CAMERA")
        self.r_mic = StatusRow("MICROPHONE")
        self.r_ai = StatusRow("AI CONNECTION")
        self.r_user = StatusRow("RECOGNIZED USER")
        self.rows = [self.r_state, self.r_cam, self.r_mic, self.r_ai, self.r_user]
        for r in self.rows:
            ll.addWidget(r)
        ll.addSpacing(6)
        ll.addWidget(heading("CAMERA VIEW (LOCAL ONLY)"))
        self.preview = QtWidgets.QLabel("CAMERA OFF")
        self.preview.setFixedSize(272, 204)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet(f"border:1px solid {C_EDGE}; border-radius:8px; color:{C_DIM};")
        ll.addWidget(self.preview)
        ll.addStretch(1)
        body.addWidget(left)

        mid = QtWidgets.QVBoxLayout()
        self.reactor = ReactorWidget()
        mid.addWidget(self.reactor, 1)
        chips = QtWidgets.QHBoxLayout()
        chips.addStretch(1)
        self.chip_listen = Chip("LISTENING", C_ACCENT)
        self.chip_think = Chip("THINKING", "#9a7bff")
        self.chip_speak = Chip("SPEAKING", "#aef0ff")
        for c in (self.chip_listen, self.chip_think, self.chip_speak):
            chips.addWidget(c)
        chips.addStretch(1)
        mid.addLayout(chips)
        self.wave = WaveformWidget()
        mid.addWidget(self.wave)
        body.addLayout(mid, 1)

        right = panel()
        right.setFixedWidth(400)
        rl = QtWidgets.QVBoxLayout(right)
        rl.addWidget(heading("CONVERSATION"))
        self.log_view = QtWidgets.QTextBrowser()
        self.log_view.setObjectName("transcript")
        rl.addWidget(self.log_view, 1)
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("Type to JARVIS...  (Enter to send)")
        self.input.returnPressed.connect(self._send_text)
        rl.addWidget(self.input)
        body.addWidget(right)

        # controls -------------------------------------------------------------------
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(8)
        self.b_start = QtWidgets.QPushButton("START ASSISTANT")
        self.b_start.setObjectName("primary")
        self.b_cam = QtWidgets.QPushButton("CAMERA ON")
        self.b_cam.setCheckable(True)
        self.b_cam.setChecked(True)
        self.b_mic = QtWidgets.QPushButton("MIC ON")
        self.b_mic.setCheckable(True)
        self.b_mic.setChecked(True)
        self.b_listen = QtWidgets.QPushButton("LISTEN")
        self.b_stop = QtWidgets.QPushButton("STOP SPEECH")
        self.vol = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(int(settings.volume * 100))
        self.vol.setFixedWidth(120)
        self.vol_lbl = QtWidgets.QLabel(f"VOL {int(settings.volume * 100)}")
        self.vol_lbl.setObjectName("h")
        b_enroll = QtWidgets.QPushButton("ENROLL")
        b_users = QtWidgets.QPushButton("USERS")
        b_set = QtWidgets.QPushButton("SETTINGS")
        for w in (self.b_start, self.b_cam, self.b_mic, self.b_listen, self.b_stop):
            bar.addWidget(w)
        bar.addSpacing(10)
        bar.addWidget(self.vol_lbl)
        bar.addWidget(self.vol)
        bar.addStretch(1)
        for w in (b_enroll, b_users, b_set):
            bar.addWidget(w)
        outer.addLayout(bar)

        # wiring ---------------------------------------------------------------------
        self.b_start.clicked.connect(self._toggle_run)
        self.b_cam.toggled.connect(self._cam_toggled)
        self.b_mic.toggled.connect(self._mic_toggled)
        self.b_listen.clicked.connect(core.listen_now)
        self.b_stop.clicked.connect(core.stop_speaking)
        self.vol.valueChanged.connect(self._vol_changed)
        b_enroll.clicked.connect(lambda: EnrollDialog(core, self).exec())
        b_users.clicked.connect(lambda: UsersDialog(core, self).exec())
        b_set.clicked.connect(lambda: SettingsDialog(settings, core, self).exec())
        core.state_changed.connect(self._on_state)
        core.transcript.connect(self._on_transcript)
        core.camera_status.connect(self._on_cam_status)
        core.mic_status.connect(self._on_mic_status)
        core.ai_status.connect(self._on_ai_status)
        core.user_changed.connect(self._on_user)
        core.people_changed.connect(self._on_people)
        core.level.connect(self._on_level)
        core.frame.connect(self._on_frame)
        core.running_changed.connect(self._on_running)

        self._cam_kind, self._mic_kind = "off", "off"
        self._on_state("OFFLINE", "")
        self._on_cam_status("off", "")
        self._on_mic_status("off", "")
        self._on_ai_status(*core.ai_initial_status())
        self._on_user("")
        self._render_log()
        self._add("system", "Press START ASSISTANT to begin. The camera and microphone stay off until you do.", True)

        self._tick = QTimer(self)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start(50)

    # ---- handlers ----------------------------------------------------------------------
    def _on_tick(self) -> None:
        for r in self.rows:
            r.led.tick()
        self.clock.setText(dt.datetime.now().strftime("%a %d %b  %H:%M:%S"))

    def _toggle_run(self) -> None:
        self.core.pause() if self.core.running else self.core.start()

    def _on_running(self, run: bool) -> None:
        self.b_start.setText("PAUSE ASSISTANT" if run else "START ASSISTANT")
        self.b_start.setObjectName("" if run else "primary")
        self.b_start.setStyle(self.b_start.style())
        if not run:
            self.preview.clear()
            self.preview.setText("CAMERA OFF")

    def _cam_toggled(self, on: bool) -> None:
        self.b_cam.setText("CAMERA ON" if on else "CAMERA OFF")
        self.core.set_camera(on)

    def _mic_toggled(self, on: bool) -> None:
        self.b_mic.setText("MIC ON" if on else "MIC OFF")
        self.core.set_mic(on)

    def _vol_changed(self, v: int) -> None:
        self.vol_lbl.setText(f"VOL {v}")
        self.core.set_volume(v / 100.0)

    def _send_text(self) -> None:
        t = self.input.text()
        self.input.clear()
        self.core.submit_text(t)

    def _on_state(self, state: str, detail: str) -> None:
        st = State(state)
        self._state = st
        label = state.replace("_", " ")
        if st == S.OFFLINE and detail == "paused":
            label = "OFFLINE - PAUSED"
        self.reactor.set_state(st, label if st != S.OFFLINE else "OFFLINE")
        col = STATE_STYLE[st][0]
        self.wave.set_color(col)
        self.state_pill.setText(f"  {label}  ")
        self.state_pill.setStyleSheet(f"border:1px solid {col}; border-radius:12px; padding:4px 10px; color:{col};"
                                      f"letter-spacing:3px; font-size:12px;")
        kind = {S.OFFLINE: "off", S.ERROR: "err", S.INITIALIZING: "warn"}.get(st, "on")
        self.r_state.set(label.title(), kind, pulse=st not in (S.OFFLINE, S.ERROR), tip=detail)
        self.chip_listen.set_active(st == S.LISTENING)
        self.chip_think.set_active(st == S.THINKING)
        self.chip_speak.set_active(st in (S.SPEAKING, S.GREETING))

    def _on_cam_status(self, kind: str, detail: str) -> None:
        self._cam_kind = kind
        if kind == "active":
            self.r_cam.set("ACTIVE", "ok", True, "Camera is on. Frames are processed locally and never saved.")
        elif kind == "error":
            self.r_cam.set("ERROR", "err", False, detail)
        else:
            self.r_cam.set("OFF", "off", False, "Camera released")
            self.preview.clear()
            self.preview.setText("CAMERA OFF")

    def _on_mic_status(self, kind: str, detail: str) -> None:
        self._mic_kind = kind
        if kind == "active":
            self.r_mic.set("ACTIVE", "ok", True, "Microphone is open. Audio is processed locally for the wake word; "
                                                 "only speech after the wake word is transcribed.")
        elif kind == "error":
            self.r_mic.set("ERROR", "err", False, detail)
        else:
            self.r_mic.set("OFF", "off", False, "Microphone closed")

    def _on_ai_status(self, kind: str, detail: str) -> None:
        text, led, pulse = {"ready": ("READY", "on", False), "online": ("ONLINE", "ok", False),
                            "busy": ("WORKING", "warn", True), "nokey": ("NO API KEY", "warn", False),
                            "error": ("ERROR", "err", False)}.get(kind, (kind.upper(), "off", False))
        self.r_ai.set(text, led, pulse, detail or self.core.ai.name)

    def _on_user(self, text: str) -> None:
        if not text:
            self.r_user.set("-", "off", False, "")
        elif text == "Unknown visitor":
            self.r_user.set(text, "warn", False, "Visitors are never identified or enrolled automatically")
        else:
            self.r_user.set(text, "ok", False, "")

    def _on_people(self, n: int) -> None:
        self.r_user.title.setText("RECOGNIZED USER" + (f"  ({n} IN VIEW)" if n else ""))

    def _on_level(self, source: str, v: float) -> None:
        if source == "mic" and self._state in (S.LISTENING, S.IDLE, S.DETECTING_PERSON, S.RECOGNIZING_USER):
            self.wave.push(v * (1.0 if self._state == S.LISTENING else 0.35))
            if self._state == S.LISTENING:
                self.reactor.set_level(v)
        elif source == "speaker" and self._state in (S.SPEAKING, S.GREETING, S.THINKING):
            self.wave.push(v)
            self.reactor.set_level(v)

    def _on_frame(self, img) -> None:
        if self.core.running and self.b_cam.isChecked() and self.s.preview:
            self.preview.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
                self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    # ---- transcript --------------------------------------------------------------------
    def _on_transcript(self, role: str, text: str, new: bool) -> None:
        self._add(role, text, new)

    def _add(self, role: str, text: str, new: bool) -> None:
        if new or not self._lines or self._lines[-1][0] != role:
            self._lines.append([role, text])
        else:
            self._lines[-1][1] += text
        self._lines = self._lines[-200:]
        self._render_log()

    def _render_log(self) -> None:
        from html import escape

        style = {"user": ("YOU", "#e6f4ff"), "assistant": ("JARVIS", C_ACCENT), "system": ("SYSTEM", C_DIM)}
        parts = []
        for role, text in self._lines:
            who, col = style.get(role, (role.upper(), C_TEXT))
            body = escape(text).replace("\n", "<br>")
            parts.append(f"<p style='margin:0 0 10px 0'><span style='color:{col};letter-spacing:2px;font-size:11px'>"
                         f"{who}</span><br><span style='color:{col if role != 'assistant' else '#bfeaff'}'>{body}</span></p>")
        self.log_view.setHtml("".join(parts))
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def paintEvent(self, e) -> None:
        p = QtGui.QPainter(self)
        g = QtGui.QRadialGradient(QPointF(self.width() * 0.5, self.height() * 0.4), max(self.width(), self.height()) * 0.8)
        g.setColorAt(0, QtGui.QColor("#0a1630"))
        g.setColorAt(1, QtGui.QColor(C_BG))
        p.fillRect(self.rect(), g)


# =======================================================================================
# 11. DIAGNOSTICS + ENTRY POINT
# =======================================================================================
def run_diagnostics() -> int:
    """`python jarvis.py --check` - answers: is my PC ready? Prints hardware + dependency status."""
    import platform

    ok = lambda b: "OK  " if b else "FAIL"
    print(f"JARVIS V1 diagnostics\n  Python {platform.python_version()} on {platform.platform()}")
    print(f"  OpenCV {cv2.__version__}  numpy {np.__version__}  PySide6 {QtCore.__version__}")
    print("\n[GPU]")
    try:
        import torch

        print(f"  torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}",
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(CPU build or no NVIDIA driver)")
    except ImportError:
        print("  torch not installed (YOLO will fall back to OpenCV HOG)")
    try:
        import ctranslate2

        print(f"  CTranslate2 CUDA devices (Whisper): {ctranslate2.get_cuda_device_count()}")
    except ImportError:
        print("  ctranslate2 not installed (faster-whisper missing)")
    print("\n[Webcam]")
    found = False
    for i in range(4):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(i)
        good = cap.isOpened() and cap.read()[0]
        cap.release()
        if good:
            found = True
            print(f"  {ok(True)} camera index {i} delivers frames")
    if not found:
        print(f"  {ok(False)} no webcam delivered a frame (indexes 0-3)")
    print("\n[Audio]")
    try:
        import sounddevice as sd

        dd = sd.default.device
        print(f"  default input/output device ids: {dd[0]} / {dd[1]}")
        ins, outs = list_audio_devices()
        print(f"  {ok(bool(ins))} microphones: " + ("; ".join(f"{i}:{n}" for i, n in ins[:6]) or "none"))
        print(f"  {ok(bool(outs))} speakers:    " + ("; ".join(f"{i}:{n}" for i, n in outs[:6]) or "none"))
    except Exception as e:
        print(f"  {ok(False)} sounddevice unavailable: {e}")
    print("\n[Packages]")
    for mod, optional in (("ultralytics", 0), ("faster_whisper", 0), ("openwakeword", 0), ("onnxruntime", 0),
                          ("openai", 0), ("edge_tts", 0), ("miniaudio", 0), ("cryptography", 0), ("keyring", 0),
                          ("piper", 1)):
        try:
            __import__(mod)
            print(f"  {ok(True)} {mod}")
        except ImportError:
            print("  info" if optional else f"  {ok(False)}", mod,
                  "(optional offline voice: pip install piper-tts)" if optional else "(pip install -r requirements.txt)")
    print("\n[Configuration]")
    print(f"  {ok(bool(os.getenv('OPENAI_API_KEY')))} OPENAI_API_KEY set (needed for AI answers)")
    print(f"  {'OK  ' if os.getenv('ELEVENLABS_API_KEY') else 'info'} ELEVENLABS_API_KEY "
          f"{'set' if os.getenv('ELEVENLABS_API_KEY') else 'not set (optional; Edge voices are the default)'}")
    s = Settings()
    print(f"  voice provider: {s.tts_provider}   AI provider: {s.ai_provider} / {s.openai_model}")
    print(f"  enrolled users: {len(ProfileStore().list_users())}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="JARVIS V1 desktop assistant")
    ap.add_argument("--check", action="store_true", help="print hardware/dependency diagnostics and exit")
    ap.add_argument("--verbose", action="store_true", help="debug logging")
    args = ap.parse_args()
    setup_logging(args.verbose)
    if args.check:
        return run_diagnostics()

    import signal

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("JARVIS")
    app.setStyleSheet(QSS)
    ppid = int(os.getenv("JARVIS_PARENT_PID", "0") or 0)
    if ppid:  # launched by the setup stage: quit if the IDE's Stop button killed it, so no camera/mic is left on
        import psutil

        watch = QTimer()
        watch.timeout.connect(lambda: None if psutil.pid_exists(ppid) else os._exit(0))  # OS frees camera/mic
        watch.start(1500)
        app._watch = watch
    settings = Settings()
    skip = USERS_DIR / ".skip_key_prompt"
    if not os.getenv("OPENAI_API_KEY") and not skip.exists():
        text, accepted = QtWidgets.QInputDialog.getText(
            None, "JARVIS - AI key",
            "Paste your OpenAI API key so JARVIS can answer questions.\n"
            "Create one at https://platform.openai.com/api-keys  (it is stored only in the local .env file).\n\n"
            "Leave empty to skip: JARVIS still runs, and you can add the key later in Settings.",
            QtWidgets.QLineEdit.EchoMode.Password)
        if accepted and text.strip():
            env_file_set("OPENAI_API_KEY", text.strip())
        else:
            skip.write_text("skipped")
    core = JarvisCore(settings)
    win = MainWindow(core, settings)
    win.show()
    app.aboutToQuit.connect(core.shutdown)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    keepalive = QTimer()  # lets Python handle Ctrl+C while Qt runs
    keepalive.start(300)
    keepalive.timeout.connect(lambda: None)
    if os.getenv("JARVIS_AUTOSTART", "").lower() in ("1", "true", "yes"):
        QTimer.singleShot(400, core.start)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
