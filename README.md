# J.A.R.V.I.S. V1

A single-file desktop AI assistant for Windows 11: webcam person detection, optional (consent-based) face
recognition with automatic greetings, a local "Hey JARVIS" wake word, speech recognition, natural spoken
replies, a configurable AI backend, and a futuristic PySide6 dashboard.

```
jarvis/
├── jarvis.py            <- the whole application (python jarvis.py)
├── requirements.txt
├── .env.example         <- copy to .env and add your keys
├── users/enrolled_users <- encrypted face templates (created for you)
├── models/              <- downloaded models (YOLO, face models, Piper voices)
└── README.md
```

## 1. What you need

* Windows 11, **Python 3.11 or 3.12** (64-bit) from python.org
* A webcam, microphone and speakers/headphones
* An OpenAI API key for answers (see step 5). Everything else works without any account.
* Optional: NVIDIA GPU (your RTX 3060 12 GB is ideal). Everything falls back to CPU automatically.

## 2. Install

Open PowerShell in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3. (GPU) Install the CUDA build of PyTorch first

`pip install ultralytics` alone installs a **CPU-only** torch on Windows. For the RTX 3060, get the exact
command for "Stable / Windows / Pip / CUDA 12.x" from <https://pytorch.org/get-started/locally/> and run it
before the next step. It looks like:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

(Skip this step to run YOLO on the CPU. It is lightweight at the 3-8 fps JARVIS uses.)

## 4. Install the rest

```powershell
pip install -r requirements.txt
```

Optional, for GPU speech recognition (faster-whisper needs the CUDA 12 runtime libraries):

```powershell
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```

If these are missing, JARVIS notices at start-up and uses the CPU model instead (a bit slower, still fine).

Check your machine at any time:

```powershell
python jarvis.py --check
```

It lists which webcams deliver frames, your microphones/speakers, whether torch and CTranslate2 can see the
GPU, which packages are installed and whether keys are configured.

## 5. Configure the AI key

1. Create a key at <https://platform.openai.com/api-keys>.
2. `Copy-Item .env.example .env`
3. Open `.env` and set `OPENAI_API_KEY=sk-...`

Without a key JARVIS still starts and says so (and prints these steps in the transcript); it will not pretend
to answer. The key is only ever read from `.env` / the environment, never saved to settings or logs. The default
model is `gpt-4o-mini`; change it in Settings or with `JARVIS_OPENAI_MODEL=` in `.env`.
`OPENAI_BASE_URL` lets you point at any OpenAI-compatible server instead.

## 6. Run

```powershell
python jarvis.py
```

The camera and microphone stay **off until you press START ASSISTANT**. First start downloads models (about
40 MB of face models, the YOLO weights, the wake-word models and the Whisper model, up to ~500 MB for
`small.en`), so it takes a few minutes once. After that start-up takes a few seconds.

### Using it

| Do this | What happens |
|---|---|
| Say **"Hey JARVIS"** | chime, then it listens. Ask anything. |
| Keep talking after an answer | follow-ups work without the wake word (8 s window, configurable) |
| Say "thanks JARVIS", "that's all", "never mind" | ends the conversation locally; back to wake-word mode |
| LISTEN button | same as the wake word (also the fallback if the wake word model can't load) |
| STOP SPEECH | silences JARVIS immediately and ends the conversation |
| Type in the box | talk to JARVIS without a microphone |
| CAMERA / MIC buttons | releases the device completely (OS camera/mic indicator goes off) |
| PAUSE ASSISTANT | releases both devices and goes OFFLINE |

### Enrolling yourself (optional)

ENROLL -> type your name -> tick the consent box -> START CAPTURE. Look at the camera and slowly turn your
head a little; 8 clear samples are captured. Only the person enrolling may be visible in the frame.
USERS lists enrolled people and **removes** them (the template file is deleted immediately).

When you walk in, JARVIS confirms your face across several consecutive frames and then greets you
("Good morning, Lorenzo."), at most once per cooldown (default 30 min), and never while it is talking or in a
conversation. Strangers are shown as "Unknown visitor": they are never named, never enrolled automatically,
and by default get no greeting (Settings -> Unknown visitors -> Neutral greeting if you want one).

## 7. Design choices

| Area | Choice | Why |
|---|---|---|
| Person detection | YOLOv8n via ultralytics (CUDA when torch sees it) | accurate, tiny, 3-8 fps is plenty |
| Face recognition | OpenCV YuNet + SFace (ONNX) | no dlib/CMake build pain on Windows; embeddings only |
| Wake word | openWakeWord `hey_jarvis` (ONNX) | fully local, ships a pre-trained "hey jarvis" model |
| Speech-to-text | faster-whisper `small.en` on GPU / `base.en` on CPU | accurate and fast; energy endpointing + Whisper VAD |
| AI | `AIProvider` interface; OpenAI implementation streams sentence by sentence | first words are spoken before the answer is finished; Claude can be added later |
| UI | PySide6, all heavy work on threads, one central state machine | UI never blocks |

### Voice engines (change in Settings or `JARVIS_TTS_PROVIDER`)

| Provider | Quality | Cost / needs | Latency | Notes |
|---|---|---|---|---|
| `edge` (default) | very good neural voices | free, no key, internet | sentence-level pipelining | default voice `en-GB-RyanNeural`; uses Microsoft's online service via the `edge-tts` package (unofficial), so it could change without notice |
| `elevenlabs` | best, most human | `ELEVENLABS_API_KEY`, free tier is quota-limited | true audio streaming | pick any voice id in Settings; default model `eleven_flash_v2_5` is the low-latency one |
| `piper` | good, a little more synthetic | fully offline; `pip install piper-tts` | fast on CPU | voice downloads to `models/piper/` on first use |

If the chosen provider fails (no internet, bad key) JARVIS says so once and falls back to `edge`, then `piper`,
then text-only. To add another engine subclass `TTSProvider` and add it to `TTS_PROVIDERS` in `jarvis.py`.

### Adding Claude (or another model) later

Subclass `AIProvider` (four small methods: `configured`, `setup_help`, `check`, `stream_chat`), register it in
`AI_PROVIDERS`, set `JARVIS_AI_PROVIDER`. Nothing else changes.

## 8. Privacy and safety

* Camera frames are processed in memory, never written to disk and never uploaded. Face recognition is local.
* Only a 128-number face *template* is stored (no photos), encrypted with Fernet; the key lives in Windows
  Credential Manager (falls back to `users/.profile.key`). This stops casual copying of the folder. It is not a
  defence against malware running as you.
* The microphone stream is open only while the assistant is running and MIC is on. The wake word runs locally;
  audio is only transcribed after the wake word (or LISTEN), and only your transcribed words are sent to the AI.
* Camera and microphone status LEDs pulse in the dashboard while active; PAUSE / CAMERA / MIC release devices at once.
* JARVIS cannot run commands, open apps or read your files. The AI replies are text only.
* `jarvis.log` records events (state changes, errors), never transcripts, keys or images.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `python jarvis.py --check` shows no webcam | close other apps using it; try `JARVIS_CAMERA_SOURCE=1` |
| Wake word never fires | lower the threshold in Settings (0.3-0.4), check the right microphone is selected, or use LISTEN |
| Wake word fires on its own | raise the threshold (0.6-0.7) or use headphones |
| Whisper says CUDA/cuBLAS error and uses CPU | `pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"` |
| No voice | check Settings -> Voice provider; the transcript says which voice failed |
| First answer is slow | the first call loads models; later ones are quick |

## 10. Verification status

JARVIS was built and tested in a Linux cloud container with **no webcam, microphone, speakers or GPU**, so
hardware paths could not be exercised. Tests used real models on real images/video, plus fake audio devices.
Run `python jarvis.py --check` first on your PC; it tells you immediately what your hardware supports.

| # | Scenario | Status |
|---|---|---|
| 1 | Application launches | Verified (dashboard renders headless; real app starts and stays up) |
| 2 | Webcam connects | **Not verified on a real webcam.** Capture loop verified with a video file as the camera; the "no camera" path reports once and doesn't crash |
| 3 | Person enters / leaves, multiple people, no repeat triggers | Verified (YOLOv8n on CPU, 2 people, exactly one enter/leave event each) |
| 4 | Enrolled user recognized (multi-frame) | Verified with the real YuNet/SFace models: same person 0.93 cosine, other faces 0.06-0.13 vs 0.363 threshold |
| 5 | Greets by name, time-of-day, no repeats, cooldown, not during a conversation | Verified (text, state and cooldown logic) |
| 6 | Unknown visitor not identified | Verified (stays "Unknown visitor"; silent by default, neutral greeting if enabled) |
| 7 | Wake word | Model downloads and loads (verified). **Detection of a real spoken "Hey JARVIS" not verified**: tests injected a fake score |
| 8 | Speech transcribed correctly | **Not verified**: the Whisper model could not be downloaded in the test container. Endpointing (silence detection, noise floor) and the wrapper's logic/CPU fallback were tested with synthetic audio and a stub model |
| 9 | AI answers | Verified with the real `openai` client streaming from a local OpenAI-compatible test server (incl. bad key, missing key). **Not verified against OpenAI itself** (no key available) |
| 10 | Natural speech | **Not verified audibly.** Edge TTS was unreachable from the container; ElevenLabs request/stream handling tested against a stub; Piper's API checked, voice download not tested. Sentence pipelining, volume, instant stop and no-overlap were verified with a fake audio device |
| 11 | Follow-up questions | Verified (no wake word needed; conversation history sent with the second request) |
| 12 | Pause | Verified |
| 13 | Camera / mic disable | Verified (streams/devices closed) |
| 14 | Clean shutdown | Verified (no mic/audio streams or threads left running) |

Also **not verified**: any NVIDIA/CUDA path (YOLO on CUDA, Whisper on CUDA; the CPU fallbacks are tested),
Windows-specific behaviour (DirectShow camera open, Windows Credential Manager key storage, DLL loading for
cuBLAS/cuDNN), and tuning of the wake-word/voice-activity thresholds for your room and microphone
(Settings has sliders for both).
