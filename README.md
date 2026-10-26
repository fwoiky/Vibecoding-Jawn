# Vibecoding-Jawn — ENGR-126 Project 15a: From Pictures to Video (AnimateDiff)

A sentence in, a short clip out. One Google Colab cell that:

1. **Part 1 – Flicker demo:** generates 16 *independent* Stable Diffusion 1.5 images from one prompt and plays them as a looping video. Each frame is a fresh "roll of the dice", so the video flickers.
2. **Part 2 – AnimateDiff:** generates all 16 frames *together* with the AnimateDiff motion adapter, so the clip is usually far more consistent from frame to frame.
3. **Part 3 – Prompt-to-video panel:** type any prompt, pick frames/steps, press **Make clip**, then **Download all** to get `project15a_videos.zip`.

| File | What it is |
|---|---|
| [`project15a_animatediff.py`](project15a_animatediff.py) | The complete program — paste it into **one** Colab cell |
| [`project15a_animatediff.ipynb`](project15a_animatediff.ipynb) | The same program as a ready-made notebook (open in Colab) |
| [`TESTING.md`](TESTING.md) | Testing checklist, experiment log and verification status |
| [`docs/Project15a_Project_Plan.docx`](docs/Project15a_Project_Plan.docx) | Polished project plan (export to PDF from Word / Google Docs) |

> **Verification status:** the code has been syntax-checked, its Diffusers API calls were checked against the Diffusers 0.42 source, and its control flow, MP4 export (H.264), ZIP packaging and panel buttons were exercised in a mocked dry run *without* real models. It has **not yet been run in Colab on a GPU** — see [`TESTING.md`](TESTING.md).

---

## 1. Setup and run

1. Open <https://colab.research.google.com> and create a new notebook (or upload `project15a_animatediff.ipynb`).
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Paste the whole of `project15a_animatediff.py` into a single code cell.
4. Edit the settings at the top if you like:
   ```python
   SHOW_FLICKER = True
   SHOW_ANIMATEDIFF = True
   SHOW_PANEL = True

   PROMPT = "A red flag waving in the wind on a sunny day"
   FRAMES = 16
   STEPS = 25
   SEED = 42
   ```
5. Run the cell (▶ or Shift+Enter).
   - First run: it may remove `torchao` and install/upgrade `diffusers` and video packages. If it says **"Packages updated. Please restart"**, choose **Runtime → Restart session** and run the cell again.
   - It downloads the models (~2–3 GB) the first time; later runs reuse the cache.
6. Scroll down: GPU status → Part 1 → Part 2 → side-by-side comparison → Part 3 panel.
7. When you're done, **Runtime → Disconnect and delete runtime** so the shared T4 is freed.

Outputs are saved in `/content/project15a_outputs/` (MP4s, `metadata.json`, `project15a_videos.zip`). Colab deletes this folder when the runtime ends, so press **Download all** first.

### Settings reference

| Setting | Default | Meaning |
|---|---|---|
| `SHOW_FLICKER` / `SHOW_ANIMATEDIFF` / `SHOW_PANEL` | `True` | Turn each part on/off. Off = that part's model work is skipped. |
| `PROMPT` | red flag | What to draw. Moving subjects (waves, flags, cars) show motion best. |
| `FRAMES` | 16 | Frames per clip, 8–24. More frames = longer clip, more memory. |
| `STEPS` | 25 | Denoising steps, 10–50. More steps = more detail, slower. |
| `SEED` | 42 | Starting random number. Part 1 uses `SEED, SEED+1, …`; Part 2 uses `SEED`. |
| `NEGATIVE_PROMPT`, `GUIDANCE`, `WIDTH/HEIGHT`, `FPS` | see file | Advanced; shared by all parts so the comparison is fair. |

Invalid settings are reported and never silently changed. Re-running with identical settings shows the earlier result instead of regenerating; change the seed or prompt to make a new one.

---

## 2. How it works (plain English)

| Piece | What it does |
|---|---|
| **Stable Diffusion 1.5** (`stable-diffusion-v1-5/stable-diffusion-v1-5`) | Starts from random noise and removes noise step by step until an image matching the prompt appears. The **seed** decides the starting noise. |
| **Independent frames (Part 1)** | 16 separate images, each with its own seed. Nothing tells frame 5 what frame 4 looked like, so the flag's color, shape and position jump around → **flicker**. |
| **AnimateDiff motion adapter** (`guoyww/animatediff-motion-adapter-v1-5-2`) | Extra "temporal" layers added to SD 1.5's UNet. While denoising, frames can look at each other, so all 16 are generated *together* and stay consistent. It's better, not perfect — expect some warping or odd motion. |
| `AnimateDiffPipeline.from_pipe(sd, motion_adapter=adapter)` | Builds the AnimateDiff pipeline by reusing the already-loaded SD text encoder and VAE, instead of loading them twice. The old SD pipeline is then deleted so only **one large pipeline is on the GPU**. |
| `DDIMScheduler(clip_sample=False, timestep_spacing="linspace", beta_schedule="linear", steps_offset=1)` | The noise-removal recipe the motion adapter expects. (Part 1's still images keep SD 1.5's own noise schedule with the same DDIM sampler.) |
| `float16`, `pipe.vae.enable_slicing()`, `torch.inference_mode()`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Memory savers so everything fits on a free 15 GB T4. |
| `diffusers.utils.export_to_video` | Writes the frames to an H.264 MP4 that plays in the browser. A GIF preview is offered as a fallback under each video. |
| ipywidgets panel | Prompt box, Frames/Steps dropdowns, **Make clip** (new random seed each time, shown on the label) and **Download all** (zips every finished MP4 + `metadata.json`). |
| `STATE` (global) | Remembers the loaded model and finished clips between cell re-runs; if a previous run crashed mid-generation, the GPU is cleaned up first. |

---

## 3. Troubleshooting

| Problem | What to do |
|---|---|
| **"No GPU detected"** | Runtime → Change runtime type → **T4 GPU** → Save, then re-run. Free T4s are shared; if none is available, try later. |
| **CUDA out of memory** | Settings are not changed for you. Try 12 or 8 frames, fewer steps, close other GPU notebooks, or Runtime → Restart session (memory can get fragmented), then re-run. Completed clips stay on disk. |
| **"Packages updated. Please restart"** | Runtime → Restart session, then run the cell again. |
| **Import / dependency error** | Expand "Full traceback" in the red box, read the *last* lines, and paste the whole traceback when asking for help before changing package versions. |
| **Video won't play** | Click "Video not playing? Show GIF preview" under it. The MP4 is still saved in `/content/project15a_outputs/`. |
| **Panel doesn't appear / buttons do nothing** | Make sure `SHOW_PANEL = True` and re-run the cell. Widgets need the cell's output to be visible; scroll to the bottom. |
| **Download all says "No finished clips yet"** | Make at least one clip first (or run Part 1/Part 2). |
| **Download doesn't start** | Allow pop-ups/downloads for colab.research.google.com, or use the folder icon on the left → `project15a_outputs` → right-click the ZIP → Download. |
| **A frame is solid black (Part 1)** | The SD safety checker flagged it. Use an ordinary, everyday prompt or a different seed. |
| **Very slow** | First run downloads models. 24 frames or 40 steps take noticeably longer. |

---

## 4. Demo script (≈3 minutes)

1. **Show the setup:** point at the three switches and the `PROMPT / FRAMES / STEPS / SEED` settings at the top.
2. **Run the cell.** Point out the GPU status box (T4, free memory) and the progress bars.
3. **Part 1:** "These 16 images all used the same prompt but different seeds, generated one by one." Show the labelled frame grid, then the looping video — *the flicker*.
4. **Part 2:** "Same prompt, seed 42, but AnimateDiff generates the frames together." Show the frames, the seed and generation time, then the looping video.
5. **Comparison:** play both side by side. Say honestly what you see — smoother? any warping?
6. **Part 3:** type a new prompt (e.g. "ocean waves crashing on rocks at sunset"), choose 16 frames / 25 steps, press **Make clip**. Show the seed label on the result.
7. **Download all:** press it and open `project15a_videos.zip` — MP4s + `metadata.json`.
8. Close with responsible use: every clip is labelled AI-generated; use fictional scenes and ordinary objects.

---

## Responsible use

All clips are AI-generated — label them that way when shared. Use fictional scenes and ordinary objects; don't make deceptive content of real people or use copyrighted characters/logos. Exact reproducibility across GPUs or library versions is not guaranteed even with the same seed.
