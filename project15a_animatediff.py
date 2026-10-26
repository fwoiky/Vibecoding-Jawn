# =============================================================================
# ENGR-126 · Project 15a — From Pictures to Video: AnimateDiff
# One complete Google Colab cell.  Runtime → Change runtime type → T4 GPU,
# paste this whole file into ONE cell, edit the settings below, then run it.
#
#   Part 1  SHOW_FLICKER      16 independent Stable Diffusion images → video
#                             (shows the flicker you get from separate images)
#   Part 2  SHOW_ANIMATEDIFF  AnimateDiff generates all frames together → video
#   Part 3  SHOW_PANEL        Type any prompt → Make clip → Download all (ZIP)
#
# Every clip is AI-generated. Label it that way if you share it.
# =============================================================================

# ----------------------------- MAIN SETTINGS ---------------------------------
SHOW_FLICKER = True        # Part 1: independent images → flickery video
SHOW_ANIMATEDIFF = True    # Part 2: AnimateDiff clip
SHOW_PANEL = True          # Part 3: interactive prompt-to-video panel

PROMPT = "A red flag waving in the wind on a sunny day"
FRAMES = 16                # try 8, 16, 24 (allowed range 8–24)
STEPS = 25                 # try 10, 25, 40 (allowed range 10–50)
SEED = 42                  # same seed + same settings → (nearly) same result

# --------------------------- ADVANCED SETTINGS -------------------------------
NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed, watermark, text"
GUIDANCE = 7.5             # how strongly the image follows the prompt
WIDTH, HEIGHT = 512, 512   # Stable Diffusion 1.5's native size
FPS = 8                    # 16 frames at 8 fps = a 2-second loop

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"

# =============================================================================
# STEP 0 — Environment setup (runs before torch is imported)
# =============================================================================
import os, sys, subprocess, importlib.metadata

# Must be set BEFORE torch is imported to take effect.
if "torch" not in sys.modules:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
elif os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != "expandable_segments:True":
    print("ℹ️  torch was already imported in this runtime, so PYTORCH_CUDA_ALLOC_CONF "
          "applies only after Runtime → Restart session.")


def _version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def _vtuple(v):
    parts = []
    for p in v.split(".")[:3]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _pip(*args):
    print("   $ pip", " ".join(args))
    subprocess.run([sys.executable, "-m", "pip", *args, "-q"], check=True)


MIN_DIFFUSERS = (0, 31, 0)   # 0.31+ exports browser-playable H.264 MP4 via imageio
_needed = []
_dv = _version("diffusers")
if _dv is None or _vtuple(_dv) < MIN_DIFFUSERS:
    _needed.append("diffusers>=0.31")
for _pkg in ["transformers", "accelerate", "imageio", "imageio-ffmpeg", "ipywidgets"]:
    if _version(_pkg) is None:
        _needed.append(_pkg)

# Handout hint: torchao conflicts with Diffusers in Colab and is not used here.
if _version("torchao") is not None:
    print("🔧 Removing torchao (known conflict with Diffusers; not used by this project)…")
    _pip("uninstall", "-y", "torchao")
if _needed:
    print("🔧 Installing / upgrading:", ", ".join(_needed))
    _pip("install", "-U", *_needed)
    if any(m in sys.modules for m in ("diffusers", "transformers")):
        raise SystemExit("✅ Packages updated. Please use Runtime → Restart session, "
                         "then run this cell again (old versions are still loaded).")
else:
    print(f"✅ Packages already installed (diffusers {_dv}).")

# =============================================================================
# STEP 1 — Imports, GPU check, shared state
# =============================================================================
import base64, gc, html, io, json, math, secrets, time, traceback, zipfile
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from IPython.display import display, HTML
import diffusers
from diffusers import (AnimateDiffPipeline, DDIMScheduler, MotionAdapter,
                       StableDiffusionPipeline)
from diffusers.utils import export_to_video

try:
    import ipywidgets as widgets
except ImportError:
    widgets = None

try:
    from google.colab import files as colab_files   # browser download in Colab
except ImportError:
    colab_files = None

OUT_DIR = Path("/content/project15a_outputs" if Path("/content").exists()
               else "project15a_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = OUT_DIR / "project15a_videos.zip"
DTYPE = torch.float16
OOM_ERRORS = (torch.cuda.OutOfMemoryError,) if hasattr(torch.cuda, "OutOfMemoryError") else ()

# Survives re-running the cell, so loaded models and finished clips are reused.
STATE = globals().setdefault("_P15A_STATE", {
    "pipe": None,      # the ONE large pipeline allowed on the GPU
    "kind": None,      # "sd" or "animatediff"
    "busy": False,     # True while loading/generating (left True if interrupted)
    "clips": [],       # metadata for every finished clip (newest last)
    "results": {},     # Part 1 / Part 2 results, keyed by their settings
})


def box(msg, kind="info"):
    colors = {"info": "#e8f0fe", "ok": "#e6f4ea", "warn": "#fef7e0", "error": "#fce8e6"}
    display(HTML(f'<div style="background:{colors[kind]};color:#202124;padding:8px 12px;'
                 f'border-radius:6px;margin:6px 0;font:14px sans-serif">{msg}</div>'))


def header(title):
    display(HTML(f'<h2 style="font-family:sans-serif;border-bottom:2px solid #888;'
                 f'padding-bottom:4px;margin-top:24px">{title}</h2>'))


def gpu_memory():
    if not torch.cuda.is_available():
        return "no GPU"
    free, total = torch.cuda.mem_get_info()
    return f"{free / 1e9:.1f} GB free of {total / 1e9:.1f} GB"


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_pipeline(reason=""):
    if STATE["pipe"] is not None:
        print(f"🧹 Releasing the {STATE['kind']} pipeline{' (' + reason + ')' if reason else ''}…")
    STATE["pipe"], STATE["kind"] = None, None
    free_gpu()


header("Project 15a — From Pictures to Video (AnimateDiff)")
GPU_OK = torch.cuda.is_available()
if GPU_OK:
    _props = torch.cuda.get_device_properties(0)
    GPU_NAME = _props.name
    box(f"🖥️ <b>GPU:</b> {GPU_NAME} · total {_props.total_memory / 1e9:.1f} GB · "
        f"currently {gpu_memory()}<br>torch {torch.__version__} · diffusers "
        f"{diffusers.__version__} · outputs → <code>{OUT_DIR}</code>", "ok")
else:
    GPU_NAME = "none"
    box("❌ <b>No GPU detected.</b> Generating video on a CPU would take hours, so all "
        "parts are skipped.<br>Fix: <b>Runtime → Change runtime type → T4 GPU → Save</b>, "
        "then run this cell again. (Free T4s are shared and sometimes unavailable.)", "error")

if STATE["busy"]:
    # A previous run was interrupted or crashed mid-generation: start clean.
    release_pipeline("previous run did not finish cleanly")
    STATE["busy"] = False

# =============================================================================
# STEP 2 — Input checks and helpers
# =============================================================================
def validate(prompt, frames, steps, seed=0):
    """Return a list of problems (empty list = OK). Settings are never silently changed."""
    problems = []
    if not isinstance(prompt, str) or not prompt.strip():
        problems.append("The prompt is empty — describe a scene, e.g. 'ocean waves at sunset'.")
    if not isinstance(frames, int) or not 8 <= frames <= 24:
        problems.append(f"FRAMES must be a whole number from 8 to 24 (got {frames!r}).")
    if not isinstance(steps, int) or not 10 <= steps <= 50:
        problems.append(f"STEPS must be a whole number from 10 to 50 (got {steps!r}).")
    if not isinstance(seed, int) or seed < 0:
        problems.append(f"SEED must be a whole number ≥ 0 (got {seed!r}).")
    return problems


def pick_evenly(items, k=8):
    if len(items) <= k:
        return list(range(len(items)))
    return sorted({round(i * (len(items) - 1) / (k - 1)) for i in range(k)})


def frame_grid(frames, labels, cols=4, thumb=160):
    """Small labelled contact sheet of frames (a PIL image)."""
    w, h = frames[0].size
    th = int(thumb * h / w)
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * (th + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (frame, label) in enumerate(zip(frames, labels)):
        x, y = (i % cols) * thumb, (i // cols) * (th + 18)
        sheet.paste(frame.convert("RGB").resize((thumb, th)), (x, y))
        draw.text((x + 4, y + th + 3), label, fill="black")
    return sheet


def gif_base64(frames, size=256):
    small = [f.convert("RGB").resize((size, int(size * f.height / f.width))) for f in frames]
    buf = io.BytesIO()
    small[0].save(buf, format="GIF", save_all=True, append_images=small[1:],
                  duration=int(1000 / FPS), loop=0)
    return base64.b64encode(buf.getvalue()).decode()


def video_card(clip, width=320):
    """HTML for one looping, labelled video (with a GIF fallback if it won't play)."""
    data = base64.b64encode(Path(clip["file"]).read_bytes()).decode()
    seed_text = (f"seeds {clip['seeds'][0]}–{clip['seeds'][-1]}" if clip.get("seeds")
                 else f"seed {clip['seed']}")
    caption = (f"<b>{html.escape(clip['title'])}</b> · <span style='background:#fde68a;"
               f"padding:0 4px;border-radius:3px'>AI-generated</span><br>"
               f"“{html.escape(clip['prompt'])}”<br>{seed_text} · {clip['frames']} frames · "
               f"{clip['steps']} steps · {clip['seconds']:.0f} s<br>"
               f"<code>{html.escape(Path(clip['file']).name)}</code>")
    gif = (f"<details><summary style='cursor:pointer'>Video not playing? Show GIF preview</summary>"
           f"<img width='{width}' src='data:image/gif;base64,{clip['_gif']}'></details>"
           if clip.get("_gif") else "")
    return (f'<figure style="display:inline-block;margin:6px;vertical-align:top;'
            f'font:12px sans-serif;max-width:{width}px">'
            f'<video width="{width}" autoplay loop muted playsinline controls '
            f'src="data:video/mp4;base64,{data}"></video>'
            f'<figcaption>{caption}{gif}</figcaption></figure>')


class PageProgress:
    """A progress bar printed in the notebook output that updates in place."""

    def __init__(self, label, total):
        self.label, self.total = label, total
        self.handle = display(HTML(self._html(0)), display_id=True)

    def _html(self, done):
        return (f'<div style="font:13px sans-serif">{self.label} '
                f'<progress value="{done}" max="{self.total}" style="width:240px"></progress> '
                f'{done}/{self.total}</div>')

    def __call__(self, done):
        self.handle.update(HTML(self._html(done)))


def step_callback(on_step, offset=0):
    """Diffusers calls this after every denoising step; we only report progress."""
    def _cb(pipe, step, timestep, callback_kwargs):
        if on_step:
            on_step(offset + step + 1)
        return callback_kwargs
    return _cb


def show_error(err, what):
    if OOM_ERRORS and isinstance(err, OOM_ERRORS):
        free_gpu()
        box(f"💥 <b>CUDA out of memory during {what}.</b> Your settings were <i>not</i> changed "
            "automatically. Try: (1) fewer frames (12 or 8), (2) fewer steps, (3) close other "
            "GPU notebooks, (4) Runtime → Restart session if memory is fragmented, then run "
            "again. Completed clips are still saved in "
            f"<code>{OUT_DIR}</code>.<br>GPU now: {gpu_memory()}", "error")
    else:
        tb = html.escape(traceback.format_exc())
        box(f"❌ <b>{what} failed:</b> {html.escape(type(err).__name__)}: "
            f"{html.escape(str(err))}<br>Completed clips are kept. "
            f"<details><summary>Full traceback (copy this when asking for help)</summary>"
            f"<pre style='white-space:pre-wrap;font-size:11px'>{tb}</pre></details>", "error")

# =============================================================================
# STEP 3 — Model loading (only ONE large pipeline on the GPU at a time)
# =============================================================================
def _load_sd_from_disk_or_hub():
    try:
        return StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=DTYPE,
                                                       variant="fp16")
    except (OSError, ValueError):
        return StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=DTYPE)


def _finish_pipeline(pipe, scheduler):
    pipe.scheduler = scheduler
    pipe.vae.enable_slicing()          # decode frames in slices to save VRAM
    pipe.set_progress_bar_config(disable=True)
    return pipe


def get_sd_pipeline():
    if STATE["kind"] == "sd":
        return STATE["pipe"]
    release_pipeline("switching to Stable Diffusion for Part 1")
    print(f"⏳ Loading Stable Diffusion 1.5 ({BASE_MODEL}) — first run downloads ~2 GB…")
    t0 = time.perf_counter()
    sd = _load_sd_from_disk_or_hub().to("cuda")
    # Still images keep SD 1.5's own (scaled_linear) noise schedule, with the DDIM sampler.
    _finish_pipeline(sd, DDIMScheduler.from_config(sd.scheduler.config))
    STATE["pipe"], STATE["kind"] = sd, "sd"
    print(f"✅ Stable Diffusion ready in {time.perf_counter() - t0:.0f} s · GPU {gpu_memory()}")
    return sd


def get_animatediff_pipeline():
    if STATE["kind"] == "animatediff":
        return STATE["pipe"]
    print(f"⏳ Loading motion adapter ({MOTION_ADAPTER})…")
    t0 = time.perf_counter()
    adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER, torch_dtype=DTYPE)
    if STATE["kind"] == "sd":
        # Reuse the already-loaded SD parts (VAE, text encoder) instead of loading twice.
        sd = STATE["pipe"]
        pipe = AnimateDiffPipeline.from_pipe(sd, motion_adapter=adapter, torch_dtype=DTYPE)
        # Drop every reference to the old SD pipeline (and its 2D UNet) before moving
        # the new motion UNet onto the GPU, so two big UNets never share the GPU.
        STATE["pipe"], STATE["kind"] = None, None
        del sd
        free_gpu()
    else:
        release_pipeline()
        try:
            pipe = AnimateDiffPipeline.from_pretrained(BASE_MODEL, motion_adapter=adapter,
                                                       torch_dtype=DTYPE, variant="fp16")
        except (OSError, ValueError):
            pipe = AnimateDiffPipeline.from_pretrained(BASE_MODEL, motion_adapter=adapter,
                                                       torch_dtype=DTYPE)
    pipe = pipe.to("cuda")
    # Settings recommended for the AnimateDiff v1-5-2 motion module.
    _finish_pipeline(pipe, DDIMScheduler.from_pretrained(
        BASE_MODEL, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace",
        beta_schedule="linear", steps_offset=1))
    STATE["pipe"], STATE["kind"] = pipe, "animatediff"
    print(f"✅ AnimateDiff ready in {time.perf_counter() - t0:.0f} s · GPU {gpu_memory()}")
    return pipe

# =============================================================================
# STEP 4 — Generation and saving
# =============================================================================
def generate_independent(prompt, frames, steps, seed, on_step=None):
    """Part 1: every frame is its own image with its own seed (seed, seed+1, …)."""
    pipe = get_sd_pipeline()
    images, seeds, flagged = [], [], 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(frames):
            s = seed + i
            out = pipe(prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
                       num_inference_steps=steps, guidance_scale=GUIDANCE,
                       width=WIDTH, height=HEIGHT,
                       generator=torch.Generator("cpu").manual_seed(s),
                       callback_on_step_end=step_callback(on_step, offset=i * steps))
            images.append(out.images[0])
            seeds.append(s)
            flagged += bool(out.nsfw_content_detected and out.nsfw_content_detected[0])
            del out
    return images, seeds, time.perf_counter() - t0, flagged


def generate_animatediff(prompt, frames, steps, seed, on_step=None):
    """Part 2/3: AnimateDiff denoises all frames together with ONE seed."""
    pipe = get_animatediff_pipeline()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = pipe(prompt=prompt, negative_prompt=NEGATIVE_PROMPT, num_frames=frames,
                   num_inference_steps=steps, guidance_scale=GUIDANCE,
                   width=WIDTH, height=HEIGHT,
                   generator=torch.Generator("cpu").manual_seed(seed),
                   decode_chunk_size=4,
                   callback_on_step_end=step_callback(on_step))
    images = out.frames[0]
    del out
    return images, time.perf_counter() - t0


def save_clip(images, kind, title, prompt, steps, seed, seconds, seeds=None):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = OUT_DIR / f"{kind}_seed{seed}_{len(images)}f_{steps}s_{stamp}.mp4"
    export_to_video(images, str(path), fps=FPS)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Video export produced no file at {path}")
    clip = {
        "title": title, "kind": kind, "label": "AI-generated",
        "file": str(path), "prompt": prompt, "negative_prompt": NEGATIVE_PROMPT,
        "seed": seed, "seeds": seeds, "frames": len(images), "steps": steps,
        "guidance": GUIDANCE, "width": WIDTH, "height": HEIGHT, "fps": FPS,
        "seconds": round(seconds, 1), "created": stamp,
        "models": [BASE_MODEL] + ([MOTION_ADAPTER] if kind == "animatediff" else []),
        "gpu": GPU_NAME, "torch": torch.__version__, "diffusers": diffusers.__version__,
        "_gif": gif_base64(images),
    }
    STATE["clips"].append(clip)
    write_metadata()
    return clip


def clip_metadata():
    return [{k: v for k, v in c.items() if not k.startswith("_")} for c in STATE["clips"]]


def write_metadata():
    note = ("All clips are AI-generated. Exact reproducibility across GPUs or "
            "library versions is not guaranteed.")
    (OUT_DIR / "metadata.json").write_text(json.dumps(
        {"project": "ENGR-126 Project 15a", "note": note, "clips": clip_metadata()}, indent=2))


def build_zip():
    """Rebuild project15a_videos.zip from scratch. Returns (path, clip count) or (None, 0)."""
    clips = [c for c in STATE["clips"]
             if Path(c["file"]).exists() and Path(c["file"]).stat().st_size > 0]
    if not clips:
        return None, 0
    write_metadata()
    tmp = ZIP_PATH.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for c in clips:
            zf.write(c["file"], arcname=Path(c["file"]).name)
        zf.write(OUT_DIR / "metadata.json", arcname="metadata.json")
        zf.writestr("README.txt", "ENGR-126 Project 15a — AI-generated video clips.\n"
                    "Made with Stable Diffusion 1.5 and AnimateDiff. Prompts, seeds, frame "
                    "counts and step counts are in metadata.json.\n")
    os.replace(tmp, ZIP_PATH)
    return ZIP_PATH, len(clips)

# =============================================================================
# PART 1 — Independent images → flicker
# =============================================================================
part1_clip = part2_clip = None
SETTINGS_OK = not validate(PROMPT, FRAMES, STEPS, SEED)
if not SETTINGS_OK:
    box("⚠️ Please fix the settings at the top of the cell (Parts 1–2 are skipped, and the "
        "panel starts at 16 frames / 25 steps instead):<br>• " +
        "<br>• ".join(html.escape(p) for p in validate(PROMPT, FRAMES, STEPS, SEED)), "warn")

if SHOW_FLICKER and GPU_OK and SETTINGS_OK:
    header("Part 1 — Independent Stable Diffusion images (expect flicker)")
    key = ("independent", PROMPT, FRAMES, STEPS, SEED, NEGATIVE_PROMPT, GUIDANCE, WIDTH, HEIGHT)
    part1_clip = STATE["results"].get(key)
    if part1_clip and Path(part1_clip["file"]).exists():
        box("♻️ Same settings as an earlier run: showing that result instead of regenerating. "
            "Change SEED/PROMPT/FRAMES/STEPS to make a new one.")
    else:
        print(f"Generating {FRAMES} separate images for: “{PROMPT}” "
              f"(seeds {SEED}–{SEED + FRAMES - 1}, {STEPS} steps each)")
        if FRAMES >= 24:
            print("ℹ️  24 frames is the slowest setting.")
        STATE["busy"] = True
        try:
            get_sd_pipeline()   # load first so its messages print above the progress bar
            imgs, seeds, secs, flagged = generate_independent(
                PROMPT, FRAMES, STEPS, SEED, PageProgress("Denoising steps", FRAMES * STEPS))
            part1_clip = save_clip(imgs, "independent", "Part 1 · independent SD images",
                                   PROMPT, STEPS, SEED, secs, seeds=seeds)
            idx = pick_evenly(imgs)
            part1_clip["_grid"] = frame_grid([imgs[i] for i in idx],
                                             [f"frame {i} · seed {seeds[i]}" for i in idx])
            STATE["results"][key] = part1_clip
            if flagged:
                box(f"⚠️ The safety checker blacked out {flagged} frame(s). Try a different "
                    "prompt or seed.", "warn")
            STATE["busy"] = False
        except Exception as err:
            part1_clip = None
            show_error(err, "Part 1")
    if part1_clip:
        print(f"Representative frames (each from a different seed) — "
              f"{part1_clip['seconds']:.0f} s total, "
              f"{part1_clip['seconds'] / part1_clip['frames']:.1f} s per image:")
        display(part1_clip["_grid"])
        display(HTML(video_card(part1_clip)))
        box("👀 Each frame was drawn from scratch with a different seed, so shapes, colors and "
            "layout jump from frame to frame. That jumping is the <b>flicker</b>.")

# =============================================================================
# PART 2 — AnimateDiff (frames generated together)
# =============================================================================
if SHOW_ANIMATEDIFF and GPU_OK and SETTINGS_OK:
    header("Part 2 — AnimateDiff (all frames generated together)")
    key = ("animatediff", PROMPT, FRAMES, STEPS, SEED, NEGATIVE_PROMPT, GUIDANCE, WIDTH, HEIGHT)
    part2_clip = STATE["results"].get(key)
    if part2_clip and Path(part2_clip["file"]).exists():
        box("♻️ Same settings as an earlier run: showing that result instead of regenerating.")
    else:
        print(f"Generating a {FRAMES}-frame AnimateDiff clip for: “{PROMPT}” "
              f"(seed {SEED}, {STEPS} steps)")
        STATE["busy"] = True
        try:
            get_animatediff_pipeline()
            imgs, secs = generate_animatediff(PROMPT, FRAMES, STEPS, SEED,
                                              PageProgress("Denoising steps", STEPS))
            part2_clip = save_clip(imgs, "animatediff", "Part 2 · AnimateDiff",
                                   PROMPT, STEPS, SEED, secs)
            idx = pick_evenly(imgs)
            part2_clip["_grid"] = frame_grid([imgs[i] for i in idx], [f"frame {i}" for i in idx])
            STATE["results"][key] = part2_clip
            STATE["busy"] = False
        except Exception as err:
            part2_clip = None
            show_error(err, "Part 2")
    if part2_clip:
        print(f"AnimateDiff frames — seed {part2_clip['seed']}, "
              f"generated in {part2_clip['seconds']:.0f} s:")
        display(part2_clip["_grid"])
        display(HTML(video_card(part2_clip)))
        box("👀 The motion module denoises every frame <i>together</i>, so the scene usually "
            "stays consistent from frame to frame. Smoother does <b>not</b> mean perfect: look "
            "for warping, melting details or motion that makes no physical sense, and record "
            "what you actually see in the experiment log.")

if part1_clip and part2_clip:
    header("Side-by-side comparison")
    display(HTML(video_card(part1_clip) + video_card(part2_clip)))

# =============================================================================
# PART 3 — Interactive prompt-to-video panel
# =============================================================================
if SHOW_PANEL and GPU_OK:
    header("Part 3 — Make your own clip")
    if widgets is None:
        box("❌ ipywidgets is not available, so the panel can't be shown. Clips from Parts 1–2 "
            f"are saved in <code>{OUT_DIR}</code>.", "error")
    else:
        prompt_box = widgets.Textarea(value=PROMPT, description="Prompt:",
                                      placeholder="e.g. ocean waves crashing on rocks at sunset",
                                      layout=widgets.Layout(width="95%", height="60px"))
        frames_dd = widgets.Dropdown(options=sorted({8, 12, 16, 24, FRAMES} & set(range(8, 25))),
                                     value=FRAMES if 8 <= FRAMES <= 24 else 16,
                                     description="Frames:")
        steps_dd = widgets.Dropdown(options=sorted({10, 25, 40, STEPS} & set(range(10, 51))),
                                    value=STEPS if 10 <= STEPS <= 50 else 25,
                                    description="Steps:")
        make_btn = widgets.Button(description="Make clip", button_style="primary", icon="film")
        zip_btn = widgets.Button(description="Download all", button_style="success",
                                 icon="download")
        progress = widgets.IntProgress(value=0, min=0, max=1, description="Progress:")
        status = widgets.HTML(value="Ready. A new random seed is used for every clip.")
        clips_out = widgets.Output()

        def on_make(_):
            prompt, frames, steps = prompt_box.value.strip(), frames_dd.value, steps_dd.value
            problems = validate(prompt, frames, steps)
            if problems:
                status.value = "⚠️ " + " ".join(html.escape(p) for p in problems)
                return
            seed = secrets.randbelow(2**31 - 1)
            make_btn.disabled = True
            progress.max, progress.value = steps, 0
            status.value = (f"⏳ Generating {frames} frames × {steps} steps (seed {seed})… "
                            "loading the model the first time can take a minute.")
            STATE["busy"] = True
            with clips_out:   # loading messages and errors appear under the panel
                try:
                    def on_step(done):
                        progress.value = done
                    imgs, secs = generate_animatediff(prompt, frames, steps, seed, on_step)
                    clip = save_clip(imgs, "animatediff", "Panel clip", prompt, steps, seed, secs)
                    STATE["busy"] = False
                    display(HTML(video_card(clip)))
                    status.value = (f"✅ Done in {secs:.0f} s · seed {seed} · saved "
                                    f"<code>{html.escape(Path(clip['file']).name)}</code>")
                except Exception as err:
                    status.value = "❌ Generation failed — details below. Earlier clips are kept."
                    show_error(err, "Make clip")
                finally:
                    make_btn.disabled = False

        def on_zip(_):
            try:
                path, n = build_zip()
            except Exception as err:
                status.value = "❌ Could not build the ZIP — details below."
                with clips_out:
                    show_error(err, "Download all")
                return
            if path is None:
                status.value = "ℹ️ No finished clips yet — make one first."
                return
            status.value = (f"📦 {path.name}: {n} clip(s) + metadata.json "
                            f"({path.stat().st_size / 1e6:.1f} MB) at <code>{path}</code>")
            if colab_files is not None:
                colab_files.download(str(path))
            else:
                from IPython.display import FileLink
                with clips_out:
                    display(FileLink(str(path)))

        make_btn.on_click(on_make)
        zip_btn.on_click(on_zip)
        display(widgets.VBox([prompt_box, widgets.HBox([frames_dd, steps_dd]),
                              widgets.HBox([make_btn, zip_btn]), progress, status, clips_out]))
        print("Tips: moving subjects (waves, a flag, a car) show motion best. Use fictional "
              "scenes and ordinary objects — no real people or copyrighted characters.")

if not (SHOW_FLICKER or SHOW_ANIMATEDIFF or SHOW_PANEL):
    box("All three switches are False, so nothing was generated. Set at least one to True.",
        "warn")
if GPU_OK:
    print(f"\nGPU memory now: {gpu_memory()} · loaded pipeline: {STATE['kind'] or 'none'} · "
          f"clips saved this session: {len(STATE['clips'])}")
