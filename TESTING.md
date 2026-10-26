# Project 15a — Testing checklist & experiment log

## What has and hasn't been verified

| Item | Status |
|---|---|
| Python syntax (`py_compile`) | ✅ Verified |
| Diffusers APIs used (`AnimateDiffPipeline.from_pipe(..., motion_adapter=, torch_dtype=)`, `UNetMotionModel` conversion, `DDIMScheduler` arguments, `vae.enable_slicing()`, `callback_on_step_end`, `decode_chunk_size`, `export_to_video`) | ✅ Checked against the Diffusers 0.42.0 source |
| Minimum Diffusers version | ✅ Set to 0.31 — 0.30's `export_to_video` writes OpenCV `mp4v`, which browsers won't play inline |
| Control flow with **mocked** models (switches, settings-reuse on re-run, SD → AnimateDiff transition with only one pipeline "on GPU", reload of SD on re-run, invalid settings, no-GPU message, panel Make clip / Download all, OOM handling keeps earlier clips, empty ZIP refused) | ✅ Verified in a dry run with fake `torch`/`diffusers` |
| MP4 encoding | ✅ Real `imageio-ffmpeg` output: H.264 (High), yuv420p, 8 fps — browser-playable |
| ZIP contents | ✅ MP4s + `metadata.json` + `README.txt` |
| Real model download, GPU memory use, timing, image/video quality on a Colab T4 | ⚠️ **Not verified — must be run in Colab** |
| Widget rendering and `files.download` inside Colab | ⚠️ **Not verified — must be run in Colab** |

## Testing checklist (fill in after running in Colab)

- [ ] GPU status is detected and displayed.
- [ ] `SHOW_FLICKER = False` skips Part 1 (no SD-only loading).
- [ ] `SHOW_ANIMATEDIFF = False` skips Part 2.
- [ ] `SHOW_PANEL = False` hides the panel.
- [ ] Part 1 generates independent images and a playable looping video.
- [ ] Part 2 generates an AnimateDiff clip and displays elapsed time.
- [ ] Prompt and seed information are visible on every clip.
- [ ] Interactive panel generates a new clip.
- [ ] Completed clips play inside the notebook.
- [ ] MP4 files are saved in `/content/project15a_outputs/`.
- [ ] Download all creates a valid `project15a_videos.zip`.
- [ ] Memory cleanup is handled between model pipelines (GPU free memory line after Part 2).
- [ ] A generation failure (e.g. 24 frames × 40 steps if it runs out of memory) shows a useful error and preserves completed files.
- [ ] Demo and troubleshooting instructions are ready.

## Experiment log

Record what you actually see — don't assume AnimateDiff will always look better.

| Experiment | What changed | What I saw | Time (s) | Best value |
|---|---|---|---|---|
| Part 1 vs Part 2 | Same prompt; Part 1 seeds 42–57, Part 2 seed 42 | | | |
| Frame count | 8 | | | |
| Frame count | 16 | | | |
| Frame count | 24 | | | |
| Inference steps | 10 | | | |
| Inference steps | 25 | | | |
| Inference steps | 40 | | | |
| Seed | Same prompt, different seed | | | |
| Moving prompt | Waves / flag / car | | | |
| Tricky prompt | "A cat jumping onto a horse" | | | |
