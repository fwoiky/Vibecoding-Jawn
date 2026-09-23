# Vibecoding-Jawn

## Projects

- **[Nitro Arena](nitro_arena/)**: a 2D rocket-car soccer game in Python/Pygame with an
  OpenGL post-processing renderer (ambient occlusion, reflections, bloom, and more).
  See [nitro_arena/README.md](nitro_arena/README.md).

  **Easiest way to play (no Terminal, no git):** download
  [`nitro_arena_onefile.py`](nitro_arena_onefile.py), open it in PyCharm and press Run.
  It unpacks the game, installs pygame into PyCharm's Python and starts the game.
  (Rebuild it after code changes with `python3 tools/build_onefile.py`.)

  Quick start from a terminal:

  ```bash
  cd nitro_arena
  python3 -m pip install -r requirements.txt
  python3 main.py
  ```
