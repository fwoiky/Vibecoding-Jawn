# Nitro Arena

**Nitro Arena** is an original 2D rocket-car soccer game written in Python.
Drive a rocket-powered car around a neon indoor stadium, jump, boost, fly
and smash a giant ball into the opponent's goal before the clock runs out.

It is inspired by the *gameplay idea* of side-view car soccer games. All
names, graphics, sounds and code are original. Every graphic is drawn in code
and every sound is synthesised in code, so there are no image or audio files
to go missing.

Features:

- Arcade car physics: driving, braking/reverse, jumping, double jumps, flips,
  air roll, boost, and driving up walls and across the ceiling
- A physics-based ball with bounces, spin, drag, and a speed cap
- A computer opponent with **Easy / Normal / Hard** difficulty (state-based
  AI: defend, chase, attack, recover; reaction delay; aerials and flips)
- 1-5 minute matches, countdown kickoffs, goal celebrations with slow motion,
  **overtime** (next goal wins) and a results screen
- Main menu with a live AI-vs-AI match in the background, How to Play,
  Settings, Graphics settings, and a pause menu
- A **GPU renderer** (OpenGL shaders): ambient occlusion, glossy floor
  reflections, HDR bloom, light shafts, motion blur, FXAA, chromatic
  aberration, colour grading, and vignette. Quality presets go from Low to
  Ultra, and every effect can be set individually. There is an automatic CPU
  fallback.
- Particles (boost trail, sparks, dust, goal explosions, confetti), screen
  shake, impact rings
- Procedural sound effects and music
- Saved settings and career statistics (wins, losses, goals)
- Keyboard controls, plus optional gamepad support

---

## 1. Requirements

- macOS (Windows and Linux also work)
- **Python 3.9 or newer** (3.11+ recommended). Check with `python3 --version`.
  If you don't have it, install it from <https://www.python.org/downloads/>.
- Python packages (installed in step 2):
  - `pygame` for the window, input, sound and 2D drawing
  - `moderngl` for the GPU effects (optional: without it the game uses the
    CPU renderer automatically)

## 2. Install

Open **Terminal**, go to this folder, and install the dependencies:

```bash
cd path/to/Vibecoding-Jawn/nitro_arena
python3 -m pip install -r requirements.txt
```

(Using a virtual environment is recommended but optional:
`python3 -m venv .venv && source .venv/bin/activate` before the pip command.)

## 3. Run

```bash
python3 main.py
```

### Running from PyCharm

1. **File -> Open...** and choose the `nitro_arena` folder.
2. Set up a Python interpreter if PyCharm asks (**Settings -> Project ->
   Python Interpreter**). PyCharm will offer to install `requirements.txt`;
   accept, or run the pip command above in PyCharm's Terminal tab.
3. Right-click `main.py` -> **Run 'main'**.

## 4. Controls

| Key | Action |
|-----|--------|
| **A / D** or **Left / Right** | Drive left / right (press the other way to brake and turn around) |
| **W** or **Up** | Jump. Press again in the air for a double jump |
| **W + A/D** in the air | Flip (dodge) in that direction: the strongest way to hit the ball |
| **S** or **Down** | Brake / reverse on the ground, dive in the air |
| **Space** (or Right Shift) | Boost: pushes the car in the direction its nose points |
| **Q / E** | Air roll: rotate the car counter-clockwise / clockwise in the air |
| **Esc** (or P) | Pause |

Menus: arrow keys / WASD and Enter, the mouse, or a gamepad.

**Gamepad (optional):** left stick or D-pad to drive, A = jump, B/X = boost,
LB/RB = air roll, Start = pause.

**Tips**

- *Flying:* jump, tilt the nose up with Q/E, then hold Space.
- Fast cars stick to the walls and the ceiling. Slow cars fall off.
- Holding jump longer gives a higher jump.
- A car on its roof flips itself back onto its wheels after a moment.

## 5. Game rules

- Score in the **orange goal on the right**. Defend the **blue goal on the
  left**.
- A goal counts when the whole ball crosses the goal line.
- Matches last 2 minutes by default (1, 2, 3 or 5 minutes in Settings).
- If the score is tied when time runs out, **overtime** starts: the next goal
  wins.
- Boost: you have up to 100. It drains while boosting and slowly refills when
  you are not boosting.

## 6. Settings

**Settings:** music volume, sound effects volume, fullscreen, screen shake,
match length, reset statistics.

**Graphics settings:**

| Option | What it does |
|--------|--------------|
| Renderer | GPU (OpenGL shaders) or CPU (plain pygame). Applies after a restart |
| Quality preset | LOW / MEDIUM / HIGH / ULTRA. Changing any option below switches to CUSTOM |
| Reflections | Glossy planar floor reflections that blur with distance (mip-mapped roughness, Fresnel falloff) |
| Ambient occlusion | Screen-space AO: soft contact shadows around cars, ball, walls and corners |
| Bloom | HDR-style glow from neon, flames and explosions (dual-filter mip chain) |
| Light shafts | Volumetric-looking god rays from the ceiling lights |
| Motion blur | Frame blending for a sense of speed |
| Anti-aliasing | FXAA edge smoothing |
| Chromatic aberration | Lens colour fringing that spikes on big impacts |
| Vignette / Color grading | Darker corners; filmic tone curve and richer colours |
| Particles | Amount of particle effects |
| Frame rate cap | 30 / 60 / 120 / 144 / unlimited |
| Show FPS | Frame-rate counter |

If a MacBook Air runs hot or drops frames, choose **MEDIUM** or **LOW**.
**ULTRA** is meant for Apple Silicon Pro/Max chips and dedicated GPUs.

Settings and statistics are saved in `save_data.json` next to `main.py`.
Delete that file to reset everything.

## 7. Project structure

```
nitro_arena/
  main.py          Entry point (run this)
  game.py          Game class: window, main loop, state machine
  settings.py      ALL tunable constants: physics, AI, colours, graphics presets
  match.py         One match: cars, ball, AI, scoring, effects, fixed-step physics
  car.py           Car physics (driving, jumping, flips, boost, walls) and drawing
  ball.py          Ball physics, car-ball impulses, drawing
  physics.py       Reusable collision helpers and ball trajectory prediction
  arena.py         Arena shape, collision queries, goal detection, pre-rendered visuals
  opponent.py      Computer opponent (state-based AI)
  player.py        Keyboard / gamepad input -> Controls
  controls.py      The Controls data class shared by player and AI
  renderer.py      GPU (OpenGL) and CPU renderers
  shaders.py       GLSL shaders: AO, reflections, bloom, light shafts, grading, FXAA
  graphics.py      Procedural drawing helpers (gradients, glows, shadows)
  particles.py     Particle system
  hud.py           Scoreboard, timer, boost meters, big messages
  ui.py            Fonts, buttons and menus
  screens.py       Main menu, How to Play, Settings, Graphics, Pause, Game Over
  audio.py         Procedural sound effects and music, audio manager
  save_data.py     Settings and statistics (JSON)
  assets/          Optional sound overrides (see assets/README.md)
  tests/           Automated tests
```

Game states (in `game.py`): `MAIN_MENU`, `HOW_TO_PLAY`, `SETTINGS`,
`GRAPHICS`, `COUNTDOWN`, `PLAYING`, `GOAL`, `PAUSED`, `GAME_OVER`.

## 8. Modifying the physics

Every number that affects how the game feels is in **`settings.py`**, grouped
and commented. Some good ones to experiment with:

| Constant | Effect |
|----------|--------|
| `GRAVITY`, `BALL_GRAVITY` | How floaty cars / the ball are |
| `CAR_ACCEL`, `CAR_MAX_SPEED` | Acceleration and top speed on the ground |
| `JUMP_VELOCITY`, `DOUBLE_JUMP_VELOCITY` | Jump heights |
| `BOOST_ACCEL`, `BOOST_MAX_SPEED`, `BOOST_DRAIN`, `BOOST_REGEN` | Boost power and economy |
| `AIR_ROT_SPEED` | Air roll speed |
| `BALL_BOUNCE`, `BALL_DRAG`, `BALL_MAX_SPEED` | Ball liveliness |
| `HIT_POWER_BONUS`, `HIT_LIFT`, `DODGE_HIT_BONUS` | How hard hits are and how high the ball goes |
| `WALL_MIN_SPEED` | Speed needed to stick to walls |
| `AI_DIFFICULTY` | Reaction time, accuracy, and skills per difficulty |
| `GOAL_HEIGHT`, `ARENA_*` | Arena and goal size |

Physics runs at a fixed 120 steps per second (`PHYSICS_HZ`), so it behaves
the same at any frame rate.

## 9. Running the tests

```bash
python3 -m unittest discover -s tests -v
```

The tests run without opening a window. They check car and ball physics,
goals, save data, AI-vs-AI stability, and a full game flow (menus, match,
pause, goal, overtime, game over).

## 10. Known limitations

- The camera is fixed: the whole arena is always visible.
- One player versus the computer only (no local or online multiplayer).
- Physics is intentionally arcade-style, not a simulation. Cars use a capsule
  shape against the arena and a box against the ball.
- The GPU effects are screen-space 2D techniques (the game is 2D). "Ambient
  occlusion" and "reflections" are computed from the 2D scene, not a 3D world.
- Graphics look the same on the CPU renderer, minus the GPU effects (it keeps
  a simple floor reflection).
- Switching between the GPU and CPU renderer needs a restart.
- Gamepad button numbers differ between controllers. Unusual pads may need the
  mapping at the top of `player.py` adjusted.
- If OpenGL cannot start (very old hardware, remote desktop, missing
  `moderngl`), the game prints a message in the console and uses the CPU
  renderer.
