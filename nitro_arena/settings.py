"""
Central configuration for Nitro Arena.

Every tunable number in the game lives here so you can tweak the feel of the
game without digging through the code. Units:
    distances  -> pixels (in the 1280x720 internal resolution)
    speeds     -> pixels per second
    accels     -> pixels per second squared
    angles     -> radians (angular speeds in radians per second)
    times      -> seconds
"""
import os

# ---------------------------------------------------------------------------
# General / window
# ---------------------------------------------------------------------------
GAME_TITLE = "NITRO ARENA"
GAME_SUBTITLE = "ROCKET CAR SOCCER"

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60

# Physics runs at a fixed rate that is independent of the frame rate.
PHYSICS_HZ = 120
PHYSICS_DT = 1.0 / PHYSICS_HZ
MAX_PHYSICS_STEPS_PER_FRAME = 8
MAX_FRAME_TIME = 0.1  # clamp huge frame times (e.g. after dragging the window)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
SOUNDS_DIR = os.path.join(ASSETS_DIR, "sounds")
SAVE_FILE = os.path.join(BASE_DIR, "save_data.json")

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BG_TOP = (6, 8, 22)
BG_BOTTOM = (20, 10, 42)
ARENA_BACK_TOP = (14, 20, 48)
ARENA_BACK_BOTTOM = (24, 30, 70)
ARENA_EDGE = (110, 210, 255)
ARENA_STRUCTURE = (10, 12, 30)
FLOOR_TOP = (132, 150, 196)
FLOOR_BOTTOM = (22, 26, 56)
FLOOR_LINE = (140, 240, 255)

PLAYER_COLOR = (30, 190, 255)
PLAYER_ACCENT = (210, 250, 255)
PLAYER_DARK = (10, 70, 130)
CPU_COLOR = (255, 110, 40)
CPU_ACCENT = (255, 235, 200)
CPU_DARK = (130, 40, 10)

BALL_COLOR = (246, 246, 252)
BALL_GLOW = (255, 230, 120)
BOOST_COLOR = (255, 190, 40)
BOOST_FLAME_INNER = (255, 250, 210)
BOOST_FLAME_OUTER = (255, 120, 30)
DUST_COLOR = (190, 200, 225)

TEXT_COLOR = (236, 242, 255)
TEXT_DIM = (140, 152, 190)
ACCENT = (255, 60, 170)
GOLD = (255, 214, 90)
PANEL_COLOR = (14, 18, 40)
PANEL_BORDER = (60, 80, 140)

# ---------------------------------------------------------------------------
# Arena layout (all in internal pixels)
# ---------------------------------------------------------------------------
ARENA_LEFT = 100        # x of the left wall / left goal line
ARENA_RIGHT = 1180      # x of the right wall / right goal line
ARENA_TOP = 100         # y of the ceiling
ARENA_FLOOR = 640       # y of the floor
GOAL_HEIGHT = 185       # height of the goal mouth
GOAL_DEPTH = 82         # how deep the goal box goes behind the goal line
CORNER_RADIUS = 110     # rounded ceiling corners (cars can drive around them)
CORNER_SEGMENTS = 10    # how many straight pieces approximate each rounded corner

ARENA_CENTER_X = (ARENA_LEFT + ARENA_RIGHT) / 2
CROSSBAR_Y = ARENA_FLOOR - GOAL_HEIGHT

# ---------------------------------------------------------------------------
# Car physics
# ---------------------------------------------------------------------------
GRAVITY = 1650.0

CAR_CAPSULE_HALF = 22.0     # distance from car centre to each "wheel" collision circle
CAR_RADIUS = 13.0           # radius of the collision capsule (half the car height)
CAR_HALF_LENGTH = 36.0      # hit box used against the ball
CAR_HALF_HEIGHT = 14.0

CAR_ACCEL = 1700.0
CAR_MAX_SPEED = 580.0
CAR_BRAKE = 2600.0
CAR_COAST_DECEL = 650.0
CAR_REVERSE_ACCEL = 1100.0
CAR_REVERSE_MAX_SPEED = 330.0
CAR_TURN_SPEED = 150.0          # below this speed, pressing the other way turns the car around
CAR_OVERSPEED_DECEL = 450.0     # slows a car that is faster than CAR_MAX_SPEED without boosting
CAR_MAX_TOTAL_SPEED = 1300.0
CAR_BOUNCE = 0.1
CAR_ALIGN_RATE = 18.0           # how quickly a grounded car lines up with the surface
CAR_TIP_RATE = 7.0              # how quickly a car on its side tips back onto its wheels
CAR_ROOF_FLIP_DELAY = 0.25      # time on the roof before the car flips itself upright
CAR_GROUND_PROBE = 4.0          # extra distance used to detect ground contact

WALL_MIN_SPEED = 260.0          # minimum speed needed to stick to walls / ceiling
WALL_STICK_ACCEL = 1900.0
WALL_DETACH_ACCEL = 700.0

JUMP_VELOCITY = 600.0
JUMP_HOLD_TIME = 0.15           # holding jump for this long gives a higher jump
JUMP_HOLD_ACCEL = 950.0
DOUBLE_JUMP_VELOCITY = 500.0
DOUBLE_JUMP_WINDOW = 1.6        # second jump must happen within this time in the air
DODGE_IMPULSE = 560.0           # double jump + direction = a flip that launches the car
DODGE_MAX_SPEED = 1000.0
DODGE_LIFT = 160.0
DODGE_DURATION = 0.42
GROUND_COOLDOWN_AFTER_JUMP = 0.1

AIR_ROT_SPEED = 6.0
AIR_ROT_ACCEL = 40.0
AIR_ROT_DAMP = 14.0
AIR_CONTROL_ACCEL = 520.0
AIR_CONTROL_MAX_SPEED = 520.0
AIR_DIVE_ACCEL = 900.0

# ---------------------------------------------------------------------------
# Boost
# ---------------------------------------------------------------------------
BOOST_MAX = 100.0
BOOST_START = 50.0
BOOST_DRAIN = 34.0              # per second while boosting
BOOST_REGEN = 11.0              # per second while not boosting
BOOST_REGEN_DELAY = 0.6
BOOST_ACCEL = 2300.0            # in the air (must beat gravity so aerials work)
BOOST_ACCEL_GROUND = 1900.0
BOOST_MAX_SPEED = 1050.0

# ---------------------------------------------------------------------------
# Ball physics
# ---------------------------------------------------------------------------
BALL_RADIUS = 28.0
BALL_GRAVITY = 950.0
BALL_BOUNCE = 0.74
BALL_DRAG = 0.10                # fraction of speed lost per second in the air
BALL_ROLL_FRICTION = 140.0
BALL_SURFACE_FRICTION = 0.06    # tangential speed lost on each bounce
BALL_MAX_SPEED = 1750.0
BALL_MIN_BOUNCE_SPEED = 45.0    # slower impacts do not bounce (prevents jitter)
BALL_KICKOFF_HEIGHT = 220.0     # the ball drops from this height at kickoff

# ---------------------------------------------------------------------------
# Car <-> ball / car <-> car hits
# ---------------------------------------------------------------------------
HIT_RESTITUTION = 0.35          # bounciness of the ball off a car
HIT_POWER_BONUS = 0.25          # arcade extra: extra ball speed from the car driving into it
CAR_HIT_RECOIL = 0.18           # how much of the hit pushes the car back
HIT_LIFT = 0.32                 # arcade extra: hits push the ball slightly upwards
DODGE_HIT_BONUS = 260.0         # flips hit harder
STRONG_HIT_SPEED = 650.0        # impacts above this speed trigger big effects
CAR_CAR_RESTITUTION = 0.4
CAR_CAR_BUMP = 120.0

# ---------------------------------------------------------------------------
# Match rules
# ---------------------------------------------------------------------------
MATCH_LENGTH_OPTIONS = [1, 2, 3, 5]     # minutes
DEFAULT_MATCH_MINUTES = 2
KICKOFF_STEP_TIME = 0.7                 # each of "3", "2", "1"
GOAL_CELEBRATION_TIME = 2.3
GOAL_SLOWMO_TIME = 0.7
GOAL_SLOWMO_SCALE = 0.3
KICKOFF_CAR_OFFSET = 370                # distance of each car from the centre at kickoff
CONTROLS_HINT_TIME = 9.0

# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------
MAX_PARTICLES = 450
SCREEN_SHAKE_MAX = 9.0
SCREEN_SHAKE_DECAY = 26.0

# ---------------------------------------------------------------------------
# AI difficulty. All levels use exactly the same car physics as the player;
# they only differ in how quickly and how well they make decisions.
# ---------------------------------------------------------------------------
DIFFICULTIES = ["EASY", "NORMAL", "HARD"]
DEFAULT_DIFFICULTY = "NORMAL"
AI_DIFFICULTY = {
    "EASY": {
        "reaction": 0.30,          # seconds of delay before it notices where the ball is
        "decision_interval": 0.24,
        "max_throttle": 0.78,      # never presses the pedal all the way
        "aim_error": 42.0,         # random error in where it tries to hit the ball
        "boost_chance": 0.25,
        "jump_skill": 0.55,        # chance of attempting a jump at a high ball
        "aerial_chance": 0.0,
        "dodge_chance": 0.10,
        "prediction_time": 0.6,
        "air_recovery": False,
        "defend_bias": 0.35,
    },
    "NORMAL": {
        "reaction": 0.16,
        "decision_interval": 0.12,
        "max_throttle": 0.93,
        "aim_error": 22.0,
        "boost_chance": 0.6,
        "jump_skill": 0.85,
        "aerial_chance": 0.3,
        "dodge_chance": 0.35,
        "prediction_time": 1.2,
        "air_recovery": True,
        "defend_bias": 0.5,
    },
    "HARD": {
        "reaction": 0.07,
        "decision_interval": 0.06,
        "max_throttle": 0.95,
        "aim_error": 8.0,
        "boost_chance": 0.65,
        "jump_skill": 1.0,
        "aerial_chance": 0.75,
        "dodge_chance": 0.7,
        "prediction_time": 1.8,
        "air_recovery": True,
        "defend_bias": 0.7,
    },
}

# ---------------------------------------------------------------------------
# Graphics quality. The GPU renderer (OpenGL via moderngl) runs these as
# shader passes; the CPU renderer supports only reflections.
# Numbers are levels: 0 = off, higher = better quality / heavier GPU load.
# ---------------------------------------------------------------------------
RENDERERS = ["GPU", "CPU"]
GRAPHICS_OPTIONS = {
    # key: (menu label, level names)
    "reflections": ("REFLECTIONS", ["OFF", "LOW", "HIGH", "ULTRA"]),
    "ambient_occlusion": ("AMBIENT OCCLUSION", ["OFF", "LOW", "HIGH", "ULTRA"]),
    "bloom": ("BLOOM", ["OFF", "LOW", "MEDIUM", "HIGH"]),
    "light_shafts": ("LIGHT SHAFTS", ["OFF", "LOW", "HIGH"]),
    "motion_blur": ("MOTION BLUR", ["OFF", "LOW", "HIGH"]),
    "antialiasing": ("ANTI-ALIASING", ["OFF", "FXAA"]),
    "chromatic_aberration": ("CHROMATIC ABERRATION", ["OFF", "ON"]),
    "vignette": ("VIGNETTE", ["OFF", "ON"]),
    "color_grading": ("COLOR GRADING", ["OFF", "ON"]),
    "particles": ("PARTICLES", ["LOW", "MEDIUM", "HIGH"]),
}
GRAPHICS_PRESETS = {
    "LOW": {"reflections": 1, "ambient_occlusion": 0, "bloom": 1, "light_shafts": 0, "motion_blur": 0,
            "antialiasing": 0, "chromatic_aberration": 0, "vignette": 1, "color_grading": 1, "particles": 0},
    "MEDIUM": {"reflections": 1, "ambient_occlusion": 1, "bloom": 2, "light_shafts": 0, "motion_blur": 0,
               "antialiasing": 1, "chromatic_aberration": 0, "vignette": 1, "color_grading": 1, "particles": 1},
    "HIGH": {"reflections": 2, "ambient_occlusion": 2, "bloom": 2, "light_shafts": 1, "motion_blur": 0,
             "antialiasing": 1, "chromatic_aberration": 1, "vignette": 1, "color_grading": 1, "particles": 2},
    "ULTRA": {"reflections": 3, "ambient_occlusion": 3, "bloom": 3, "light_shafts": 2, "motion_blur": 1,
              "antialiasing": 1, "chromatic_aberration": 1, "vignette": 1, "color_grading": 1, "particles": 2},
}
PRESET_NAMES = ["LOW", "MEDIUM", "HIGH", "ULTRA", "CUSTOM"]
DEFAULT_PRESET = "HIGH"
PARTICLE_LIMITS = [150, 300, 450]
PARTICLE_DENSITY = [0.45, 0.75, 1.0]
FPS_CAP_OPTIONS = [30, 60, 120, 144, 0]   # 0 = unlimited

# ---------------------------------------------------------------------------
# Default user settings (saved to save_data.json)
# ---------------------------------------------------------------------------
DEFAULT_USER_SETTINGS = {
    "music_volume": 0.5,
    "sfx_volume": 0.8,
    "fullscreen": False,
    "screen_shake": True,
    "match_minutes": DEFAULT_MATCH_MINUTES,
    "difficulty": DEFAULT_DIFFICULTY,
    "show_fps": False,
    "renderer": "GPU",
    "graphics_preset": DEFAULT_PRESET,
    "fps_cap": 60,
    **GRAPHICS_PRESETS[DEFAULT_PRESET],
}
