"""The set of inputs a car understands. Both the player and the AI produce these."""
from dataclasses import dataclass


@dataclass
class Controls:
    move_x: float = 0.0        # -1 = left, +1 = right (analog values allowed)
    down: bool = False         # brake / reverse on the ground, dive in the air
    jump_pressed: bool = False  # True only on the frame the jump button went down
    jump_held: bool = False
    boost: bool = False
    rotate: float = 0.0        # -1 = counter-clockwise, +1 = clockwise (air roll)
