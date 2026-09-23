"""Turns keyboard (and optional gamepad) input into Controls for the player's car."""
import pygame

from controls import Controls

KEYS_LEFT = (pygame.K_a, pygame.K_LEFT)
KEYS_RIGHT = (pygame.K_d, pygame.K_RIGHT)
KEYS_JUMP = (pygame.K_w, pygame.K_UP)
KEYS_DOWN = (pygame.K_s, pygame.K_DOWN)
KEYS_BOOST = (pygame.K_SPACE, pygame.K_RSHIFT)
KEYS_ROTATE_CCW = (pygame.K_q,)
KEYS_ROTATE_CW = (pygame.K_e,)

# Common gamepad layout (Xbox style through SDL). Unknown pads still work
# with the left stick + first buttons.
PAD_JUMP_BUTTONS = (0,)          # A / Cross
PAD_BOOST_BUTTONS = (1, 2)       # B / Circle, X / Square
PAD_ROTATE_CCW_BUTTONS = (4,)    # LB / L1
PAD_ROTATE_CW_BUTTONS = (5,)     # RB / R1
PAD_PAUSE_BUTTONS = (7, 9)       # Start / Options
PAD_DEADZONE = 0.3


def any_pressed(keys, key_group):
    return any(keys[k] for k in key_group)


class PlayerController:
    def __init__(self):
        self.jump_queued = False
        self.joysticks = {}
        self.refresh_joysticks()

    def refresh_joysticks(self):
        self.joysticks = {}
        try:
            for i in range(pygame.joystick.get_count()):
                joystick = pygame.joystick.Joystick(i)
                joystick.init()
                self.joysticks[joystick.get_instance_id()] = joystick
        except pygame.error:
            self.joysticks = {}

    def reset(self):
        self.jump_queued = False

    def handle_event(self, event):
        """Record jump presses as events so very quick taps are never missed."""
        if event.type == pygame.KEYDOWN and event.key in KEYS_JUMP:
            self.jump_queued = True
        elif event.type == pygame.JOYBUTTONDOWN and event.button in PAD_JUMP_BUTTONS:
            self.jump_queued = True
        elif event.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
            self.refresh_joysticks()

    def poll(self):
        keys = pygame.key.get_pressed()
        c = Controls()
        move = 0.0
        if any_pressed(keys, KEYS_LEFT):
            move -= 1.0
        if any_pressed(keys, KEYS_RIGHT):
            move += 1.0
        rotate = 0.0
        if any_pressed(keys, KEYS_ROTATE_CCW):
            rotate -= 1.0
        if any_pressed(keys, KEYS_ROTATE_CW):
            rotate += 1.0
        c.down = any_pressed(keys, KEYS_DOWN)
        c.jump_held = any_pressed(keys, KEYS_JUMP)
        c.boost = any_pressed(keys, KEYS_BOOST)

        for joystick in self.joysticks.values():
            try:
                if joystick.get_numaxes() >= 2:
                    ax = joystick.get_axis(0)
                    ay = joystick.get_axis(1)
                    if abs(ax) > PAD_DEADZONE and move == 0.0:
                        move = max(-1.0, min(1.0, ax))
                    if ay > 0.6:
                        c.down = True
                if joystick.get_numhats() > 0:
                    hat_x, hat_y = joystick.get_hat(0)
                    if hat_x and move == 0.0:
                        move = float(hat_x)
                    if hat_y < 0:
                        c.down = True
                buttons = joystick.get_numbuttons()
                pressed = lambda group: any(b < buttons and joystick.get_button(b) for b in group)
                c.jump_held = c.jump_held or pressed(PAD_JUMP_BUTTONS)
                c.boost = c.boost or pressed(PAD_BOOST_BUTTONS)
                if rotate == 0.0:
                    if pressed(PAD_ROTATE_CCW_BUTTONS):
                        rotate = -1.0
                    elif pressed(PAD_ROTATE_CW_BUTTONS):
                        rotate = 1.0
            except pygame.error:
                continue

        c.move_x = move
        c.rotate = rotate
        c.jump_pressed = self.jump_queued
        self.jump_queued = False
        return c
