"""
The rocket car: arcade physics, collision with the arena and drawing.

Orientation model
-----------------
* `angle` is the rotation of the car body (0 = wheels down). Positive angles
  rotate clockwise on screen (pygame's y axis points down).
* `facing` is +1 when the nose points right and -1 when it points left.
* forward (nose direction) = facing * (cos(angle), sin(angle))
* up (roof direction)      = (sin(angle), -cos(angle))

Collision shape
---------------
The car collides with the arena as a *capsule*: two circles (the "wheels")
joined by a straight core. The ball uses a slightly bigger oriented box so
hits with the nose or the roof feel different.
"""
import math

import pygame
from pygame.math import Vector2

import settings as S
from graphics import GLOW, scale_color, soft_ellipse
from physics import (Contact, approach, clamp, closest_point_on_segment,
                     limit_speed, sign, wrap_angle)

SPRITE_W, SPRITE_H = 96, 48
WHEEL_OFFSET_X = 22
WHEEL_OFFSET_Y = 5
WHEEL_RADIUS = 8


def build_car_sprite(body_color, accent_color, dark_color, style):
    """Draw a car facing right onto a transparent surface (centre = physics centre)."""
    surface = pygame.Surface((SPRITE_W, SPRITE_H), pygame.SRCALPHA)
    cx, cy = SPRITE_W / 2, SPRITE_H / 2

    def pts(points):
        return [(cx + x, cy + y) for x, y in points]

    if style == "wedge":
        body = [(-37, 7), (-38, -5), (-27, -9), (-13, -17), (5, -17), (24, -6), (38, -2), (38, 5), (32, 8), (-32, 8)]
        window = [(-9, -14), (3, -14), (16, -7), (-18, -7)]
        spoiler = [(-40, -18), (-24, -18), (-24, -15), (-40, -14)]
        spoiler_post = [(-34, -15), (-31, -15), (-30, -8), (-33, -8)]
    else:
        body = [(-37, 7), (-38, -3), (-34, -8), (-20, -9), (-10, -17), (9, -17), (21, -8), (34, -5), (38, 0), (37, 6), (30, 8), (-30, 8)]
        window = [(-7, -14), (7, -14), (16, -8), (-14, -8)]
        spoiler = [(-40, -12), (-30, -12), (-30, -9), (-40, -8)]
        spoiler_post = None

    pygame.draw.polygon(surface, body_color, pts(body))
    # Lower skirt (darker) for depth.
    skirt = [(x, y) for x, y in body if y >= 0]
    skirt = [(-37, 1), (37, 1)] + sorted(skirt, key=lambda p: -p[0])
    pygame.draw.polygon(surface, scale_color(body_color, 0.72), pts(skirt))
    # Racing stripe.
    pygame.draw.line(surface, accent_color, (cx - 34, cy - 2), (cx + 33, cy - 2), 3)
    # Window with a reflection.
    pygame.draw.polygon(surface, (18, 26, 48), pts(window))
    pygame.draw.line(surface, (120, 170, 220), (cx - 2, cy - 13), (cx + 6, cy - 8), 2)
    # Spoiler.
    pygame.draw.polygon(surface, dark_color, pts(spoiler))
    if spoiler_post:
        pygame.draw.polygon(surface, dark_color, pts(spoiler_post))
    # Wheel arches.
    for wx in (-WHEEL_OFFSET_X, WHEEL_OFFSET_X):
        pygame.draw.circle(surface, (12, 14, 22), (int(cx + wx), int(cy + WHEEL_OFFSET_Y)), WHEEL_RADIUS + 2)
    # Lights.
    pygame.draw.rect(surface, (255, 250, 210), (cx + 33, cy - 4, 5, 3))
    pygame.draw.rect(surface, (255, 60, 70), (cx - 39, cy - 3, 4, 3))
    # Outline.
    pygame.draw.polygon(surface, dark_color, pts(body), 2)
    if pygame.display.get_surface() is not None:
        surface = surface.convert_alpha()
    return surface


class Car:
    def __init__(self, team, body_color, accent_color, dark_color, style):
        self.team = team
        self.body_color = body_color
        self.accent_color = accent_color
        self.sprite = build_car_sprite(body_color, accent_color, dark_color, style)
        self.sprite_flipped = pygame.transform.flip(self.sprite, True, False)
        self.shadow = soft_ellipse(110, 22, max_alpha=150)
        self.events = []  # (name, position, strength) tuples read by the match each frame
        self.reset(S.ARENA_CENTER_X, 1)

    def reset(self, x, facing):
        self.pos = Vector2(x, S.ARENA_FLOOR - S.CAR_RADIUS - 0.5)
        self.vel = Vector2(0, 0)
        self.angle = 0.0
        self.ang_vel = 0.0
        self.facing = facing
        self.boost = S.BOOST_START
        self.boosting = False
        self.boost_regen_delay = 0.0
        self.grounded = True
        self.ground_normal = Vector2(0, -1)
        self.ground_cooldown = 0.0
        self.has_air_jump = False
        self.air_time = 0.0
        self.jump_hold = 0.0
        self.dodge_time = 0.0
        self.tip_normal = None
        self.roof_time = 0.0
        self.last_impact = 0.0
        self.wheel_angle = 0.0
        self.events.clear()

    # ------------------------------------------------------------------
    # Orientation helpers
    # ------------------------------------------------------------------
    @property
    def axis(self):
        return Vector2(math.cos(self.angle), math.sin(self.angle))

    @property
    def forward(self):
        return self.axis * self.facing

    @property
    def up(self):
        return Vector2(math.sin(self.angle), -math.cos(self.angle))

    @property
    def on_floor(self):
        return self.grounded and self.ground_normal.y < -0.7

    def wheel_positions(self):
        offset = self.axis * S.CAR_CAPSULE_HALF
        return self.pos - offset, self.pos + offset

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, controls, dt):
        if self.ground_cooldown > 0:
            self.ground_cooldown -= dt
        self._handle_jump(controls)
        if self.grounded:
            self._drive(controls, dt)
        else:
            self._fly(controls, dt)
        self._handle_boost(controls, dt)

        self.vel.y += S.GRAVITY * dt
        limit_speed(self.vel, S.CAR_MAX_TOTAL_SPEED)
        self.pos += self.vel * dt
        self.angle = wrap_angle(self.angle + self.ang_vel * dt)
        if self.grounded:
            self.wheel_angle += self.facing * self.vel.dot(self.forward) / WHEEL_RADIUS * dt

    def _handle_jump(self, c):
        if not c.jump_pressed:
            return
        if self.grounded:
            n = self.ground_normal
            vn = self.vel.dot(n)
            if vn < 0:
                self.vel -= n * vn
            self.vel += n * S.JUMP_VELOCITY
            self.grounded = False
            self.ground_cooldown = S.GROUND_COOLDOWN_AFTER_JUMP
            self.has_air_jump = True
            self.air_time = 0.0
            self.jump_hold = S.JUMP_HOLD_TIME
            self.events.append(("jump", Vector2(self.pos), 1.0))
        elif self.has_air_jump and self.air_time < S.DOUBLE_JUMP_WINDOW:
            self.has_air_jump = False
            self.jump_hold = 0.0
            direction = sign(c.move_x) if abs(c.move_x) > 0.3 else 0
            if direction:
                # Flip / dodge: a burst of speed plus a full spin that hits hard.
                self.vel.x = clamp(self.vel.x + direction * S.DODGE_IMPULSE,
                                   -S.DODGE_MAX_SPEED, S.DODGE_MAX_SPEED)
                self.vel.y = min(self.vel.y * 0.25, -S.DODGE_LIFT)
                self.dodge_time = S.DODGE_DURATION
                self.ang_vel = direction * 2 * math.pi / S.DODGE_DURATION
                self.events.append(("dodge", Vector2(self.pos), 1.0))
            else:
                self.vel.y = min(self.vel.y, 0.0) - S.DOUBLE_JUMP_VELOCITY
                self.events.append(("double_jump", Vector2(self.pos), 1.0))

    def _drive(self, c, dt):
        n = self.ground_normal
        on_floor = n.y < -0.7

        # Line the car body up with the surface it is driving on.
        target = math.atan2(n.x, -n.y)
        self.angle = wrap_angle(self.angle + wrap_angle(target - self.angle) * min(1.0, S.CAR_ALIGN_RATE * dt))
        self.ang_vel = 0.0

        tangent = Vector2(-n.y, n.x)
        if tangent.dot(self.forward) < 0:
            tangent = -tangent
        speed = self.vel.dot(tangent)
        move = c.move_x
        boosting = c.boost and self.boost > 0

        # Pressing the opposite direction on the floor: brake, then turn around.
        if on_floor and abs(move) > 0.1 and sign(move) != self.facing and not c.down:
            if speed > S.CAR_TURN_SPEED:
                new_speed = approach(speed, 0.0, S.CAR_BRAKE * dt)
                self.vel += tangent * (new_speed - speed)
                return
            self.facing = -self.facing
            tangent = -tangent
            speed = -speed
            self.events.append(("turn", Vector2(self.pos), 1.0))

        throttle = move * self.facing
        if c.down:
            if speed > 20:
                new_speed = approach(speed, 0.0, S.CAR_BRAKE * dt)
            else:
                new_speed = max(speed - S.CAR_REVERSE_ACCEL * dt, min(speed, -S.CAR_REVERSE_MAX_SPEED))
        elif throttle > 0.05:
            max_speed = S.CAR_MAX_SPEED * min(1.0, throttle)
            if speed < 0:
                new_speed = speed + S.CAR_BRAKE * dt
            elif speed < max_speed:
                new_speed = min(speed + S.CAR_ACCEL * throttle * dt, max_speed)
            else:
                new_speed = speed
        elif throttle < -0.05:
            if speed > 0:
                new_speed = approach(speed, 0.0, S.CAR_BRAKE * dt)
            else:
                new_speed = max(speed - S.CAR_REVERSE_ACCEL * dt, min(speed, -S.CAR_REVERSE_MAX_SPEED))
        else:
            new_speed = approach(speed, 0.0, S.CAR_COAST_DECEL * dt)

        if not boosting and abs(new_speed) > S.CAR_MAX_SPEED:
            new_speed = approach(new_speed, sign(new_speed) * S.CAR_MAX_SPEED, S.CAR_OVERSPEED_DECEL * dt)
        self.vel += tangent * (new_speed - speed)

        # Walls and ceiling: fast cars stick, slow cars fall off.
        if not on_floor:
            if abs(new_speed) > S.WALL_MIN_SPEED:
                away = self.vel.dot(n)
                if away > 0:
                    self.vel -= n * away   # arcade grip: never drift off the surface
                self.vel -= n * S.WALL_STICK_ACCEL * dt
            else:
                self.vel += n * S.WALL_DETACH_ACCEL * dt

    def _fly(self, c, dt):
        self.air_time += dt
        if self.dodge_time > 0:
            self.dodge_time -= dt
            if self.dodge_time <= 0:
                self.ang_vel = 0.0
        elif abs(c.rotate) > 0.05:
            self.ang_vel = approach(self.ang_vel, c.rotate * S.AIR_ROT_SPEED, S.AIR_ROT_ACCEL * dt)
        else:
            self.ang_vel = approach(self.ang_vel, 0.0, S.AIR_ROT_DAMP * dt)

        if abs(c.move_x) > 0.05 and self.vel.x * sign(c.move_x) < S.AIR_CONTROL_MAX_SPEED:
            self.vel.x += c.move_x * S.AIR_CONTROL_ACCEL * dt
        if c.down:
            self.vel.y += S.AIR_DIVE_ACCEL * dt
        if self.jump_hold > 0:
            if c.jump_held:
                self.vel.y -= S.JUMP_HOLD_ACCEL * dt
                self.jump_hold -= dt
            else:
                self.jump_hold = 0.0

        # Lying on the side or roof on the floor: tip back onto the wheels.
        if self.tip_normal is not None and self.dodge_time <= 0:
            n = self.tip_normal
            if self.up.dot(n) < -0.2:
                self.roof_time += dt
                if self.roof_time > S.CAR_ROOF_FLIP_DELAY:
                    self._self_right()
            else:
                target = math.atan2(n.x, -n.y)
                diff = wrap_angle(target - self.angle)
                self.angle = wrap_angle(self.angle + clamp(diff, -S.CAR_TIP_RATE * dt, S.CAR_TIP_RATE * dt))
                self.ang_vel = 0.0
        else:
            self.roof_time = 0.0

    def _self_right(self):
        # Rotating by pi and mirroring keeps the nose direction but puts the
        # wheels back on the ground.
        self.angle = wrap_angle(self.angle + math.pi)
        self.facing = -self.facing
        self.vel.y -= 260
        self.roof_time = 0.0
        self.events.append(("flip_upright", Vector2(self.pos), 1.0))

    def _handle_boost(self, c, dt):
        self.boosting = False
        if c.boost and self.boost > 0:
            self.boosting = True
            self.boost = max(0.0, self.boost - S.BOOST_DRAIN * dt)
            self.boost_regen_delay = S.BOOST_REGEN_DELAY
            accel = S.BOOST_ACCEL_GROUND if self.grounded else S.BOOST_ACCEL
            direction = self.forward
            if self.vel.dot(direction) < S.BOOST_MAX_SPEED:
                self.vel += direction * accel * dt
        elif self.boost_regen_delay > 0:
            self.boost_regen_delay -= dt
        else:
            self.boost = min(S.BOOST_MAX, self.boost + S.BOOST_REGEN * dt)

    # ------------------------------------------------------------------
    # Collisions with the arena
    # ------------------------------------------------------------------
    def collide_arena(self, arena):
        self.last_impact = 0.0
        for _ in range(3):
            best = None
            a, b = self.wheel_positions()
            for p in (a, b):
                hit = arena.deepest_contact(p, S.CAR_RADIUS)
                if hit and (best is None or hit.depth > best.depth):
                    best = hit
            # Goal crossbar edges poking into the side of the car.
            for corner in arena.convex_corners:
                q = closest_point_on_segment(corner, a, b)
                diff = q - corner
                dist = diff.length()
                if dist < S.CAR_RADIUS:
                    normal = diff / dist if dist > 1e-6 else Vector2(0, 1)
                    depth = S.CAR_RADIUS - dist
                    if best is None or depth > best.depth:
                        best = Contact(normal, depth, Vector2(corner))
            if best is None:
                break
            self.pos += best.normal * best.depth
            vn = self.vel.dot(best.normal)
            if vn < 0:
                bounce = S.CAR_BOUNCE if -vn > 250 else 0.0
                self.vel -= best.normal * vn * (1.0 + bounce)
                self.last_impact = max(self.last_impact, -vn)

    def update_ground_state(self, arena):
        was_grounded = self.grounded
        self.tip_normal = None
        if self.ground_cooldown > 0:
            self.grounded = False
        else:
            normals = []
            up = self.up
            # A longer probe on walls/ceiling keeps fast cars glued around corners.
            probe = S.CAR_GROUND_PROBE * (3.0 if was_grounded and self.ground_normal.y > -0.7 else 1.0)
            for p in self.wheel_positions():
                hit = arena.deepest_contact(p, S.CAR_RADIUS + probe)
                if hit is None:
                    continue
                if hit.normal.dot(up) > 0.5:
                    normals.append(hit.normal)
                elif hit.normal.y < -0.6:
                    self.tip_normal = hit.normal
            if normals:
                n = normals[0] if len(normals) == 1 else (normals[0] + normals[1])
                n = n.normalize()
                self.grounded = self.vel.dot(n) < 150
                if self.grounded:
                    self.ground_normal = n
                    self.tip_normal = None
            else:
                self.grounded = False

        if self.grounded and not was_grounded:
            self.events.append(("land", Vector2(self.pos), self.last_impact))
            self.has_air_jump = False
            self.dodge_time = 0.0
            self.air_time = 0.0
            self.jump_hold = 0.0
            self.roof_time = 0.0
        elif was_grounded and not self.grounded and self.ground_cooldown <= 0:
            # Drove off a ledge: allow one air jump.
            self.has_air_jump = True
            self.air_time = 0.0

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw_shadow(self, surface):
        height = S.ARENA_FLOOR - self.pos.y
        if height < 0 or self.pos.y < S.ARENA_TOP:
            return
        t = clamp(height / 450.0, 0.0, 1.0)
        width = int(100 * (1.0 - 0.5 * t))
        shadow = pygame.transform.scale(self.shadow, (width, 16))
        shadow.set_alpha(int(220 * (1.0 - t)))
        surface.blit(shadow, (self.pos.x - width / 2, S.ARENA_FLOOR - 9))

    def draw(self, surface, time):
        fwd = self.forward
        down = -self.up
        GLOW.draw(surface, self.body_color, self.pos + down * 10, 46, 0.35)
        if self.boosting:
            self._draw_flame(surface, time, fwd, down)

        sprite = self.sprite if self.facing > 0 else self.sprite_flipped
        rotated = pygame.transform.rotozoom(sprite, -math.degrees(self.angle), 1.0)
        surface.blit(rotated, rotated.get_rect(center=(round(self.pos.x), round(self.pos.y))))

        spoke = Vector2(math.cos(self.wheel_angle), math.sin(self.wheel_angle)) * 6
        for lx in (-WHEEL_OFFSET_X, WHEEL_OFFSET_X):
            c = self.pos + fwd * lx + down * WHEEL_OFFSET_Y
            pygame.draw.circle(surface, (24, 26, 36), c, WHEEL_RADIUS)
            pygame.draw.circle(surface, (175, 185, 205), c, 4)
            pygame.draw.line(surface, (70, 74, 92), c - spoke, c + spoke, 2)

    def _draw_flame(self, surface, time, fwd, down):
        rear = self.pos - fwd * 37 + down * (-1)
        side = Vector2(-fwd.y, fwd.x)
        flicker = 0.75 + 0.25 * math.sin(time * 55.0 + self.team * 2.0)
        length = 26 * flicker + 8
        outer = [rear + side * 7, rear - side * 7, rear - fwd * length]
        inner = [rear + side * 4, rear - side * 4, rear - fwd * (length * 0.6)]
        GLOW.draw(surface, S.BOOST_FLAME_OUTER, rear - fwd * 10, 34, 0.8)
        pygame.draw.polygon(surface, S.BOOST_FLAME_OUTER, outer)
        pygame.draw.polygon(surface, S.BOOST_FLAME_INNER, inner)
