"""The ball: physics (gravity, drag, bounces), car hits and drawing."""
import math

import pygame
from pygame.math import Vector2

import settings as S
from graphics import GLOW, regular_polygon, soft_ellipse
from physics import circle_vs_obb, clamp, limit_speed, sign


def _build_ball_graphics(radius):
    size = int(radius * 2 + 4)
    center = size / 2

    # Opaque shaded base (sphere gradient lit from the top left).
    base = pygame.Surface((size, size), pygame.SRCALPHA)
    steps = int(radius)
    for i in range(steps, 0, -1):
        t = i / steps
        shade = 1.0 - 0.45 * t ** 2
        color = tuple(int(c * shade) for c in S.BALL_COLOR)
        offset = (1.0 - t) * radius * 0.35
        pygame.draw.circle(base, color, (center - offset, center - offset), i)

    # Panel pattern that rotates with the ball's spin.
    pattern = pygame.Surface((size, size), pygame.SRCALPHA)
    panel_color = (46, 56, 96, 230)
    seam = (120, 130, 170, 200)
    pygame.draw.polygon(pattern, panel_color, regular_polygon((center, center), radius * 0.32, 5, -math.pi / 2))
    for k in range(5):
        a = -math.pi / 2 + k * 2 * math.pi / 5
        inner = (center + math.cos(a) * radius * 0.32, center + math.sin(a) * radius * 0.32)
        outer = (center + math.cos(a) * radius * 0.78, center + math.sin(a) * radius * 0.78)
        pygame.draw.line(pattern, seam, inner, outer, 2)
        a2 = a + math.pi / 5
        spot = (center + math.cos(a2) * radius * 0.86, center + math.sin(a2) * radius * 0.86)
        pygame.draw.polygon(pattern, panel_color, regular_polygon(spot, radius * 0.2, 5, a2))

    # Mask the pattern to the ball circle.
    mask = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(mask, (255, 255, 255, 255), (center, center), radius - 1)
    pattern.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

    # Fixed lighting drawn on top of the rotating pattern.
    light = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(light, (255, 255, 255, 120), (center - radius * 0.38, center - radius * 0.42), radius * 0.22)
    pygame.draw.circle(light, (255, 255, 255, 60), (center - radius * 0.3, center - radius * 0.34), radius * 0.4)
    pygame.draw.circle(light, (*S.BALL_GLOW, 255), (center, center), radius, 2)

    if pygame.display.get_surface() is not None:
        base, pattern, light = base.convert_alpha(), pattern.convert_alpha(), light.convert_alpha()
    return base, pattern, light


class Ball:
    def __init__(self):
        self.radius = S.BALL_RADIUS
        self.base, self.pattern, self.light = _build_ball_graphics(self.radius)
        self.shadow = soft_ellipse(120, 22, max_alpha=160)
        self.reset()

    def reset(self):
        self.pos = Vector2(S.ARENA_CENTER_X, S.ARENA_FLOOR - S.BALL_KICKOFF_HEIGHT)
        self.vel = Vector2(0, 0)
        self.spin = 0.0          # visual rotation speed (radians per second)
        self.rotation = 0.0
        self.frozen = True       # held in place until kickoff
        self.visible = True
        self.hit_flash = 0.0
        self.on_ground = False

    def update(self, dt):
        if self.frozen:
            return
        self.vel.y += S.BALL_GRAVITY * dt
        self.vel *= 1.0 - S.BALL_DRAG * dt
        if self.on_ground:
            self.vel.x = self._approach_zero(self.vel.x, S.BALL_ROLL_FRICTION * dt)
        limit_speed(self.vel, S.BALL_MAX_SPEED)
        self.pos += self.vel * dt
        self.rotation += self.spin * dt
        if self.on_ground:
            self.spin = self.vel.x / self.radius
        else:
            self.spin *= 1.0 - 0.3 * dt
        self.hit_flash = max(0.0, self.hit_flash - dt * 3.0)

    @staticmethod
    def _approach_zero(value, amount):
        if abs(value) <= amount:
            return 0.0
        return value - sign(value) * amount

    def collide_arena(self, arena):
        impact, normal = arena.collide_ball(self.pos, self.vel, self.radius)
        # Touching the floor (or resting on it) enables rolling friction.
        floor_contact = arena.deepest_contact(self.pos, self.radius + 1.5)
        self.on_ground = floor_contact is not None and floor_contact.normal.y < -0.7
        if normal is not None and not self.on_ground:
            tangent_speed = self.vel.x * -normal[1] + self.vel.y * normal[0]
            self.spin = tangent_speed / self.radius
        return impact

    def collide_car(self, car):
        """
        Resolve a car hitting the ball. Uses an impulse along the contact normal,
        plus a small arcade boost so hits feel punchy. Returns the impact speed.
        """
        contact = circle_vs_obb(self.pos, self.radius, car.pos, car.axis, -car.up,
                                S.CAR_HALF_LENGTH, S.CAR_HALF_HEIGHT)
        if contact is None:
            return 0.0
        n = contact.normal
        self.pos += n * contact.depth

        # Velocity of the touching point of the car (includes spin from flips).
        r = contact.point - car.pos
        point_vel = car.vel + Vector2(-r.y, r.x) * car.ang_vel
        rel = self.vel - point_vel
        vn = rel.dot(n)
        if vn >= 0:
            return 0.0   # already separating

        # Arcade model: the ball bounces off the car as if the car were very
        # heavy; the car only recoils a little. Extra power comes from the
        # car's OWN speed into the ball, so a parked car is a blocker, not a
        # trampoline.
        j = -(1.0 + S.HIT_RESTITUTION) * vn
        self.vel += n * j
        car.vel -= n * (j * S.CAR_HIT_RECOIL)

        drive_in = max(0.0, point_vel.dot(n))
        shot_dir = Vector2(n)
        if n.y > -0.2:
            shot_dir.y -= S.HIT_LIFT
            shot_dir = shot_dir.normalize()
        self.vel += shot_dir * drive_in * S.HIT_POWER_BONUS
        if car.dodge_time > 0:
            self.vel += shot_dir * S.DODGE_HIT_BONUS
        limit_speed(self.vel, S.BALL_MAX_SPEED)
        approach_speed = -vn

        tangent = Vector2(-n.y, n.x)
        self.spin = clamp(rel.dot(tangent) / self.radius, -25, 25)
        self.hit_flash = clamp(approach_speed / S.STRONG_HIT_SPEED, 0.3, 1.0)
        self.on_ground = False
        return approach_speed

    def separate_from_car(self, car):
        """If the ball got pinned against a wall, push the car out instead."""
        contact = circle_vs_obb(self.pos, self.radius, car.pos, car.axis, -car.up,
                                S.CAR_HALF_LENGTH, S.CAR_HALF_HEIGHT)
        if contact is None:
            return
        car.pos -= contact.normal * contact.depth
        vn = car.vel.dot(contact.normal)
        if vn > 0:
            car.vel -= contact.normal * vn

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw_shadow(self, surface):
        if not self.visible:
            return
        height = S.ARENA_FLOOR - self.pos.y
        if height < 0:
            return
        t = clamp(height / 500.0, 0.0, 1.0)
        width = int(90 * (1.0 - 0.55 * t))
        shadow = pygame.transform.scale(self.shadow, (width, 16))
        shadow.set_alpha(int(230 * (1.0 - t * 0.85)))
        surface.blit(shadow, (self.pos.x - width / 2, S.ARENA_FLOOR - 9))

    def draw(self, surface):
        if not self.visible:
            return
        center = (round(self.pos.x), round(self.pos.y))
        speed = self.vel.length()
        GLOW.draw(surface, S.BALL_GLOW, center, self.radius * 2.2, 0.35 + 0.5 * self.hit_flash + min(0.3, speed / 4000))
        surface.blit(self.base, self.base.get_rect(center=center))
        pattern = pygame.transform.rotate(self.pattern, -math.degrees(self.rotation))
        surface.blit(pattern, pattern.get_rect(center=center))
        surface.blit(self.light, self.light.get_rect(center=center))
        if self.hit_flash > 0.05:
            pygame.draw.circle(surface, (255, 255, 255), center, int(self.radius + 3 + 10 * (1 - self.hit_flash)),
                               max(1, int(4 * self.hit_flash)))
