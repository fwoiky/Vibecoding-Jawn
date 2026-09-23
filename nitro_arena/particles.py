"""Lightweight particle system for boost trails, hits, dust, goals and confetti."""
import math
import random

import pygame

import settings as S
from graphics import GLOW


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "size", "color",
                 "gravity", "drag", "glow", "shape", "angle", "spin")

    def __init__(self, x, y, vx, vy, life, size, color, gravity, drag, glow, shape):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.life = self.max_life = life
        self.size = size
        self.color = color
        self.gravity = gravity
        self.drag = drag
        self.glow = glow
        self.shape = shape
        self.angle = random.uniform(0, math.pi)
        self.spin = random.uniform(-8, 8)


class Ring:
    """An expanding circle outline (impact shockwave)."""
    __slots__ = ("x", "y", "radius", "max_radius", "life", "max_life", "color", "width")

    def __init__(self, x, y, max_radius, life, color, width):
        self.x, self.y = x, y
        self.radius = 4.0
        self.max_radius = max_radius
        self.life = self.max_life = life
        self.color = color
        self.width = width


class ParticleSystem:
    def __init__(self, max_particles=S.MAX_PARTICLES):
        self.max_particles = max_particles
        self.density = 1.0          # scales burst sizes (particle quality setting)
        self.particles = []
        self.rings = []

    def clear(self):
        self.particles.clear()
        self.rings.clear()

    def emit(self, x, y, vx, vy, life, size, color, gravity=0.0, drag=0.0, glow=False, shape="circle"):
        if len(self.particles) >= self.max_particles:
            # Recycle the oldest particle instead of growing forever.
            self.particles.pop(0)
        self.particles.append(Particle(x, y, vx, vy, life, size, color, gravity, drag, glow, shape))

    def burst(self, pos, count, speed=(100, 400), life=(0.3, 0.8), size=(2, 5), colors=((255, 255, 255),),
              gravity=0.0, drag=1.5, glow=True, direction=None, spread=math.pi, shape="circle"):
        base_angle = math.atan2(direction[1], direction[0]) if direction is not None else 0.0
        for _ in range(max(1, int(count * self.density))):
            if direction is None:
                angle = random.uniform(0, 2 * math.pi)
            else:
                angle = base_angle + random.uniform(-spread, spread)
            s = random.uniform(*speed)
            self.emit(pos[0], pos[1], math.cos(angle) * s, math.sin(angle) * s,
                      random.uniform(*life), random.uniform(*size), random.choice(colors),
                      gravity, drag, glow, shape)

    def ring(self, pos, max_radius=80, life=0.35, color=(255, 255, 255), width=4):
        self.rings.append(Ring(pos[0], pos[1], max_radius, life, color, width))

    def update(self, dt):
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                continue
            if p.drag:
                factor = max(0.0, 1.0 - p.drag * dt)
                p.vx *= factor
                p.vy *= factor
            p.vy += p.gravity * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.angle += p.spin * dt
            if p.y > S.ARENA_FLOOR - 1 and p.gravity > 0:
                p.y = S.ARENA_FLOOR - 1
                p.vy *= -0.4
                p.vx *= 0.7
            alive.append(p)
        self.particles = alive

        rings = []
        for r in self.rings:
            r.life -= dt
            if r.life > 0:
                t = 1.0 - r.life / r.max_life
                r.radius = 4 + (r.max_radius - 4) * (1 - (1 - t) ** 2)
                rings.append(r)
        self.rings = rings

    def draw(self, surface):
        for p in self.particles:
            t = p.life / p.max_life
            if p.glow:
                GLOW.draw(surface, p.color, (p.x, p.y), p.size * 3 * (0.5 + 0.5 * t), t)
            elif p.shape == "confetti":
                w = p.size * 2
                h = max(1.0, p.size * abs(math.sin(p.angle)))
                pygame.draw.rect(surface, p.color, (p.x - w / 2, p.y - h / 2, w, h))
            else:
                radius = p.size * (0.3 + 0.7 * t)
                if radius >= 0.5:
                    pygame.draw.circle(surface, p.color, (p.x, p.y), radius)
        for r in self.rings:
            t = r.life / r.max_life
            width = max(1, int(r.width * t + 0.5))
            pygame.draw.circle(surface, r.color, (int(r.x), int(r.y)), int(r.radius), width)
