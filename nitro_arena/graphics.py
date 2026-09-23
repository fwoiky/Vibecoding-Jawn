"""
Procedural drawing helpers shared by the arena, cars, ball, particles and UI.
Everything here is generated in code - the game ships without image files.
"""
import math

import pygame

from physics import clamp, lerp_color


def vertical_gradient(size, top_color, bottom_color, alpha=False):
    width, height = size
    flags = pygame.SRCALPHA if alpha else 0
    surface = pygame.Surface((width, height), flags)
    for y in range(height):
        t = y / max(1, height - 1)
        pygame.draw.line(surface, lerp_color(top_color, bottom_color, t), (0, y), (width, y))
    return surface


def blur(surface, amount=4):
    """Cheap blur: shrink the image, then scale it back up in two smooth steps."""
    width, height = surface.get_size()
    small = pygame.transform.smoothscale(surface, (max(1, width // amount), max(1, height // amount)))
    mid = pygame.transform.smoothscale(small, (max(1, width // 2), max(1, height // 2)))
    mid = pygame.transform.smoothscale(mid, (max(1, width // 3), max(1, height // 3)))
    return pygame.transform.smoothscale(mid, (width, height))


def scale_color(color, factor):
    return (int(clamp(color[0] * factor, 0, 255)),
            int(clamp(color[1] * factor, 0, 255)),
            int(clamp(color[2] * factor, 0, 255)))


def glow_layer(size, draw_fn, blur_amount=6, strength=2):
    """
    Build a glow by drawing shapes on a black layer, blurring it and returning
    it. Blit the result with BLEND_ADD for a neon look.
    """
    layer = pygame.Surface(size)
    layer.fill((0, 0, 0))
    draw_fn(layer)
    glow = blur(layer, blur_amount)
    for _ in range(strength - 1):
        glow.blit(glow, (0, 0), special_flags=pygame.BLEND_ADD)
    return glow


def draw_neon_lines(target, color, lines, width=3, glow_width=12, closed=False):
    """Draw polylines with a soft neon glow onto `target` (used for static art)."""
    size = target.get_size()

    def draw(layer):
        for points in lines:
            if len(points) >= 2:
                pygame.draw.lines(layer, color, closed, points, glow_width)

    target.blit(glow_layer(size, draw, blur_amount=6, strength=2), (0, 0), special_flags=pygame.BLEND_ADD)
    bright = lerp_color(color, (255, 255, 255), 0.55)
    for points in lines:
        if len(points) >= 2:
            pygame.draw.lines(target, bright, closed, points, width)


class GlowCache:
    """
    Pre-rendered radial glow sprites for additive blending. Sprites are cached
    by (colour, radius, intensity level) so nothing is re-created every frame.
    """
    LEVELS = 8

    def __init__(self):
        self._cache = {}

    def get(self, color, radius, intensity=1.0):
        radius = max(2, int(radius))
        level = int(clamp(intensity, 0.0, 1.0) * self.LEVELS + 0.5)
        key = (color, radius, level)
        sprite = self._cache.get(key)
        if sprite is None:
            if len(self._cache) > 600:
                self._cache.clear()
            sprite = self._build(color, radius, level / self.LEVELS)
            self._cache[key] = sprite
        return sprite

    @staticmethod
    def _build(color, radius, intensity):
        size = radius * 2
        surface = pygame.Surface((size, size))
        surface.fill((0, 0, 0))
        steps = max(4, radius)
        for i in range(steps, 0, -1):
            r = radius * i / steps
            falloff = (1.0 - i / steps) ** 1.6
            pygame.draw.circle(surface, scale_color(color, falloff * intensity), (radius, radius), max(1, int(r)))
        return surface

    def get_alpha(self, color, radius, intensity=1.0):
        """Glow sprite using per-pixel alpha (works on transparent UI layers)."""
        radius = max(2, int(radius))
        level = int(clamp(intensity, 0.0, 1.0) * self.LEVELS + 0.5)
        key = ("alpha", color, radius, level)
        sprite = self._cache.get(key)
        if sprite is None:
            sprite = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            sprite.fill((*color, 0))
            steps = max(4, radius)
            for i in range(steps, 0, -1):
                falloff = (1.0 - i / steps) ** 1.6
                alpha = int(255 * falloff * level / self.LEVELS)
                pygame.draw.circle(sprite, (*color, alpha), (radius, radius), max(1, int(radius * i / steps)))
            self._cache[key] = sprite
        return sprite

    def draw_alpha(self, target, color, pos, radius, intensity=1.0):
        if intensity <= 0.02:
            return
        sprite = self.get_alpha(color, radius, intensity)
        r = sprite.get_width() // 2
        target.blit(sprite, (int(pos[0]) - r, int(pos[1]) - r))

    def draw(self, target, color, pos, radius, intensity=1.0):
        if intensity <= 0.02:
            return
        sprite = self.get(color, radius, intensity)
        r = sprite.get_width() // 2
        target.blit(sprite, (int(pos[0]) - r, int(pos[1]) - r), special_flags=pygame.BLEND_ADD)


GLOW = GlowCache()


def soft_ellipse(width, height, color=(0, 0, 0), max_alpha=150):
    """A blurry ellipse, used for shadows."""
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    steps = 10
    # pygame.draw overwrites alpha instead of blending, so draw from the outside
    # in with increasing opacity.
    for i in range(steps):
        t = i / steps
        alpha = int(max_alpha * ((i + 1) / steps) ** 1.2)
        rect = pygame.Rect(0, 0, int(width * (1 - t * 0.7)), int(height * (1 - t * 0.7)))
        rect.center = (width // 2, height // 2)
        pygame.draw.ellipse(surface, (*color, alpha), rect)
    return surface


def rounded_panel(size, fill, border=None, radius=12, border_width=2, alpha=255):
    surface = pygame.Surface(size, pygame.SRCALPHA)
    rect = surface.get_rect()
    pygame.draw.rect(surface, (*fill, alpha), rect, border_radius=radius)
    if border:
        pygame.draw.rect(surface, (*border, 255), rect, border_width, border_radius=radius)
    return surface


def regular_polygon(center, radius, sides, rotation=0.0):
    cx, cy = center
    return [(cx + math.cos(rotation + i * 2 * math.pi / sides) * radius,
             cy + math.sin(rotation + i * 2 * math.pi / sides) * radius) for i in range(sides)]
