"""UI building blocks: fonts, cached text, key caps, buttons and keyboard/mouse/gamepad menus."""
import math

import pygame

import settings as S
from graphics import GLOW, blur, rounded_panel
from physics import lerp_color

_font_cache = {}
_text_cache = {}
_glow_text_cache = {}


def font(size):
    f = _font_cache.get(size)
    if f is None:
        try:
            f = pygame.font.Font(None, size)   # pygame's bundled font: works everywhere
        except (pygame.error, OSError):
            f = pygame.font.SysFont("arial", size, bold=True)
        _font_cache[size] = f
    return f


def render_text(text, size, color):
    key = (text, size, color)
    surface = _text_cache.get(key)
    if surface is None:
        if len(_text_cache) > 500:
            _text_cache.clear()
        surface = font(size).render(text, True, color)
        _text_cache[key] = surface
    return surface


def draw_text(surface, text, size, color, pos, anchor="center", shadow=True):
    rendered = render_text(text, size, color)
    rect = rendered.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
    if shadow:
        surface.blit(render_text(text, size, (0, 0, 0)), rect.move(2, 3))
    surface.blit(rendered, rect)
    return rect


def glow_text_surface(text, size, color, glow_color):
    """Text plus a soft neon glow made of per-pixel alpha (cached)."""
    key = (text, size, color, glow_color)
    cached = _glow_text_cache.get(key)
    if cached is None:
        text_surface = font(size).render(text, True, color)
        pad = size // 2
        layer = pygame.Surface((text_surface.get_width() + pad * 2, text_surface.get_height() + pad * 2),
                               pygame.SRCALPHA)
        layer.fill((*glow_color, 0))
        shape = font(size).render(text, True, glow_color)
        # Thicken the text before blurring so the glow is strong.
        for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3), (0, 0)):
            layer.blit(shape, (pad + dx, pad + dy))
        glow = blur(layer, 6)
        cached = (text_surface, glow, pad)
        _glow_text_cache[key] = cached
    return cached


def draw_glow_text(surface, text, size, color, glow_color, pos, scale=1.0):
    text_surface, glow, pad = glow_text_surface(text, size, color, glow_color)
    if scale != 1.0:
        text_surface = pygame.transform.smoothscale(
            text_surface, (max(1, int(text_surface.get_width() * scale)), max(1, int(text_surface.get_height() * scale))))
        glow = pygame.transform.smoothscale(glow, (max(1, int(glow.get_width() * scale)), max(1, int(glow.get_height() * scale))))
    surface.blit(glow, glow.get_rect(center=pos))
    surface.blit(glow, glow.get_rect(center=pos))
    surface.blit(text_surface, text_surface.get_rect(center=pos))


ARROWS = {"LEFT": math.pi, "RIGHT": 0.0, "UP": -math.pi / 2, "DOWN": math.pi / 2}


def draw_keycap(surface, label, x, y, height=34):
    """Draw a keyboard key. Returns the key's width. 'LEFT'/'RIGHT'/'UP'/'DOWN' draw arrows."""
    if label in ARROWS:
        width = height
    else:
        width = max(height, render_text(label, 24, S.TEXT_COLOR).get_width() + 18)
    rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(surface, (8, 10, 22), rect.move(0, 3), border_radius=7)
    pygame.draw.rect(surface, (44, 52, 90), rect, border_radius=7)
    pygame.draw.rect(surface, (120, 140, 200), rect, 2, border_radius=7)
    if label in ARROWS:
        a = ARROWS[label]
        cx, cy = rect.center
        pts = [(cx + math.cos(a) * 9, cy + math.sin(a) * 9),
               (cx + math.cos(a + 2.4) * 8, cy + math.sin(a + 2.4) * 8),
               (cx + math.cos(a - 2.4) * 8, cy + math.sin(a - 2.4) * 8)]
        pygame.draw.polygon(surface, S.TEXT_COLOR, pts)
    else:
        draw_text(surface, label, 24, S.TEXT_COLOR, rect.center, shadow=False)
    return width


class Button:
    """
    A menu button. `action` runs when activated. Option buttons also have
    `on_change(direction)` for left/right and a `value` callable shown on the right.
    """

    def __init__(self, label, action=None, on_change=None, value=None, width=380, height=54):
        self.label = label
        self.action = action
        self.on_change = on_change
        self.value = value
        self.rect = pygame.Rect(0, 0, width, height)
        self.highlight = 0.0

    def text(self):
        return self.label() if callable(self.label) else self.label

    def update(self, dt, selected):
        target = 1.0 if selected else 0.0
        self.highlight += (target - self.highlight) * min(1.0, dt * 14)

    def draw(self, surface, selected, time):
        h = self.highlight
        rect = self.rect.inflate(int(16 * h), int(4 * h))
        fill = lerp_color(S.PANEL_COLOR, (26, 40, 86), h)
        panel = rounded_panel(rect.size, fill, None, radius=10, alpha=int(200 + 40 * h))
        surface.blit(panel, rect)
        border = lerp_color(S.PANEL_BORDER, S.PLAYER_COLOR, h)
        pygame.draw.rect(surface, border, rect, 2, border_radius=10)
        if h > 0.05:
            pulse = 0.5 + 0.5 * math.sin(time * 6)
            GLOW.draw_alpha(surface, S.PLAYER_COLOR, (rect.left + 6, rect.centery), 22, h * (0.5 + 0.5 * pulse))
            GLOW.draw_alpha(surface, S.PLAYER_COLOR, (rect.right - 6, rect.centery), 22, h * (0.5 + 0.5 * pulse))
        color = lerp_color(S.TEXT_DIM, (255, 255, 255), 0.4 + 0.6 * h)
        if self.value is None:
            draw_text(surface, self.text(), 30, color, rect.center)
        else:
            draw_text(surface, self.text(), 26, color, (rect.left + 22, rect.centery), anchor="midleft")
            value_text = self.value()
            right = rect.right - 22
            if selected:
                vw = render_text(value_text, 26, S.GOLD).get_width()
                draw_text(surface, ">", 26, S.GOLD, (right, rect.centery), anchor="midright")
                draw_text(surface, "<", 26, S.GOLD, (right - vw - 28, rect.centery), anchor="midright")
                right -= 20
            draw_text(surface, value_text, 26, S.GOLD if selected else color, (right, rect.centery), anchor="midright")


class Menu:
    """A vertical list of buttons controlled by keyboard, mouse or gamepad."""

    def __init__(self, buttons, center_x, top_y, spacing=64, audio=None):
        self.buttons = buttons
        self.center_x = center_x
        self.top_y = top_y
        self.spacing = spacing
        self.audio = audio
        self.selected = 0
        self._axis_latch = 0
        self.layout()

    def layout(self):
        for i, button in enumerate(self.buttons):
            button.rect.center = (self.center_x, self.top_y + i * self.spacing)

    def _sound(self, name):
        if self.audio:
            self.audio.play(name)

    def move(self, direction):
        self.selected = (self.selected + direction) % len(self.buttons)
        self._sound("menu_move")

    def change(self, direction):
        button = self.buttons[self.selected]
        if button.on_change:
            button.on_change(direction)
            self._sound("menu_move")

    def activate(self):
        button = self.buttons[self.selected]
        if button.action:
            self._sound("menu_select")
            button.action()
        elif button.on_change:
            self.change(1)

    def handle_event(self, event):
        """Returns True if the event was used."""
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_UP, pygame.K_w):
                self.move(-1)
            elif event.key in (pygame.K_DOWN, pygame.K_s, pygame.K_TAB):
                self.move(1)
            elif event.key in (pygame.K_LEFT, pygame.K_a):
                self.change(-1)
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                self.change(1)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self.activate()
            else:
                return False
            return True
        if event.type == pygame.MOUSEMOTION:
            for i, button in enumerate(self.buttons):
                if button.rect.collidepoint(event.pos) and i != self.selected:
                    self.selected = i
                    self._sound("menu_move")
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, button in enumerate(self.buttons):
                if button.rect.collidepoint(event.pos):
                    self.selected = i
                    if button.on_change and button.action is None:
                        # Click the left half to decrease, right half to increase.
                        self.change(-1 if event.pos[0] < button.rect.centerx - 40 else 1)
                    else:
                        self.activate()
                    return True
            return False
        if event.type == pygame.JOYHATMOTION:
            hat_x, hat_y = event.value
            if hat_y:
                self.move(-hat_y)
            elif hat_x:
                self.change(hat_x)
            return True
        if event.type == pygame.JOYAXISMOTION and event.axis in (0, 1):
            direction = 1 if event.value > 0.6 else (-1 if event.value < -0.6 else 0)
            latch = (event.axis, direction)
            if direction and latch != self._axis_latch:
                if event.axis == 1:
                    self.move(direction)
                else:
                    self.change(direction)
            if direction == 0 and self._axis_latch and self._axis_latch[0] == event.axis:
                self._axis_latch = 0
            elif direction:
                self._axis_latch = latch
            return True
        if event.type == pygame.JOYBUTTONDOWN and event.button == 0:
            self.activate()
            return True
        return False

    def update(self, dt):
        for i, button in enumerate(self.buttons):
            button.update(dt, i == self.selected)

    def draw(self, surface, time):
        for i, button in enumerate(self.buttons):
            button.draw(surface, i == self.selected, time)
