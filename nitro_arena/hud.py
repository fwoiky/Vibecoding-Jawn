"""In-game heads-up display: scoreboard, timer, boost meters and big centre messages."""
import math

import pygame

import settings as S
from graphics import GLOW, rounded_panel
from ui import draw_text, glow_text_surface, render_text

SEGMENTS = 20


def format_time(seconds):
    seconds = max(0, int(math.ceil(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


class HUD:
    def __init__(self):
        self.team_panel = rounded_panel((280, 64), S.PANEL_COLOR, S.PANEL_BORDER, radius=12, alpha=215)
        self.timer_panel = rounded_panel((190, 78), (10, 12, 30), S.PANEL_BORDER, radius=14, alpha=230)
        self.boost_panel = rounded_panel((320, 56), S.PANEL_COLOR, S.PANEL_BORDER, radius=12, alpha=215)
        self.small_boost_panel = rounded_panel((200, 40), S.PANEL_COLOR, S.PANEL_BORDER, radius=10, alpha=190)
        self.hint_panel = rounded_panel((600, 30), S.PANEL_COLOR, None, radius=10, alpha=170)
        self.message = None   # (title, subtitle, colour, duration, age)
        self.time = 0.0

    def show_message(self, title, subtitle="", color=S.TEXT_COLOR, duration=1.0):
        self.message = [title, subtitle, color, duration, 0.0]

    def clear_message(self):
        self.message = None

    def update(self, dt):
        self.time += dt
        if self.message:
            self.message[4] += dt
            if self.message[4] >= self.message[3]:
                self.message = None

    # ------------------------------------------------------------------
    def draw(self, surface, match, show_hint=False):
        self._draw_scoreboard(surface, match)
        self._draw_boost(surface, match.cars[0].boost)
        self._draw_small_boost(surface, match.cars[1].boost)
        if show_hint:
            self._draw_hint(surface, match.elapsed_play)
        if self.message:
            self._draw_message(surface)

    def _draw_scoreboard(self, surface, match):
        # Player (top-left), timer (top-centre), CPU (top-right).
        for team, x in ((0, S.ARENA_LEFT), (1, S.ARENA_RIGHT - 280)):
            rect = pygame.Rect(x, 16, 280, 64)
            surface.blit(self.team_panel, rect)
            color = S.PLAYER_COLOR if team == 0 else S.CPU_COLOR
            stripe = pygame.Rect(rect.left + 10, rect.top + 10, 8, rect.height - 20) if team == 0 else \
                pygame.Rect(rect.right - 18, rect.top + 10, 8, rect.height - 20)
            pygame.draw.rect(surface, color, stripe, border_radius=4)
            name = "PLAYER" if team == 0 else "CPU"
            score = str(match.score[team])
            if team == 0:
                draw_text(surface, name, 32, S.TEXT_COLOR, (rect.left + 32, rect.centery), anchor="midleft")
                draw_text(surface, score, 58, color, (rect.right - 24, rect.centery + 2), anchor="midright")
            else:
                draw_text(surface, name, 32, S.TEXT_COLOR, (rect.right - 32, rect.centery), anchor="midright")
                draw_text(surface, score, 58, color, (rect.left + 24, rect.centery + 2), anchor="midleft")

        rect = self.timer_panel.get_rect(midtop=(S.SCREEN_WIDTH // 2, 10))
        surface.blit(self.timer_panel, rect)
        if match.overtime:
            timer_text = "+" + format_time(match.overtime_time)
            sub, sub_color = "OVERTIME", S.GOLD
        else:
            timer_text = format_time(match.time_left)
            sub, sub_color = match.difficulty, S.TEXT_DIM
        urgent = not match.overtime and match.time_left <= 10
        color = S.ACCENT if urgent and int(self.time * 4) % 2 == 0 else S.TEXT_COLOR
        draw_text(surface, timer_text, 56, color, (rect.centerx, rect.top + 30))
        draw_text(surface, sub, 20, sub_color, (rect.centerx, rect.bottom - 14), shadow=False)

    def _draw_boost(self, surface, boost):
        rect = self.boost_panel.get_rect(bottomleft=(18, S.SCREEN_HEIGHT - 10))
        surface.blit(self.boost_panel, rect)
        draw_text(surface, "BOOST", 22, S.TEXT_DIM, (rect.left + 16, rect.top + 14), anchor="midleft", shadow=False)
        draw_text(surface, str(int(boost)), 36, S.BOOST_COLOR, (rect.right - 16, rect.centery + 1), anchor="midright")
        bar = pygame.Rect(rect.left + 16, rect.top + 26, 230, 18)
        filled = boost / S.BOOST_MAX * SEGMENTS
        seg_w = bar.width / SEGMENTS
        for i in range(SEGMENTS):
            seg = pygame.Rect(int(bar.left + i * seg_w) + 1, bar.top, int(seg_w) - 2, bar.height)
            if i < int(filled):
                color = S.BOOST_COLOR if boost > 25 else S.ACCENT
                pygame.draw.rect(surface, color, seg, border_radius=2)
            elif i < filled:
                pygame.draw.rect(surface, (120, 90, 30), seg, border_radius=2)
            else:
                pygame.draw.rect(surface, (40, 44, 70), seg, border_radius=2)
        if boost >= S.BOOST_MAX - 0.5:
            GLOW.draw_alpha(surface, S.BOOST_COLOR, (rect.right - 38, rect.centery), 30, 0.4)

    def _draw_small_boost(self, surface, boost):
        rect = self.small_boost_panel.get_rect(bottomright=(S.SCREEN_WIDTH - 18, S.SCREEN_HEIGHT - 14))
        surface.blit(self.small_boost_panel, rect)
        draw_text(surface, "CPU BOOST", 18, S.TEXT_DIM, (rect.left + 12, rect.centery), anchor="midleft", shadow=False)
        bar = pygame.Rect(rect.left + 100, rect.centery - 6, 86, 12)
        pygame.draw.rect(surface, (40, 44, 70), bar, border_radius=4)
        fill = bar.copy()
        fill.width = int(bar.width * boost / S.BOOST_MAX)
        if fill.width > 0:
            pygame.draw.rect(surface, S.CPU_COLOR, fill, border_radius=4)

    def _draw_hint(self, surface, elapsed):
        remaining = S.CONTROLS_HINT_TIME - elapsed
        if remaining <= 0:
            return
        alpha = int(255 * min(1.0, remaining))
        rect = self.hint_panel.get_rect(midbottom=(S.SCREEN_WIDTH // 2, S.SCREEN_HEIGHT - 16))
        panel = self.hint_panel.copy()
        text = render_text("A/D DRIVE   W JUMP   SPACE BOOST   Q/E AIR ROLL   ESC PAUSE", 18, S.TEXT_COLOR)
        panel.blit(text, text.get_rect(center=panel.get_rect().center))
        panel.set_alpha(alpha)
        surface.blit(panel, rect)

    def _draw_message(self, surface):
        title, subtitle, color, duration, age = self.message
        pop = max(0.0, 1.0 - age / 0.2)
        scale = 1.0 + 0.7 * pop * pop
        fade = min(1.0, (duration - age) / 0.25)
        text_surface, glow, _ = glow_text_surface(title, 150, (255, 255, 255), color)
        center = (S.SCREEN_WIDTH // 2, S.SCREEN_HEIGHT // 2 - 50)
        for image in (glow, glow, text_surface):
            if scale != 1.0:
                image = pygame.transform.smoothscale(image, (int(image.get_width() * scale),
                                                             int(image.get_height() * scale)))
            if fade < 1.0:
                image = image.copy()
                image.set_alpha(int(255 * max(0.0, fade)))
            surface.blit(image, image.get_rect(center=center))
        if subtitle:
            sub = render_text(subtitle, 40, color)
            if fade < 1.0:
                sub = sub.copy()
                sub.set_alpha(int(255 * max(0.0, fade)))
            surface.blit(sub, sub.get_rect(center=(center[0], center[1] + 90)))
