"""Menu screens: main menu, how to play, settings, graphics settings, pause and game over."""
import math

import pygame

import settings as S
from graphics import rounded_panel
from ui import Button, Menu, draw_glow_text, draw_keycap, draw_text

BACK_KEYS = (pygame.K_ESCAPE, pygame.K_BACKSPACE)


def is_back(event):
    return ((event.type == pygame.KEYDOWN and event.key in BACK_KEYS)
            or (event.type == pygame.JOYBUTTONDOWN and event.button == 1))


class Screen:
    def __init__(self, game):
        self.game = game
        self.menu = None
        self._overlay = None

    def on_enter(self):
        pass

    def handle_event(self, event):
        if self.menu:
            self.menu.handle_event(event)

    def update(self, dt):
        if self.menu:
            self.menu.update(dt)

    def draw(self, surface):
        pass

    def dim(self, surface, alpha=150):
        # Cached overlay: allocating a full-screen surface every frame is wasteful.
        if self._overlay is None or self._overlay.get_alpha() != alpha:
            self._overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            self._overlay.fill((4, 6, 18, alpha))
        surface.blit(self._overlay, (0, 0))


# ----------------------------------------------------------------------
class MainMenuScreen(Screen):
    def __init__(self, game):
        super().__init__(game)
        g = game
        self.menu = Menu([
            Button("PLAY MATCH", g.start_match),
            Button("DIFFICULTY", on_change=self._change_difficulty,
                   value=lambda: g.save.settings["difficulty"]),
            Button("HOW TO PLAY", lambda: g.change_state(g.STATES.HOW_TO_PLAY)),
            Button("SETTINGS", lambda: g.open_settings(g.STATES.MAIN_MENU)),
            Button("QUIT", g.quit),
        ], S.SCREEN_WIDTH // 2, 330, spacing=66, audio=g.audio)

    def _change_difficulty(self, direction):
        names = S.DIFFICULTIES
        current = names.index(self.game.save.settings["difficulty"])
        self.game.save.settings["difficulty"] = names[(current + direction) % len(names)]
        self.game.save.save()

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return  # nothing to go back to; use QUIT
        super().handle_event(event)

    def draw(self, surface):
        self.dim(surface, 140)
        t = self.game.time
        draw_glow_text(surface, S.GAME_TITLE, 130, (255, 255, 255), S.PLAYER_COLOR,
                       (S.SCREEN_WIDTH // 2, 150 + math.sin(t * 1.6) * 4))
        draw_text(surface, S.GAME_SUBTITLE, 34, S.CPU_COLOR, (S.SCREEN_WIDTH // 2, 232))
        self.menu.draw(surface, t)
        st = self.game.save.stats
        footer = (f"MATCHES {st['matches_played']}    WINS {st['wins']}    LOSSES {st['losses']}    "
                  f"GOALS {st['goals_scored']} : {st['goals_conceded']}")
        draw_text(surface, footer, 24, S.TEXT_DIM, (S.SCREEN_WIDTH // 2, S.SCREEN_HEIGHT - 60))
        draw_text(surface, "ARROWS / WASD + ENTER, MOUSE OR GAMEPAD", 20, S.TEXT_DIM,
                  (S.SCREEN_WIDTH // 2, S.SCREEN_HEIGHT - 30), shadow=False)
        r = self.game.renderer
        label = "RENDERER: GPU (OPENGL)" if r.gpu else "RENDERER: CPU"
        draw_text(surface, label, 18, S.TEXT_DIM, (S.SCREEN_WIDTH - 16, S.SCREEN_HEIGHT - 30), anchor="midright",
                  shadow=False)


# ----------------------------------------------------------------------
class HowToPlayScreen(Screen):
    CONTROLS = [
        (["A", "D"], ["LEFT", "RIGHT"], "Drive left / right"),
        (["W"], ["UP"], "Jump  (again in the air = double jump)"),
        (["W", "+ A/D"], None, "In the air: flip for a power hit"),
        (["S"], ["DOWN"], "Brake / reverse  -  dive in the air"),
        (["SPACE"], None, "Boost (pushes where the nose points)"),
        (["Q", "E"], None, "Air roll: rotate the car in the air"),
        (["ESC"], None, "Pause"),
    ]
    TIPS = [
        "Score more goals than your opponent",
        "before time runs out.",
        "",
        "Tied at 0:00? OVERTIME - next goal wins!",
        "",
        "Fly: jump, tilt the nose up with Q/E,",
        "then hold SPACE to boost into the air.",
        "",
        "Flip into the ball for your hardest shots.",
        "Drive fast to stick to walls and the ceiling.",
        "Boost slowly refills when not in use.",
    ]

    def __init__(self, game):
        super().__init__(game)
        self.menu = Menu([Button("BACK", self.back, width=260)], S.SCREEN_WIDTH // 2, 650, audio=game.audio)
        self.panel = rounded_panel((1140, 500), S.PANEL_COLOR, S.PANEL_BORDER, radius=18, alpha=225)

    def back(self):
        self.game.change_state(self.game.STATES.MAIN_MENU)

    def handle_event(self, event):
        if is_back(event):
            self.game.audio.play("menu_back")
            self.back()
            return
        super().handle_event(event)

    def draw(self, surface):
        self.dim(surface, 150)
        draw_glow_text(surface, "HOW TO PLAY", 70, (255, 255, 255), S.PLAYER_COLOR, (S.SCREEN_WIDTH // 2, 60))
        panel_rect = self.panel.get_rect(midtop=(S.SCREEN_WIDTH // 2, 110))
        surface.blit(self.panel, panel_rect)
        x0, y = panel_rect.left + 40, panel_rect.top + 30
        draw_text(surface, "CONTROLS", 34, S.GOLD, (x0, y), anchor="topleft")
        y += 52
        for keys, alt, text in self.CONTROLS:
            x = x0
            for i, key in enumerate(keys):
                if i and not key.startswith("+"):
                    draw_text(surface, "/", 24, S.TEXT_DIM, (x + 6, y + 17))
                    x += 14
                if key.startswith("+"):
                    draw_text(surface, key, 24, S.TEXT_DIM, (x + 6, y + 17), anchor="midleft")
                    x += 70
                    continue
                x += draw_keycap(surface, key, x, y) + 6
            if alt:
                draw_text(surface, "or", 20, S.TEXT_DIM, (x + 16, y + 17))
                x += 32
                for key in alt:
                    x += draw_keycap(surface, key, x, y) + 6
            draw_text(surface, text, 24, S.TEXT_COLOR, (x0 + 230, y + 17), anchor="midleft")
            y += 52
        x1 = panel_rect.left + 720
        y = panel_rect.top + 30
        draw_text(surface, "HOW TO WIN", 34, S.GOLD, (x1, y), anchor="topleft")
        y += 52
        for line in self.TIPS:
            if line:
                draw_text(surface, line, 24, S.TEXT_COLOR, (x1, y), anchor="topleft")
            y += 30
        self.menu.draw(surface, self.game.time)


# ----------------------------------------------------------------------
def _on_off(value):
    return "ON" if value else "OFF"


class SettingsScreen(Screen):
    def __init__(self, game):
        super().__init__(game)
        g = game
        s = g.save.settings
        self.return_state = g.STATES.MAIN_MENU
        self.confirm_reset = False
        self.menu = Menu([
            Button("MUSIC VOLUME", on_change=lambda d: self._volume("music_volume", d),
                   value=lambda: f"{round(s['music_volume'] * 100)}%", width=560),
            Button("SOUND EFFECTS", on_change=lambda d: self._volume("sfx_volume", d),
                   value=lambda: f"{round(s['sfx_volume'] * 100)}%", width=560),
            Button("FULLSCREEN", on_change=lambda d: g.toggle_fullscreen(),
                   value=lambda: _on_off(s["fullscreen"]), width=560),
            Button("SCREEN SHAKE", on_change=lambda d: self._toggle("screen_shake"),
                   value=lambda: _on_off(s["screen_shake"]), width=560),
            Button("MATCH LENGTH", on_change=self._match_length,
                   value=lambda: f"{s['match_minutes']} MIN", width=560),
            Button("GRAPHICS SETTINGS  >", lambda: g.change_state(g.STATES.GRAPHICS), width=560),
            Button(lambda: "PRESS AGAIN TO CONFIRM" if self.confirm_reset else "RESET STATISTICS",
                   self._reset_stats, width=560),
            Button("BACK", self.back, width=560),
        ], S.SCREEN_WIDTH // 2, 190, spacing=62, audio=g.audio)
        self.panel = rounded_panel((640, 540), S.PANEL_COLOR, S.PANEL_BORDER, radius=18, alpha=215)

    def on_enter(self):
        self.confirm_reset = False

    def _volume(self, key, direction):
        s = self.game.save.settings
        s[key] = round(min(1.0, max(0.0, s[key] + 0.1 * direction)), 2)
        self.game.apply_audio_settings()
        self.game.save.save()

    def _toggle(self, key):
        s = self.game.save.settings
        s[key] = not s[key]
        self.game.apply_gameplay_settings()
        self.game.save.save()

    def _match_length(self, direction):
        s = self.game.save.settings
        options = S.MATCH_LENGTH_OPTIONS
        s["match_minutes"] = options[(options.index(s["match_minutes"]) + direction) % len(options)]
        self.game.save.save()

    def _reset_stats(self):
        if self.confirm_reset:
            self.game.save.reset_stats()
            self.confirm_reset = False
        else:
            self.confirm_reset = True

    def back(self):
        self.game.save.save()
        self.game.change_state(self.return_state)

    def handle_event(self, event):
        if is_back(event):
            self.game.audio.play("menu_back")
            self.back()
            return
        super().handle_event(event)

    def draw(self, surface):
        self.dim(surface, 170)
        draw_glow_text(surface, "SETTINGS", 76, (255, 255, 255), S.PLAYER_COLOR, (S.SCREEN_WIDTH // 2, 70))
        surface.blit(self.panel, self.panel.get_rect(midtop=(S.SCREEN_WIDTH // 2, 140)))
        self.menu.draw(surface, self.game.time)
        if self.game.STATES.MAIN_MENU != self.return_state:
            draw_text(surface, "Match length applies to the next match.", 20, S.TEXT_DIM,
                      (S.SCREEN_WIDTH // 2, S.SCREEN_HEIGHT - 26), shadow=False)


# ----------------------------------------------------------------------
class GraphicsScreen(Screen):
    DESCRIPTIONS = {
        "renderer": "GPU uses OpenGL shaders for all effects. CPU is a simple fallback. Applies after restart.",
        "graphics_preset": "Sets every option below at once. Changing any option switches to CUSTOM.",
        "reflections": "Glossy mirror reflections on the arena floor, blurring with distance.",
        "ambient_occlusion": "Soft contact shadows where cars, ball, walls and corners meet.",
        "bloom": "Bright neon lights, boost flames and explosions glow and bleed light.",
        "light_shafts": "Volumetric light rays from the stadium ceiling lights.",
        "motion_blur": "Blends frames together for a sense of speed.",
        "antialiasing": "FXAA smooths jagged edges.",
        "chromatic_aberration": "Subtle lens colour fringing that spikes on big impacts.",
        "vignette": "Darkens the screen corners to focus on the action.",
        "color_grading": "Filmic tone curve, richer colours and cooler shadows.",
        "particles": "Amount of sparks, boost trail, dust and confetti.",
        "fps_cap": "Maximum frames per second. Physics always runs at a fixed rate.",
        "show_fps": "Shows the current frame rate in the top-left corner.",
        "back": "",
    }

    def __init__(self, game):
        super().__init__(game)
        g = game
        s = g.save.settings
        self.keys = []
        buttons = []

        def add(key, button):
            self.keys.append(key)
            buttons.append(button)

        add("renderer", Button("RENDERER", on_change=self._renderer, value=self._renderer_value, width=640, height=34))
        add("graphics_preset", Button("QUALITY PRESET", on_change=self._preset,
                                      value=lambda: s["graphics_preset"], width=640, height=34))
        for key, (label, levels) in S.GRAPHICS_OPTIONS.items():
            add(key, Button(label, on_change=lambda d, k=key: self._option(k, d),
                            value=lambda k=key: self._option_value(k), width=640, height=34))
        add("fps_cap", Button("FRAME RATE CAP", on_change=self._fps_cap,
                              value=lambda: f"{s['fps_cap']} FPS" if s["fps_cap"] else "UNLIMITED",
                              width=640, height=34))
        add("show_fps", Button("SHOW FPS", on_change=lambda d: self._flip("show_fps"),
                               value=lambda: _on_off(s["show_fps"]), width=640, height=34))
        add("back", Button("BACK", self.back, width=640, height=34))
        self.menu = Menu(buttons, S.SCREEN_WIDTH // 2, 118, spacing=38.5, audio=g.audio)
        self.panel = rounded_panel((720, 600), S.PANEL_COLOR, S.PANEL_BORDER, radius=18, alpha=220)

    # --- option handlers ----------------------------------------------
    def _renderer_value(self):
        wanted = self.game.save.settings["renderer"]
        active = self.game.renderer.name
        if wanted != active:
            return f"{wanted} (RESTART)"
        return "GPU (OPENGL)" if wanted == "GPU" else "CPU (CLASSIC)"

    def _renderer(self, direction):
        s = self.game.save.settings
        s["renderer"] = "CPU" if s["renderer"] == "GPU" else "GPU"
        self.game.save.save()

    def _preset(self, direction):
        s = self.game.save.settings
        names = S.PRESET_NAMES[:-1]  # CUSTOM is not selectable directly
        current = names.index(s["graphics_preset"]) if s["graphics_preset"] in names else 1
        name = names[(current + direction) % len(names)]
        s["graphics_preset"] = name
        s.update(S.GRAPHICS_PRESETS[name])
        self.game.apply_graphics_settings()

    def _option(self, key, direction):
        s = self.game.save.settings
        levels = S.GRAPHICS_OPTIONS[key][1]
        s[key] = (s[key] + direction) % len(levels)
        s["graphics_preset"] = "CUSTOM"
        for name in S.PRESET_NAMES[:-1]:
            if all(s[k] == v for k, v in S.GRAPHICS_PRESETS[name].items()):
                s["graphics_preset"] = name
        self.game.apply_graphics_settings()

    def _option_value(self, key):
        levels = S.GRAPHICS_OPTIONS[key][1]
        text = levels[self.game.save.settings[key]]
        if not self.game.renderer.gpu and key not in ("reflections", "particles") and text != "OFF":
            text += " (GPU ONLY)"
        return text

    def _fps_cap(self, direction):
        s = self.game.save.settings
        options = S.FPS_CAP_OPTIONS
        s["fps_cap"] = options[(options.index(s["fps_cap"]) + direction) % len(options)]
        self.game.save.save()

    def _flip(self, key):
        s = self.game.save.settings
        s[key] = not s[key]
        self.game.save.save()

    def back(self):
        self.game.save.save()
        self.game.change_state(self.game.STATES.SETTINGS)

    def handle_event(self, event):
        if is_back(event):
            self.game.audio.play("menu_back")
            self.back()
            return
        super().handle_event(event)

    def draw(self, surface):
        self.dim(surface, 120)
        draw_glow_text(surface, "GRAPHICS", 64, (255, 255, 255), S.PLAYER_COLOR, (S.SCREEN_WIDTH // 2, 52))
        surface.blit(self.panel, self.panel.get_rect(midtop=(S.SCREEN_WIDTH // 2, 92)))
        self.menu.draw(surface, self.game.time)
        key = self.keys[self.menu.selected]
        description = self.DESCRIPTIONS.get(key, "")
        if key == "renderer" and self.game.renderer.gpu_error:
            description = "OpenGL could not start, so the CPU renderer is active. See README."
        draw_text(surface, description, 22, S.TEXT_COLOR, (S.SCREEN_WIDTH // 2, S.SCREEN_HEIGHT - 22))


# ----------------------------------------------------------------------
class PauseScreen(Screen):
    def __init__(self, game):
        super().__init__(game)
        g = game
        self.menu = Menu([
            Button("RESUME", g.resume),
            Button("RESTART", g.restart_match),
            Button("SETTINGS", lambda: g.open_settings(g.STATES.PAUSED)),
            Button("MAIN MENU", g.go_to_menu),
        ], S.SCREEN_WIDTH // 2, 300, spacing=68, audio=g.audio)

    def on_enter(self):
        self.menu.selected = 0

    def handle_event(self, event):
        if (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_p)) or \
                (event.type == pygame.JOYBUTTONDOWN and event.button in (7, 9)):
            self.game.resume()
            return
        super().handle_event(event)

    def draw(self, surface):
        self.dim(surface, 160)
        draw_glow_text(surface, "PAUSED", 110, (255, 255, 255), S.PLAYER_COLOR, (S.SCREEN_WIDTH // 2, 170))
        self.menu.draw(surface, self.game.time)


# ----------------------------------------------------------------------
class GameOverScreen(Screen):
    def __init__(self, game):
        super().__init__(game)
        g = game
        self.menu = Menu([
            Button("PLAY AGAIN", g.restart_match),
            Button("MAIN MENU", g.go_to_menu),
        ], S.SCREEN_WIDTH // 2, 470, spacing=68, audio=g.audio)
        self.won = False
        self.score = (0, 0)
        self.age = 0.0

    def setup(self, score):
        self.score = tuple(score)
        self.won = score[0] > score[1]
        self.age = 0.0
        self.menu.selected = 0

    def update(self, dt):
        super().update(dt)
        self.age += dt

    def handle_event(self, event):
        if self.age < 0.6:
            return  # avoid skipping the result by accident while still pressing keys
        super().handle_event(event)

    def draw(self, surface):
        self.dim(surface, 150)
        pop = max(0.0, 1.0 - self.age / 0.3)
        title = "VICTORY!" if self.won else "DEFEAT"
        color = S.PLAYER_COLOR if self.won else S.CPU_COLOR
        draw_glow_text(surface, title, 130, (255, 255, 255), color, (S.SCREEN_WIDTH // 2, 150),
                       scale=1.0 + 0.5 * pop * pop)
        draw_text(surface, "PLAYER", 36, S.PLAYER_COLOR, (S.SCREEN_WIDTH // 2 - 170, 290))
        draw_text(surface, "CPU", 36, S.CPU_COLOR, (S.SCREEN_WIDTH // 2 + 170, 290))
        draw_text(surface, f"{self.score[0]}   -   {self.score[1]}", 110, S.TEXT_COLOR, (S.SCREEN_WIDTH // 2, 300))
        st = self.game.save.stats
        draw_text(surface, f"CAREER RECORD   {st['wins']} W  -  {st['losses']} L", 26, S.TEXT_DIM,
                  (S.SCREEN_WIDTH // 2, 385))
        self.menu.draw(surface, self.game.time)
