"""
The Game class: window, main loop and the game-state machine.

States
    MAIN_MENU   -> title screen (an AI-vs-AI match plays in the background)
    HOW_TO_PLAY -> controls screen
    SETTINGS    -> audio / gameplay settings (from the main menu or pause menu)
    GRAPHICS    -> graphics settings
    COUNTDOWN   -> "3, 2, 1, GO!" before every kickoff
    PLAYING     -> the match is running
    GOAL        -> goal celebration, then back to COUNTDOWN (or GAME_OVER)
    PAUSED      -> pause menu, physics frozen
    GAME_OVER   -> results screen
"""
import math
from enum import Enum, auto

import pygame

import settings as S
from arena import Arena
from audio import AudioManager
from hud import HUD
from match import Match
from player import PAD_PAUSE_BUTTONS, PlayerController
from renderer import create_renderer
from save_data import SaveData
from screens import (GameOverScreen, GraphicsScreen, HowToPlayScreen, MainMenuScreen, PauseScreen,
                     SettingsScreen)
from ui import draw_text


class GameState(Enum):
    MAIN_MENU = auto()
    HOW_TO_PLAY = auto()
    SETTINGS = auto()
    GRAPHICS = auto()
    COUNTDOWN = auto()
    PLAYING = auto()
    GOAL = auto()
    PAUSED = auto()
    GAME_OVER = auto()


MATCH_STATES = (GameState.COUNTDOWN, GameState.PLAYING, GameState.GOAL)
MENU_STATES = (GameState.MAIN_MENU, GameState.HOW_TO_PLAY, GameState.SETTINGS, GameState.GRAPHICS)


def _make_icon():
    icon = pygame.Surface((32, 32), pygame.SRCALPHA)
    pygame.draw.circle(icon, S.BALL_COLOR, (16, 16), 14)
    pygame.draw.circle(icon, S.PLAYER_COLOR, (16, 16), 14, 3)
    pygame.draw.polygon(icon, S.CPU_COLOR, [(10, 9), (20, 16), (10, 23)])
    return icon


class Game:
    STATES = GameState

    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        pygame.display.set_caption(f"{S.GAME_TITLE} - {S.GAME_SUBTITLE.title()}")
        try:
            pygame.display.set_icon(_make_icon())
        except pygame.error:
            pass

        self.save = SaveData()
        settings = self.save.settings
        self.renderer = create_renderer(settings)
        settings["fullscreen"] = self.renderer.fullscreen
        self.clock = pygame.time.Clock()
        self.audio = AudioManager(settings["music_volume"], settings["sfx_volume"])
        self.arena = Arena()
        self.arena.build_graphics()
        self.player_input = PlayerController()
        self.hud = HUD()

        self.demo = Match(self.arena, None, "HARD", 2, demo=True)
        self.match = None
        self.screens = {
            GameState.MAIN_MENU: MainMenuScreen(self),
            GameState.HOW_TO_PLAY: HowToPlayScreen(self),
            GameState.SETTINGS: SettingsScreen(self),
            GameState.GRAPHICS: GraphicsScreen(self),
            GameState.PAUSED: PauseScreen(self),
            GameState.GAME_OVER: GameOverScreen(self),
        }
        self.state = GameState.MAIN_MENU
        self.paused_from = GameState.PLAYING
        self.countdown = 0.0
        self.goal_timer = 0.0
        self.time = 0.0
        self.fade = 1.0
        self.fps_display = 0.0
        self._last_count = None
        self._fade_surface = None
        self.running = True
        self.apply_graphics_settings()
        self.apply_gameplay_settings()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def apply_audio_settings(self):
        s = self.save.settings
        self.audio.set_volumes(s["music_volume"], s["sfx_volume"])

    def apply_gameplay_settings(self):
        for match in (self.match, self.demo):
            if match:
                match.shake_enabled = self.save.settings["screen_shake"]

    def apply_graphics_settings(self):
        self.save.save()
        for match in (self.match, self.demo):
            if match:
                match.set_particle_quality(self.save.settings["particles"])

    def toggle_fullscreen(self):
        self.save.settings["fullscreen"] = self.renderer.toggle_fullscreen()
        self.save.save()

    # ------------------------------------------------------------------
    # State changes
    # ------------------------------------------------------------------
    def change_state(self, state):
        self.state = state
        screen = self.screens.get(state)
        if screen:
            screen.on_enter()
        self.fade = max(self.fade, 0.35)
        in_match = state in MATCH_STATES or state == GameState.GAME_OVER
        self.audio.set_music_duck(0.55 if in_match else 1.0)

    def open_settings(self, return_state):
        self.screens[GameState.SETTINGS].return_state = return_state
        self.change_state(GameState.SETTINGS)

    def start_match(self):
        s = self.save.settings
        self.match = Match(self.arena, self.audio, s["difficulty"], s["match_minutes"],
                           player_controller=self.player_input)
        self.match.shake_enabled = s["screen_shake"]
        self.match.set_particle_quality(s["particles"])
        self.hud.clear_message()
        self.fade = 1.0
        self._begin_countdown()

    def restart_match(self):
        if self.match:
            self.match.stop_sounds()
        self.start_match()

    def _begin_countdown(self):
        self.countdown = 3 * S.KICKOFF_STEP_TIME
        self._last_count = None
        self.change_state(GameState.COUNTDOWN)

    def pause(self):
        if self.state in MATCH_STATES:
            self.paused_from = self.state
            self.match.stop_sounds()
            self.change_state(GameState.PAUSED)
            self.fade = 0.0

    def resume(self):
        self.player_input.reset()
        self.change_state(self.paused_from)
        self.fade = 0.0

    def go_to_menu(self):
        if self.match:
            self.match.stop_sounds()
        self.match = None
        self.hud.clear_message()
        self.change_state(GameState.MAIN_MENU)

    def quit(self):
        self.running = False

    def _on_goal(self, team):
        self.goal_timer = S.GOAL_CELEBRATION_TIME
        name = "PLAYER SCORES!" if team == 0 else "CPU SCORES!"
        color = S.PLAYER_COLOR if team == 0 else S.CPU_COLOR
        score = self.match.score
        self.hud.show_message("GOAL!", f"{name}    {score[0]} - {score[1]}", color, S.GOAL_CELEBRATION_TIME - 0.2)
        self.change_state(GameState.GOAL)

    def _after_goal(self):
        m = self.match
        if m.overtime or m.time_left <= 0:
            if m.score[0] != m.score[1]:
                self._end_match()
                return
            self._start_overtime()
            return
        m.reset_kickoff()
        self._begin_countdown()

    def _start_overtime(self):
        self.match.start_overtime()
        self.match.reset_kickoff()
        self.audio.play("whistle")
        self.hud.show_message("OVERTIME", "NEXT GOAL WINS", S.GOLD, 1.2)
        self._begin_countdown()
        self.countdown += 1.2   # extra time to read the banner before "3, 2, 1"

    def _end_match(self):
        m = self.match
        m.stop_sounds()
        self.save.record_match(m.score[0], m.score[1])
        self.screens[GameState.GAME_OVER].setup(m.score)
        self.audio.play("whistle")
        self.audio.play("win" if m.score[0] > m.score[1] else "lose")
        self.hud.clear_message()
        self.change_state(GameState.GAME_OVER)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        while self.running:
            cap = self.save.settings["fps_cap"]
            dt = self.clock.tick(cap if cap else 0) / 1000.0
            dt = min(dt, S.MAX_FRAME_TIME)
            self.process_input()
            self.update(dt)
            self.draw()
        self.shutdown()

    def shutdown(self):
        if self.match:
            self.match.stop_sounds()
        self.save.save()
        pygame.quit()

    def process_input(self):
        for event in pygame.event.get():
            if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                event = pygame.event.Event(event.type, {**event.dict,
                                                        "pos": self.renderer.window_to_internal(event.pos)})
            self.handle_event(event)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
            self.player_input.handle_event(event)
            return
        if event.type == pygame.WINDOWFOCUSLOST and self.state in MATCH_STATES:
            self.pause()
            return
        if self.state in MATCH_STATES:
            is_pause = ((event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_p))
                        or (event.type == pygame.JOYBUTTONDOWN and event.button in PAD_PAUSE_BUTTONS))
            if is_pause:
                self.pause()
            else:
                self.player_input.handle_event(event)
            return
        screen = self.screens.get(self.state)
        if screen:
            screen.handle_event(event)

    def update(self, dt):
        self.time += dt
        self.fade = max(0.0, self.fade - dt * 3.0)
        self.audio.update()
        self.hud.update(dt)
        state = self.state
        screen = self.screens.get(state)
        if screen:
            screen.update(dt)

        if state == GameState.COUNTDOWN:
            self._update_countdown(dt)
        elif state == GameState.PLAYING:
            self._update_playing(dt)
        elif state == GameState.GOAL:
            self._update_goal(dt)
        elif state == GameState.GAME_OVER:
            self.match.update(dt, cars_active=False, ball_active=False)
            if self.screens[GameState.GAME_OVER].won and self.screens[GameState.GAME_OVER].age < 6.0:
                self.match.celebrate(0)
        elif self._showing_demo():
            self.demo.update(dt)

    def _showing_demo(self):
        if self.state in (GameState.SETTINGS, GameState.GRAPHICS):
            return self.screens[GameState.SETTINGS].return_state == GameState.MAIN_MENU or self.match is None
        return self.state in MENU_STATES

    def _update_countdown(self, dt):
        self.match.update(dt, cars_active=False)
        self.countdown -= dt
        count = int(math.ceil(self.countdown / S.KICKOFF_STEP_TIME))
        if count != self._last_count and 1 <= count <= 3:
            self._last_count = count
            self.audio.play("countdown")
            self.hud.show_message(str(count), "", (120, 200, 255), S.KICKOFF_STEP_TIME)
        if self.countdown <= 0:
            self.match.release_ball()
            self.audio.play("go")
            self.hud.show_message("GO!", "", S.GOLD, 0.6)
            self.change_state(GameState.PLAYING)

    def _update_playing(self, dt):
        m = self.match
        events = m.update(dt)
        m.tick_clock(dt)
        for event in events:
            if event[0] == "goal":
                self._on_goal(event[1])
                return
        if not m.overtime and m.time_left <= 0:
            if m.score[0] != m.score[1]:
                self._end_match()
            else:
                self._start_overtime()

    def _update_goal(self, dt):
        elapsed = S.GOAL_CELEBRATION_TIME - self.goal_timer
        scale = S.GOAL_SLOWMO_SCALE if elapsed < S.GOAL_SLOWMO_TIME else 1.0
        self.match.update(dt * scale, cars_active=True, ball_active=False)
        self.goal_timer -= dt
        if self.goal_timer <= 0:
            self._after_goal()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw(self):
        renderer = self.renderer
        world = renderer.world_surface()
        match = self.demo if self._showing_demo() else self.match
        match.draw(world, show_player_marker=self.state != GameState.GAME_OVER)
        impact = match.shake / S.SCREEN_SHAKE_MAX if match.shake_enabled else 0.0
        renderer.finish_world(match.occluders(), {"impact": min(1.0, impact)})

        ui = renderer.ui_surface()
        if match is self.match and self.state != GameState.GAME_OVER:
            self.hud.draw(ui, match, show_hint=self.state in (GameState.PLAYING, GameState.COUNTDOWN))
        screen = self.screens.get(self.state)
        if screen:
            screen.draw(ui)
        if self.save.settings["show_fps"]:
            self.fps_display += (self.clock.get_fps() - self.fps_display) * 0.1
            draw_text(ui, f"{self.fps_display:.0f} FPS", 20, S.GOLD, (12, 8), anchor="topleft")
        if self.fade > 0:
            if self._fade_surface is None:
                self._fade_surface = pygame.Surface(ui.get_size(), pygame.SRCALPHA)
            self._fade_surface.fill((0, 0, 0, int(255 * min(1.0, self.fade))))
            ui.blit(self._fade_surface, (0, 0))
        renderer.present()
