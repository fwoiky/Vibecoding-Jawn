"""
Automated checks for Nitro Arena. They run without a visible window or sound:

    python -m unittest discover -s tests -v
"""
import os
import random
import sys
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame  # noqa: E402

pygame.init()

import settings as S  # noqa: E402
from arena import Arena  # noqa: E402
from car import Car  # noqa: E402
from controls import Controls  # noqa: E402
from match import Match  # noqa: E402
from opponent import AIController  # noqa: E402
from save_data import SaveData  # noqa: E402

DT = S.PHYSICS_DT


def make_car():
    car = Car(0, S.PLAYER_COLOR, S.PLAYER_ACCENT, S.PLAYER_DARK, "coupe")
    car.reset(400, 1)
    return car


def step(car, arena, controls, seconds):
    for i in range(int(seconds / DT)):
        car.update(controls, DT)
        car.collide_arena(arena)
        car.update_ground_state(arena)
        controls.jump_pressed = False


class CarPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.arena = Arena()
        self.car = make_car()

    def test_car_rests_on_floor(self):
        step(self.car, self.arena, Controls(), 1.0)
        self.assertTrue(self.car.grounded)
        self.assertAlmostEqual(self.car.pos.y, S.ARENA_FLOOR - S.CAR_RADIUS, delta=1.0)

    def test_drives_and_turns_around(self):
        step(self.car, self.arena, Controls(move_x=1), 1.0)
        self.assertGreater(self.car.vel.x, S.CAR_MAX_SPEED * 0.9)
        step(self.car, self.arena, Controls(move_x=-1), 1.0)
        self.assertEqual(self.car.facing, -1)
        self.assertLess(self.car.vel.x, 0)

    def test_boost_is_faster_and_drains(self):
        self.car.boost = 100
        step(self.car, self.arena, Controls(move_x=1, boost=True), 0.6)
        self.assertGreater(self.car.vel.x, S.CAR_MAX_SPEED + 100)
        self.assertLess(self.car.boost, 100)

    def test_jump_and_double_jump(self):
        c = Controls(jump_pressed=True, jump_held=True)
        best_single = 0
        for i in range(int(1.2 / DT)):
            self.car.update(c, DT)
            self.car.collide_arena(self.arena)
            self.car.update_ground_state(self.arena)
            c.jump_pressed = False
            best_single = max(best_single, S.ARENA_FLOOR - self.car.pos.y)
        self.assertGreater(best_single, 100)
        self.assertTrue(self.car.grounded)
        self.car.reset(400, 1)
        best_double = 0
        for i in range(int(1.6 / DT)):
            c = Controls(jump_pressed=i in (0, 30), jump_held=True)
            self.car.update(c, DT)
            self.car.collide_arena(self.arena)
            self.car.update_ground_state(self.arena)
            best_double = max(best_double, S.ARENA_FLOOR - self.car.pos.y)
        self.assertGreater(best_double, best_single + 50)

    def test_upside_down_car_rights_itself(self):
        self.car.pos.y = 400
        self.car.angle = 3.1
        self.car.grounded = False
        step(self.car, self.arena, Controls(), 2.0)
        self.assertTrue(self.car.grounded)
        self.assertLess(abs(self.car.angle), 0.1)

    def test_fast_car_drives_up_wall_onto_ceiling(self):
        import math
        car = self.car
        car.pos.update(S.ARENA_RIGHT - S.CAR_RADIUS - 0.5, 380)
        car.angle = -math.pi / 2
        car.vel.update(0, -600)
        car.ground_normal.update(-1, 0)
        car.grounded = True
        car.boost = 100
        step(car, self.arena, Controls(move_x=1, boost=True), 1.0)
        self.assertTrue(car.grounded)
        self.assertGreater(car.ground_normal.y, 0.9)   # on the ceiling
        self.assertLess(car.pos.y, S.ARENA_TOP + 30)


class BallAndArenaTests(unittest.TestCase):
    def test_ball_bounces_and_settles(self):
        m = Match(Arena(), None, "NORMAL", 2)
        m.release_ball()
        heights = []
        for _ in range(int(6 / (1 / 60))):
            m.update(1 / 60, cars_active=False)
            heights.append(m.ball.pos.y)
        self.assertLess(min(heights), S.ARENA_FLOOR - S.BALL_KICKOFF_HEIGHT + 5)
        self.assertAlmostEqual(m.ball.pos.y, S.ARENA_FLOOR - S.BALL_RADIUS, delta=2.0)

    def test_goal_detection(self):
        arena = Arena()
        from pygame.math import Vector2
        self.assertEqual(arena.goal_scored(Vector2(S.ARENA_RIGHT + 40, 600), S.BALL_RADIUS), 0)
        self.assertEqual(arena.goal_scored(Vector2(S.ARENA_LEFT - 40, 600), S.BALL_RADIUS), 1)
        self.assertIsNone(arena.goal_scored(Vector2(S.ARENA_CENTER_X, 600), S.BALL_RADIUS))

    def test_car_hits_ball(self):
        m = Match(Arena(), None, "NORMAL", 2)
        m.ai = [None, None]
        m.ball.pos.update(700, S.ARENA_FLOOR - S.BALL_RADIUS)
        m.release_ball()
        m.cars[0].reset(500, 1)
        m.cars[1].reset(1100, -1)
        hits = []
        m.player_controller = None
        for _ in range(90):
            m.cars[0].vel.x = max(m.cars[0].vel.x, 500)
            hits += [e for e in m.update(1 / 60) if e[0] == "hit"]
        self.assertTrue(hits)
        self.assertGreater(m.ball.vel.x, 300)


class MatchSimulationTests(unittest.TestCase):
    def test_ai_vs_ai_match_is_stable_and_scores(self):
        random.seed(3)
        m = Match(Arena(), None, "HARD", 2, demo=False)
        m.ai[0] = AIController(m.cars[0], m.ball, "NORMAL")
        m.release_ball()
        goals = 0
        for _ in range(60 * 90):
            for event in m.update(1 / 60):
                if event[0] == "goal":
                    goals += 1
                    m.reset_kickoff()
                    m.release_ball()
            for body in m.cars + [m.ball]:
                self.assertTrue(S.ARENA_LEFT - S.GOAL_DEPTH - 5 < body.pos.x < S.ARENA_RIGHT + S.GOAL_DEPTH + 5)
                self.assertTrue(S.ARENA_TOP - 5 < body.pos.y < S.ARENA_FLOOR + 5)
            self.assertLessEqual(m.ball.vel.length(), S.BALL_MAX_SPEED + 1)
        self.assertGreater(goals, 0)


class SaveDataTests(unittest.TestCase):
    def test_round_trip_and_bad_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "save.json")
            data = SaveData(path)
            data.settings["music_volume"] = 0.3
            data.record_match(3, 1)
            again = SaveData(path)
            self.assertEqual(again.settings["music_volume"], 0.3)
            self.assertEqual(again.stats["wins"], 1)
            with open(path, "w") as f:
                f.write("{not json")
            broken = SaveData(path)
            self.assertEqual(broken.settings["difficulty"], S.DEFAULT_DIFFICULTY)


class GameFlowTests(unittest.TestCase):
    """Runs the real Game object (CPU renderer, no window) through every state."""

    def setUp(self):
        import save_data
        self.tmp = tempfile.TemporaryDirectory()
        self._orig = save_data.SaveData.__init__.__defaults__
        save_data.SaveData.__init__.__defaults__ = (os.path.join(self.tmp.name, "save.json"),)
        from game import Game
        self.Game = Game

    def tearDown(self):
        import save_data
        save_data.SaveData.__init__.__defaults__ = self._orig
        self.tmp.cleanup()

    def frames(self, game, n):
        for _ in range(n):
            game.process_input()
            game.update(1 / 60)
            game.draw()

    def press(self, key):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0, unicode="", scancode=0))

    def test_full_flow(self):
        game = self.Game()
        game.save.settings["renderer"] = "CPU"
        S_ = game.STATES
        self.frames(game, 5)
        self.assertEqual(game.state, S_.MAIN_MENU)
        # Menu navigation: HOW TO PLAY and back.
        self.press(pygame.K_DOWN)
        self.press(pygame.K_DOWN)
        self.press(pygame.K_RETURN)
        self.frames(game, 2)
        self.assertEqual(game.state, S_.HOW_TO_PLAY)
        self.press(pygame.K_ESCAPE)
        self.frames(game, 2)
        self.assertEqual(game.state, S_.MAIN_MENU)
        # Settings -> graphics -> change an option -> back.
        game.open_settings(S_.MAIN_MENU)
        game.change_state(S_.GRAPHICS)
        screen = game.screens[S_.GRAPHICS]
        screen.menu.selected = screen.keys.index("bloom")
        self.press(pygame.K_RIGHT)
        self.frames(game, 2)
        self.assertEqual(game.save.settings["graphics_preset"], "CUSTOM")
        self.press(pygame.K_ESCAPE)
        self.press(pygame.K_ESCAPE)
        self.frames(game, 2)
        self.assertEqual(game.state, S_.MAIN_MENU)
        # Play a match.
        game.start_match()
        self.assertEqual(game.state, S_.COUNTDOWN)
        self.frames(game, int(60 * 3 * S.KICKOFF_STEP_TIME) + 5)
        self.assertEqual(game.state, S_.PLAYING)
        self.frames(game, 60)
        self.assertLess(game.match.time_left, game.match.match_length)
        # Pause freezes the clock.
        self.press(pygame.K_ESCAPE)
        self.frames(game, 2)
        self.assertEqual(game.state, S_.PAUSED)
        frozen = game.match.time_left
        self.frames(game, 30)
        self.assertEqual(game.match.time_left, frozen)
        self.press(pygame.K_ESCAPE)
        self.frames(game, 2)
        self.assertEqual(game.state, S_.PLAYING)
        # Force a goal.
        game.match.ball.pos.update(S.ARENA_RIGHT + 50, S.ARENA_FLOOR - 40)
        self.frames(game, 2)
        self.assertEqual(game.state, S_.GOAL)
        self.assertEqual(game.match.score[0], 1)
        self.frames(game, int(60 * S.GOAL_CELEBRATION_TIME) + 5)
        self.assertEqual(game.state, S_.COUNTDOWN)
        # Tied at full time -> overtime.
        self.frames(game, int(60 * 3 * S.KICKOFF_STEP_TIME) + 5)
        game.match.score = [1, 1]
        game.match.time_left = 0.01
        self.frames(game, 2)
        self.assertTrue(game.match.overtime)
        self.assertEqual(game.state, S_.COUNTDOWN)
        # Golden goal ends the match.
        self.frames(game, int(60 * (3 * S.KICKOFF_STEP_TIME + 1.5)))
        self.assertEqual(game.state, S_.PLAYING)
        game.match.ball.pos.update(S.ARENA_LEFT - 50, S.ARENA_FLOOR - 40)
        self.frames(game, int(60 * S.GOAL_CELEBRATION_TIME) + 10)
        self.assertEqual(game.state, S_.GAME_OVER)
        self.assertEqual(game.save.stats["losses"], 1)
        # Play again.
        game.restart_match()
        self.assertEqual(game.state, S_.COUNTDOWN)
        self.assertEqual(game.match.score, [0, 0])
        game.go_to_menu()
        self.assertEqual(game.state, S_.MAIN_MENU)
        game.save.save()  # (no pygame.quit here: other tests still need pygame)


if __name__ == "__main__":
    unittest.main()
