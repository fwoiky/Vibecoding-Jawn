"""
A single match: owns the arena, cars, ball, AI and particle effects, runs the
fixed-timestep physics and keeps score. Game-flow decisions (countdown,
pause, overtime, game over) are made by game.py.
"""
import math
import random

import pygame
from pygame.math import Vector2

import settings as S
from ball import Ball
from car import Car
from controls import Controls
from opponent import AIController
from particles import ParticleSystem
from physics import capsule_vs_capsule, clamp

PLAYER, CPU = 0, 1
TEAM_COLORS = (S.PLAYER_COLOR, S.CPU_COLOR)


class Match:
    def __init__(self, arena, audio=None, difficulty=S.DEFAULT_DIFFICULTY,
                 match_minutes=S.DEFAULT_MATCH_MINUTES, player_controller=None, demo=False):
        self.arena = arena
        self.audio = audio
        self.demo = demo
        self.difficulty = difficulty
        self.player_controller = player_controller
        self.ball = Ball()
        self.cars = [
            Car(PLAYER, S.PLAYER_COLOR, S.PLAYER_ACCENT, S.PLAYER_DARK, "coupe"),
            Car(CPU, S.CPU_COLOR, S.CPU_ACCENT, S.CPU_DARK, "wedge"),
        ]
        # In demo mode (menu background) the computer drives both cars.
        self.ai = [
            AIController(self.cars[PLAYER], self.ball, difficulty) if demo or player_controller is None else None,
            AIController(self.cars[CPU], self.ball, difficulty),
        ]
        self.particles = ParticleSystem()
        self.score = [0, 0]
        self.match_length = match_minutes * 60.0
        self.time_left = self.match_length
        self.overtime = False
        self.overtime_time = 0.0
        self.shake_enabled = True
        self.shake = 0.0
        self.flash = 0.0
        self.time = 0.0
        self.elapsed_play = 0.0
        self.accumulator = 0.0
        self.last_goal_team = None
        self.last_touch = None
        self.world_surface = None
        self.flash_surface = None
        self.demo_timer = 0.0
        self.reset_kickoff()
        if demo:
            self.ball.frozen = False

    # ------------------------------------------------------------------
    # Match flow helpers (called by the Game)
    # ------------------------------------------------------------------
    def reset_kickoff(self):
        self.ball.reset()
        self.cars[PLAYER].reset(S.ARENA_CENTER_X - S.KICKOFF_CAR_OFFSET, 1)
        self.cars[CPU].reset(S.ARENA_CENTER_X + S.KICKOFF_CAR_OFFSET, -1)
        for ai in self.ai:
            if ai:
                ai.reset()
        if self.player_controller:
            self.player_controller.reset()
        self.accumulator = 0.0
        self.last_touch = None
        self.stop_sounds()

    def release_ball(self):
        self.ball.frozen = False

    def tick_clock(self, dt):
        self.elapsed_play += dt
        if self.overtime:
            self.overtime_time += dt
        else:
            self.time_left = max(0.0, self.time_left - dt)

    def start_overtime(self):
        self.overtime = True
        self.overtime_time = 0.0

    def set_difficulty(self, difficulty):
        self.difficulty = difficulty
        for ai in self.ai:
            if ai:
                ai.set_difficulty(difficulty)

    def set_particle_quality(self, level):
        level = max(0, min(len(S.PARTICLE_LIMITS) - 1, level))
        self.particles.max_particles = S.PARTICLE_LIMITS[level]
        self.particles.density = S.PARTICLE_DENSITY[level]

    def occluders(self):
        """Shapes of the moving objects, used by the GPU ambient occlusion pass."""
        shapes = []
        for car in self.cars:
            along = car.axis * S.CAR_HALF_LENGTH
            across = car.up * S.CAR_HALF_HEIGHT
            p = car.pos
            shapes.append(("poly", [tuple(p + along + across), tuple(p + along - across),
                                    tuple(p - along - across), tuple(p - along + across)]))
        if self.ball.visible:
            shapes.append(("circle", tuple(self.ball.pos), self.ball.radius))
        return shapes

    def stop_sounds(self):
        if self.audio:
            self.audio.stop_loops()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def _controls_for(self, index, dt):
        ai = self.ai[index]
        if ai is not None:
            return ai.update(dt)
        if self.player_controller is not None:
            return self.player_controller.poll()
        return Controls()

    def update(self, dt, cars_active=True, ball_active=True):
        """Advance the world by dt seconds. Returns a list of gameplay events."""
        self.time += dt
        events = []
        if cars_active:
            controls = [self._controls_for(i, dt) for i in range(2)]
        else:
            if self.player_controller:
                self.player_controller.poll()  # discard input during countdowns
            controls = [Controls(), Controls()]
            for car in self.cars:
                car.boosting = False

        self.accumulator += dt
        steps = 0
        while self.accumulator >= S.PHYSICS_DT and steps < S.MAX_PHYSICS_STEPS_PER_FRAME:
            self._step(S.PHYSICS_DT, controls, cars_active, ball_active, events)
            for c in controls:
                c.jump_pressed = False   # a jump press is used only once
            self.accumulator -= S.PHYSICS_DT
            steps += 1
        if steps >= S.MAX_PHYSICS_STEPS_PER_FRAME:
            self.accumulator = 0.0

        self._process_car_events()
        self._update_effects(dt, cars_active)
        if self.demo:
            self._update_demo(dt)
        return events

    def _step(self, dt, controls, cars_active, ball_active, events):
        ball = self.ball
        if cars_active:
            for car, c in zip(self.cars, controls):
                car.update(c, dt)
            for car in self.cars:
                car.collide_arena(self.arena)
            self._collide_cars()

        if ball_active and ball.visible and not ball.frozen:
            ball.update(dt)
            if cars_active:
                for car in self.cars:
                    impact = ball.collide_car(car)
                    if impact > 20:
                        self.last_touch = car.team
                        events.append(("hit", car.team, impact))
                        self._hit_effects(car, impact)
            impact = ball.collide_arena(self.arena)
            if impact > 160:
                self._bounce_effects(impact)
            if cars_active:
                for car in self.cars:
                    ball.separate_from_car(car)
            team = self.arena.goal_scored(ball.pos, ball.radius)
            if team is not None:
                self._score_goal(team)
                events.append(("goal", team))

        if cars_active:
            for car in self.cars:
                car.update_ground_state(self.arena)

    def _collide_cars(self):
        a, b = self.cars
        a1, a2 = a.wheel_positions()
        b1, b2 = b.wheel_positions()
        contact = capsule_vs_capsule(a1, a2, b1, b2, S.CAR_RADIUS, S.CAR_RADIUS)
        if contact is None:
            return
        n = contact.normal
        a.pos -= n * (contact.depth * 0.5)
        b.pos += n * (contact.depth * 0.5)
        rel = (b.vel - a.vel).dot(n)
        if rel < 0:
            j = -(1.0 + S.CAR_CAR_RESTITUTION) * rel * 0.5
            a.vel -= n * j
            b.vel += n * j
            if -rel > S.CAR_CAR_BUMP:
                point = (a.pos + b.pos) * 0.5
                self.particles.burst(point, 10, speed=(80, 260), life=(0.2, 0.4), size=(2, 4),
                                     colors=((255, 255, 255), S.GOLD))
                self._play("bounce", clamp(-rel / 600, 0.3, 1.0))
                self.add_shake(-rel / 250)

    # ------------------------------------------------------------------
    # Effects
    # ------------------------------------------------------------------
    def _play(self, name, volume=1.0):
        if self.audio and not self.demo:
            self.audio.play(name, volume)

    def add_shake(self, amount):
        self.shake = min(S.SCREEN_SHAKE_MAX, max(self.shake, amount))

    def _hit_effects(self, car, impact):
        pos = self.ball.pos
        strength = clamp(impact / S.STRONG_HIT_SPEED, 0.0, 1.5)
        color = TEAM_COLORS[car.team]
        count = int(6 + 18 * min(1.0, strength))
        self.particles.burst(pos, count, speed=(120, 320 + 400 * strength), life=(0.2, 0.5),
                             size=(2, 4), colors=((255, 255, 255), color, S.GOLD))
        if impact > S.STRONG_HIT_SPEED:
            self.particles.ring(pos, 90, 0.35, (255, 255, 255), 5)
            self._play("hit_hard", min(1.0, strength))
            self.add_shake(4 + 3 * min(1.0, strength))
        else:
            self._play("hit_soft", 0.4 + 0.6 * strength)

    def _bounce_effects(self, impact):
        self._play("bounce", clamp(impact / 900, 0.2, 1.0))
        if impact > 700:
            self.particles.ring(self.ball.pos, 60, 0.3, S.BALL_GLOW, 3)
            self.add_shake(2.5)

    def _score_goal(self, team):
        ball = self.ball
        pos = Vector2(ball.pos)
        ball.visible = False
        ball.frozen = True
        self.score[team] += 1
        self.last_goal_team = team
        goal_side = 1 if team == PLAYER else 0
        color = TEAM_COLORS[team]
        self.arena.goal_flash[goal_side] = 1.0
        self.particles.burst(pos, 70, speed=(200, 950), life=(0.5, 1.3), size=(3, 7),
                             colors=(color, (255, 255, 255), S.GOLD), gravity=500, drag=1.2)
        self.particles.burst(pos, 40, speed=(150, 600), life=(0.8, 1.6), size=(2, 4),
                             colors=(color, (255, 255, 255), S.GOLD, S.ACCENT), gravity=600, drag=0.8,
                             glow=False, shape="confetti")
        self.particles.ring(pos, 220, 0.6, color, 8)
        self.particles.ring(pos, 140, 0.45, (255, 255, 255), 5)
        self.flash = 0.6
        self.add_shake(S.SCREEN_SHAKE_MAX)
        self._play("goal")
        self.stop_sounds()

    def celebrate(self, team):
        """Confetti rain for the game-over screen."""
        color = TEAM_COLORS[team]
        for _ in range(3):
            x = random.uniform(S.ARENA_LEFT, S.ARENA_RIGHT)
            self.particles.emit(x, S.ARENA_TOP + 5, random.uniform(-60, 60), random.uniform(50, 200),
                                random.uniform(1.5, 3.0), random.uniform(2, 4),
                                random.choice((color, (255, 255, 255), S.GOLD, S.ACCENT)),
                                gravity=120, drag=0.5, shape="confetti")

    def _process_car_events(self):
        for car in self.cars:
            is_player = car.team == PLAYER and not self.demo
            volume = 1.0 if is_player else 0.55
            for name, pos, strength in car.events:
                if name in ("jump", "double_jump"):
                    self._play(name, volume)
                    self._dust(pos + Vector2(0, S.CAR_RADIUS), 8)
                elif name == "dodge":
                    self._play("dodge", volume)
                    self.particles.ring(pos, 50, 0.25, TEAM_COLORS[car.team], 3)
                elif name == "land" and strength > 280:
                    self._play("land", clamp(strength / 900, 0.3, 1.0) * volume)
                    self._dust(pos + Vector2(0, S.CAR_RADIUS), int(clamp(strength / 60, 4, 14)))
                elif name == "turn":
                    self._dust(pos + Vector2(0, S.CAR_RADIUS), 4)
                elif name == "flip_upright":
                    self._play("flip_upright", volume)
                    self._dust(pos + Vector2(0, S.CAR_RADIUS), 6)
            car.events.clear()

    def _dust(self, pos, count):
        if pos.y < S.ARENA_FLOOR - 30:
            return  # only kick up dust on the floor
        self.particles.burst((pos.x, S.ARENA_FLOOR - 2), count, speed=(40, 170), life=(0.3, 0.6),
                             size=(2, 5), colors=(S.DUST_COLOR, (230, 235, 250)), gravity=-40,
                             drag=3.0, glow=False, direction=(0, -1), spread=1.3)

    def _update_effects(self, dt, cars_active):
        self.arena.update(dt)
        for i, car in enumerate(self.cars):
            if car.boosting and cars_active:
                fwd = car.forward
                rear = car.pos - fwd * 40
                for _ in range(2 if random.random() < self.particles.density else 1):
                    spread = Vector2(random.uniform(-40, 40), random.uniform(-40, 40))
                    v = car.vel * 0.3 - fwd * random.uniform(180, 320) + spread
                    self.particles.emit(rear.x, rear.y, v.x, v.y, random.uniform(0.18, 0.35),
                                        random.uniform(3, 5), random.choice((S.BOOST_FLAME_OUTER, S.BOOST_COLOR)),
                                        drag=2.0, glow=True)
                if car.vel.length() > 850 and random.random() < 0.5:
                    p = car.pos + Vector2(random.uniform(-20, 20), random.uniform(-14, 14))
                    self.particles.emit(p.x, p.y, -car.vel.x * 0.1, -car.vel.y * 0.1, 0.2, 2,
                                        (200, 230, 255), drag=0.0, glow=True)
            elif car.on_floor and abs(car.vel.x) > 420 and random.random() < 0.35:
                self._dust(car.pos + Vector2(-car.forward.x * 20, S.CAR_RADIUS), 1)
            if self.audio and not self.demo:
                self.audio.set_boost(i, car.boosting and cars_active, 1.0 if i == PLAYER else 0.45)

        ball = self.ball
        if ball.visible and not ball.frozen and ball.vel.length() > 900 and random.random() < 0.7:
            self.particles.emit(ball.pos.x, ball.pos.y, 0, 0, 0.25, ball.radius * 0.35, S.BALL_GLOW, glow=True)

        self.particles.update(dt)
        self.shake = max(0.0, self.shake - S.SCREEN_SHAKE_DECAY * dt)
        self.flash = max(0.0, self.flash - dt * 1.5)

    def _update_demo(self, dt):
        """Menu background: restart automatically after goals."""
        if not self.ball.visible:
            self.demo_timer += dt
            if self.demo_timer > 2.0:
                self.demo_timer = 0.0
                self.reset_kickoff()
                self.ball.frozen = False
                if max(self.score) >= 9:
                    self.score = [0, 0]

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw(self, surface, show_player_marker=True):
        target = surface
        offset = (0, 0)
        if self.shake_enabled and self.shake > 0.3:
            if self.world_surface is None:
                self.world_surface = pygame.Surface(surface.get_size()).convert()
            target = self.world_surface
            offset = (random.uniform(-self.shake, self.shake), random.uniform(-self.shake, self.shake))

        self.arena.draw_background(target)
        for car in self.cars:
            car.draw_shadow(target)
        self.ball.draw_shadow(target)
        self.particles.draw(target)
        self.ball.draw(target)
        for car in self.cars:
            car.draw(target, self.time)
        self.arena.draw_foreground(target, self.time)
        if show_player_marker and not self.demo:
            self._draw_player_marker(target)

        if target is not surface:
            surface.fill(S.BG_TOP)
            surface.blit(target, offset)
        if self.flash > 0:
            if self.flash_surface is None:
                self.flash_surface = pygame.Surface(surface.get_size())
                self.flash_surface.fill((255, 255, 255))
            self.flash_surface.set_alpha(int(160 * self.flash))
            surface.blit(self.flash_surface, (0, 0))

    def _draw_player_marker(self, surface):
        car = self.cars[PLAYER]
        bob = math.sin(self.time * 5) * 3
        x, y = car.pos.x, car.pos.y - 34 + bob
        if car.pos.y < S.ARENA_TOP + 80:
            y = car.pos.y + 34 - bob
            points = [(x - 7, y), (x + 7, y), (x, y - 8)]
        else:
            points = [(x - 7, y), (x + 7, y), (x, y + 8)]
        pygame.draw.polygon(surface, S.PLAYER_COLOR, points)
        pygame.draw.polygon(surface, (255, 255, 255), points, 1)
