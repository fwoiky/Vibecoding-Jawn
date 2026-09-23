"""
Computer-controlled opponent.

The AI produces the same Controls as the keyboard, so it obeys exactly the
same physics as the player. It works like this:

1. Perception: it sees the ball with a small delay (reaction time).
2. Decision (a few times per second): pick a state
       DEFEND      - get back between the ball and its own goal
       CHASE_BALL  - drive to a spot behind the ball (predicting where it goes)
       ATTACK      - close to the ball on the correct side: hit it goal-wards
       RECOVER     - regain position when out of the play / low on boost
3. Steering (every frame): turn the current target into key presses, plus
   jumps, flips and aerials when the ball is high.

Difficulty only changes reaction time, accuracy and how often it uses its
skills - never the physics.
"""
import math
import random
from collections import deque
from enum import Enum

from pygame.math import Vector2

import settings as S
from controls import Controls
from physics import clamp, predict_ball_path, sign, wrap_angle


class AIState(Enum):
    DEFEND = "DEFEND"
    CHASE_BALL = "CHASE"
    ATTACK = "ATTACK"
    RECOVER = "RECOVER"


FIELD_WIDTH = S.ARENA_RIGHT - S.ARENA_LEFT


class AIController:
    def __init__(self, car, ball, difficulty=S.DEFAULT_DIFFICULTY):
        self.car = car
        self.ball = ball
        self.set_difficulty(difficulty)
        # Team 0 defends the left goal and attacks to the right.
        self.dir = 1 if car.team == 0 else -1
        self.own_goal_x = S.ARENA_LEFT if self.dir == 1 else S.ARENA_RIGHT
        self.reset()

    def set_difficulty(self, difficulty):
        self.difficulty = difficulty if difficulty in S.AI_DIFFICULTY else S.DEFAULT_DIFFICULTY
        self.p = S.AI_DIFFICULTY[self.difficulty]

    def reset(self):
        self.state = AIState.CHASE_BALL
        self.clock = 0.0
        self.history = deque()
        self.decision_timer = 0.0
        self.target_x = self.car.pos.x
        self.intercept = Vector2(self.ball.pos)
        self.aim_offset = 0.0
        self.want_boost = False
        self.jump_queue = []        # pending jump presses: [delay, move_x, hold]
        self.jump_hold_timer = 0.0
        self.aerial_timer = 0.0
        self.stuck_timer = 0.0
        self.unstick_timer = 0.0
        self.jump_cooldown = 0.0
        self.skill_timer = 0.0
        self.allow_jump = True
        self.allow_aerial = False
        self.hang_back = True

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------
    def _perceive(self, dt):
        self.clock += dt
        self.history.append((self.clock, Vector2(self.ball.pos), Vector2(self.ball.vel)))
        cutoff = self.clock - self.p["reaction"]
        while len(self.history) > 1 and self.history[1][0] <= cutoff:
            self.history.popleft()
        _, pos, vel = self.history[0]
        return pos, vel

    def _x_from_own_goal(self, x):
        """Distance from our own goal line towards the enemy goal."""
        return (x - self.own_goal_x) * self.dir

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------
    def _find_intercept(self, ball_pos, ball_vel):
        car = self.car
        reach_height = 95 + 150 * self.p["jump_skill"] + (140 if self.p["aerial_chance"] > 0.5 else 0)
        top_speed = S.CAR_MAX_SPEED * self.p["max_throttle"]
        if car.boost > 20 and self.p["boost_chance"] > 0.5:
            top_speed += 250
        points = predict_ball_path(ball_pos, ball_vel, self.p["prediction_time"])
        for t, point in points:
            distance = max(0.0, abs(point.x - car.pos.x) - 60)
            if distance / top_speed <= t and S.ARENA_FLOOR - point.y < reach_height:
                return point
        return points[-1][1]

    def _decide(self, ball_pos, ball_vel):
        car = self.car
        p = self.p
        self.intercept = self._find_intercept(ball_pos, ball_vel)
        self.aim_offset = random.uniform(-p["aim_error"], p["aim_error"])

        ball_x = self._x_from_own_goal(self.intercept.x)
        ball_now_x = self._x_from_own_goal(ball_pos.x)
        car_x = self._x_from_own_goal(car.pos.x)
        towards_us = -ball_vel.x * self.dir
        behind_ball = car_x < min(ball_x, ball_now_x) - 25
        threat = towards_us > 150 and (ball_now_x < FIELD_WIDTH * 0.65 or towards_us > 500)
        distance = car.pos.distance_to(ball_pos)
        deep_attack = ball_now_x > FIELD_WIDTH * 0.8 and towards_us < 0

        flying_away = ball_now_x > FIELD_WIDTH * 0.75 and towards_us < -350
        if not behind_ball:
            self.state = AIState.DEFEND
        elif flying_away and self.hang_back:
            # We just shot it deep: don't follow it into the corner.
            self.state = AIState.RECOVER
        elif distance < 240:
            self.state = AIState.ATTACK
        elif deep_attack and self.hang_back:
            # Ball is deep in the opponent's corner: wait for the rebound
            # instead of leaving our goal empty.
            self.state = AIState.RECOVER
        elif car.boost < 12 and ball_now_x > FIELD_WIDTH * 0.7 and not threat and self.hang_back:
            self.state = AIState.RECOVER
        else:
            self.state = AIState.CHASE_BALL

        # Decide boost use for this decision period. Racing back to stop a
        # shot always gets boost if there is any.
        if self.state == AIState.DEFEND and threat:
            self.want_boost = random.random() < p["boost_chance"] + 0.3
        else:
            wants_speed = self.state in (AIState.ATTACK, AIState.CHASE_BALL)
            self.want_boost = wants_speed and random.random() < p["boost_chance"]

        # Target position on the x axis for this state.
        if self.state == AIState.DEFEND:
            self.target_x = self.own_goal_x + self.dir * 60
        elif self.state == AIState.RECOVER:
            self.target_x = self.own_goal_x + self.dir * FIELD_WIDTH * 0.3
        elif self.state == AIState.CHASE_BALL:
            self.target_x = self.intercept.x - self.dir * (S.BALL_RADIUS + 30) + self.aim_offset
        else:  # ATTACK: drive straight through the ball towards the enemy goal
            self.target_x = self.intercept.x + self.dir * 40 + self.aim_offset * 0.5

        # Should we try a flip into the ball?
        if (self.state == AIState.ATTACK and car.on_floor and distance < 130
                and not self.jump_queue and self.jump_cooldown <= 0
                and S.ARENA_FLOOR - ball_pos.y < 80 and car.vel.x * self.dir > 200
                and (ball_pos.x - car.pos.x) * self.dir > 0 and random.random() < p["dodge_chance"]):
            self._queue_jump(0.0)
            self._queue_jump(0.09, self.dir)

    def _queue_jump(self, delay, move_x=0.0, hold=True):
        self.jump_queue.append([delay, move_x, hold])
        self.jump_cooldown = 0.6

    # ------------------------------------------------------------------
    # Per-frame control
    # ------------------------------------------------------------------
    def update(self, dt):
        car = self.car
        ball_pos, ball_vel = self._perceive(dt)
        self.jump_cooldown = max(0.0, self.jump_cooldown - dt)

        # Re-roll "will I try a jump / aerial?" every so often, not every frame.
        self.skill_timer -= dt
        if self.skill_timer <= 0:
            self.skill_timer = 1.2
            self.allow_jump = random.random() < self.p["jump_skill"]
            self.allow_aerial = random.random() < self.p["aerial_chance"]
            self.hang_back = random.random() < self.p["defend_bias"]

        self.decision_timer -= dt
        if self.decision_timer <= 0:
            self.decision_timer = self.p["decision_interval"] * random.uniform(0.8, 1.2)
            self._decide(ball_pos, ball_vel)

        c = Controls()
        if car.grounded:
            self._drive(c, ball_pos, ball_vel, dt)
        else:
            self._air(c, ball_pos, dt)
        self._process_jump_queue(c, dt)
        return c

    def _drive(self, c, ball_pos, ball_vel, dt):
        car = self.car
        self.aerial_timer = 0.0

        # On a wall or the ceiling: let go and drop back down.
        if not car.on_floor:
            return

        dx = self.target_x - car.pos.x
        if abs(dx) > 18:
            c.move_x = sign(dx) * self.p["max_throttle"]
            # Ease off when closing in on the approach point so we arrive
            # under control instead of overshooting it.
            if self.state == AIState.CHASE_BALL:
                desired_speed = max(160.0, abs(dx) * 2.6)
                if car.vel.x * sign(dx) > desired_speed:
                    c.move_x = 0.0
        elif self.state in (AIState.DEFEND, AIState.RECOVER):
            # Arrived: face the ball and wait.
            to_ball = ball_pos.x - car.pos.x
            if sign(to_ball) != car.facing and abs(car.vel.x) < 50:
                c.move_x = sign(to_ball) * 0.5
        # Brake instead of overshooting a defensive position.
        if self.state in (AIState.DEFEND, AIState.RECOVER):
            speed_to_target = car.vel.x * sign(dx)
            if speed_to_target > 0 and abs(dx) < speed_to_target ** 2 / (2 * S.CAR_BRAKE) + 30:
                c.move_x = 0.0
                c.down = abs(car.vel.x) > 60

        heading_right_way = c.move_x != 0 and sign(c.move_x) == car.facing
        far = abs(dx) > 160
        c.boost = self.want_boost and heading_right_way and far and car.boost > 5

        self._maybe_jump_for_ball(ball_pos, ball_vel)
        self._avoid_own_goal(c, ball_pos)
        self._check_stuck(c, dt)

    def _maybe_jump_for_ball(self, ball_pos, ball_vel):
        car = self.car
        if self.jump_cooldown > 0 or self.jump_queue or not self.allow_jump:
            return
        # Never jump at the ball from the wrong side: that hit would go
        # towards our own goal.
        if self.state not in (AIState.ATTACK, AIState.CHASE_BALL):
            return
        dx = ball_pos.x - car.pos.x
        if dx * car.facing < -10:
            return  # ball is behind us
        closing = (car.vel.x - ball_vel.x) * sign(dx)
        if closing < 60 and abs(dx) > 50:
            return
        time_to_reach = max(0.0, abs(dx) - 45) / max(closing, 60)
        if time_to_reach > 0.9:
            return
        # Where will the ball be (vertically) when we get there?
        t = time_to_reach
        future_y = min(ball_pos.y + ball_vel.y * t + 0.5 * S.BALL_GRAVITY * t * t,
                       S.ARENA_FLOOR - S.BALL_RADIUS)
        height = car.pos.y - future_y      # how far the ball is above the car
        if height < 70:
            return                          # low enough to hit without jumping
        # Aerial: jump, then fly to the ball with boost.
        if height > 250:
            if self.allow_aerial and car.boost > 30 and abs(dx) < 380:
                self._queue_jump(0.0)
                self._queue_jump(0.16)
                self.aerial_timer = 2.0
            return
        rise_time = 0.1 + height / 700.0
        if time_to_reach <= rise_time:
            self._queue_jump(0.0)
            if height > 185:
                self._queue_jump(0.2)

    def _avoid_own_goal(self, c, ball_pos):
        """
        On the wrong side of the ball we must not push it towards our own goal:
        hop over it (double jump if it bounces high) or brake and wait.
        """
        car = self.car
        if self.state != AIState.DEFEND:
            return
        rel = ball_pos - car.pos
        ball_height = S.ARENA_FLOOR - ball_pos.y
        heading_into_ball = rel.x * car.vel.x > 0 and abs(car.vel.x) > 80
        if not heading_into_ball or ball_height > 170:
            return   # not in our way (or high enough to drive under)
        edge_gap = abs(rel.x) - (S.CAR_HALF_LENGTH + S.BALL_RADIUS)
        time_to_contact = edge_gap / abs(car.vel.x)
        if time_to_contact > 0.3:
            return
        if self.jump_cooldown <= 0 and not self.jump_queue:
            self._queue_jump(0.0, hold=ball_height > 70)
            if ball_height > 110:
                self._queue_jump(0.14)
        elif self.jump_cooldown > 0 and car.on_floor:
            c.move_x = 0.0
            c.down = True
            c.boost = False

    def _check_stuck(self, c, dt):
        car = self.car
        if self.unstick_timer > 0:
            self.unstick_timer -= dt
            c.move_x = 0.0
            c.down = True
            c.boost = False
            return
        if abs(c.move_x) > 0.3 and car.vel.length() < 40:
            self.stuck_timer += dt
            if self.stuck_timer > 1.0:
                self.stuck_timer = 0.0
                self.unstick_timer = 0.4
                if self.jump_cooldown <= 0:
                    self._queue_jump(0.0)
        else:
            self.stuck_timer = 0.0

    def _air(self, c, ball_pos, dt):
        car = self.car
        if car.dodge_time > 0:
            return
        c.jump_held = self.jump_hold_timer > 0
        self.jump_hold_timer = max(0.0, self.jump_hold_timer - dt)

        if self.aerial_timer > 0:
            self.aerial_timer -= dt
            to_ball = ball_pos + Vector2(-self.dir * 12, 0) - car.pos
            if to_ball.y > 40 or to_ball.length() < 30:
                self.aerial_timer = 0.0
            else:
                direction = to_ball.normalize()
                self._rotate_towards(c, direction)
                facing_error = math.acos(clamp(car.forward.dot(direction), -1.0, 1.0))
                c.boost = facing_error < 0.6 and car.boost > 0
                c.move_x = sign(to_ball.x) * 0.5
                return

        # Normal air control: drift towards the target and land on the wheels.
        dx = self.target_x - car.pos.x
        if abs(dx) > 40:
            c.move_x = sign(dx) * 0.6
        # Jumped over the ball on the way back to goal: get down quickly.
        if self.state == AIState.DEFEND and (ball_pos.x - car.pos.x) * car.vel.x < 0 and car.vel.y > -200:
            c.down = True
        if self.p["air_recovery"]:
            error = wrap_angle(-car.angle)   # angle 0 = wheels down
            if abs(error) > 0.15:
                c.rotate = clamp(error * 2.0, -1.0, 1.0)

    def _rotate_towards(self, c, direction):
        car = self.car
        desired = math.atan2(direction.y * car.facing, direction.x * car.facing)
        error = wrap_angle(desired - car.angle)
        c.rotate = clamp(error * 2.5, -1.0, 1.0)

    def _process_jump_queue(self, c, dt):
        if not self.jump_queue:
            return
        entry = self.jump_queue[0]
        entry[0] -= dt
        if entry[0] <= 0:
            self.jump_queue.pop(0)
            c.jump_pressed = True
            self.jump_hold_timer = S.JUMP_HOLD_TIME if entry[2] else 0.0
            c.jump_held = entry[2]
            # A direction turns an air jump into a flip, so always set it explicitly.
            c.move_x = entry[1]
