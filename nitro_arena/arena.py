"""
The arena: its collision boundary, goal detection and (pre-rendered) visuals.

The playable area is described by a closed outline (polygon) made of straight
segments. The rounded ceiling corners are approximated by several short
segments, which lets cars drive smoothly from a wall onto the ceiling.

    ceiling  ___________________________
            /                           \\
           |                             |   <- side walls
     ______|  (goal roof / crossbar)     |______
    |goal                                  goal|
    |______________ floor _____________________|
"""
import math
import random

import pygame
from pygame.math import Vector2

import settings as S
from graphics import (GLOW, draw_neon_lines, regular_polygon, scale_color,
                      vertical_gradient)
from physics import Contact, lerp_color

LEFT_GOAL = 0    # goal defended by the player (team 0)
RIGHT_GOAL = 1   # goal defended by the CPU (team 1)


class Arena:
    def __init__(self):
        self.outline = self._build_outline()
        self.segments = self._build_segments(self.outline)
        # Edges that stick out into the play area (the top of each goal mouth).
        self.convex_corners = [Vector2(S.ARENA_LEFT, S.CROSSBAR_Y),
                               Vector2(S.ARENA_RIGHT, S.CROSSBAR_Y)]
        self.goal_rects = [
            pygame.Rect(S.ARENA_LEFT - S.GOAL_DEPTH, S.CROSSBAR_Y, S.GOAL_DEPTH, S.GOAL_HEIGHT),
            pygame.Rect(S.ARENA_RIGHT, S.CROSSBAR_Y, S.GOAL_DEPTH, S.GOAL_HEIGHT),
        ]
        self.goal_colors = [S.PLAYER_COLOR, S.CPU_COLOR]
        self.goal_flash = [0.0, 0.0]
        self.background = None
        self.goal_fronts = []
        self.goal_flash_surfaces = []

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @staticmethod
    def _arc(center, radius, start_deg, end_deg, segments):
        points = []
        for i in range(segments + 1):
            a = math.radians(start_deg + (end_deg - start_deg) * i / segments)
            points.append(Vector2(center.x + math.cos(a) * radius, center.y - math.sin(a) * radius))
        return points

    def _build_outline(self):
        L, R, T, B = S.ARENA_LEFT, S.ARENA_RIGHT, S.ARENA_TOP, S.ARENA_FLOOR
        CR, GD, CB = S.CORNER_RADIUS, S.GOAL_DEPTH, S.CROSSBAR_Y
        pts = [
            Vector2(L - GD, B),        # left goal, back bottom
            Vector2(R + GD, B),        # floor across to the right goal
            Vector2(R + GD, CB),       # right goal back wall
            Vector2(R, CB),            # right goal roof (crossbar)
        ]
        pts += self._arc(Vector2(R - CR, T + CR), CR, 0, 90, S.CORNER_SEGMENTS)       # right wall + corner
        pts += self._arc(Vector2(L + CR, T + CR), CR, 90, 180, S.CORNER_SEGMENTS)     # ceiling + corner
        pts += [
            Vector2(L, CB),            # left wall down to crossbar
            Vector2(L - GD, CB),       # left goal roof
        ]
        return pts

    @staticmethod
    def _build_segments(points):
        """
        Each segment stores precomputed values for fast collision tests.
        Points go around the arena so that the inward normal is (dy, -dx).
        """
        segments = []
        count = len(points)
        for i in range(count):
            a = points[i]
            b = points[(i + 1) % count]
            dx, dy = b.x - a.x, b.y - a.y
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            nx, ny = dy / length, -dx / length
            segments.append((a.x, a.y, b.x, b.y, dx, dy, length * length, nx, ny,
                             min(a.x, b.x), min(a.y, b.y), max(a.x, b.x), max(a.y, b.y)))
        return segments

    @staticmethod
    def _segment_contact(seg, px, py, r):
        """Circle (px, py, r) against one boundary segment -> (nx, ny, depth) or None."""
        ax, ay, bx, by, dx, dy, len_sq, nx, ny, minx, miny, maxx, maxy = seg
        if px < minx - r or px > maxx + r or py < miny - r or py > maxy + r:
            return None
        t = ((px - ax) * dx + (py - ay) * dy) / len_sq
        if t <= 0.0 or t >= 1.0:
            # Closest point is a segment end: treat it as a point obstacle.
            # This gives correct bounces off the goal crossbar edges.
            qx, qy = (ax, ay) if t <= 0.0 else (bx, by)
            ex, ey = px - qx, py - qy
            dist_sq = ex * ex + ey * ey
            if dist_sq >= r * r:
                return None
            dist = math.sqrt(dist_sq)
            if dist < 1e-6:
                return nx, ny, r
            return ex / dist, ey / dist, r - dist
        side = (px - ax) * nx + (py - ay) * ny   # distance in front of the wall
        if side >= r:
            return None
        if side < -(2 * r + 20):
            return None   # far behind this wall -> not ours to resolve
        return nx, ny, r - side

    def deepest_contact(self, pos, radius):
        best = None
        px, py = pos.x, pos.y
        for seg in self.segments:
            hit = self._segment_contact(seg, px, py, radius)
            if hit and (best is None or hit[2] > best[2]):
                best = hit
        if best is None:
            return None
        normal = Vector2(best[0], best[1])
        return Contact(normal, best[2], pos - normal * radius)

    def collide_ball(self, pos, vel, radius):
        """
        Push a circle out of the arena walls and bounce it. `pos` and `vel` are
        modified in place. Returns (impact_speed, normal_tuple_or_None).
        """
        impact = 0.0
        impact_normal = None
        for seg in self.segments:
            hit = self._segment_contact(seg, pos.x, pos.y, radius)
            if hit is None:
                continue
            nx, ny, depth = hit
            pos.x += nx * depth
            pos.y += ny * depth
            vn = vel.x * nx + vel.y * ny
            if vn < 0:
                tx, ty = -ny, nx
                vt = (vel.x * tx + vel.y * ty) * (1.0 - S.BALL_SURFACE_FRICTION)
                new_vn = -vn * S.BALL_BOUNCE if -vn > S.BALL_MIN_BOUNCE_SPEED else 0.0
                vel.x = tx * vt + nx * new_vn
                vel.y = ty * vt + ny * new_vn
                if -vn > impact:
                    impact = -vn
                    impact_normal = (nx, ny)
        return impact, impact_normal

    def goal_scored(self, ball_pos, radius):
        """Returns the team that scored (0 = player, 1 = CPU) or None."""
        if ball_pos.y > S.CROSSBAR_Y:
            if ball_pos.x + radius < S.ARENA_LEFT:
                return 1      # ball fully inside the player's goal
            if ball_pos.x - radius > S.ARENA_RIGHT:
                return 0
        return None

    # ------------------------------------------------------------------
    # Visuals
    # ------------------------------------------------------------------
    def build_graphics(self):
        """Pre-render the static arena once (requires an initialised display)."""
        size = (S.SCREEN_WIDTH, S.SCREEN_HEIGHT)
        bg = vertical_gradient(size, S.BG_TOP, S.BG_BOTTOM)
        self._draw_stadium_backdrop(bg)
        self._draw_interior(bg)
        self._draw_goal_interiors(bg)
        self._draw_floor(bg)
        self._draw_outline(bg)
        self.background = bg.convert()
        self._build_goal_fronts()

    def _draw_stadium_backdrop(self, surface):
        rng = random.Random(7)
        # Distant stadium lights / crowd sparkle outside the arena.
        for _ in range(140):
            x = rng.randint(0, S.SCREEN_WIDTH)
            y = rng.randint(0, S.SCREEN_HEIGHT)
            c = rng.choice([(60, 80, 140), (90, 60, 140), (50, 110, 150), (120, 90, 60)])
            pygame.draw.circle(surface, c, (x, y), rng.choice([1, 1, 2]))
        # Structural pillars on both sides.
        for x in (S.ARENA_LEFT - S.GOAL_DEPTH - 14, S.ARENA_RIGHT + S.GOAL_DEPTH + 4):
            pygame.draw.rect(surface, S.ARENA_STRUCTURE, (x, 0, 10, S.SCREEN_HEIGHT))

    def _interior_polygon(self):
        return [(p.x, p.y) for p in self.outline]

    def _draw_interior(self, surface):
        size = surface.get_size()
        layer = vertical_gradient(size, S.ARENA_BACK_TOP, S.ARENA_BACK_BOTTOM, alpha=True)

        # Subtle grid on the back wall.
        grid_color = (34, 44, 92)
        for x in range(0, S.SCREEN_WIDTH, 48):
            pygame.draw.line(layer, grid_color, (x, 0), (x, S.SCREEN_HEIGHT), 1)
        for y in range(S.ARENA_TOP, S.ARENA_FLOOR, 48):
            pygame.draw.line(layer, grid_color, (0, y), (S.SCREEN_WIDTH, y), 1)

        # Crowd stands: rows of little heads behind the glass near the top.
        rng = random.Random(3)
        for row in range(4):
            y = S.ARENA_TOP + 26 + row * 16
            for x in range(S.ARENA_LEFT + 20, S.ARENA_RIGHT - 20, 11):
                if rng.random() < 0.8:
                    shade = rng.choice([(40, 50, 100), (55, 45, 95), (35, 70, 110), (80, 60, 90)])
                    pygame.draw.circle(layer, shade, (x + rng.randint(-2, 2), y + rng.randint(-2, 2)), 4)
        pygame.draw.line(layer, (70, 110, 180), (0, S.ARENA_TOP + 92), (S.SCREEN_WIDTH, S.ARENA_TOP + 92), 2)

        # Centre emblem (original design): hexagon ring with a chevron.
        cx, cy = S.ARENA_CENTER_X, (S.ARENA_TOP + S.ARENA_FLOOR) / 2 + 40
        emblem = (40, 58, 120)
        pygame.draw.polygon(layer, emblem, regular_polygon((cx, cy), 120, 6, math.pi / 6), 6)
        pygame.draw.polygon(layer, emblem, regular_polygon((cx, cy), 96, 6, math.pi / 6), 2)
        chevron = [(cx - 50, cy - 40), (cx + 10, cy), (cx - 50, cy + 40), (cx - 30, cy),
                   ]
        pygame.draw.polygon(layer, emblem, chevron)
        chevron2 = [(cx - 5, cy - 40), (cx + 55, cy), (cx - 5, cy + 40), (cx + 15, cy)]
        pygame.draw.polygon(layer, emblem, chevron2)
        # Centre line.
        pygame.draw.line(layer, (48, 70, 140), (cx, S.ARENA_TOP + 100), (cx, S.ARENA_FLOOR), 3)

        # Mask the layer to the arena shape.
        mask = pygame.Surface(size, pygame.SRCALPHA)
        mask.fill((255, 255, 255, 0))
        pygame.draw.polygon(mask, (255, 255, 255, 255), self._interior_polygon())
        layer.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surface.blit(layer, (0, 0))

    def _draw_goal_interiors(self, surface):
        for rect, color in zip(self.goal_rects, self.goal_colors):
            dark = scale_color(color, 0.22)
            pygame.draw.rect(surface, dark, rect)
            net = scale_color(color, 0.5)
            for x in range(rect.left, rect.right + 1, 12):
                pygame.draw.line(surface, net, (x, rect.top), (x, rect.bottom), 1)
            for y in range(rect.top, rect.bottom + 1, 12):
                pygame.draw.line(surface, net, (rect.left, y), (rect.right, y), 1)
            GLOW.draw(surface, color, rect.center, 110, 0.45)

    def _draw_floor(self, surface):
        floor_rect = pygame.Rect(0, S.ARENA_FLOOR, S.SCREEN_WIDTH, S.SCREEN_HEIGHT - S.ARENA_FLOOR)
        floor = vertical_gradient(floor_rect.size, S.FLOOR_TOP, S.FLOOR_BOTTOM)
        # Stripes and centre marker on the floor.
        for x in range(0, S.SCREEN_WIDTH, 80):
            pygame.draw.line(floor, lerp_color(S.FLOOR_TOP, S.FLOOR_BOTTOM, 0.3),
                             (x, 6), (x - 30, floor_rect.height), 2)
        cx = int(S.ARENA_CENTER_X)
        pygame.draw.polygon(floor, (255, 255, 255), [(cx - 26, 3), (cx + 26, 3), (cx + 14, 14), (cx - 14, 14)])
        for x, color in ((S.ARENA_LEFT, S.PLAYER_COLOR), (S.ARENA_RIGHT, S.CPU_COLOR)):
            pygame.draw.rect(floor, color, (x - 3, 0, 6, 16))
        surface.blit(floor, floor_rect)
        draw_neon_lines(surface, S.FLOOR_LINE, [[(0, S.ARENA_FLOOR), (S.SCREEN_WIDTH, S.ARENA_FLOOR)]],
                        width=3, glow_width=10)

    def _draw_outline(self, surface):
        # Arena walls + ceiling (neutral colour) and goal frames (team colours).
        L, R, CB, B, GD = S.ARENA_LEFT, S.ARENA_RIGHT, S.CROSSBAR_Y, S.ARENA_FLOOR, S.GOAL_DEPTH
        # outline[3:-1] runs from the right crossbar, up and over, to the left crossbar.
        walls = [(p.x, p.y) for p in self.outline[3:-1]]
        draw_neon_lines(surface, S.ARENA_EDGE, [walls], width=3, glow_width=12)
        left_frame = [(L, CB), (L - GD, CB), (L - GD, B)]
        right_frame = [(R, CB), (R + GD, CB), (R + GD, B)]
        draw_neon_lines(surface, S.PLAYER_COLOR, [left_frame], width=4, glow_width=14)
        draw_neon_lines(surface, S.CPU_COLOR, [right_frame], width=4, glow_width=14)
        # Ceiling light fixtures.
        for x in range(S.ARENA_LEFT + 180, S.ARENA_RIGHT - 150, 150):
            pygame.draw.rect(surface, (220, 240, 255), (x, S.ARENA_TOP + 4, 40, 4), border_radius=2)
            GLOW.draw(surface, (120, 170, 255), (x + 20, S.ARENA_TOP + 10), 40, 0.5)

    def _build_goal_fronts(self):
        """Semi-transparent goal-mouth 'light curtains', drawn in front of the ball."""
        self.goal_fronts = []
        self.goal_flash_surfaces = []
        for side, (rect, color) in enumerate(zip(self.goal_rects, self.goal_colors)):
            curtain = pygame.Surface((14, rect.height), pygame.SRCALPHA)
            for x in range(14):
                edge = x if side == 0 else 13 - x
                alpha = int(90 * (edge / 13) ** 2)
                pygame.draw.line(curtain, (*color, alpha), (x, 0), (x, rect.height))
            pos = (S.ARENA_LEFT - 14, rect.top) if side == 0 else (S.ARENA_RIGHT, rect.top)
            self.goal_fronts.append((curtain, pos))
            flash = pygame.Surface(rect.size)
            flash.fill(lerp_color(color, (255, 255, 255), 0.4))
            self.goal_flash_surfaces.append(flash)

    def update(self, dt):
        for i in range(2):
            self.goal_flash[i] = max(0.0, self.goal_flash[i] - dt * 0.8)

    def draw_background(self, surface):
        if self.background is None:
            self.build_graphics()
        surface.blit(self.background, (0, 0))
        for i, rect in enumerate(self.goal_rects):
            if self.goal_flash[i] > 0:
                flash = self.goal_flash_surfaces[i]
                flash.set_alpha(int(200 * self.goal_flash[i]))
                surface.blit(flash, rect)

    def draw_foreground(self, surface, time):
        for i, (curtain, pos) in enumerate(self.goal_fronts):
            surface.blit(curtain, pos)
            # Pulsing post light at the top of each goal mouth.
            pulse = 0.6 + 0.4 * math.sin(time * 3.0 + i * math.pi)
            post = self.convex_corners[i]
            GLOW.draw(surface, self.goal_colors[i], post, 26, 0.6 * pulse + self.goal_flash[i] * 0.4)
            pygame.draw.circle(surface, (255, 255, 255), (int(post.x), int(post.y)), 4)
