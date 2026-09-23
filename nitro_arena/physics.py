"""
Small, reusable physics helpers.

The game uses simple hand-written arcade physics:
    * objects have a position and a velocity (pygame Vector2)
    * collisions are detected with circles, capsules and oriented boxes
    * overlaps are fixed by pushing objects apart, then an *impulse* (an
      instant change in velocity) is applied along the collision normal.
"""
import math

from pygame.math import Vector2

import settings as S


def clamp(value, low, high):
    return max(low, min(high, value))


def sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def approach(value, target, max_delta):
    """Move value towards target by at most max_delta (never overshoots)."""
    if value < target:
        return min(value + max_delta, target)
    return max(value - max_delta, target)


def wrap_angle(angle):
    """Wrap an angle to the range [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    t = clamp(t, 0.0, 1.0)
    return (int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t))


def limit_speed(velocity, max_speed):
    speed_sq = velocity.length_squared()
    if speed_sq > max_speed * max_speed:
        velocity.scale_to_length(max_speed)


class Contact:
    """Result of a collision test. `normal` points out of the obstacle."""
    __slots__ = ("normal", "depth", "point")

    def __init__(self, normal, depth, point):
        self.normal = normal
        self.depth = depth
        self.point = point


def closest_point_on_segment(p, a, b):
    ab = b - a
    length_sq = ab.length_squared()
    if length_sq == 0:
        return Vector2(a)
    t = clamp((p - a).dot(ab) / length_sq, 0.0, 1.0)
    return a + ab * t


def circle_vs_obb(center, radius, box_center, axis_x, axis_y, half_w, half_h):
    """
    Circle against an oriented box. axis_x / axis_y are the box's unit axes.
    Returns a Contact whose normal points from the box towards the circle.
    """
    d = center - box_center
    lx = d.dot(axis_x)
    ly = d.dot(axis_y)
    cx = clamp(lx, -half_w, half_w)
    cy = clamp(ly, -half_h, half_h)

    if cx != lx or cy != ly:
        # Circle centre is outside the box: use the closest point on the box.
        dx = lx - cx
        dy = ly - cy
        dist_sq = dx * dx + dy * dy
        if dist_sq >= radius * radius:
            return None
        dist = math.sqrt(dist_sq)
        nx, ny = dx / dist, dy / dist
        depth = radius - dist
    else:
        # Circle centre is inside the box: push out along the shallowest axis.
        pen_x = half_w - abs(lx)
        pen_y = half_h - abs(ly)
        if pen_x < pen_y:
            nx, ny = (1.0 if lx >= 0 else -1.0), 0.0
            cx = half_w * nx
            depth = pen_x + radius
        else:
            nx, ny = 0.0, (1.0 if ly >= 0 else -1.0)
            cy = half_h * ny
            depth = pen_y + radius

    normal = axis_x * nx + axis_y * ny
    point = box_center + axis_x * cx + axis_y * cy
    return Contact(normal, depth, point)


def capsule_vs_capsule(a1, a2, b1, b2, radius_a, radius_b):
    """
    Distance test between two capsules (segments with a radius).
    In 2D the closest points of two non-crossing segments always involve an
    endpoint, so checking the four endpoint/segment pairs is enough.
    Returns a Contact with a normal pointing from capsule A to capsule B.
    """
    best = None
    best_dist_sq = float("inf")
    for p, s1, s2, flip in ((a1, b1, b2, True), (a2, b1, b2, True),
                            (b1, a1, a2, False), (b2, a1, a2, False)):
        q = closest_point_on_segment(p, s1, s2)
        dist_sq = (p - q).length_squared()
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            # Vector from A's point to B's point.
            best = (q - p) if flip else (p - q)
    total = radius_a + radius_b
    if best_dist_sq >= total * total:
        return None
    dist = math.sqrt(best_dist_sq)
    if dist < 1e-4:
        # Segments cross: separate along the line between the centres.
        ca = (a1 + a2) * 0.5
        cb = (b1 + b2) * 0.5
        normal = cb - ca
        if normal.length_squared() < 1e-6:
            normal = Vector2(1, 0)
        normal = normal.normalize()
    else:
        normal = best / dist
    return Contact(normal, total - dist, None)


def predict_ball_path(pos, vel, duration, step=1.0 / 30.0):
    """
    Cheap ball trajectory prediction used by the AI. It ignores the rounded
    corners and cars but handles gravity, drag and bounces off the main walls.
    Returns a list of (time, Vector2 position).
    """
    p = Vector2(pos)
    v = Vector2(vel)
    r = S.BALL_RADIUS
    points = [(0.0, Vector2(p))]
    t = 0.0
    drag = 1.0 - S.BALL_DRAG * step
    while t < duration:
        t += step
        v.y += S.BALL_GRAVITY * step
        v *= drag
        p += v * step
        if p.y + r > S.ARENA_FLOOR:
            p.y = S.ARENA_FLOOR - r
            if v.y > S.BALL_MIN_BOUNCE_SPEED:
                v.y = -v.y * S.BALL_BOUNCE
            else:
                v.y = 0
        elif p.y - r < S.ARENA_TOP:
            p.y = S.ARENA_TOP + r
            v.y = abs(v.y) * S.BALL_BOUNCE
        in_goal_mouth = p.y > S.CROSSBAR_Y + r * 0.5
        if not in_goal_mouth:
            if p.x - r < S.ARENA_LEFT:
                p.x = S.ARENA_LEFT + r
                v.x = abs(v.x) * S.BALL_BOUNCE
            elif p.x + r > S.ARENA_RIGHT:
                p.x = S.ARENA_RIGHT - r
                v.x = -abs(v.x) * S.BALL_BOUNCE
        points.append((t, Vector2(p)))
    return points
