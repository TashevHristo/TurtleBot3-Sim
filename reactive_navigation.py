"""Map-free reactive goal navigation for the TurtleBot3 demo.

The controller is a Tangent Bug variant.  Every control step it sees only:

* the robot pose (x, y, yaw) and the goal position, and
* the obstacle points measured *right now* by the range sensors
  (lidar + depth camera), in the robot frame.

It keeps no map, no planned path and no waypoints.  It has two modes:

Motion to goal
    Drive straight at the goal while that direction is free.  If an obstacle
    blocks it, find the two visible ends of that obstacle and head past the
    end that gives the shortest estimated route (distance to the end plus
    straight-line distance from the end to the goal).

Boundary following
    If that estimate gets worse or stops improving, the robot is in a dead end
    (for example a wall between it and the goal).  It then follows the
    obstacle boundary, on the side that leads toward the more promising end,
    until it can reach a free point that is closer to the goal than any
    obstacle point it saw while following.  Then it returns to motion to goal.
"""
import math
from dataclasses import dataclass

import numpy as np

MOTION_TO_GOAL = "motion to goal"
BOUNDARY_FOLLOWING = "boundary following"


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class NavigationCommand:
    linear: float
    angular: float
    mode: str
    heading: float        # chosen travel direction, robot frame (rad)
    clearance: float      # free travel distance along that direction (m)
    follow_side: int      # +1 obstacle kept on the left, -1 on the right, 0 none
    detail: str


class TangentBugNavigator:
    def __init__(
        self,
        goal,
        sensor_range=3.5,
        robot_radius=0.11,
        safety_margin=0.14,
        follow_distance=0.36,
        max_speed=0.22,
        max_turn=1.1,
        bin_degrees=2.0,
        approach_gain=0.5,
        min_approach_speed=0.025,
    ):
        self.goal = np.asarray(goal, dtype=float)
        self.sensor_range = float(sensor_range)
        self.robot_radius = float(robot_radius)
        self.full_inflate = float(robot_radius + safety_margin)
        self.inflate = self.full_inflate
        self.pass_offset = self.full_inflate + 0.10
        self.follow_distance = float(follow_distance)
        self.max_speed = float(max_speed)
        self.max_turn = float(max_turn)
        self.approach_gain = float(approach_gain)            # 1/s: speed = gain * goal distance
        self.min_approach_speed = float(min_approach_speed)  # m/s creep speed at the goal
        self.bin_angles = np.radians(np.arange(-180.0, 180.0, bin_degrees))
        self.bin_width = math.radians(bin_degrees)
        self.bin_units = np.stack([np.cos(self.bin_angles), np.sin(self.bin_angles)], axis=1)

        self.mode = MOTION_TO_GOAL
        self.time = 0.0
        self.mode_switches = 0
        # motion to goal
        self.pass_side = 0            # +1 pass obstacle ends counter-clockwise, -1 clockwise
        self.best_cost = math.inf
        self.best_cost_time = 0.0
        # boundary following
        self.follow_side = 0
        self.follow_anchor = None     # world point to drive up to before following
        self.follow_start = None
        self.follow_left_start = False
        self.followed_distance = math.inf
        self.failed_follows = []      # (world start point, side) that looped without exit

    # ------------------------------------------------------------- sensing
    def _bin_points(self, points):
        """Reduce robot-frame points to one nearest range per angular bin."""
        count = len(self.bin_angles)
        ranges = np.full(count, self.sensor_range)
        hits = np.zeros(count, dtype=bool)
        if len(points):
            distance = np.hypot(points[:, 0], points[:, 1])
            bearing = np.arctan2(points[:, 1], points[:, 0])
            index = np.round((bearing - self.bin_angles[0]) / self.bin_width).astype(int) % count
            keep = distance < self.sensor_range
            np.minimum.at(ranges, index[keep], distance[keep])
            hits[index[keep]] = True
            # The sensors are not at the robot centre, so a bin can stay empty
            # between two hit bins.  Treat such single holes as surface.
            hole = ~hits & np.roll(hits, 1) & np.roll(hits, -1)
            ranges[hole] = 0.5 * (np.roll(ranges, 1)[hole] + np.roll(ranges, -1)[hole])
            hits |= hole
        return ranges, hits

    def _clearance(self, points, units):
        """Free travel distance of the inflated robot along each unit vector.

        Returns (clearance per direction, index of the limiting point or -1).
        Space beyond sensor range is treated as free.
        """
        units = np.atleast_2d(units)
        if not len(points):
            return np.full(len(units), self.sensor_range), np.full(len(units), -1)
        along = points @ units.T
        lateral_sq = np.sum(points ** 2, axis=1)[:, None] - along ** 2
        radius_sq = self.inflate ** 2
        blocking = (along > 0.0) & (lateral_sq < radius_sq)
        contact = along - np.sqrt(np.maximum(radius_sq - lateral_sq, 0.0))
        contact = np.where(blocking, contact, np.inf)
        limiting = np.argmin(contact, axis=0)
        clearance = contact[limiting, np.arange(len(units))]
        free = ~np.isfinite(clearance)
        limiting[free] = -1
        clearance = np.clip(np.where(free, self.sensor_range, clearance), 0.0, self.sensor_range)
        return clearance, limiting

    def _clearance_along(self, points, heading):
        unit = np.array([[math.cos(heading), math.sin(heading)]])
        return float(self._clearance(points, unit)[0][0])

    def _bin_of(self, angle):
        return int(np.round((angle - self.bin_angles[0]) / self.bin_width)) % len(self.bin_angles)

    def _discontinuous(self, ranges, hits, a, b):
        if hits[a] != hits[b]:
            return True
        if not hits[a]:
            return False
        near = min(ranges[a], ranges[b])
        return abs(ranges[a] - ranges[b]) > 0.25 + 0.15 * near

    def _obstacle_ends(self, ranges, hits, start_bin):
        """Both visible ends of the continuous obstacle that contains start_bin.

        Returns a list of (bin index, pass sign); pass sign +1 means the free
        side lies counter-clockwise of that end.
        """
        count = len(ranges)
        ends = []
        for sign in (+1, -1):
            current = start_bin
            for _ in range(count):
                following = (current + sign) % count
                if following == start_bin:
                    return []        # enclosed: no visible end
                if self._discontinuous(ranges, hits, current, following):
                    ends.append((current, sign))
                    break
                current = following
        return ends

    # ------------------------------------------------------------- helpers
    def _to_world(self, position, yaw, local):
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.atleast_2d(local)
        world = np.stack([
            position[0] + c * local[:, 0] - s * local[:, 1],
            position[1] + s * local[:, 0] + c * local[:, 1],
        ], axis=1)
        return world

    def _end_candidates(self, position, yaw, ranges, hits, start_bin):
        candidates = []
        for end_bin, sign in self._obstacle_ends(ranges, hits, start_bin):
            distance = float(ranges[end_bin])
            offset = math.asin(min(1.0, self.pass_offset / max(distance, 1e-6)))
            heading = wrap_angle(float(self.bin_angles[end_bin]) + sign * offset)
            local = distance * np.array([math.cos(heading), math.sin(heading)])
            world = self._to_world(position, yaw, local)[0]
            cost = distance + float(np.linalg.norm(world - self.goal))
            candidates.append({"heading": heading, "sign": sign, "cost": cost,
                               "distance": distance})
        return candidates

    def _open_heading(self, points, heading, sign, needed):
        """Turn from heading toward the pass side until the inflated robot fits.

        The visible end of an obstacle can lie far beyond a nearer corner of
        the same obstacle, so the straight line to it may clip that corner.
        """
        steps = heading + sign * np.radians(np.arange(0.0, 92.0, 2.0))
        units = np.stack([np.cos(steps), np.sin(steps)], axis=1)
        clearance, _ = self._clearance(points, units)
        fits = np.nonzero(clearance >= needed)[0]
        index = int(fits[0]) if len(fits) else int(np.argmax(clearance))
        return wrap_angle(float(steps[index])), float(clearance[index])

    def _drive(self, heading, clearance, slow=False):
        speed = min(self.max_speed, 0.5 * clearance)
        if slow:
            speed = min(speed, 0.6 * self.max_speed)
        if clearance < 0.06:
            speed = 0.0
        if abs(heading) > math.radians(60.0):
            speed = 0.0
        elif abs(heading) > math.radians(25.0):
            speed *= 0.4
        angular = max(-self.max_turn, min(self.max_turn, 2.2 * heading))
        return speed, angular

    def _switch(self, mode):
        if mode != self.mode:
            self.mode_switches += 1
        self.mode = mode
        self.best_cost = math.inf
        self.best_cost_time = self.time

    def _reach_distance(self, position, yaw, clearances):
        """Smallest goal distance of any point the robot can reach in a straight line."""
        directions = self.bin_angles + yaw
        unit = np.stack([np.cos(directions), np.sin(directions)], axis=1)
        along = np.clip(unit @ (self.goal - position), 0.0, clearances)
        closest = position + unit * along[:, None]
        return float(np.min(np.linalg.norm(closest - self.goal, axis=1)))

    # ---------------------------------------------------------------- main
    def update(self, position, yaw, points, dt):
        """Return a NavigationCommand from the pose and robot-frame obstacle points."""
        self.time += dt
        position = np.asarray(position[:2], dtype=float)
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        ranges, hits = self._bin_points(points)
        # Already closer than the safety margin (e.g. after a tight corner):
        # shrink the margin to the space actually available so moving along
        # or away from the obstacle is still allowed.
        nearest = float(np.min(ranges))
        self.inflate = min(self.full_inflate, max(self.robot_radius + 0.005, nearest - 0.03))

        to_goal = self.goal - position
        goal_distance = float(np.hypot(*to_goal))
        goal_heading = wrap_angle(math.atan2(to_goal[1], to_goal[0]) - yaw)
        goal_unit = np.array([[math.cos(goal_heading), math.sin(goal_heading)]])
        goal_clear, goal_limiter = self._clearance(points, goal_unit)
        goal_clear = float(goal_clear[0])
        goal_is_free = goal_clear >= min(goal_distance, self.sensor_range - 0.05)

        context = {
            "position": position, "yaw": yaw, "points": points, "ranges": ranges,
            "hits": hits, "goal_distance": goal_distance, "goal_heading": goal_heading,
            "goal_clear": goal_clear, "goal_limiter": int(goal_limiter[0]),
            "goal_is_free": goal_is_free,
        }
        if self.mode == BOUNDARY_FOLLOWING:
            command = self._boundary_following(context)
            if command is not None:
                return command
        return self._motion_to_goal(context)

    def _blocking_obstacle(self, context):
        """Where the goal direction is blocked.

        Returns the scan bin of the point that limits the inflated robot's
        clearance toward the goal (used to find obstacle ends), and the world
        point to drive up to before following a boundary: the surface hit
        straight along the goal bearing when there is one, so that grazing a
        nearby corner does not make the robot follow that corner.
        """
        ranges, hits = context["ranges"], context["hits"]
        limiter = context["points"][context["goal_limiter"]]
        start_bin = self._bin_of(math.atan2(limiter[1], limiter[0]))
        goal_bin = self._bin_of(context["goal_heading"])
        local = limiter
        if hits[goal_bin] and ranges[goal_bin] < context["goal_distance"]:
            local = ranges[goal_bin] * self.bin_units[goal_bin]
        anchor = self._to_world(context["position"], context["yaw"], local)[0]
        return start_bin, anchor

    def _motion_to_goal(self, context):
        points = context["points"]
        if context["goal_is_free"]:
            self.pass_side = 0
            cost = context["goal_distance"]
            heading, clearance = context["goal_heading"], context["goal_clear"]
            detail = "goal in sight"
        else:
            start_bin, anchor = self._blocking_obstacle(context)
            candidates = self._end_candidates(context["position"], context["yaw"],
                                              context["ranges"], context["hits"], start_bin)
            if not candidates:
                return self._start_following(context, start_bin, anchor,
                                             "no visible obstacle end")
            lowest = min(candidates, key=lambda item: item["cost"])
            best = lowest
            same_side = [item for item in candidates if item["sign"] == self.pass_side]
            if same_side and same_side[0]["cost"] < lowest["cost"] + 0.4:
                best = same_side[0]       # avoid flip-flopping between two ends
            self.pass_side = best["sign"]
            cost = lowest["cost"]
            heading, clearance = self._open_heading(
                points, best["heading"], best["sign"], min(best["distance"], 0.8))
            detail = "heading past obstacle end ({})".format(
                "counter-clockwise" if best["sign"] > 0 else "clockwise")

            # Dead end: the route estimate got worse or stopped improving.
            stalled = self.time - self.best_cost_time > 6.0
            if cost > self.best_cost + 0.3 or stalled:
                return self._start_following(context, start_bin, anchor,
                                             "no progress" if stalled else "estimate rose")

        if cost < self.best_cost - 0.03:
            self.best_cost = cost
            self.best_cost_time = self.time
        linear, angular = self._drive(heading, clearance)
        if context["goal_is_free"]:
            # Slow down on the final approach so the robot can stop on the goal
            # centre instead of driving through it at full speed.
            linear = min(linear, max(self.min_approach_speed,
                                     self.approach_gain * context["goal_distance"]))
            if context["goal_distance"] < 0.5:
                detail = "final approach"
        return NavigationCommand(linear, angular, MOTION_TO_GOAL, heading, clearance, 0, detail)

    def _start_following(self, context, start_bin, anchor, reason):
        position = context["position"]
        # Follow toward the most promising end that the robot is not already at.
        candidates = [item for item in self._end_candidates(
            position, context["yaw"], context["ranges"], context["hits"], start_bin)
            if item["distance"] > 0.6]
        if candidates:
            sign = min(candidates, key=lambda item: item["cost"])["sign"]
        else:
            sign = self.pass_side or 1
        # Passing an end clockwise keeps the obstacle on the robot's left.
        side = -sign
        # A follow started here before went all the way round: go the other way.
        for start, failed_side in self.failed_follows:
            if failed_side == side and np.hypot(*(start - position)) < 1.0:
                side = -side
                break
        self.follow_side = side
        self.follow_anchor = anchor
        self.follow_start = position.copy()
        self.follow_left_start = False
        self.followed_distance = math.inf
        self._switch(BOUNDARY_FOLLOWING)
        command = self._boundary_following(context)
        if command is None:
            return NavigationCommand(0.0, 0.0, self.mode, 0.0, 0.0, 0, reason)
        command.detail = f"{reason}; {command.detail}"
        return command

    def _leave_boundary(self):
        self.follow_side = 0
        self.follow_anchor = None
        self._switch(MOTION_TO_GOAL)
        return None

    def _boundary_following(self, context):
        position, yaw, points = context["position"], context["yaw"], context["points"]
        if context["goal_is_free"]:
            return self._leave_boundary()

        # First drive up to the obstacle that blocked the goal.
        if self.follow_anchor is not None:
            local = self.follow_anchor - position
            heading = wrap_angle(math.atan2(local[1], local[0]) - yaw)
            clearance = self._clearance_along(points, heading)
            if np.hypot(*local) > self.follow_distance + 0.12 and clearance > 0.12:
                linear, angular = self._drive(heading, clearance)
                return NavigationCommand(linear, angular, BOUNDARY_FOLLOWING, heading,
                                         clearance, self.follow_side, "approaching obstacle")
            self.follow_anchor = None
            self.follow_start = position.copy()
            self.follow_left_start = False

        # Closest approach to the goal of any obstacle point seen while following.
        if len(points):
            world = self._to_world(position, yaw, points)
            self.followed_distance = min(
                self.followed_distance, float(np.min(np.linalg.norm(world - self.goal, axis=1))))

        # Leave once a reachable free point is closer to the goal than that.
        clearances, _ = self._clearance(points, self.bin_units)
        if self._reach_distance(position, yaw, clearances) < self.followed_distance - 0.05:
            return self._leave_boundary()

        # Went all the way round without finding a way out: remember it and
        # trace the boundary the other way.
        away = float(np.hypot(*(position - self.follow_start)))
        if away > 1.0:
            self.follow_left_start = True
        elif self.follow_left_start and away < 0.35:
            self.failed_follows.append((self.follow_start.copy(), self.follow_side))
            self.follow_side = -self.follow_side
            self.follow_left_start = False
            self.follow_start = position.copy()
            self.followed_distance = math.inf
        side = self.follow_side

        if len(points):
            distance = np.hypot(points[:, 0], points[:, 1])
            bearing = np.arctan2(points[:, 1], points[:, 0])
            # The followed side, plus a little of the front.  A wall straight
            # ahead is handled as an inside corner instead.
            on_side = (bearing * side >= -math.radians(15.0)) & (distance < 1.2)
        else:
            on_side = np.zeros(0, dtype=bool)
        if not np.any(on_side):
            # Lost the wall: arc toward the followed side to find it again.
            heading = side * math.radians(50.0)
            clearance = self._clearance_along(points, heading)
            linear, angular = self._drive(heading, clearance, slow=True)
            return NavigationCommand(linear, angular, BOUNDARY_FOLLOWING, heading,
                                     clearance, side, "searching for wall")

        nearest = np.argmin(np.where(on_side, distance, np.inf))
        wall_bearing, wall_distance = float(bearing[nearest]), float(distance[nearest])
        # In a narrow passage keep to the middle instead of the usual distance.
        target = self.follow_distance
        opposite = ((bearing * side <= -math.radians(20.0)) & (bearing * side >= -math.radians(160.0))
                    & (distance < 2.0 * self.follow_distance + 0.3))
        if np.any(opposite):
            target = min(target, 0.5 * (wall_distance + float(np.min(distance[opposite]))))
        correction = max(-1.2, min(1.2, 1.6 * (wall_distance - target)))
        heading = wrap_angle(wall_bearing - side * math.pi / 2.0 + side * correction)
        clearance = self._clearance_along(points, heading)
        if clearance < 0.06:
            # Wall ahead (inside corner): turn on the spot away from the wall.
            return NavigationCommand(0.0, -side * 0.7 * self.max_turn, BOUNDARY_FOLLOWING,
                                     heading, clearance, side, "turning at corner")
        linear, angular = self._drive(heading, clearance, slow=True)
        return NavigationCommand(linear, angular, BOUNDARY_FOLLOWING, heading, clearance,
                                 side, "wall on {}".format("left" if side > 0 else "right"))
