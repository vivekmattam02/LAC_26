"""
================================================================================
TRAJECTORY FOLLOWING CONTROLLER
================================================================================

This module converts a planned path into velocity commands that the rover
can execute. We use two classic control algorithms:

1. PURE PURSUIT: For steering (angular velocity)
2. PID CONTROL: For speed (linear velocity)

WHY THESE ALGORITHMS:
---------------------
Pure Pursuit is simple, robust, and widely used in robotics because:
- Naturally smooths the path (doesn't try to hit every waypoint exactly)
- Single tuning parameter (lookahead distance)
- Stable behavior even with noisy localization

PID Control for speed because:
- Simple and well-understood
- Easy to tune
- Handles varying terrain resistance

HOW PURE PURSUIT WORKS:
-----------------------
1. Find a "lookahead point" on the path that's a fixed distance ahead
2. Compute the steering angle needed to drive toward that point
3. The geometry gives us: curvature = 2 * sin(angle) / lookahead_distance

The lookahead distance affects behavior:
- Short lookahead: Follows path closely but may oscillate
- Long lookahead: Smoother but may cut corners

FOR LAC SIMULATOR:
------------------
The LAC rover uses VehicleVelocityControl with:
- linear_target_velocity: Forward/backward speed (m/s)
- angular_target_velocity: Turning rate (rad/s)

This is a "differential drive" or "skid-steer" model - the rover turns
by driving wheels at different speeds.

================================================================================
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any


class Controller:
    """
    Trajectory following controller using Pure Pursuit and PID.

    This controller takes the current robot pose and a target path,
    and outputs velocity commands (linear and angular) to follow the path.

    Attributes:
        lookahead_distance (float): How far ahead to look on path (meters)
        max_linear_velocity (float): Maximum forward speed (m/s)
        max_angular_velocity (float): Maximum turning rate (rad/s)
        goal_tolerance (float): Distance to consider goal reached (meters)

    Example Usage:
        >>> controller = Controller(config)
        >>> linear_vel, angular_vel = controller.get_control(pose, trajectory)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the controller.

        Args:
            config: Configuration dictionary containing:
                - lookahead_distance: Pure pursuit lookahead (default 1.0)
                - max_linear_velocity: Max forward speed (default 0.4)
                - max_angular_velocity: Max turning rate (default 1.0)
                - goal_tolerance: Distance to goal to stop (default 0.3)
                - kp, ki, kd: PID gains for speed control
        """
        # =====================================================================
        # PURE PURSUIT PARAMETERS
        # =====================================================================

        # Lookahead distance: How far ahead on the path to aim for
        # Larger = smoother but less precise path following
        # Smaller = more precise but may oscillate
        # Rule of thumb: 1-3x the robot length
        self.lookahead_distance = config.get('lookahead_distance', 1.0)

        # Minimum lookahead (prevent division issues when close to goal)
        self.min_lookahead = config.get('min_lookahead', 0.3)

        # =====================================================================
        # VELOCITY LIMITS
        # =====================================================================

        # LAC IPEx rover limits (from autonomous_agent.py):
        # MAX_LINEAR_SPEED = 0.49 m/s
        # MAX_ANGULAR_SPEED = 4.13 rad/s
        self.max_linear_velocity = config.get('max_linear_velocity', 0.4)
        self.max_angular_velocity = config.get('max_angular_velocity', 1.0)

        # Minimum speed (to avoid getting stuck)
        self.min_linear_velocity = config.get('min_linear_velocity', 0.05)

        # =====================================================================
        # GOAL AND WAYPOINT TOLERANCES
        # =====================================================================

        # Distance at which we consider the goal reached
        self.goal_tolerance = config.get('goal_tolerance', 0.3)

        # Distance to switch to next waypoint
        self.waypoint_tolerance = config.get('waypoint_tolerance', 0.5)

        # =====================================================================
        # PID CONTROLLER FOR SPEED
        # =====================================================================

        # PID gains
        self.kp = config.get('kp', 1.0)   # Proportional
        self.ki = config.get('ki', 0.0)   # Integral (usually 0 for rovers)
        self.kd = config.get('kd', 0.1)   # Derivative

        # PID state
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_time = None

        # =====================================================================
        # STATE
        # =====================================================================

        # Current waypoint index in the trajectory
        self.current_waypoint_index = 0

        # Flag if we've reached the goal
        self.goal_reached = False

        print(f"[Controller] Initialized with lookahead={self.lookahead_distance}m")
        print(f"[Controller] Max velocities: linear={self.max_linear_velocity}m/s, "
              f"angular={self.max_angular_velocity}rad/s")

    def get_control(
        self,
        current_pose: Tuple[float, float, float],
        trajectory: List[Tuple[float, float]],
        current_velocity: float = 0.0,
        dt: float = 0.1
    ) -> Tuple[float, float]:
        """
        Compute velocity commands to follow the trajectory.

        This is the main function called each control loop.

        Args:
            current_pose: Current robot pose (x, y, theta)
                         x, y in meters, theta in radians (0 = +x direction)
            trajectory: List of (x, y) waypoints to follow
            current_velocity: Current forward velocity (m/s)
            dt: Time step since last call (seconds)

        Returns:
            linear_velocity: Forward velocity command (m/s)
            angular_velocity: Turning rate command (rad/s)

        Example:
            >>> pose = (1.0, 2.0, 0.5)  # x, y, heading
            >>> path = [(2.0, 2.5), (3.0, 3.0), (4.0, 3.5)]
            >>> v, w = controller.get_control(pose, path)
        """
        if not trajectory or len(trajectory) == 0:
            return 0.0, 0.0

        x, y, theta = current_pose

        # Check if we've reached the goal (last waypoint)
        goal = trajectory[-1]
        distance_to_goal = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)

        if distance_to_goal < self.goal_tolerance:
            self.goal_reached = True
            return 0.0, 0.0

        # Find lookahead point on trajectory
        lookahead_point = self._find_lookahead_point(current_pose, trajectory)

        if lookahead_point is None:
            # No valid lookahead point, stop
            return 0.0, 0.0

        # Pure pursuit: compute angular velocity
        angular_velocity = self.pure_pursuit(current_pose, lookahead_point)

        # PID: compute linear velocity
        # Slow down when turning sharply or near obstacles
        curvature = abs(angular_velocity) / max(self.max_linear_velocity, 0.1)
        target_speed = self.max_linear_velocity * (1.0 - 0.5 * min(curvature, 1.0))

        # Also slow down near goal
        slowdown_distance = 1.0  # Start slowing down 1m from goal
        if distance_to_goal < slowdown_distance:
            target_speed *= (distance_to_goal / slowdown_distance)

        target_speed = max(target_speed, self.min_linear_velocity)

        linear_velocity = self.pid_speed(current_velocity, target_speed, dt)

        # Clamp to limits
        linear_velocity = np.clip(linear_velocity, -self.max_linear_velocity, self.max_linear_velocity)
        angular_velocity = np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)

        return linear_velocity, angular_velocity

    def _find_lookahead_point(
        self,
        current_pose: Tuple[float, float, float],
        trajectory: List[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """
        Find the lookahead point on the trajectory.

        The lookahead point is the intersection of:
        1. A circle centered at robot position with radius = lookahead_distance
        2. The path (trajectory)

        If multiple intersections, pick the one furthest along the path.

        Args:
            current_pose: Robot pose (x, y, theta)
            trajectory: Path waypoints

        Returns:
            lookahead_point: (x, y) of lookahead point, or None if not found
        """
        x, y, theta = current_pose
        robot_pos = np.array([x, y])

        # Use dynamic lookahead: shorter when close to goal
        goal = np.array(trajectory[-1])
        dist_to_goal = np.linalg.norm(robot_pos - goal)
        lookahead = min(self.lookahead_distance, dist_to_goal * 0.8)
        lookahead = max(lookahead, self.min_lookahead)

        # Find intersection with path segments
        best_point = None
        best_dist_along_path = -1

        for i in range(len(trajectory) - 1):
            p1 = np.array(trajectory[i])
            p2 = np.array(trajectory[i + 1])

            # Find intersection of circle (robot_pos, lookahead) with segment (p1, p2)
            intersection = self._circle_segment_intersection(
                robot_pos, lookahead, p1, p2
            )

            if intersection is not None:
                # Compute distance along path to this intersection
                dist_along_path = i + np.linalg.norm(intersection - p1) / (np.linalg.norm(p2 - p1) + 1e-6)

                if dist_along_path > best_dist_along_path:
                    best_point = intersection
                    best_dist_along_path = dist_along_path

        # If no intersection found, use closest waypoint ahead
        if best_point is None:
            # Find closest waypoint that's ahead of robot
            for i in range(len(trajectory)):
                wp = np.array(trajectory[i])
                dist = np.linalg.norm(wp - robot_pos)

                # Check if waypoint is roughly ahead (dot product with heading)
                to_wp = wp - robot_pos
                heading = np.array([np.cos(theta), np.sin(theta)])

                if np.dot(to_wp, heading) > 0 and dist > 0.1:
                    return tuple(wp)

            # Last resort: return goal
            return trajectory[-1]

        return tuple(best_point)

    def _circle_segment_intersection(
        self,
        center: np.ndarray,
        radius: float,
        p1: np.ndarray,
        p2: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Find intersection of a circle with a line segment.

        Uses the quadratic formula to solve for intersection points,
        then checks if they lie within the segment.

        Args:
            center: Circle center (x, y)
            radius: Circle radius
            p1, p2: Line segment endpoints

        Returns:
            intersection: Point of intersection closest to p2, or None
        """
        d = p2 - p1  # Direction vector
        f = p1 - center  # Vector from center to p1

        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return None  # No intersection

        discriminant = np.sqrt(discriminant)

        # Two possible solutions
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)

        # Check if solutions are within segment [0, 1]
        intersections = []
        for t in [t1, t2]:
            if 0 <= t <= 1:
                point = p1 + t * d
                intersections.append((t, point))

        if not intersections:
            return None

        # Return intersection closest to p2 (furthest along segment)
        intersections.sort(key=lambda x: x[0], reverse=True)
        return intersections[0][1]

    def pure_pursuit(
        self,
        current_pose: Tuple[float, float, float],
        lookahead_point: Tuple[float, float]
    ) -> float:
        """
        Pure pursuit steering control.

        Computes the angular velocity needed to drive toward the lookahead point.

        THE MATH:
        ---------
        Let (gx, gy) be the lookahead point in robot's local frame.

        The curvature of the arc to reach this point is:
            curvature = 2 * gy / L²

        Where L is the distance to the lookahead point.

        Angular velocity is:
            omega = curvature * velocity

        But since we're computing omega directly (not steering angle),
        we use: omega = 2 * v * sin(alpha) / L

        Where alpha is the angle to the lookahead point.

        Args:
            current_pose: Robot pose (x, y, theta)
            lookahead_point: Target point (x, y)

        Returns:
            angular_velocity: Turning rate (rad/s)
        """
        x, y, theta = current_pose
        gx, gy = lookahead_point

        # Vector from robot to goal
        dx = gx - x
        dy = gy - y

        # Distance to lookahead point
        L = np.sqrt(dx**2 + dy**2)
        if L < 0.01:
            return 0.0

        # Angle to goal in world frame
        angle_to_goal = np.arctan2(dy, dx)

        # Angle error (how much we need to turn)
        # This is the angle to goal in robot's local frame
        alpha = self._normalize_angle(angle_to_goal - theta)

        # Pure pursuit formula: curvature = 2 * sin(alpha) / L
        # Angular velocity = curvature * linear_velocity
        # Simplified: omega = 2 * v * sin(alpha) / L
        # We use a gain instead of velocity for direct omega control

        curvature = 2.0 * np.sin(alpha) / L
        angular_velocity = curvature * self.max_linear_velocity

        return angular_velocity

    def pid_speed(
        self,
        current_speed: float,
        target_speed: float,
        dt: float
    ) -> float:
        """
        PID controller for speed.

        Args:
            current_speed: Current forward velocity (m/s)
            target_speed: Desired forward velocity (m/s)
            dt: Time step (seconds)

        Returns:
            velocity_command: Velocity to command
        """
        # Error
        error = target_speed - current_speed

        # Proportional term
        P = self.kp * error

        # Integral term
        self.integral_error += error * dt
        self.integral_error = np.clip(self.integral_error, -1.0, 1.0)  # Anti-windup
        I = self.ki * self.integral_error

        # Derivative term
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0
        D = self.kd * derivative

        self.prev_error = error

        # Total control output
        control = P + I + D

        # Simple approach: just use target + small correction
        velocity = target_speed + 0.5 * control

        return velocity

    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-pi, pi].

        Args:
            angle: Angle in radians

        Returns:
            Normalized angle in [-pi, pi]
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def reset(self):
        """Reset controller state for a new trajectory."""
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.current_waypoint_index = 0
        self.goal_reached = False
        print("[Controller] Reset")

    def is_goal_reached(self) -> bool:
        """Check if the goal has been reached."""
        return self.goal_reached


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_controller(config: Dict[str, Any]) -> Controller:
    """Factory function to create a Controller."""
    return Controller(config)


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    """Test the controller module."""
    print("Testing Controller...")

    config = {
        'lookahead_distance': 1.0,
        'max_linear_velocity': 0.4,
        'max_angular_velocity': 1.0,
        'kp': 1.0,
        'ki': 0.0,
        'kd': 0.1
    }

    controller = Controller(config)

    # Create a simple path
    trajectory = [
        (0.0, 0.0),
        (1.0, 0.5),
        (2.0, 1.0),
        (3.0, 1.0),
        (4.0, 0.5),
        (5.0, 0.0)
    ]

    # Simulate robot following the path
    print("\nSimulating path following...")
    x, y, theta = -0.5, 0.0, 0.0  # Start pose
    dt = 0.1

    for step in range(100):
        # Get control commands
        v, w = controller.get_control((x, y, theta), trajectory, v if step > 0 else 0, dt)

        if controller.is_goal_reached():
            print(f"\nGoal reached at step {step}!")
            break

        # Simulate robot motion (simple kinematic model)
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt

        if step % 10 == 0:
            print(f"Step {step}: pos=({x:.2f}, {y:.2f}), heading={np.degrees(theta):.1f}deg, "
                  f"v={v:.2f}m/s, w={np.degrees(w):.1f}deg/s")

    print(f"\nFinal position: ({x:.2f}, {y:.2f})")
    print(f"Goal position: {trajectory[-1]}")
    print(f"Distance to goal: {np.sqrt((x-trajectory[-1][0])**2 + (y-trajectory[-1][1])**2):.2f}m")

    print("\nTest complete!")
