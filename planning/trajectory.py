"""
Trajectory generation and smoothing.

Converts discrete path to smooth, time-parameterized trajectory.
"""

import numpy as np


class TrajectoryGenerator:
    def __init__(self, config):
        self.max_velocity = config.get('max_velocity', 1.0)
        self.max_acceleration = config.get('max_acceleration', 0.5)
        self.smoothing_weight = config.get('smoothing_weight', 0.5)

    def smooth_path(self, path, num_iterations=100):
        """
        Smooth path using iterative averaging while preserving endpoints.
        """
        if path is None or len(path) < 3:
            return path

        smoothed = np.array(path, dtype=np.float64)
        original = np.array(path, dtype=np.float64)

        alpha = np.clip(self.smoothing_weight, 0.0, 1.0)
        beta = 1.0 - alpha

        for _ in range(num_iterations):
            prev = smoothed.copy()
            for i in range(1, len(smoothed) - 1):
                smooth_term = 0.5 * (smoothed[i - 1] + smoothed[i + 1])
                smoothed[i] = alpha * original[i] + beta * smooth_term
            if np.max(np.abs(smoothed - prev)) < 1e-4:
                break

        return [tuple(p) for p in smoothed]

    def generate_trajectory(self, path, dt=0.1):
        """
        Generate time-parameterized trajectory.

        Returns:
            trajectory: List of (time, x, y, vx, vy)
        """
        if path is None or len(path) == 0:
            return []

        if len(path) == 1:
            x, y = path[0]
            return [(0.0, float(x), float(y), 0.0, 0.0)]

        path_np = np.array(path, dtype=np.float64)
        traj = []
        t = 0.0

        # Start point
        traj.append((t, float(path_np[0, 0]), float(path_np[0, 1]), 0.0, 0.0))

        for i in range(1, len(path_np)):
            p0 = path_np[i - 1]
            p1 = path_np[i]
            delta = p1 - p0
            dist = np.linalg.norm(delta)
            if dist < 1e-8:
                continue

            direction = delta / dist
            speed = min(self.max_velocity, max(1e-3, dist / max(dt, 1e-6)))
            vx, vy = direction * speed

            steps = max(1, int(np.ceil(dist / (speed * dt))))
            for s in range(1, steps + 1):
                ratio = s / steps
                p = p0 + ratio * delta
                t += dt
                traj.append((t, float(p[0]), float(p[1]), float(vx), float(vy)))

        return traj
