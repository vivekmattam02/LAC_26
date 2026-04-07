"""
Visualization for lunar navigation system.

Displays:
- 3D point cloud map
- Costmap with layers
- Planned path
- Robot pose
"""

import numpy as np


class Visualizer:
    def __init__(self, config):
        self.window_name = config.get('window_name', 'Lunar Navigation')
        self.show_uncertainty = config.get('show_uncertainty', True)
        self.show_path = config.get('show_path', True)
        self._fig = None
        self._ax = None

    def setup(self):
        """Initialize visualization window (matplotlib if available)."""
        try:
            import matplotlib.pyplot as plt
            self._fig, self._ax = plt.subplots(num=self.window_name)
            self._ax.set_title(self.window_name)
            self._ax.set_aspect('equal')
            return True
        except Exception:
            self._fig, self._ax = None, None
            return False

    def update(self, voxel_grid, costmap, path, robot_pose):
        """Update visualization frame."""
        if self._ax is None:
            return False

        import matplotlib.pyplot as plt

        self._ax.clear()
        cm = costmap.get_combined_costmap().T
        self._ax.imshow(cm, origin='lower', cmap='inferno')

        if self.show_path and path:
            p = np.array(path)
            self._ax.plot(p[:, 0], p[:, 1], 'c-', linewidth=2)

        if robot_pose is not None and len(robot_pose) >= 2:
            self._ax.plot(robot_pose[0], robot_pose[1], 'go')

        self._ax.set_title(self.window_name)
        plt.pause(0.001)
        return True

    def visualize_costmap(self, costmap):
        """Visualize costmap as 2D image."""
        if self._ax is None:
            return None
        cm = costmap.get_combined_costmap()
        return cm

    def visualize_point_cloud(self, points, colors=None):
        """Return point cloud arrays for optional external plotting."""
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('points must be shape (N, 3)')
        return points, colors

    def uncertainty_to_color(self, uncertainty):
        """Convert uncertainty to color (green=confident, red=uncertain)."""
        u = np.clip(np.asarray(uncertainty, dtype=np.float32), 0.0, 1.0)
        r = u
        g = 1.0 - u
        b = np.zeros_like(u)
        return np.stack([r, g, b], axis=-1)

    def close(self):
        """Close visualization window."""
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._fig)
            except Exception:
                pass
        self._fig, self._ax = None, None
