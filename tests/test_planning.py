"""Tests for planning module"""

import pathlib
import sys
import unittest
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from costmap.costmap import PerceptionAwareCostmap
from planning.astar import AStarPlanner
from planning.coverage_planner import CoveragePlanner
from planning.trajectory import TrajectoryGenerator


class DummyVoxelGrid:
    def __init__(self):
        self.resolution = 1.0
        self.origin = np.array([0.0, 0.0, 0.0])
        self._valid = np.zeros((6, 6), dtype=bool)
        self._valid[1:5, 1:3] = True

    def get_height_map(self):
        h = np.zeros_like(self._valid, dtype=np.float32)
        u = np.zeros_like(self._valid, dtype=np.float32)
        return h, u, self._valid

    def voxel_to_world(self, idx):
        idx = np.atleast_2d(idx)
        world = np.zeros((idx.shape[0], 3), dtype=np.float32)
        world[:, 0] = idx[:, 0] + 0.5
        world[:, 1] = idx[:, 1] + 0.5
        world[:, 2] = idx[:, 2] + 0.5
        return world


class TestAStarPlanner(unittest.TestCase):
    def setUp(self):
        self.costmap = PerceptionAwareCostmap({
            'resolution': 1.0,
            'size_x': 10,
            'size_y': 10,
            'origin': [0.0, 0.0],
            'uncertainty_weight': 10.0,
        })
        self.costmap.unknown_mask[:] = False

    def test_simple_path(self):
        planner = AStarPlanner(self.costmap, {'allow_diagonal': True})
        path, cost = planner.plan((1.0, 1.0), (8.0, 8.0))
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 2)
        self.assertGreater(cost, 0)

    def test_obstacle_avoidance(self):
        # Vertical wall with one gap at y=5
        for y in range(10):
            if y == 5:
                continue
            self.costmap.obstacle_layer[5, y] = self.costmap.LETHAL_COST
        self.costmap._costmap_dirty = True

        planner = AStarPlanner(self.costmap, {'allow_diagonal': True})
        path, _ = planner.plan((1.0, 1.0), (8.0, 8.0))
        self.assertIsNotNone(path)

        # Ensure path passes through the gap vicinity
        through_gap = any(abs(px - 5.5) < 1.1 and abs(py - 5.5) < 1.1 for px, py in path)
        self.assertTrue(through_gap)

    def test_uncertainty_avoidance(self):
        # High uncertainty strip in the middle
        unc = np.zeros((10, 10), dtype=np.float32)
        unc[4:6, :] = 5.0
        self.costmap.update_uncertainty_layer(unc)

        planner = AStarPlanner(self.costmap, {'allow_diagonal': True})
        path, _ = planner.plan((1.0, 1.0), (8.0, 8.0))
        self.assertIsNotNone(path)

        # Path should not spend many points in high-uncertainty strip
        risky_points = sum(1 for px, py in path if 4.0 <= px <= 6.0)
        self.assertLess(risky_points, max(2, len(path) // 2))


class TestCoveragePlanner(unittest.TestCase):
    def test_frontier_detection(self):
        planner = CoveragePlanner({'min_frontier_size': 2})
        voxel = DummyVoxelGrid()
        costmap = PerceptionAwareCostmap({'resolution': 1.0, 'size_x': 6, 'size_y': 6, 'origin': [0.0, 0.0]})
        costmap.unknown_mask[:] = False

        frontiers = planner.find_frontiers(voxel)
        self.assertGreater(len(frontiers), 0)

        goal = planner.get_next_goal((1.0, 1.0, 0.0), voxel, costmap)
        self.assertIsNotNone(goal)


class TestTrajectoryGenerator(unittest.TestCase):
    def test_smooth_and_generate(self):
        gen = TrajectoryGenerator({'max_velocity': 1.0, 'smoothing_weight': 0.4})
        path = [(0.0, 0.0), (1.0, 1.5), (2.0, 0.0), (3.0, 0.0)]
        smoothed = gen.smooth_path(path, num_iterations=50)
        self.assertEqual(smoothed[0], path[0])
        self.assertEqual(smoothed[-1], path[-1])

        traj = gen.generate_trajectory(smoothed, dt=0.1)
        self.assertGreater(len(traj), len(path))
        self.assertEqual(traj[0][1:3], (path[0][0], path[0][1]))


if __name__ == '__main__':
    unittest.main()
