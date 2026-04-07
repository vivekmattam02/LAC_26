"""Tests for mapping module"""

import pathlib
import sys
import types
import unittest
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cv2  # noqa: F401
except Exception:
    sys.modules['cv2'] = types.ModuleType('cv2')

from mapping.voxel_grid import VoxelGrid


class TestVoxelGrid(unittest.TestCase):
    def setUp(self):
        self.grid = VoxelGrid({
            'resolution': 1.0,
            'grid_size': [5, 5, 5],
            'origin': [0.0, 0.0, 0.0],
            'prob_hit': 0.7,
            'prob_miss': 0.4,
            'occ_threshold': 0.5,
        })

    def test_world_to_voxel_conversion(self):
        pts = np.array([[0.2, 0.9, 1.1], [3.9, 4.0, 2.5]])
        vox = self.grid.world_to_voxel(pts)
        np.testing.assert_array_equal(vox[0], [0, 0, 1])
        np.testing.assert_array_equal(vox[1], [3, 4, 2])

    def test_depth_integration(self):
        depth = np.ones((4, 4), dtype=np.float32) * 2.0
        uncertainty = np.ones((4, 4), dtype=np.float32) * 0.2
        pose = np.eye(4, dtype=np.float32)
        intrinsics = {'fx': 1.0, 'fy': 1.0, 'cx': 1.5, 'cy': 1.5}

        self.grid.integrate_depth(depth, uncertainty, pose, intrinsics, subsample=1)

        self.assertGreater(np.sum(self.grid.observation_count), 0)
        self.assertTrue(np.any(self.grid.occupancy > 0))


class TestHeightMap(unittest.TestCase):
    def test_height_extraction(self):
        grid = VoxelGrid({
            'resolution': 1.0,
            'grid_size': [3, 3, 4],
            'origin': [0.0, 0.0, 0.0],
            'occ_threshold': 0.5,
        })

        # Mark a specific occupied voxel with height stats
        i, j, k = 1, 1, 2
        grid.occupancy[i, j, k] = 5.0
        grid.observation_count[i, j, k] = 2
        grid.height_sum[i, j, k] = 5.0  # mean height 2.5
        grid.uncertainty[i, j, k] = 0.3

        heights, uncertainties, valid = grid.get_height_map()
        self.assertTrue(valid[i, j])
        self.assertAlmostEqual(float(heights[i, j]), 2.5, places=5)
        self.assertAlmostEqual(float(uncertainties[i, j]), 0.3, places=5)


if __name__ == '__main__':
    unittest.main()
