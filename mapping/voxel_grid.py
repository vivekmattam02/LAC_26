"""
================================================================================
VOXEL GRID MODULE FOR 3D TERRAIN MAPPING
================================================================================

A voxel grid is a 3D array of cubic cells (voxels) that stores information
about the environment. Think of it as a 3D version of a 2D grid map.

WHY VOXEL GRIDS FOR LUNAR NAVIGATION:
-------------------------------------
The lunar terrain has complex 3D structure:
- Craters with varying depths
- Rocks and boulders
- Slopes and inclines
- Overhanging features (rare, but possible)

A voxel grid captures all this 3D information, which is essential for:
1. Detecting obstacles at different heights
2. Understanding terrain traversability
3. Building accurate height maps for scoring
4. Tracking what areas have been explored

HOW THE VOXEL GRID WORKS:
-------------------------
1. The 3D world is divided into a regular grid of cubic cells (voxels)
2. Each voxel stores:
   - Occupancy: Is there something here? (probability)
   - Height statistics: Average and variance of observed heights
   - Uncertainty: How confident are we?
   - Observation count: How many times has this been observed?

3. When we get a new depth measurement:
   - Convert it to 3D points in world frame
   - Update the voxels that these points fall into
   - Also update the "free space" between camera and points

COORDINATE SYSTEMS:
-------------------
- World frame: Fixed reference frame (x: forward, y: left, z: up)
- Voxel indices: Integer (i, j, k) indices into the 3D array
- Conversion: voxel_idx = (world_pos - origin) / resolution

================================================================================
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import cv2


class VoxelGrid:
    """
    3D voxel grid for terrain mapping with uncertainty tracking.

    The voxel grid maintains a probabilistic occupancy map of the environment.
    Each voxel stores occupancy probability (in log-odds form for numerical
    stability) along with height statistics and uncertainty.

    Attributes:
        resolution (float): Size of each voxel in meters
        grid_size (tuple): Number of voxels in (x, y, z)
        origin (ndarray): World coordinates of grid corner (0, 0, 0)
        occupancy (ndarray): Log-odds occupancy probability
        height_sum (ndarray): Sum of heights (for running average)
        height_sq_sum (ndarray): Sum of squared heights (for variance)
        observation_count (ndarray): Number of observations per voxel
        uncertainty (ndarray): Accumulated uncertainty

    Example Usage:
        >>> config = {'resolution': 0.1, 'grid_size': [200, 200, 50]}
        >>> grid = VoxelGrid(config)
        >>> grid.integrate_depth(depth, uncertainty, pose, intrinsics)
        >>> height_map = grid.get_height_map()
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the voxel grid.

        Args:
            config: Configuration dictionary containing:
                - resolution: Voxel size in meters (default 0.1)
                - grid_size: [nx, ny, nz] number of voxels (default [200, 200, 50])
                - origin: [x, y, z] world coords of grid corner (default [-10, -10, -2])
                - prob_hit: Probability of hit for occupied (default 0.7)
                - prob_miss: Probability of miss for free (default 0.4)
                - occ_threshold: Threshold for considering occupied (default 0.5)
        """
        # =====================================================================
        # GRID GEOMETRY
        # =====================================================================

        # Resolution: size of each voxel cube in meters
        # Smaller = more detail but more memory
        # 10cm (0.1m) is a good balance for rover navigation
        self.resolution = config.get('resolution', 0.1)

        # Grid dimensions: number of voxels in each direction
        # Default: 200 x 200 x 50 = 20m x 20m x 5m world coverage
        self.grid_size = np.array(config.get('grid_size', [200, 200, 50]))

        # Origin: world coordinates of the (0, 0, 0) corner of the grid
        # Grid covers from origin to origin + grid_size * resolution
        self.origin = np.array(config.get('origin', [-10.0, -10.0, -2.0]))

        # Compute world extent
        self.extent = self.origin + self.grid_size * self.resolution
        print(f"[VoxelGrid] Resolution: {self.resolution}m")
        print(f"[VoxelGrid] Grid size: {self.grid_size} voxels")
        print(f"[VoxelGrid] World extent: {self.origin} to {self.extent}")

        # =====================================================================
        # OCCUPANCY PROBABILITIES (LOG-ODDS FORM)
        # =====================================================================
        # We use log-odds for numerical stability when multiplying probabilities
        #
        # Log-odds: l = log(p / (1 - p))
        # Probability: p = 1 / (1 + exp(-l))
        #
        # Benefits:
        # - Adding log-odds = multiplying probabilities
        # - No risk of underflow/overflow
        # - Prior of p=0.5 corresponds to l=0

        # Probability that a measured point is truly occupied
        self.prob_hit = config.get('prob_hit', 0.7)

        # Probability that a ray passing through indicates free space
        self.prob_miss = config.get('prob_miss', 0.4)

        # Convert to log-odds
        self.log_odds_hit = np.log(self.prob_hit / (1 - self.prob_hit))
        self.log_odds_miss = np.log(self.prob_miss / (1 - self.prob_miss))

        # Clamping bounds to prevent over-confidence
        self.log_odds_min = -5.0  # Corresponds to p ≈ 0.007
        self.log_odds_max = 5.0   # Corresponds to p ≈ 0.993

        # Threshold for considering a voxel occupied
        self.occ_threshold = config.get('occ_threshold', 0.5)
        self.log_odds_threshold = np.log(self.occ_threshold / (1 - self.occ_threshold))

        # =====================================================================
        # INITIALIZE GRID ARRAYS
        # =====================================================================

        nx, ny, nz = self.grid_size

        # Occupancy in log-odds (initialize to 0 = prior p=0.5)
        self.occupancy = np.zeros((nx, ny, nz), dtype=np.float32)

        # Running sum of heights (for computing mean)
        self.height_sum = np.zeros((nx, ny, nz), dtype=np.float32)

        # Running sum of squared heights (for computing variance)
        self.height_sq_sum = np.zeros((nx, ny, nz), dtype=np.float32)

        # Observation count per voxel
        self.observation_count = np.zeros((nx, ny, nz), dtype=np.int32)

        # Accumulated uncertainty (weighted by observations)
        self.uncertainty = np.ones((nx, ny, nz), dtype=np.float32) * 10.0  # High initial uncertainty

        print(f"[VoxelGrid] Initialized with {nx * ny * nz:,} voxels")
        print(f"[VoxelGrid] Memory usage: ~{(nx * ny * nz * 4 * 5) / 1e6:.1f} MB")

    def world_to_voxel(self, points: np.ndarray) -> np.ndarray:
        """
        Convert world coordinates to voxel indices.

        The conversion is:
            voxel_idx = floor((world_pos - origin) / resolution)

        Args:
            points: World coordinates, shape (N, 3) or (3,)

        Returns:
            voxel_indices: Integer voxel indices, same shape as input
        """
        # Ensure 2D array
        points = np.atleast_2d(points)

        # Convert to voxel coordinates
        voxel_coords = (points - self.origin) / self.resolution

        # Floor to get integer indices
        voxel_indices = np.floor(voxel_coords).astype(np.int32)

        return voxel_indices

    def voxel_to_world(self, voxel_indices: np.ndarray) -> np.ndarray:
        """
        Convert voxel indices to world coordinates (center of voxel).

        Args:
            voxel_indices: Voxel indices, shape (N, 3) or (3,)

        Returns:
            world_coords: World coordinates of voxel centers
        """
        # Ensure 2D array
        voxel_indices = np.atleast_2d(voxel_indices)

        # Convert to world coordinates (add 0.5 to get center)
        world_coords = self.origin + (voxel_indices + 0.5) * self.resolution

        return world_coords

    def is_in_bounds(self, voxel_indices: np.ndarray) -> np.ndarray:
        """
        Check if voxel indices are within grid bounds.

        Args:
            voxel_indices: Voxel indices, shape (N, 3)

        Returns:
            valid: Boolean mask, shape (N,)
        """
        voxel_indices = np.atleast_2d(voxel_indices)

        valid = (
            (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < self.grid_size[0]) &
            (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < self.grid_size[1]) &
            (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < self.grid_size[2])
        )

        return valid

    def integrate_depth(
        self,
        depth: np.ndarray,
        uncertainty: np.ndarray,
        pose: np.ndarray,
        intrinsics: Dict[str, float],
        subsample: int = 4
    ):
        """
        Integrate a depth map into the voxel grid.

        This is the main function for building the map. For each depth measurement:
        1. Convert to 3D point in camera frame
        2. Transform to world frame using pose
        3. Update the voxel containing the point (mark as occupied)
        4. Optionally ray-cast to mark free space

        Args:
            depth: Depth map (H, W) in meters
            uncertainty: Uncertainty map (H, W) in meters
            pose: Camera pose as 4x4 transformation matrix (camera to world)
            intrinsics: Camera intrinsics {'fx', 'fy', 'cx', 'cy'}
            subsample: Process every Nth pixel for speed (default 4)

        Example:
            >>> pose = np.eye(4)  # Camera at origin
            >>> intrinsics = {'fx': 458, 'fy': 458, 'cx': 320, 'cy': 240}
            >>> grid.integrate_depth(depth, uncertainty, pose, intrinsics)
        """
        h, w = depth.shape
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        # Create pixel coordinate grids (subsampled for speed)
        u = np.arange(0, w, subsample)
        v = np.arange(0, h, subsample)
        u, v = np.meshgrid(u, v)
        u = u.flatten()
        v = v.flatten()

        # Get depth and uncertainty at these pixels
        d = depth[v, u]
        unc = uncertainty[v, u]

        # Filter valid depths
        valid = d > 0
        u, v, d, unc = u[valid], v[valid], d[valid], unc[valid]

        if len(d) == 0:
            return

        # =====================================================================
        # STEP 1: Convert depth to 3D points in camera frame
        # =====================================================================
        # Using the pinhole camera model:
        #   X = (u - cx) * Z / fx
        #   Y = (v - cy) * Z / fy
        #   Z = depth

        X_cam = (u - cx) * d / fx
        Y_cam = (v - cy) * d / fy
        Z_cam = d

        # Stack into (N, 3) array
        points_cam = np.stack([X_cam, Y_cam, Z_cam], axis=-1)

        # =====================================================================
        # STEP 2: Transform to world frame
        # =====================================================================
        # pose is a 4x4 matrix: [R | t]
        #                       [0 | 1]
        # world_point = R @ camera_point + t

        # Extract rotation and translation
        R = pose[:3, :3]
        t = pose[:3, 3]

        # Transform points
        points_world = (R @ points_cam.T).T + t

        # =====================================================================
        # STEP 3: Update voxels
        # =====================================================================

        # Convert to voxel indices
        voxel_indices = self.world_to_voxel(points_world)

        # Filter points within bounds
        in_bounds = self.is_in_bounds(voxel_indices)
        voxel_indices = voxel_indices[in_bounds]
        z_world = points_world[in_bounds, 2]  # Height
        unc = unc[in_bounds]

        if len(voxel_indices) == 0:
            return

        # Update each voxel
        for i in range(len(voxel_indices)):
            ix, iy, iz = voxel_indices[i]

            # Update occupancy (log-odds)
            self.occupancy[ix, iy, iz] = np.clip(
                self.occupancy[ix, iy, iz] + self.log_odds_hit,
                self.log_odds_min, self.log_odds_max
            )

            # Update height statistics
            self.height_sum[ix, iy, iz] += z_world[i]
            self.height_sq_sum[ix, iy, iz] += z_world[i] ** 2
            self.observation_count[ix, iy, iz] += 1

            # Update uncertainty (weighted average)
            n = self.observation_count[ix, iy, iz]
            self.uncertainty[ix, iy, iz] = (
                (n - 1) * self.uncertainty[ix, iy, iz] + unc[i]
            ) / n

    def integrate_free_space(
        self,
        camera_position: np.ndarray,
        points_world: np.ndarray,
        max_range: float = 10.0
    ):
        """
        Mark voxels along rays from camera to points as free space.

        This is important for clearing out stale obstacles. If we observe
        free space where we previously thought there was an obstacle,
        we should update our belief.

        Uses ray marching (stepping along the ray at voxel resolution).

        Args:
            camera_position: Camera position in world frame (3,)
            points_world: Observed 3D points in world frame (N, 3)
            max_range: Maximum range for ray casting (meters)
        """
        # This is computationally expensive, so we subsample
        n_points = min(len(points_world), 1000)
        indices = np.random.choice(len(points_world), n_points, replace=False)
        points_subset = points_world[indices]

        for point in points_subset:
            # Direction from camera to point
            direction = point - camera_position
            distance = np.linalg.norm(direction)

            if distance < 0.1 or distance > max_range:
                continue

            direction = direction / distance  # Normalize

            # Step along ray at half voxel resolution
            step_size = self.resolution * 0.5
            n_steps = int(distance / step_size) - 1  # Don't mark the endpoint

            for i in range(1, n_steps):  # Start from 1 to skip camera position
                t = i * step_size
                ray_point = camera_position + t * direction

                # Get voxel index
                voxel_idx = self.world_to_voxel(ray_point.reshape(1, 3))[0]

                if self.is_in_bounds(voxel_idx.reshape(1, 3))[0]:
                    ix, iy, iz = voxel_idx
                    # Decrease occupancy (mark as free)
                    self.occupancy[ix, iy, iz] = np.clip(
                        self.occupancy[ix, iy, iz] + self.log_odds_miss,
                        self.log_odds_min, self.log_odds_max
                    )

    def get_height_map(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract a 2.5D height map from the voxel grid.

        For each (x, y) column, find the highest occupied voxel and
        return its height. This is used for:
        1. LAC geometric map scoring
        2. Costmap generation
        3. Visualization

        Returns:
            heights: 2D array (nx, ny) of heights in meters
            uncertainties: 2D array (nx, ny) of height uncertainties
            valid: 2D boolean array indicating valid cells
        """
        nx, ny, nz = self.grid_size

        # Output arrays
        heights = np.zeros((nx, ny), dtype=np.float32)
        uncertainties = np.ones((nx, ny), dtype=np.float32) * 10.0  # High default
        valid = np.zeros((nx, ny), dtype=bool)

        # Convert log-odds to probability
        prob = 1.0 / (1.0 + np.exp(-self.occupancy))

        # For each column, find highest occupied voxel
        for i in range(nx):
            for j in range(ny):
                # Get occupancy column
                col_prob = prob[i, j, :]
                col_count = self.observation_count[i, j, :]

                # Find occupied voxels (probability > threshold)
                occupied = col_prob > self.occ_threshold

                if np.any(occupied):
                    # Get highest occupied voxel index
                    occupied_indices = np.where(occupied)[0]
                    highest_k = occupied_indices[-1]  # Highest z index

                    # Compute mean height from statistics
                    if col_count[highest_k] > 0:
                        mean_height = self.height_sum[i, j, highest_k] / col_count[highest_k]
                        heights[i, j] = mean_height
                        uncertainties[i, j] = self.uncertainty[i, j, highest_k]
                        valid[i, j] = True
                    else:
                        # Fall back to voxel center height
                        voxel_center = self.voxel_to_world(np.array([[i, j, highest_k]]))[0]
                        heights[i, j] = voxel_center[2]
                        valid[i, j] = True

        return heights, uncertainties, valid

    def get_point_cloud(
        self,
        prob_threshold: float = 0.5,
        max_points: int = 100000
    ) -> np.ndarray:
        """
        Extract occupied voxels as a point cloud.

        Useful for visualization and debugging.

        Args:
            prob_threshold: Minimum occupancy probability
            max_points: Maximum number of points to return

        Returns:
            points: Point cloud (N, 3) in world coordinates
        """
        # Convert log-odds to probability
        prob = 1.0 / (1.0 + np.exp(-self.occupancy))

        # Find occupied voxels
        log_odds_thresh = np.log(prob_threshold / (1 - prob_threshold))
        occupied = self.occupancy > log_odds_thresh

        # Get indices of occupied voxels
        indices = np.array(np.where(occupied)).T  # (N, 3)

        if len(indices) > max_points:
            # Randomly subsample
            choice = np.random.choice(len(indices), max_points, replace=False)
            indices = indices[choice]

        # Convert to world coordinates
        points = self.voxel_to_world(indices)

        return points

    def get_occupancy_probability(self, point: np.ndarray) -> float:
        """
        Get occupancy probability at a world point.

        Args:
            point: World coordinates (3,) or (N, 3)

        Returns:
            probability: Occupancy probability [0, 1]
        """
        voxel_idx = self.world_to_voxel(point.reshape(1, 3))[0]

        if not self.is_in_bounds(voxel_idx.reshape(1, 3))[0]:
            return 0.5  # Unknown outside grid

        ix, iy, iz = voxel_idx
        log_odds = self.occupancy[ix, iy, iz]
        prob = 1.0 / (1.0 + np.exp(-log_odds))

        return prob

    def clear_region(
        self,
        center: np.ndarray,
        radius: float
    ):
        """
        Clear a spherical region (set to unknown).

        Useful for resetting a region when we know it's changed.

        Args:
            center: Center of region in world coordinates (3,)
            radius: Radius in meters
        """
        # Get bounding box in voxel coordinates
        min_corner = center - radius
        max_corner = center + radius

        min_voxel = self.world_to_voxel(min_corner.reshape(1, 3))[0]
        max_voxel = self.world_to_voxel(max_corner.reshape(1, 3))[0]

        # Clamp to grid bounds
        min_voxel = np.maximum(min_voxel, 0)
        max_voxel = np.minimum(max_voxel, self.grid_size - 1)

        # Reset region
        for i in range(min_voxel[0], max_voxel[0] + 1):
            for j in range(min_voxel[1], max_voxel[1] + 1):
                for k in range(min_voxel[2], max_voxel[2] + 1):
                    # Check if within sphere
                    voxel_center = self.voxel_to_world(np.array([[i, j, k]]))[0]
                    if np.linalg.norm(voxel_center - center) <= radius:
                        self.occupancy[i, j, k] = 0.0
                        self.height_sum[i, j, k] = 0.0
                        self.height_sq_sum[i, j, k] = 0.0
                        self.observation_count[i, j, k] = 0
                        self.uncertainty[i, j, k] = 10.0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the voxel grid state.

        Returns:
            stats: Dictionary with grid statistics
        """
        prob = 1.0 / (1.0 + np.exp(-self.occupancy))

        occupied = prob > self.occ_threshold
        free = prob < (1 - self.occ_threshold)
        unknown = ~occupied & ~free

        observed = self.observation_count > 0

        stats = {
            'total_voxels': np.prod(self.grid_size),
            'occupied_voxels': np.sum(occupied),
            'free_voxels': np.sum(free),
            'unknown_voxels': np.sum(unknown),
            'observed_voxels': np.sum(observed),
            'mean_observations': np.mean(self.observation_count[observed]) if np.any(observed) else 0,
            'mean_uncertainty': np.mean(self.uncertainty[observed]) if np.any(observed) else 10.0,
        }

        return stats


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_voxel_grid(config: Dict[str, Any]) -> VoxelGrid:
    """
    Factory function to create a VoxelGrid.

    Args:
        config: Configuration dictionary

    Returns:
        VoxelGrid instance
    """
    return VoxelGrid(config)


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    """
    Test the voxel grid module.
    """
    print("Testing VoxelGrid...")

    # Create configuration
    config = {
        'resolution': 0.1,
        'grid_size': [100, 100, 30],
        'origin': [-5.0, -5.0, -1.0]
    }

    # Create voxel grid
    grid = VoxelGrid(config)

    # Create synthetic depth data
    h, w = 480, 640
    fx, fy, cx, cy = 458.0, 458.0, 320.0, 240.0

    # Depth map: planar surface at 2m with some variation
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)
    depth = 2.0 + 0.01 * ((u - cx) + (v - cy)) / fx  # Slight slope
    depth = depth.astype(np.float32)

    # Uncertainty: higher at edges
    uncertainty = 0.05 + 0.1 * np.sqrt((u - cx)**2 + (v - cy)**2) / fx
    uncertainty = uncertainty.astype(np.float32)

    # Camera pose at origin looking forward
    pose = np.eye(4)

    # Intrinsics
    intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

    # Integrate depth
    print("\nIntegrating depth map...")
    grid.integrate_depth(depth, uncertainty, pose, intrinsics, subsample=4)

    # Get statistics
    stats = grid.get_statistics()
    print(f"\nGrid statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Extract height map
    print("\nExtracting height map...")
    heights, uncertainties, valid = grid.get_height_map()
    print(f"Valid cells: {np.sum(valid)} / {valid.size}")
    if np.any(valid):
        print(f"Height range: {heights[valid].min():.2f}m - {heights[valid].max():.2f}m")

    # Extract point cloud
    points = grid.get_point_cloud(prob_threshold=0.5)
    print(f"\nPoint cloud: {len(points)} points")

    print("\nTest complete!")
