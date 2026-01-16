"""
================================================================================
HEIGHT MAP MODULE FOR LAC GEOMETRIC MAPPING
================================================================================

A height map (also called a 2.5D map or elevation map) represents terrain as
a 2D grid where each cell stores the height of the ground at that location.

WHY HEIGHT MAPS FOR LAC:
------------------------
The Lunar Autonomy Challenge scores teams on their geometric mapping accuracy.
You must provide a height map (and rock locations) that the scoring system
compares against ground truth. Better maps = higher scores.

The height map is also essential for:
1. Determining terrain traversability (too steep = impassable)
2. Planning paths around obstacles
3. Identifying craters and rocks
4. Understanding the exploration coverage

HOW IT RELATES TO THE VOXEL GRID:
---------------------------------
The voxel grid stores full 3D information, but for many purposes we only
need the "top surface" - the highest occupied voxel at each (x, y) position.

Height Map = 2D projection of 3D voxel grid

    Voxel Grid (3D)          Height Map (2D)
    +---+---+---+            +---+---+---+
    | | | # |                |   |   | 2 |
    +---+---+---+  ====>     +---+---+---+
    | # | | |                | 1 |   |   |
    +---+---+---+            +---+---+---+

    # = occupied voxel       number = height

LAC GEOMETRIC MAP API:
----------------------
The LAC simulator provides a GeometricMap class with these methods:
- set_height(x, y, height): Set height at world coordinates
- set_rock(x, y, is_rock): Mark if location has a rock
- get_map_array(): Get the map as numpy array for scoring

Our HeightMap class bridges our internal representation to LAC's API.

================================================================================
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import cv2


class HeightMap:
    """
    2.5D height map for terrain representation and LAC scoring.

    This class maintains a 2D grid of height values with associated
    uncertainties and rock flags. It can be updated from the voxel grid
    and exported in the format required by LAC scoring.

    Attributes:
        resolution (float): Grid cell size in meters
        size_x (int): Number of cells in x direction
        size_y (int): Number of cells in y direction
        origin (ndarray): World coordinates of grid corner
        heights (ndarray): Height values in meters
        uncertainties (ndarray): Height uncertainties in meters
        rock_flags (ndarray): Boolean flags for rock presence
        valid_mask (ndarray): Boolean mask for observed cells

    Example Usage:
        >>> config = {'resolution': 0.1, 'size_x': 200, 'size_y': 200}
        >>> height_map = HeightMap(config)
        >>> height_map.update_from_voxel_grid(voxel_grid)
        >>> h = height_map.get_height(5.0, 3.0)  # Height at world (5, 3)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the height map.

        Args:
            config: Configuration dictionary containing:
                - resolution: Cell size in meters (default 0.1)
                - size_x: Number of cells in x (default 200)
                - size_y: Number of cells in y (default 200)
                - origin: [x, y] world coords of corner (default [-10, -10])
                - unknown_height: Default height for unobserved cells (default 0)
                - max_slope: Maximum traversable slope in degrees (default 30)
        """
        # =====================================================================
        # GRID GEOMETRY
        # =====================================================================

        # Resolution: size of each grid cell in meters
        self.resolution = config.get('resolution', 0.1)

        # Grid dimensions
        self.size_x = config.get('size_x', 200)
        self.size_y = config.get('size_y', 200)

        # Origin: world coordinates of the (0, 0) corner
        self.origin = np.array(config.get('origin', [-10.0, -10.0]))

        # Compute world extent
        self.extent = self.origin + np.array([self.size_x, self.size_y]) * self.resolution

        # Default height for unobserved cells
        self.unknown_height = config.get('unknown_height', 0.0)

        # Maximum traversable slope (degrees)
        self.max_slope = config.get('max_slope', 30.0)

        print(f"[HeightMap] Resolution: {self.resolution}m")
        print(f"[HeightMap] Size: {self.size_x} x {self.size_y} cells")
        print(f"[HeightMap] World extent: {self.origin} to {self.extent}")

        # =====================================================================
        # INITIALIZE DATA ARRAYS
        # =====================================================================

        # Height values in meters (z coordinate)
        self.heights = np.full((self.size_x, self.size_y), self.unknown_height, dtype=np.float32)

        # Height uncertainties (standard deviation in meters)
        self.uncertainties = np.full((self.size_x, self.size_y), 10.0, dtype=np.float32)

        # Rock flags (True if rock detected at this location)
        self.rock_flags = np.zeros((self.size_x, self.size_y), dtype=bool)

        # Valid mask (True if this cell has been observed)
        self.valid_mask = np.zeros((self.size_x, self.size_y), dtype=bool)

        # Observation count per cell
        self.observation_count = np.zeros((self.size_x, self.size_y), dtype=np.int32)

        print(f"[HeightMap] Initialized with {self.size_x * self.size_y:,} cells")

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to cell indices.

        Args:
            x: World x coordinate (meters)
            y: World y coordinate (meters)

        Returns:
            (i, j): Cell indices

        Example:
            >>> i, j = height_map.world_to_cell(5.0, 3.0)
        """
        i = int(np.floor((x - self.origin[0]) / self.resolution))
        j = int(np.floor((y - self.origin[1]) / self.resolution))
        return i, j

    def cell_to_world(self, i: int, j: int) -> Tuple[float, float]:
        """
        Convert cell indices to world coordinates (cell center).

        Args:
            i: Cell x index
            j: Cell y index

        Returns:
            (x, y): World coordinates of cell center
        """
        x = self.origin[0] + (i + 0.5) * self.resolution
        y = self.origin[1] + (j + 0.5) * self.resolution
        return x, y

    def is_in_bounds(self, i: int, j: int) -> bool:
        """
        Check if cell indices are within grid bounds.

        Args:
            i, j: Cell indices

        Returns:
            True if within bounds
        """
        return 0 <= i < self.size_x and 0 <= j < self.size_y

    def update_from_voxel_grid(self, voxel_grid):
        """
        Update height map from a voxel grid.

        This extracts the 2.5D height map from the 3D voxel grid by
        finding the highest occupied voxel in each (x, y) column.

        Args:
            voxel_grid: VoxelGrid instance
        """
        # Get height map from voxel grid
        heights_3d, uncertainties_3d, valid_3d = voxel_grid.get_height_map()

        # The voxel grid might have different dimensions, so we need to map
        # cells from voxel grid to our height map

        # Get voxel grid parameters
        vg_origin = voxel_grid.origin[:2]  # Only x, y
        vg_resolution = voxel_grid.resolution
        vg_size = voxel_grid.grid_size[:2]  # Only x, y

        # For each cell in our height map, find corresponding voxel grid cell
        for i in range(self.size_x):
            for j in range(self.size_y):
                # Get world coordinates of our cell center
                x, y = self.cell_to_world(i, j)

                # Convert to voxel grid indices
                vi = int(np.floor((x - vg_origin[0]) / vg_resolution))
                vj = int(np.floor((y - vg_origin[1]) / vg_resolution))

                # Check bounds
                if 0 <= vi < vg_size[0] and 0 <= vj < vg_size[1]:
                    if valid_3d[vi, vj]:
                        self.heights[i, j] = heights_3d[vi, vj]
                        self.uncertainties[i, j] = uncertainties_3d[vi, vj]
                        self.valid_mask[i, j] = True
                        self.observation_count[i, j] += 1

        # After updating heights, detect rocks
        self._detect_rocks()

        print(f"[HeightMap] Updated from voxel grid. Valid cells: {np.sum(self.valid_mask)}")

    def update_point(
        self,
        x: float,
        y: float,
        z: float,
        uncertainty: float = 0.1
    ):
        """
        Update height at a single world point.

        Uses incremental mean and variance update (Welford's algorithm).

        Args:
            x, y: World coordinates
            z: Height value
            uncertainty: Measurement uncertainty
        """
        i, j = self.world_to_cell(x, y)

        if not self.is_in_bounds(i, j):
            return

        n = self.observation_count[i, j]

        if n == 0:
            # First observation
            self.heights[i, j] = z
            self.uncertainties[i, j] = uncertainty
        else:
            # Incremental update (weighted average)
            old_height = self.heights[i, j]
            old_unc = self.uncertainties[i, j]

            # Weight inversely proportional to uncertainty
            w_old = 1.0 / (old_unc ** 2 + 1e-6)
            w_new = 1.0 / (uncertainty ** 2 + 1e-6)

            # Weighted average
            self.heights[i, j] = (w_old * old_height + w_new * z) / (w_old + w_new)

            # Combined uncertainty
            self.uncertainties[i, j] = 1.0 / np.sqrt(w_old + w_new)

        self.observation_count[i, j] += 1
        self.valid_mask[i, j] = True

    def get_height(self, x: float, y: float) -> float:
        """
        Get height at world position.

        Args:
            x, y: World coordinates

        Returns:
            Height in meters (or unknown_height if not observed)
        """
        i, j = self.world_to_cell(x, y)

        if not self.is_in_bounds(i, j):
            return self.unknown_height

        return self.heights[i, j]

    def get_uncertainty(self, x: float, y: float) -> float:
        """
        Get height uncertainty at world position.

        Args:
            x, y: World coordinates

        Returns:
            Uncertainty in meters
        """
        i, j = self.world_to_cell(x, y)

        if not self.is_in_bounds(i, j):
            return 10.0  # High uncertainty outside grid

        return self.uncertainties[i, j]

    def is_valid(self, x: float, y: float) -> bool:
        """
        Check if position has been observed.

        Args:
            x, y: World coordinates

        Returns:
            True if observed
        """
        i, j = self.world_to_cell(x, y)

        if not self.is_in_bounds(i, j):
            return False

        return self.valid_mask[i, j]

    def is_rock(self, x: float, y: float) -> bool:
        """
        Check if position has a rock.

        Args:
            x, y: World coordinates

        Returns:
            True if rock detected
        """
        i, j = self.world_to_cell(x, y)

        if not self.is_in_bounds(i, j):
            return False

        return self.rock_flags[i, j]

    def set_rock(self, x: float, y: float, is_rock: bool):
        """
        Set rock flag at position.

        Args:
            x, y: World coordinates
            is_rock: Whether there's a rock
        """
        i, j = self.world_to_cell(x, y)

        if self.is_in_bounds(i, j):
            self.rock_flags[i, j] = is_rock

    def _detect_rocks(self):
        """
        Automatically detect rocks based on height discontinuities.

        Rocks are characterized by:
        1. Sudden height changes (high gradient)
        2. Small connected regions that are elevated
        3. High local variance in height

        This is a simple heuristic - in practice you might use
        more sophisticated methods or direct rock detection from images.
        """
        # Compute height gradient (slope)
        grad_x = np.gradient(self.heights, self.resolution, axis=0)
        grad_y = np.gradient(self.heights, self.resolution, axis=1)
        slope = np.sqrt(grad_x**2 + grad_y**2)

        # Compute local variance using sliding window
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

        # Mean height in neighborhood
        local_mean = cv2.filter2D(self.heights, -1, kernel)

        # Local variance
        local_var = cv2.filter2D(self.heights**2, -1, kernel) - local_mean**2
        local_var = np.maximum(local_var, 0)  # Numerical stability

        # Rock detection criteria:
        # 1. High slope (steep change) OR
        # 2. High local variance (rough surface)
        # 3. Cell must be observed

        slope_threshold = np.tan(np.radians(45))  # 45 degree slope
        variance_threshold = 0.01  # 10cm^2 variance

        rock_mask = (
            self.valid_mask &
            ((slope > slope_threshold) | (local_var > variance_threshold))
        )

        self.rock_flags = rock_mask

    def compute_slope_map(self) -> np.ndarray:
        """
        Compute slope at each cell (in degrees).

        Slope is computed from the height gradient:
            slope = atan(sqrt(dz/dx^2 + dz/dy^2))

        Returns:
            slope: 2D array of slopes in degrees
        """
        # Compute gradients
        grad_x = np.gradient(self.heights, self.resolution, axis=0)
        grad_y = np.gradient(self.heights, self.resolution, axis=1)

        # Slope magnitude
        slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        slope_deg = np.degrees(slope_rad)

        # Mark invalid cells with -1
        slope_deg[~self.valid_mask] = -1

        return slope_deg

    def compute_traversability(self) -> np.ndarray:
        """
        Compute traversability based on slope and rocks.

        Traversability is a value from 0 (impassable) to 1 (easy).

        Returns:
            traversability: 2D array of traversability values [0, 1]
        """
        slope = self.compute_slope_map()

        # Slope-based traversability
        # 0-15 degrees: fully traversable
        # 15-30 degrees: decreasing traversability
        # >30 degrees: impassable
        trav_slope = np.clip(1.0 - (slope - 15) / 15, 0, 1)

        # Rocks are obstacles
        trav_rock = (~self.rock_flags).astype(np.float32)

        # Unobserved areas are uncertain
        trav_observed = self.valid_mask.astype(np.float32) * 0.5 + 0.5

        # Combined traversability
        traversability = trav_slope * trav_rock * trav_observed

        return traversability

    def export_for_lac(self, geometric_map):
        """
        Export height map to LAC GeometricMap format.

        This copies our height and rock data to the LAC-provided
        geometric map object for scoring.

        Args:
            geometric_map: LAC GeometricMap instance
        """
        for i in range(self.size_x):
            for j in range(self.size_y):
                if self.valid_mask[i, j]:
                    # Get world coordinates
                    x, y = self.cell_to_world(i, j)

                    # Set height
                    geometric_map.set_height(x, y, self.heights[i, j])

                    # Set rock flag
                    geometric_map.set_rock(x, y, self.rock_flags[i, j])

        print(f"[HeightMap] Exported {np.sum(self.valid_mask)} cells to LAC GeometricMap")

    def export_to_file(self, filepath: str):
        """
        Export height map to file for offline analysis.

        Saves as numpy compressed format (.npz) containing:
        - heights
        - uncertainties
        - rock_flags
        - valid_mask
        - metadata

        Args:
            filepath: Output file path
        """
        np.savez_compressed(
            filepath,
            heights=self.heights,
            uncertainties=self.uncertainties,
            rock_flags=self.rock_flags,
            valid_mask=self.valid_mask,
            resolution=self.resolution,
            origin=self.origin,
            size_x=self.size_x,
            size_y=self.size_y
        )
        print(f"[HeightMap] Exported to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> 'HeightMap':
        """
        Load height map from file.

        Args:
            filepath: Input file path

        Returns:
            HeightMap instance
        """
        data = np.load(filepath)

        config = {
            'resolution': float(data['resolution']),
            'origin': data['origin'].tolist(),
            'size_x': int(data['size_x']),
            'size_y': int(data['size_y'])
        }

        height_map = cls(config)
        height_map.heights = data['heights']
        height_map.uncertainties = data['uncertainties']
        height_map.rock_flags = data['rock_flags']
        height_map.valid_mask = data['valid_mask']

        print(f"[HeightMap] Loaded from {filepath}")
        return height_map

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the height map.

        Returns:
            Dictionary with height map statistics
        """
        valid_heights = self.heights[self.valid_mask]
        valid_uncertainties = self.uncertainties[self.valid_mask]

        stats = {
            'total_cells': self.size_x * self.size_y,
            'valid_cells': np.sum(self.valid_mask),
            'coverage': np.sum(self.valid_mask) / (self.size_x * self.size_y) * 100,
            'rock_cells': np.sum(self.rock_flags),
            'mean_height': np.mean(valid_heights) if len(valid_heights) > 0 else 0,
            'min_height': np.min(valid_heights) if len(valid_heights) > 0 else 0,
            'max_height': np.max(valid_heights) if len(valid_heights) > 0 else 0,
            'mean_uncertainty': np.mean(valid_uncertainties) if len(valid_uncertainties) > 0 else 10,
        }

        return stats


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_height_map(config: Dict[str, Any]) -> HeightMap:
    """
    Factory function to create a HeightMap.

    Args:
        config: Configuration dictionary

    Returns:
        HeightMap instance
    """
    return HeightMap(config)


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    """
    Test the height map module.
    """
    print("Testing HeightMap...")

    # Create configuration
    config = {
        'resolution': 0.1,
        'size_x': 100,
        'size_y': 100,
        'origin': [-5.0, -5.0]
    }

    # Create height map
    height_map = HeightMap(config)

    # Add some synthetic height data
    print("\nAdding synthetic height data...")
    for i in range(50):
        for j in range(50):
            x, y = height_map.cell_to_world(i + 25, j + 25)
            # Create a simple slope with some noise
            z = 0.5 + 0.02 * i + 0.01 * j + np.random.randn() * 0.02
            height_map.update_point(x, y, z, uncertainty=0.05)

    # Add some "rocks" (high points)
    for _ in range(10):
        i = np.random.randint(25, 75)
        j = np.random.randint(25, 75)
        x, y = height_map.cell_to_world(i, j)
        z = height_map.get_height(x, y) + 0.3  # Rock is 30cm above ground
        height_map.update_point(x, y, z, uncertainty=0.1)
        height_map.set_rock(x, y, True)

    # Get statistics
    stats = height_map.get_statistics()
    print(f"\nHeight map statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # Compute slope and traversability
    slope = height_map.compute_slope_map()
    traversability = height_map.compute_traversability()

    print(f"\nSlope range: {slope[slope >= 0].min():.1f} - {slope[slope >= 0].max():.1f} degrees")
    print(f"Traversability range: {traversability.min():.2f} - {traversability.max():.2f}")

    # Test get functions
    test_x, test_y = 0.0, 0.0
    print(f"\nAt ({test_x}, {test_y}):")
    print(f"  Height: {height_map.get_height(test_x, test_y):.3f}m")
    print(f"  Uncertainty: {height_map.get_uncertainty(test_x, test_y):.3f}m")
    print(f"  Valid: {height_map.is_valid(test_x, test_y)}")
    print(f"  Rock: {height_map.is_rock(test_x, test_y)}")

    print("\nTest complete!")
