"""
================================================================================
PERCEPTION-AWARE COSTMAP MODULE
================================================================================

A costmap assigns a "cost" value to each location in the environment.
The path planner uses these costs to find paths that are not just short,
but also SAFE and traverse through well-observed areas.

WHY PERCEPTION-AWARE COSTMAPS:
------------------------------
Traditional costmaps only consider obstacles (binary: free or occupied).
But for a lunar rover with limited sensing, this isn't enough:

1. UNCERTAINTY: Some areas are well-mapped, others poorly observed.
   We should prefer paths through areas we're confident about.

2. SHADOWS: The rover's cameras struggle in shadows - SLAM may fail.
   Avoid shadow regions where possible.

3. TERRAIN: Steep slopes are harder to traverse than flat ground.

The perception-aware costmap combines multiple "layers":

    Total Cost = w_obs * Obstacle + w_unc * Uncertainty + w_shd * Shadow + ...

COST LAYERS:
------------
1. Obstacle Layer:
   - High cost (infinite) for occupied cells
   - Zero cost for free cells
   - Inflated around obstacles for safety margin

2. Uncertainty Layer:
   - Low cost where we have confident measurements
   - High cost where uncertainty is large
   - Encourages exploration of uncertain areas when safe

3. Shadow Layer:
   - High cost in shadow regions where perception degrades
   - Based on lighting analysis or SLAM confidence

4. Slope Layer:
   - Cost increases with terrain slope
   - Prevents rover from attempting steep climbs

HOW IT'S USED:
--------------
The A* planner queries: "What's the cost to traverse cell (i, j)?"
It then finds the path that minimizes total cost while reaching the goal.

================================================================================
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import cv2
from scipy.ndimage import distance_transform_edt


class PerceptionAwareCostmap:
    """
    Multi-layer costmap for perception-aware path planning.

    This costmap combines multiple information sources to guide the planner
    toward safe, well-observed paths rather than just short paths.

    Attributes:
        resolution (float): Grid cell size in meters
        size_x (int): Number of cells in x direction
        size_y (int): Number of cells in y direction
        origin (ndarray): World coordinates of grid corner
        layers (dict): Individual cost layers
        weights (dict): Weights for combining layers

    Example Usage:
        >>> costmap = PerceptionAwareCostmap(config)
        >>> costmap.update_from_height_map(height_map)
        >>> cost = costmap.get_cost(5.0, 3.0)  # Cost at world (5, 3)
    """

    # Cost value meanings
    FREE_COST = 0.0              # Completely free to traverse
    UNKNOWN_COST = 50.0          # Unknown area (neither free nor occupied)
    INSCRIBED_COST = 200.0       # Too close to obstacle for robot body
    LETHAL_COST = 255.0          # Obstacle - cannot traverse

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the perception-aware costmap.

        Args:
            config: Configuration dictionary containing:
                - resolution: Cell size in meters (default 0.2)
                - size_x: Number of cells in x (default 100)
                - size_y: Number of cells in y (default 100)
                - origin: [x, y] world coords of corner (default [-10, -10])
                - obstacle_weight: Weight for obstacle layer (default 100)
                - uncertainty_weight: Weight for uncertainty layer (default 5)
                - shadow_weight: Weight for shadow layer (default 10)
                - inflation_radius: Robot inflation radius in meters (default 0.3)
        """
        # =====================================================================
        # GRID GEOMETRY
        # =====================================================================

        self.resolution = config.get('resolution', 0.2)
        self.size_x = config.get('size_x', 100)
        self.size_y = config.get('size_y', 100)
        self.origin = np.array(config.get('origin', [-10.0, -10.0]))

        # Robot parameters
        self.inflation_radius = config.get('inflation_radius', 0.3)  # meters
        self.inflation_cells = int(np.ceil(self.inflation_radius / self.resolution))

        # =====================================================================
        # LAYER WEIGHTS
        # These control how much each factor contributes to total cost
        # =====================================================================

        self.weights = {
            'obstacle': config.get('obstacle_weight', 100.0),
            'uncertainty': config.get('uncertainty_weight', 5.0),
            'shadow': config.get('shadow_weight', 10.0),
            'slope': config.get('slope_weight', 3.0),
            'inflation': config.get('inflation_weight', 20.0)
        }

        # =====================================================================
        # INITIALIZE COST LAYERS
        # Each layer is a 2D array of the same size
        # =====================================================================

        # Obstacle layer: LETHAL where obstacles exist
        self.obstacle_layer = np.zeros((self.size_x, self.size_y), dtype=np.float32)

        # Inflation layer: Distance-based cost around obstacles
        self.inflation_layer = np.zeros((self.size_x, self.size_y), dtype=np.float32)

        # Uncertainty layer: High where depth/mapping uncertainty is high
        self.uncertainty_layer = np.zeros((self.size_x, self.size_y), dtype=np.float32)

        # Shadow layer: High in shadow regions
        self.shadow_layer = np.zeros((self.size_x, self.size_y), dtype=np.float32)

        # Slope layer: Cost based on terrain steepness
        self.slope_layer = np.zeros((self.size_x, self.size_y), dtype=np.float32)

        # Unknown mask: True where we haven't observed
        self.unknown_mask = np.ones((self.size_x, self.size_y), dtype=bool)

        # Combined costmap (cached for efficiency)
        self._combined_costmap = None
        self._costmap_dirty = True

        print(f"[Costmap] Resolution: {self.resolution}m")
        print(f"[Costmap] Size: {self.size_x} x {self.size_y} cells")
        print(f"[Costmap] Inflation radius: {self.inflation_radius}m ({self.inflation_cells} cells)")

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to cell indices.

        Args:
            x, y: World coordinates (meters)

        Returns:
            (i, j): Cell indices
        """
        i = int(np.floor((x - self.origin[0]) / self.resolution))
        j = int(np.floor((y - self.origin[1]) / self.resolution))
        return i, j

    def cell_to_world(self, i: int, j: int) -> Tuple[float, float]:
        """
        Convert cell indices to world coordinates (cell center).

        Args:
            i, j: Cell indices

        Returns:
            (x, y): World coordinates
        """
        x = self.origin[0] + (i + 0.5) * self.resolution
        y = self.origin[1] + (j + 0.5) * self.resolution
        return x, y

    def is_in_bounds(self, i: int, j: int) -> bool:
        """Check if cell indices are within grid bounds."""
        return 0 <= i < self.size_x and 0 <= j < self.size_y

    def update_from_height_map(self, height_map):
        """
        Update costmap from a HeightMap object.

        This is the main update function that derives obstacles, slopes,
        and uncertainty from the height map.

        Args:
            height_map: HeightMap instance
        """
        # Get height map parameters
        hm_resolution = height_map.resolution
        hm_origin = height_map.origin

        # For each cell in costmap, sample from height map
        for i in range(self.size_x):
            for j in range(self.size_y):
                # Get world coordinates
                x, y = self.cell_to_world(i, j)

                # Check if valid in height map
                if height_map.is_valid(x, y):
                    self.unknown_mask[i, j] = False

                    # Get uncertainty
                    unc = height_map.get_uncertainty(x, y)
                    self.uncertainty_layer[i, j] = np.clip(unc * 10, 0, 50)

                    # Check for rocks (obstacles)
                    if height_map.is_rock(x, y):
                        self.obstacle_layer[i, j] = self.LETHAL_COST

        # Compute slope from height map
        self._update_slope_layer(height_map)

        # Inflate obstacles
        self._inflate_obstacles()

        # Mark as needing recomputation
        self._costmap_dirty = True

        print(f"[Costmap] Updated from height map. Unknown: {np.sum(self.unknown_mask)} cells")

    def _update_slope_layer(self, height_map):
        """
        Compute slope-based costs from height map.

        Steeper slopes = higher cost.
        """
        # Get slope from height map
        slope = height_map.compute_slope_map()

        # Map slope to cost
        # 0-15 degrees: 0 cost
        # 15-30 degrees: increasing cost
        # >30 degrees: maximum cost

        for i in range(self.size_x):
            for j in range(self.size_y):
                x, y = self.cell_to_world(i, j)
                hm_i, hm_j = height_map.world_to_cell(x, y)

                if height_map.is_in_bounds(hm_i, hm_j):
                    s = slope[hm_i, hm_j]
                    if s >= 0:  # Valid slope
                        if s < 15:
                            self.slope_layer[i, j] = 0
                        elif s < 30:
                            self.slope_layer[i, j] = (s - 15) / 15 * 50
                        else:
                            self.slope_layer[i, j] = 100

    def _inflate_obstacles(self):
        """
        Inflate obstacles to create a safety buffer around them.

        Uses distance transform to compute distance from obstacles,
        then applies exponential decay for cost.

        INFLATION EXPLAINED:
        The robot has a physical size. If we plan a path that passes
        right next to an obstacle, the robot body might collide.

        Inflation creates a "buffer zone" around obstacles:
        - Inside inflation radius: Very high cost
        - Cost decreases with distance from obstacle
        """
        # Create binary obstacle mask
        obstacle_mask = self.obstacle_layer >= self.LETHAL_COST

        if not np.any(obstacle_mask):
            self.inflation_layer[:] = 0
            return

        # Distance transform: distance to nearest obstacle
        # For non-obstacles, this gives the distance to the nearest obstacle
        dist_to_obstacle = distance_transform_edt(~obstacle_mask)

        # Convert distance (in cells) to meters
        dist_meters = dist_to_obstacle * self.resolution

        # Compute inflation cost
        # Cost = INSCRIBED if within inscribed radius
        # Cost decays exponentially with distance

        inscribed_radius = self.inflation_radius * 0.5  # Inner circle

        # Cells within inscribed radius
        inscribed_mask = dist_meters < inscribed_radius
        self.inflation_layer[inscribed_mask] = self.INSCRIBED_COST

        # Cells between inscribed and inflation radius: exponential decay
        inflation_mask = (dist_meters >= inscribed_radius) & (dist_meters < self.inflation_radius)

        # Exponential decay: cost = max_cost * exp(-distance/scale)
        scale = self.inflation_radius / 3.0  # Decay scale
        decay_cost = self.INSCRIBED_COST * np.exp(-(dist_meters - inscribed_radius) / scale)
        self.inflation_layer[inflation_mask] = decay_cost[inflation_mask]

        # Outside inflation radius: zero inflation cost
        outside_mask = dist_meters >= self.inflation_radius
        self.inflation_layer[outside_mask] = 0

    def update_uncertainty_layer(self, uncertainty_map: np.ndarray):
        """
        Update uncertainty layer directly from an uncertainty map.

        Args:
            uncertainty_map: 2D array of uncertainty values (same size as costmap)
        """
        assert uncertainty_map.shape == (self.size_x, self.size_y)

        # Normalize and scale uncertainty to cost
        unc_normalized = np.clip(uncertainty_map / 5.0, 0, 1)  # 5m is max uncertainty
        self.uncertainty_layer = unc_normalized * 50  # Max uncertainty cost is 50

        self._costmap_dirty = True

    def update_shadow_layer(self, shadow_mask: np.ndarray):
        """
        Update shadow layer from a shadow detection mask.

        In shadow regions, the cameras get poor images and SLAM
        may fail. We want to avoid these areas if possible.

        Args:
            shadow_mask: Boolean array (True = shadow) or float [0, 1]
        """
        assert shadow_mask.shape == (self.size_x, self.size_y)

        if shadow_mask.dtype == bool:
            self.shadow_layer = shadow_mask.astype(np.float32) * 50
        else:
            self.shadow_layer = np.clip(shadow_mask, 0, 1) * 50

        self._costmap_dirty = True

    def update_obstacle_at(self, x: float, y: float, is_obstacle: bool):
        """
        Update a single cell as obstacle or free.

        Args:
            x, y: World coordinates
            is_obstacle: True if obstacle, False if free
        """
        i, j = self.world_to_cell(x, y)

        if self.is_in_bounds(i, j):
            if is_obstacle:
                self.obstacle_layer[i, j] = self.LETHAL_COST
            else:
                self.obstacle_layer[i, j] = self.FREE_COST

            self.unknown_mask[i, j] = False
            self._costmap_dirty = True

    def get_cost(self, x: float, y: float) -> float:
        """
        Get total cost at a world position.

        This is the main function used by the path planner.

        Args:
            x, y: World coordinates

        Returns:
            cost: Total cost at this position (0 = free, 255 = obstacle)
        """
        i, j = self.world_to_cell(x, y)
        return self.get_cost_at_cell(i, j)

    def get_cost_at_cell(self, i: int, j: int) -> float:
        """
        Get total cost at a cell.

        Combines all layers with their weights.

        Args:
            i, j: Cell indices

        Returns:
            cost: Total cost
        """
        if not self.is_in_bounds(i, j):
            return self.LETHAL_COST  # Out of bounds is lethal

        # Unknown cells have moderate cost
        if self.unknown_mask[i, j]:
            return self.UNKNOWN_COST

        # Obstacle is lethal (overrides everything)
        if self.obstacle_layer[i, j] >= self.LETHAL_COST:
            return self.LETHAL_COST

        # Combine layers with weights
        cost = (
            self.weights['obstacle'] * (self.obstacle_layer[i, j] / 255.0) +
            self.weights['inflation'] * (self.inflation_layer[i, j] / 255.0) +
            self.weights['uncertainty'] * (self.uncertainty_layer[i, j] / 50.0) +
            self.weights['shadow'] * (self.shadow_layer[i, j] / 50.0) +
            self.weights['slope'] * (self.slope_layer[i, j] / 100.0)
        )

        # Clamp to valid range
        return np.clip(cost, 0, self.LETHAL_COST)

    def get_combined_costmap(self) -> np.ndarray:
        """
        Get the combined costmap as a 2D array.

        This caches the result for efficiency.

        Returns:
            costmap: 2D array of total costs
        """
        if self._costmap_dirty or self._combined_costmap is None:
            self._combined_costmap = np.zeros((self.size_x, self.size_y), dtype=np.float32)

            for i in range(self.size_x):
                for j in range(self.size_y):
                    self._combined_costmap[i, j] = self.get_cost_at_cell(i, j)

            self._costmap_dirty = False

        return self._combined_costmap

    def is_collision_free(self, x: float, y: float) -> bool:
        """
        Check if a position is collision-free (safe to traverse).

        Args:
            x, y: World coordinates

        Returns:
            True if position is safe (not lethal or inscribed)
        """
        cost = self.get_cost(x, y)
        return cost < self.INSCRIBED_COST

    def is_path_collision_free(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        step_size: float = 0.1
    ) -> bool:
        """
        Check if a straight-line path is collision-free.

        Args:
            start: Start (x, y) in world coordinates
            end: End (x, y) in world coordinates
            step_size: Check interval in meters

        Returns:
            True if entire path is collision-free
        """
        start = np.array(start)
        end = np.array(end)

        direction = end - start
        distance = np.linalg.norm(direction)

        if distance < 0.001:
            return self.is_collision_free(start[0], start[1])

        direction = direction / distance
        n_steps = int(np.ceil(distance / step_size))

        for i in range(n_steps + 1):
            t = i * step_size
            if t > distance:
                t = distance

            point = start + t * direction

            if not self.is_collision_free(point[0], point[1]):
                return False

        return True

    def get_neighbors(self, i: int, j: int) -> list:
        """
        Get valid neighboring cells (8-connected).

        Used by the path planner.

        Args:
            i, j: Cell indices

        Returns:
            List of (ni, nj, cost_to_move) tuples
        """
        neighbors = []

        # 8-connected: cardinal and diagonal
        moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),  # Cardinal
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)  # Diagonal
        ]

        for di, dj, base_cost in moves:
            ni, nj = i + di, j + dj

            if self.is_in_bounds(ni, nj):
                cell_cost = self.get_cost_at_cell(ni, nj)

                if cell_cost < self.LETHAL_COST:
                    # Total cost to move = base movement cost + cell cost
                    move_cost = base_cost * self.resolution + cell_cost * 0.1
                    neighbors.append((ni, nj, move_cost))

        return neighbors

    def clear(self):
        """Reset all layers to default values."""
        self.obstacle_layer[:] = 0
        self.inflation_layer[:] = 0
        self.uncertainty_layer[:] = 0
        self.shadow_layer[:] = 0
        self.slope_layer[:] = 0
        self.unknown_mask[:] = True
        self._costmap_dirty = True

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the costmap."""
        combined = self.get_combined_costmap()

        free_cells = np.sum(combined < 10)
        occupied_cells = np.sum(combined >= self.LETHAL_COST)
        unknown_cells = np.sum(self.unknown_mask)

        stats = {
            'total_cells': self.size_x * self.size_y,
            'free_cells': free_cells,
            'occupied_cells': occupied_cells,
            'unknown_cells': unknown_cells,
            'mean_cost': np.mean(combined[combined < self.LETHAL_COST]),
            'max_cost': np.max(combined),
        }

        return stats


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_costmap(config: Dict[str, Any]) -> PerceptionAwareCostmap:
    """Factory function to create a PerceptionAwareCostmap."""
    return PerceptionAwareCostmap(config)


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    """Test the costmap module."""
    print("Testing PerceptionAwareCostmap...")

    config = {
        'resolution': 0.2,
        'size_x': 50,
        'size_y': 50,
        'origin': [-5.0, -5.0],
        'inflation_radius': 0.4
    }

    costmap = PerceptionAwareCostmap(config)

    # Add some obstacles
    print("\nAdding obstacles...")
    for _ in range(10):
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        costmap.update_obstacle_at(x, y, True)

    # Add some free space
    for i in range(20):
        for j in range(20):
            x, y = costmap.cell_to_world(i + 15, j + 15)
            costmap.update_obstacle_at(x, y, False)

    # Recompute inflation
    costmap._inflate_obstacles()

    # Get statistics
    stats = costmap.get_statistics()
    print(f"\nCostmap statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Test cost queries
    test_x, test_y = 0.0, 0.0
    print(f"\nCost at ({test_x}, {test_y}): {costmap.get_cost(test_x, test_y):.1f}")
    print(f"Collision-free: {costmap.is_collision_free(test_x, test_y)}")

    # Test path collision check
    print(f"\nPath from (-2, -2) to (2, 2) collision-free: {costmap.is_path_collision_free((-2, -2), (2, 2))}")

    print("\nTest complete!")
