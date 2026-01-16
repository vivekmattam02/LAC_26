"""
================================================================================
A* PATH PLANNER WITH PERCEPTION-AWARE COSTS
================================================================================

A* is a graph search algorithm that finds the optimal path from start to goal.
It's one of the most widely used path planning algorithms in robotics because:
1. It's guaranteed to find the optimal path (if one exists)
2. It's efficient due to the use of heuristics
3. It's easy to understand and implement

HOW A* WORKS:
-------------
A* maintains two values for each node:
- g(n): Cost to reach node n from the start
- h(n): Heuristic estimate of cost from n to goal
- f(n) = g(n) + h(n): Total estimated cost through n

The algorithm:
1. Start with the start node in the "open set"
2. Pick the node with lowest f(n) from open set
3. If it's the goal, we're done!
4. Otherwise, expand it (add neighbors to open set)
5. Repeat until goal found or open set empty

The heuristic h(n) must be "admissible" (never overestimate actual cost).
Common choices:
- Euclidean distance: sqrt((x2-x1)² + (y2-y1)²)
- Manhattan distance: |x2-x1| + |y2-y1|

WHY A* FOR LUNAR NAVIGATION:
----------------------------
The perception-aware costmap encodes not just obstacles, but also:
- Uncertainty (prefer well-mapped areas)
- Shadows (avoid where SLAM fails)
- Slopes (avoid steep terrain)

A* naturally handles these because it minimizes total cost, not just distance.
Paths will prefer safe, well-observed routes over risky shortcuts.

================================================================================
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass(order=True)
class Node:
    """
    A node in the A* search tree.

    Attributes:
        f_cost: Total estimated cost (g + h), used for priority queue ordering
        g_cost: Cost from start to this node
        position: (i, j) cell indices
        parent: Parent node for path reconstruction
    """
    f_cost: float
    g_cost: float = field(compare=False)
    position: Tuple[int, int] = field(compare=False)
    parent: Optional['Node'] = field(compare=False, default=None)


class AStarPlanner:
    """
    A* path planner with perception-aware cost integration.

    This planner finds optimal paths on the costmap grid, considering
    obstacles, uncertainty, shadows, and terrain difficulty.

    Attributes:
        costmap: PerceptionAwareCostmap instance
        config: Configuration dictionary

    Example Usage:
        >>> planner = AStarPlanner(costmap, config)
        >>> path, cost = planner.plan((0, 0), (5, 5))
        >>> if path:
        ...     print(f"Found path with {len(path)} waypoints, cost {cost}")
    """

    def __init__(self, costmap, config: Dict[str, Any] = None):
        """
        Initialize the A* planner.

        Args:
            costmap: PerceptionAwareCostmap instance for cost queries
            config: Configuration dictionary containing:
                - heuristic_weight: Weight for heuristic (default 1.0)
                - max_iterations: Maximum search iterations (default 100000)
                - allow_diagonal: Allow diagonal movement (default True)
        """
        self.costmap = costmap
        self.config = config or {}

        # =====================================================================
        # A* PARAMETERS
        # =====================================================================

        # Heuristic weight: >1 makes search faster but less optimal
        # This is known as "Weighted A*" or "WA*"
        self.heuristic_weight = self.config.get('heuristic_weight', 1.0)

        # Maximum iterations to prevent infinite loops
        self.max_iterations = self.config.get('max_iterations', 100000)

        # Allow diagonal movement (8-connected) or only cardinal (4-connected)
        self.allow_diagonal = self.config.get('allow_diagonal', True)

        # =====================================================================
        # MOVEMENT DEFINITIONS
        # =====================================================================

        if self.allow_diagonal:
            # 8-connected: cardinal + diagonal movements
            self.movements = [
                (1, 0), (-1, 0), (0, 1), (0, -1),        # Cardinal (N, S, E, W)
                (1, 1), (1, -1), (-1, 1), (-1, -1)       # Diagonal
            ]
            # Cost for each movement (diagonal costs sqrt(2) more)
            self.move_costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
        else:
            # 4-connected: only cardinal movements
            self.movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            self.move_costs = [1.0, 1.0, 1.0, 1.0]

        print(f"[AStarPlanner] Initialized with heuristic_weight={self.heuristic_weight}")
        print(f"[AStarPlanner] Movement: {'8-connected' if self.allow_diagonal else '4-connected'}")

    def plan(
        self,
        start_world: Tuple[float, float],
        goal_world: Tuple[float, float]
    ) -> Tuple[Optional[List[Tuple[float, float]]], float]:
        """
        Plan a path from start to goal in world coordinates.

        This is the main function you call to get a path.

        Args:
            start_world: Start position (x, y) in world coordinates (meters)
            goal_world: Goal position (x, y) in world coordinates (meters)

        Returns:
            path: List of (x, y) world coordinates representing the path,
                  or None if no path found
            cost: Total path cost (0 if no path found)

        Example:
            >>> path, cost = planner.plan((0, 0), (5, 3))
            >>> for waypoint in path:
            ...     robot.go_to(waypoint)
        """
        # Convert world coordinates to grid cells
        start_cell = self.costmap.world_to_cell(start_world[0], start_world[1])
        goal_cell = self.costmap.world_to_cell(goal_world[0], goal_world[1])

        # Validate start and goal
        if not self.costmap.is_in_bounds(start_cell[0], start_cell[1]):
            print(f"[AStarPlanner] Start {start_world} is out of bounds")
            return None, 0.0

        if not self.costmap.is_in_bounds(goal_cell[0], goal_cell[1]):
            print(f"[AStarPlanner] Goal {goal_world} is out of bounds")
            return None, 0.0

        # Check if start is in collision
        if self.costmap.get_cost_at_cell(start_cell[0], start_cell[1]) >= self.costmap.LETHAL_COST:
            print(f"[AStarPlanner] Start {start_world} is in obstacle")
            return None, 0.0

        # Check if goal is in collision
        if self.costmap.get_cost_at_cell(goal_cell[0], goal_cell[1]) >= self.costmap.LETHAL_COST:
            print(f"[AStarPlanner] Goal {goal_world} is in obstacle")
            return None, 0.0

        # Run A* search
        path_cells, cost = self._astar(start_cell, goal_cell)

        if path_cells is None:
            print(f"[AStarPlanner] No path found from {start_world} to {goal_world}")
            return None, 0.0

        # Convert path from cells to world coordinates
        path_world = []
        for cell in path_cells:
            world = self.costmap.cell_to_world(cell[0], cell[1])
            path_world.append(world)

        print(f"[AStarPlanner] Found path with {len(path_world)} waypoints, cost={cost:.2f}")

        return path_world, cost

    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Tuple[Optional[List[Tuple[int, int]]], float]:
        """
        Core A* search algorithm.

        Args:
            start: Start cell (i, j)
            goal: Goal cell (i, j)

        Returns:
            path: List of cells from start to goal, or None
            cost: Total path cost
        """
        # =====================================================================
        # DATA STRUCTURES
        # =====================================================================

        # Open set: nodes to explore (priority queue sorted by f_cost)
        # Using heapq which is a min-heap
        open_set = []

        # Closed set: nodes already explored
        # Use a set for O(1) lookup
        closed_set = set()

        # g_costs: best known g(n) for each node
        # Default to infinity
        g_costs = {}
        g_costs[start] = 0.0

        # Create start node
        h_start = self._heuristic(start, goal)
        start_node = Node(
            f_cost=h_start,
            g_cost=0.0,
            position=start,
            parent=None
        )
        heapq.heappush(open_set, start_node)

        # =====================================================================
        # MAIN A* LOOP
        # =====================================================================

        iterations = 0

        while open_set and iterations < self.max_iterations:
            iterations += 1

            # Pop node with lowest f_cost
            current = heapq.heappop(open_set)

            # Check if we've reached the goal
            if current.position == goal:
                path = self._reconstruct_path(current)
                return path, current.g_cost

            # Skip if already in closed set (we might have added duplicates)
            if current.position in closed_set:
                continue

            # Add to closed set
            closed_set.add(current.position)

            # Expand neighbors
            for k, (di, dj) in enumerate(self.movements):
                ni, nj = current.position[0] + di, current.position[1] + dj
                neighbor_pos = (ni, nj)

                # Skip if already explored
                if neighbor_pos in closed_set:
                    continue

                # Skip if out of bounds
                if not self.costmap.is_in_bounds(ni, nj):
                    continue

                # Get cell cost
                cell_cost = self.costmap.get_cost_at_cell(ni, nj)

                # Skip if obstacle (lethal)
                if cell_cost >= self.costmap.LETHAL_COST:
                    continue

                # Compute g(n) = g(current) + cost to move to neighbor
                # Movement cost * resolution + cell traversal cost
                move_cost = self.move_costs[k] * self.costmap.resolution
                edge_cost = move_cost + cell_cost * 0.01  # Scale cell cost

                tentative_g = current.g_cost + edge_cost

                # Skip if we already have a better path to this node
                if neighbor_pos in g_costs and tentative_g >= g_costs[neighbor_pos]:
                    continue

                # This is a better path
                g_costs[neighbor_pos] = tentative_g

                # Compute f = g + h
                h = self._heuristic(neighbor_pos, goal)
                f = tentative_g + self.heuristic_weight * h

                # Create neighbor node and add to open set
                neighbor_node = Node(
                    f_cost=f,
                    g_cost=tentative_g,
                    position=neighbor_pos,
                    parent=current
                )
                heapq.heappush(open_set, neighbor_node)

        # No path found
        if iterations >= self.max_iterations:
            print(f"[AStarPlanner] Reached max iterations ({self.max_iterations})")

        return None, 0.0

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Compute heuristic estimate of cost from pos to goal.

        Uses Euclidean distance multiplied by resolution to get
        distance in meters. This is admissible because the shortest
        path can never be less than the straight-line distance.

        Args:
            pos: Current position (i, j)
            goal: Goal position (i, j)

        Returns:
            h: Estimated cost to goal
        """
        # Euclidean distance in cells, scaled by resolution
        di = pos[0] - goal[0]
        dj = pos[1] - goal[1]
        distance_cells = np.sqrt(di**2 + dj**2)

        return distance_cells * self.costmap.resolution

    def _reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from goal to start by following parent pointers.

        Args:
            node: Goal node with parent chain

        Returns:
            path: List of (i, j) cells from start to goal
        """
        path = []
        current = node

        while current is not None:
            path.append(current.position)
            current = current.parent

        # Reverse to get start -> goal order
        path.reverse()

        return path

    def smooth_path(
        self,
        path: List[Tuple[float, float]],
        max_iterations: int = 100
    ) -> List[Tuple[float, float]]:
        """
        Smooth the path by removing unnecessary waypoints.

        Uses line-of-sight checks: if we can go directly from A to C,
        we don't need waypoint B in between.

        Args:
            path: Original path as list of (x, y) world coordinates
            max_iterations: Maximum smoothing iterations

        Returns:
            smoothed_path: Smoothed path with fewer waypoints
        """
        if path is None or len(path) <= 2:
            return path

        smoothed = list(path)  # Copy

        for _ in range(max_iterations):
            modified = False

            i = 0
            while i < len(smoothed) - 2:
                # Check if we can skip the middle point
                if self.costmap.is_path_collision_free(smoothed[i], smoothed[i + 2]):
                    # Remove the middle point
                    smoothed.pop(i + 1)
                    modified = True
                else:
                    i += 1

            if not modified:
                break

        return smoothed


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_astar_planner(costmap, config: Dict[str, Any] = None) -> AStarPlanner:
    """Factory function to create an AStarPlanner."""
    return AStarPlanner(costmap, config)


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    """Test the A* planner module."""
    print("Testing AStarPlanner...")

    # Need to create a costmap first
    from costmap import PerceptionAwareCostmap

    costmap_config = {
        'resolution': 0.2,
        'size_x': 50,
        'size_y': 50,
        'origin': [-5.0, -5.0]
    }
    costmap = PerceptionAwareCostmap(costmap_config)

    # Add some obstacles
    print("\nSetting up environment...")
    obstacles = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]  # L-shaped obstacle
    for ox, oy in obstacles:
        costmap.update_obstacle_at(ox, oy, True)

    # Mark rest as free
    for i in range(costmap.size_x):
        for j in range(costmap.size_y):
            x, y = costmap.cell_to_world(i, j)
            if (x, y) not in [(ox, oy) for ox, oy in obstacles]:
                costmap.update_obstacle_at(x, y, False)

    costmap._inflate_obstacles()

    # Create planner
    planner_config = {
        'heuristic_weight': 1.0,
        'allow_diagonal': True
    }
    planner = AStarPlanner(costmap, planner_config)

    # Test planning
    print("\nPlanning path from (-3, -3) to (3, 3)...")
    start = (-3.0, -3.0)
    goal = (3.0, 3.0)

    path, cost = planner.plan(start, goal)

    if path:
        print(f"\nPath found!")
        print(f"  Waypoints: {len(path)}")
        print(f"  Total cost: {cost:.2f}")
        print(f"  First 5 waypoints: {path[:5]}")

        # Smooth the path
        smoothed = planner.smooth_path(path)
        print(f"\nSmoothed path:")
        print(f"  Waypoints: {len(smoothed)} (reduced from {len(path)})")
    else:
        print("\nNo path found!")

    # Test with blocked goal
    print("\nTesting with blocked goal (0, 0)...")
    path2, cost2 = planner.plan(start, (0, 0))
    if path2 is None:
        print("  Correctly detected blocked goal")

    print("\nTest complete!")
