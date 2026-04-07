"""
Coverage planner for efficient terrain mapping.

Generates waypoints to maximize:
- Area coverage
- Mapping productivity (LAC scoring)
- Rock detection opportunities
"""

import numpy as np


class CoveragePlanner:
    def __init__(self, config):
        self.coverage_radius = config.get('coverage_radius', 5.0)
        self.min_frontier_size = config.get('min_frontier_size', 3)

    def get_next_goal(self, current_pose, voxel_grid, costmap):
        """
        Get next exploration goal.

        Args:
            current_pose: Current robot pose (x, y, theta) or (x, y)
            voxel_grid: Current map
            costmap: Perception-aware costmap

        Returns:
            goal: (x, y) position of next goal or None
        """
        frontiers = self.find_frontiers(voxel_grid)
        if not frontiers:
            return None

        scored = [
            (self.score_frontier(frontier, current_pose, costmap), frontier)
            for frontier in frontiers
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        best_frontier = scored[0][1]
        centroid = np.mean(np.array(best_frontier), axis=0)
        return float(centroid[0]), float(centroid[1])

    def find_frontiers(self, voxel_grid):
        """Find frontier cells (known cells adjacent to unknown cells)."""
        _, _, valid = voxel_grid.get_height_map()
        known = valid.astype(bool)
        unknown = ~known

        nx, ny = known.shape
        frontier_mask = np.zeros_like(known, dtype=bool)

        for i in range(nx):
            for j in range(ny):
                if not known[i, j]:
                    continue
                i0, i1 = max(0, i - 1), min(nx, i + 2)
                j0, j1 = max(0, j - 1), min(ny, j + 2)
                if np.any(unknown[i0:i1, j0:j1]):
                    frontier_mask[i, j] = True

        # Connected components in frontier mask (8-connected)
        visited = np.zeros_like(frontier_mask, dtype=bool)
        frontiers = []

        for i in range(nx):
            for j in range(ny):
                if not frontier_mask[i, j] or visited[i, j]:
                    continue

                stack = [(i, j)]
                component = []
                visited[i, j] = True

                while stack:
                    ci, cj = stack.pop()
                    x, y, _ = voxel_grid.voxel_to_world(np.array([[ci, cj, 0]]))[0]
                    component.append((float(x), float(y)))

                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = ci + di, cj + dj
                            if 0 <= ni < nx and 0 <= nj < ny:
                                if frontier_mask[ni, nj] and not visited[ni, nj]:
                                    visited[ni, nj] = True
                                    stack.append((ni, nj))

                if len(component) >= self.min_frontier_size:
                    frontiers.append(component)

        return frontiers

    def score_frontier(self, frontier, current_pose, costmap):
        """Score frontier using size, distance, and local traversal cost."""
        if len(frontier) == 0:
            return -np.inf

        if len(current_pose) >= 2:
            cx, cy = current_pose[0], current_pose[1]
        else:
            cx, cy = current_pose

        frontier_np = np.array(frontier)
        centroid = np.mean(frontier_np, axis=0)

        # Prefer larger frontier and closer frontier
        size_score = float(len(frontier))
        distance = np.linalg.norm(centroid - np.array([cx, cy]))
        distance_score = -distance

        # Penalize high-cost frontier neighborhoods
        local_costs = []
        for x, y in frontier[:: max(1, len(frontier) // 8)]:
            local_costs.append(costmap.get_cost(x, y))
        mean_cost = float(np.mean(local_costs)) if local_costs else 0.0
        cost_penalty = -0.1 * mean_cost

        return size_score + distance_score + cost_penalty
