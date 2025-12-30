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
            current_pose: Current robot pose
            voxel_grid: Current map
            costmap: Perception-aware costmap
        
        Returns:
            goal: (x, y) position of next goal
        """
        # TODO: Implement
        pass
    
    def find_frontiers(self, voxel_grid):
        """Find frontier cells (boundary between known and unknown)"""
        # TODO: Implement
        pass
    
    def score_frontier(self, frontier, current_pose, costmap):
        """Score a frontier based on distance, size, and path cost"""
        # TODO: Implement
        pass
