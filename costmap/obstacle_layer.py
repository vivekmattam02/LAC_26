"""
Obstacle layer for costmap.

Detects obstacles from:
- Height gradients (steep terrain)
- Unknown regions
"""

import numpy as np


class ObstacleLayer:
    def __init__(self, config):
        self.gradient_threshold = config.get('gradient_threshold', 0.3)
        self.unknown_cost = config.get('unknown_cost', 0.8)
    
    def compute(self, height_map, valid_mask):
        """
        Compute obstacle costs from height map.
        
        Args:
            height_map: 2D array of heights
            valid_mask: Boolean mask of observed cells
        
        Returns:
            obstacle_cost: 2D array of costs [0, 1]
        """
        # TODO: Implement
        pass
