"""
Uncertainty layer for costmap.

High uncertainty regions get high cost because:
- Map might be wrong
- Robot might collide with unmapped obstacles
"""

import numpy as np


class UncertaintyLayer:
    def __init__(self, config):
        self.max_uncertainty = config.get('max_uncertainty', 1.0)
    
    def compute(self, uncertainty_map):
        """
        Compute uncertainty costs.
        
        Args:
            uncertainty_map: 2D array of uncertainties
        
        Returns:
            uncertainty_cost: 2D array of costs [0, 1]
        """
        # TODO: Implement
        pass
