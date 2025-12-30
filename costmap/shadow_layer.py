"""
Shadow layer for costmap.

Penalizes shadow regions because:
- SLAM loses tracking in dark areas
- No features to track
- Lunar shadows are extremely dark (no atmosphere)
"""

import numpy as np
import cv2


class ShadowLayer:
    def __init__(self, config):
        self.shadow_threshold = config.get('shadow_threshold', 30)  # pixel intensity
    
    def detect_shadows(self, image):
        """
        Detect shadow regions in image.
        
        Args:
            image: RGB or grayscale image
        
        Returns:
            shadow_mask: Boolean mask (True = shadow)
        """
        # TODO: Implement
        pass
    
    def compute(self, shadow_mask):
        """
        Compute shadow costs.
        
        Args:
            shadow_mask: Boolean mask of shadow regions
        
        Returns:
            shadow_cost: 2D array of costs [0, 1]
        """
        # TODO: Implement
        pass
