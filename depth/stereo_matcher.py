"""
Stereo matching for depth computation.

Computes dense depth maps from stereo image pairs
using Semi-Global Block Matching (SGBM).
"""

import numpy as np
import cv2


class StereoMatcher:
    def __init__(self, config):
        self.baseline = config.get('baseline', 0.162)
        self.fx = config.get('fx', 458.0)
        self.max_disp = config.get('max_disparity', 128)
        
        # TODO: Initialize SGBM matcher
        self.stereo = None
    
    def compute_depth(self, img_left, img_right):
        """
        Compute depth from stereo pair.
        
        Args:
            img_left: Left image (H, W, 3)
            img_right: Right image (H, W, 3)
        
        Returns:
            depth: Depth map in meters (H, W)
            disparity: Raw disparity map (H, W)
        """
        # TODO: Implement
        pass
    
    def disparity_to_depth(self, disparity):
        """Convert disparity to depth using: depth = baseline * fx / disparity"""
        # TODO: Implement
        pass
