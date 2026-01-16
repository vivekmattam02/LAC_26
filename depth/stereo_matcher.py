"""
================================================================================
STEREO MATCHING MODULE FOR LUNAR NAVIGATION
================================================================================

This module computes dense depth maps from stereo image pairs. Depth information
is essential for building 3D maps of the environment and detecting obstacles.

THEORY - HOW STEREO DEPTH WORKS:
--------------------------------
Stereo vision mimics human binocular vision. Two cameras separated by a known
distance (baseline) capture the same scene from slightly different viewpoints.

Key Insight: Objects appear at different horizontal positions in the two images.
- Close objects: Large horizontal shift (disparity)
- Far objects: Small horizontal shift (disparity)

The depth formula is derived from similar triangles:

    depth = (baseline * focal_length) / disparity

Where:
    - baseline: Physical distance between cameras (in meters)
    - focal_length: Camera focal length (in pixels)
    - disparity: Horizontal pixel difference between left/right images

IMPLEMENTATION CHOICES:
-----------------------
1. SGBM (Semi-Global Block Matching): Classical algorithm, fast, works on CPU
   - Uses block matching to find correspondences
   - Semi-global optimization for smoothness
   - Good for real-time applications

2. FoundationStereo (Deep Learning): More accurate, needs GPU
   - Neural network trained on millions of stereo pairs
   - Better at handling textureless regions, reflections
   - Outputs confidence along with disparity

We implement both and allow switching based on available hardware.

FOR LAC SIMULATOR:
------------------
- We use FrontLeft + FrontRight cameras as stereo pair
- Baseline is approximately 0.162m (from Lunar.yaml configuration)
- Cameras provide grayscale images at up to 2448x2048 resolution

================================================================================
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
from enum import Enum


class StereoMethod(Enum):
    """Available stereo matching methods"""
    SGBM = "sgbm"                    # Classical Semi-Global Block Matching
    FOUNDATION = "foundation"        # Deep learning FoundationStereo


class StereoMatcher:
    """
    Computes dense depth maps from stereo image pairs.

    This class handles the entire stereo processing pipeline:
    1. Rectification (if needed) - align images so epipolar lines are horizontal
    2. Disparity computation - find pixel correspondences
    3. Depth conversion - convert disparity to metric depth
    4. Confidence estimation - how reliable is each depth measurement

    Attributes:
        baseline (float): Distance between cameras in meters
        fx (float): Focal length in pixels (horizontal)
        fy (float): Focal length in pixels (vertical)
        cx (float): Principal point x-coordinate
        cy (float): Principal point y-coordinate
        method (StereoMethod): Which algorithm to use

    Example Usage:
        >>> config = {'baseline': 0.162, 'fx': 458.0, 'method': 'sgbm'}
        >>> matcher = StereoMatcher(config)
        >>> depth, confidence = matcher.compute_depth(left_img, right_img)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the stereo matcher with camera parameters.

        Args:
            config: Dictionary containing:
                - baseline: Distance between cameras (meters), default 0.162
                - fx: Focal length x (pixels), default 458.0
                - fy: Focal length y (pixels), default 458.0
                - cx: Principal point x, default 320.0
                - cy: Principal point y, default 240.0
                - method: 'sgbm' or 'foundation', default 'sgbm'
                - max_disparity: Maximum disparity to search, default 128
                - min_disparity: Minimum disparity, default 0
                - block_size: SGBM block size (odd number), default 5
        """
        # =====================================================================
        # CAMERA INTRINSIC PARAMETERS
        # These define the camera's internal geometry
        # =====================================================================

        # Baseline: physical distance between left and right cameras
        # For LAC: 16.2 cm = 0.162 m (from Lunar.yaml)
        self.baseline = config.get('baseline', 0.162)

        # Focal length: how strongly the lens converges light
        # Measured in pixels (fx, fy). For ideal cameras, fx = fy
        # This affects the depth scale: larger fx = larger depth for same disparity
        self.fx = config.get('fx', 458.0)
        self.fy = config.get('fy', 458.0)

        # Principal point: where the optical axis hits the image sensor
        # Usually near image center. (cx, cy) in pixels
        self.cx = config.get('cx', 320.0)
        self.cy = config.get('cy', 240.0)

        # =====================================================================
        # STEREO MATCHING PARAMETERS
        # =====================================================================

        # Maximum disparity to search for
        # Larger = can detect closer objects but slower computation
        # Rule of thumb: max_disp = baseline * fx / min_depth
        # For 0.162m baseline, fx=458, min_depth=0.5m: max_disp = 148
        self.max_disparity = config.get('max_disparity', 128)
        self.min_disparity = config.get('min_disparity', 0)

        # Block size for matching (must be odd: 3, 5, 7, 9, 11...)
        # Larger = more robust to noise but less detail
        self.block_size = config.get('block_size', 5)

        # Which method to use
        method_str = config.get('method', 'sgbm')
        self.method = StereoMethod(method_str)

        # =====================================================================
        # INITIALIZE THE STEREO MATCHER
        # =====================================================================

        if self.method == StereoMethod.SGBM:
            self._init_sgbm()
        elif self.method == StereoMethod.FOUNDATION:
            self._init_foundation_stereo(config)

        # Depth range limits (for filtering outliers)
        self.min_depth = config.get('min_depth', 0.1)   # 10 cm minimum
        self.max_depth = config.get('max_depth', 50.0)  # 50 m maximum

        print(f"[StereoMatcher] Initialized with method={self.method.value}")
        print(f"[StereoMatcher] baseline={self.baseline}m, fx={self.fx}px")
        print(f"[StereoMatcher] Depth range: {self.min_depth}m - {self.max_depth}m")

    def _init_sgbm(self):
        """
        Initialize OpenCV's Semi-Global Block Matching stereo matcher.

        SGBM works by:
        1. For each pixel in left image, search along horizontal line in right image
        2. Find best matching block using Sum of Absolute Differences (SAD)
        3. Use semi-global optimization to enforce smoothness

        The "semi-global" part means it optimizes along multiple directions
        (8 paths) to get smoother disparity maps than simple block matching.
        """
        # Number of disparities must be divisible by 16
        num_disparities = ((self.max_disparity - self.min_disparity) // 16) * 16
        if num_disparities < 16:
            num_disparities = 16

        # P1, P2: Smoothness penalties
        # P1: penalty for disparity change of 1 between neighbors
        # P2: penalty for disparity change > 1 (should be > P1)
        # These are scaled by block_size^2 and number of channels
        P1 = 8 * 1 * self.block_size ** 2   # 1 channel (grayscale)
        P2 = 32 * 1 * self.block_size ** 2

        # Create the SGBM matcher
        self.stereo_left = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=num_disparities,
            blockSize=self.block_size,
            P1=P1,
            P2=P2,
            disp12MaxDiff=1,          # Max allowed difference in left-right check
            uniquenessRatio=10,        # Margin for best match vs second best
            speckleWindowSize=100,     # Max size of smooth disparity regions
            speckleRange=32,           # Max disparity variation within region
            preFilterCap=63,           # Truncation value for prefilter
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # Faster 3-way mode
        )

        # Create right matcher for left-right consistency check
        self.stereo_right = cv2.ximgproc.createRightMatcher(self.stereo_left)

        # WLS filter for smoothing disparity and filling holes
        # Lambda: regularization, higher = smoother
        # Sigma: edge-preserving parameter
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(
            matcher_left=self.stereo_left
        )
        self.wls_filter.setLambda(8000)
        self.wls_filter.setSigmaColor(1.5)

        print(f"[StereoMatcher] SGBM initialized: numDisparities={num_disparities}")

    def _init_foundation_stereo(self, config: Dict[str, Any]):
        """
        Initialize FoundationStereo deep learning model.

        FoundationStereo is a neural network that learns stereo matching from data.
        It consists of:
        1. Feature extraction: CNN extracts features from both images
        2. Cost volume: Build 4D volume of matching costs
        3. Cost aggregation: 3D CNN processes the volume
        4. Disparity regression: Soft-argmin to get sub-pixel disparity

        Advantages over SGBM:
        - Better in textureless regions (learns context)
        - Handles reflections and transparency
        - More accurate overall

        Disadvantages:
        - Requires GPU
        - Slower than SGBM
        - May not generalize to very different domains
        """
        try:
            import torch

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"[StereoMatcher] Using device: {self.device}")

            # Path to pretrained model weights
            model_path = config.get('foundation_model_path', None)

            if model_path is not None:
                # Load the model (architecture depends on FoundationStereo version)
                # This is a placeholder - actual loading depends on the model
                self.foundation_model = self._load_foundation_model(model_path)
                self.foundation_model.to(self.device)
                self.foundation_model.eval()
                print(f"[StereoMatcher] FoundationStereo loaded from {model_path}")
            else:
                print("[StereoMatcher] WARNING: No model path provided, falling back to SGBM")
                self.method = StereoMethod.SGBM
                self._init_sgbm()

        except ImportError:
            print("[StereoMatcher] PyTorch not available, falling back to SGBM")
            self.method = StereoMethod.SGBM
            self._init_sgbm()

    def _load_foundation_model(self, model_path: str):
        """
        Load FoundationStereo model from checkpoint.

        This is a placeholder - the actual implementation depends on
        which FoundationStereo version/architecture you're using.
        """
        import torch
        import torch.nn as nn

        # Placeholder: In reality, you would load the actual model architecture
        # Example: from foundation_stereo import FoundationStereo
        #          model = FoundationStereo()
        #          model.load_state_dict(torch.load(model_path))

        print(f"[StereoMatcher] Loading model from {model_path}")
        # For now, we'll just return None and fall back to SGBM
        return None

    def compute_depth(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute depth map from stereo image pair.

        This is the main function you call to get depth from stereo images.

        Args:
            img_left: Left camera image, shape (H, W) for grayscale or (H, W, 3) for color
            img_right: Right camera image, same shape as left

        Returns:
            depth: Depth map in meters, shape (H, W). Invalid pixels are set to 0.
            confidence: Confidence map [0, 1], shape (H, W). Higher = more reliable.

        Example:
            >>> depth, conf = matcher.compute_depth(left_img, right_img)
            >>> # Get depth at pixel (100, 200)
            >>> d = depth[100, 200]  # in meters
            >>> c = conf[100, 200]   # confidence 0-1
        """
        # Ensure images are grayscale for SGBM
        if len(img_left.shape) == 3:
            img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        else:
            img_left_gray = img_left

        if len(img_right.shape) == 3:
            img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            img_right_gray = img_right

        # Compute disparity based on method
        if self.method == StereoMethod.SGBM:
            disparity, confidence = self._compute_disparity_sgbm(
                img_left_gray, img_right_gray
            )
        else:
            disparity, confidence = self._compute_disparity_foundation(
                img_left, img_right
            )

        # Convert disparity to depth
        depth = self.disparity_to_depth(disparity)

        # Apply depth range filter
        invalid_mask = (depth < self.min_depth) | (depth > self.max_depth) | (disparity <= 0)
        depth[invalid_mask] = 0
        confidence[invalid_mask] = 0

        return depth, confidence

    def _compute_disparity_sgbm(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute disparity using Semi-Global Block Matching.

        The pipeline:
        1. Compute left disparity (left image as reference)
        2. Compute right disparity (right image as reference)
        3. Use WLS filter to combine them and fill holes
        4. Estimate confidence from left-right consistency

        Args:
            img_left: Grayscale left image
            img_right: Grayscale right image

        Returns:
            disparity: Disparity map (float, in pixels)
            confidence: Confidence map [0, 1]
        """
        # Compute disparity from left and right perspectives
        # SGBM returns disparity as fixed-point (multiply by 1/16 to get float)
        disp_left = self.stereo_left.compute(img_left, img_right)
        disp_right = self.stereo_right.compute(img_right, img_left)

        # Apply WLS filter to smooth and fill holes
        # This uses both left and right disparities for consistency
        disp_filtered = self.wls_filter.filter(
            disparity_map_left=disp_left,
            left_view=img_left,
            disparity_map_right=disp_right
        )

        # Convert to float disparity
        disparity = disp_filtered.astype(np.float32) / 16.0

        # Compute confidence based on:
        # 1. Left-right consistency
        # 2. Disparity validity
        # 3. WLS confidence
        confidence = self._compute_sgbm_confidence(
            disp_left, disp_right, disp_filtered
        )

        return disparity, confidence

    def _compute_sgbm_confidence(
        self,
        disp_left: np.ndarray,
        disp_right: np.ndarray,
        disp_filtered: np.ndarray
    ) -> np.ndarray:
        """
        Estimate confidence of SGBM disparity estimates.

        We use multiple cues:
        1. Left-Right Consistency: If left and right disparities agree, it's reliable
        2. Valid Range: Disparity should be positive and within range
        3. Local Variance: Flat regions (low variance) might be unreliable

        Args:
            disp_left: Raw left disparity (fixed-point)
            disp_right: Raw right disparity (fixed-point)
            disp_filtered: WLS-filtered disparity (fixed-point)

        Returns:
            confidence: Confidence map [0, 1]
        """
        h, w = disp_left.shape
        confidence = np.ones((h, w), dtype=np.float32)

        # Convert to float
        dl = disp_left.astype(np.float32) / 16.0
        dr = disp_right.astype(np.float32) / 16.0
        df = disp_filtered.astype(np.float32) / 16.0

        # 1. Left-Right Consistency Check
        # For each pixel in left image, check if corresponding pixel in right agrees
        # Create coordinate arrays
        x_coords = np.arange(w)
        x_grid = np.tile(x_coords, (h, 1))

        # Where in right image does left pixel map to?
        x_right = (x_grid - dl).astype(np.int32)

        # Clamp to valid range
        x_right = np.clip(x_right, 0, w - 1)

        # Get right disparity at those locations
        y_grid = np.tile(np.arange(h).reshape(-1, 1), (1, w))
        dr_at_left = dr[y_grid, x_right]

        # Compute consistency error (should be close to 0 if consistent)
        lr_diff = np.abs(dl - dr_at_left)

        # Convert to confidence (larger diff = lower confidence)
        lr_confidence = np.exp(-lr_diff / 2.0)  # Exponential decay

        # 2. Valid disparity range
        valid_mask = (df > 0) & (df < self.max_disparity)

        # 3. Combine confidences
        confidence = lr_confidence * valid_mask.astype(np.float32)

        # Clamp to [0, 1]
        confidence = np.clip(confidence, 0, 1)

        return confidence

    def _compute_disparity_foundation(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute disparity using FoundationStereo deep learning model.

        The pipeline:
        1. Preprocess images (normalize, resize if needed)
        2. Run through neural network
        3. Post-process output

        FoundationStereo typically outputs both disparity and confidence.

        Args:
            img_left: Left image (can be color or grayscale)
            img_right: Right image

        Returns:
            disparity: Disparity map (float, in pixels)
            confidence: Confidence map [0, 1]
        """
        import torch

        # Preprocess: normalize to [0, 1] and convert to tensor
        left_tensor = self._preprocess_for_network(img_left)
        right_tensor = self._preprocess_for_network(img_right)

        # Run inference
        with torch.no_grad():
            # FoundationStereo forward pass
            # Output format depends on model version
            output = self.foundation_model(left_tensor, right_tensor)

            if isinstance(output, tuple):
                disp_tensor, conf_tensor = output
            else:
                disp_tensor = output
                conf_tensor = None

        # Convert to numpy
        disparity = disp_tensor.squeeze().cpu().numpy()

        if conf_tensor is not None:
            confidence = conf_tensor.squeeze().cpu().numpy()
        else:
            # Estimate confidence from disparity gradient
            confidence = self._estimate_confidence_from_disparity(disparity)

        return disparity, confidence

    def _preprocess_for_network(self, img: np.ndarray) -> 'torch.Tensor':
        """
        Preprocess image for neural network input.

        Standard preprocessing:
        1. Convert to RGB if grayscale
        2. Normalize to [0, 1] or [-1, 1]
        3. Convert to tensor with shape (1, C, H, W)
        """
        import torch

        # Ensure 3 channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to tensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        # Move to device
        tensor = tensor.to(self.device)

        return tensor

    def _estimate_confidence_from_disparity(
        self,
        disparity: np.ndarray
    ) -> np.ndarray:
        """
        Estimate confidence when model doesn't provide it.

        Heuristics:
        1. Valid disparity range
        2. Smoothness (low gradient = more confident)
        3. Edge regions (high gradient = less confident)
        """
        # Valid range
        valid = (disparity > 0) & (disparity < self.max_disparity)

        # Compute gradient magnitude
        grad_x = cv2.Sobel(disparity, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(disparity, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Convert gradient to confidence (high gradient = low confidence)
        # Normalize gradient
        grad_normalized = grad_mag / (np.max(grad_mag) + 1e-6)
        smoothness_conf = 1.0 - grad_normalized

        # Combine
        confidence = valid.astype(np.float32) * smoothness_conf

        return confidence

    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity map to depth map using stereo geometry.

        THE STEREO DEPTH EQUATION:
        --------------------------
        From similar triangles in the stereo geometry:

            depth = (baseline * focal_length) / disparity

        Where:
            - depth: Distance to the point (meters)
            - baseline: Distance between cameras (meters)
            - focal_length: Camera focal length (pixels)
            - disparity: Horizontal shift between images (pixels)

        IMPORTANT:
            - When disparity = 0, depth = infinity (point at infinity)
            - Small disparity = far object
            - Large disparity = close object
            - We handle disparity <= 0 by setting depth to 0 (invalid)

        Args:
            disparity: Disparity map (pixels), shape (H, W)

        Returns:
            depth: Depth map (meters), shape (H, W)
        """
        # Avoid division by zero
        safe_disparity = np.where(disparity > 0, disparity, 1e-6)

        # Apply stereo formula
        depth = (self.baseline * self.fx) / safe_disparity

        # Mark invalid disparities as 0 depth
        depth = np.where(disparity > 0, depth, 0)

        return depth

    def depth_to_disparity(self, depth: np.ndarray) -> np.ndarray:
        """
        Convert depth map to disparity map (inverse of disparity_to_depth).

        Args:
            depth: Depth map (meters), shape (H, W)

        Returns:
            disparity: Disparity map (pixels), shape (H, W)
        """
        # Avoid division by zero
        safe_depth = np.where(depth > 0, depth, 1e-6)

        # Inverse of depth formula
        disparity = (self.baseline * self.fx) / safe_depth

        # Mark invalid depths as 0 disparity
        disparity = np.where(depth > 0, disparity, 0)

        return disparity

    def depth_to_pointcloud(
        self,
        depth: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.3
    ) -> np.ndarray:
        """
        Convert depth map to 3D point cloud in camera frame.

        COORDINATE SYSTEM:
        ------------------
        Camera frame (standard):
            - X: right
            - Y: down
            - Z: forward (into the scene)

        For each pixel (u, v) with depth d:
            X = (u - cx) * d / fx
            Y = (v - cy) * d / fy
            Z = d

        Args:
            depth: Depth map (H, W) in meters
            confidence: Optional confidence map (H, W) for filtering
            confidence_threshold: Minimum confidence to include point

        Returns:
            points: Point cloud (N, 3) where N is number of valid points
                    Each row is [X, Y, Z] in camera frame
        """
        h, w = depth.shape

        # Create pixel coordinate grids
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)

        # Compute 3D coordinates
        z = depth
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        # Stack into (H, W, 3)
        points_image = np.stack([x, y, z], axis=-1)

        # Create valid mask
        valid = depth > 0
        if confidence is not None:
            valid = valid & (confidence >= confidence_threshold)

        # Extract valid points
        points = points_image[valid]

        return points


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_stereo_matcher(config: Dict[str, Any]) -> StereoMatcher:
    """
    Factory function to create a StereoMatcher with the given configuration.

    Args:
        config: Configuration dictionary

    Returns:
        StereoMatcher instance
    """
    return StereoMatcher(config)


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    """
    Quick test of the stereo matcher.

    This creates synthetic stereo images and computes depth.
    """
    print("Testing StereoMatcher...")

    # Create test configuration
    config = {
        'baseline': 0.162,
        'fx': 458.0,
        'fy': 458.0,
        'cx': 320.0,
        'cy': 240.0,
        'method': 'sgbm',
        'max_disparity': 128,
        'min_depth': 0.5,
        'max_depth': 20.0
    }

    # Create matcher
    matcher = StereoMatcher(config)

    # Create synthetic test images (simple gradient)
    h, w = 480, 640
    img_left = np.random.randint(0, 255, (h, w), dtype=np.uint8)

    # Shift right image to simulate disparity of 20 pixels (depth = 0.162 * 458 / 20 = 3.7m)
    shift = 20
    img_right = np.roll(img_left, shift, axis=1)

    # Compute depth
    depth, confidence = matcher.compute_depth(img_left, img_right)

    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: {depth[depth > 0].min():.2f}m - {depth[depth > 0].max():.2f}m")
    print(f"Confidence range: {confidence.min():.2f} - {confidence.max():.2f}")

    # Test point cloud generation
    points = matcher.depth_to_pointcloud(depth, confidence, confidence_threshold=0.1)
    print(f"Point cloud: {points.shape[0]} points")

    print("Test complete!")
