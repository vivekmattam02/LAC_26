"""
================================================================================
DEPTH UNCERTAINTY ESTIMATION MODULE
================================================================================

This module estimates and propagates uncertainty in depth measurements.
Understanding uncertainty is CRITICAL for robust autonomous navigation because:

1. Not all depth measurements are equally reliable
2. Planning should prefer paths through well-observed regions
3. The robot should slow down or stop when uncertain about obstacles

WHY DEPTH UNCERTAINTY MATTERS:
------------------------------
Stereo depth has inherent limitations:
- Textureless regions (like smooth lunar surfaces): Can't find matches
- Repetitive patterns: Ambiguous matches
- Occlusions: One camera sees something the other doesn't
- Low light / shadows: Poor image quality
- Far objects: Small disparity = large depth uncertainty

The uncertainty in depth follows a specific pattern based on the stereo geometry:

    sigma_depth = (depth^2 * sigma_disparity) / (baseline * focal_length)

Key insight: Depth uncertainty grows QUADRATICALLY with distance!
At 2x distance, uncertainty is 4x larger.

This module provides:
1. Analytical depth uncertainty based on stereo geometry
2. Data-driven uncertainty from stereo matching confidence
3. Combined uncertainty estimation
4. Uncertainty propagation to downstream modules

================================================================================
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import cv2


class DepthUncertainty:
    """
    Estimates uncertainty in depth measurements from stereo vision.

    This class computes how confident we should be in each depth measurement.
    Higher uncertainty = less reliable depth = should be treated with caution.

    The uncertainty model combines:
    1. Geometric uncertainty: Based on stereo triangulation math
    2. Matching uncertainty: From stereo matching confidence
    3. Environmental factors: Texture, lighting, occlusions

    Attributes:
        baseline (float): Stereo baseline in meters
        fx (float): Focal length in pixels
        sigma_disp (float): Base disparity uncertainty in pixels

    Example Usage:
        >>> config = {'baseline': 0.162, 'fx': 458.0}
        >>> uncertainty = DepthUncertainty(config)
        >>> sigma_depth = uncertainty.compute_depth_uncertainty(depth, confidence)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the depth uncertainty estimator.

        Args:
            config: Configuration dictionary containing:
                - baseline: Stereo baseline (meters)
                - fx: Focal length (pixels)
                - sigma_disparity: Base disparity uncertainty (pixels), default 0.5
                - min_confidence: Minimum valid confidence, default 0.1
                - max_uncertainty: Cap on uncertainty (meters), default 10.0
        """
        # =====================================================================
        # STEREO GEOMETRY PARAMETERS
        # These are used for analytical uncertainty computation
        # =====================================================================

        self.baseline = config.get('baseline', 0.162)
        self.fx = config.get('fx', 458.0)

        # Base disparity uncertainty in pixels
        # This represents the minimum error in stereo matching
        # Typical values: 0.25 - 1.0 pixels for good stereo systems
        # SGBM typically achieves ~0.5 pixel accuracy
        self.sigma_disparity = config.get('sigma_disparity', 0.5)

        # Legacy parameter for compatibility
        self.max_consistency_error = config.get('max_consistency_error', 2.0)

        # =====================================================================
        # UNCERTAINTY BOUNDS
        # =====================================================================

        # Minimum confidence to consider a measurement valid
        self.min_confidence = config.get('min_confidence', 0.1)

        # Maximum uncertainty cap (prevents numerical issues)
        self.max_uncertainty = config.get('max_uncertainty', 10.0)

        # Minimum uncertainty floor (even perfect measurements have some error)
        self.min_uncertainty = config.get('min_uncertainty', 0.01)

        # =====================================================================
        # WEIGHTING FACTORS
        # How much each uncertainty source contributes
        # =====================================================================

        # Weight for geometric (analytical) uncertainty
        self.weight_geometric = config.get('weight_geometric', 0.5)

        # Weight for confidence-based uncertainty
        self.weight_confidence = config.get('weight_confidence', 0.5)

        print(f"[DepthUncertainty] Initialized with baseline={self.baseline}m, fx={self.fx}px")
        print(f"[DepthUncertainty] Base disparity uncertainty: {self.sigma_disparity} pixels")

    def compute_uncertainty(
        self,
        disp_left: np.ndarray,
        disp_right: np.ndarray
    ) -> np.ndarray:
        """
        Compute uncertainty from left-right consistency check.

        This is a key quality metric for stereo: if the left and right
        disparity maps don't agree, the match is unreliable.

        HOW IT WORKS:
        For a pixel at (x, y) in the left image with disparity d:
        - It corresponds to pixel (x - d, y) in the right image
        - The right disparity at that location should be approximately d
        - Large differences indicate matching errors

        Args:
            disp_left: Left disparity map (H, W)
            disp_right: Right disparity map (H, W), computed with right as reference

        Returns:
            uncertainty: Uncertainty map (H, W), higher = less confident [0, 1]
        """
        h, w = disp_left.shape

        # Create coordinate grids
        x_coords = np.arange(w)
        x_grid = np.tile(x_coords, (h, 1))
        y_grid = np.tile(np.arange(h).reshape(-1, 1), (1, w))

        # For each pixel in left, find corresponding location in right
        x_right = (x_grid - disp_left).astype(np.int32)
        x_right = np.clip(x_right, 0, w - 1)

        # Get right disparity at those locations
        disp_right_at_left = disp_right[y_grid, x_right]

        # Compute consistency error
        consistency_error = np.abs(disp_left - disp_right_at_left)

        # Convert to uncertainty [0, 1]
        # Larger error = higher uncertainty
        uncertainty = np.tanh(consistency_error / self.max_consistency_error)

        # Invalid disparities get maximum uncertainty
        invalid = (disp_left <= 0) | (disp_right_at_left <= 0)
        uncertainty[invalid] = 1.0

        return uncertainty

    def texture_uncertainty(self, image: np.ndarray) -> np.ndarray:
        """
        Compute uncertainty from image texture.

        Stereo matching relies on texture (image gradients) to find correspondences.
        - High texture: Easy to match, low uncertainty
        - Low texture (smooth surfaces): Hard to match, high uncertainty

        The lunar surface has large textureless regions (smooth regolith),
        which creates significant matching challenges.

        Args:
            image: Input image (H, W) grayscale or (H, W, 3) color

        Returns:
            uncertainty: Texture-based uncertainty (H, W) in [0, 1]
                        0 = rich texture (low uncertainty)
                        1 = no texture (high uncertainty)
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = gray.astype(np.float32)

        # Compute image gradients using Sobel operator
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Apply local averaging to get texture richness in a neighborhood
        kernel_size = 11
        texture_richness = cv2.boxFilter(grad_mag, -1, (kernel_size, kernel_size))

        # Normalize to [0, 1] using adaptive thresholding
        # Use 95th percentile to avoid outliers
        percentile_95 = np.percentile(texture_richness, 95)
        if percentile_95 > 0:
            texture_normalized = texture_richness / percentile_95
        else:
            texture_normalized = np.zeros_like(texture_richness)

        texture_normalized = np.clip(texture_normalized, 0, 1)

        # Convert to uncertainty: low texture = high uncertainty
        uncertainty = 1.0 - texture_normalized

        return uncertainty

    def compute_depth_uncertainty(
        self,
        depth: np.ndarray,
        confidence: np.ndarray,
        texture_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute uncertainty for each depth measurement.

        This is the main function that combines multiple uncertainty sources
        into a single uncertainty estimate per pixel.

        Args:
            depth: Depth map (H, W) in meters
            confidence: Confidence map (H, W) in [0, 1] from stereo matcher
            texture_map: Optional texture richness map (H, W) in [0, 1]

        Returns:
            uncertainty: Uncertainty map (H, W) in meters (standard deviation)
                        Higher values = less reliable depth

        Example:
            >>> sigma = uncertainty.compute_depth_uncertainty(depth, confidence)
            >>> # Points with sigma > 1.0m are quite uncertain
            >>> reliable_mask = sigma < 0.5  # Reliable within 0.5m
        """
        # 1. Compute geometric (analytical) uncertainty
        sigma_geometric = self.compute_geometric_uncertainty(depth)

        # 2. Compute confidence-based uncertainty
        sigma_confidence = self.compute_confidence_uncertainty(depth, confidence)

        # 3. Optionally include texture-based uncertainty
        if texture_map is not None:
            sigma_texture = self.compute_texture_uncertainty(depth, texture_map)
        else:
            sigma_texture = np.zeros_like(depth)

        # 4. Combine uncertainties (root sum of squares for independent sources)
        # sigma_combined = sqrt(w1*s1^2 + w2*s2^2 + w3*s3^2)
        sigma_combined = np.sqrt(
            self.weight_geometric * sigma_geometric**2 +
            self.weight_confidence * sigma_confidence**2 +
            0.3 * sigma_texture**2  # Texture has lower weight
        )

        # 5. Apply bounds
        sigma_combined = np.clip(sigma_combined, self.min_uncertainty, self.max_uncertainty)

        # 6. Mark invalid depths with maximum uncertainty
        invalid_mask = (depth <= 0) | (confidence < self.min_confidence)
        sigma_combined[invalid_mask] = self.max_uncertainty

        return sigma_combined

    def compute_geometric_uncertainty(self, depth: np.ndarray) -> np.ndarray:
        """
        Compute analytical depth uncertainty from stereo geometry.

        DERIVATION:
        -----------
        From the stereo depth equation:
            depth = (baseline * fx) / disparity

        Taking the derivative with respect to disparity:
            d(depth)/d(disparity) = -(baseline * fx) / disparity^2

        Rearranging:
            d(depth)/d(disparity) = -depth^2 / (baseline * fx)

        For small errors, propagating disparity uncertainty:
            sigma_depth = |d(depth)/d(disparity)| * sigma_disparity
            sigma_depth = depth^2 * sigma_disparity / (baseline * fx)

        KEY INSIGHT:
        Depth uncertainty grows with the SQUARE of depth!
        - At 1m: sigma = 0.5 * 1^2 / (0.162 * 458) = 0.007m = 7mm
        - At 5m: sigma = 0.5 * 25 / 74.2 = 0.17m = 17cm
        - At 10m: sigma = 0.5 * 100 / 74.2 = 0.67m = 67cm

        This is why stereo is great for close objects but poor for far ones.

        Args:
            depth: Depth map (H, W) in meters

        Returns:
            sigma: Uncertainty map (H, W) in meters
        """
        # Apply the quadratic depth uncertainty formula
        # sigma_depth = depth^2 * sigma_disparity / (baseline * fx)

        bf = self.baseline * self.fx  # baseline-focal product

        # Avoid issues with zero depth
        safe_depth = np.where(depth > 0, depth, 1e-6)

        sigma = (safe_depth ** 2) * self.sigma_disparity / bf

        # Mark invalid depths
        sigma = np.where(depth > 0, sigma, self.max_uncertainty)

        return sigma

    def compute_confidence_uncertainty(
        self,
        depth: np.ndarray,
        confidence: np.ndarray
    ) -> np.ndarray:
        """
        Convert stereo matching confidence to depth uncertainty.

        The confidence from stereo matching indicates how reliable the match is:
        - High confidence (close to 1): Good match, low uncertainty
        - Low confidence (close to 0): Poor match, high uncertainty

        We model this as:
            sigma = base_sigma * (1 + k * (1 - confidence))

        Where:
            - base_sigma is the geometric uncertainty
            - k is a scaling factor (how much confidence affects uncertainty)
            - Low confidence multiplies uncertainty significantly

        Args:
            depth: Depth map (H, W) in meters
            confidence: Confidence map (H, W) in [0, 1]

        Returns:
            sigma: Uncertainty map (H, W) in meters
        """
        # Get base geometric uncertainty
        base_sigma = self.compute_geometric_uncertainty(depth)

        # Scale factor based on confidence
        # At confidence=1: factor = 1 (no increase)
        # At confidence=0: factor = 1 + k (k times increase)
        k = 5.0  # Uncertainty multiplier for low confidence

        # Clamp confidence to avoid numerical issues
        conf = np.clip(confidence, 0.01, 1.0)

        # Higher uncertainty for lower confidence
        # Using exponential decay: sigma increases as confidence decreases
        confidence_factor = 1.0 + k * (1.0 - conf)

        sigma = base_sigma * confidence_factor

        return sigma

    def compute_texture_uncertainty(
        self,
        depth: np.ndarray,
        texture_map: np.ndarray
    ) -> np.ndarray:
        """
        Compute uncertainty contribution from texture richness.

        Stereo matching relies on texture (image gradients) to find correspondences.
        - High texture: Easy to match, low uncertainty
        - Low texture (smooth surfaces): Hard to match, high uncertainty

        The lunar surface has large textureless regions (smooth regolith),
        which creates significant matching challenges.

        Args:
            depth: Depth map (H, W) in meters
            texture_map: Texture richness (H, W) in [0, 1]
                        0 = no texture, 1 = rich texture

        Returns:
            sigma: Uncertainty map (H, W) in meters
        """
        # Base uncertainty
        base_sigma = self.compute_geometric_uncertainty(depth)

        # Texture factor: low texture = high uncertainty
        # At texture=1: factor = 1
        # At texture=0: factor = 1 + k
        k = 3.0  # Uncertainty multiplier for low texture

        texture_safe = np.clip(texture_map, 0.05, 1.0)
        texture_factor = 1.0 + k * (1.0 - texture_safe)

        sigma = base_sigma * texture_factor

        return sigma

    def get_reliability_mask(
        self,
        depth: np.ndarray,
        confidence: np.ndarray,
        max_depth: float = 15.0,
        min_confidence: float = 0.3,
        max_uncertainty: float = 0.5
    ) -> np.ndarray:
        """
        Create a binary mask of reliable depth measurements.

        This is useful for downstream processing that needs to know
        which depth measurements can be trusted.

        Criteria for reliability:
        1. Depth is positive and below maximum
        2. Confidence is above threshold
        3. Computed uncertainty is below threshold

        Args:
            depth: Depth map (H, W) in meters
            confidence: Confidence map (H, W) in [0, 1]
            max_depth: Maximum reliable depth (meters)
            min_confidence: Minimum confidence threshold
            max_uncertainty: Maximum acceptable uncertainty (meters)

        Returns:
            reliable_mask: Boolean mask (H, W) - True = reliable
        """
        # Compute uncertainty
        sigma = self.compute_depth_uncertainty(depth, confidence)

        # Apply all criteria
        mask = (
            (depth > 0) &
            (depth < max_depth) &
            (confidence >= min_confidence) &
            (sigma <= max_uncertainty)
        )

        return mask


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_depth_uncertainty(config: Dict[str, Any]) -> DepthUncertainty:
    """
    Factory function to create a DepthUncertainty estimator.

    Args:
        config: Configuration dictionary

    Returns:
        DepthUncertainty instance
    """
    return DepthUncertainty(config)


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    """
    Test the depth uncertainty module.
    """
    print("Testing DepthUncertainty...")

    # Create test configuration
    config = {
        'baseline': 0.162,
        'fx': 458.0,
        'sigma_disparity': 0.5
    }

    # Create uncertainty estimator
    uncertainty = DepthUncertainty(config)

    # Test at various depths
    print("\nGeometric uncertainty at various depths:")
    print("=" * 50)
    for d in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        depth_array = np.array([[d]])
        sigma = uncertainty.compute_geometric_uncertainty(depth_array)
        print(f"  Depth: {d:5.1f}m -> Uncertainty: {sigma[0,0]:.3f}m ({sigma[0,0]*100:.1f}cm)")

    # Test with synthetic depth and confidence maps
    print("\n\nFull uncertainty computation:")
    print("=" * 50)

    h, w = 100, 100
    # Create a depth map with gradient (1m to 10m)
    depth = np.linspace(1, 10, w).reshape(1, -1).repeat(h, axis=0)

    # Create confidence map (higher confidence at center)
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    confidence = np.exp(-(xx**2 + yy**2))  # Gaussian centered confidence

    # Compute uncertainty
    sigma = uncertainty.compute_depth_uncertainty(depth, confidence)

    print(f"Depth range: {depth.min():.1f}m - {depth.max():.1f}m")
    print(f"Confidence range: {confidence.min():.2f} - {confidence.max():.2f}")
    print(f"Uncertainty range: {sigma.min():.3f}m - {sigma.max():.3f}m")

    # Test reliability mask
    reliable = uncertainty.get_reliability_mask(
        depth, confidence,
        max_depth=8.0,
        min_confidence=0.3,
        max_uncertainty=0.5
    )
    print(f"\nReliable measurements: {reliable.sum()} / {reliable.size} ({100*reliable.sum()/reliable.size:.1f}%)")

    print("\nTest complete!")
