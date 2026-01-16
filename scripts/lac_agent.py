#!/usr/bin/env python3
"""
================================================================================
LAC AUTONOMOUS AGENT - PERCEPTION-AWARE LUNAR NAVIGATION
================================================================================

This is the main autonomous agent for the Lunar Autonomy Challenge.
It integrates all our navigation modules:
    - Stereo depth estimation with FoundationStereo/SGBM
    - Uncertainty-aware voxel mapping
    - Perception-aware costmap
    - A* path planning
    - Pure pursuit control

The agent extends the LAC AutonomousAgent base class and interfaces
with the CARLA-based lunar simulator.

ARCHITECTURE:
-------------
    Cameras (FrontLeft/FrontRight)
           |
           v
    Stereo Depth + Uncertainty
           |
           v
    Voxel Grid (3D Map)
           |
           v
    Height Map (2.5D)
           |
           v
    Costmap (Obstacles + Uncertainty + Slopes)
           |
           v
    A* Path Planner
           |
           v
    Pure Pursuit Controller
           |
           v
    VehicleVelocityControl (LAC API)

USAGE:
------
This agent is loaded by the LAC leaderboard system. To register it:

    # In your agents folder
    from scripts.lac_agent import PerceptionAwareAgent

================================================================================
"""

import sys
import os
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from depth.stereo_matcher import StereoMatcher
from depth.uncertainty import DepthUncertainty
from mapping.voxel_grid import VoxelGrid
from mapping.height_map import HeightMap
from costmap.costmap import PerceptionAwareCostmap
from planning.astar import AStarPlanner
from control.controller import Controller


class PerceptionAwareAgent:
    """
    Perception-aware autonomous agent for the Lunar Autonomy Challenge.

    This agent uses stereo vision with uncertainty estimation to build
    a map of the environment, then plans and executes paths that prefer
    well-observed, safe regions.

    Key Features:
    - Stereo depth from FrontLeft/FrontRight cameras
    - Uncertainty propagation through the pipeline
    - Multi-layer costmap (obstacles, uncertainty, slopes)
    - A* planning with perception-aware costs
    - Pure pursuit trajectory following

    Attributes:
        config (dict): Configuration parameters
        stereo_matcher (StereoMatcher): Stereo depth computation
        uncertainty (DepthUncertainty): Depth uncertainty estimation
        voxel_grid (VoxelGrid): 3D occupancy mapping
        height_map (HeightMap): 2.5D terrain representation
        costmap (PerceptionAwareCostmap): Multi-layer cost map
        planner (AStarPlanner): Path planning
        controller (Controller): Trajectory following
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the perception-aware agent.

        Args:
            config: Configuration dictionary. If None, uses defaults.
        """
        print("=" * 60)
        print("PERCEPTION-AWARE LUNAR NAVIGATION AGENT")
        print("=" * 60)

        # Use provided config or defaults
        self.config = config or self._default_config()

        # =====================================================================
        # INITIALIZE MODULES
        # =====================================================================

        print("\n[Agent] Initializing modules...")

        # Stereo depth estimation
        self.stereo_matcher = StereoMatcher(self.config['stereo'])

        # Depth uncertainty estimation
        self.uncertainty_estimator = DepthUncertainty(self.config['uncertainty'])

        # 3D voxel mapping
        self.voxel_grid = VoxelGrid(self.config['mapping'])

        # 2.5D height map (for LAC scoring)
        self.height_map = HeightMap(self.config['height_map'])

        # Perception-aware costmap
        self.costmap = PerceptionAwareCostmap(self.config['costmap'])

        # Path planner
        self.planner = AStarPlanner(self.costmap, self.config['planning'])

        # Trajectory controller
        self.controller = Controller(self.config['control'])

        # =====================================================================
        # STATE VARIABLES
        # =====================================================================

        # Current robot pose (x, y, theta)
        self.pose = (0.0, 0.0, 0.0)

        # Current linear velocity
        self.velocity = 0.0

        # Current planned path
        self.current_path = None

        # Current goal
        self.current_goal = None

        # Exploration targets
        self.exploration_targets = []

        # Mission phase
        self.phase = "INITIALIZING"

        # Frame counter
        self.frame_count = 0

        # Timing
        self.last_update_time = time.time()

        print("\n[Agent] Initialization complete!")
        print("=" * 60)

    def _default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            'stereo': {
                'baseline': 0.162,
                'fx': 458.0,
                'fy': 458.0,
                'cx': 320.0,
                'cy': 240.0,
                'method': 'sgbm',
                'max_disparity': 128,
                'min_depth': 0.5,
                'max_depth': 20.0
            },
            'uncertainty': {
                'baseline': 0.162,
                'fx': 458.0,
                'sigma_disparity': 0.5
            },
            'mapping': {
                'resolution': 0.1,
                'grid_size': [200, 200, 50],
                'origin': [-10.0, -10.0, -2.0]
            },
            'height_map': {
                'resolution': 0.1,
                'size_x': 200,
                'size_y': 200,
                'origin': [-10.0, -10.0]
            },
            'costmap': {
                'resolution': 0.2,
                'size_x': 100,
                'size_y': 100,
                'origin': [-10.0, -10.0],
                'inflation_radius': 0.3,
                'obstacle_weight': 100.0,
                'uncertainty_weight': 5.0,
                'slope_weight': 3.0
            },
            'planning': {
                'heuristic_weight': 1.0,
                'allow_diagonal': True
            },
            'control': {
                'lookahead_distance': 1.0,
                'max_linear_velocity': 0.4,
                'max_angular_velocity': 1.0,
                'goal_tolerance': 0.3
            }
        }

    def run_step(
        self,
        images: Dict[str, np.ndarray],
        imu_data: Dict[str, Any],
        pose_estimate: Optional[Tuple[float, float, float]] = None,
        goal: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, float]:
        """
        Execute one step of autonomous navigation.

        This is the main function called every control cycle.

        Args:
            images: Dictionary of camera images
                   {'FrontLeft': np.ndarray, 'FrontRight': np.ndarray, ...}
            imu_data: IMU measurements {'accel': ..., 'gyro': ..., 'orientation': ...}
            pose_estimate: Optional external pose estimate (x, y, theta) from SLAM
            goal: Optional goal position (x, y) to navigate to

        Returns:
            linear_velocity: Forward velocity command (m/s)
            angular_velocity: Turning rate command (rad/s)
        """
        self.frame_count += 1
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # =====================================================================
        # STEP 1: UPDATE POSE
        # =====================================================================

        if pose_estimate is not None:
            self.pose = pose_estimate
        # If no external pose, we would integrate IMU here
        # For now, assume pose comes from ORB-SLAM3

        # =====================================================================
        # STEP 2: PROCESS STEREO IMAGES
        # =====================================================================

        if 'FrontLeft' in images and 'FrontRight' in images:
            # Compute stereo depth
            img_left = images['FrontLeft']
            img_right = images['FrontRight']

            depth, confidence = self.stereo_matcher.compute_depth(img_left, img_right)

            # Compute uncertainty
            uncertainty = self.uncertainty_estimator.compute_depth_uncertainty(
                depth, confidence
            )

            # Update map (every few frames for efficiency)
            if self.frame_count % 5 == 0:
                self._update_map(depth, uncertainty)

        # =====================================================================
        # STEP 3: UPDATE GOAL AND PLAN
        # =====================================================================

        # Update goal if provided
        if goal is not None and goal != self.current_goal:
            self.current_goal = goal
            self._plan_to_goal(goal)

        # If no current path, plan one
        if self.current_path is None and self.current_goal is not None:
            self._plan_to_goal(self.current_goal)

        # =====================================================================
        # STEP 4: EXECUTE CONTROL
        # =====================================================================

        if self.current_path is not None and len(self.current_path) > 0:
            linear_vel, angular_vel = self.controller.get_control(
                self.pose,
                self.current_path,
                self.velocity,
                dt
            )

            # Check if goal reached
            if self.controller.is_goal_reached():
                print(f"[Agent] Goal reached!")
                self.current_path = None
                self.current_goal = None
                linear_vel, angular_vel = 0.0, 0.0
        else:
            # No path, stop
            linear_vel, angular_vel = 0.0, 0.0

        # Update velocity state
        self.velocity = linear_vel

        return linear_vel, angular_vel

    def _update_map(self, depth: np.ndarray, uncertainty: np.ndarray):
        """
        Update the voxel grid and costmap from depth measurements.

        Args:
            depth: Depth map (H, W) in meters
            uncertainty: Uncertainty map (H, W) in meters
        """
        # Create camera pose matrix from current pose
        x, y, theta = self.pose
        pose_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0, x],
            [np.sin(theta), np.cos(theta), 0, y],
            [0, 0, 1, 0],  # Assuming 2D ground plane
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Camera intrinsics
        intrinsics = {
            'fx': self.config['stereo']['fx'],
            'fy': self.config['stereo']['fy'],
            'cx': self.config['stereo']['cx'],
            'cy': self.config['stereo']['cy']
        }

        # Integrate into voxel grid
        self.voxel_grid.integrate_depth(depth, uncertainty, pose_matrix, intrinsics)

        # Update height map from voxel grid
        self.height_map.update_from_voxel_grid(self.voxel_grid)

        # Update costmap from height map
        self.costmap.update_from_height_map(self.height_map)

    def _plan_to_goal(self, goal: Tuple[float, float]):
        """
        Plan a path to the given goal.

        Args:
            goal: Goal position (x, y) in world coordinates
        """
        start = (self.pose[0], self.pose[1])

        print(f"[Agent] Planning path from {start} to {goal}")

        path, cost = self.planner.plan(start, goal)

        if path is not None:
            # Smooth the path
            self.current_path = self.planner.smooth_path(path)
            self.controller.reset()
            print(f"[Agent] Path found with {len(self.current_path)} waypoints")
        else:
            self.current_path = None
            print(f"[Agent] No path found to {goal}")

    def get_geometric_map(self):
        """
        Get the geometric map for LAC scoring.

        Returns:
            height_map: HeightMap instance with current terrain data
        """
        return self.height_map

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics for debugging.

        Returns:
            Dictionary with agent state and module statistics
        """
        stats = {
            'frame_count': self.frame_count,
            'phase': self.phase,
            'pose': self.pose,
            'velocity': self.velocity,
            'has_path': self.current_path is not None,
            'path_length': len(self.current_path) if self.current_path else 0,
            'goal': self.current_goal,
            'voxel_grid': self.voxel_grid.get_statistics(),
            'height_map': self.height_map.get_statistics(),
            'costmap': self.costmap.get_statistics()
        }
        return stats


# =============================================================================
# LAC COMPATIBLE WRAPPER
# =============================================================================

class LACAutonomousAgent:
    """
    LAC-compatible wrapper for the PerceptionAwareAgent.

    This class provides the interface expected by the LAC leaderboard system.
    It inherits from the LAC AutonomousAgent base class (when available).

    Usage in LAC:
        The LAC system will instantiate this class and call:
        - setup(): Initialize the agent
        - sensors(): Return required sensors
        - run_step(): Execute one navigation step
        - destroy(): Clean up
    """

    def __init__(self, path_to_conf_file: str = None):
        """
        Initialize the LAC agent.

        Args:
            path_to_conf_file: Path to configuration file (optional)
        """
        self._agent = None
        self._config_path = path_to_conf_file
        self._geometric_map = None

    def setup(self, path_to_conf_file: str = None):
        """
        Setup the agent (called by LAC before run_step).

        Args:
            path_to_conf_file: Path to configuration file
        """
        # Load config if provided
        config = None
        if path_to_conf_file or self._config_path:
            import yaml
            config_file = path_to_conf_file or self._config_path
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

        # Create the perception-aware agent
        self._agent = PerceptionAwareAgent(config)

    def sensors(self):
        """
        Return the sensor configuration required by this agent.

        Returns:
            List of sensor configurations
        """
        # We need stereo cameras (FrontLeft and FrontRight)
        # The LAC simulator provides 8 camera positions
        sensors = [
            {
                'type': 'sensor.camera.rgb',
                'id': 'FrontLeft',
                'x': 0.7, 'y': -0.4, 'z': 1.6,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -15.0,
                'width': 640, 'height': 480,
                'fov': 90
            },
            {
                'type': 'sensor.camera.rgb',
                'id': 'FrontRight',
                'x': 0.7, 'y': 0.4, 'z': 1.6,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 15.0,
                'width': 640, 'height': 480,
                'fov': 90
            },
            {
                'type': 'sensor.other.imu',
                'id': 'IMU',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
            }
        ]
        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of autonomous navigation.

        This is called by the LAC system at each simulation step.

        Args:
            input_data: Dictionary of sensor data
                       {'FrontLeft': (timestamp, array), 'FrontRight': ..., 'IMU': ...}
            timestamp: Current simulation timestamp

        Returns:
            control: VehicleVelocityControl or tuple (linear_vel, angular_vel)
        """
        # Extract images
        images = {}
        imu_data = {}

        for sensor_id, (ts, data) in input_data.items():
            if 'Front' in sensor_id:
                images[sensor_id] = data
            elif sensor_id == 'IMU':
                imu_data = data

        # Get pose from SLAM (if available) or use dead reckoning
        # In a real implementation, this would come from ORB-SLAM3
        pose_estimate = self._agent.pose  # Use last known pose

        # Run navigation step
        linear_vel, angular_vel = self._agent.run_step(
            images, imu_data, pose_estimate, goal=self._current_goal
        )

        # Return control command
        # Format depends on LAC version - adjust as needed
        return (linear_vel, angular_vel)

    def set_goal(self, goal: Tuple[float, float]):
        """
        Set the navigation goal.

        Args:
            goal: Goal position (x, y) in world coordinates
        """
        self._current_goal = goal

    def get_geometric_map(self, geometric_map):
        """
        Export our map to the LAC geometric map for scoring.

        Args:
            geometric_map: LAC GeometricMap instance
        """
        if self._agent is not None:
            self._agent.height_map.export_for_lac(geometric_map)

    def destroy(self):
        """Clean up resources."""
        print("[LACAgent] Destroying agent")
        self._agent = None


# =============================================================================
# STANDALONE TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the agent with synthetic data.
    """
    print("Testing PerceptionAwareAgent...")

    # Create agent with default config
    agent = PerceptionAwareAgent()

    # Create synthetic images
    h, w = 480, 640
    img_left = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    img_right = np.roll(img_left, 20, axis=1)  # Simulate 20 pixel disparity

    images = {
        'FrontLeft': img_left,
        'FrontRight': img_right
    }

    imu_data = {
        'accel': np.array([0, 0, 9.81]),
        'gyro': np.array([0, 0, 0])
    }

    # Set a goal
    goal = (5.0, 5.0)

    # Run a few steps
    print("\nRunning navigation steps...")
    for i in range(10):
        linear_vel, angular_vel = agent.run_step(
            images, imu_data,
            pose_estimate=(i * 0.1, 0.0, 0.0),  # Simulated pose
            goal=goal
        )
        print(f"Step {i}: v={linear_vel:.2f} m/s, w={angular_vel:.2f} rad/s")

    # Get statistics
    stats = agent.get_statistics()
    print("\nAgent statistics:")
    print(f"  Frame count: {stats['frame_count']}")
    print(f"  Pose: {stats['pose']}")
    print(f"  Has path: {stats['has_path']}")

    print("\nTest complete!")
