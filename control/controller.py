"""
Trajectory following controller.

Converts trajectory to velocity commands for CARLA vehicle.
Uses pure pursuit for path tracking.
"""

import numpy as np


class Controller:
    def __init__(self, config):
        self.lookahead_distance = config.get('lookahead_distance', 2.0)
        self.max_steer = config.get('max_steer', 0.5)
        self.max_throttle = config.get('max_throttle', 0.5)
        
        # PID gains for speed control
        self.kp = config.get('kp', 1.0)
        self.ki = config.get('ki', 0.0)
        self.kd = config.get('kd', 0.1)
        
        self.integral_error = 0.0
        self.prev_error = 0.0
    
    def get_control(self, current_pose, current_velocity, trajectory):
        """
        Compute control commands.
        
        Args:
            current_pose: Current (x, y, theta)
            current_velocity: Current speed
            trajectory: Target trajectory
        
        Returns:
            throttle: Throttle command [-1, 1]
            steer: Steering command [-1, 1]
            brake: Brake command [0, 1]
        """
        # TODO: Implement
        pass
    
    def pure_pursuit(self, current_pose, trajectory):
        """Pure pursuit steering control"""
        # TODO: Implement
        pass
    
    def pid_speed(self, current_speed, target_speed, dt):
        """PID speed control"""
        # TODO: Implement
        pass
    
    def reset(self):
        """Reset controller state"""
        self.integral_error = 0.0
        self.prev_error = 0.0
