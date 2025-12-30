#!/usr/bin/env python3
"""
Record sensor data from CARLA for offline development.

Saves:
- Stereo images
- IMU data
- Ground truth poses

Usage:
    python record_data.py --output data/mission_01
"""

import argparse
import os
import time


def main(args):
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'left'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'right'), exist_ok=True)
    
    print(f"Recording to: {args.output}")
    print("Press Ctrl+C to stop recording")
    
    # TODO: Initialize CARLA client
    # client = carla.Client('localhost', 2000)
    # world = client.get_world()
    
    # TODO: Initialize sensors
    
    frame_count = 0
    
    try:
        while True:
            # TODO: Get sensor data
            # TODO: Save images
            # TODO: Save IMU
            # TODO: Save ground truth
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Recorded {frame_count} frames")
            
            time.sleep(0.1)  # 10 Hz
            
    except KeyboardInterrupt:
        print(f"\nRecording complete. Total frames: {frame_count}")
    
    # TODO: Cleanup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Record sensor data')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    args = parser.parse_args()
    main(args)
