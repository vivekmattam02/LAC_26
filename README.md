# Perception-Aware Lunar Rover Navigation

Autonomous navigation system for the Lunar Autonomy Challenge using perception-aware path planning.

## Overview

```
Stereo Cameras → Stereo Depth → Dense Mapping → Perception-Aware Costmap → Path Planning → Control
       ↓
   ORB-SLAM3 → Pose
```

**Key Features:**
- Dense stereo depth with uncertainty estimation
- Voxel-based 3D terrain mapping
- Perception-aware costmap (obstacles + uncertainty + shadows)
- A* path planning that avoids uncertain regions
- Coverage planning for efficient exploration

## Project Structure

```
lunar_navigation/
├── sensors/          # Camera and IMU interfaces
├── depth/            # Stereo matching and uncertainty
├── mapping/          # Voxel grid and height map
├── costmap/          # Perception-aware costmap layers
├── planning/         # A* and coverage planning
├── control/          # Trajectory following
├── visualization/    # Real-time visualization
├── scripts/          # Main execution scripts
├── config/           # Configuration files
└── tests/            # Unit tests
```

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/lunar_navigation.git
cd lunar_navigation

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install CARLA (see CARLA documentation)
# Install ORB-SLAM3 (see ORB-SLAM3 documentation)
```

## Usage

### Record Data
```bash
python scripts/record_data.py --output data/mission_01
```

### Run Autonomous Navigation
```bash
python scripts/run_autonomous.py --config config/params.yaml
```

## Architecture

### Perception
- **ORB-SLAM3**: Visual odometry and sparse mapping
- **Stereo Depth**: Dense depth from stereo matching (SGBM)
- **Uncertainty**: Left-right consistency check

### Mapping
- **Voxel Grid**: 3D occupancy with uncertainty
- **Height Map**: 2.5D representation for LAC scoring

### Planning
- **Costmap Layers**:
  - Obstacle layer (height gradients)
  - Uncertainty layer (high uncertainty = high cost)
  - Shadow layer (dark regions penalized)
- **A* Planner**: Perception-aware path planning
- **Coverage Planner**: Frontier-based exploration

### Control
- Pure pursuit path tracking
- PID speed control

## TODO

- [ ] Week 1: Sensor interface
- [ ] Week 2: Stereo depth
- [ ] Week 3: Dense mapping
- [ ] Week 4: Perception-aware costmap
- [ ] Week 5: Path planning
- [ ] Week 6: Control + integration
- [ ] Week 7: Testing
- [ ] Week 8: Polish + demo

## License

MIT
