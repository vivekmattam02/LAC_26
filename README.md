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


## Resume / Portfolio Description

**Lunar Autonomy Challenge (NASA-style), 2024–Present**  
Building an autonomous navigation stack for a simulated lunar rover exploring unknown terrain without GPS or prior maps. The system integrates ORB-SLAM3-based pose estimation with stereo depth, voxel/height mapping, perception-aware costmaps, A* planning, and pure-pursuit + PID control.

**What makes this project different:** depth is treated as uncertain, not ground truth. Uncertainty from stereo is carried into mapping and planning so the rover can penalize and avoid low-confidence areas such as shadows, crater rims, and texture-poor regions.

**Current stereo status:** SGBM is the default working backend, while FoundationStereo is integrated as a configurable inference path (including `foundation_fast`) when a model loader/checkpoint is provided.

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


### Stereo Backend (SGBM vs FoundationStereo)
- `stereo.method: 'sgbm'` is the current default and works without deep model weights.
- `stereo.method: 'foundation'` runs deep-model inference when both `foundation_model_path` and `foundation_loader` are configured.
- `stereo.method: 'foundation_fast'` enables faster-by-default inference knobs (AMP, FP16/channels-last defaults, optional `torch.compile`) while using the same loader path.
- `foundation_loader` must be a callable path like `your_package.loader:load_model` and should return a `torch.nn.Module` from the checkpoint path.
- If model path, loader, or PyTorch setup is unavailable, the code safely falls back to SGBM.


## Collaboration Workflow

When making updates, use a feature branch and local Git identity so commits are attributable to the correct user.

```bash
# Create and switch to a working branch
git checkout -b feature/short-description

# Set commit identity for this repository only
git config user.name "YOUR_NAME"
git config user.email "YOUR_EMAIL"

# Verify before committing
git config --get user.name
git config --get user.email
```

This avoids accidental commits with machine/default identities and keeps history clean for pull requests.

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
