# MizukiROBOT

Robot control system with neural network-based controllers for Affetto.

## Description

This project is a comprehensive robot control system that includes:

- Neural network-based controllers for robot motion
- Data collection and training tools
- PID controller tuning utilities
- Trajectory recording and tracking capabilities
- Performance evaluation and analysis tools

## Getting Started

### Dependencies

This project uses Python 3.12+ and is managed with [uv](https://docs.astral.sh/uv/).

### Installation

1. Install uv if you haven't already:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Clone this repository and install dependencies:
   ```bash
   git clone <your-repo-url>
   cd MizukiROBOT
   uv sync
   ```

3. Run programs using uv:
   ```bash
   uv run python apps/collect_data_myrobot.py -h
   ```

## Features

- **Data Collection**: Collect sensory and actuation data during robot movements
- **Neural Network Training**: Train MLP models for motion control
- **Trajectory Management**: Record and track joint angle trajectories
- **Performance Analysis**: Calculate RMSE and performance scores
- **PID Tuning**: Automated PID controller parameter optimization

## Project Structure

- `apps/` - Main application scripts
- `src/` - Core source code
- `affetto-nn-ctrl/` - Neural network controller implementation
- `desktop/` - Desktop application components
- `data/` - Data files and datasets
- `docs/` - Documentation

## License

MIT License