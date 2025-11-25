#!/bin/bash
# Mobile VLA Training Script

set -e

echo "========================================="
echo "Mobile VLA Training with RoboVLMs"
echo "========================================="

# Check if config exists
CONFIG_PATH="configs/mobile_vla/train_mobile_vla_full_ft.json"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Check if data exists
DATA_DIR="../ROS_action/mobile_vla_dataset"
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

# Count .h5 files
NUM_FILES=$(find "$DATA_DIR" -name "*.h5" | wc -l)
echo "Found $NUM_FILES .h5 files in dataset"

if [ "$NUM_FILES" -eq 0 ]; then
    echo "Error: No .h5 files found in dataset"
    exit 1
fi

# Parse arguments
MODE="train"
RESUME=""
TEST_MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE="--test"
            echo "Running in TEST mode"
            shift
            ;;
        --resume)
            RESUME="--resume $2"
            echo "Resuming from checkpoint: $2"
            shift 2
            ;;
        --docker)
            MODE="docker"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directories
mkdir -p runs/mobile_vla/checkpoints
mkdir -p runs/mobile_vla/logs

if [ "$MODE" == "docker" ]; then
    echo "Running in Docker mode..."
    docker-compose -f docker-compose-mobile-vla.yml up train_mobile_vla
else
    echo "Running in local mode..."
    
    # Check if ROS2 is sourced
    if [ -z "$ROS_DISTRO" ]; then
        echo "Warning: ROS2 not sourced. Sourcing /opt/ros/humble/setup.bash"
        source /opt/ros/humble/setup.bash
    fi
    
    # Run training
    python3 train_mobile_vla.py \
        --config "$CONFIG_PATH" \
        $RESUME \
        $TEST_MODE
fi

echo "========================================="
echo "Training completed!"
echo "========================================="

