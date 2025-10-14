#!/bin/bash
# Mobile VLA Inference Script

set -e

echo "========================================="
echo "Mobile VLA Inference with RoboVLMs"
echo "========================================="

# Parse arguments
CHECKPOINT=""
CONFIG="configs/mobile_vla/train_mobile_vla_full_ft.json"
DEVICE="cuda"
ROS2_MODE=""
DOCKER_MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --ros2)
            ROS2_MODE="--ros2"
            shift
            ;;
        --docker)
            DOCKER_MODE="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check checkpoint
if [ -z "$CHECKPOINT" ]; then
    # Find latest checkpoint
    CHECKPOINT=$(find runs/mobile_vla/checkpoints -name "*.ckpt" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -z "$CHECKPOINT" ]; then
        echo "Error: No checkpoint found. Please specify --checkpoint"
        exit 1
    fi
    
    echo "Using latest checkpoint: $CHECKPOINT"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Check config
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config not found: $CONFIG"
    exit 1
fi

if [ "$DOCKER_MODE" == "true" ]; then
    echo "Running in Docker mode..."
    docker-compose -f docker-compose-mobile-vla.yml up inference_mobile_vla
else
    echo "Running in local mode..."
    echo "Checkpoint: $CHECKPOINT"
    echo "Config: $CONFIG"
    echo "Device: $DEVICE"
    
    if [ -n "$ROS2_MODE" ]; then
        echo "ROS2 mode: ENABLED"
        
        # Check if ROS2 is sourced
        if [ -z "$ROS_DISTRO" ]; then
            echo "Sourcing ROS2..."
            source /opt/ros/humble/setup.bash
        fi
    else
        echo "ROS2 mode: DISABLED (test mode)"
    fi
    
    # Run inference
    python3 eval/mobile_vla/inference_wrapper.py \
        --checkpoint "$CHECKPOINT" \
        --config "$CONFIG" \
        --device "$DEVICE" \
        $ROS2_MODE
fi

echo "========================================="
echo "Inference completed!"
echo "========================================="

