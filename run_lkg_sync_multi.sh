#!/usr/bin/env bash
# Synchronized playback for combined LKG videos with per-loop re-sync.
# Improved version for execution within the RGBD output directory.

set -euo pipefail

# --- Configuration ---
# BASE_DIR is now set to the script's own directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PLAYER_DIR is the location of the player engine and environment
PLAYER_DIR="$BASE_DIR"

# Video files (relative to BASE_DIR)
V1_FILE="display2_combined.mp4"
V2_FILE="display1_combined.mp4"

V1="$BASE_DIR/$V1_FILE"
V2="$BASE_DIR/$V2_FILE"

# Calibration files (remain in shared directory)
CALIB1="/home/nttdmse/share/calibration/LKG-E12651.json" # Monitor 1 (HDMI 2)
CALIB2="/home/nttdmse/share/calibration/LKG-E12592.json" # Monitor 2 (HDMI 1)

SYNC_FILE="/tmp/lkg_sync_go"
DONE1="/tmp/lkg_p1_done"
DONE2="/tmp/lkg_p2_done"

# --- Validation ---
if [[ ! -f "$V1" ]]; then
    echo "Error: Video file not found: $V1"
    exit 1
fi
if [[ ! -f "$V2" ]]; then
    echo "Error: Video file not found: $V2"
    exit 1
fi

# --- Execution ---
cleanup() {
    echo -e "\n\033[1;31mStopping players...\033[0m"
    kill $PID1 $PID2 2>/dev/null || true
    rm -f "$SYNC_FILE" "$DONE1" "$DONE2"
    exit
}
trap cleanup INT TERM

# Move to player directory to activate environment and run player
cd "$PLAYER_DIR"
if [ -d "lkg-env" ]; then
    source lkg-env/bin/activate
else
    echo "Warning: lkg-env not found, using system python"
fi

export DISPLAY=${DISPLAY:-:1}

echo "================================================"
echo "Starting Synchronized Playback"
echo "Source: $BASE_DIR"
echo "Video 1: $V1_FILE (Monitor 1)"
echo "Video 2: $V2_FILE (Monitor 2)"
echo "================================================"

rm -f "$SYNC_FILE" "$DONE1" "$DONE2"

# Start players
# Display 1 (Right-side LKG)
python lkg_rgbd_player.py "$V1" --monitor 1 --windowed --wait-trigger "$SYNC_FILE" --done-signal "$DONE1" --depthiness 0.80 --focus 0.53 --calib-file "$CALIB1" --loop &
PID1=$!

# Display 2 (Left-side LKG)
python lkg_rgbd_player.py "$V2" --monitor 2 --windowed --wait-trigger "$SYNC_FILE" --done-signal "$DONE2" --depthiness 0.45 --focus 0.84 --calib-file "$CALIB2" --loop &
PID2=$!

echo "Waiting for players to initialize..."
sleep 5

while true; do
    echo "------------------------------------------------"
    echo "Starting Loop Iteration: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------"

    # Send trigger
    touch "$SYNC_FILE"
    sleep 1
    rm -f "$SYNC_FILE"

    # Wait for both to finish this loop
    while [[ ! -f "$DONE1" || ! -f "$DONE2" ]]; do
        sleep 0.1
        if ! kill -0 $PID1 2>/dev/null || ! kill -0 $PID2 2>/dev/null; then
            echo "Error: Player process died."
            cleanup
        fi
    done

    echo "Loop finished. Re-syncing in 1s..."
    rm -f "$DONE1" "$DONE2"
    sleep 1
done
