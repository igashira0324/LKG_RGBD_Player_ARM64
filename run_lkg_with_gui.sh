#!/bin/bash

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/lkg-env"

# If venv not found in script dir, check parent (for export_repo case)
if [ ! -d "$VENV_DIR" ]; then
    VENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/lkg-env"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment (lkg-env) not found."
    echo "Please create it using: python3 -m venv lkg-env && source lkg-env/bin/activate && pip install opencv-python PyOpenGL PyOpenGL-accelerate glfw PyQt5"
    exit 1
fi
VENV_PYTHON="$VENV_DIR/bin/python3"
PLAYER_SCRIPT="$SCRIPT_DIR/lkg_rgbd_player.py"
GUI_SCRIPT="$SCRIPT_DIR/lkg_control_panel.py"
CALIB_FILE="$SCRIPT_DIR/lkg_calibration.json"

export DISPLAY=${DISPLAY:-:1}

# Help message
usage() {
    echo "Usage: $0 <input_video_or_image> [--gui-only] [--force-player]"
    echo "Example: $0 VideoProject1-1.mp4"
    echo "Options:"
    echo "  --gui-only       Launch only the control panel (no player)"
    echo "  --force-player   Force player launch even if LKG not detected"
    echo "  --monitor N        Monitor index (1 or 2)"
    echo "  --calib-file PATH  Path to specific calibration JSON"
    exit 1
}

# Auto-detect Looking Glass Go
is_lkg_connected() {
    # Check USB for "Looking Glass" string or known Vendor IDs
    if lsusb | grep -qi "Looking Glass"; then return 0; fi
    if lsusb | grep -q "21cf:"; then return 0; fi
    if lsusb | grep -q "05df:16c0"; then return 0; fi
    # Check xrandr for LKG Go resolution (1440x2560)
    if xrandr --current 2>/dev/null | grep -q "1440x2560"; then return 0; fi
    return 1
}

# Parse arguments
GUI_ONLY=false
MONITOR=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gui-only)
            GUI_ONLY=true
            shift
            ;;
        --force-player)
            GUI_ONLY=false
            shift
            ;;
        --monitor)
            MONITOR="$2"
            shift 2
            ;;
        --calib-file)
            USER_CALIB_FILE="$2"
            shift 2
            ;;
        *)
            if [[ "$1" == --* ]]; then
                EXTRA_ARGS+=("$1")
                if [[ $# -gt 1 && "$2" != --* ]]; then
                    EXTRA_ARGS+=("$2")
                    shift
                fi
            else
                INPUT_FILE="$1"
            fi
            shift
            ;;

    esac
done

if ! is_lkg_connected; then
    if [ "$GUI_ONLY" = false ] && [ -n "$INPUT_FILE" ]; then
        echo "Looking Glass Go not detected. Starting in GUI ONLY mode."
        GUI_ONLY=true
    fi
else
    if [ "$GUI_ONLY" = false ]; then
        echo "Looking Glass Go detected! Initializing system..."
    fi
fi

if [ "$GUI_ONLY" != true ] && [ -z "$INPUT_FILE" ]; then
    usage
fi

# Cleanup function
cleanup() {
    if [ -n "${PLAYER_PID:-}" ]; then
        echo "Closing player..."
        kill $PLAYER_PID 2>/dev/null
        wait $PLAYER_PID 2>/dev/null
    fi
    echo "Done."
}
trap cleanup EXIT INT TERM

if [ "$GUI_ONLY" = true ]; then
    echo "Running in GUI ONLY mode (No Player)."
else
    # Build player command with proper flags
    PLAYER_ARGS=("$INPUT_FILE" "--monitor" "$MONITOR" "--loop" "--depth-loc" "3")
    
    # Add calibration file if exists
    if [ -n "${USER_CALIB_FILE:-}" ] && [ -f "$USER_CALIB_FILE" ]; then
        PLAYER_ARGS+=("--calib-file" "$USER_CALIB_FILE")
    elif [ -f "$CALIB_FILE" ]; then
        PLAYER_ARGS+=("--calib-file" "$CALIB_FILE")
    fi
    
    # Start the Player in the background
    echo "Starting player: $INPUT_FILE on Monitor $MONITOR"
    "$VENV_PYTHON" "$PLAYER_SCRIPT" "${PLAYER_ARGS[@]}" "${EXTRA_ARGS[@]}" &

    PLAYER_PID=$!
    
    # Wait for player to initialize
    sleep 2
fi

# Start the Control Panel
echo "Launching Control Panel for Monitor $MONITOR..."
GUI_ARGS=("--monitor" "$MONITOR")

if [ -n "${USER_CALIB_FILE:-}" ] && [ -f "$USER_CALIB_FILE" ]; then
    GUI_ARGS+=("--calib-file" "$USER_CALIB_FILE")
fi

"$VENV_PYTHON" "$GUI_SCRIPT" "${GUI_ARGS[@]}" "${EXTRA_ARGS[@]}"

