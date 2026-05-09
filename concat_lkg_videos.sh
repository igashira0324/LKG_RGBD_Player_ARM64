#!/usr/bin/env bash
# Concatenate unified RGBD videos into one combined video for each display.
# Improved version for execution within the RGBD output directory.

set -euo pipefail

# BASE_DIR is the directory where the script resides
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# Input Subdirectories
D1_DIR="display1"
D2_DIR="display2"

# Output Files
OUT1="display1_combined.mp4"
OUT2="display2_combined.mp4"

echo "Checking input files..."

# Check if directories exist
if [[ ! -d "$D1_DIR" || ! -d "$D2_DIR" ]]; then
    echo "Error: Input directories $D1_DIR or $D2_DIR not found in $BASE_DIR"
    exit 1
fi

shopt -s nullglob

echo "Gathering files for Display 1..."
D1_FILES=("$D1_DIR"/*.mp4)
if (( ${#D1_FILES[@]} == 0 )); then
    echo "Error: No mp4 files found in $D1_DIR"
    exit 1
fi

FILTER1=""
INPUTS1=()
idx=0
for f in "${D1_FILES[@]}"; do
    INPUTS1+=("-i" "$f")
    FILTER1="$FILTER1[$idx:v]scale=1152:1024[v$idx];"
    idx=$((idx+1))
done
CONCAT_V=""
for i in $(seq 0 $((idx-1))); do CONCAT_V="${CONCAT_V}[v$i]"; done
FILTER1="${FILTER1}${CONCAT_V}concat=n=$idx:v=1:a=0[v]"

echo "Starting concatenation for Display 1 (Visual only)..."
ffmpeg -y "${INPUTS1[@]}" -filter_complex "$FILTER1" -map "[v]" -c:v libx264 -crf 10 -pix_fmt yuv420p "$OUT1"

echo "Gathering files for Display 2..."
D2_FILES=("$D2_DIR"/*.mp4)
if (( ${#D2_FILES[@]} == 0 )); then
    echo "Error: No mp4 files found in $D2_DIR"
    exit 1
fi

FILTER2=""
INPUTS2=()
idx=0
for f in "${D2_FILES[@]}"; do
    INPUTS2+=("-i" "$f")
    FILTER2="$FILTER2[$idx:v]scale=1152:1024[v$idx];"
    idx=$((idx+1))
done
CONCAT_VA=""
for i in $(seq 0 $((idx-1))); do CONCAT_VA="${CONCAT_VA}[v$i][$i:a]"; done
FILTER2="${FILTER2}${CONCAT_VA}concat=n=$idx:v=1:a=1[v][a]"

echo "Starting concatenation for Display 2 (With audio)..."
ffmpeg -y "${INPUTS2[@]}" -filter_complex "$FILTER2" -map "[v]" -map "[a]" -c:v libx264 -crf 10 -pix_fmt yuv420p -c:a aac -b:a 192k "$OUT2"

echo "Concatenation complete!"
echo "Generated: $OUT1"
echo "Generated: $OUT2"
