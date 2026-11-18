#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/data/isaac-sim/isaac-sim-6.0.1}"

if [[ ! -x "$ISAAC_SIM_ROOT/python.sh" ]]; then
    echo "Isaac Sim python.sh not found under $ISAAC_SIM_ROOT" >&2
    echo "Set ISAAC_SIM_ROOT to your Isaac Sim installation." >&2
    exit 1
fi

exec "$ISAAC_SIM_ROOT/python.sh" "$HERE/turtlebot3_goal_demo.py" \
    --headless \
    --capture-images \
    --capture-interval 120 \
    --record-video \
    --output-dir "$HERE/output" \
    "$@"
