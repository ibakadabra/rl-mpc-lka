#!/usr/bin/env bash
# Launch the CARLA server (run on the machine that owns the simulator).
#
# Examples:
#   ./scripts/carla_setup.sh                       # default port 2000, off-screen
#   ./scripts/carla_setup.sh --windowed            # show the spectator window
#   CARLA_ROOT=/opt/carla-0.9.15 ./scripts/carla_setup.sh
#
# After this is running, in a separate terminal:
#   python scripts/evaluate_carla.py --n-seeds 3

set -e

CARLA_ROOT="${CARLA_ROOT:-$HOME/CARLA_0.9.15}"
PORT="${CARLA_PORT:-2000}"
QUALITY="${CARLA_QUALITY:-Low}"      # Low | Epic
RENDER_FLAG="-RenderOffScreen"

for arg in "$@"; do
  case "$arg" in
    --windowed) RENDER_FLAG="" ;;
  esac
done

if [ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]; then
  echo "CARLA not found at $CARLA_ROOT — set CARLA_ROOT env var." >&2
  exit 1
fi

echo "starting CARLA   root=$CARLA_ROOT  port=$PORT  quality=$QUALITY  flags=$RENDER_FLAG"
exec "$CARLA_ROOT/CarlaUE4.sh" -carla-rpc-port="$PORT" -quality-level="$QUALITY" $RENDER_FLAG
