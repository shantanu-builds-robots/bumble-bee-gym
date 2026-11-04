#!/bin/bash
# Record the Gazebo side of the parity test into generated/gazebo_parity.{csv,switch.txt}.
#
# Needs the bumble-bee Docker image (built from your bumble-bee checkout):
#   docker build -f <bumble-bee>/docker/Dockerfile -t bumble-bee <bumble-bee>
# Runs headless (no Gazebo window) and limits the container to a few CPUs.
# BUMBLE_BEE_WS=<colcon workspace>: use that built workspace (mounted at /out) instead of the image's own.
set -euo pipefail
GYM="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${BUMBLE_BEE_IMAGE:-bumble-bee}"
CPUS="${PARITY_CPUS:-6}"

"$GYM/.venv/bin/python" -c "
from pathlib import Path
from bumble_bee_gym import course, worlds
out = Path('$GYM/generated')
world, _ = worlds.write(course.load('empty'), out)
# Same world, physics slowed to a tenth of real time so each command reaches the plugin within one step.
slow = world.read_text().replace('<real_time_update_rate>1000<', '<real_time_update_rate>100<')
(out / 'empty_parity.world').write_text(slow)"

WS_ARGS=()
[ -n "${BUMBLE_BEE_WS:-}" ] && WS_ARGS=(-v "$BUMBLE_BEE_WS":/out)

docker run --rm --cpus="$CPUS" -v "$GYM":/gym "${WS_ARGS[@]}" -e PYTHONPATH=/gym --entrypoint bash "$IMAGE" -c '
  source /opt/ros/humble/setup.bash
  source /usr/share/gazebo/setup.sh
  for ws in /out/install /ws/install; do [ -f $ws/setup.bash ] && { source $ws/setup.bash; break; }; done
  ros2 launch drone_delivery gazebo.launch.py world:=/gym/generated/empty_parity.world headless:=true pause:=true \
      x:=0.0 y:=0.0 z:=2.0 > /tmp/gazebo.log 2>&1 &
  until ros2 topic list 2>/dev/null | grep -q /drone_pose; do sleep 1; done
  sleep 3
  python3 /gym/scripts/gazebo_parity_probe.py /gym/generated/gazebo_parity
  grep -E "Thrust map|Max rotor" /tmp/gazebo.log || true
  chown '"$(id -u):$(id -g)"' /gym/generated/gazebo_parity.*
'
