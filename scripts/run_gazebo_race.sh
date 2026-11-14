#!/bin/bash
# Fly a MuJoCo-trained race-gates policy in Gazebo, headless, and record the flight.
#
#   scripts/run_gazebo_race.sh outputs/gates_02/policy_000024576000.npz [seconds]
#
# Needs the bumble-bee Docker image (docker build -f <bumble-bee>/docker/Dockerfile -t bumble-bee <bumble-bee>) and
# a policy exported with `python -m bumble_bee_gym.learn.export_policy`. Physics runs at full real time (1000 steps of
# 1 ms per second), and the pilot flies like a flight computer: a 100 Hz wall-clock loop acting on the newest pose, so
# ROS delivery delays reach the controller as they would on real hardware. The drone spawns at rest on the start line
# (30% of the way from the last gate to gate 0, facing gate 0), as in learn/play_gates.py's filmed flight.
# Writes <policy dir>/<NAME or gazebo_race>.csv and .log; the log ends with the measured pose age and real-time factor.
#   scripts/run_gazebo_race.sh hover [seconds]   -> no policy, hover speed on every rotor: a timing check only
#                                                   (writes outputs/timing/<NAME or hover>.csv)
#   scripts/run_gazebo_race.sh mpc:outputs/mpc/plan.npz [seconds]
#                                                -> the NMPC tracking a time-optimal plan (bumble_bee_gym/mpc), with
#                                                   CasADi for the image's Python 3.10 from ~/.cache/bumble-bee-gym/pydeps310
#                                                   (uv pip install --target ~/.cache/bumble-bee-gym/pydeps310
#                                                    --python-version 3.10 --python-platform x86_64-manylinux_2_28
#                                                    --no-deps casadi); writes <plan dir>/<NAME or gazebo_race_mpc>.*
# SLOW=1: run physics at a tenth of real time instead, so ROS delivery adds no delay (how it was first tested).
# VIDEO=<file.mp4>: film the flight from Gazebo's own window on a virtual display inside the container (installs xvfb
# and ffmpeg there), the camera following the drone; nothing on your desktop is touched.
# LOG=<dir>: also record a Gazebo state log of the flight into <dir>/state.log (under the gym repo), for replay
# filming from any camera afterwards.
# GUI=1: also open Gazebo's window on $DISPLAY (run `xhost +local:docker` once first), with the camera following the
# drone.
set -euo pipefail
GYM="$(cd "$(dirname "$0")/.." && pwd)"
PYDEPS="$HOME/.cache/bumble-bee-gym/pydeps310"
if [ "$1" = hover ]; then
  POLICY=hover; OUT="$GYM/outputs/timing"; mkdir -p "$OUT"; NAME="${NAME:-hover}"
elif [[ "$1" == mpc:* ]]; then
  PLAN="$(realpath "${1#mpc:}")"; POLICY="mpc:$PLAN"; OUT="$(dirname "$PLAN")"; NAME="${NAME:-gazebo_race_mpc}"
  [ -d "$PYDEPS/casadi" ] || { echo "CasADi for Python 3.10 missing in $PYDEPS (see the header)"; exit 1; }
else
  POLICY="$(realpath "$1")"; OUT="$(dirname "$POLICY")"
fi
SECONDS_="${2:-15}"
IMAGE="${BUMBLE_BEE_IMAGE:-bumble-bee}"
RTF_RATE=1000; [ "${SLOW:-0}" = 1 ] && RTF_RATE=100

read -r X Y Z YAW < <("$GYM/.venv/bin/python" -c "
from pathlib import Path
import numpy as np
from bumble_bee_gym import course, worlds
c = course.load('race_gates')
world, _ = worlds.write(c, Path('$GYM/generated'))
slow = world.read_text().replace('<real_time_update_rate>1000<', '<real_time_update_rate>${RTF_RATE}<')
if '${VIDEO:-}':   # chase camera: Gazebo's window follows the drone from just behind it
    slow = slow.replace('<view_controller>orbit</view_controller>', '<view_controller>orbit</view_controller>'
        '<track_visual><name>drone</name><static>true</static><use_model_frame>true</use_model_frame>'
        '<xyz>-1.2 0 0.45</xyz><inherit_yaw>true</inherit_yaw></track_visual>')
(Path('$GYM/generated') / 'race_gates_slow.world').write_text(slow)
g7, g0 = c.gates[-1].xyz, c.gates[0].xyz
p = g7 + 0.3 * (g0 - g7)
print(p[0], p[1], p[2], np.arctan2(*(g0 - p)[[1, 0]]))")

HEADLESS=true; GUI_ARGS=(); FOLLOW=""
if [ "${GUI:-0}" = 1 ]; then
  HEADLESS=false
  GUI_ARGS=(--gpus all -e DISPLAY="$DISPLAY" -e NVIDIA_DRIVER_CAPABILITIES=all -v /tmp/.X11-unix:/tmp/.X11-unix)
  FOLLOW="(sleep 8; gz camera -c gzclient_camera -f ${FOLLOW_MODEL:-drone}) &"
fi
REC=""; REC_START=""; STOP_REC=""
if [ -n "${VIDEO:-}" ]; then
  HEADLESS=true; GUI_WAIT=20
  VID="/gym/$(realpath --relative-to="$GYM" "$VIDEO")"
  REC="apt-get update -qq >/dev/null && apt-get install -y -qq xvfb ffmpeg >/dev/null 2>&1
  Xvfb :99 -screen 0 1920x1080x24 >/dev/null 2>&1 & sleep 2; export DISPLAY=:99
  mkdir -p ~/.gazebo; printf '[geometry]\\nx=0\\ny=0\\nwidth=1920\\nheight=1080\\n' > ~/.gazebo/gui.ini
  gzclient >/dev/null 2>&1 & sleep 10"
  REC_START="ffmpeg -loglevel error -y -f x11grab -framerate 30 -video_size 1920x1080 -i :99 -c:v libx264 -preset veryfast -pix_fmt yuv420p $VID &
  FFPID=\$!"
  STOP_REC="kill -INT \$FFPID; wait \$FFPID"
fi

if [ "$POLICY" = hover ]; then PILOT_ARG=hover
elif [[ "$POLICY" == mpc:* ]]; then PILOT_ARG="mpc:/gym/${PLAN#$GYM/}"
else PILOT_ARG="/gym/${POLICY#$GYM/}"; fi
docker run --rm --cpus="${GAZEBO_CPUS:-6}" "${GUI_ARGS[@]}" -v "$GYM":/gym -v "$PYDEPS":/pydeps:ro -e PYTHONPATH=/gym:/pydeps \
    -e BUMBLE_BEE_ROOT=/ws/src/drone_delivery --entrypoint bash "$IMAGE" -c "
  source /opt/ros/humble/setup.bash; source /usr/share/gazebo/setup.sh; source /ws/install/setup.bash
  ros2 launch drone_delivery gazebo.launch.py world:=/gym/generated/race_gates_slow.world headless:=$HEADLESS pause:=true \
      x:=$X y:=$Y z:=$Z yaw:=$YAW > /tmp/gazebo.log 2>&1 &
  until ros2 topic list 2>/dev/null | grep -q /drone_pose; do sleep 1; done
  $FOLLOW
  $REC
  sleep ${GUI_WAIT:-3}
  # Gazebo's window can start the physics running; put the drone back on the start line, paused, before flying.
  ros2 service call /pause_physics std_srvs/srv/Empty > /dev/null
  ros2 service call /reset_world std_srvs/srv/Empty > /dev/null
  $REC_START
  ${LOG:+gz log -d 1}
  python3 /gym/scripts/gazebo_race_pilot.py $PILOT_ARG /gym/${OUT#$GYM/}/${NAME:-gazebo_race} $SECONDS_ 2>&1 | tee /gym/${OUT#$GYM/}/${NAME:-gazebo_race}.log
  $STOP_REC
  ${LOG:+gz log -d 0; sleep 2; mkdir -p /gym/$LOG; cp \$(ls -t ~/.gazebo/log/*/gzserver/state.log | head -1) /gym/$LOG/state.log}
  chown -R $(id -u):$(id -g) /gym/${OUT#$GYM/} ${LOG:+/gym/$LOG}
"
