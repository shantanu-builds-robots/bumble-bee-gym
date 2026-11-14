#!/usr/bin/env bash
# Run a learn/ module in sprout-gym's GPU image (JAX + MuJoCo Playground + Brax), in the background.
#
#   bumble_bee_gym/learn/run.sh <name> train [train.py args...]     -> outputs/<name>/train.log
#   bumble_bee_gym/learn/run.sh <name> play  [play.py args...]      -> outputs/<name>/play.log
#
# Image: sprout-playground (build it from sprout-gym: docker build -f docker/Dockerfile.playground -t sprout-playground .)
# Container: bee-<name>-<module>. bumble-bee is mounted read-only for the drone's URDF.
# JAX may use at most XLA_MEM_FRACTION of GPU memory (default 0.35, so it can share the GPU with another run), and
# allocates on demand rather than all at once.
set -euo pipefail
name="$1"; mod="$2"; shift 2
repo="$(cd "$(dirname "$0")/../.." && pwd)"
bee="${BUMBLE_BEE_ROOT:-$repo/../bumble-bee}"
mkdir -p "$repo/outputs/$name" "$HOME/.cache/bumble-bee-gym"
docker rm -f "bee-$name-$mod" >/dev/null 2>&1 || true
docker run -d --name "bee-$name-$mod" --gpus all \
    -v "$repo":/app -v "$(cd "$bee" && pwd)":/bumble-bee:ro -v "$HOME/.cache/bumble-bee-gym":/cache \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp -e BUMBLE_BEE_ROOT=/bumble-bee \
    -e JAX_COMPILATION_CACHE_DIR=/cache/jax -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.35}" \
    sprout-playground bash -c "python -m bumble_bee_gym.learn.$mod --name $name $* > /app/outputs/$name/$mod.log 2>&1"
echo "started bee-$name-$mod -> outputs/$name/$mod.log"
