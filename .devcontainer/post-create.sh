#!/usr/bin/env bash
# Runs once after the container is created. Must stay idempotent.
set -uo pipefail
if [ -d /home/node/.ssh-host ]; then
  mkdir -p /home/node/.ssh
  cp -r /home/node/.ssh-host/. /home/node/.ssh/ 2>/dev/null || true
  chmod 700 /home/node/.ssh; chmod 600 /home/node/.ssh/id_* 2>/dev/null || true
fi
sudo uv pip install --system -e "/workspace[dev]" || echo "post-create: WARNING - install failed."
sudo chown -R node:node /workspace/*.egg-info /workspace/src/*.egg-info 2>/dev/null || true
