#!/bin/bash
# The FIXED run contract. Identical on every node in the tree — the only thing
# that differs between arms is experiments/config.json on that node's branch.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/train_arm.py experiments/config.json
