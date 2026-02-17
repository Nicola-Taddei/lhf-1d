#!/usr/bin/env bash

# Usage:
#   ./launch_agents.sh N SWEEP_ID
# Example:
#   ./launch_agents.sh 4 USER/PROJECT/abc123

set -e  # fail on error

N=$1
SWEEP_ID=$2

if [ -z "$N" ] || [ -z "$SWEEP_ID" ]; then
  echo "Usage: ./launch_agents.sh <num_agents> <sweep_id>"
  exit 1
fi

# If using conda, uncomment and adjust:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv

echo "Launching $N agents for sweep $SWEEP_ID"

for i in $(seq 1 $N)
do
  nohup wandb agent "$SWEEP_ID" > "agent_${i}.log" 2>&1 &
  echo "Started agent $i (PID $!)"
done

echo "All agents launched."
