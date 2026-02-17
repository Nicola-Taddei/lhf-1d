#!/usr/bin/env bash

# Usage:
#   ./kill_sweep.sh USER/PROJECT/SWEEP_ID
# Example:
#   ./kill_sweep.sh USER/PROJECT/abc123

set -e

SWEEP_ID=$1

if [ -z "$SWEEP_ID" ]; then
  echo "Usage: ./kill_sweep.sh <sweep_id>"
  exit 1
fi

echo "Looking for agents running sweep: $SWEEP_ID"
echo

MATCHING=$(pgrep -af "wandb agent $SWEEP_ID" || true)

if [ -z "$MATCHING" ]; then
  echo "No matching agents found."
else
  echo "$MATCHING"
  echo
  read -p "Kill these processes? (y/n): " confirm

  if [[ "$confirm" == "y" ]]; then
    pkill -f "wandb agent $SWEEP_ID"
    echo "Agents terminated."
  else
    echo "Aborted."
    exit 0
  fi
fi

echo
echo "Looking for agent log files..."

LOGS=$(ls agent_*.log 2>/dev/null || true)

if [ -z "$LOGS" ]; then
  echo "No agent log files found."
else
  echo "$LOGS"
  echo
  read -p "Delete these log files? (y/n): " confirm_logs

  if [[ "$confirm_logs" == "y" ]]; then
    rm agent_*.log
    echo "Log files removed."
  else
    echo "Log file deletion skipped."
  fi
fi

# Optional: remove PID tracking file if present
if [ -f agent_pids.txt ]; then
  rm agent_pids.txt
  echo "Removed agent_pids.txt"
fi

echo
echo "Cleanup complete."
