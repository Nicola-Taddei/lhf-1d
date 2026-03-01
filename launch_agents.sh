#!/usr/bin/env bash

# Usage:
#   ./launch_agents.sh N SWEEP_ID GPU_LIST CPU_BLOCKS
#
# Example:
#   ./launch_agents.sh 4 USER/PROJECT/abc123 "0,1,2,3" "0-7;8-15;16-23;24-31"
#
# Notes:
#   - GPU_LIST: comma-separated GPU ids
#   - CPU_BLOCKS: semicolon-separated CPU core ranges (one per agent)
#   - N must match number of GPUs and CPU blocks provided

set -e  # fail on error

N=$1
SWEEP_ID=$2
GPU_LIST=$3
CPU_BLOCKS=$4

if [ -z "$N" ] || [ -z "$SWEEP_ID" ] || [ -z "$GPU_LIST" ] || [ -z "$CPU_BLOCKS" ]; then
  echo "Usage: ./launch_agents.sh <num_agents> <sweep_id> <gpu_list> <cpu_blocks>"
  echo "Example:"
  echo "  ./launch_agents.sh 4 USER/PROJECT/abc123 \"0,1,2,3\" \"0-7;8-15;16-23;24-31\""
  exit 1
fi

# Convert comma-separated GPUs into array
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

# Convert semicolon-separated CPU blocks into array
IFS=';' read -r -a CPUS <<< "$CPU_BLOCKS"

# Validation
if [ "${#GPUS[@]}" -ne "$N" ]; then
  echo "Error: Number of GPUs (${#GPUS[@]}) does not match N ($N)"
  exit 1
fi

if [ "${#CPUS[@]}" -ne "$N" ]; then
  echo "Error: Number of CPU blocks (${#CPUS[@]}) does not match N ($N)"
  exit 1
fi

echo "Launching $N agents for sweep $SWEEP_ID"

for i in $(seq 0 $((N-1)))
do
  GPU_ID=${GPUS[$i]}
  CPU_BLOCK=${CPUS[$i]}

  echo "Starting agent $((i+1)) on GPU $GPU_ID using CPUs $CPU_BLOCK"

  CUDA_VISIBLE_DEVICES=$GPU_ID \
  taskset -c $CPU_BLOCK \
  nohup wandb agent "$SWEEP_ID" \
  > "agent_${i}_gpu${GPU_ID}.log" 2>&1 &

  echo "  -> PID $!"
done

echo "All agents launched."