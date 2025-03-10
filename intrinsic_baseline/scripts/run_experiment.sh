#!/bin/bash
# scripts/run_experiment.sh
# This script is used to start the experiment on the server.
# Logs will be saved in the specified output directory.

# Activate virtual environment if needed
# Uncomment and modify the following line if you use a virtual environment.
# source /path/to/your/venv/bin/activate

# Set the GPU device (e.g., use GPU 0)
export CUDA_VISIBLE_DEVICES=3

# Define the output directory (make sure this matches the output_dir in your config)
OUTPUT_DIR="./outputs/experiment1"
mkdir -p "${OUTPUT_DIR}"

# Run the experiment and redirect both stdout and stderr to log.txt in the output directory
python experiments/run_experiment.py > "${OUTPUT_DIR}/log.txt" 2>&1

# To run the experiment in the background, uncomment the line below:
# nohup python experiments/run_experiment.py > "${OUTPUT_DIR}/log.txt" 2>&1 &