#!/bin/bash
# scripts/run_experiment.sh
# This script loops over 20 values of d in log scale [10..10000],
# modifies config.yaml, runs the experiment, and records the test accuracy.

# GPU device selection (e.g., GPU 0 or 7, etc.)
export CUDA_VISIBLE_DEVICES=0

# Output directory (ensure it matches your config's output_dir)
OUTPUT_DIR="./outputs/experiment1"
mkdir -p "${OUTPUT_DIR}"

# Generate 20 log-spaced integers between 10 and 10000
D_VALUES=($(python3 -c "import numpy as np; vals = np.logspace(np.log10(10), np.log10(10000), 20); print(' '.join(map(str, map(int, vals))))"))

# We'll store (d, test_accuracy) pairs here
RESULTS_CSV="${OUTPUT_DIR}/results_vs_d.csv"
echo "d,test_accuracy" > "$RESULTS_CSV"  # overwrite or create new

for d in "${D_VALUES[@]}"; do
  echo "====================================="
  echo "Running experiment with d=$d"
  echo "====================================="

  # 1) Modify the config file to set 'global_intrinsic_dimension: d'
  #    We'll assume your config has a line like:
  #    global_intrinsic_dimension: 1900
  #    and we want to replace that with the new d.
  sed -i "s/^  global_intrinsic_dimension:.*/  global_intrinsic_dimension: $d/" ./configs/config.yaml

  # 2) Run the experiment, saving log to a file
  LOGFILE="${OUTPUT_DIR}/log_d_${d}.txt"
  python3 -m experiments.run_experiment > "$LOGFILE" 2>&1

  # 3) Parse the test accuracy from the resulting YAML (results.yaml)
  #    We assume the final test metrics are stored at
  #    outputs/experiment1/results.yaml with a structure like:
  #    test:
  #      accuracy: 0.873
  #      ...
  #
  #    We'll use a small Python snippet to extract test['accuracy'].
  ACC=$(python3 -c "
import yaml
with open('${OUTPUT_DIR}/results.yaml','r') as f:
    data = yaml.safe_load(f)
test_acc = data.get('test', {}).get('eval_accuracy', None)
if test_acc is None:
    print('NaN')
else:
    print(test_acc)
")

  # 4) Append (d, test_accuracy) to the CSV
  echo "${d},${ACC}" >> "$RESULTS_CSV"

done

echo "====================================="
echo "Done! Results stored in $RESULTS_CSV"
echo "You can now plot d vs. test_accuracy."
echo "====================================="