#!/bin/bash
set -x # Print commands and their arguments as they are executed

# the same dataset_dir with run.sh
data_dir="/inspire/ssd/project/sais-auto-scientist/public/mle-data/mle-bench/data/"

# launch a server which tells agent whether the submission is valid or not, allowed by MLE-Bench rules
nohup python -u grade_server.py \
  data_dir="${data_dir}" \
  desc_file="none" > grade_server.out 2>&1 &