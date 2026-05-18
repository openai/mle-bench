#!/bin/bash

mlebench_dir=$HOME/repos/mle-bench # adjust path accordingly
log_dir=$mlebench_dir/runs/2024-06-27_16-40-46

example_dir=$mlebench_dir/examples/log_analysis
logs_file=$example_dir/logs.jsonl

python $example_dir/make_logs_file.py \
  --runs-dir $log_dir \
  --output $logs_file

python extras/rule_violation_detector/run.py --content-type logs --comps-file $logs_file --output-dir $example_dir
