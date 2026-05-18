#!/bin/bash
set -x # Print commands and their arguments as they are executed

# These will be provided by the environment
INSTRUCTIONS_FILE="/home/instructions.txt"
VALIDATION_SCRIPT="/home/validate_submission.sh"
DATA_DIR="/home/data"

eval "$(conda shell.bash hook)" # make conda available to the shell
conda activate agent

# --------- Prep some variables

# determine hardware available
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed 's/^[ \t]*//' \
    | sed 's/[ \t]*$//' \
    | sort \
    | uniq -c \
    | sed 's/^ *\([0-9]*\) *\(.*\)$/\1 \2/' \
    | paste -sd ', ' -)
else
  HARDWARE="a CPU"
fi
export HARDWARE

# convert $TIME_LIMIT to more readable format for additional_notes
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)

# --------- Environment checks

# Confirm that we don't have access to this
ls /private  # Expect "permission denied"
# Data should be mounted correctly
ls /home/data  # Expect to see the data files
# Confirm that validation tool works
bash /home/validate_submission.sh /home/data/sample_submission.csv  # Expect "Submission is valid."

# --------- Check that we can use the GPU in PyTorch and TensorFlow

python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')"
python -c "import tensorflow as tf; print('GPUs Available: ', tf.config.list_physical_devices('GPU'))"

# --------- Construct instructions for agent (this is the MLAB "research problem")

# Update instructions for agent-specific details: replace `/home/` paths to make paths relative
# (since the agent will have its own copies of these files in its workspace).
# e.g. /home/submission/submission.csv -> submission/submission.csv
sed -i 's|/home/||g' $INSTRUCTIONS_FILE
# further, add a linebreak and add the additional_notes (after substituting in env variables)
echo "" >> $INSTRUCTIONS_FILE
envsubst < /home/additional_notes.txt >> $INSTRUCTIONS_FILE

# --------- Prepare necessary files for the agent workspace
task="mlebench"
mlab_env_dir="$AGENT_DIR/MLAgentBench/MLAgentBench/benchmarks/$task/env"  # Everything we want to give to the agent, put here
workdir="workdir"  # contents of $mlab_env_dir later get copied by MLAB into this workdir where the agent operates
logdir="$AGENT_DIR/logs" && mkdir -p $logdir

# Agent will write submissions to $mlab_env_dir/submission, we will extract and grade from $SUBMISSION_DIR
ln -s $SUBMISSION_DIR $mlab_env_dir/submission
# Agent will generate a bunch of files under $logdir, we only want $logdir/agent_log so we link that to $LOGS_DIR which is what gets exported
ln -s $LOGS_DIR $logdir/agent_log
# DATA_DIR is a mounted volume prepared with the data, agent will interact with it at $mlab_env_dir/data
ln -s $DATA_DIR $mlab_env_dir/data
# Populate research_problem.txt with the updated instructions.txt, this is given to the agent via a prompt
cp $INSTRUCTIONS_FILE $AGENT_DIR/MLAgentBench/MLAgentBench/benchmarks/$task/scripts/research_problem.txt
# Agent will receive instructions via the `research_problem.txt` prompt, but we also make a copy available to the agent
cp $INSTRUCTIONS_FILE $mlab_env_dir/instructions.txt
# Give the agent access to the validation tool
cp $VALIDATION_SCRIPT $mlab_env_dir/validate_submission.sh

# --------- MLAB steps

# Not much left to do, but need to run this to initialize $workdir with the contents of $mlab_env_dir
python -u -m MLAgentBench.prepare_task $task $(which python)

# Run the agent with a timeout
timeout $TIME_LIMIT_SECS python -u -m MLAgentBench.runner \
    --python $(which python) \
    --task $task \
    --device 0 \
    --log-dir $logdir \
    --work-dir $workdir \
    $@ # forward the kwargs from config to MLAB

# Copy agent code out to $CODE_DIR
# find all files that are less than 10MB, copy while preserving directory structure
find "$workdir" -type f -size -10M -exec cp --parents {} "$CODE_DIR" \;