#!/bin/bash

export SUBMISSION_DIR="/home/submission"
export LOGS_DIR="/home/logs"
export CODE_DIR="/home/code"
export AGENT_DIR="/home/agent"

# export OPENAI_API_KEY="xxx"
export AZURE_OPENAI_ENDPOINT="xxx"
export OPENAI_API_VERSION="xxx"
export CHAT_MODEL="xxx"
export AZURE_OPENAI_AD_TOKEN="xxx"

export MLE_BENCH_ABSOLUTE_PATH="xxxx"

nohup /home/v-yuanteli/miniconda3/envs/mle/bin/python /data/userdata/v-yuanteli/mle-bench/run_agent.py \
    --agent-id=aide \
    --competition-set=experiments/exp1.txt \
    > output2.log 2>&1 &
