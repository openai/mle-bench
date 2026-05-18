#!/bin/bash

# ============================================
# 1. 参数接收
# ============================================
DATA_DIR="$1"
GPU_ID="$2"
LOCK_FILE_BUSY="$3" # 如果主脚本不传这个，下面归还逻辑会自动跳过
CPU_RANGE="$4" 

# 提取任务名：从路径中剥离出任务标识
TASK_NAME=$(basename "$(dirname "$(dirname "$DATA_DIR")")")

LOG_DIR="run_logs_v214"

mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/${TASK_NAME}_run_v214.log"
# ============================================
# 2. 资源隔离与环境变量
# ============================================
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# 动态计算核心数
if [[ "$CPU_RANGE" == *"-"* ]]; then
    IFS='-' read -r start end <<< "$CPU_RANGE"
    NUM_CORES=$((end - start + 1))
else
    NUM_CORES=1
fi

# 限制线程数，防止 OpenBLAS/MKL 竞争导致 CPU 效率下降
export OMP_NUM_THREADS=$(( NUM_CORES > 8 ? 8 : NUM_CORES ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
# 优化显存碎片
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1

# ============================================
# 3. 任务执行
# ============================================
{
    echo "============================================"
    echo ">>> [START] $(date)"
    echo ">>> [TASK]  $TASK_NAME"
    echo ">>> [GPU]   $GPU_ID"
    echo ">>> [CPU]   $CPU_RANGE (Cores: $NUM_CORES, Threads: $OMP_NUM_THREADS)"
    echo "============================================"
    
    # 使用 taskset 强制绑定 CPU 核心，防止 OS 调度导致的任务漂移
    # timeout 设置为 12 小时，防止单个任务卡死占坑
    taskset -c "$CPU_RANGE" timeout 86400 aide \
        data_dir="$DATA_DIR" \
        desc_file="$DATA_DIR/description.md" \
        exp_name="${TASK_NAME}" \
        cpu_number="${NUM_CORES}"  # 修改这里，使用脚本内计算好的 NUM_CORES
        
    EXIT_CODE=$?
    
    echo "============================================"
    echo ">>> [END]   $(date) | EXIT_CODE: $EXIT_CODE"
    echo "============================================"
} >> "$LOG_FILE" 2>&1

# ============================================
# 4. 归还资源 (双重保险)
# ============================================
# 如果 scheduler 还没回收，这里主动改名回 .free
if [ -n "$LOCK_FILE_BUSY" ] && [ -f "$LOCK_FILE_BUSY" ]; then
    mv "$LOCK_FILE_BUSY" "${LOCK_FILE_BUSY%.busy}.free" 2>/dev/null
fi

exit $EXIT_CODE