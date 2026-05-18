#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
# ============================================
# 1. 核心配置
# ============================================
DATA_ROOT="/inspire/ssd/project/sais-auto-scientist/public/mle-data/mle-bench/data"
AIDE_RUN_LOG_DIR="BaguetteSota/logs/run_v214"
GPU_IDS=(0)
MAX_PARALLEL=8 # 建议先从 8 开始
TOTAL_CPUS=80
CPUS_PER_TASK=$(( TOTAL_CPUS / MAX_PARALLEL ))

# ============================================
# 2. 任务发现
# ============================================
TARGET_TASKS=("siim-isic-melanoma-classification")

TASK_LIST=()

# for task_folder in "$DATA_ROOT"/*; do
#     [ ! -d "$task_folder" ] && continue
#     TASK_NAME=$(basename "$task_folder")
#     # 跳过已运行成功的
#     if [ -d "${AIDE_RUN_LOG_DIR}/${TASK_NAME}" ]; then continue; fi
    
#     TARGET_DIR="${task_folder}/prepared/public"
#     if [ -f "${TARGET_DIR}/description.md" ]; then
#         TASK_LIST+=("$TARGET_DIR")
#     fi  
# done
for TASK_NAME in "${TARGET_TASKS[@]}"; do
    task_folder="$DATA_ROOT/$TASK_NAME"
    
    [ ! -d "$task_folder" ] && continue
    # 跳过已运行成功的
    if [ -d "${AIDE_RUN_LOG_DIR}/${TASK_NAME}" ]; then continue; fi
    
    TARGET_DIR="${task_folder}/prepared/public"
    if [ -f "${TARGET_DIR}/description.md" ]; then
        TASK_LIST+=("$TARGET_DIR")
    fi  
done

TOTAL_TASKS=${#TASK_LIST[@]}
echo ">>> [SCHEDULER] 待处理任务: $TOTAL_TASKS | 并发: $MAX_PARALLEL | 每任务核心: $CPUS_PER_TASK"

# ============================================
# 3. 资源池初始化 (使用内存文件系统 /dev/shm 提升速度)
# ============================================
LOCK_DIR="/dev/shm/gpu_balancer_$(whoami)"
rm -rf "$LOCK_DIR" && mkdir -p "$LOCK_DIR"

# 计算每个 GPU 承载几个任务
NUM_GPUS=${#GPU_IDS[@]}
TASKS_PER_GPU=$((MAX_PARALLEL / NUM_GPUS))

for i in "${!GPU_IDS[@]}"; do
    g_id=${GPU_IDS[$i]}
    for ((t=0; t<TASKS_PER_GPU; t++)); do
        # 文件名格式: slot_[GPU索引]_[SLOT索引].free
        touch "${LOCK_DIR}/slot_${i}_${t}.free"
    done
done

cleanup() {
    echo -e "\n>>> 正在强行终止所有子任务并清理资源..."
    pkill -P $$
    rm -rf "$LOCK_DIR"
    exit 1
}
trap cleanup SIGINT SIGTERM

# ============================================
# 4. 负载均衡循环
# ============================================
current_idx=0
declare -A active_tasks  # 使用关联数组记录 PID 和对应的资源文件

while [ $current_idx -lt $TOTAL_TASKS ] || [ ${#active_tasks[@]} -gt 0 ]; do
    
    # --- A. 回收已完成的资源 ---
    for pid in "${!active_tasks[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            BUSY_TICKET=${active_tasks[$pid]}
            FREE_TICKET="${BUSY_TICKET%.busy}.free"
            mv "$BUSY_TICKET" "$FREE_TICKET" 2>/dev/null
            unset active_tasks[$pid]
            # echo "[RELEASE] PID $pid 释放了资源 $(basename $FREE_TICKET)"
        fi
    done

    # --- B. 启动新任务 ---
    while [ ${#active_tasks[@]} -lt $MAX_PARALLEL ] && [ $current_idx -lt $TOTAL_TASKS ]; do
        # 随机挑选一个空闲槽位
        FREE_TICKET=$(ls $LOCK_DIR/*.free 2>/dev/null | shuf -n 1)
        
        if [ -n "$FREE_TICKET" ]; then
            # 1. 解析资源 ID
            TICKET_BASE=$(basename "$FREE_TICKET")
            # slot_0_1.free -> GPU_IDX=0, S_ID=1
            GPU_IDX=$(echo "$TICKET_BASE" | cut -d'_' -f2)
            S_ID=$(echo "$TICKET_BASE" | cut -d'_' -f3 | cut -d'.' -f1)
            
            REAL_GPU_ID=${GPU_IDS[$GPU_IDX]}

            # 2. 计算精确的 CPU 绑定区间 (避免任务间 CPU 争抢)
            GLOBAL_SLOT_ID=$(( GPU_IDX * TASKS_PER_GPU + S_ID ))
            CPU_START=$(( GLOBAL_SLOT_ID * CPUS_PER_TASK ))
            CPU_END=$(( CPU_START + CPUS_PER_TASK - 1 ))
            CURRENT_CPU_RANGE="${CPU_START}-${CPU_END}"

            # 3. 锁定并执行
            BUSY_TICKET="${FREE_TICKET%.free}.busy"
            mv "$FREE_TICKET" "$BUSY_TICKET"
            
            TASK_DIR="${TASK_LIST[$current_idx]}"
            T_NAME=$(basename "$(dirname "$(dirname "$TASK_DIR")")")

            echo "[LAUNCH] $((current_idx+1))/$TOTAL_TASKS -> $T_NAME | GPU: $REAL_GPU_ID | CPUs: $CURRENT_CPU_RANGE"

            # 启动包装脚本
            ./task_wrapper_max.sh "$TASK_DIR" "$REAL_GPU_ID" "$BUSY_TICKET" "$CURRENT_CPU_RANGE" &
            
            # 记录任务信息
            child_pid=$!
            active_tasks[$child_pid]="$BUSY_TICKET"
            
            ((current_idx++))
            sleep 0.5 # 稍微错开启动时间，防止 IO 并发峰值
        else
            break # 暂无空闲槽位
        fi
    done
    
    # 打印状态 (可选)
    # echo -ne "Active Tasks: ${#active_tasks[@]} / $MAX_PARALLEL \r"
    sleep 2
done

rm -rf "$LOCK_DIR"
echo ">>> [FINISH] 所有任务执行完毕。"