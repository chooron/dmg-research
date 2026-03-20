#!/bin/bash
# Run Causal-dPL group-holdout cross-validation (并行限制为2个任务)
# Usage: bash scripts/run_causal_dhbv.sh [holdout_group]

set -e

CONFIG=conf/config_vrex_dhbv.yaml
SCRIPT=train_causal_dpl.py

# 确保在项目根目录运行
cd "$(dirname "$0")/.."

# 定义运行单个任务的函数（不带 &，由外部控制是否后台）
run_single() {
    local g=$1
    echo "========================================"
    echo " Starting Holdout Group ${g}..."
    echo "========================================"
    # 建议将输出重定向到文件，否则两个并行任务的日志会交织在一起
    python ${SCRIPT} \
        --config ${CONFIG} \
        --holdout ${g} \
        --mode train_test > "log_group_${g}.txt" 2>&1
}

if [ -n "$1" ]; then
    # 如果指定了单个 group，则直接前台运行
    run_single "$1"
else
    # 并行运行逻辑：每次运行两个，分两批完成 (1,2 一组; 3,4 一组)
    
    # 第一批：Group 1 和 2
    run_single 1 &
    pid1=$!
    echo "Launched Group 1 (PID: $pid1), waiting 10s..."
    sleep 10
    
    run_single 2 &
    pid2=$!
    echo "Launched Group 2 (PID: $pid2). Waiting for batch 1 to finish..."
    
    wait $pid1 $pid2  # 等待前两个任务结束
    echo "Batch 1 (Group 1 & 2) complete."

    # 第二批：Group 3 和 4
    run_single 3 &
    pid3=$!
    echo "Launched Group 3 (PID: $pid3), waiting 10s..."
    sleep 10
    
    run_single 4 &
    pid4=$!
    echo "Launched Group 4 (PID: $pid4). Waiting for batch 2 to finish..."
    
    wait $pid3 $pid4  # 等待后两个任务结束
    echo "Batch 2 (Group 3 & 4) complete."

    echo "All runs complete. Shutting down in 1 minute..."
    
    # 执行关机命令
    # 注意：通常需要 root 权限，如果是普通用户可能需要 sudo
    # sudo /usr/bin/shutdown -h now
fi