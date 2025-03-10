#!/bin/bash
# scripts/run_experiment.sh
# 说明：此脚本用于在服务器上启动实验，日志将保存在指定的输出目录中。

# 激活虚拟环境（如果需要）
# source /path/to/your/venv/bin/activate

# 设置使用的 GPU 设备（例如使用第0号GPU）
export CUDA_VISIBLE_DEVICES=0

# 从配置文件中输出目录与脚本中保持一致
OUTPUT_DIR="./outputs/experiment1"
mkdir -p ${OUTPUT_DIR}

# 运行实验，日志输出到OUTPUT_DIR下的log.txt文件
python experiments/run_experiment.py > ${OUTPUT_DIR}/log.txt 2>&1

# 如果希望后台运行实验，可以使用以下命令（去掉注释）：
# nohup python experiments/run_experiment.py > ${OUTPUT_DIR}/log.txt 2>&1 &