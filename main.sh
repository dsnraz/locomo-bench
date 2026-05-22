#!/bin/bash
#SBATCH -p gpu_chen
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -o job_obs_rag.out
#
# Observation RAG 预测脚本
# 对标 hyperbolic_memory 的 scripts/main.sh 参数风格
#
# 提取模型（事实提取）和生成模型（QA 回答）完全分离。
# 支持 llama3.2 / qwen2.5 / deepseek。
# 嵌入模型与 hymemory 一致：sentence-transformers/all-mpnet-base-v2
#

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bank
cd /share/home/leiyh5/locomo-bench

python main.py \
  --data-file /share/home/leiyh5/Memory/data/locomo/locomo10.json \
  --out-file /share/home/leiyh5/locomo-bench/data/locomo10_obs_rag_pred.json \
  --prediction-key observation_rag_prediction \
  --extraction-model-path /share/home/leiyh5/models/Qwen2.5-7B-Instruct \
  --generation-model-path /share/home/leiyh5/models/Llama-3.2-3B-Instruct \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --rag-mode observation \
  --top-k 10 \
  --emb-dir /tmp/locomo_emb \
  --device auto \
  "$@"
