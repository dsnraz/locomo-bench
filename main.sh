#!/bin/bash
#SBATCH -p gpu_chen
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -o job_obs_rag.out
#
# Observation RAG 预测脚本 — DeepSeek API 生成
# 对标 hyperbolic_memory 的 scripts/main.sh 参数风格
#
# 提取: Qwen2.5-7B 本地 | 嵌入: all-mpnet-base-v2 本地 | 生成: DeepSeek API
# 使用前设置: export OPENAI_API_KEY="sk-xxx"
#

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bank
cd /share/home/leiyh5/locomo-bench

python main.py \
  --data-file /share/home/leiyh5/Memory/data/locomo/locomo10.json \
  --out-file /share/home/leiyh5/locomo-bench/data/locomo10_obs_rag_pred_deepseek.json \
  --prediction-key observation_rag_prediction \
  --extraction-model-path /share/home/leiyh5/models/Qwen2.5-7B-Instruct \
  --generation-handler-type openai \
  --generation-api-base https://api.deepseek.com \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --rag-mode observation \
  --top-k 10 \
  --emb-dir /tmp/locomo_emb \
  --device auto \
  "$@"
