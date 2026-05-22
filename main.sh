#!/bin/bash
# Summary RAG 预测脚本
# 对标 hyperbolic_memory 的 scripts/main.sh 参数风格
#
# 提取模型（生成 session summary）和生成模型（QA 回答）完全分离。
# 支持 llama3.2 / qwen2.5 / deepseek 等通过 transformers 本地加载的模型。
# 嵌入模型与 hymemory 一致：sentence-transformers/all-mpnet-base-v2
#
# 用法:
#   bash main.sh
#   bash main.sh --top-k 10 --max-samples 3
#

source ~/miniconda3/etc/profile.d/conda.sh
conda activate memory
cd /share/home/leiyh5/Memory

python /path/to/locomo-bench/main.py \
  --data-file /share/home/leiyh5/Memory/data/locomo/locomo10.json \
  --out-file /share/home/leiyh5/Memory/data/locomo/locomo10_summary_rag_pred.json \
  --prediction-key summary_rag_prediction \
  --extraction-model-path /share/home/leiyh5/models/Qwen2.5-7B-Instruct \
  --extraction-device auto \
  --extraction-batch-size 8 \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --top-k 5 \
  --generation-model-path /share/home/leiyh5/models/Llama-3.2-3B-Instruct \
  --generation-device auto \
  --device auto \
  "$@"
