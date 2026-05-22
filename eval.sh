#!/bin/bash
# Observation RAG 评估脚本
# 对标 hyperbolic_memory 的 scripts/eval/eval.sh 参数风格
#
# 用法:
#   bash eval.sh
#   bash eval.sh --pred-file /path/to/other_pred.json --prediction-key my_key
#

source ~/miniconda3/etc/profile.d/conda.sh
conda activate memory
cd /share/home/leiyh5/Memory

ANNFILE=/share/home/leiyh5/Memory/data/locomo/locomo10.json
PREDFILE=/share/home/leiyh5/Memory/data/locomo/locomo10_obs_rag_pred.json
PREDKEY=observation_rag_prediction
MODELKEY=observation_rag

python /path/to/locomo-bench/scripts/eval/evaluate_locomo_predictions.py \
  --ann-file "$ANNFILE" \
  --pred-file "$PREDFILE" \
  --prediction-key "$PREDKEY" \
  --model-key "$MODELKEY" \
  --locomo-root /path/to/locomo-bench \
  --scored-file /share/home/leiyh5/Memory/data/locomo/locomo10_obs_rag_scored.json \
  --stats-file /share/home/leiyh5/Memory/data/locomo/locomo10_obs_rag_stats.json \
  "$@"
