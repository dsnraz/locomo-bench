#!/bin/bash
#SBATCH -p gpu_chen
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -o job_obs_rag_eval.out
#
# Observation RAG 评估脚本
# 对标 hyperbolic_memory 的 scripts/eval/eval.sh 参数风格
#

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bank
cd /share/home/leiyh5/locomo-bench

python /share/home/leiyh5/Memory/scripts/eval/evaluate_locomo_predictions.py \
  --ann-file /share/home/leiyh5/Memory/data/locomo/locomo10.json \
  --pred-file /share/home/leiyh5/locomo-bench/data/locomo10_obs_rag_pred.json \
  --prediction-key observation_rag_prediction \
  --model-key observation_rag \
  --locomo-root /share/home/leiyh5/locomo-bench \
  --scored-file /share/home/leiyh5/locomo-bench/data/locomo10_obs_rag_scored.json \
  --stats-file /share/home/leiyh5/locomo-bench/data/locomo10_obs_rag_stats.json \
  "$@"
