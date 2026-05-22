#!/bin/bash
# Summary RAG 完整 pipeline：预测 → 评估
# 对标 hyperbolic_memory 的 scripts/main.sh + scripts/eval/eval.sh 参数风格
#
# 提取模型（生成 session summary）和生成模型（QA 回答）完全分离。
# 支持 llama3.2 / qwen2.5 / deepseek 等通过 transformers 本地加载的模型。
# 嵌入模型与 hymemory 一致：sentence-transformers/all-mpnet-base-v2
# 评估直接复用 hyperbolic_memory 的 evaluate_locomo_predictions.py
#
# 用法:
#   bash run.sh                           # 完整 pipeline
#   bash run.sh --top-k 10 --max-samples 3
#   bash run.sh --predict-only            # 仅预测
#   bash run.sh --eval-only               # 仅评估
#

source ~/miniconda3/etc/profile.d/conda.sh
conda activate memory
cd /share/home/leiyh5/Memory

PRED_FILE=/share/home/leiyh5/Memory/data/locomo/locomo10_summary_rag_pred.json
ANNFILE=/share/home/leiyh5/Memory/data/locomo/locomo10.json
PREDKEY=summary_rag_prediction
MODELKEY=summary_rag

# ─── 预测 ────────────────────────────────────────────────────────────────
if [[ "${*}" != *"--eval-only"* ]]; then
  echo "=== Step 1/2: 预测 ==="
  python d:/project/summary_rag/main.py \
    --data-file "$ANNFILE" \
    --out-file "$PRED_FILE" \
    --prediction-key "$PREDKEY" \
    --extraction-model-path /share/home/leiyh5/models/Qwen2.5-7B-Instruct \
    --extraction-device auto \
    --extraction-batch-size 8 \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --top-k 5 \
    --generation-model-path /share/home/leiyh5/models/Llama-3.2-3B-Instruct \
    --generation-device auto \
    --device auto \
    "$@"
  echo "预测完成: $PRED_FILE"
fi

# ─── 评估（复用 hyperbolic_memory 的 eval 脚本）─────────────────────────
if [[ "${*}" != *"--predict-only"* ]]; then
  echo "=== Step 2/2: 评估（调用 hyperbolic_memory eval）==="
  python -m scripts.eval.evaluate_locomo_predictions \
    --ann-file "$ANNFILE" \
    --pred-file "$PRED_FILE" \
    --prediction-key "$PREDKEY" \
    --model-key "$MODELKEY" \
    --locomo-root /share/home/leiyh5/locomo \
    --scored-file /share/home/leiyh5/Memory/data/locomo/locomo10_summary_rag_scored.json \
    --stats-file /share/home/leiyh5/Memory/data/locomo/locomo10_summary_rag_stats.json \
    "$@"
fi
