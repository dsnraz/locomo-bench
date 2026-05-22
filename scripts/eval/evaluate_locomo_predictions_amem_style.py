"""
对 LoCoMo 预测 JSON 做「A-mem 论文脚本风格」的分数：集合词重叠 F1 + BLEU-1。

- F1：逻辑来自 A-mem `utils.py`（`simple_tokenize` + 唯一词集合上的 precision/recall/F1），
  与 LoCoMo 官方 `eval_question_answering` 不是同一指标。
- BLEU-1：与 `evaluate_locomo_predictions.py` 中实现一致（NLTK sentence_bleu + method1），
  便于与 A-mem 侧 BLEU 口径对齐。

不依赖 LoCoMo 仓库与 `task_eval`，也不 import A-mem。
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


# --- A-mem/utils.py: simple_tokenize（原样复制）---


def _simple_tokenize_amem(text: Any) -> list[str]:
    """A-mem `simple_tokenize`：小写后替换标点为空格再 split。"""
    text = str(text)
    return text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def _calculate_amem_style_f1(prediction: Any, reference: Any) -> float:
    """
    A-mem `calculate_metrics` 中的 token F1：pred/ref 先做成词集合，再算 overlap F1。
    """
    if not prediction or not reference:
        return 0.0
    prediction = str(prediction).strip()
    reference = str(reference).strip()
    pred_tokens = set(_simple_tokenize_amem(prediction))
    ref_tokens = set(_simple_tokenize_amem(reference))
    common_tokens = pred_tokens & ref_tokens
    if not pred_tokens or not ref_tokens:
        return 0.0
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# --- evaluate_locomo_predictions.py: BLEU-1（原样复制）---


def _calculate_bleu1_score(prediction: Any, reference: Any) -> float:
    pred_tokens = nltk.word_tokenize(str(prediction).lower())
    ref_tokens = [nltk.word_tokenize(str(reference).lower())]
    if not pred_tokens or not ref_tokens[0]:
        return 0.0

    smooth = SmoothingFunction().method1
    try:
        return float(
            sentence_bleu(
                ref_tokens,
                pred_tokens,
                weights=(1.0, 0.0, 0.0, 0.0),
                smoothing_function=smooth,
            )
        )
    except Exception:
        return 0.0


# --- 与 A-mem load_dataset.QA.final_answer 一致：cat5 用 adversarial_answer ---


def _reference_text_for_amem(qa: dict[str, Any]) -> str:
    cat = int(qa.get("category", 0))
    if cat == 5:
        adv = qa.get("adversarial_answer")
        if adv is not None and str(adv).strip():
            return str(adv).strip()
    ans = qa.get("answer")
    return "" if ans is None else str(ans).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoCoMo 预测：A-mem 风格 F1 + BLEU-1（与 evaluate_locomo_predictions 中 BLEU 实现一致）。"
    )
    p.add_argument(
        "--ann-file",
        type=str,
        default="/share/home/leiyh5/Memory/data/locomo/locomo_qa_test.json",
        help="带标准答案的标注 JSON（用于缺失字段回填）。",
    )
    p.add_argument(
        "--pred-file",
        type=str,
        default="/share/home/leiyh5/Memory/data/locomo/locomo_qa_test_pred_category.json",
        help="推理输出的预测 JSON。",
    )
    p.add_argument(
        "--prediction-key",
        type=str,
        default="memory_prediction",
        help="每条 qa 上存放模型预测的字段名。",
    )
    p.add_argument(
        "--model-key",
        type=str,
        default="memory",
        help="写入字段前缀，例如 memory -> memory_amem_f1, memory_amem_bleu1。",
    )
    p.add_argument(
        "--scored-file",
        type=str,
        default="/share/home/leiyh5/Memory/data/locomo/locomo_qa_test_pred_amem_style_scored.json",
        help="写入逐题分数后的 JSON。",
    )
    p.add_argument(
        "--stats-file",
        type=str,
        default="/share/home/leiyh5/Memory/data/locomo/locomo_qa_test_pred_amem_style_stats.json",
        help="汇总统计 JSON（overall + 按 category）。",
    )
    return p.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _print_category_metric_stats(
    samples: list[dict[str, Any]], metric_key: str, label: str
) -> None:
    category_names = {
        1: "multi-hop retrieval",
        2: "temporal reasoning",
        3: "open-domain knowledge",
        4: "single-hop retrieval",
        5: "adversarial",
    }
    score_sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)

    for sample in samples:
        for qa in sample.get("qa", []):
            category = qa.get("category")
            try:
                category_int = int(category)
            except (TypeError, ValueError):
                continue
            if category_int not in category_names:
                continue
            counts[category_int] += 1
            score_sums[category_int] += float(qa.get(metric_key, 0.0))

    print(f"\nCategory {label} stats (ordered 1 -> 5):")
    total_count = 0
    total_score = 0.0
    for category in (1, 2, 3, 4, 5):
        count = counts[category]
        score_sum = score_sums[category]
        avg = (score_sum / count) if count > 0 else 0.0
        total_count += count
        total_score += score_sum
        print(
            f"  {category}. {category_names[category]}: "
            f"count={count}, avg_{label}={avg:.3f}"
        )
    overall = (total_score / total_count) if total_count > 0 else 0.0
    print(f"  overall: count={total_count}, avg_{label}={overall:.3f}")


def _build_aggregate_stats(
    samples: list[dict[str, Any]],
    f1_key: str,
    bleu_key: str,
    model_key: str,
    prediction_key: str,
) -> dict[str, Any]:
    """overall + 按 category 的 mean/std/count。"""
    f1_all: list[float] = []
    bleu_all: list[float] = []
    by_cat: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"amem_f1": [], "bleu1": []}
    )

    for sample in samples:
        for qa in sample.get("qa", []):
            try:
                c = int(qa["category"])
            except (TypeError, ValueError, KeyError):
                continue
            fv = float(qa.get(f1_key, 0.0))
            bv = float(qa.get(bleu_key, 0.0))
            f1_all.append(fv)
            bleu_all.append(bv)
            by_cat[c]["amem_f1"].append(fv)
            by_cat[c]["bleu1"].append(bv)

    def _agg(vals: list[float]) -> dict[str, float | int]:
        if not vals:
            return {"mean": 0.0, "std": 0.0, "count": 0}
        return {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "count": len(vals),
        }

    out: dict[str, Any] = {
        "description": "A-mem style set-F1 + BLEU-1 (NLTK, same as evaluate_locomo_predictions)",
        "model_key": model_key,
        "prediction_key": prediction_key,
        "overall": {
            "amem_f1": _agg(f1_all),
            "bleu1": _agg(bleu_all),
        },
        "by_category": {},
    }
    for c in sorted(by_cat.keys()):
        out["by_category"][str(c)] = {
            "amem_f1": _agg(by_cat[c]["amem_f1"]),
            "bleu1": _agg(by_cat[c]["bleu1"]),
        }
    return out


def main() -> None:
    args = parse_args()
    ann_path = Path(args.ann_file)
    pred_path = Path(args.pred_file)
    scored_path = Path(args.scored_file)
    stats_path = Path(args.stats_file)

    if not ann_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    if not pred_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    prediction_key = args.prediction_key
    model_key = args.model_key
    f1_key = f"{model_key}_amem_f1"
    bleu_key = f"{model_key}_amem_bleu1"

    samples = _load_json(pred_path)
    ann_samples = _load_json(ann_path)
    if not isinstance(samples, list):
        raise ValueError("Prediction file must be a JSON list of samples.")
    if not isinstance(ann_samples, list):
        raise ValueError("Annotation file must be a JSON list of samples.")

    ann_by_id: dict[str, dict[str, Any]] = {}
    for ann_sample in ann_samples:
        sid = str(ann_sample.get("sample_id", ""))
        if sid:
            ann_by_id[sid] = ann_sample

    total_qa = 0
    missing_pred = 0
    filled_required = 0
    filled_from_adversarial = 0
    for sample in samples:
        sid = str(sample.get("sample_id", ""))
        qas = sample.get("qa", [])
        total_qa += len(qas)
        ann_qas = ann_by_id.get(sid, {}).get("qa", []) if sid else []
        for i, qa in enumerate(qas):
            if "answer" not in qa and "adversarial_answer" in qa:
                qa["answer"] = (
                    ann_qas[i].get("adversarial_answer", qa["adversarial_answer"])
                    if i < len(ann_qas)
                    else qa["adversarial_answer"]
                )
                filled_from_adversarial += 1

            if i < len(ann_qas):
                ann_qa = ann_qas[i]
                for required_key in ("answer", "category", "evidence"):
                    if required_key not in qa:
                        if required_key == "answer" and "adversarial_answer" in ann_qa:
                            qa[required_key] = ann_qa["adversarial_answer"]
                        else:
                            qa[required_key] = ann_qa.get(required_key)
                        filled_required += 1
                # cat 5 参考答案：与 A-mem `final_answer` 一致，需要 adversarial_answer
                if int(qa.get("category", 0)) == 5 and (
                    "adversarial_answer" not in qa or qa.get("adversarial_answer") in (None, "")
                ):
                    if ann_qa.get("adversarial_answer") is not None:
                        qa["adversarial_answer"] = ann_qa["adversarial_answer"]
                        filled_required += 1

            if prediction_key not in qa:
                missing_pred += 1
                qa[prediction_key] = ""

    if missing_pred > 0:
        print(f"[warning] {missing_pred} QA items missing `{prediction_key}`. Filled with empty strings.")
    if filled_required > 0:
        print(
            f"[warning] Backfilled {filled_required} missing fields from annotation file "
            "(including adversarial_answer where applicable)."
        )
    if filled_from_adversarial > 0:
        print(
            f"[info] Filled {filled_from_adversarial} missing `answer` from `adversarial_answer`."
        )

    for sample in samples:
        for qa in sample.get("qa", []):
            pred = qa.get(prediction_key, "")
            ref = _reference_text_for_amem(qa)
            qa[f1_key] = round(_calculate_amem_style_f1(pred, ref), 3)
            qa[bleu_key] = round(_calculate_bleu1_score(pred, ref), 3)

    _dump_json(scored_path, samples)

    agg = _build_aggregate_stats(samples, f1_key, bleu_key, model_key, prediction_key)
    _dump_json(stats_path, agg)

    o = agg["overall"]
    print("\n=== A-mem style (set-F1 + BLEU-1) ===")
    print(
        f"Overall amem_f1 mean: {o['amem_f1']['mean']:.4f} "
        f"(n={o['amem_f1']['count']})"
    )
    print(
        f"Overall bleu1 mean:   {o['bleu1']['mean']:.4f} "
        f"(n={o['bleu1']['count']})"
    )
    _print_category_metric_stats(samples, f1_key, "amem_f1")
    _print_category_metric_stats(samples, bleu_key, "bleu1")

    print("\nEvaluation completed.")
    print(f"Total QA evaluated: {total_qa}")
    print(f"Per-question scored file: {scored_path}")
    print(f"Aggregate stats file: {stats_path}")


if __name__ == "__main__":
    main()
