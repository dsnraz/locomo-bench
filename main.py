"""
Summary RAG — observation 模式预测脚本（薄封装）。

核心逻辑全部调用 locomo 现有 task_eval 代码:
  - memory_utils.get_session_facts()        事实提取
  - rag_utils.get_embeddings()              嵌入 & 检索
  - gpt_utils.get_gpt_answers()             QA 问答
  - evaluation.eval_question_answering()    评估

只替换三个后端:
  1. run_chatgpt / run_chatgpt_with_examples → 本地模型 (llama/qwen/deepseek)
  2. get_embeddings → sentence-transformers/all-mpnet-base-v2
  3. 预测/评估分离 (predict → JSON → eval.sh)

对标 hyperbolic_memory 的 session_run.py + main.sh 参数风格。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# 让 locomo 的 task_eval 等模块可以被 import
sys.path.insert(0, str(Path(__file__).parent))


# ─── 本地模型加载（仅此一段是新增的）────────────────────────────────────

MODEL_TYPE_SIGNATURES = {
    "qwen": ["qwen"],
    "llama": ["llama", "vicuna", "alpaca"],
    "deepseek": ["deepseek"],
}


def _detect_model_type(model_path: str) -> str:
    lower = model_path.lower()
    for t, sigs in MODEL_TYPE_SIGNATURES.items():
        if any(s in lower for s in sigs):
            return t
    return "default"


def _load_local_model(model_path: str, device: str = "auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_type = _detect_model_type(model_path)
    print(f"[本地模型] {model_path}  (类型: {model_type})")

    tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": True, "use_fast": False}
    if model_type in ("qwen", "llama", "deepseek", "mistral", "baichuan", "internlm"):
        tokenizer_kwargs["padding_side"] = "left"

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|extra_0|>"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map="auto" if device == "auto" else device,
        torch_dtype=torch.float16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if (gc := getattr(model, "generation_config", None)) and gc.pad_token_id is None:
        gc.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer, model_type


def _llm_generate(model, tokenizer, model_type: str, messages: List[Dict], max_tokens: int) -> str:
    """用 chat template 构建 prompt 并生成。messages 格式同 OpenAI API。"""
    if model_type in ("qwen", "deepseek", "llama"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # fallback: 拼接消息
        prompt = "\n".join(m["content"] for m in messages)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if model_type == "deepseek" and "</think>" in response:
        response = response.rsplit("</think>", 1)[-1]
    if model_type == "qwen" and "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0].strip()
    return response.strip()


# ─── Monkey-patch locomo 的 OpenAI 调用 ──────────────────────────────────

_patched_model = None
_patched_tokenizer = None
_patched_model_type = None


def _patched_run_chatgpt(query, num_gen=1, num_tokens_request=1000, model="chatgpt",
                         use_16k=False, temperature=1.0, wait_time=1):
    """替换 global_methods.run_chatgpt：用本地模型生成。"""
    messages = [{"role": "user", "content": query}]
    return _llm_generate(_patched_model, _patched_tokenizer, _patched_model_type, messages, num_tokens_request)


def _patched_run_chatgpt_with_examples(query, examples, input, num_gen=1, num_tokens_request=1000,
                                       use_16k=False, wait_time=1, temperature=1.0):
    """替换 global_methods.run_chatgpt_with_examples：few-shot → 本地模型。"""
    messages = [{"role": "system", "content": query}]
    for inp, out in examples:
        messages.append({"role": "user", "content": inp})
        messages.append({"role": "assistant", "content": out})
    messages.append({"role": "user", "content": input})
    return _llm_generate(_patched_model, _patched_tokenizer, _patched_model_type, messages, num_tokens_request)


# ─── Monkey-patch 嵌入 ──────────────────────────────────────────────────

_embedding_model = None


def _patched_get_embeddings(retriever, inputs, mode="context"):
    """替换 rag_utils.get_embeddings：用 all-mpnet-base-v2。"""
    embeddings = _embedding_model.encode(inputs, normalize_embeddings=True, convert_to_tensor=False)
    return np.array(embeddings)


# ─── 参数解析 ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Observation RAG 预测")
    p.add_argument("--data-file", type=str, required=True)
    p.add_argument("--out-file", type=str, required=True)
    p.add_argument("--prediction-key", type=str, default="observation_rag_prediction")
    # 提取模型 — 用于 get_session_facts 事实提取
    p.add_argument("--extraction-model-path", type=str, default=None)
    # 生成模型 — 用于 get_gpt_answers QA
    p.add_argument("--generation-model-path", type=str, default=None)
    # 嵌入模型
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    # RAG 参数
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--rag-mode", type=str, default="observation",
                   choices=("observation", "dialog", "summary"))
    p.add_argument("--emb-dir", type=str, default="/tmp/locomo_emb")
    p.add_argument("--prompt-dir", type=str, default=str(Path(__file__).parent / "prompt_examples"))
    # 运行控制
    p.add_argument("--max-samples", type=int, default=100000000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use-rag", action="store_true", default=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--model", type=str, default="chatgpt",  # 兼容 gpt_utils 内部逻辑
                   help="传给 gpt_utils 的 model 名，本地模型时用 chatgpt 即可")
    p.add_argument("--retriever", type=str, default="dragon",  # 被 patch 覆盖，仅占位
                   help="嵌入检索器名（被本地嵌入模型覆盖）")
    return p.parse_args()


# ─── 主流程 ──────────────────────────────────────────────────────────────

def main() -> None:
    global _patched_model, _patched_tokenizer, _patched_model_type, _embedding_model
    args = parse_args()

    # 确保 emb_dir 和 prompt_dir 存在
    Path(args.emb_dir).mkdir(parents=True, exist_ok=True)

    # 1. 加载本地模型 & 嵌入模型
    if args.extraction_model_path:
        _patched_model, _patched_tokenizer, _patched_model_type = _load_local_model(
            args.extraction_model_path, args.device
        )

    gen_model = gen_tokenizer = gen_model_type = None
    if args.generation_model_path and args.generation_model_path != args.extraction_model_path:
        gen_model, gen_tokenizer, gen_model_type = _load_local_model(
            args.generation_model_path, args.device
        )

    from sentence_transformers import SentenceTransformer
    print(f"[嵌入模型] {args.embedding_model}")
    _embedding_model = SentenceTransformer(args.embedding_model, device=None if args.device == "auto" else args.device)

    # 2. Monkey-patch
    import global_methods
    global_methods.run_chatgpt = _patched_run_chatgpt
    global_methods.run_chatgpt_with_examples = _patched_run_chatgpt_with_examples

    import task_eval.rag_utils
    task_eval.rag_utils.get_embeddings = _patched_get_embeddings

    # 3. 导入 locomo 核心逻辑
    from generative_agents.memory_utils import get_session_facts
    from task_eval.gpt_utils import get_gpt_answers

    data_path = Path(args.data_file)
    if not data_path.is_file():
        raise FileNotFoundError(f"数据文件不存在: {args.data_file}")
    with open(data_path, encoding="utf-8") as f:
        samples: List[Dict[str, Any]] = json.load(f)
    print(f"加载了 {len(samples)} 个样本")

    dataset_prefix = data_path.stem
    n_samples = min(len(samples), max(1, args.max_samples))
    output_samples: List[Dict[str, Any]] = []

    for si in range(n_samples):
        sample = samples[si]
        sid = sample.get("sample_id", f"index_{si}")
        print(f"\n{'='*60}\n样本 {sid} ({si + 1}/{n_samples})\n{'='*60}")

        # ── 步骤 A: 生成 observations（调 locomo get_session_facts）──
        print("提取 observations ...")
        conversation = sample.get("conversation", {})
        session_nums = sorted(
            int(k.split("_")[-1]) for k in conversation
            if k.startswith("session_") and "date_time" not in k
        )
        observations = []
        date_times = []
        context_ids = []

        for sess_idx in session_nums:
            facts = get_session_facts(args, conversation, conversation, sess_idx, return_embeddings=False)
            dt = conversation.get(f"session_{sess_idx}_date_time", "")
            for speaker, fact_list in facts.items():
                for fact_text, dia_id in fact_list:
                    observations.append(fact_text)
                    context_ids.append(dia_id)
                    date_times.append(dt)

        # ── 步骤 B: 嵌入 observations → 写 pickle（调 patched get_embeddings）──
        print(f"嵌入 {len(observations)} 条 observations ...")
        emb_inputs = [f"{dt}. {obs}" for dt, obs in zip(date_times, observations)] if args.rag_mode == "observation" else observations
        import task_eval.rag_utils as rag_utils
        embeddings = rag_utils.get_embeddings(args.retriever, emb_inputs, "context")

        pkl_path = Path(args.emb_dir) / f"{dataset_prefix}_observation_{sid}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({
                "embeddings": embeddings,
                "date_time": date_times,
                "dia_id": context_ids,
                "context": observations,
            }, f)

        # ── 步骤 C: 切换生成模型（如果和提取模型不同）──
        if gen_model is not None:
            _patched_model, _patched_tokenizer, _patched_model_type = gen_model, gen_tokenizer, gen_model_type

        # ── 步骤 D: QA（调 locomo get_gpt_answers）──
        print("QA 生成 ...")
        out_sample: Dict[str, Any] = {"sample_id": sid, "qa": [dict(q) for q in (sample.get("qa") or [])]}
        out_sample = get_gpt_answers(sample, out_sample, args.prediction_key, args)
        output_samples.append(out_sample)

    # 4. 写预测 JSON
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_samples, f, ensure_ascii=False, indent=2)
    print(f"\n预测结果已写入: {out_path}")


if __name__ == "__main__":
    main()
