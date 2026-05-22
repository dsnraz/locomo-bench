"""
Summary RAG — 基于 LoCoMo session summary 检索的问答预测。

架构（对标 hyperbolic_memory 的 predict-then-eval 模式）:
  提取模型（extraction_model）: 为每个 session 生成摘要
  嵌入模型（embedding_model）: all-mpnet-base-v2，编码摘要和问题
  生成模型（generation_model）: 基于检索到的摘要上下文回答问题

运行: python main.py --data-file ... --out-file ... [其他参数]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import random
import torch


# ─── 模型类型检测 & 加载（内联，避免跨仓库依赖）──────────────────────────

MODEL_TYPE_SIGNATURES = {
    "qwen": ["qwen"],
    "llama": ["llama", "vicuna", "alpaca"],
    "chatglm": ["chatglm"],
    "baichuan": ["baichuan"],
    "internlm": ["internlm"],
    "mistral": ["mistral"],
    "deepseek": ["deepseek"],
}


def detect_model_type(model_path: str) -> str:
    model_path_lower = model_path.lower()
    for model_type, signatures in MODEL_TYPE_SIGNATURES.items():
        if any(sig in model_path_lower for sig in signatures):
            return model_type
    return "default"


def load_model(model_path: str, device: str = "auto"):
    """加载 transformers 模型，返回 (model, tokenizer, model_type)。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_type = detect_model_type(model_path)
    print(f"加载模型: {model_path}")
    print(f"检测到模型类型: {model_type}")

    device_map = "auto" if device == "auto" else ({"": "cuda"} if device == "cuda" else {"": "cpu"})

    tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": True, "use_fast": False}
    decoder_only_types = ["qwen", "llama", "mistral", "baichuan", "internlm", "deepseek"]
    if model_type in decoder_only_types:
        tokenizer_kwargs["padding_side"] = "left"

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|extra_0|>"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and getattr(gen_cfg, "pad_token_id", None) is None:
        gen_cfg.pad_token_id = tokenizer.pad_token_id

    model.eval()
    print("模型加载完成")
    return model, tokenizer, model_type


def build_prompt(tokenizer, model_type: str, user_prompt: str) -> str:
    """根据模型类型构建 chat template 或原始 prompt。"""
    if model_type in ("qwen", "deepseek", "llama"):
        messages = [{"role": "user", "content": user_prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif model_type == "chatglm":
        return f"[Round 0]\n问：{user_prompt}\n答："
    else:
        return user_prompt


def generate(
    model, tokenizer, model_type: str, prompt: str,
    max_new_tokens: int = 256, temperature: float = 0.1
) -> str:
    """单条生成。"""
    formatted = build_prompt(tokenizer, model_type, prompt)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=10000000)
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()

    if model_type == "deepseek" and "</think>" in response:
        response = response.rsplit("</think>", 1)[-1]
    if model_type == "qwen" and "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0].strip()
    return response.strip()


def batch_generate(
    model, tokenizer, model_type: str, prompts: List[str],
    max_new_tokens: int = 256
) -> List[str]:
    """批量生成。"""
    if len(prompts) == 0:
        return []
    if len(prompts) == 1:
        return [generate(model, tokenizer, model_type, prompts[0], max_new_tokens)]

    formatted = [build_prompt(tokenizer, model_type, p) for p in prompts]
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                       max_length=10000000, padding=True)
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    responses = []
    for output in outputs:
        gen = output[input_len:]
        resp = tokenizer.decode(gen, skip_special_tokens=True).strip()
        if model_type == "deepseek" and "</think>" in resp:
            resp = resp.rsplit("</think>", 1)[-1]
        if model_type == "qwen" and "<|im_end|>" in resp:
            resp = resp.split("<|im_end|>")[0].strip()
        responses.append(resp.strip())
    return responses


# ─── 嵌入模型 ────────────────────────────────────────────────────────────

def load_embedding_model(embedding_model: str, device: str = "auto"):
    """加载 sentence-transformers 嵌入模型。"""
    from sentence_transformers import SentenceTransformer
    dev = None if device == "auto" else device
    print(f"加载嵌入模型: {embedding_model}")
    model = SentenceTransformer(embedding_model, device=dev, local_files_only=True)
    dim = model.get_sentence_embedding_dimension()
    print(f"嵌入维度: {dim}")
    return model


def embed_batch(model, texts: List[str]) -> np.ndarray:
    """批量嵌入，返回 (N, dim) numpy 数组。"""
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()))
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_tensor=False)
    return np.array(embeddings)


# ─── 数据工具 ────────────────────────────────────────────────────────────

def get_session_numbers(conversation: Dict[str, Any]) -> List[int]:
    nums = []
    for key in conversation:
        if key.startswith("session_") and not key.endswith("date_time"):
            try:
                nums.append(int(key.split("_")[-1]))
            except ValueError:
                pass
    return sorted(nums)


def build_session_text(session: List[Dict[str, Any]], date_time: str) -> str:
    """构建单个 session 的文本表示（对标 locomo 的 get_summary_query）。"""
    lines = [date_time]
    for dialog in session:
        line = f'{dialog["speaker"]} said, "{dialog["text"]}"'
        if "blip_caption" in dialog:
            line += f' and shared {dialog["blip_caption"]}.'
        lines.append(line)
    return "\n".join(lines)


# ─── 摘要生成（提取模型）────────────────────────────────────────────────

SUMMARY_PROMPT = (
    "Generate a concise summary of the following conversation using exact words "
    "from the conversation wherever possible. The summary should contain all facts "
    "about the two speakers, as well as references to time.\n\n"
    "{conversation}\n\n"
    "Summary:"
)


def generate_summaries(
    sample: Dict[str, Any],
    model, tokenizer, model_type: str,
    batch_size: int = 8,
) -> Tuple[List[str], List[str], List[str]]:
    """为一个样本的所有 session 生成摘要。
    返回 (summaries, date_times, context_ids)。
    """
    conversation = sample.get("conversation", {})
    session_nums = get_session_numbers(conversation)

    texts = []
    date_times = []
    context_ids = []

    for i in session_nums:
        session = conversation.get(f"session_{i}", [])
        dt = conversation.get(f"session_{i}_date_time", "")
        texts.append(build_session_text(session, dt))
        date_times.append(dt)
        context_ids.append(f"S{i}")

    prompts = [SUMMARY_PROMPT.format(conversation=t) for t in texts]

    summaries = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        summaries.extend(batch_generate(model, tokenizer, model_type, batch, max_new_tokens=256))

    return summaries, date_times, context_ids


# ─── 问答 Prompt ─────────────────────────────────────────────────────────

QA_PROMPT = (
    "Based on the above context, write an answer in the form of a short phrase "
    "for the following question. Answer with exact words from the context "
    "whenever possible.\n\n"
    "Question: {question} Short answer:"
)

QA_PROMPT_TEMPORAL = (
    "Based on the above context, write an answer in the form of a short phrase "
    "for the following question. Answer with exact words from the context "
    "whenever possible. Use DATE of CONVERSATION to answer with an approximate date.\n\n"
    "Question: {question} Short answer:"
)

QA_PROMPT_CAT_5 = (
    "Based on the above context, answer the following question.\n\n"
    "Question: {question} Short answer:"
)


def build_context(retrieved_summaries: List[str], retrieved_dates: List[str]) -> str:
    """构建 RAG 上下文（对标 locomo 的格式）。"""
    parts = [f"{dt}: {summary}" for dt, summary in zip(retrieved_dates, retrieved_summaries)]
    return "\n\n".join(parts)


# ─── 参数解析 ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summary RAG — session-summary 检索 + LLM 问答预测"
    )
    # 数据
    p.add_argument("--data-file", type=str, required=True,
                   help="LoCoMo 数据 JSON（如 locomo10.json 或 locomo_qa_test.json）")
    p.add_argument("--out-file", type=str, required=True,
                   help="输出预测 JSON 路径")
    p.add_argument("--prediction-key", type=str, default="summary_rag_prediction",
                   help="写入每个 qa 条目的预测字段名")

    # 提取模型（生成 session summary）
    p.add_argument("--extraction-model-path", type=str, default=None,
                   help="提取模型本地路径（如 Qwen2.5-7B-Instruct），不传则跳过摘要生成直接用原始 session 文本")
    p.add_argument("--extraction-device", type=str, default="auto")
    p.add_argument("--extraction-batch-size", type=int, default=8,
                   help="摘要生成的 batch size")

    # 嵌入模型
    p.add_argument("--embedding-model", type=str,
                   default="sentence-transformers/all-mpnet-base-v2",
                   help="句向量模型名或本地路径（默认 all-mpnet-base-v2，与 hymemory 一致）")

    # 检索
    p.add_argument("--top-k", type=int, default=5,
                   help="检索 top-K 个 summary 作为上下文")

    # 生成模型（问答）
    p.add_argument("--generation-model-path", type=str, default=None,
                   help="生成模型本地路径（如 Llama-3.2-3B-Instruct）")
    p.add_argument("--generation-device", type=str, default="auto")

    # 运行控制
    p.add_argument("--max-samples", type=int, default=100000000)
    p.add_argument("--max-questions", type=int, default=100000000)
    p.add_argument("--device", type=str, default="auto",
                   help="嵌入模型设备")

    return p.parse_args()


# ─── 主流程 ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # 1. 加载数据
    data_path = Path(args.data_file)
    if not data_path.is_file():
        raise FileNotFoundError(f"数据文件不存在: {args.data_file}")
    with open(data_path, encoding="utf-8") as f:
        samples: List[Dict[str, Any]] = json.load(f)
    print(f"加载了 {len(samples)} 个样本")

    # 2. 加载提取模型（可选——不传则用原始 session 文本代替摘要）
    extraction_model = extraction_tokenizer = extraction_model_type = None
    if args.extraction_model_path:
        extraction_model, extraction_tokenizer, extraction_model_type = load_model(
            args.extraction_model_path, args.extraction_device
        )

    # 3. 加载嵌入模型
    embedding_model = load_embedding_model(args.embedding_model, args.device)

    # 4. 加载生成模型
    generation_model = generation_tokenizer = generation_model_type = None
    if args.generation_model_path:
        generation_model, generation_tokenizer, generation_model_type = load_model(
            args.generation_model_path, args.generation_device
        )

    # 5. 逐样本处理
    n_samples = min(len(samples), max(1, args.max_samples))
    output_samples: List[Dict[str, Any]] = []

    for si in range(n_samples):
        sample = samples[si]
        sid = sample.get("sample_id", f"index_{si}")
        print(f"\n{'='*60}")
        print(f"样本 {sid} ({si + 1}/{n_samples})")
        print(f"{'='*60}")

        # 5a. 生成/获取 session summaries
        if extraction_model is not None:
            print("生成 session summaries ...")
            summaries, date_times, context_ids = generate_summaries(
                sample, extraction_model, extraction_tokenizer, extraction_model_type,
                batch_size=args.extraction_batch_size,
            )
            print(f"  共 {len(summaries)} 个 summaries")
        else:
            # 不用提取模型：直接用原始 session 文本作为"摘要"
            print("未提供提取模型，使用原始 session 文本 ...")
            conversation = sample.get("conversation", {})
            session_nums = get_session_numbers(conversation)
            summaries, date_times, context_ids = [], [], []
            for i in session_nums:
                session = conversation.get(f"session_{i}", [])
                dt = conversation.get(f"session_{i}_date_time", "")
                summaries.append(build_session_text(session, dt))
                date_times.append(dt)
                context_ids.append(f"S{i}")
            print(f"  共 {len(summaries)} 个原始 session 文本")

        # 5b. 嵌入所有 summaries
        print("嵌入 summaries ...")
        summary_embeddings = embed_batch(embedding_model, summaries)  # (N, dim)

        # 5c. 对每个问题：嵌入 → 检索 → 生成
        out_sample: Dict[str, Any] = {
            "sample_id": sid,
            "qa": [dict(q) for q in (sample.get("qa") or [])],
        }
        qa_list = out_sample["qa"]
        n_q = min(len(qa_list), max(1, args.max_questions))

        # 预先嵌入所有问题
        questions = [str(item.get("question", "")) for item in qa_list[:n_q]]
        question_embeddings = embed_batch(embedding_model, questions)  # (n_q, dim)

        for qi in range(n_q):
            item = qa_list[qi]
            question = questions[qi]
            if not question.strip():
                continue

            # 检索 top-K summaries（余弦相似度）
            q_emb = question_embeddings[qi]  # (dim,)
            scores = np.dot(summary_embeddings, q_emb)  # (N,)
            top_indices = np.argsort(scores)[::-1][:args.top_k]

            retrieved_summaries = [summaries[idx] for idx in top_indices]
            retrieved_dates = [date_times[idx] for idx in top_indices]

            context = build_context(retrieved_summaries, retrieved_dates)

            # 构建 QA prompt（对标 locomo gpt_utils 的分类处理）
            cat = item.get("category", 0)
            if cat == 5:
                # 对抗性问题：选项随机排列，模型需选择 (a) 或 (b)
                ground_truth = str(item.get("answer", ""))
                if random.random() < 0.5:
                    q_text = question + " Select the correct answer: (a) Not mentioned in the conversation (b) " + ground_truth + ". "
                    answer_map = {"a": "Not mentioned in the conversation", "b": ground_truth}
                else:
                    q_text = question + " Select the correct answer: (a) " + ground_truth + " (b) Not mentioned in the conversation. "
                    answer_map = {"a": ground_truth, "b": "Not mentioned in the conversation"}
                qa_prompt = QA_PROMPT_CAT_5.format(question=q_text)
                full_prompt = context + "\n\n" + qa_prompt

                if generation_model is not None:
                    raw_answer = generate(generation_model, generation_tokenizer,
                                          generation_model_type, full_prompt, max_new_tokens=32)
                else:
                    raw_answer = ""

                # 解析 (a)/(b) 选择，映射回实际答案
                raw_lower = raw_answer.strip().lower()
                if len(raw_lower) >= 1 and raw_lower[0] in ("a", "b"):
                    answer = answer_map.get(raw_lower[0], raw_answer)
                elif "(a)" in raw_lower:
                    answer = answer_map.get("a", raw_answer)
                elif "(b)" in raw_lower:
                    answer = answer_map.get("b", raw_answer)
                else:
                    answer = raw_answer
            elif cat == 2:
                qa_prompt = QA_PROMPT_TEMPORAL.format(question=question)
                full_prompt = context + "\n\n" + qa_prompt
                if generation_model is not None:
                    answer = generate(generation_model, generation_tokenizer,
                                      generation_model_type, full_prompt, max_new_tokens=32)
                else:
                    answer = ""
            else:
                qa_prompt = QA_PROMPT.format(question=question)
                full_prompt = context + "\n\n" + qa_prompt
                if generation_model is not None:
                    answer = generate(generation_model, generation_tokenizer,
                                      generation_model_type, full_prompt, max_new_tokens=32)
                else:
                    answer = ""

            item[args.prediction_key] = answer.strip() if answer else ""
            print(f"  Q{qi+1} (cat{cat}): {question[:80]}... → {answer[:80] if answer else '(无)'}")

        output_samples.append(out_sample)

    # 6. 写入预测 JSON
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_samples, f, ensure_ascii=False, indent=2)
    print(f"\n预测结果已写入: {out_path}")
    print(f"预测字段: {args.prediction_key}")


if __name__ == "__main__":
    main()
