"""
LLM Reranker — 用 LLM 对检索结果重排序

比 cross-encoder 简单，但利用了 LLM 的语义理解能力。
后续可替换为 fine-tuned cross-encoder。
"""

from __future__ import annotations

from typing import Any

from .llm import LLMClient


def llm_rerank(
    query: str,
    candidates: list[dict[str, Any]],
    llm: LLMClient,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """用 LLM 对候选文档重排序

    Args:
        query: 用户查询
        candidates: 候选文档列表，每个包含 "text" 和可选 "metadata"
        llm: LLM 客户端
        top_k: 返回前 k 个

    Returns:
        重排序后的候选列表
    """
    if len(candidates) <= 1:
        return candidates[:top_k]

    # 构建 prompt，让 LLM 打分
    docs_text = ""
    for i, c in enumerate(candidates[:20]):  # 最多 20 个候选
        text = c.get("text", "")[:300]
        docs_text += f"\n[{i}] {text}\n"

    prompt = (
        f"你是一个学术检索评估器。给定一个研究问题和一组候选文档，"
        f"请按相关性从高到低排序，返回文档编号列表。\n\n"
        f"研究问题: {query}\n\n"
        f"候选文档:{docs_text}\n\n"
        f"请只返回排序后的编号列表，用逗号分隔，如: 2,0,3,1\n"
        f"只返回前 {top_k} 个最相关的。"
    )

    response = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    # 解析排序结果
    try:
        content = response.content or ""
        # 提取数字
        indices = []
        for part in content.replace(" ", "").split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if 0 <= idx < len(candidates):
                    indices.append(idx)

        if indices:
            reranked = [candidates[i] for i in indices[:top_k]]
            # 补充未被选中的
            seen = set(indices[:top_k])
            for i, c in enumerate(candidates):
                if i not in seen and len(reranked) < top_k:
                    reranked.append(c)
            return reranked
    except Exception:
        pass

    # 解析失败，返回原序
    return candidates[:top_k]
