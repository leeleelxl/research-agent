"""
Embedding 工具 — 文本向量化

策略：
1. 优先使用本地 sentence-transformers（效果好）
2. 回退到 API-based embedding（方便）
3. 最后回退到简单的 TF-IDF 向量（零依赖 baseline）
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Callable

_embed_fn: Callable[[str], list[float]] | None = None
_dimension: int = 384


def get_embed_fn() -> tuple[Callable[[str], list[float]], int]:
    """获取 embedding 函数和维度"""
    global _embed_fn, _dimension

    if _embed_fn is not None:
        return _embed_fn, _dimension

    # 尝试 sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        _dimension = model.get_sentence_embedding_dimension()

        def _st_embed(text: str) -> list[float]:
            return model.encode(text, normalize_embeddings=True).tolist()

        _embed_fn = _st_embed
        return _embed_fn, _dimension
    except ImportError:
        pass

    # 回退到简单的哈希向量（保持功能可用，但效果差）
    _dimension = 256

    def _hash_embed(text: str, dim: int = 256) -> list[float]:
        """基于字符 n-gram 的简单 embedding（零依赖 baseline）"""
        tokens = re.findall(r'\w+', text.lower())
        vec = [0.0] * dim
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    _embed_fn = _hash_embed
    return _embed_fn, _dimension


def embed_text(text: str) -> list[float]:
    """向量化一段文本"""
    fn, _ = get_embed_fn()
    return fn(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """批量向量化"""
    fn, _ = get_embed_fn()
    return [fn(t) for t in texts]


def get_dimension() -> int:
    """获取当前 embedding 维度"""
    _, dim = get_embed_fn()
    return dim
