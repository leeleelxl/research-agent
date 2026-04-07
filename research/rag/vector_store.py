"""
向量存储 — 基于 numpy 的轻量级向量检索（开发阶段）

生产环境可替换为 FAISS。开发阶段用 numpy 避免安装依赖。
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SearchResult:
    text: str
    metadata: dict[str, Any]
    score: float
    index: int


class VectorStore:
    """轻量级向量存储"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._vectors: list[np.ndarray] = []
        self._texts: list[str] = []
        self._metadatas: list[dict] = []

    def add(self, text: str, vector: list[float] | np.ndarray, metadata: dict | None = None):
        """添加一条记录"""
        vec = np.array(vector, dtype=np.float32)
        if vec.shape[0] != self.dimension:
            raise ValueError(f"维度不匹配: 期望 {self.dimension}, 实际 {vec.shape[0]}")
        self._vectors.append(vec)
        self._texts.append(text)
        self._metadatas.append(metadata or {})

    def add_batch(self, texts: list[str], vectors: list[list[float]], metadatas: list[dict] | None = None):
        """批量添加"""
        metadatas = metadatas or [{}] * len(texts)
        for text, vec, meta in zip(texts, vectors, metadatas):
            self.add(text, vec, meta)

    def search(self, query_vector: list[float] | np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """余弦相似度检索"""
        if not self._vectors:
            return []

        query = np.array(query_vector, dtype=np.float32)
        query_norm = query / (np.linalg.norm(query) + 1e-8)

        matrix = np.stack(self._vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        matrix_norm = matrix / norms

        scores = matrix_norm @ query_norm
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(SearchResult(
                text=self._texts[idx],
                metadata=self._metadatas[idx],
                score=float(scores[idx]),
                index=int(idx),
            ))
        return results

    @property
    def count(self) -> int:
        return len(self._vectors)

    def save(self, path: str | Path):
        """持久化到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dimension": self.dimension,
            "texts": self._texts,
            "metadatas": self._metadatas,
            "vectors": [v.tolist() for v in self._vectors],
        }
        path.write_text(json.dumps(data, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> VectorStore:
        """从文件加载"""
        data = json.loads(Path(path).read_text())
        store = cls(dimension=data["dimension"])
        store._texts = data["texts"]
        store._metadatas = data["metadatas"]
        store._vectors = [np.array(v, dtype=np.float32) for v in data["vectors"]]
        return store
