"""
RAG 检索器 — 统一封装向量检索 + BM25 + 经验增强

核心创新点：经验增强检索
- 普通: query → embedding → search
- 增强: query + experience → rewrite query → embedding → search → experience rerank
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .vector_store import VectorStore, SearchResult


@dataclass
class ExperienceRecord:
    """一条检索经验"""
    original_query: str
    rewritten_query: str | None
    result_score: float  # Critic 评分
    keywords_that_helped: list[str] = field(default_factory=list)
    keywords_that_missed: list[str] = field(default_factory=list)


class BM25:
    """简单 BM25 实现"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._texts: list[str] = []
        self._avg_dl: float = 0
        self._df: Counter = Counter()

    def add(self, text: str):
        tokens = self._tokenize(text)
        self._docs.append(tokens)
        self._texts.append(text)
        self._df.update(set(tokens))
        total = sum(len(d) for d in self._docs)
        self._avg_dl = total / len(self._docs)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        query_tokens = self._tokenize(query)
        n = len(self._docs)
        scores = []
        for i, doc in enumerate(self._docs):
            score = 0
            dl = len(doc)
            doc_counter = Counter(doc)
            for t in query_tokens:
                if t not in self._df:
                    continue
                df = self._df[t]
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                tf = doc_counter.get(t, 0)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl))
            scores.append((self._texts[i], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())


class RAGRetriever:
    """统一检索器，支持多种策略"""

    def __init__(self, vector_store: VectorStore, embed_fn=None):
        self.vector_store = vector_store
        self.embed_fn = embed_fn  # text → vector 的函数
        self.bm25 = BM25()
        self._experiences: list[ExperienceRecord] = []

    def add_document(self, text: str, metadata: dict | None = None):
        """添加文档到所有索引"""
        if self.embed_fn:
            vec = self.embed_fn(text)
            self.vector_store.add(text, vec, metadata)
        self.bm25.add(text)

    def search_vector(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """纯向量检索"""
        if not self.embed_fn:
            return []
        vec = self.embed_fn(query)
        return self.vector_store.search(vec, top_k)

    def search_bm25(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """纯 BM25 检索"""
        return self.bm25.search(query, top_k)

    def search_hybrid(self, query: str, top_k: int = 5,
                      vector_weight: float = 0.7) -> list[SearchResult]:
        """混合检索：向量 + BM25"""
        vec_results = self.search_vector(query, top_k * 2)
        bm25_results = self.search_bm25(query, top_k * 2)

        # 分数归一化并融合
        scores: dict[str, float] = {}
        for r in vec_results:
            scores[r.text] = vector_weight * r.score
        bm25_max = max((s for _, s in bm25_results), default=1)
        for text, score in bm25_results:
            norm_score = score / bm25_max if bm25_max > 0 else 0
            scores[text] = scores.get(text, 0) + (1 - vector_weight) * norm_score

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [SearchResult(text=text, metadata={}, score=score, index=i)
                for i, (text, score) in enumerate(sorted_results)]

    # ========================================
    # 经验增强检索（核心创新）
    # ========================================

    def add_experience(self, record: ExperienceRecord):
        """记录一条检索经验"""
        self._experiences.append(record)

    def search_with_experience(self, query: str, top_k: int = 5) -> tuple[list[SearchResult], str | None]:
        """经验增强检索

        返回: (检索结果, 改写后的 query 或 None)
        """
        rewritten = self._rewrite_with_experience(query)
        actual_query = rewritten or query
        results = self.search_hybrid(actual_query, top_k)
        return results, rewritten

    def _rewrite_with_experience(self, query: str) -> str | None:
        """基于历史经验改写 query"""
        if not self._experiences:
            return None

        # 找到历史上评分高的经验
        good_experiences = [e for e in self._experiences if e.result_score >= 7.0]
        if not good_experiences:
            return None

        # 收集有效关键词
        helpful_keywords = []
        for exp in good_experiences[-10:]:  # 最近 10 条好经验
            helpful_keywords.extend(exp.keywords_that_helped)

        if not helpful_keywords:
            return None

        # 用高频有效关键词增强 query
        keyword_counts = Counter(helpful_keywords)
        top_keywords = [kw for kw, _ in keyword_counts.most_common(3)]

        # 只添加 query 中不包含的关键词
        additions = [kw for kw in top_keywords if kw.lower() not in query.lower()]
        if not additions:
            return None

        return f"{query} {' '.join(additions)}"

    @property
    def experience_count(self) -> int:
        return len(self._experiences)
