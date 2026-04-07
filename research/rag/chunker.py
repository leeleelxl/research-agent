"""
文本分块策略 — 实现 3 种 chunk 方式用于消融实验

1. naive: 固定字符数切分
2. sentence_window: 按句子切分，保留上下文窗口
3. semantic: 按语义段落切分（利用换行+标题检测）
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class ChunkStrategy(str, Enum):
    NAIVE = "naive"
    SENTENCE_WINDOW = "sentence_window"
    SEMANTIC = "semantic"


@dataclass
class Chunk:
    text: str
    metadata: dict
    index: int


class Chunker:
    """文本分块器"""

    def __init__(self, strategy: ChunkStrategy = ChunkStrategy.SENTENCE_WINDOW,
                 chunk_size: int = 512, overlap: int = 64, window_size: int = 2):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.window_size = window_size

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        if self.strategy == ChunkStrategy.NAIVE:
            return self._naive_chunk(text, metadata)
        elif self.strategy == ChunkStrategy.SENTENCE_WINDOW:
            return self._sentence_window_chunk(text, metadata)
        elif self.strategy == ChunkStrategy.SEMANTIC:
            return self._semantic_chunk(text, metadata)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def _naive_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        """固定字符数切分"""
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append(Chunk(
                text=chunk_text,
                metadata={**metadata, "strategy": "naive", "char_start": start},
                index=idx,
            ))
            start = end - self.overlap
            idx += 1
        return chunks

    def _sentence_window_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        """句子切分 + 上下文窗口"""
        sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        for i, sent in enumerate(sentences):
            # 取窗口内的上下文
            start = max(0, i - self.window_size)
            end = min(len(sentences), i + self.window_size + 1)
            window_text = " ".join(sentences[start:end])

            # 截断到 chunk_size
            if len(window_text) > self.chunk_size:
                window_text = window_text[:self.chunk_size]

            chunks.append(Chunk(
                text=window_text,
                metadata={**metadata, "strategy": "sentence_window",
                         "center_sentence": i, "window": f"{start}-{end}"},
                index=i,
            ))
        return chunks

    def _semantic_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        """语义段落切分（基于双换行、标题检测）"""
        # 按双换行或标题模式切分
        segments = re.split(r'\n\s*\n|\n(?=#{1,4}\s)', text)
        segments = [s.strip() for s in segments if s.strip()]

        chunks = []
        current = ""
        idx = 0
        for seg in segments:
            if len(current) + len(seg) <= self.chunk_size:
                current = current + "\n\n" + seg if current else seg
            else:
                if current:
                    chunks.append(Chunk(
                        text=current,
                        metadata={**metadata, "strategy": "semantic"},
                        index=idx,
                    ))
                    idx += 1
                current = seg

        if current:
            chunks.append(Chunk(
                text=current,
                metadata={**metadata, "strategy": "semantic"},
                index=idx,
            ))
        return chunks
