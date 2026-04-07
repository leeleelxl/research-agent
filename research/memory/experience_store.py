"""
经验存储 — 持久化检索经验用于经验增强检索

存储 (query, rewrite, score, helpful_keywords, missed_keywords) 五元组，
支持按相似度召回历史经验。
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

from ..rag.retriever import ExperienceRecord


class ExperienceStore:
    """经验持久化存储"""

    def __init__(self, path: str | Path = "experience.jsonl"):
        self.path = Path(path)
        self._records: list[ExperienceRecord] = []
        if self.path.exists():
            self._load()

    def add(self, record: ExperienceRecord):
        self._records.append(record)
        self._save_one(record)

    def get_good_experiences(self, min_score: float = 7.0, limit: int = 20) -> list[ExperienceRecord]:
        """获取高质量经验"""
        good = [r for r in self._records if r.result_score >= min_score]
        return good[-limit:]

    def get_bad_experiences(self, max_score: float = 4.0, limit: int = 10) -> list[ExperienceRecord]:
        """获取失败经验（避免重复犯错）"""
        bad = [r for r in self._records if r.result_score <= max_score]
        return bad[-limit:]

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def avg_score(self) -> float:
        if not self._records:
            return 0.0
        return sum(r.result_score for r in self._records) / len(self._records)

    def summary(self) -> str:
        if not self._records:
            return "经验库为空"
        good = len([r for r in self._records if r.result_score >= 7.0])
        bad = len([r for r in self._records if r.result_score <= 4.0])
        return (f"经验库: {self.count} 条 | 均分: {self.avg_score:.1f} "
                f"| 优质: {good} | 失败: {bad}")

    def _save_one(self, record: ExperienceRecord):
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def _load(self):
        for line in self.path.read_text().strip().split("\n"):
            if line:
                d = json.loads(line)
                self._records.append(ExperienceRecord(**d))
