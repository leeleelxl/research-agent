"""
Retriever Agent — 论文检索与知识库管理

学术背景：
- Self-RAG（Asai et al., ICLR 2024）：Agent 自主决定何时检索、检索什么
  我们的 Retriever 也有自主性——根据上下文选择 Semantic Scholar 还是 arXiv
- CRAG（Yan et al., 2024）：Corrective RAG，检索不准时自动修正查询
  我们的经验增强检索就是这个思想的实现——用历史反馈修正未来的 query
- 经验增强检索（核心创新）：
  普通 RAG: query → search
  我们: query + experience → rewrite → search → experience-based rerank

设计决策：
- Retriever 拥有最多的工具（4 个），因为检索是最复杂的步骤
- 同时持有外部检索（API）和内部检索（向量库）能力，可以互补
- 经验记忆持久化到 ExperienceStore，跨 session 可用
"""

from __future__ import annotations

from .base import BaseAgent


class RetrieverAgent(BaseAgent):
    role = "retriever"
    max_rounds = 5  # 检索可能需要多轮尝试

    system_prompt = """你是一个学术论文检索专家。你的任务是根据研究子问题找到最相关的论文。

## 可用检索策略
1. **semantic_scholar_search** — 学术论文语义检索（推荐，覆盖面广）
2. **arxiv_search** — arXiv 预印本检索（最新成果）
3. **vector_store_query** — 本地知识库检索（已读过的论文）
4. **semantic_scholar_details** — 获取特定论文的详细信息

## 检索规则
1. 对每个子问题至少用 2 种不同的检索策略
2. 优先搜索英文关键词（学术论文以英文为主）
3. 注意年份过滤，优先最近 3 年的论文
4. 如果某次检索结果不理想，尝试改写 query（加同义词、换表述）
5. 如果收到 Critic 的反馈说某方面论文不足，重点补充

## 输出
汇总你找到的所有相关论文，按子问题分组。"""
