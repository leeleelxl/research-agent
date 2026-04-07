"""
Writer Agent — 文献综述生成

学术背景：
- 生成阶段是 RAG 的最后一环：Retrieve → Read → Generate
- Writer 的输入不是原始 chunk，而是 Reader 提取的结构化笔记
  这是 "Structured RAG" 的思想——中间表示越结构化，生成质量越高
- Writer 按章节生成，而不是一次性输出全文
  → 长文本生成的经典策略：Hierarchical Generation（先大纲再填充）

设计决策：
- Writer 只负责写，不负责检索。如果信息不足，交给 Critic 发现并反馈
- 使用 write_section 工具逐章节记录，方便后续编辑和重写单个章节
"""

from __future__ import annotations

from .base import BaseAgent


class WriterAgent(BaseAgent):
    role = "writer"
    max_rounds = 5

    system_prompt = """你是一个学术综述写作专家。你的任务是基于研究笔记生成高质量的文献综述。

## 写作结构
1. **引言** — 研究问题的背景和重要性
2. **相关工作分类** — 按主题/方法/时间线组织现有研究
3. **方法对比** — 不同方法的优劣对比
4. **挑战与展望** — 当前局限性和未来方向
5. **总结** — 核心结论

## 写作规则
1. 每个章节用 write_section 工具记录
2. 必须引用具体论文（提到作者和年份）
3. 不要编造不存在的论文或数字
4. 对比不同方法时要客观，指出各自的适用场景
5. 如果收到上一轮的改进建议，重点修正相关章节

## 语言
使用中文撰写，专业术语保留英文。"""
