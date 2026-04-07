"""
Critic Agent — 综述质量评估（Agent-as-a-Judge）

学术背景：
- LLM-as-a-Judge（Zheng et al., NeurIPS 2023）：用 LLM 评估 LLM 输出
  简单但粗糙——LLM 只能基于文本打分，无法验证事实
- Agent-as-a-Judge（Pan et al., ICML 2024 Workshop）：用 Agent 评估 Agent
  比 LLM-as-Judge 强在：Agent 可以调用工具验证（如检查引用是否存在）
  论文数据：与人类专家一致性 ~90%，vs LLM-as-Judge 的 ~70%

我们的 Critic 实现了 Agent-as-a-Judge：
  1. 多维度评分（不是单一分数）
  2. 识别具体的 gaps（哪些子问题没被覆盖）
  3. 给出可执行的 suggestions（不是泛泛的"需要改进"）
  4. 评估结果驱动 Retriever 的反馈循环（Reflexion）

设计决策：
- Critic 的评分会被记录到 ExperienceStore
  → 下次 Retriever 检索时可以参考历史评分改进 query
  → 这就是"经验增强检索"的经验来源
- 多维度评分（覆盖度/准确性/连贯性/深度）比单一分数更有用
  → 不同维度低分对应不同的改进动作
"""

from __future__ import annotations

from .base import BaseAgent


class CriticAgent(BaseAgent):
    role = "critic"
    max_rounds = 3

    system_prompt = """你是一个严格的学术评审专家。你的任务是评估文献综述的质量，给出具体的分数和改进建议。

## 评估维度（每项 0-10 分）
1. **覆盖度** (coverage) — 研究问题的各个方面是否都被讨论？
2. **准确性** (accuracy) — 引用的论文信息是否准确？是否有编造？
3. **连贯性** (coherence) — 文章结构是否清晰？段落之间是否有逻辑过渡？
4. **深度** (depth) — 分析是否有洞见？是否只是简单堆砌论文？

## 评估规则
1. 必须使用 score_review 工具记录你的评分
2. 如果某个维度低于 7 分，必须在 gaps 中具体说明缺什么
3. suggestions 要具体可执行，如"需要补充 X 方向 2024 年之后的论文"
4. 综合分 = 四个维度的平均值
5. 综合分 ≥ 7.0 视为通过

## 评分标准
- 9-10: 优秀，接近发表水平
- 7-8: 良好，覆盖全面但可以更深入
- 5-6: 一般，有明显遗漏或浅尝辄止
- 3-4: 较差，大量信息缺失
- 1-2: 很差，几乎没有有效内容"""
