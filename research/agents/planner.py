"""
Planner Agent — 研究问题分解与策略规划

学术背景：
- Plan-and-Execute 范式（Wang et al., 2023）：先生成计划，再逐步执行
  vs ReAct（Yao et al., ICLR 2023）边想边做
  Plan-and-Execute 的优势是有全局视角，不容易跑偏
- 当 Critic 反馈不合格时，Planner 会修正计划
  这是 Reflexion（Shinn et al., NeurIPS 2023）的思想：从失败中反思，改进策略

设计决策：
- Planner 不做检索，只做规划。职责单一，prompt 短，推理质量高
- 输出结构化的子问题列表，而不是自然语言计划。方便下游 Agent 消费
"""

from __future__ import annotations

from .base import BaseAgent


class PlannerAgent(BaseAgent):
    role = "planner"
    max_rounds = 3

    system_prompt = """你是一个学术研究规划专家。你的任务是将用户的研究问题分解为具体的、可检索的子问题。

## 规则
1. 将研究问题分解为 3-5 个子问题
2. 每个子问题应该足够具体，可以直接用于论文检索
3. 子问题应该覆盖：背景/定义、核心方���、最新进展、对比/评估、挑战与未来方向
4. 使用 decompose_question 工具记录你的分解结果
5. 如果收到上一轮的反馈（gaps），优先补充这些方面的子问题

## 输出格式
先简要分析研究问题的范围，然后调用 decompose_question 工具。"""
