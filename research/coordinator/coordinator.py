"""
多 Agent 协调器 — 管理 5 个 Agent 的执行流程和反馈循环

执行流程：
  Planner → Retriever → Reader → Writer → Critic
                ↑                            │
                └──── 不合格时反馈循环 ────────┘

协调器负责：
1. 按顺序调度 Agent
2. 传递 Agent 之间的数据（SharedState）
3. 管理反馈循环（Critic 打回时触发 Retriever 补充检索）
4. 记录完整执行轨迹
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..agents.base import BaseAgent, AgentResult


@dataclass
class SharedState:
    """Agent 之间的共享状态"""
    question: str = ""
    sub_questions: list[str] = field(default_factory=list)
    papers: list[dict[str, Any]] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)
    draft: str = ""
    review: dict[str, Any] = field(default_factory=dict)
    final_output: str = ""

    # 执行追踪
    rounds_completed: int = 0
    agent_results: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})


@dataclass
class CriticFeedback:
    """Critic 的评估反馈"""
    score: float  # 0-10
    passed: bool
    gaps: list[str] = field(default_factory=list)  # 缺失的方面
    suggestions: list[str] = field(default_factory=list)


class ResearchCoordinator:
    """多 Agent 研究协调器"""

    def __init__(
        self,
        planner: BaseAgent,
        retriever: BaseAgent,
        reader: BaseAgent,
        writer: BaseAgent,
        critic: BaseAgent,
        max_rounds: int = 3,
        pass_threshold: float = 7.0,
    ):
        self.planner = planner
        self.retriever = retriever
        self.reader = reader
        self.writer = writer
        self.critic = critic
        self.max_rounds = max_rounds
        self.pass_threshold = pass_threshold

    def run(self, question: str) -> SharedState:
        """执行完整研究流程"""
        state = SharedState(question=question)

        # Phase 1: Planner 分解问题
        state = self._run_agent("planner", self.planner, state,
                                task=f"请将以下研究问题分解为 3-5 个具体的子问题：\n{question}")

        for round_num in range(self.max_rounds):
            state.rounds_completed = round_num + 1

            # Phase 2: Retriever 检索论文
            retriever_task = self._build_retriever_task(state)
            state = self._run_agent("retriever", self.retriever, state, task=retriever_task)

            # Phase 3: Reader 精读
            reader_task = self._build_reader_task(state)
            state = self._run_agent("reader", self.reader, state, task=reader_task)

            # Phase 4: Writer 生成综述
            writer_task = self._build_writer_task(state)
            state = self._run_agent("writer", self.writer, state, task=writer_task)

            # Phase 5: Critic 评估
            critic_task = self._build_critic_task(state)
            state = self._run_agent("critic", self.critic, state, task=critic_task)

            # 判断是否通过
            score = state.review.get("score", 0)
            if score >= self.pass_threshold:
                state.final_output = state.draft
                break

            # 未通过 → 下一轮用 Critic 的反馈改进

        if not state.final_output:
            state.final_output = state.draft  # 达到最大轮次，返回最后版本

        return state

    def _run_agent(self, name: str, agent: BaseAgent, state: SharedState, task: str) -> SharedState:
        """运行单个 Agent 并更新共享状态"""
        context = {
            "question": state.question,
            "sub_questions": state.sub_questions,
            "papers_count": len(state.papers),
            "round": state.rounds_completed,
        }
        if state.review.get("gaps"):
            context["previous_gaps"] = state.review["gaps"]

        result = agent.run(task, context=context)

        # 更新 token 统计
        state.total_tokens["input"] += result.total_tokens.get("input", 0)
        state.total_tokens["output"] += result.total_tokens.get("output", 0)

        # 记录
        state.agent_results.append({
            "agent": name,
            "round": state.rounds_completed,
            "output_length": len(result.output),
            "tool_calls": len(result.tool_calls_made),
            "rounds": result.rounds,
            "tokens": result.total_tokens,
        })

        return state

    def _build_retriever_task(self, state: SharedState) -> str:
        task = f"针对以下子问题检索相关学术论文：\n"
        for i, q in enumerate(state.sub_questions, 1):
            task += f"{i}. {q}\n"
        if state.review.get("gaps"):
            task += f"\n上一轮评审发现的不足：{', '.join(state.review['gaps'])}"
            task += "\n请重点补充这些方面的论文。"
        return task

    def _build_reader_task(self, state: SharedState) -> str:
        return f"精读检索到的 {len(state.papers)} 篇论文，提取每篇的核心方法、实验结果和关键结论。"

    def _build_writer_task(self, state: SharedState) -> str:
        task = f"基于以下子问题和论文笔记，生成一篇结构化的文献综述：\n"
        task += f"研究问题：{state.question}\n"
        task += f"子问题数量：{len(state.sub_questions)}\n"
        task += f"论文数量：{len(state.papers)}\n"
        if state.review.get("suggestions"):
            task += f"\n上一轮的改进建议：{', '.join(state.review['suggestions'])}"
        return task

    def _build_critic_task(self, state: SharedState) -> str:
        return (
            f"评估以下文献综述的质量（满分 10 分）：\n\n"
            f"研究问题：{state.question}\n"
            f"综述长度：{len(state.draft)} 字\n\n"
            f"请从以下维度评分并给出具体反馈：\n"
            f"1. 覆盖度：子问题是否都被回答\n"
            f"2. 准确性：信息是否有论文支撑\n"
            f"3. 连贯性：文章结构是否清晰\n"
            f"4. 深度：分析是否有洞见\n"
        )

    def summary(self, state: SharedState) -> str:
        """生成执行摘要"""
        lines = [
            f"研究问题: {state.question}",
            f"执行轮次: {state.rounds_completed}/{self.max_rounds}",
            f"子问题: {len(state.sub_questions)} 个",
            f"检索论文: {len(state.papers)} 篇",
            f"最终评分: {state.review.get('score', '—')}",
            f"Token 消耗: input={state.total_tokens['input']}, output={state.total_tokens['output']}",
            f"Agent 调用: {len(state.agent_results)} 次",
        ]
        return "\n".join(lines)
