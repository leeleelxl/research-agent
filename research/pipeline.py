"""
端到端 Pipeline — 一行代码启动完整研究流程

用法：
    from research.pipeline import run_research
    result = run_research("What are the recent advances in self-evolving agents?")
"""

from __future__ import annotations

import os
from pathlib import Path

from .agents import PlannerAgent, RetrieverAgent, ReaderAgent, WriterAgent, CriticAgent
from .coordinator import ResearchCoordinator
from .tools.llm import LLMClient
from .tools.all_tools import create_registry


def run_research(
    question: str,
    model: str = "gpt-4o-mini",
    max_rounds: int = 2,
    pass_threshold: float = 7.0,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict:
    """运行完整的多 Agent 研究流程

    Args:
        question: 研究问题
        model: 使用的模型
        max_rounds: 最大反馈轮次
        pass_threshold: Critic 通过阈值
        api_base: API 地址（默认从环境变量读取）
        api_key: API Key（默认从环境变量读取）

    Returns:
        包含完整执行结果的字典
    """
    # 初始化 LLM 客户端
    llm = LLMClient(
        api_base=api_base or os.getenv("OPENAI_API_BASE", ""),
        api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
        default_model=model,
    )

    # 创建工具注册表
    registry = create_registry()

    # 创建 5 个 Agent
    planner = PlannerAgent(llm=llm, registry=registry)
    retriever = RetrieverAgent(llm=llm, registry=registry)
    reader = ReaderAgent(llm=llm, registry=registry)
    writer = WriterAgent(llm=llm, registry=registry)
    critic = CriticAgent(llm=llm, registry=registry)

    # 创建协调器
    coordinator = ResearchCoordinator(
        planner=planner,
        retriever=retriever,
        reader=reader,
        writer=writer,
        critic=critic,
        max_rounds=max_rounds,
        pass_threshold=pass_threshold,
    )

    # 执行
    print(f"🔬 研究问题: {question}")
    print(f"📋 模型: {model} | 最大轮次: {max_rounds} | 通过阈值: {pass_threshold}")
    print("=" * 60)

    state = coordinator.run(question)

    # 输出摘要
    print("\n" + "=" * 60)
    print("📊 执行摘要")
    print(coordinator.summary(state))

    return {
        "question": state.question,
        "sub_questions": state.sub_questions,
        "papers": state.papers,
        "draft": state.draft,
        "final_output": state.final_output,
        "review": state.review,
        "rounds": state.rounds_completed,
        "tokens": state.total_tokens,
        "agent_results": state.agent_results,
    }
