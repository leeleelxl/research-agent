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

        print(f"  [{name}] running...")
        result = agent.run(task, context=context)
        print(f"  [{name}] done | tools: {len(result.tool_calls_made)} | output: {len(result.output)} chars")

        # 更新 token 统计
        state.total_tokens["input"] += result.total_tokens.get("input", 0)
        state.total_tokens["output"] += result.total_tokens.get("output", 0)

        # 根据 Agent 角色，把输出写回 SharedState
        self._update_state(name, result, state)

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

    def _update_state(self, agent_name: str, result: AgentResult, state: SharedState):
        """根据 Agent 角色解析输出，更新 SharedState"""
        if agent_name == "planner":
            # 从 tool_calls 或文本中提取子问题
            for tc in result.tool_calls_made:
                if tc["tool"] == "decompose_question":
                    sub_qs = tc["args"].get("sub_questions", [])
                    if sub_qs:
                        state.sub_questions = sub_qs
                        return
            # 回退：从输出文本中按行提取
            lines = result.output.strip().split("\n")
            questions = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                    # 去掉编号前缀
                    clean = line.lstrip("0123456789.-*) ").strip()
                    if clean and len(clean) > 10:
                        questions.append(clean)
            if questions:
                state.sub_questions = questions[:5]

        elif agent_name == "retriever":
            # Retriever 的输出就是检索摘要，存为 papers 信息
            if result.output and len(result.output) > 50:
                state.papers.append({
                    "round": state.rounds_completed,
                    "source": "retriever_output",
                    "content": result.output,
                })
            # 从 tool calls 中也收集
            for tc in result.tool_calls_made:
                if tc["tool"] in ("semantic_scholar_search", "arxiv_search"):
                    state.papers.append({
                        "round": state.rounds_completed,
                        "source": tc["tool"],
                        "query": tc["args"].get("query", ""),
                        "content": tc["result"][:1000],
                    })

        elif agent_name == "reader":
            # Reader 的结构化笔记
            for tc in result.tool_calls_made:
                if tc["tool"] == "extract_paper_info":
                    state.notes.append(tc["args"])
            # 也保存自由文本输出
            if result.output and len(result.output) > 50:
                state.notes.append({
                    "source": "reader_summary",
                    "content": result.output,
                })

        elif agent_name == "writer":
            # Writer 的输出就是综述草稿
            state.draft = result.output
            # 也从 write_section 工具调用中拼接
            if not state.draft or len(state.draft) < 100:
                sections = []
                for tc in result.tool_calls_made:
                    if tc["tool"] == "write_section":
                        title = tc["args"].get("section_title", "")
                        content = tc["args"].get("content", "")
                        sections.append(f"## {title}\n\n{content}")
                if sections:
                    state.draft = "\n\n".join(sections)

        elif agent_name == "critic":
            # Critic 的评分和反馈
            for tc in result.tool_calls_made:
                if tc["tool"] == "score_review":
                    args = tc["args"]
                    scores = [
                        args.get("coverage", 0),
                        args.get("accuracy", 0),
                        args.get("coherence", 0),
                        args.get("depth", 0),
                    ]
                    avg = sum(scores) / len(scores) if scores else 0
                    state.review = {
                        "score": avg,
                        "coverage": args.get("coverage", 0),
                        "accuracy": args.get("accuracy", 0),
                        "coherence": args.get("coherence", 0),
                        "depth": args.get("depth", 0),
                        "gaps": args.get("gaps", []),
                        "suggestions": args.get("suggestions", []),
                    }
                    print(f"  [{agent_name}] score: {avg:.1f} | gaps: {args.get('gaps', [])}")
                    return
            # 回退：从文本中提取
            if result.output:
                state.review = {"score": 0, "text": result.output[:500]}

    def _build_retriever_task(self, state: SharedState) -> str:
        task = f"针对以下子问题检索相关学术论文：\n"
        for i, q in enumerate(state.sub_questions, 1):
            task += f"{i}. {q}\n"
        if state.review.get("gaps"):
            task += f"\n上一轮评审发现的不足：\n"
            for gap in state.review["gaps"][:3]:  # 最多 3 个 gap 避免 prompt 过长
                task += f"- {gap}\n"
            task += "请重点补充这些方面的论文。"
        return task

    def _build_reader_task(self, state: SharedState) -> str:
        task = "精读以下检索到的论文，对每篇用 extract_paper_info 工具提取结构化信息。\n\n"
        task += "## 检索到的论文\n\n"
        for i, paper in enumerate(state.papers[-8:], 1):  # 最近 8 篇
            content = paper.get("content", "")[:400]
            source = paper.get("source", "unknown")
            task += f"### 论文 {i} (来源: {source})\n{content}\n\n"
        if not state.papers:
            task += "(暂无检索结果，请根据研究问题自行分析)\n"
        return task

    def _build_writer_task(self, state: SharedState) -> str:
        task = "基于以下材料，生成一篇结构化的文献综述。\n\n"
        task += f"## 研究问题\n{state.question}\n\n"

        task += "## 子问题\n"
        for i, q in enumerate(state.sub_questions, 1):
            task += f"{i}. {q}\n"

        task += "\n## 论文笔记\n"
        if state.notes:
            for i, note in enumerate(state.notes[-10:], 1):  # 最近 10 条笔记
                if isinstance(note, dict):
                    title = note.get("title", note.get("source", f"笔记{i}"))
                    findings = note.get("key_findings", note.get("content", ""))[:300]
                    method = note.get("methodology", "")[:200]
                    task += f"\n### {title}\n"
                    task += f"- 关键发现: {findings}\n"
                    if method:
                        task += f"- 方法: {method}\n"
        else:
            task += "(暂无结构化笔记，请根据检索结果自行综合)\n"

        if state.review.get("suggestions"):
            task += "\n## 上一轮的改进建议\n"
            for s in state.review["suggestions"][:3]:
                task += f"- {s}\n"

        task += ("\n## 写作要求\n"
                 "1. 必须引用具体论文（作者+年份），不要编造\n"
                 "2. 对比不同方法的优劣\n"
                 "3. 用中文撰写，专业术语保留英文\n"
                 "4. 每个章节用 write_section 工具记录\n")
        return task

    def _build_critic_task(self, state: SharedState) -> str:
        # 给 Critic 看综述原文（截断避免过长）
        draft_preview = state.draft[:3000] if state.draft else "(综述为空)"
        if len(state.draft) > 3000:
            draft_preview += f"\n\n... (共 {len(state.draft)} 字，已截断)"

        return (
            f"评估以下文献综述的质量，使用 score_review 工具给出评分。\n\n"
            f"## 研究问题\n{state.question}\n\n"
            f"## 子问题\n" + "\n".join(f"- {q}" for q in state.sub_questions) + "\n\n"
            f"## 综述内容\n{draft_preview}\n\n"
            f"## 评估维度（每项 0-10 分）\n"
            f"1. 覆盖度 (coverage): 子问题是否都被回答\n"
            f"2. 准确性 (accuracy): 引用的论文是否真实，数据是否有出处\n"
            f"3. 连贯性 (coherence): 结构是否清晰，段落过渡是否自然\n"
            f"4. 深度 (depth): 分析是否有洞见，还是简单堆砌\n\n"
            f"如果某维度低于 7 分，必须在 gaps 中说明具体缺什么。"
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
