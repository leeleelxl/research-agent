"""
Agent 基类 — 所有 Agent 的共同行为

每个 Agent 有：
- 独立的 system prompt
- 独立的工具集（通过 role 过滤）
- 独立的记忆（短期 + 长期）
- ReAct 执行循环（思考→工具调用→观察→...→最终回答）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..tools.llm import LLMClient, LLMResponse
from ..tools.registry import ToolRegistry


@dataclass
class AgentResult:
    """Agent 执行结果"""
    output: str
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})
    rounds: int = 0


class BaseAgent:
    """Agent 基类"""

    role: str = "base"
    system_prompt: str = "You are a helpful assistant."
    max_rounds: int = 10

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        model: str | None = None,
    ):
        self.llm = llm
        self.registry = registry
        self.model = model
        self._short_memory: list[dict[str, Any]] = []  # 当前任务的对话记忆

    @property
    def available_tools(self) -> list[dict]:
        """该 Agent 可用的工具 schema"""
        return self.registry.get_schemas(role=self.role)

    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        """执行任务（ReAct 循环）"""
        messages = [
            {"role": "system", "content": self._build_system_prompt(context)},
            {"role": "user", "content": task},
        ]
        tools = self.available_tools or None
        tool_calls_made = []
        total_tokens = {"input": 0, "output": 0}

        for round_num in range(self.max_rounds):
            response = self.llm.chat(
                messages=messages,
                model=self.model,
                tools=tools,
            )
            total_tokens["input"] += response.usage.get("input", 0)
            total_tokens["output"] += response.usage.get("output", 0)

            # 没有工具调用 → Agent 完成
            if not response.tool_calls:
                return AgentResult(
                    output=response.content or "",
                    tool_calls_made=tool_calls_made,
                    messages=messages,
                    total_tokens=total_tokens,
                    rounds=round_num + 1,
                )

            # 有工具调用 → 执行并继续
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if response.content:
                assistant_msg["content"] = response.content
            assistant_msg["tool_calls"] = response.tool_calls
            messages.append(assistant_msg)

            for tc in response.tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                result = self.registry.execute(func_name, func_args)
                tool_calls_made.append({
                    "tool": func_name,
                    "args": func_args,
                    "result": result[:500],  # 截断避免 context 爆炸
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        # 达到最大轮次
        return AgentResult(
            output="达到最大执行轮次",
            tool_calls_made=tool_calls_made,
            messages=messages,
            total_tokens=total_tokens,
            rounds=self.max_rounds,
        )

    def _build_system_prompt(self, context: dict[str, Any] | None = None) -> str:
        """构建 system prompt，子类可重写"""
        prompt = self.system_prompt
        if context:
            prompt += f"\n\n## 上下文\n{json.dumps(context, ensure_ascii=False, indent=2)}"

        tool_names = self.registry.list_tools(role=self.role)
        if tool_names:
            prompt += f"\n\n## 可用工具\n{', '.join(tool_names)}"

        return prompt
