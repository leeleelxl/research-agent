"""
工具注册表 — 统一管理所有 Agent 可用的工具

每个工具是一个函数 + schema，支持：
- 按名称调用
- 自动生成 OpenAI function calling 格式的 schema
- 按 Agent 角色过滤可用工具
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    """一个可被 Agent 调用的工具"""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    func: Callable[..., str]
    roles: list[str] = field(default_factory=list)  # 哪些 Agent 可用

    def to_openai_schema(self) -> dict:
        """转为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs) -> str:
        """执行工具"""
        return self.func(**kwargs)


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        roles: list[str] | None = None,
    ) -> Callable:
        """装饰器：注册一个工具

        用法：
            @registry.register(
                name="web_search",
                description="搜索网页",
                parameters={"type": "object", "properties": {...}},
                roles=["retriever"],
            )
            def web_search(query: str) -> str:
                ...
        """
        def decorator(func: Callable) -> Callable:
            tool = Tool(
                name=name,
                description=description,
                parameters=parameters,
                func=func,
                roles=roles or [],
            )
            self._tools[name] = tool
            return func
        return decorator

    def add(self, tool: Tool):
        """直接添加一个 Tool 对象"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(self, name: str, args: dict[str, Any]) -> str:
        """按名称执行工具"""
        tool = self._tools.get(name)
        if tool is None:
            return f"错误：未知工具 '{name}'"
        try:
            return tool.execute(**args)
        except Exception as e:
            return f"工具执行错误 [{name}]: {e}"

    def get_schemas(self, role: str | None = None) -> list[dict]:
        """获取 OpenAI function calling 格式的 schema 列表"""
        tools = self._tools.values()
        if role:
            tools = [t for t in tools if not t.roles or role in t.roles]
        return [t.to_openai_schema() for t in tools]

    def list_tools(self, role: str | None = None) -> list[str]:
        """列出工具名称"""
        if role:
            return [t.name for t in self._tools.values() if not t.roles or role in t.roles]
        return list(self._tools.keys())

    @property
    def count(self) -> int:
        return len(self._tools)
