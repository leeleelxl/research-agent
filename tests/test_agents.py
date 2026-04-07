"""Agent 基类和工具注册测试"""

from research.tools.registry import Tool, ToolRegistry
from research.agents.base import BaseAgent, AgentResult


class TestToolRegistry:
    def test_register_and_execute(self):
        reg = ToolRegistry()

        @reg.register(
            name="add",
            description="加法",
            parameters={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
            roles=["calculator"],
        )
        def add(a: int, b: int) -> str:
            return str(a + b)

        assert reg.count == 1
        assert reg.execute("add", {"a": 3, "b": 4}) == "7"

    def test_role_filtering(self):
        reg = ToolRegistry()
        reg.add(Tool(name="t1", description="", parameters={}, func=lambda: "", roles=["planner"]))
        reg.add(Tool(name="t2", description="", parameters={}, func=lambda: "", roles=["retriever"]))
        reg.add(Tool(name="t3", description="", parameters={}, func=lambda: "", roles=["planner", "retriever"]))

        assert set(reg.list_tools("planner")) == {"t1", "t3"}
        assert set(reg.list_tools("retriever")) == {"t2", "t3"}
        assert len(reg.list_tools()) == 3

    def test_schema_generation(self):
        reg = ToolRegistry()
        reg.add(Tool(
            name="search",
            description="搜索",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            func=lambda q: q,
        ))
        schemas = reg.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "search"

    def test_unknown_tool(self):
        reg = ToolRegistry()
        result = reg.execute("nonexistent", {})
        assert "未知工具" in result
