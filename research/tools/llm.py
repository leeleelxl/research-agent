"""
LLM 调用层 — 统一的 OpenAI 兼容 API 客户端

支持 stream 模式（兼容中转站）和 tool calling。
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[dict[str, Any]] | None
    usage: dict[str, int]
    model: str


class LLMClient:
    """OpenAI 兼容的 LLM 客户端（stream 模式）"""

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        default_model: str = "gpt-4o-mini",
        timeout: int = 120,
    ):
        self.api_base = (api_base or os.getenv("OPENAI_API_BASE", "")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.default_model = default_model
        self.timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """发送 chat completion 请求（stream 模式）"""
        model = model or self.default_model
        url = f"{self.api_base}/v1/chat/completions"
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            body["tools"] = tools

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })

        with self._opener.open(req, timeout=self.timeout) as resp:
            content = ""
            tool_calls_data: dict[int, dict] = {}
            usage = {}

            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0].get("delta", {})

                if delta.get("content"):
                    content += delta["content"]
                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        idx = tc["index"]
                        if idx not in tool_calls_data:
                            tool_calls_data[idx] = {
                                "id": "", "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc.get("id"):
                            tool_calls_data[idx]["id"] = tc["id"]
                        if tc.get("function", {}).get("name"):
                            tool_calls_data[idx]["function"]["name"] = tc["function"]["name"]
                        if tc.get("function", {}).get("arguments"):
                            tool_calls_data[idx]["function"]["arguments"] += tc["function"]["arguments"]
                if chunk.get("usage"):
                    usage = chunk["usage"]

        tool_calls = list(tool_calls_data.values()) if tool_calls_data else None

        # Token 统计：优先用 API 返回值，回退到字符数估算
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        if input_tokens == 0 and output_tokens == 0:
            # 中转站 stream 模式不返回 usage，用字符数估算（中文约 1.5 字/token）
            input_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
            output_chars = len(content) + sum(
                len(tc["function"]["arguments"]) for tc in tool_calls_data.values()
            )
            input_tokens = max(1, input_chars // 2)
            output_tokens = max(1, output_chars // 2)

        return LLMResponse(
            content=content or None,
            tool_calls=tool_calls,
            usage={"input": input_tokens, "output": output_tokens},
            model=model,
        )
