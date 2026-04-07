"""
学术检索工具 — Semantic Scholar + arXiv API

无需 API key，直接调用公开接口。
"""

from __future__ import annotations

import json
import urllib.request
import urllib.parse
import time
from typing import Any


def semantic_scholar_search(query: str, limit: int = 5, year: str | None = None) -> str:
    """Semantic Scholar 论文检索

    Args:
        query: 搜索关键词
        limit: 返回数量
        year: 年份过滤，如 "2023-2026"
    """
    params = {
        "query": query,
        "limit": str(limit),
        "fields": "title,abstract,year,authors,citationCount,url,externalIds",
    }
    if year:
        params["year"] = year

    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{urllib.parse.urlencode(params)}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ReSearch-Agent/0.1"})
        with opener.open(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return f"检索失败: {e}"

    papers = data.get("data", [])
    if not papers:
        return f"未找到关于 '{query}' 的论文"

    results = []
    for p in papers:
        authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:3])
        if len(p.get("authors", [])) > 3:
            authors += " et al."
        abstract = (p.get("abstract") or "")[:200]
        results.append(
            f"**{p.get('title', 'Untitled')}** ({p.get('year', '?')})\n"
            f"  作者: {authors}\n"
            f"  引用: {p.get('citationCount', 0)}\n"
            f"  摘要: {abstract}...\n"
            f"  URL: {p.get('url', '')}"
        )

    return f"找到 {len(results)} 篇论文：\n\n" + "\n\n".join(results)


def semantic_scholar_details(paper_id: str) -> str:
    """获取论文详细信息（含完整摘要和引用）"""
    fields = "title,abstract,year,authors,citationCount,references.title,references.year,tldr"
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields={fields}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ReSearch-Agent/0.1"})
        with opener.open(req, timeout=15) as resp:
            p = json.loads(resp.read().decode())
    except Exception as e:
        return f"获取失败: {e}"

    authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:5])
    tldr = p.get("tldr", {}).get("text", "无")
    abstract = p.get("abstract", "无")

    refs = p.get("references", [])[:10]
    ref_text = "\n".join(f"  - {r.get('title', '?')} ({r.get('year', '?')})" for r in refs)

    return (
        f"**{p.get('title')}** ({p.get('year')})\n"
        f"作者: {authors}\n"
        f"引用数: {p.get('citationCount', 0)}\n"
        f"TL;DR: {tldr}\n\n"
        f"摘要:\n{abstract}\n\n"
        f"主要参考文献:\n{ref_text}"
    )


def arxiv_search(query: str, limit: int = 5) -> str:
    """arXiv 论文检索"""
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": "0",
        "max_results": str(limit),
        "sortBy": "relevance",
    })
    url = f"http://export.arxiv.org/api/query?{params}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ReSearch-Agent/0.1"})
        with opener.open(req, timeout=15) as resp:
            raw = resp.read().decode()
    except Exception as e:
        return f"arXiv 检索失败: {e}"

    # 简单的 XML 解析（避免依赖 lxml）
    results = []
    entries = raw.split("<entry>")[1:]  # 跳过 header
    for entry in entries[:limit]:
        title = _extract_xml(entry, "title").replace("\n", " ").strip()
        summary = _extract_xml(entry, "summary").replace("\n", " ").strip()[:200]
        published = _extract_xml(entry, "published")[:10]

        # 提取作者
        authors = []
        for author_block in entry.split("<author>")[1:]:
            name = _extract_xml(author_block, "name")
            if name:
                authors.append(name)
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."

        # 提取 arXiv ID
        arxiv_id = ""
        if "<id>" in entry:
            arxiv_id = _extract_xml(entry, "id").split("/abs/")[-1]

        results.append(
            f"**{title}** ({published})\n"
            f"  作者: {author_str}\n"
            f"  arXiv: {arxiv_id}\n"
            f"  摘要: {summary}..."
        )

    if not results:
        return f"arXiv 未找到关于 '{query}' 的论文"

    return f"arXiv 找到 {len(results)} 篇：\n\n" + "\n\n".join(results)


def _extract_xml(text: str, tag: str) -> str:
    """简单 XML 标签提取"""
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    if start == -1 or end == -1:
        # 尝试带属性的标签
        start = text.find(f"<{tag} ")
        if start != -1:
            start = text.find(">", start) + 1
            end = text.find(f"</{tag}>")
    else:
        start += len(f"<{tag}>")
    if start == -1 or end == -1:
        return ""
    return text[start:end].strip()
