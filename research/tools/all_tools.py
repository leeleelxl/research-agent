"""
工具注册入口 — 注册所有工具到 ToolRegistry，按 Agent 角色分配
"""

from __future__ import annotations

from .registry import ToolRegistry
from . import search, pdf_parser, embedding


def create_registry() -> ToolRegistry:
    """创建并注册所有工具"""
    reg = ToolRegistry()

    # ============================================================
    # Planner 工具
    # ============================================================
    @reg.register(
        name="decompose_question",
        description="将研究问题分解为多个具体的子问题（由 LLM 在推理中完成，此工具仅做记录）",
        parameters={
            "type": "object",
            "properties": {
                "sub_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "分解后的子问题列表",
                }
            },
            "required": ["sub_questions"],
        },
        roles=["planner"],
    )
    def decompose_question(sub_questions: list[str]) -> str:
        return f"已记录 {len(sub_questions)} 个子问题：\n" + "\n".join(f"  {i+1}. {q}" for i, q in enumerate(sub_questions))

    # ============================================================
    # Retriever 工具
    # ============================================================
    @reg.register(
        name="semantic_scholar_search",
        description="在 Semantic Scholar 上搜索学术论文，返回标题、摘要、引用数等信息",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词（建议英文）"},
                "limit": {"type": "integer", "description": "返回数量，默认 5", "default": 5},
                "year": {"type": "string", "description": "年份范围，如 '2023-2026'"},
            },
            "required": ["query"],
        },
        roles=["retriever"],
    )
    def _ss_search(query: str, limit: int = 5, year: str | None = None) -> str:
        return search.semantic_scholar_search(query, limit, year)

    @reg.register(
        name="semantic_scholar_details",
        description="获取指定论文的详细信息，包括完整摘要、TL;DR 和参考文献",
        parameters={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Semantic Scholar paper ID 或 arXiv ID（如 'arxiv:2310.12931'）"},
            },
            "required": ["paper_id"],
        },
        roles=["retriever", "reader"],
    )
    def _ss_details(paper_id: str) -> str:
        return search.semantic_scholar_details(paper_id)

    @reg.register(
        name="arxiv_search",
        description="在 arXiv 上搜索论文",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "limit": {"type": "integer", "description": "返回数量，默认 5", "default": 5},
            },
            "required": ["query"],
        },
        roles=["retriever"],
    )
    def _arxiv_search(query: str, limit: int = 5) -> str:
        return search.arxiv_search(query, limit)

    @reg.register(
        name="vector_store_query",
        description="在本地知识库中进行向量相似度检索",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索文本"},
                "top_k": {"type": "integer", "description": "返回数量", "default": 5},
            },
            "required": ["query"],
        },
        roles=["retriever"],
    )
    def _vector_query(query: str, top_k: int = 5) -> str:
        # 运行时由 Retriever Agent 绑定具体的 vector store
        return "向量库未初始化（需在运行时绑定）"

    # ============================================================
    # Reader 工具
    # ============================================================
    @reg.register(
        name="download_arxiv_pdf",
        description="下载 arXiv 论文 PDF 到本地",
        parameters={
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string", "description": "arXiv ID，如 '2310.12931'"},
            },
            "required": ["arxiv_id"],
        },
        roles=["reader"],
    )
    def _download_pdf(arxiv_id: str) -> str:
        return pdf_parser.download_arxiv_pdf(arxiv_id)

    @reg.register(
        name="parse_pdf",
        description="解析 PDF 文件，提取纯文本内容",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "PDF 文件路径"},
            },
            "required": ["path"],
        },
        roles=["reader"],
    )
    def _parse_pdf(path: str) -> str:
        text = pdf_parser.parse_pdf(path)
        # 截断避免 context 爆炸
        if len(text) > 5000:
            return text[:5000] + f"\n\n... (截断，全文 {len(text)} 字符)"
        return text

    @reg.register(
        name="extract_paper_info",
        description="从论文文本中提取结构化信息（方法、实验、结论）",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "论文标题"},
                "key_findings": {"type": "string", "description": "关键发现"},
                "methodology": {"type": "string", "description": "研究方法"},
                "limitations": {"type": "string", "description": "局限性"},
            },
            "required": ["title", "key_findings"],
        },
        roles=["reader"],
    )
    def _extract_info(title: str, key_findings: str, methodology: str = "", limitations: str = "") -> str:
        return f"已记录论文信息：{title}"

    # ============================================================
    # Writer 工具
    # ============================================================
    @reg.register(
        name="write_section",
        description="写入综述的一个章节",
        parameters={
            "type": "object",
            "properties": {
                "section_title": {"type": "string", "description": "章节标题"},
                "content": {"type": "string", "description": "章节内容"},
            },
            "required": ["section_title", "content"],
        },
        roles=["writer"],
    )
    def _write_section(section_title: str, content: str) -> str:
        return f"已写入章节: {section_title} ({len(content)} 字)"

    # ============================================================
    # Critic 工具
    # ============================================================
    @reg.register(
        name="score_review",
        description="为文献综述各维度打分",
        parameters={
            "type": "object",
            "properties": {
                "coverage": {"type": "number", "description": "覆盖度评分 (0-10)"},
                "accuracy": {"type": "number", "description": "准确性评分 (0-10)"},
                "coherence": {"type": "number", "description": "连贯性评分 (0-10)"},
                "depth": {"type": "number", "description": "深度评分 (0-10)"},
                "gaps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "缺失的方面",
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "改进建议",
                },
            },
            "required": ["coverage", "accuracy", "coherence", "depth"],
        },
        roles=["critic"],
    )
    def _score_review(coverage: float, accuracy: float, coherence: float, depth: float,
                      gaps: list[str] | None = None, suggestions: list[str] | None = None) -> str:
        avg = (coverage + accuracy + coherence + depth) / 4
        return (
            f"评分结果：覆盖度={coverage}, 准确性={accuracy}, "
            f"连贯性={coherence}, 深度={depth}, 综合={avg:.1f}\n"
            f"不足: {', '.join(gaps or ['无'])}\n"
            f"建议: {', '.join(suggestions or ['无'])}"
        )

    return reg
