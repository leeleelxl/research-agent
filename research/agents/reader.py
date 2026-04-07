"""
Reader Agent — 论文精读与信息提取

学术背景：
- 在 RAG 系统中，Document Understanding 是关键但常被忽视的环节
- 直接把原始 chunk 喂给 LLM 效果差，因为学术论文结构复杂（表格、引用关系）
- Reader 的作用是 Structured Extraction：把非结构化文本转成结构化笔记
  （方法、实验设置、关键数字、局限性）
- 这对应 LlamaIndex 中的 "Document Agent" 概念

设计决策：
- Reader 可以访问 semantic_scholar_details（获取论文详情和引用关系）
- Reader 可以下载和解析 PDF（深度阅读）
- 输出是结构化的 extract_paper_info，而不是自由文本
  → 结���化输出对下游 Writer 更友好，也便于存入向量库
"""

from __future__ import annotations

from .base import BaseAgent


class ReaderAgent(BaseAgent):
    role = "reader"
    max_rounds = 8  # 多篇论文需要多轮

    system_prompt = """你是一个学术论文精读专家。你的任务是深入阅读论文并提取结构化信息。

## 工作流程
1. 用 semantic_scholar_details 获取论文的完整摘要和 TL;DR
2. 如果需要更深入的信息，用 download_arxiv_pdf + parse_pdf 获取全文
3. 对每篇论文，用 extract_paper_info 记录结构化信息

## 提取要点
对每篇论文提取：
- **title**: 论文标题
- **key_findings**: 核心发现（1-3 句话）
- **methodology**: 研究方法（用了什么模型/算法/数据集）
- **limitations**: 局限性和���足

## 注意
- 只提取论文中明确写的信息，不要推测
- 关注具体数字（准确率、提升百分比等）
- 注意论文之间的引用关系和对比"""
