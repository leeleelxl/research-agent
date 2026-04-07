"""
PDF 解析工具 — 提取论文文本

优先使用 PyMuPDF（fitz），回退到 pdfplumber，再回退到纯文本提示。
"""

from __future__ import annotations

import urllib.request
from pathlib import Path


def parse_pdf(path: str) -> str:
    """解析 PDF 文件，返回纯文本"""
    path = Path(path)
    if not path.exists():
        return f"文件不存在: {path}"

    # 尝试 PyMuPDF
    try:
        import fitz
        doc = fitz.open(str(path))
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip() if text.strip() else "PDF 解析为空（可能是扫描件）"
    except ImportError:
        pass

    # 尝试 pdfplumber
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip() if text.strip() else "PDF 解析为空"
    except ImportError:
        pass

    return "错误：需要安装 PyMuPDF (pip install pymupdf) 或 pdfplumber (pip install pdfplumber)"


def download_arxiv_pdf(arxiv_id: str, save_dir: str = "papers") -> str:
    """下载 arXiv 论文 PDF

    Args:
        arxiv_id: arXiv ID，如 "2310.12931"
        save_dir: 保存目录
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    filename = save_path / f"{arxiv_id.replace('/', '_')}.pdf"

    if filename.exists():
        return str(filename)

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ReSearch-Agent/0.1"})
        with opener.open(req, timeout=60) as resp:
            filename.write_bytes(resp.read())
        return str(filename)
    except Exception as e:
        return f"下载失败: {e}"
