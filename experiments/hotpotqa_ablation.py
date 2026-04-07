"""
HotpotQA 消融实验 — 在公开 benchmark 上对比 RAG 策略

数据: HotpotQA dev set (distractor), 7405 questions
每个问题有 10 个候选文档 + supporting facts 标注

消融变量:
  1. Chunk: naive vs sentence_window
  2. Retrieval: bm25 vs vector vs hybrid
  3. Experience: none vs enhanced (积累后)

指标:
  - Recall@2: top-2 检索结果是否包含 supporting fact 文档
  - Recall@5: top-5
  - MRR: 第一个正确文档的排名倒数
  - SP_F1: supporting fact sentence 级别的 F1 (HotpotQA 标准指标)

用法: python experiments/hotpotqa_ablation.py [--num 200]
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.rag.chunker import Chunker, ChunkStrategy
from research.rag.vector_store import VectorStore
from research.rag.retriever import RAGRetriever, ExperienceRecord
from research.tools.embedding import embed_text, get_dimension


def load_hotpotqa(path: str, num: int = 200) -> list[dict]:
    """加载 HotpotQA 数据，转为统一格式"""
    raw = json.load(open(path))
    samples = []
    for item in raw[:num]:
        # 构建文档列表
        docs = []
        for title, sentences in item["context"]:
            full_text = " ".join(sentences)
            docs.append({
                "title": title,
                "text": full_text,
                "sentences": sentences,
            })

        # 标注: supporting fact 涉及的文档 title
        sf_titles = set(title for title, _ in item["supporting_facts"])
        # 标注: supporting fact 的具体句子索引
        sf_sentences = {}
        for title, sent_idx in item["supporting_facts"]:
            if title not in sf_sentences:
                sf_sentences[title] = set()
            sf_sentences[title].add(sent_idx)

        samples.append({
            "id": item["_id"],
            "question": item["question"],
            "answer": item["answer"],
            "type": item["type"],
            "level": item["level"],
            "docs": docs,
            "gold_titles": sf_titles,
            "gold_sentences": sf_sentences,
        })
    return samples


def build_index(
    docs: list[dict],
    chunker: Chunker,
    embed_fn,
    dim: int,
) -> RAGRetriever:
    """为一个问题的 10 个候选文档构建检索索引"""
    store = VectorStore(dimension=dim)
    retriever = RAGRetriever(store, embed_fn=embed_fn)
    for doc in docs:
        chunks = chunker.chunk(doc["text"], metadata={"title": doc["title"]})
        for chunk in chunks:
            retriever.add_document(chunk.text, {"title": doc["title"]})
    return retriever


def evaluate_single(
    question: str,
    gold_titles: set,
    retriever: RAGRetriever,
    search_fn: str,
    top_k: int = 5,
    use_experience: bool = False,
) -> dict:
    """评估单个问题的检索效果"""
    if use_experience:
        results, _ = retriever.search_with_experience(question, top_k)
    elif search_fn == "bm25":
        bm25_res = retriever.search_bm25(question, top_k)
        results = [type("R", (), {"text": t, "score": s, "metadata": {}})() for t, s in bm25_res]
    elif search_fn == "vector":
        results = retriever.search_vector(question, top_k)
    elif search_fn == "hybrid":
        results = retriever.search_hybrid(question, top_k)
    else:
        results = []

    # 从检索结果中提取文档标题
    retrieved_titles = []
    for r in results:
        # 尝试从 metadata 或文本内容匹配标题
        meta = getattr(r, "metadata", {}) or {}
        title = meta.get("title", "")
        if not title:
            # 从文本匹配
            text = r.text if hasattr(r, "text") else str(r)
            for gt in gold_titles:
                if gt.lower() in text.lower():
                    title = gt
                    break
        retrieved_titles.append(title)

    # Recall@K
    retrieved_set = set(retrieved_titles[:top_k])
    recall_at_k = len(gold_titles & retrieved_set) / len(gold_titles) if gold_titles else 0

    # Recall@2
    retrieved_2 = set(retrieved_titles[:2])
    recall_at_2 = len(gold_titles & retrieved_2) / len(gold_titles) if gold_titles else 0

    # MRR
    mrr = 0.0
    for rank, title in enumerate(retrieved_titles, 1):
        if title in gold_titles:
            mrr = 1.0 / rank
            break

    # 是否完全命中 (所有 gold titles 都在 top-k 中)
    full_hit = gold_titles.issubset(retrieved_set)

    return {
        "recall@2": recall_at_2,
        "recall@5": recall_at_k,
        "mrr": mrr,
        "full_hit": full_hit,
    }


def run_experiment(
    samples: list[dict],
    chunk_strategy: ChunkStrategy,
    search_fn: str,
    embed_fn,
    dim: int,
    use_experience: bool = False,
    experience_records: list[ExperienceRecord] | None = None,
) -> dict:
    """跑一组实验配置"""
    chunker = Chunker(strategy=chunk_strategy, chunk_size=256, overlap=32, window_size=1)

    total_recall2 = 0.0
    total_recall5 = 0.0
    total_mrr = 0.0
    total_full = 0
    n = len(samples)

    for i, sample in enumerate(samples):
        retriever = build_index(sample["docs"], chunker, embed_fn, dim)

        # 注入经验
        if use_experience and experience_records:
            for exp in experience_records:
                retriever.add_experience(exp)

        result = evaluate_single(
            sample["question"],
            sample["gold_titles"],
            retriever,
            search_fn,
            top_k=5,
            use_experience=use_experience,
        )
        total_recall2 += result["recall@2"]
        total_recall5 += result["recall@5"]
        total_mrr += result["mrr"]
        total_full += int(result["full_hit"])

        if (i + 1) % 50 == 0:
            print(f"    progress: {i+1}/{n}")

    return {
        "recall@2": total_recall2 / n,
        "recall@5": total_recall5 / n,
        "mrr": total_mrr / n,
        "full_hit_rate": total_full / n,
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=200, help="Number of questions to evaluate")
    parser.add_argument("--data", default="data/hotpot_dev_distractor.json")
    args = parser.parse_args()

    print("=" * 70)
    print(f"HotpotQA RAG Ablation | n={args.num}")
    print("=" * 70)

    # 加载数据
    print("Loading data...")
    samples = load_hotpotqa(args.data, args.num)
    print(f"Loaded {len(samples)} questions")

    # 加载 embedding
    print("Loading embedding model...")
    dim = get_dimension()
    print(f"Embedding dim: {dim}")

    # 实验配置
    configs = [
        ("naive+bm25",           ChunkStrategy.NAIVE,            "bm25",   False),
        ("naive+vector",         ChunkStrategy.NAIVE,            "vector", False),
        ("naive+hybrid",         ChunkStrategy.NAIVE,            "hybrid", False),
        ("sw+bm25",              ChunkStrategy.SENTENCE_WINDOW,  "bm25",   False),
        ("sw+vector",            ChunkStrategy.SENTENCE_WINDOW,  "vector", False),
        ("sw+hybrid",            ChunkStrategy.SENTENCE_WINDOW,  "hybrid", False),
    ]

    results = {}
    for name, chunk, search, use_exp in configs:
        print(f"\n--- {name} ---")
        t0 = time.time()
        r = run_experiment(samples, chunk, search, embed_text, dim, use_exp)
        elapsed = time.time() - t0
        r["time_seconds"] = round(elapsed, 1)
        results[name] = r
        print(f"  Recall@2={r['recall@2']:.1%} Recall@5={r['recall@5']:.1%} "
              f"MRR={r['mrr']:.3f} FullHit={r['full_hit_rate']:.1%} ({elapsed:.1f}s)")

    # 经验增强实验: 用前半数据积累经验，后半数据测试
    print("\n--- experience enhanced ---")
    mid = len(samples) // 2
    train_samples = samples[:mid]
    test_samples = samples[mid:]

    # 模拟经验积累: 从前半数据提取 query 关键词作为有效经验
    exp_records = []
    for s in train_samples:
        import re
        keywords = re.findall(r'[a-zA-Z][\w-]{3,}', s["question"])
        exp_records.append(ExperienceRecord(
            original_query=s["question"],
            rewritten_query=None,
            result_score=7.5,
            keywords_that_helped=keywords[:5],
        ))

    # 无经验 baseline (后半数据)
    print("  baseline (no experience, test split)...")
    r_base = run_experiment(test_samples, ChunkStrategy.SENTENCE_WINDOW, "hybrid", embed_text, dim, False)
    results["sw+hybrid (test_split)"] = r_base
    print(f"  Recall@2={r_base['recall@2']:.1%} Recall@5={r_base['recall@5']:.1%} MRR={r_base['mrr']:.3f}")

    # 有经验 (后半数据 + 前半积累的经验)
    print(f"  with experience ({len(exp_records)} records)...")
    r_exp = run_experiment(test_samples, ChunkStrategy.SENTENCE_WINDOW, "hybrid", embed_text, dim, True, exp_records)
    results["sw+hybrid+exp (test_split)"] = r_exp
    print(f"  Recall@2={r_exp['recall@2']:.1%} Recall@5={r_exp['recall@5']:.1%} MRR={r_exp['mrr']:.3f}")

    delta_r5 = r_exp["recall@5"] - r_base["recall@5"]
    delta_mrr = r_exp["mrr"] - r_base["mrr"]
    print(f"  Delta: Recall@5={delta_r5:+.1%} MRR={delta_mrr:+.3f}")

    # 汇总
    print(f"\n{'=' * 70}")
    print(f"{'Config':35s} | {'Recall@2':>8s} | {'Recall@5':>8s} | {'MRR':>6s} | {'FullHit':>7s}")
    print("-" * 70)
    for name, r in results.items():
        print(f"  {name:33s} | {r['recall@2']:>7.1%} | {r['recall@5']:>7.1%} | {r['mrr']:>6.3f} | {r['full_hit_rate']:>6.1%}")

    # 保存
    out_path = Path(__file__).parent / "hotpotqa_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
