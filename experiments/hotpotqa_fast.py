"""
HotpotQA 消融实验 — 优化版（共享 embedding 缓存）

优化: 每个文档只 embed 一次，不同检索策略共享缓存
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.rag.chunker import Chunker, ChunkStrategy
from research.rag.vector_store import VectorStore
from research.rag.retriever import RAGRetriever, ExperienceRecord
from research.tools.embedding import embed_text, embed_batch, get_dimension


def load_hotpotqa(path: str, num: int) -> list[dict]:
    raw = json.load(open(path))
    samples = []
    for item in raw[:num]:
        docs = []
        for title, sentences in item["context"]:
            docs.append({"title": title, "text": " ".join(sentences), "sentences": sentences})
        sf_titles = set(t for t, _ in item["supporting_facts"])
        samples.append({
            "question": item["question"],
            "answer": item["answer"],
            "docs": docs,
            "gold_titles": sf_titles,
        })
    return samples


def build_all_indexes(samples, chunker, embed_fn, dim):
    """预计算: 为所有问题构建索引并缓存 embedding"""
    indexes = []
    total_embeds = 0

    for i, sample in enumerate(samples):
        texts = []
        metas = []
        for doc in sample["docs"]:
            chunks = chunker.chunk(doc["text"], metadata={"title": doc["title"]})
            for chunk in chunks:
                texts.append(chunk.text)
                metas.append({"title": doc["title"]})

        # 批量 embedding
        vectors = embed_batch(texts)
        total_embeds += len(texts)

        # 构建 vector store
        store = VectorStore(dimension=dim)
        store.add_batch(texts, vectors, metas)

        # 构建 BM25
        from research.rag.retriever import BM25
        bm25 = BM25()
        for t in texts:
            bm25.add(t)

        indexes.append({
            "store": store,
            "bm25": bm25,
            "texts": texts,
            "metas": metas,
            "vectors": vectors,
        })

        if (i + 1) % 50 == 0:
            print(f"  indexed {i+1}/{len(samples)} ({total_embeds} embeddings)")

    print(f"  total: {total_embeds} embeddings cached")
    return indexes


def search_and_eval(question, gold_titles, index, search_fn, embed_fn, dim, top_k=5, experience_records=None):
    """单次检索评估"""
    store = index["store"]
    retriever = RAGRetriever(store, embed_fn=embed_fn)
    retriever.bm25 = index["bm25"]

    if experience_records:
        for exp in experience_records:
            retriever.add_experience(exp)

    use_exp = experience_records is not None
    if use_exp:
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

    # 匹配
    retrieved_titles = []
    for r in results:
        meta = getattr(r, "metadata", {}) or {}
        title = meta.get("title", "")
        if not title:
            text = r.text if hasattr(r, "text") else str(r)
            for gt in gold_titles:
                if gt.lower() in text.lower():
                    title = gt
                    break
        retrieved_titles.append(title)

    retrieved_set = set(retrieved_titles[:top_k])
    recall5 = len(gold_titles & retrieved_set) / len(gold_titles) if gold_titles else 0
    recall2 = len(gold_titles & set(retrieved_titles[:2])) / len(gold_titles) if gold_titles else 0

    mrr = 0.0
    for rank, t in enumerate(retrieved_titles, 1):
        if t in gold_titles:
            mrr = 1.0 / rank
            break

    return {"recall@2": recall2, "recall@5": recall5, "mrr": mrr}


def run_config(samples, indexes, search_fn, embed_fn, dim, exp_records=None):
    """跑一个配置"""
    r2, r5, mrr_sum = 0.0, 0.0, 0.0
    n = len(samples)
    for i, (sample, index) in enumerate(zip(samples, indexes)):
        r = search_and_eval(sample["question"], sample["gold_titles"], index, search_fn, embed_fn, dim,
                           experience_records=exp_records)
        r2 += r["recall@2"]
        r5 += r["recall@5"]
        mrr_sum += r["mrr"]
    return {"recall@2": r2/n, "recall@5": r5/n, "mrr": mrr_sum/n, "n": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--data", default="data/hotpot_dev_distractor.json")
    args = parser.parse_args()

    print("=" * 70)
    print(f"HotpotQA RAG Ablation (Fast) | n={args.num}")
    print("=" * 70)

    samples = load_hotpotqa(args.data, args.num)
    print(f"Loaded {len(samples)} questions")

    dim = get_dimension()
    print(f"Embedding dim: {dim}")

    results = {}

    # Naive chunk
    print("\n[Phase 1] Building indexes (naive chunk)...")
    t0 = time.time()
    chunker_naive = Chunker(strategy=ChunkStrategy.NAIVE, chunk_size=256, overlap=32)
    naive_indexes = build_all_indexes(samples, chunker_naive, embed_text, dim)
    print(f"  done in {time.time()-t0:.1f}s")

    for fn in ["bm25", "vector", "hybrid"]:
        key = f"naive+{fn}"
        print(f"  {key}...")
        r = run_config(samples, naive_indexes, fn, embed_text, dim)
        results[key] = r
        print(f"    Recall@2={r['recall@2']:.1%} Recall@5={r['recall@5']:.1%} MRR={r['mrr']:.3f}")

    # Sentence window chunk
    print("\n[Phase 2] Building indexes (sentence_window chunk)...")
    t0 = time.time()
    chunker_sw = Chunker(strategy=ChunkStrategy.SENTENCE_WINDOW, chunk_size=256, window_size=1)
    sw_indexes = build_all_indexes(samples, chunker_sw, embed_text, dim)
    print(f"  done in {time.time()-t0:.1f}s")

    for fn in ["bm25", "vector", "hybrid"]:
        key = f"sw+{fn}"
        print(f"  {key}...")
        r = run_config(samples, sw_indexes, fn, embed_text, dim)
        results[key] = r
        print(f"    Recall@2={r['recall@2']:.1%} Recall@5={r['recall@5']:.1%} MRR={r['mrr']:.3f}")

    # Experience enhanced (v2: similarity-based matching)
    print("\n[Phase 3] Experience enhanced retrieval...")
    mid = len(samples) // 2
    train_samples, test_samples = samples[:mid], samples[mid:]
    train_indexes, test_indexes = sw_indexes[:mid], sw_indexes[mid:]

    # 从 train set 积累经验
    exp_records = []
    for s in train_samples:
        keywords = re.findall(r'[a-zA-Z][\w-]{3,}', s["question"])
        exp_records.append(ExperienceRecord(
            original_query=s["question"],
            rewritten_query=None,
            result_score=7.5,
            keywords_that_helped=keywords[:5],
        ))
    print(f"  accumulated {len(exp_records)} experience records from train split")

    # Baseline (test split, no experience)
    print("  baseline (test split)...")
    r_base = run_config(test_samples, test_indexes, "hybrid", embed_text, dim)
    results["sw+hybrid [test]"] = r_base
    print(f"    Recall@2={r_base['recall@2']:.1%} Recall@5={r_base['recall@5']:.1%} MRR={r_base['mrr']:.3f}")

    # With experience (v2)
    print(f"  with experience v2 ({len(exp_records)} records)...")
    r_exp = run_config(test_samples, test_indexes, "hybrid", embed_text, dim, exp_records)
    results["sw+hybrid+exp_v2 [test]"] = r_exp
    print(f"    Recall@2={r_exp['recall@2']:.1%} Recall@5={r_exp['recall@5']:.1%} MRR={r_exp['mrr']:.3f}")

    dr5 = r_exp["recall@5"] - r_base["recall@5"]
    dmrr = r_exp["mrr"] - r_base["mrr"]
    print(f"    Delta: Recall@5={dr5:+.1%} MRR={dmrr:+.3f}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  {'Config':35s} | {'Recall@2':>8s} | {'Recall@5':>8s} | {'MRR':>6s}")
    print(f"  {'-'*35}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    for name, r in results.items():
        print(f"  {name:35s} | {r['recall@2']:>7.1%} | {r['recall@5']:>7.1%} | {r['mrr']:>6.3f}")

    out_path = Path(__file__).parent / "hotpotqa_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
