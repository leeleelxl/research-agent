"""
RAG 策略消融实验 — 在 HotpotQA 子集上对比不同检索策略

消融变量：
1. Chunk 策略: naive vs sentence_window vs semantic
2. 检索方式: BM25 vs vector vs hybrid
3. Rerank: 无 vs LLM rerank
4. 经验增强: 无 vs 有

评估指标：
- Recall@5: 前 5 个检索结果中包含正确答案的比例
- MRR: 正确答案首次出现的排名倒数

用法: python experiments/rag_ablation.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.rag.chunker import Chunker, ChunkStrategy, Chunk
from research.rag.vector_store import VectorStore
from research.rag.retriever import RAGRetriever, BM25, ExperienceRecord
from research.tools.embedding import embed_text, get_dimension

# ============================================================
# HotpotQA 小规模测试集（手工构造，模拟多跳问答）
# ============================================================

DOCUMENTS = [
    {"id": "d1", "text": "Transformer architecture was introduced by Vaswani et al. in 2017 in the paper 'Attention Is All You Need'. It uses self-attention mechanisms instead of recurrence."},
    {"id": "d2", "text": "BERT (Bidirectional Encoder Representations from Transformers) was proposed by Devlin et al. in 2018. It is pre-trained using masked language modeling and next sentence prediction."},
    {"id": "d3", "text": "GPT-3, developed by OpenAI in 2020, has 175 billion parameters. It demonstrated impressive few-shot learning capabilities across many NLP tasks."},
    {"id": "d4", "text": "Retrieval-Augmented Generation (RAG) combines a retriever with a generator. The retriever finds relevant documents, and the generator produces answers conditioned on them."},
    {"id": "d5", "text": "Dense Passage Retrieval (DPR) by Karpukhin et al. 2020 uses dual encoders to retrieve passages. It outperforms BM25 on open-domain QA tasks."},
    {"id": "d6", "text": "Self-RAG by Asai et al. 2024 enables the model to decide when to retrieve and what to retrieve, improving both generation quality and factuality."},
    {"id": "d7", "text": "ReAct (Reasoning and Acting) by Yao et al. 2023 interleaves reasoning traces with actions, enabling LLMs to use tools and interact with environments."},
    {"id": "d8", "text": "Reflexion by Shinn et al. 2023 equips agents with verbal self-reflection, allowing them to learn from failures through natural language feedback stored in memory."},
    {"id": "d9", "text": "Tree of Thought (ToT) by Yao et al. 2023 enables deliberate problem solving by exploring multiple reasoning paths using tree search algorithms like BFS and DFS."},
    {"id": "d10", "text": "FAISS (Facebook AI Similarity Search) is an efficient library for similarity search of dense vectors. It supports GPU acceleration and various index types."},
    {"id": "d11", "text": "BM25 is a ranking function based on term frequency and inverse document frequency. It is widely used as a baseline in information retrieval."},
    {"id": "d12", "text": "HyDE (Hypothetical Document Embeddings) generates a hypothetical answer first, then uses its embedding to retrieve relevant documents, improving zero-shot retrieval."},
    {"id": "d13", "text": "Sentence-window chunking preserves context around each sentence by including neighboring sentences. This helps maintain coherence in retrieved chunks."},
    {"id": "d14", "text": "Cross-encoder rerankers score query-document pairs jointly, providing more accurate relevance judgments than bi-encoders but at higher computational cost."},
    {"id": "d15", "text": "Experience replay, originally from DQN by Mnih et al. 2015, stores past experiences in a buffer and samples them for training, breaking correlations in sequential data."},
    {"id": "d16", "text": "Agent-as-a-Judge by Pan et al. 2024 uses agentic systems to evaluate other agents, achieving 90% agreement with human experts compared to 70% for LLM-as-Judge."},
    {"id": "d17", "text": "DSPy by Khattab et al. 2024 is a framework for programming LMs declaratively. It can optimize prompts algorithmically, raising ReAct agent accuracy from 24% to 51%."},
    {"id": "d18", "text": "Mem0 is a memory layer for AI applications that provides long-term memory persisting across sessions, supporting episodic and semantic memory types."},
    {"id": "d19", "text": "CRAG (Corrective RAG) by Yan et al. 2024 detects low-quality retrieval results and triggers corrective actions like query rewriting or web search fallback."},
    {"id": "d20", "text": "Multi-agent systems coordinate multiple specialized agents. Role isolation prevents conflicts, and structured communication protocols ensure reliable collaboration."},
]

QA_PAIRS = [
    {"question": "What paper introduced the Transformer architecture?", "answer_doc_ids": ["d1"], "answer_text": "Attention Is All You Need"},
    {"question": "How does Self-RAG improve over standard RAG?", "answer_doc_ids": ["d6", "d4"], "answer_text": "decides when and what to retrieve"},
    {"question": "What technique does Reflexion use for agent learning?", "answer_doc_ids": ["d8"], "answer_text": "verbal self-reflection"},
    {"question": "How does DPR differ from BM25?", "answer_doc_ids": ["d5", "d11"], "answer_text": "dual encoders"},
    {"question": "What is the key idea behind experience replay?", "answer_doc_ids": ["d15"], "answer_text": "stores past experiences in a buffer"},
    {"question": "How does CRAG handle low-quality retrieval?", "answer_doc_ids": ["d19"], "answer_text": "corrective actions like query rewriting"},
    {"question": "What accuracy improvement did DSPy achieve on ReAct?", "answer_doc_ids": ["d17"], "answer_text": "24% to 51%"},
    {"question": "What is the advantage of cross-encoder rerankers?", "answer_doc_ids": ["d14"], "answer_text": "more accurate relevance judgments"},
    {"question": "How do multi-agent systems prevent conflicts?", "answer_doc_ids": ["d20"], "answer_text": "role isolation"},
    {"question": "What does HyDE generate before retrieval?", "answer_doc_ids": ["d12"], "answer_text": "hypothetical answer"},
]


# ============================================================
# 评估函数
# ============================================================

def evaluate_retrieval(
    questions: list[dict],
    retriever: RAGRetriever,
    search_fn: str,
    top_k: int = 5,
    use_experience: bool = False,
) -> dict:
    """评估检索策略"""
    recall_hits = 0
    mrr_sum = 0.0
    total = len(questions)

    for qa in questions:
        query = qa["question"]
        answer_ids = set(qa["answer_doc_ids"])
        answer_text = qa["answer_text"].lower()

        # 根据策略选择检索方式
        if use_experience:
            results, _ = retriever.search_with_experience(query, top_k)
        elif search_fn == "bm25":
            bm25_results = retriever.search_bm25(query, top_k)
            results = [type("R", (), {"text": t, "score": s})() for t, s in bm25_results]
        elif search_fn == "vector":
            results = retriever.search_vector(query, top_k)
        elif search_fn == "hybrid":
            results = retriever.search_hybrid(query, top_k)
        else:
            results = []

        # 计算 Recall@K 和 MRR
        hit = False
        for rank, r in enumerate(results, 1):
            text = r.text.lower() if hasattr(r, "text") else str(r).lower()
            if answer_text in text:
                if not hit:
                    mrr_sum += 1.0 / rank
                    hit = True
        if hit:
            recall_hits += 1

    return {
        "recall@5": recall_hits / total if total > 0 else 0,
        "mrr": mrr_sum / total if total > 0 else 0,
        "total": total,
        "hits": recall_hits,
    }


# ============================================================
# 消融实验
# ============================================================

def run_ablation():
    print("=" * 70)
    print("RAG Ablation Study")
    print("=" * 70)

    dim = get_dimension()
    results = {}

    for chunk_strategy in [ChunkStrategy.NAIVE, ChunkStrategy.SENTENCE_WINDOW]:
        chunker = Chunker(strategy=chunk_strategy, chunk_size=256, overlap=32, window_size=1)

        # 构建索引
        store = VectorStore(dimension=dim)
        retriever = RAGRetriever(store, embed_fn=embed_text)

        for doc in DOCUMENTS:
            chunks = chunker.chunk(doc["text"], metadata={"doc_id": doc["id"]})
            for chunk in chunks:
                retriever.add_document(chunk.text, chunk.metadata)

        strategy_name = chunk_strategy.value
        print(f"\n--- Chunk: {strategy_name} | docs: {store.count} ---")

        # 测试不同检索方式
        for search_fn in ["bm25", "vector", "hybrid"]:
            key = f"{strategy_name}+{search_fn}"
            r = evaluate_retrieval(QA_PAIRS, retriever, search_fn, top_k=5)
            results[key] = r
            print(f"  {key:30s} | Recall@5={r['recall@5']:.0%} ({r['hits']}/{r['total']}) | MRR={r['mrr']:.3f}")

        # 测试经验增强（先注入一些模拟经验）
        for _ in range(10):
            retriever.add_experience(ExperienceRecord(
                original_query="RAG retrieval",
                rewritten_query=None,
                result_score=8.0,
                keywords_that_helped=["retrieval", "augmented", "generation", "dense", "passage"],
            ))
        key = f"{strategy_name}+hybrid+experience"
        r = evaluate_retrieval(QA_PAIRS, retriever, "hybrid", top_k=5, use_experience=True)
        results[key] = r
        print(f"  {key:30s} | Recall@5={r['recall@5']:.0%} ({r['hits']}/{r['total']}) | MRR={r['mrr']:.3f}")

    # 汇总表
    print(f"\n{'=' * 70}")
    print(f"{'Strategy':35s} | {'Recall@5':>8s} | {'MRR':>6s} | {'Hits':>4s}")
    print("-" * 70)
    for key, r in sorted(results.items()):
        print(f"  {key:33s} | {r['recall@5']:>7.0%} | {r['mrr']:>6.3f} | {r['hits']:>3d}/{r['total']}")

    # 保存结果
    out_path = Path(__file__).parent / "rag_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_ablation()
