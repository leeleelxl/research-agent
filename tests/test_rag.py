"""RAG 核心模块测试"""

import tempfile
from pathlib import Path

from research.rag.chunker import Chunker, ChunkStrategy
from research.rag.vector_store import VectorStore, SearchResult
from research.rag.retriever import RAGRetriever, BM25, ExperienceRecord


class TestChunker:
    def test_naive_chunk(self):
        c = Chunker(strategy=ChunkStrategy.NAIVE, chunk_size=20, overlap=5)
        chunks = c.chunk("Hello world. This is a test document for chunking.")
        assert len(chunks) >= 2
        assert all(len(ch.text) <= 20 for ch in chunks)

    def test_sentence_window(self):
        text = "第一句话。第二句话。第三句话。第四句话。第五句话。"
        c = Chunker(strategy=ChunkStrategy.SENTENCE_WINDOW, window_size=1)
        chunks = c.chunk(text)
        assert len(chunks) == 5
        # 中间的 chunk 应包含上下文
        assert "第二句话" in chunks[2].text  # window 包含前后

    def test_semantic_chunk(self):
        text = ("# Title\n\nFirst paragraph with enough text to exceed the chunk size limit. " * 20
                + "\n\n## Section\n\nSecond paragraph also with substantial content. " * 20)
        c = Chunker(strategy=ChunkStrategy.SEMANTIC, chunk_size=200)
        chunks = c.chunk(text)
        assert len(chunks) >= 2


class TestVectorStore:
    def test_add_and_search(self):
        store = VectorStore(dimension=3)
        store.add("cat", [1.0, 0.0, 0.0])
        store.add("dog", [0.9, 0.1, 0.0])
        store.add("car", [0.0, 0.0, 1.0])

        results = store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].text == "cat"
        assert results[0].score > results[1].score

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_store.json"
            store = VectorStore(dimension=2)
            store.add("hello", [1.0, 0.0], {"source": "test"})
            store.save(path)

            loaded = VectorStore.load(path)
            assert loaded.count == 1
            results = loaded.search([1.0, 0.0], top_k=1)
            assert results[0].text == "hello"


class TestBM25:
    def test_search(self):
        bm25 = BM25()
        bm25.add("machine learning is a subset of artificial intelligence")
        bm25.add("deep learning uses neural networks")
        bm25.add("the weather is sunny today")

        results = bm25.search("neural network deep learning", top_k=2)
        assert len(results) == 2
        assert "neural" in results[0][0].lower()


class TestRAGRetriever:
    def test_bm25_search(self):
        store = VectorStore(dimension=3)
        retriever = RAGRetriever(store)
        retriever.add_document("attention mechanism in transformers")
        retriever.add_document("recurrent neural networks for NLP")
        retriever.add_document("cooking recipes for beginners")

        results = retriever.search_bm25("transformer attention", top_k=2)
        assert len(results) == 2
        assert "attention" in results[0][0].lower()

    def test_experience_rewrite(self):
        store = VectorStore(dimension=3)
        retriever = RAGRetriever(store)

        # 添加好的经验
        for _ in range(3):
            retriever.add_experience(ExperienceRecord(
                original_query="agent memory",
                rewritten_query=None,
                result_score=8.0,
                keywords_that_helped=["episodic", "retrieval"],
            ))

        # 经验增强检索应该改写 query
        _, rewritten = retriever.search_with_experience("agent memory systems")
        assert rewritten is not None
        assert "episodic" in rewritten or "retrieval" in rewritten
