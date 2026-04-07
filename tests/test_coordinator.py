"""Coordinator 和 SharedState 测试"""

from research.coordinator.coordinator import SharedState, ResearchCoordinator
from research.memory.experience_store import ExperienceStore
from research.rag.retriever import ExperienceRecord


class TestSharedState:
    def test_initial_state(self):
        state = SharedState(question="What is RAG?")
        assert state.question == "What is RAG?"
        assert state.rounds_completed == 0
        assert len(state.papers) == 0

    def test_token_tracking(self):
        state = SharedState()
        state.total_tokens["input"] += 100
        state.total_tokens["output"] += 50
        assert state.total_tokens == {"input": 100, "output": 50}


class TestExperienceStore:
    def test_add_and_query(self, tmp_path):
        store = ExperienceStore(tmp_path / "exp.jsonl")
        store.add(ExperienceRecord(
            original_query="agent memory",
            rewritten_query="agent episodic memory",
            result_score=8.5,
            keywords_that_helped=["episodic"],
        ))
        store.add(ExperienceRecord(
            original_query="bad query",
            rewritten_query=None,
            result_score=2.0,
        ))

        assert store.count == 2
        good = store.get_good_experiences(min_score=7.0)
        assert len(good) == 1
        bad = store.get_bad_experiences(max_score=4.0)
        assert len(bad) == 1

    def test_persistence(self, tmp_path):
        path = tmp_path / "exp.jsonl"
        store1 = ExperienceStore(path)
        store1.add(ExperienceRecord("q1", None, 9.0))
        store1.add(ExperienceRecord("q2", None, 3.0))

        store2 = ExperienceStore(path)  # 重新加载
        assert store2.count == 2
        assert store2.avg_score == 6.0

    def test_summary(self, tmp_path):
        store = ExperienceStore(tmp_path / "exp.jsonl")
        assert "为空" in store.summary()
        store.add(ExperienceRecord("q", None, 8.0))
        summary = store.summary()
        assert "1 条" in summary
