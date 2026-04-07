"""
端到端演示 — 连续跑多个研究问题，验证经验跨任务积累

经验增强检索的核心假设：
  第 1 个任务积累的检索经验（哪些关键词有效、哪些方向被遗漏）
  能帮助第 2 个相关任务的检索更精准。

运行：python examples/run_demo.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from research.pipeline import run_research

EXPERIENCE_FILE = "demo_experience.jsonl"


def print_result(result: dict, task_num: int):
    print(f"\n{'=' * 60}")
    print(f"[TASK {task_num} RESULT]")
    print(f"{'=' * 60}")

    output = result.get("final_output", "")
    if output:
        print(output[:1500])
        if len(output) > 1500:
            print(f"\n... (total {len(output)} chars)")
    else:
        print("(no output)")

    print(f"\n[AGENTS]")
    for ar in result.get("agent_results", []):
        print(f"  round={ar['round']} {ar['agent']:10s} tools={ar['tool_calls']} output={ar['output_length']}")

    review = result.get("review", {})
    print(f"\n[SCORES] coverage={review.get('coverage','-')} accuracy={review.get('accuracy','-')} "
          f"coherence={review.get('coherence','-')} depth={review.get('depth','-')} "
          f"avg={review.get('score',0):.1f}")
    print(f"[TOKENS] input={result['tokens']['input']} output={result['tokens']['output']}")
    print(f"[EXPERIENCE] store={result.get('experience_count', 0)} records")


def main():
    # 清理旧经验文件
    exp_path = Path(EXPERIENCE_FILE)
    if exp_path.exists():
        exp_path.unlink()

    # 任务 1: 第一次检索，没有经验
    print("\n" + "#" * 60)
    print("# TASK 1: No prior experience")
    print("#" * 60)
    r1 = run_research(
        question="What are the recent advances in self-evolving AI agents?",
        model="gpt-4o-mini",
        max_rounds=2,
        pass_threshold=7.0,
        experience_path=EXPERIENCE_FILE,
    )
    print_result(r1, 1)

    # 任务 2: 相关主题，有任务 1 的经验
    print("\n" + "#" * 60)
    print("# TASK 2: With experience from Task 1")
    print("#" * 60)
    r2 = run_research(
        question="How do LLM agents use memory systems for long-term learning?",
        model="gpt-4o-mini",
        max_rounds=2,
        pass_threshold=7.0,
        experience_path=EXPERIENCE_FILE,
    )
    print_result(r2, 2)

    # 对比
    print("\n" + "=" * 60)
    print("[COMPARISON]")
    s1 = r1.get("review", {}).get("score", 0)
    s2 = r2.get("review", {}).get("score", 0)
    print(f"  Task 1 score: {s1:.1f} (no experience)")
    print(f"  Task 2 score: {s2:.1f} (with {r1.get('experience_count', 0)} prior experiences)")
    print(f"  Delta: {s2 - s1:+.1f}")


if __name__ == "__main__":
    main()
