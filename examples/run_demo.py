"""
端到端演示 — 用真实 API 跑一个研究问题
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from research.pipeline import run_research


def main():
    question = "What are the recent advances in experience-enhanced retrieval for RAG systems?"

    result = run_research(
        question=question,
        model="gpt-4o-mini",
        max_rounds=2,
        pass_threshold=7.0,
    )

    print("\n" + "=" * 60)
    print("[FINAL OUTPUT]")
    print("=" * 60)
    output = result.get("final_output", "")
    if output:
        print(output[:2000])
        if len(output) > 2000:
            print(f"\n... (truncated, total {len(output)} chars)")
    else:
        print("No output generated")

    print("\n[AGENT EXECUTION LOG]")
    for ar in result.get("agent_results", []):
        print(f"  Round {ar['round']} | {ar['agent']:10s} | tools: {ar['tool_calls']} | output: {ar['output_length']} chars")

    print(f"\n[TOKENS] input={result['tokens']['input']}, output={result['tokens']['output']}")


if __name__ == "__main__":
    main()
