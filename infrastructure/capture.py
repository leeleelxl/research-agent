#!/usr/bin/env python3
"""
证据链捕获工具 — 借鉴 SOAR capture.py 设计

用途：包裹任何命令执行，自动保存原始输出和元数据，防止 LLM 编造结果。

用法：
    python capture.py run --output-dir <dir> --tag <tag> -- <command...>
    python capture.py verify --output-dir <dir> --tag <tag> --expected <pattern>

示例：
    python capture.py run --output-dir results/exp01 --tag pytest -- pytest tests/ -v
    python capture.py verify --output-dir results/exp01 --tag pytest --expected "passed"
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def cmd_run(args):
    """执行命令并捕获完整输出"""
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag
    log_file = raw_dir / f"{tag}.log"
    meta_file = raw_dir / f"{tag}.meta.json"

    command = args.command
    cmd_str = " ".join(command)

    print(f"[capture] 执行: {cmd_str}")
    print(f"[capture] 输出: {log_file}")

    start_time = time.time()
    start_ts = datetime.now(timezone.utc).isoformat()

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout,
        )
        output = result.stdout
        exit_code = result.returncode
    except subprocess.TimeoutExpired as e:
        output = e.stdout or ""
        exit_code = -1
        print(f"[capture] 超时 ({args.timeout}s)")
    except FileNotFoundError:
        output = f"命令未找到: {command[0]}"
        exit_code = 127
        print(f"[capture] {output}")

    elapsed = time.time() - start_time

    # 写日志
    with open(log_file, "w") as f:
        f.write(output)

    line_count = output.count("\n")

    # 写元数据
    meta = {
        "command": cmd_str,
        "tag": tag,
        "exit_code": exit_code,
        "start_time": start_ts,
        "elapsed_seconds": round(elapsed, 2),
        "line_count": line_count,
        "log_file": str(log_file),
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 打印摘要
    status = "✅ 成功" if exit_code == 0 else f"❌ 失败 (exit={exit_code})"
    print(f"[capture] {status} | {elapsed:.1f}s | {line_count} 行")

    return exit_code


def cmd_verify(args):
    """验证捕获的输出是否包含预期内容"""
    output_dir = Path(args.output_dir)
    log_file = output_dir / "raw" / f"{args.tag}.log"
    meta_file = output_dir / "raw" / f"{args.tag}.meta.json"

    if not log_file.exists():
        print(f"[verify] ❌ 日志不存在: {log_file}")
        return 1

    if not meta_file.exists():
        print(f"[verify] ❌ 元数据不存在: {meta_file}")
        return 1

    with open(log_file) as f:
        content = f.read()

    with open(meta_file) as f:
        meta = json.load(f)

    # 检查退出码
    if meta["exit_code"] != 0:
        print(f"[verify] ⚠️ 命令退出码非零: {meta['exit_code']}")

    # 检查预期模式
    if args.expected:
        if args.expected in content:
            print(f"[verify] ✅ 找到预期内容: '{args.expected}'")
            return 0
        else:
            print(f"[verify] ❌ 未找到预期内容: '{args.expected}'")
            return 2

    print(f"[verify] ✅ 日志存在 ({meta['line_count']} 行, {meta['elapsed_seconds']}s)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="证据链捕获工具")
    sub = parser.add_subparsers(dest="action", required=True)

    # run 子命令
    p_run = sub.add_parser("run", help="执行命令并捕获输出")
    p_run.add_argument("--output-dir", required=True, help="输出目录")
    p_run.add_argument("--tag", required=True, help="输出标签")
    p_run.add_argument("--timeout", type=int, default=300, help="超时秒数 (默认 300)")
    p_run.add_argument("command", nargs=argparse.REMAINDER, help="要执行的命令")

    # verify 子命令
    p_verify = sub.add_parser("verify", help="验证捕获的输出")
    p_verify.add_argument("--output-dir", required=True, help="输出目录")
    p_verify.add_argument("--tag", required=True, help="输出标签")
    p_verify.add_argument("--expected", help="预期包含的字符串")

    args = parser.parse_args()

    # 处理 -- 分隔符
    if args.action == "run":
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        if not args.command:
            parser.error("缺少要执行的命令")

    if args.action == "run":
        sys.exit(cmd_run(args))
    elif args.action == "verify":
        sys.exit(cmd_verify(args))


if __name__ == "__main__":
    main()
