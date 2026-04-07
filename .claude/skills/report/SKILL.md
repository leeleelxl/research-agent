# /report — 实验收尾

当用户说"收尾"、"总结"、"/report"时触发。

## 执行步骤

1. 确认本次实验范围和编号
2. 确保 `experiments/<exp_id>/report.md` 包含：目标、方法、结果、结论
3. 运行 `python infrastructure/capture.py verify` 验证证据链
4. 在 `memory/experiments.md` 添加索引条目
5. 更新 `memory/task_board.md` 对应任务状态
6. 如有值得记录的经验 → 更新 `memory/retro/` 对应文件
