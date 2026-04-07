# 工程踩坑记录

| 日期 | 问题 | 解决方案 | 教训 |
|------|------|---------|------|
| 2026-04-07 | 中转站非 stream 模式返回空 content | 改用 stream=True，手动拼接 delta | 中转站 API 和官方行为不一致，永远先测试 |
| 2026-04-07 | stream 模式不返回 usage（token 统计为 0） | 用字符数估算 token（中文约 1.5 字/token） | 不能依赖中转站返回标准字段 |
| 2026-04-07 | Writer 自由发挥编造论文引用 | Coordinator 把 papers/notes 实际内容传入 Writer task | Agent 的输出质量强依赖输入质量，必须确保数据流贯通 |
| 2026-04-07 | Coordinator._run_agent 没有把 Agent 输出写回 SharedState | 加 _update_state 按角色解析输出 | 多 Agent 系统的核心难题：状态管理和数据传递 |
| 2026-04-07 | demo 运行时 PDF 被 git commit | .gitignore 加 papers/ | 大文件要提前排除 |
| 2026-04-07 | Semantic Scholar API 429 限流 | 加延迟或减少请求频率 | 公开 API 有 rate limit，实验时注意 |
