# 多 Agent 协调经验

## 已验证的设计

#1 5 Agent 各自有独立工具集（通过 role 过滤），Planner 1 个，Retriever 4 个，Reader 4 个，Writer 1 个，Critic 1 个
#2 SharedState 作为 Agent 间通信载体，每个 Agent 只写自己负责的字段
#3 Coordinator._update_state 按角色解析 Agent 输出（tool_calls + 文本）写回 SharedState
#4 Critic->Planner 反馈循环：Critic 的 gaps 和 suggestions 注入下一轮的 Retriever task

## 端到端运行数据

- 任务: "experience-enhanced retrieval for RAG"
- Planner: 分解 5 个子问题，1 轮
- Retriever: 6-16 次工具调用（Semantic Scholar + arXiv）
- Reader: 2-11 次工具调用
- Writer: 5-7 次 write_section 调用，生成 5000-14000 字综述
- Critic: 评分 7.2-7.5，识别 5-7 个 gaps
- Token: 约 input=30k-58k, output=12k-27k（单次完整运行）
- 1 轮即通过（阈值 7.0）

## 待改进

- Reader 工具调用次数不稳定（有时 0 次），可能是 prompt 不够明确
- Writer output 字段有时为空（内容通过 write_section 工具传递），需统一
- 反馈循环还没真正跑多轮（都是 1 轮通过），需调低阈值测试
