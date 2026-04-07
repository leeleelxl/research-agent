# 任务看板

> 更新时间：2026-04-07

## Week 1: RAG 核心

| # | 任务 | 模块 | 状态 |
|---|------|------|------|
| 1 | 项目骨架 + 基础设施 | 全局 | done |
| 2 | LLM 调用层（stream + tool calling） | tools | done |
| 3 | Semantic Scholar + arXiv 检索工具 | tools | done |
| 4 | PDF 解析 + 3 种 chunk 策略 | rag | done |
| 5 | FAISS 向量库 + BM25 检索 | rag | done |
| 6 | LLM reranker | rag | done |
| 7 | RAG 策略消融实验 | experiments | done (小规模，需接 HotpotQA) |

## Week 2-3: 多 Agent + 自进化

| # | 任务 | 模块 | 状态 |
|---|------|------|------|
| 8 | Agent 基类 + 工具注册机制 | agents | done |
| 9 | Planner Agent | agents | done |
| 10 | Retriever Agent + 经验增强检索 | agents | done |
| 11 | Reader Agent | agents | done |
| 12 | Writer Agent | agents | done |
| 13 | Critic Agent（Agent-as-Judge） | agents | done |
| 14 | Coordinator（协调 + 反馈循环） | coordinator | done |
| 15 | 经验增强检索闭环 | memory | done |
| 16 | 端到端 pipeline 跑通 | 全局 | done |

## 接下来要做

| # | 任务 | 优先级 | 说明 |
|---|------|--------|------|
| 17 | 接入 HotpotQA 跑正式消融实验 | done | naive+vector 最优 71.2% Recall@5, 0.852 MRR |
| 18 | 改进经验匹配 v2（相似度匹配） | done | 消除负面效果，但正向提升未实现，需 v3 |
| 19 | Web 可视化（多 Agent 执行过程） | P1 | 复用 AgentProbe |
| 20 | README + 技术博客 | P1 | |
| 21 | Fine-tune embedding（AutoDL 租卡） | P2 | 用 Agent 运行数据训练，最后做 |
| 22 | 训练 reranker | P2 | |
| 23 | 训练前后对比实验 | P2 | |
