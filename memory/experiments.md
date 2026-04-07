# 实验记录索引

| # | 日期 | 主题 | 结论 | 数据 |
|---|------|------|------|------|
| 1 | 2026-04-07 | 端到端 pipeline 首次运行 | 5 Agent 协调跑通，Critic 评分 7.25-7.5，1 轮通过 | examples/run_demo.py |
| 2 | 2026-04-07 | 经验增强检索闭环测试 | 连续 2 个任务，经验从 0->5->10 正确积累，分数持平 7.5 | examples/run_demo.py |
| 3 | 2026-04-07 | RAG 消融实验（小规模） | Vector>BM25(+10% Recall), Hybrid 最佳 MRR(0.900), 经验注入 MRR 下降 | experiments/rag_ablation.py |

## 关键发现

- #1: 中转站必须用 stream 模式，非 stream 返回空 content
- #1: Writer 需要拿到 Retriever/Reader 的实际数据，否则会自由发挥编造引用
- #2: 5 条经验太少，跨任务改善不明显
- #3: **简单关键词注入反而降低 MRR（0.900->0.750）**，通用关键词引入噪声，需改为相似度匹配经验
