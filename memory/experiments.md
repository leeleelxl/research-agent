# 实验记录索引

| # | 日期 | 主题 | 结论 | 数据 |
|---|------|------|------|------|
| 1 | 2026-04-07 | 端到端 pipeline 首次运行 | 5 Agent 协调跑通，Critic 评分 7.25-7.5，1 轮通过 | examples/run_demo.py |
| 2 | 2026-04-07 | 经验增强检索闭环测试 | 连续 2 个任务，经验从 0->5->10 正确积累，分数持平 7.5 | examples/run_demo.py |
| 3 | 2026-04-07 | RAG 消融(小规模) | Vector>BM25(+10%), 全局关键词注入 MRR 下降 | experiments/rag_ablation.py |
| 4 | 2026-04-07 | **HotpotQA 消融(200题)** | Vector>>BM25(+12% Recall), naive>sw, 经验v2中性 | experiments/hotpotqa_fast.py |

## 关键发现

- #1: 中转站必须用 stream 模式
- #1: Writer 必须拿到 Retriever/Reader 实际数据
- #3: 全局关键词注入 MRR 大幅下降(-0.150)
- #4: **HotpotQA 正式结果**: naive+vector 最优 (Recall@5=71.2%, MRR=0.852)
- #4: sentence_window 在短文档上反而更差 (Recall@5: 71.2% -> 63.2%)
- #4: Hybrid 检索因 BM25 归一化问题反而拉低了 vector 的效果
- #4: 经验增强 v2 (相似度匹配) 消除了 v1 的负面效果，但还没有正向提升 (Recall 持平, MRR -0.011)
