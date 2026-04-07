# ReSearch — 自进化多 Agent 学术研究系统

多 Agent 协作 + RAG + 经验增强检索，给定研究问题，自动检索论文、构建知识库、生成文献综述，并从反馈中持续改进。

## 硬性规则

0. 保持诚实，不编造实验数据
1. **回答使用中文**
2. **实验数据必须走证据链**：benchmark/评测通过 `infrastructure/capture.py` 执行
3. **实验后记录**：执行 `/report` 更新实验索引
4. **控制输出量**：大文件用 offset/limit，命令输出取关键部分
5. **提交前测试**：推 GitHub 前确保测试通过

## 架构概览

```
Planner → Retriever → Reader → Writer → Critic
   ↑                                      │
   └──────── 反馈循环（不合格打回）──────────┘
```

五个 Agent，各自有独立工具集和记忆：
- **Planner** — 拆解问题、规划检索策略
- **Retriever** — 检索论文、管理向量知识库、经验增强检索
- **Reader** — 精读论文、提取结构化信息
- **Writer** — 生成文献综述
- **Critic** — 多维度评估、驱动反馈循环

## 核心创新：经验增强检索

普通 RAG：query → embedding → 搜索
我们的：query + 历史经验 → query 改写 → embedding → 搜索 → 经验 rerank

经验来源：Critic 对每次检索结果的质量评分，积累形成 (query, rewrite, score) 三元组。

## 文件导航

### Session 启动必读

| 文件 | 内容 |
|------|------|
| `memory/task_board.md` | 当前任务进度 |
| `memory/experiments.md` | 实验记录索引 |

### 按需读取

| 触发条件 | 文件 |
|---------|------|
| RAG 策略对比经验 | `memory/retro/rag_strategies.md` |
| 检索优化经验 | `memory/retro/retrieval_optimization.md` |
| 工程踩坑 | `memory/retro/engineering.md` |
| 多 Agent 协调经验 | `memory/retro/multi_agent.md` |

### Skills

- `/report` — 实验收尾：补完报告、更新索引、验证数据
