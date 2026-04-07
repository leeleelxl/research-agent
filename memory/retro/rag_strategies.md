# RAG 策略经验

## HotpotQA 消融实验 (200 questions, 正式结果)

| 策略 | Recall@2 | Recall@5 | MRR |
|------|----------|----------|-----|
| naive+bm25 (baseline) | 46.8% | 59.2% | 0.722 |
| **naive+vector** | **56.0%** | **71.2%** | **0.852** |
| naive+hybrid | 49.2% | 58.5% | 0.775 |
| sw+bm25 | 37.0% | 55.2% | 0.688 |
| sw+vector | 43.5% | 63.2% | 0.816 |
| sw+hybrid | 44.5% | 55.8% | 0.763 |
| sw+hybrid [test split] | 41.5% | 53.0% | 0.760 |
| sw+hybrid+exp_v2 [test] | 41.5% | 53.0% | 0.749 |

## 规则

#1 **Vector >> BM25**: Recall@5 +12%(59.2->71.2%), MRR +0.130。在 HotpotQA 上验证一致
#2 **naive chunk > sentence_window**: 短文档场景下 naive 效果更好(71.2% vs 63.2%)，sentence_window 切得太碎反而引入噪声
#3 **Hybrid 未必最优**: naive+hybrid(58.5%) 反而低于 naive+vector(71.2%)。原因是 BM25 分数归一化不精确，拉低了 vector 的准确匹配
#4 **经验增强 v2 效果中性**: Recall 持平(53.0%)，MRR 微降(-0.011)。比 v1(MRR 大幅下降)好很多，但还没有正向提升
#5 经验增强需要更精准的匹配：当前 Jaccard 相似度可能不够，应尝试用 embedding 相似度匹配经验
#6 sentence_window 在长文档场景可能更有优势，当前 HotpotQA 段落偏短

## 面试话术
"在 HotpotQA 200 题上，向量检索比 BM25 Recall@5 提升 12 个点（59.2%->71.2%），MRR 提升 0.130。经验增强检索 v2 用相似度匹配替代全局注入后，消除了 v1 的负面效果，但正向提升还需要进一步优化匹配策略。"
