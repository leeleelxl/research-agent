# RAG 策略经验

## 消融实验数据（小规模，20 doc / 10 QA）

| 策略 | Recall@5 | MRR |
|------|----------|-----|
| naive+bm25 | 80% | 0.800 |
| naive+vector | 90% | 0.850 |
| naive+hybrid | 90% | 0.900 |
| naive+hybrid+experience | 90% | 0.750 |
| sentence_window+bm25 | 80% | 0.800 |
| sentence_window+vector | 90% | 0.833 |
| sentence_window+hybrid | 90% | 0.770 |
| sentence_window+hybrid+experience | 90% | 0.500 |

## 规则

#1 Vector 检索比 BM25 Recall 高约 10%，但 BM25 在精确关键词匹配上有优势
#2 Hybrid 检索（vector 0.7 + BM25 0.3）在 naive chunk 上 MRR 最高
#3 sentence_window chunk 产生更多 chunk（20->41），但检索效果未必更好（文档短时 naive 就够）
#4 **简单的全局关键词注入会引入噪声**，MRR 显著下降。经验增强需要按 query 相似度精准匹配，不能无脑注入
#5 当前数据集太小（20 doc），结论需在 HotpotQA 上验证
