# 检索优化经验

## 经验增强检索现状

已实现闭环：
  Critic 评分 -> 提取 (query, score, helpful/missed keywords)
  -> ExperienceStore (jsonl 持久化)
  -> 下次 Retriever 读取 -> 注入 prompt

问题：
- 当前实现是全局关键词注入（取所有好经验的高频关键词）
- 消融实验显示 MRR 反而下降（0.900->0.750）
- 原因：通用关键词与当前 query 不相关时引入噪声

## 改进方向

#1 按 query 相似度匹配经验（不是全局注入）
  - 用 embedding 计算当前 query 和历史 query 的相似度
  - 只注入相似 query 的经验关键词
  - 这在学术上叫 "experience-conditioned query rewriting"

#2 区分 query-level 和 domain-level 经验
  - domain-level: 某个领域普遍有效的关键词（如 "RAG" 领域的 "dense retrieval"）
  - query-level: 只对特定类型 query 有效的关键词

#3 引入负经验（避免无效 query）
  - 已有 get_bad_experiences()，但还没在 query 改写中使用
  - 可以用负经验做 "这些关键词不要加" 的过滤
