# DeepRevision Competition Pitch

## One-Liner

DeepRevision 是一个基于课件证据和错题 mastery 的个性化期末冲刺 Agent。

## Core Problem

期末复习最大的问题不是“没有 AI 回答”，而是学生很难把课件、练习、错题和复习计划连成一个可信闭环。普通聊天机器人能解释概念，但不会持续追踪学生在哪些知识点反复出错，也很难证明答案来自课件。

## Product Promise

DeepRevision 把复习过程压缩成一个闭环：

1. 课件入库，回答必须锚定资料。
2. 根据课件和薄弱点生成练习。
3. 学生作答后自动记录错因和知识点。
4. mastery 画像更新掌握度、连错和复习优先级。
5. Planner 根据最近错题和薄弱点生成短期复习计划。
6. 下一轮出题继续围绕薄弱点纠偏。

## Differentiators

- 多 Agent 分工：Supervisor 路由到 RAG、Quiz、Exam、Planner、History、Ops。
- 可信来源：知识回答强调课件证据，避免纯大模型自由发挥。
- 学习闭环：练习记录会进入 mastery，并影响后续计划和推荐。
- 安全工具：危险删除类操作必须二次确认，不会被一句话诱导执行。
- 可测评：已有标准 Agent 测评和生产测评报告，能量化 route、SSE、mastery、安全和稳定性。

## Judge-Facing Summary

如果普通 AI 是“会答题的聊天框”，DeepRevision 是“会看课件、会出题、会判分、会记住薄弱点、会安排复习”的期末冲刺系统。
