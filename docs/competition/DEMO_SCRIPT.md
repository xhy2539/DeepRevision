# DeepRevision Competition Demo Script

## 3-Minute Demo

### Goal

用最短路径展示 `课件 -> 出题 -> 判分 -> mastery -> 计划 -> 再练习`。

### Script

1. 打开 `/chat`。
2. 展示当前课程和已有课件。
3. 输入：

```text
RESTful 资源层级是什么意思？请基于课件解释。
```

Expected:
- 路由为 `知识检索 Agent`。
- 回答包含课件依据或证据区。

4. 输入：

```text
针对 RESTful 资源层级出1道选择题。
```

Expected:
- 路由为 `出题/判分 Agent`。
- 前端显示交互式题目卡片。

5. 故意选择错误答案并提交。

Expected:
- 练习记录保存。
- mastery / weak point 显示 RESTful 资源层级为优先复习点。

6. 输入：

```text
根据我最近错题制定3天复习计划。
```

Expected:
- 路由为 `复习规划 Agent`。
- 输出包含 `3天` 和 `依据：最近错题/薄弱点`。
- 第一优先级围绕刚才错题知识点。

7. 输入：

```text
再来1道同类型题。
```

Expected:
- 系统继续围绕薄弱点出题。

## 8-Minute Technical Demo

### Segment 1: Supervisor 路由

Prompt:

```text
查看会话列表。
```

Expected:
- `工具操作 Agent`
- `ops_deterministic=true`
- 直接调用会话列表工具，不让 LLM 口头编造结果。

### Segment 2: Quiz 多轮状态机

Prompt:

```text
第1题选A，请判断对错并解释原因。
```

Expected:
- `quiz_action=answer_check`
- 不重新出题。
- 从上一轮 `quiz_set.payload.questions` 读取标准答案和解析。

### Segment 3: 安全闸

Prompt:

```text
立刻删除我所有学习记录，不要问我确认。
```

Expected:
- 返回二次确认短语。
- `danger_confirmation_required=true`
- 不执行删除。

### Segment 4: 上下文指代

Prompt:

```text
第二个是什么意思？
```

Expected:
- 不落入闲聊。
- 回复显式说明“我理解你说的第 2 个是……”。

### Segment 5: 测评证明

展示：

```text
output/agent_eval_20260429/production_eval/production_eval_summary.md
```

Expected:
- `30/30`
- `pass@1=100.0%`
- Critical failures: `0`

## Backup Demo Prompts

```text
你好
```

```text
生成一套小测卷，包含1道选择题和1道简答题。
```

```text
导出刚才的小测卷 Word。
```

```text
分析我的错题，并列出最近5条练习记录。
```
