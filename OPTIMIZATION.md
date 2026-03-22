# DeepRevision — 后续优化思路

> 基于当前代码逐模块分析，按实际代码位置标注，优先级评估见末尾汇总表。

---

## 一、Agent 架构

### 1. SubAgent 无错误降级
**位置**：`agent/multi_agent/supervisor.py` — 各 SubAgent 节点函数

任一 SubAgent 抛出未捕获异常，整个 `supervisor_workflow.ainvoke` 直接崩溃，前端收到 500。应在每个 SubAgent 节点加 try/except，失败时向 `final_answer` 写友好提示并正常走 END，不中断整条流。

```python
# 当前
async def rag_subagent_node(state):
    ...  # 异常直接向上抛

# 应该
async def rag_subagent_node(state):
    try:
        ...
    except Exception as e:
        logger.error(f"[RAG SubAgent] 异常: {e}")
        return {"final_answer": "抱歉，知识库查询暂时出现问题，请稍后再试。"}
```

---

### 2. `_call_llm` / `_extract_json` 代码重复
**位置**：`agent/multi_agent/supervisor.py:114` 和 `agent/multi_agent/quiz_agent.py:69`

两个文件各自定义了一份几乎相同的 `_call_llm` 和 `_extract_json`。修改时容易只改一处。应提取到 `utils/llm_utils.py` 统一维护。

---

### 3. Reflexion JSON 解析失败兜底过于宽松
**位置**：`agent/multi_agent/quiz_agent.py:314-317`

```python
except Exception:
    critique = {"approved": True, "overall_score": 80, ...}
```

LLM 输出格式异常时静默给 80 分直接通过，掩盖了真实错误。应记录原始 LLM 输出便于排查，而不是直接放行。

---

### 4. Supervisor 路由参数无范围校验
**位置**：`agent/multi_agent/supervisor.py:196-197`

`num`、`total_questions` 直接从 LLM JSON 取用，没有合法值限制。用户说"出 1000 道题"会被原样传入，触发超长 LLM 调用。应在 SubAgent 入口做 clamp：

```python
num = min(max(int(params.get('num', 3)), 1), 20)
total = min(max(int(params.get('total_questions', 10)), 1), 50)
```

---

### 5. quiz/exam 流式输出是伪流式
**位置**：`api/routers/chat.py:95-103`

结果已完整生成后才开始按行输出（30ms/行），是 UX 层面的模拟，Reflexion 循环期间（最多 6 次 LLM 调用）用户看到的是空白。

根本解决方案：将 `quiz_workflow` / `exam_workflow` 注册为 LangGraph 子图节点（`graph.add_node("quiz_agent", quiz_workflow)`），内层图的事件会自动冒泡到外层 `astream_events`，实现真正的逐 token 流式。代价是需要做 `SupervisorState` ↔ `QuizState` 的输入/输出映射。

---

## 二、RAG 管线

### 6. `RagSummarizeService` 每次请求重建 BM25 索引
**位置**：`agent/multi_agent/supervisor.py:168`、`agent/multi_agent/quiz_agent.py:65`

每次 SubAgent 调用都 `RagSummarizeService()`，`__init__` 内的 `self.refresh()` 会从 ChromaDB 拉取全量文档重建 BM25 索引。高频请求下这是明显性能瓶颈。

改进方向：按 `session_id` 做单例缓存，上传新文件后调 `refresh()` 刷新，其余请求复用已有实例。

---

### 7. ~~并发图片处理无速率限制~~ ✅ 已修复
**位置**：`rag/vector_store.py` — `_enrich_images()`

`asyncio.gather(*tasks)` 同时发出所有图片 API 请求，无并发上限。图片密集的课件（100+ 张）会瞬间打满 API rate limit。

**修复方案：** 类属性定义 `MAX_CONCURRENT_IMAGES = 5`，`__init__` 中创建 `Semaphore`，`_describe_one` 内部用 `async with self._image_semaphore` 控制并发。

---

### 8. Query Rewrite 与 Rerank 使用主力模型
**位置**：`rag/rag_service.py:124-127`

Query Rewrite（改写查询词）和 LLM Rerank（排序）不需要强推理能力，但目前与出题、回答共用同一个 `chat_model`（MiniMax abab6.5s）。可配置为更轻量的模型，降低延迟和费用。

---

### 9. 无向量查询缓存
**位置**：`rag/rag_service.py` — `retrieve_context()`

相同或相近的问题每次都走完整检索流程（Query Rewrite + BM25 + Vector + Rerank）。可对 `retrieve_context` 的结果按 query 做 LRU 缓存（TTL 10 分钟），重复提问直接命中缓存。

---

### 10. ~~文件上传无去重校验~~ ✅ 已实现
**位置**：`rag/vector_store.py`

同一课件重传会在 ChromaDB 里写两份向量，检索结果出现重复片段，影响召回质量。

**实现方案：** `load_document()` 中已有 `chunk_md5_hex()` 和 `save_md5_hex()` 函数，入库前检查 MD5 是否已存在，存在则跳过。

---

## 三、记忆服务

### 11. trash 数据在提纯失败时永久丢失
**位置**：`utils/memory_service.py:153-157`

```python
# 当前（有风险）
trash_data = list(self.store[session_id]["trash"])
self.store[session_id]["trash"].clear()   # ← 先清空
await self._distill_to_graph(...)         # ← 这里失败 = 数据永久丢失
```

LLM 调用失败（网络超时、rate limit）后，这批历史对话记录无法重试。应改为提纯成功后再清空：

```python
try:
    await self._distill_to_graph(session_id, trash_data)
    self.store[session_id]["trash"].clear()  # 成功才清
except Exception as e:
    logger.error(f"[记忆提纯失败] {e}，trash 保留待下次重试")
```

---

### 12. 图谱节点无上限、无 TTL
**位置**：`utils/memory_service.py` — `_distill_to_graph()`

`graph` 列表只增不减，长期使用后无限堆积。每轮对话都将全部图谱节点注入 Supervisor prompt，最终超出 token 上限导致请求失败。

改进方向：设置上限（如 50 条），超出时按插入时间淘汰最旧节点；或定期合并相似节点。

---

### 13. ~~SQLite 缺索引和 WAL 模式~~ ✅ 已修复
**位置**：`utils/memory_service.py` — `_create_tables()`

`messages` 和 `graph_nodes` 表按 `session_id` 查询，但没有对该列建索引。会话数据增多后退化为全表扫描。另外默认日志模式下并发读写性能差。

**修复方案：** `_create_tables()` 中添加：
```python
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_sid ON messages(session_id);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_sid ON graph_nodes(session_id);")
```

---

## 四、文档解析

### 14. ~~PPT 标题为 None 时有潜在崩溃~~ ✅ 已修复
**位置**：`utils/file_handler.py:244`

```python
if shape.text != slide.shapes.title.text:  # slide.shapes.title 可能为 None
```

**修复方案：** 提前获取 title_shape 和 title_text，避免空指针：
```python
title_shape = slide.shapes.title
title_text = title_shape.text if title_shape else ""
for shape in slide.shapes:
    ...
    if text and text != title_text:
```

---

### 15. OCR 功能静默失败
**位置**：`utils/file_handler.py:364-366`、`489-495`

`pytesseract` 的 `ImportError` 被 except 吞掉，扫描件课件会全部解析为空白，用户无任何感知。应在 `SessionMemoryManager` 启动时或首次调用时检测一次并打印明确警告：

```python
try:
    import pytesseract
    pytesseract.get_tesseract_version()
except Exception:
    logger.warning("[OCR] pytesseract 未安装或 Tesseract 未配置，扫描件将无法识别文字")
```

---

## 五、API 层

### 16. 用户输入无长度限制
**位置**：`api/routers/chat.py:49`

极长输入（配合 `recent_history` 后 prompt 更长）可能超出模型 context 限制。应在入口截断：

```python
query = body.get("query", "")[:2000]
```

---

### 17. session_id 未做路径安全校验
**位置**：`api/routers/chat.py:49`、`api/routers/knowledge.py`

`session_id` 由前端传入，用于构建 `data/{session_id}/` 路径。若包含 `../` 存在路径穿越风险。应在入口正则校验：

```python
import re
if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', session_id):
    raise HTTPException(status_code=400, detail="非法 session_id")
```

---

### 18. 错误响应格式不统一
**位置**：`api/routers/chat.py`、`api/routers/knowledge.py`

部分接口返回 `{"code": 404, "message": "..."}` 自定义格式，部分直接抛 `HTTPException`，前端需要处理两套格式。应统一为 FastAPI 标准异常机制。

---

### 19. `agent.yml` 的 web_search 开关已失效
**位置**：`config/agent.yml`、`agent/tools/agent_tools.py:10`

`agent_tools.py` 读取该配置，但自从 `AgentExecutor` 被替换为 Supervisor 工作流后，`tools` 列表不再被使用。该配置项目前没有任何代码路径读取它来实际控制行为，是误导性的死配置。应接线（给 planner_agent 接入 web_search）或删除该配置项。

---

## 六、可观测性

### 20. Token 统计不持久化
**位置**：`utils/logger_handler.py` — `update_token_stats()`

Token 统计存在内存，重启归零，无法查看历史消耗趋势或估算费用。应每次更新时追加一条记录到 SQLite 的 `token_stats` 表（时间戳 + prompt_tokens + completion_tokens）。

---

### 21. 无请求链路追踪
**位置**：所有 Agent 节点

一次出题请求经过 Supervisor → quiz_agent → generate → critique → revise → critique 多个节点，各节点日志之间无关联 ID，排查问题时无法把一次完整请求的所有日志串在一起。

改进方向：在 `SupervisorState` 加 `request_id: str` 字段，`chat.py` 入口生成 UUID，所有节点日志都带上它。

---

### 22. Reflexion 效果数据无聚合
**位置**：`agent/multi_agent/quiz_agent.py` — `run_quiz_agent()`

`initial_score → final_score` 目前只打日志，无法做统计分析。应存到 SQLite 的 `reflexion_stats` 表，积累后可查询"平均提升分数"、"轮次=0（直接通过）占比"等指标。

---

## 七、配置

### 23. Supervisor 参数硬编码
**位置**：`agent/multi_agent/supervisor.py:136`

`recent_history` 每条截断 200 字、传入全部历史——这两个值硬编码在节点函数里。应提取到 `config/agent.yml`：

```yaml
supervisor:
  history_max_turns: 6          # 传给 Supervisor 的最近对话轮数
  history_max_chars_per_turn: 200  # 每条消息截断字数
```

---

### 24. 无健康检查接口
**位置**：`api/main.py`

没有 `/health` 端点，无法在容器或负载均衡场景做存活探测。只需加一个返回 200 的简单路由即可。

---

## 八、测试

### 25. 零测试覆盖

没有任何单元测试或集成测试。最高风险的函数应优先补：

| 函数 | 风险点 |
|------|--------|
| `_critique_needs_revision` | 三条件逻辑，边界 case 多 |
| `_rrf_fuse` | 排序正确性，分数计算 |
| `_extract_json` | LLM 输出格式多变，健壮性关键 |
| `supervisor_node` 路由 | mock LLM，测各类意图的路由正确性 |
| `_migrate_from_json` | 数据迁移正确性 |

---

## 优先级汇总

| 优先级 | 编号 | 描述 | 分类 |
|--------|------|------|------|
| 🔴 高 | 1 | SubAgent 无错误降级 | 稳定性 |
| 🔴 高 | 11 | trash 数据提纯失败时丢失 | 稳定性 |
| 🔴 高 | 14 | PPT 标题 None 崩溃 | Bug |
| 🔴 高 | ~~16~~ | ~~用户输入无长度限制~~ ✅ | 安全 |
| 🔴 高 | ~~17~~ | ~~session_id 路径穿越~~ ✅ | 安全 |
| 🟡 中 | 6 | RagSummarizeService 重建 BM25 | 性能 |
| 🟡 中 | ~~7~~ | ~~并发图片无速率限制~~ ✅ | 稳定性 |
| 🟡 中 | ~~10~~ | ~~文件上传无去重~~ ✅ | 数据质量 |
| 🟡 中 | 12 | 图谱节点无上限 | 长期稳定性 |
| 🟡 中 | ~~13~~ | ~~SQLite 缺索引 + WAL~~ ✅ | 性能 |
| 🟡 中 | 19 | agent.yml web_search 死配置 | 准确性 |
| 🟡 中 | 20 | Token 统计不持久化 | 可观测性 |
| 🟢 低 | 2 | _call_llm 代码重复 | 工程质量 |
| 🟢 低 | 3 | JSON 解析失败兜底宽松 | 健壮性 |
| 🟢 低 | 4 | 路由参数无范围校验 | 健壮性 |
| 🟢 低 | 5 | quiz/exam 伪流式 | 体验 |
| 🟢 低 | 8 | Query Rewrite 用主力模型 | 成本 |
| 🟢 低 | 9 | 无查询缓存 | 性能 |
| 🟢 低 | ~~14~~ | ~~PPT 标题 None 崩溃~~ ✅ | Bug |
| 🟢 低 | 15 | OCR 静默失败 | 可观测性 |
| 🟢 低 | 18 | 错误响应格式不统一 | 工程质量 |
| 🟢 低 | 21 | 无链路追踪 | 可观测性 |
| 🟢 低 | 22 | Reflexion 数据无聚合 | 可观测性 |
| 🟢 低 | 23 | Supervisor 参数硬编码 | 可维护性 |
| 🟢 低 | 24 | 无健康检查接口 | 部署 |
| 🟢 低 | 25 | 零测试覆盖 | 工程质量 |
