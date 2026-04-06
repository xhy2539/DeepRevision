# DeepRevision — AI 期末复习引擎

> 基于 RAG + Supervisor 多 Agent + Reflexion 出题 + LangGraph 的智能复习助手

![Python](https://img.shields.io/badge/Python-3.11+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Async%20Backend-009688.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3-FFD700.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Supervisor%20%2B%20Reflexion-orange.svg)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 一、项目定位

**DeepRevision** 是一个面向大学生的 AI 期末复习助手。学生上传自己的课件，系统在课件范围内智能问答、按需出题，同时通过长短期记忆记录学习状态。

### 核心能力

| 能力 | 描述 |
|------|------|
| 📥 多格式课件摄入 | PDF / Word / PPT / TXT / 图片，按页解析，图片内容通过多模态模型并发理解 |
| 🔍 混合检索 | Query Rewrite + BM25 + 向量检索 RRF 融合 + LLM 重排序 |
| 🤖 Supervisor 多 Agent | LangGraph Supervisor 分析意图 → 路由到专业 SubAgent（RAG / 出题 / 出卷 / 规划 / 闲聊） |
| 📝 Reflexion 出题 | LangGraph Reflexion 架构：出题 + 推理链 → Critic 质疑推理 → Revise 针对批评修订（最多 2 轮） |
| 🧠 双轨记忆 | 短期滑窗 + 后台异步图谱提纯，SQLite 本地持久化，重启不丢失 |
| 📄 试卷导出 | 生成完整试卷并导出 Word |
| 🗂️ 多科目隔离 | ContextVar + ChromaDB Collection，每科目独立知识库 |
| 📊 算力监控 | AOP 装饰器实时统计 Token 消耗与调用延迟 |

### 当前实现补充（2026-04）

- 对话流式协议已升级为 `start / delta / complete` 事件，支持后端携带 `kind/render_mode/payload`，前端可稳定渲染 `quiz_set` 与 `exam_paper`。
- 出卷链路默认并发上限为 2，并加入结构化失败时的分批文本补题兜底，避免单批超时导致整卷空白。
- 知识库上传状态改为持久化 `ingest_status.json`（`processing/completed/failed`），并支持失败文件一键重试。
- 新增健康检查接口：`GET /health`。

---

## 二、技术栈

| 层级 | 技术选型 |
|------|----------|
| 语言 | Python 3.11+ |
| Web 框架 | FastAPI + uvicorn（ASGI 异步） |
| LLM 框架 | LangChain 0.3 + LangGraph |
| Agent 编排 | LangGraph StateGraph（Supervisor + Reflexion） |
| 向量库 | ChromaDB（本地持久化） |
| 检索策略 | BM25 + 向量 RRF 混合 + LLM Rerank |
| 大模型 | MiniMax（可替换为任意 OpenAI 兼容模型） |
| 多模态 | MiniMax 视觉模型（图片/PPT 图片理解） |
| 前端 | React + Tailwind CSS |

---

## 三、系统架构

```
Browser (React)
      │  SSE 流式 / REST
      ▼
FastAPI (ASGI) /api/chat/stream
      │
      ▼
Supervisor Agent (LangGraph StateGraph)
  ├── 分析用户意图 → 输出结构化路由决策（JSON）
  └── 条件路由 add_conditional_edges
        ├── rag_agent    → RAG Service（混合检索 + LLM 总结）
        ├── quiz_agent   → Reflexion 出题工作流（3 Agent）
        ├── exam_agent   → Reflexion 出卷工作流（3 Agent）
        ├── planner_agent→ 复习计划生成
        └── chitchat     → 闲聊回复
              │
    ┌─────────┴──────────────────────────────┐
    │           RAG Service                   │
    │  Query Rewrite (LLM)                    │
    │  → BM25 + 向量 RRF 混合检索             │
    │  → LLM Rerank                           │
    │  → 上下文总结                            │
    └────────────────┬───────────────────────┘
                     │
            ChromaDB（per session collection）
```

### Supervisor 路由工作流

```
用户输入
   │
   ▼
Supervisor LLM（分析意图）
   │ 输出 {"route": "rag|quiz|exam|planner|chitchat", "params": {...}}
   ▼
add_conditional_edges（条件路由）
   ├── rag      → rag_agent      → END
   ├── quiz     → quiz_agent     → END
   ├── exam     → exam_agent     → END
   ├── planner  → planner_agent  → END
   └── chitchat → chitchat       → END
```

### Reflexion 出题工作流（LangGraph）

```
出题请求（topic / quiz_type / num）
   │
   ▼
Agent 1: Generate（出题 + 推理链）
  - RAG 检索课件资料
  - 生成题目，同时输出 JSON 推理链：
    knowledge_points / answer_evidence / distractor_design
   │
   ▼
Agent 2: Critic（质疑推理链）
  - 逐条核实推理链声明 vs. 课件原文
  - 输出 approved / overall_score / reasoning_flaws / specific_issues
   │
  需要修订？（满足任一：overall_score<70 / 存在high severity缺陷 / approved=False）
  rounds >= 2？（强制结束）
  ├── 通过或已达2轮 → END（输出最终题目）
  └── 需修订且未到2轮 → Agent 3: Revise
               - 针对每条批评逐条修订
               - 输出 revised_quiz / revision_notes / addressed_issues
               └── 回到 Critic（最多 2 轮）
```

> Reflexion 核心：推理链（Chain of Thought）显式化，Critic 质疑的是推理过程而非最终输出，Revise 必须逐条回应批评并说明修改依据。

### 双轨记忆架构

```
每轮对话
   │
   ▼
短期记忆（滑窗 N 条）
   │ 超出阈值
   ▼
trash 暂存区
   │ 累积到 2N 条
   ▼
后台异步图谱提纯（asyncio.create_task，不阻塞对话）
   │
   ▼
长期图谱记忆（知识点结构化摘要）
   │
   └─── 下次对话时注入 Supervisor system prompt
```

---

## 四、目录结构

```
DeepRevision/
├── api/
│   ├── main.py                  # FastAPI 入口，路由挂载
│   └── routers/
│       ├── chat.py              # 对话 SSE（接入 Supervisor 工作流）
│       ├── knowledge.py         # 文件上传 + 样卷管理
│       └── exam_export.py       # 试卷导出 Word
│
├── agent/
│   ├── tools/
│   │   └── agent_tools.py       # RAG 缓存清理工具（clear_rag_cache）
│   └── multi_agent/
│       ├── supervisor.py        # Supervisor 路由 + 5 个 SubAgent 节点
│       └── quiz_agent.py        # Reflexion 出题 + 出卷工作流（3 Agent）
│
├── rag/
│   ├── rag_service.py           # 混合检索 + Query Rewrite + LLM Rerank
│   └── vector_store.py          # ChromaDB 封装 + 并发图片处理
│
├── model/
│   └── factory.py               # LLM / Embedding / Vision 模型工厂
│
├── utils/
│   ├── memory_service.py        # 双轨记忆（滑窗 + 异步图谱提纯）
│   ├── session_context.py       # ContextVar 会话隔离
│   ├── logger_handler.py        # AOP 装饰器 + 线程安全算力统计
│   ├── file_handler.py          # 多格式文档解析（PDF/PPT/图片/Word）
│   ├── config_handler.py        # YAML 配置读取
│   └── prompt_loader.py         # 提示词加载
│
├── config/
│   ├── chroma.yml               # 向量库 + 分块 + 检索参数
│   ├── rag.yml                  # 模型配置
│   ├── agent.yml                # Agent 行为配置
│   └── prompts.yml              # 提示词路径配置
│
├── prompts/                     # 提示词模板
├── react/                       # React 前端
└── data/                        # 上传文件（按 session_id 隔离）
    └── {session_id}/
```

---

## 五、工程亮点

### 1. Supervisor 多 Agent 路由（LangGraph）

用一个 LLM 做意图分类和参数提取，输出结构化 JSON，再由 `add_conditional_edges` 分发到专业 SubAgent。Supervisor 同时注入近期对话历史，支持多轮追问的正确路由（"再出 3 道"、"换成判断题"等）。Supervisor 节点的 LLM token 不透传给用户。

```python
# supervisor_node — 注入近期对话 + 过滤路由 JSON
history_lines = [f"{'学生' if isinstance(m, HumanMessage) else '助手'}: {m.content[:200]}"
                 for m in state.get('chat_history', [])]
result = await _call_llm(SUPERVISOR_PROMPT,
    input=state['input'],
    memory_context=state.get('memory_context', ''),
    recent_history="\n".join(history_lines) or "（无近期对话）")

# chat.py — 过滤 Supervisor 的 JSON token
async for event in supervisor_workflow.astream_events(initial_state, version="v2"):
    if event.get("metadata", {}).get("langgraph_node", "") == "supervisor":
        continue   # 路由决策 JSON 不暴露给用户
```

### 2. Reflexion 出题：推理链显式化

传统质检只看最终输出对不对。Reflexion 要求出题者先写下推理链（每道题的知识点来源、答案依据、干扰项设计逻辑），Critic 逐条核实推理链 vs. 课件原文。

```python
# Agent 1 输出 JSON 推理链
{
  "quiz": "完整题目文本...",
  "reasoning": {
    "knowledge_points": ["第1题考察X，来源：课件原文'...'"],
    "answer_evidence":  ["第1题答案A的依据：课件原文'...'"],
    "distractor_design":["第1题干扰项B：容易与X混淆，因为..."]
  }
}

# Agent 2 质疑推理链（不只看题目对错）
{
  "approved": false, "overall_score": 68,
  "reasoning_flaws": [
    {"question":"第1题","flaw":"推理链称依据是'...'，但课件原文是'...'，两者不符","severity":"high"}
  ]
}

# Agent 3 针对批评逐条修订
{
  "revised_quiz": "修订后完整题目...",
  "revision_notes": "1. 针对'第1题答案依据不准确'：将答案从B改为C，因为课件明确写道'...'"
}
```

### 3. Session 隔离：ContextVar 上下文穿透

FastAPI async handler 与 LangGraph 节点跨越多个协程栈。用 `threading.local` 无法穿透协程，用全局变量无法隔离并发用户。

```python
# utils/session_context.py
current_session_id: ContextVar[str] = ContextVar("current_session_id", default="default")

# 请求入口绑定
current_session_id.set(session_id)

# RAG 服务深处自动获取，无需层层传参
class VectorStoreService:
    def __init__(self):
        sid = current_session_id.get()   # 自动拿到当前请求的 session
        col = f"col_{hashlib.md5(sid.encode()).hexdigest()[:16]}"
        self.vector_store = Chroma(collection_name=col, ...)
```

### 4. 混合检索 RRF 融合 + 检索/回答分离

BM25 擅长精确关键词匹配，向量检索擅长语义理解，两者互补。RRF 不依赖人工调权重，按名次倒数融合。

RAG Service 分为两个方法，避免 SubAgent 中产生冗余 LLM 调用：

```python
# retrieve_context(query) — 只检索，返回原始课件片段
# 供 rag_subagent_node 和 quiz_agent 使用：SubAgent 自己调一次 LLM 完成回答/出题
async def retrieve_context(self, query: str) -> str:
    expanded_query = await self.rewrite_chain.ainvoke(...)  # LLM ×1
    docs = await self.retriever_docs(expanded_query)
    docs = await self._rerank(expanded_query, docs)         # LLM ×1
    return 格式化原始片段

# rag_summarize(query) — 检索 + 最终 LLM 总结
# 保留用于直接调用场景（不经过 Supervisor）
async def rag_summarize(self, query: str) -> str:
    context = await self.retrieve_context(query)
    return await self.chain.ainvoke({"input": query, "context": context})  # LLM ×1
```

```python
# RRF 公式: score(d) = Σ 1/(k + rank(d))，k=60
def _rrf_fuse(self, docs1, docs2):
    for rank, doc in enumerate(docs1, 1):
        doc_scores[key] += 1 / (self.rrf_k + rank)
    for rank, doc in enumerate(docs2, 1):
        doc_scores[key] += 1 / (self.rrf_k + rank)
    return sorted by score
```

流程：Query Rewrite → BM25 + 向量并行检索 → RRF 融合 → LLM Rerank 精筛

### 5. 图片并发处理

PDF/PPT 内每张图片都需要调用视觉模型。串行调用在图多时极慢（150 张 ≈ 150 次串行等待）。

```python
# vector_store.py — _enrich_images()
async def _describe_one(blob, context, label):
    loop = asyncio.get_event_loop()
    # 同步视觉模型调用放入线程池，不阻塞事件循环
    desc = await loop.run_in_executor(None, _summarize_image, blob, context)
    return label, desc

# 所有图片并发处理
results = await asyncio.gather(*tasks, return_exceptions=True)
```

### 6. 文档按页解析

自定义解析器按页生成 Document，页面边界不会被 splitter 打破，避免跨页知识点混入同一 chunk。Unstructured 作为兜底仅在自定义解析失败时启用。

```python
# 每页独立 Document → splitter 切割时天然隔离
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    ...
    documents.append(Document(page_content=..., metadata={"page": page_num+1}))
```

### 7. SQLite 持久化 + 异步写入

会话数据（对话历史、图谱节点）持久化到本地 `sessions.db`，重启不丢失。写入通过 `run_in_executor` 放入线程池，不阻塞事件循环。首次启动自动迁移旧 `sessions.json` 数据。

```python
# memory_service.py — 异步写入 SQLite
async def _persist(self, session_id: str):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, partial(_save_session_sync, self.db, session_id, session)
    )

# 启动时同步加载，之后所有写操作全异步
self.store = _load_all_sync(self.db)   # __init__ 一次性加载到内存
await self._persist(session_id)        # 每次变更异步落盘
```

### 8. 异步记忆提纯

图谱提纯是重操作（LLM 调用），不能阻塞对话响应。

```python
# memory_service.py
async def add_message(self, session_id, role, content):
    ...
    if len(session["trash"]) >= self.trash_threshold * 2:
        asyncio.create_task(self._trigger_background_distillation(session_id))
        # fire-and-forget，立即返回，不等待提纯完成
```

---

## 六、API 接口

### 对话

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/chat/stream` | SSE 流式对话（Supervisor 路由） |
| GET | `/api/chat/sessions` | 获取所有会话列表 |
| POST | `/api/chat/session` | 创建新会话 |
| PUT | `/api/chat/session/{id}` | 重命名会话 |
| DELETE | `/api/chat/session/{id}` | 销毁会话及知识库 |
| GET | `/api/chat/messages` | 获取会话消息历史 |
| GET | `/api/chat/tokens` | 算力消耗统计 |

### 知识库

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/knowledge/upload` | 上传课件（最多5个，后台向量化） |
| GET | `/api/knowledge/list` | 已上传文件列表（含 processing/completed/failed 状态） |
| POST | `/api/knowledge/retry-failed` | 重试当前会话中失败文件 |
| POST | `/api/knowledge/sample/upload` | 上传样卷（学习试卷风格） |
| GET | `/api/knowledge/sample` | 获取样卷格式 |
| DELETE | `/api/knowledge/sample` | 删除样卷 |

### 试卷导出

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/exam/format` | 格式化试卷（添加难度、分值等元信息） |
| POST | `/api/exam/export/docx` | 导出试卷为 Word |
| POST | `/api/exam/answersheet` | 生成答案卷 Word |
| GET | `/api/exam/download/{filename}` | 下载导出文件 |
| GET | `/health` | 服务健康检查 |

### 用户认证

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/auth/register` | 用户注册 |
| POST | `/api/auth/login` | 用户登录 |
| GET | `/api/auth/check` | 检查登录状态 |

---

## 七、配置说明

### `config/chroma.yml` — 检索参数（改此文件即生效）

```yaml
chunk_size: 1000        # 分块大小（字符数）
chunk_overlap: 200      # 分块重叠
k: 8                    # 向量检索 top-k

retrieve_top_k: 10      # 混合检索召回数量
rerank_top_k: 8         # LLM 重排后保留数量
rrf_k: 60               # RRF 公式参数
```

### `config/rag.yml` — 模型配置

```yaml
chat_model_name: abab6.5s-chat       # 主对话模型
embedding_model_name: text-embedding-v3
vision_model_name: abab6.5g-chat     # 图片理解模型
```

### `config/agent.yml` — Agent 行为

```yaml
web_search:
  enabled: true    # DuckDuckGo 联网搜索开关（agent_tools.py 中定义，按需启用）
```

---

## 八、快速开始

### 1. 安装依赖

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# .env
MINIMAX_API_KEY=your_minimax_api_key
```

> 使用其他模型只需修改 `model/factory.py` 中的 `ChatModelFactory`，接口兼容 OpenAI 标准。

### 3. 启动服务

```bash
# 后端
python run.py
# 或
uvicorn api.main:app --reload --port 8001

# 前端
cd react && npm install && npm run dev
```

访问 `http://127.0.0.1:3000`

---

## 九、面试核心问答

**Q: 多轮对话中追问如何正确路由？**

Supervisor prompt 注入 `{recent_history}`（最近 N 条对话格式化为"学生/助手"文本）。追问"再出 3 道"时 Supervisor 能看到上一轮是出题请求，从而正确路由到 quiz 而非 rag。每条消息截断 200 字防止 prompt 过长。

**Q: RAG 路径有几次 LLM 调用？**

`retrieve_context` 路径（Supervisor RAG SubAgent / 出题 Agent 使用）：Query Rewrite × 1 + LLM Rerank × 1 = 2 次，最终回答由 SubAgent 再调 1 次，共 3 次。`rag_summarize` 路径（完整链路）：额外 +1 次总结 LLM，共 4 次。两个方法分开暴露，避免 SubAgent 把 LLM 摘要当 context 再喂给另一个 LLM（信息被稀释两层）。

**Q: SQLite 写入策略？**

正常追加消息走单行 `INSERT`（`_append_message_sync`），不重写历史。只有滑动窗口触发（recent → trash 的 bucket 迁移）时才全量覆写，保证 bucket 状态一致。图谱提纯后的新节点也是追加写。

**Q: 怎么知道 Reflexion 是否真的有效？**

`critique_quiz_node` 在 `reflection_rounds==0` 时把首轮评分存入 `initial_score`。最终日志输出 `评分变化=65 → 82 (+17)`，可逐次积累判断 Reflexion 修订有无实质提升，还是 Critic 大多直接放行（轮次=0 占比高）。

**Q: 持久化怎么做的？**

会话数据存 SQLite（stdlib `sqlite3`，无额外依赖）。启动时同步加载全量数据到内存 `store` 字典作为读缓存，每次变更通过 `run_in_executor` 异步写回 SQLite，不阻塞事件循环。ChromaDB 向量库自带持久化。首次启动自动检测并迁移旧 JSON 数据。

**Q: 整体架构是什么？**

两层 LangGraph 工作流。外层：Supervisor 节点分析意图，`add_conditional_edges` 路由到 5 个 SubAgent（RAG/出题/出卷/规划/闲聊）。内层（出题/出卷）：Reflexion 架构，Generate+推理链 → Critic质疑推理 → Revise修订，最多 2 轮。

**Q: Reflexion 与普通质检有什么区别？**

普通质检只看输出对不对（"答案是否来自课件"）。Reflexion 要求 Agent 1 先显式写出推理链，Critic 质疑的是推理链声明是否成立，Revise 必须逐条回应批评并给出修改说明。这样使 Agent 的内部推理过程变得可审计、可批评、可改进。

**Q: 如何防止大模型幻觉？**

两层防御：① Supervisor 路由到 RAG SubAgent 时，系统 prompt 强制"只能基于课件回答"；② Reflexion Critic 专门核实每道题的推理链引用是否与课件原文一致，发现引用错误触发 Revise 修订循环。

**Q: Session 隔离怎么实现的？**

`ContextVar` 在协程粒度绑定 session_id，从请求入口一直透传到 ChromaDB Collection 选择，无需层层传参，且天然隔离并发用户。Supervisor SubAgent 节点在调用 RAG 前都会显式 `current_session_id.set(sid)` 确保在新协程栈中也能正确拿到。

**Q: SSE 流式输出中如何处理 Supervisor 的路由 JSON？**

用 `astream_events(version="v2")`，每个事件携带 `metadata.langgraph_node`。在 `on_chat_model_stream` 事件里先判断 `if node == "supervisor": continue`，Supervisor 的 JSON 输出（`{"route":"quiz","reason":"..."}`）不会透传给前端。

**Q: 出题/出卷结果如何从 SSE 输出？**

quiz_agent/exam_agent 内部调用 `ainvoke`（非流式），结果存在 `final_answer` 字段。在 `on_chain_end` 事件中检测到这两个节点完成后，从 `output["final_answer"]` 取值，按行拆分逐行 yield（间隔 30ms），给用户呈现流式感而不是一次性弹出全文。

**Q: 多 Agent 出题系统 LLM 调用次数？**

最优路径：2 次（Generate × 1 + Critique × 1，直接通过）。最多：6 次（Generate × 1 + Critique × 3 + Revise × 2，走满 2 轮修订后第 3 次 Critique 因 rounds≥2 强制结束）。触发修订的条件是三选一：`overall_score < 70`、存在 high severity 推理缺陷、或 `approved=False`，避免不必要的循环。

**Q: 如何扩展到更大规模？**

| 瓶颈 | 扩展方案 |
|------|----------|
| 内存记忆 | 迁移 Redis |
| ChromaDB | 换用 Milvus / Qdrant |
| 单进程 FastAPI | Gunicorn 多 Worker + Nginx |
| 模型费用 | 接入语义缓存 + 小模型做 Supervisor 路由 |

---

## 十、已知待改进点

> 以下问题真实存在于当前代码中，列出便于后续迭代。

### 1. ~~Supervisor 路由不感知多轮对话~~ ✅ 已修复

`SUPERVISOR_PROMPT` 现已注入 `{recent_history}`，将最近 N 条对话格式化为"学生/助手"文本传入。追问如"再出 3 道"、"换成判断题"能被正确感知并路由。

### 2. ~~RAG SubAgent 存在冗余 LLM 调用~~ ✅ 已修复

新增 `RagSummarizeService.retrieve_context(query)` 方法，只做检索（Query Rewrite + RRF + Rerank），返回格式化原始课件片段，不做最终 LLM 总结。`rag_subagent_node` 和 `quiz_agent` 的 `get_rag_context` 均改用此方法，RAG 路径从 4 次 LLM 调用减为 2 次（Query Rewrite + Rerank），最终回答由 SubAgent 的一次 LLM 调用完成。

### 3. ~~并发图片处理无速率限制~~ ✅ 已修复

`asyncio.gather(*tasks)` 对文档中所有图片同时发起视觉模型 API 请求，无并发上限。图片密集的课件（如每页含多张图的 PPT）会在短时间内打出大量并发请求，容易触发 API rate limit 报错。

**修复方案：** `VectorStoreService` 类添加 `MAX_CONCURRENT_IMAGES = 5` 属性，`__init__` 中创建 `asyncio.Semaphore`，`_describe_one` 内部用 `async with self._image_semaphore` 控制并发。现在 100 张图片会排队处理，最多 5 个同时请求 API。

### 4. ~~SQLite 全量覆写~~ ✅ 已修复

`add_message` 在正常追加场景改为单行 `INSERT`（`_append_message_sync`），只在滑动窗口触发（消息 bucket 发生迁移）时才全量覆写。写放大从"每条消息 O(n)"降为"仅在窗口滑动时 O(n)"。

### 5. ~~无法验证 Reflexion 循环的实际效果~~ ✅ 已修复

`critique_quiz_node` / `critique_exam_node` 在首轮（`reflection_rounds==0`）执行后将评分存入 `initial_score` 字段。最终日志格式为：

```
[Reflexion统计] topic=XX | 修订轮次=1 | 评分变化=65 → 82 (+17) | 高危缺陷=0 | 使用修订版=True
```

积累若干样本后可定量评估 Reflexion 的实际提升幅度。

### 6. "图谱记忆"命名与实现不符

代码中的长期记忆是 LLM 把对话历史提炼为结构化三元组（subject / relation / object），以列表形式存储。没有图结构、没有节点间的边关系查询、没有路径搜索。对外介绍时称"知识图谱"会引起误解。

实际上它是**结构化摘要记忆**，表达能力介于纯文本摘要和真正的知识图谱之间。

### 7. ~~用户输入无长度限制~~ ✅ 已修复

极长输入可能超出模型 context 限制。

**修复方案：** `chat.py` 入口处 `query = query[:2000]` 截断超长输入。

### 8. ~~session_id 路径穿越风险~~ ✅ 已修复

`session_id` 由前端传入用于构建路径，若包含 `../` 存在路径穿越风险。

**修复方案：** 入口正则校验 `^[a-zA-Z0-9_\-]{1,64}$`，不合规返回 400。

### 9. ~~PPT 标题为 None 时崩溃~~ ✅ 已修复

`slide.shapes.title` 为 None 时访问 `.text` 抛 `AttributeError`。

**修复方案：** 提前获取 `title_text = title_shape.text if title_shape else ""`。

### 10. ~~SQLite 缺索引和 WAL~~ ✅ 已修复

`messages` 和 `graph_nodes` 表按 `session_id` 查询无索引，默认日志模式并发性能差。

**修复方案：** `_create_tables()` 添加 `PRAGMA journal_mode=WAL` 和 `CREATE INDEX`。

### 11. ~~文件上传无去重~~ ✅ 已实现

`load_document()` 已有 `chunk_md5_hex()` 和 `save_md5_hex()` 函数，入库前检查 MD5 存在则跳过。

### 12. 缺少已上传文件删除 API

用户上传课件后无法删除，只能创建新会话隔离。需要在 `knowledge.py` 添加 `DELETE /api/knowledge/file/{file_id}` 端点。

---
