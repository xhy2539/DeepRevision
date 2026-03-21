# DeepRevision — AI 期末复习引擎 · 完整技术文档

> 本文档依据截至 2026-03-11 的代码状态生成，供下一个开发会话或面试复盘使用。

---

## 一、项目定位与背景

**DeepRevision** 是基于 **RAG (检索增强生成)** + **ReAct (推理与行动) Agent 范式** 构建的垂直领域大模型智能体，专为解决大学生期末复习"课件资料分散、知识点零乱、无法定向出题"的痛点而设计。

通过上传 PDF / Word / TXT 格式课件，用户可以：
- 向引擎提问任意知识点，得到有课件文本出处支撑的回答
- 要求引擎按考点自动生成选择题（含解析）
- 在多个独立科目会话之间切换，彼此记忆与知识库完全隔离

---

## 二、完整技术栈

| 层级 | 组件 | 说明 |
|---|---|---|
| 语言运行时 | Python 3.11 | 主控语言 |
| LLM 框架 | LangChain v1.2+ | `langchain-core`, `langchain-community`, `langchain-classic` |
| 大模型后端 | 阿里云通义千问 (Qwen) DashScope API | 可在 `model/factory.py` 替换任意厂商 |
| 向量数据库 | ChromaDB (Local Persist) | 每个 Session 独立 Collection |
| API 网关 | FastAPI + Uvicorn (ASGI) | 异步非阻塞 |
| 数据传输协议 | Server-Sent Events (SSE) | 实现"打字机"流式Token输出 |
| 前端 | 原生 HTML5 + Tailwind CSS + Vanilla JS | 无框架, 极简工业设计 |
| Markdown渲染 | Marked.js | 纯前端渲染LLM长文本 |
| 异步提交 | FastAPI BackgroundTasks | 文件摄入不阻塞HTTP响应 |

---

## 三、目录结构速览

```
DeepRevision/
│
├── api/                        # FastAPI 应用层
│   ├── main.py                 # 应用入口, 挂载路由和静态文件
│   └── routers/
│       ├── chat.py             # /api/chat/* — 对话、会话管理、Token统计
│       └── knowledge.py        # /api/knowledge/upload — 文件上传与摄入
│
├── agent/
│   └── tools/
│       └── agent_tools.py      # ReAct 工具箱 (search_courseware, generate_quiz)
│
├── rag/
│   ├── rag_service.py          # 混合检索 + Query Rewrite + 汇总Chain
│   └── vector_store.py         # ChromaDB 包装层，Session 隔离逻辑在此
│
├── model/
│   └── factory.py              # LLM & Embedding 模型工厂 (统一入口)
│
├── utils/
│   ├── memory_service.py       # 双轨记忆引擎 (短期滑窗 + 长期图谱萃取)
│   ├── session_context.py      # ContextVar: 传递当前 session_id 穿越调用栈
│   ├── logger_handler.py       # AOP 装饰器 + Token / 延迟统计探针
│   ├── config_handler.py       # YAML 配置文件读取
│   ├── path_tool.py            # 路径工具函数
│   ├── prompt_loader.py        # 加载 prompts/ 目录下的提示词文件
│   └── file_handler.py         # PDF / DOCX / TXT Loader 封装
│
├── config/
│   ├── chroma.yml              # 向量库核心参数（chunk_size, k, 文件类型等）
│   └── prompts.yml             # 提示词文件路径配置
│
├── prompts/
│   ├── main_prompt.txt         # Agent ReAct 主系统提示词
│   └── rag_summarize.txt       # RAG 课件总结 Prompt
│
├── static/
│   └── index.html              # 全量前端单页应用 (SPA)
│
└── data/                       # 上传文件物理存储根目录
    └── {session_id}/           # 每个科目会话独立子目录
```

---

## 四、核心模块深度解析

### 4.1 会话隔离架构 (`utils/session_context.py`)

这是本次迭代最底层的基础设施改造。

**问题**：FastAPI 的异步协程中，传统 `threading.local()` 失效。如果多个并发用户同时发起请求，一个全局变量会被所有请求串改，造成"用户 A 的文件摄入到用户 B 的知识库"这类灾难性数据污染。

**方案**：使用 Python 3.7+ 原生的 `contextvars.ContextVar`：

```python
from contextvars import ContextVar

current_session_id: ContextVar[str] = ContextVar("current_session_id", default="default")
```

- 每一个进入 `/api/chat/stream` 的异步请求都会立即调用 `current_session_id.set(req.session_id)`
- 这个值在整个异步调用链中自动透传（包括 `await` 切换后的续帧），并且**完全线程/协程安全**
- `VectorStoreService.__init__` 中通过 `current_session_id.get()` 拿到当前请求对应的 Session，据此构建 ChromaDB 集合名

---

### 4.2 Knowledge Base 会话级隔离 (`rag/vector_store.py`)

```python
class VectorStoreService:
    def __init__(self):
        self.session_id = current_session_id.get()  # 从 ContextVar 读取
        safe_col_name = "col_" + self.session_id.replace("-", "_").lower()

        self.vector_store = Chroma(
            collection_name=safe_col_name,   # 每个 session 独立 Collection
            persist_directory=chroma_conf['persist_directory'],  # 公共 chroma_db/
        )
        # 物理文件目录也按 session 隔离
        self.session_data_path = os.path.join(get_abs_path("data"), self.session_id)
```

- ChromaDB 在一个 `persist_directory` 下可以共存多个 `collection_name`，每个科目对应一个 Collection
- `destroy_knowledge_base()` 方法同时执行 `collection.delete()` 和 `shutil.rmtree(session_data_path)` 实现彻底物理清除

---

### 4.3 混合高精度 RAG 检索 (`rag/rag_service.py`)

分三步执行：

**步骤一：Query Rewrite（查询改写）**
```
"详细讲一下它的定义" → LLM改写 → "TCP三次握手 定义 握手过程 状态机"
```
解决学生口语化提问对向量检索友好度极差的问题。

**步骤二：混合检索（BM25 + Vector Embedding）**
```python
EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.6, 0.4]  # BM25权重更高，确保专有名词精准命中
)
```
- **BM25（权重0.6）**：传统倒排索引，对专有名词词频绝对敏感，不会"模糊化"
- **Embedding（权重0.4）**：语义相似度兜底，处理同义词和句意推断

**步骤三：课件总结 Chain**
将检索到的 context 与用户问题一起丢给 `rag_summarize.txt` 定制提示词链，强制要求 LLM 优先引用课件原文，并标注来源。

---

### 4.4 ReAct 智能体工具调度 (`agent/tools/agent_tools.py`)

两个 LangChain Tool：

| 工具名 | 触发场景 | 内部逻辑 |
|---|---|---|
| `search_courseware` | 用户问知识点、请求解释 | 调用 `RagSummarizeService` 执行混合检索+总结 |
| `generate_quiz` | 用户要出题、练习 | 先查资料→再驱动 LLM 基于真实课件内容出3道题（含解析） |

Agent 由 `create_tool_calling_agent` + `AgentExecutor` 组装，ReAct 主提示词在 `prompts/main_prompt.txt`，约束其：
1. 必须先调用工具获取课件资料作为底料
2. 回答时必须标注"依据课件XXX章"
3. 没有课件支撑时坦白说明，禁止幻觉生成

---

### 4.5 双轨记忆引擎 (`utils/memory_service.py`)

**数据结构**：
```python
store[session_id] = {
    "name": "计算机网络期末",     # 展示名称（可重命名）
    "recent": [...],              # 最近 N 轮对话（短期滑动窗口）
    "trash": [...],               # 被挤出的旧记忆，等待提纯
    "graph": [...]                # 结构化知识图谱节点（长期记忆）
}
```

**内存淘汰策略**：
1. 每次 `add_message` 后，若 `recent` 超过 `max_recent_turns * 2 = 6` 条，最老的 2 条被移入 `trash`
2. `trash` 积累到 `trash_threshold * 2 = 8` 条时，触发后台 `_distill_to_graph`
3. 提纯链（小模型 + JsonOutputParser）将8条旧对话浓缩成若干 KnowledgeGraphNode 结构塞入 `graph`

这样 Token 用量被严格上限控制，同时长期记忆以高密度结构化形式保留。

---

### 4.6 流式 SSE 输出链路 (`api/routers/chat.py`)

```python
async for event in app_agent.astream_events(..., version="v1"):
    if event["event"] == "on_chat_model_stream":
        yield f"data: {json.dumps({'text': chunk.content})}\\n\\n"
    elif event["event"] == "on_tool_start":
        yield "data: [系统思考日志]\\n\\n"
    elif event["event"] == "on_chat_model_end":
        # 提取 usage_metadata 更新 token 计数
        update_token_stats(usage)
```

前端 `fetch` + `ReadableStream` 接收，按 `data:` 帧实时拼接并渲染 Markdown。

---

### 4.7 AOP 算力探针 (`utils/logger_handler.py`)

```python
@timer_and_token_logger         # 挂在任何工具函数上
def search_courseware(query):   # 核心业务代码保持纯净
    ...
```

装饰器内部记录执行耗时；Token 消耗通过 `on_chat_model_end` 事件钩子从 LLM 响应元数据中提取。对外暴露 `/api/chat/tokens` 接口，前端每3秒轮询并更新顶部看板。

---

## 五、API 接口速查

| 方法 | 路径 | 功能 |
|---|---|---|
| `POST` | `/api/chat/stream` | 流式对话 (SSE)，Body: `{query, session_id}` |
| `GET` | `/api/chat/sessions` | 获取所有活跃会话列表 `[{id, name}]` |
| `PUT` | `/api/chat/session/{id}` | 重命名会话，Body: `{new_name}` |
| `DELETE` | `/api/chat/session/{id}` | 彻底销毁会话（记忆 + ChromaDB集合 + 物理文件） |
| `GET` | `/api/chat/tokens` | 获取 Token 消耗与响应延迟统计 |
| `POST` | `/api/knowledge/upload?session_id=xxx` | 上传文件到指定科目知识库 |

---

## 六、关键配置项 (`config/chroma.yml`)

```yaml
collection_name: agent         # 仅用于兜底，实际按 session_id 动态生成
persist_directory: chroma_db   # ChromaDB 物理持久化目录
k: 3                           # 向量检索每次返回文档数
chunk_size: 500                # 文本切片大小（字符数）
chunk_overlap: 50              # 切片重叠区（保留上下文连接）
allow_knowledge_file_type: ["txt", "pdf", "docx"]
```

---

## 七、本地启动与开发指南

```bash
# 1. 激活项目 Python 虚拟环境
# (根据你的环境可能是 conda activate xxx)

# 2. 启动开发服务器（热重载模式）
uvicorn api.main:app --reload

# 3. 打开前端界面
# 浏览器访问：http://127.0.0.1:8000/
```

**环境变量**（需提前配置）：
```bash
DASHSCOPE_API_KEY=sk-xxxx     # 阿里云通义千问 API Key
```

---

## 八、面试核心问答

**Q：如何解决大模型幻觉问题？**
采用双重防御：① ReAct Agent 在提示词层面强制工具调用顺序（先查课件再答话）；② BM25 检索确保专有名词精准命中，杜绝向量模型"语义过泛"导致的张冠李戴。

**Q：为什么用 FastAPI 不用 Flask？**
大模型调用是纯 I/O 密集型任务（大量时间在等第三方 API 返回）。FastAPI 基于 ASGI 原生支持 `async/await` 协程，单进程可支撑超高并发。Flask 同步阻塞，一个请求挂起10秒，整个服务器就瘫了。

**Q：EnsembleRetriever 权重为什么 BM25 更高？**
大学期末复习场景中精准名词命中优先级远高于语义泛化推断。"赫兹赫伯姆图"错一个字就是另一个知识点，BM25 的倒排索引能绝对精确定位，Embedding 做语义兜底。

**Q：ContextVar 为什么比 threading.local 更合适？**
FastAPI 的协程用 `await` 切换时，不同的请求可以共用同一个线程（事件循环线程），`threading.local` 按线程隔离会发生数据串改。`ContextVar` 是按"执行上下文（Context）"隔离的，与 `asyncio` 的协程调度模型完美匹配。

**Q：如果系统要扩展到高并发，瓶颈在哪里？**
1. 内存 `dict` 形式的 `memory_manager` → 迁移到 Redis 集群
2. Local 模式 ChromaDB → 独立为网络服务节点或换用 Milvus
3. FastAPI 单进程 → Nginx + Gunicorn 多 Worker 横向扩展

---

*文档由 DeepRevision 开发助手自动生成 · 2026-03-11*
