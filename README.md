# DeepRevision — AI 期末复习引擎

> 基于 RAG + ReAct Agent 的智能复习助手 | 面试项目推荐

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Async%20Backend-009688.svg)
![LangChain](https://img.shields.io/badge/LangChain-Agent%20Framework-FFD700.svg)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)

---

## 一、项目定位

**DeepRevision** 是一个面向大学生的 AI 期末复习助手，基于 **RAG (检索增强生成)** + **ReAct Agent** 架构构建。

### 核心能力

| 能力 | 描述 |
|------|------|
| 📥 课件摄入 | 支持 PDF/Word/TXT/PPT/图片，自动向量化存储 |
| 🔍 混合检索 | BM25 + 向量检索 RRF 融合，提升召回率 |
| 🤖 智能问答 | ReAct Agent 自动决策：查资料 → 生成答案 |
| 📝 智能出题 | 多 Agent 协作：出题 → 验证 → 审核 → 监督 |
| 🧠 双轨记忆 | 短期滑窗 + 长期图谱，Token 精打细算 |
| 📊 算力监控 | 实时 Token 消耗与响应延迟统计 |
| 🗂️ 会话隔离 | 多科目并行，物理级知识库隔离 |

---

## 二、技术栈

### 后端架构

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI (ASGI)                        │
│                    uvicorn --reload                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌────────┐    ┌──────────┐   ┌──────────┐
│  API   │    │  Agent   │   │   RAG    │
│ Router │    │  Tools   │   │ Service  │
└────┬───┘    └────┬─────┘   └────┬─────┘
     │              │               │
     ▼              ▼               ▼
┌────────┐    ┌──────────┐   ┌──────────┐
│ Chat/   │    │ ReAct    │   │ ChromaDB │
│ Knowledge│    │ Agent    │   │ + BM25   │
└────────┘    └──────────┘   └──────────┘
```

| 层级 | 技术选型 | 面试亮点 |
|------|----------|----------|
| 语言 | Python 3.11+ | 协程 async/await |
| Web 框架 | FastAPI | ASGI 非阻塞、高并发 |
| LLM 框架 | LangChain | Agent Tool Calling |
| 向量库 | ChromaDB | 本地持久化、Session 隔离 |
| 检索 | BM25 + RRF | 混合检索降本增效 |
| 大模型 | 阿里云 Qwen (DashScope) | 可替换为 OpenAI/Anthropic |
| 前端 | Next.js 14 + React | App Router, Server Components |
| UI | Tailwind CSS | 响应式、组件化 |

---

## 三、核心架构

### 1. 系统架构图

```
                              ┌──────────────────────────────────────┐
                              │           Browser (Next.js)           │
                              │  ┌────────┐  ┌────────┐  ┌────────┐ │
                              │  │  Chat  │  │ Upload │  │ Stats  │ │
                              │  └────┬───┘  └────┬───┘  └───┬────┘ │
                              └───────┼────────────┼──────────┼───────┘
                                      │            │          │
                               SSE Stream   Form Upload  Polling
                                      │            │          │
      ┌───────────────────────────────▼────────────▼──────────▼────────┐
      │                          FastAPI Backend                         │
      │  ┌─────────────────────────────────────────────────────────────┐ │
      │  │                   API Router Layer                         │ │
      │  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │ │
      │  │  │ /chat/stream │  │ /knowledge/ │  │ /chat/session │  │ │
      │  │  │   (SSE)     │  │   /upload   │  │   (CRUD)     │  │ │
      │  │  └──────┬──────┘  └──────┬──────┘  └───────┬───────┘  │ │
      │  └─────────┼────────────────┼────────────────┼──────────┘ │
      │            │                │                │              │
      │  ┌─────────▼────────────────▼────────────────▼──────────┐  │
      │  │              ReAct Agent (LangChain)                 │  │
      │  │  ┌─────────────────────────────────────────────────┐ │  │
      │  │  │              Tool Calling Agent                 │ │  │
      │  │  │  ┌────────────┐  ┌────────────┐  ┌──────────┐│ │  │
      │  │  │  │  search_   │  │ generate_  │  │  工具    ││ │  │
      │  │  │  │  courseware │  │   quiz    │  │  调用链  ││ │  │
      │  │  │  └─────┬──────┘  └─────┬──────┘  └──────────┘│ │  │
      │  │  └────────┼────────────────┼──────────────────────┘ │  │
      │  └───────────┼────────────────┼────────────────────────┘ │
      │              │                │                          │
      │  ┌───────────▼────────────────▼───────────────────────┐  │
      │  │                   RAG Service                       │  │
      │  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  │  │
      │  │  │ Query       │  │  Hybrid     │  │ Context  │  │  │
      │  │  │ Rewrite     │  │  Retrieval  │  │ Summary  │  │  │
      │  │  │ (LLM)      │  │ (BM25+RRF) │  │          │  │  │
      │  │  └──────┬─────┘  └──────┬──────┘  └────┬─────┘  │  │
      │  └─────────┼────────────────┼────────────────┼────────┘  │
      │            │                │                │            │
      │  ┌────────▼─────┐  ┌──────▼──────┐  ┌───▼─────┐    │
      │  │  Memory      │  │  ChromaDB   │  │  LLM    │    │
      │  │  Service     │  │  Vector+BM25│  │ (Qwen)  │    │
      │  │  (双轨记忆)  │  │             │  │         │    │
      │  └──────────────┘  └─────────────┘  └─────────┘    │
      └────────────────────────────────────────────────────────┘
```

### 2. Agent 工作流程

```
User: "出一道关于进程调度的选择题"

┌─────────────────────────────────────────────────────────────┐
│                    ReAct Agent                              │
├─────────────────────────────────────────────────────────────┤
│  Step 1: Think                                          │
│    - 用户要求出题，需要调用 generate_quiz 工具             │
│                                                             │
│  Step 2: Action                                          │
│    - 调用 search_courseware("进程调度") 检索课件           │
│    - 调用 generate_quiz(topic="进程调度", type="选择题")  │
│                                                             │
│  Step 3: Observe                                         │
│    - 获取 RAG 检索结果作为上下文                          │
│    - 4 Agent 协作: 出题→验证→审核→监督                   │
│                                                             │
│  Step 4: Output                                         │
│    - 流式返回题目 + 答案 + 解析                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、目录结构

```
DeepRevision/
├── api/                          # FastAPI 应用层
│   ├── main.py                   # 入口，路由挂载
│   └── routers/
│       ├── chat.py               # 对话 API (SSE 流式)
│       ├── knowledge.py           # 知识库 API (上传/列表)
│       └── auth.py               # 认证 API
│
├── agent/                        # Agent 智能体
│   ├── tools/
│   │   └── agent_tools.py        # ReAct 工具: 搜索/出题
│   └── multi_agent/
│       └── quiz_agent.py         # 多 Agent 出题系统
│           ├── Agent 1: 出题
│           ├── Agent 2: 验证
│           ├── Agent 3: 审核
│           └── Agent 4: 监督
│
├── rag/                          # RAG 检索系统
│   ├── rag_service.py             # 混合检索 + Query Rewrite
│   └── vector_store.py           # ChromaDB 封装 + Session 隔离
│
├── model/                        # LLM 工厂
│   └── factory.py                # 统一入口，可切换模型
│
├── utils/                        # 工具模块
│   ├── memory_service.py         # 双轨记忆引擎
│   ├── session_context.py        # ContextVar 会话隔离
│   ├── logger_handler.py        # AOP 装饰器 + 算力探针
│   ├── config_handler.py         # YAML 配置读取
│   ├── file_handler.py          # 文档加载器
│   └── prompt_loader.py         # 提示词加载
│
├── config/                       # 配置文件
│   ├── chroma.yml               # 向量库参数
│   └── prompts.yml              # 提示词配置
│
├── prompts/                      # 提示词模板
│   ├── main_prompt.txt          # ReAct 主提示词
│   ├── rag_summarize.txt        # RAG 总结提示词
│   └── quiz_prompt.txt          # 出题提示词
│
├── react/                        # Next.js 前端
│   ├── app/
│   │   ├── page.tsx             # 首页 (重定向)
│   │   ├── chat/
│   │   │   └── page.tsx         # 聊天页面
│   │   └── components/
│   │       ├── ExamCanvas.tsx    # 试卷渲染组件
│   │       └── ui/               # shadcn/ui 组件
│   └── package.json
│
└── data/                         # 上传文件存储
    └── {session_id}/            # Session 隔离目录
```

---

## 五、面试核心问答

### Q1: 如何解决大模型幻觉问题？

**方案：双重防御机制**

```python
# 第一层：ReAct Agent 强制工具调用
prompt = """
你是一个复习助手。规则：
1. 必须先调用 search_courseware 工具查资料
2. 回答时必须引用课件原文
3. 没有资料时坦白说"我没有相关信息"
"""
```

```python
# 第二层：BM25 精确匹配兜底
# 专有名词不会被"语义相似"误导
ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.6, 0.4]  # 关键词优先
)
```

### Q2: 为什么用 FastAPI 而不是 Flask？

| 对比 | FastAPI | Flask |
|------|---------|-------|
| 并发模型 | ASGI 非阻塞 | WSGI 同步阻塞 |
| async/await | 原生支持 | 需配合 gevent |
| 类型校验 | Pydantic 自动 | 需手动 |
| 自动文档 | Swagger 内置 | 需 flask-restful |

```python
# FastAPI 的异步优势
@app.post("/chat/stream")
async def chat_stream(request: Request):
    # 整个链路都是 async，I/O 等待不阻塞其他请求
    async for event in agent.astream_events(input):
        yield f"data: {event}\n\n"
```

### Q3: Session 隔离如何实现？

**问题**：多用户并发时，如何保证用户 A 的文件不进入用户 B 的知识库？

**方案：ContextVar + ChromaDB Collection**

```python
# utils/session_context.py
from contextvars import ContextVar
current_session_id: ContextVar[str] = ContextVar("current_session_id", default="default")

# api/routers/chat.py
@app.post("/chat/stream")
async def chat_stream(request: Request):
    body = await request.json()
    session_id = body.get("session_id", "default")
    current_session_id.set(session_id)  # 绑定到当前协程上下文
    # 后续所有调用自动继承这个 session_id
```

```python
# rag/vector_store.py
class VectorStoreService:
    def __init__(self):
        self.session_id = current_session_id.get()  # 自动获取当前会话
        safe_col_name = f"col_{hash(self.session_id)}"
        self.vector_store = Chroma(collection_name=safe_col_name)
```

### Q4: RAG 检索效果如何优化？

**方案：混合检索 + MMR**

```python
# 1. Query Rewrite - 口语化 → 关键词
query = "详细讲一下TCP三次握手"
rewritten = llm.invoke("改写成适合检索的关键词")
# → "TCP 三次握手 握手过程 状态机"

# 2. 混合检索 RRF
# RRF 公式: score(d) = Σ 1/(k + rank(d))
rrf = RRFRetriever(bm25_retriever, vector_retriever, rrf_k=60)

# 3. MMR 多样性 (可选)
mmr = vectorstore.as_retriever(
    search_type="mmr",
    fetch_k=20,  # 获取更多候选
    lambda_mult=0.5  # 平衡相关性与多样性
)
```

### Q5: 多 Agent 出题系统如何设计？

```python
# agent/multi_agent/quiz_agent.py
class QuizState(TypedDict):
    topic: str
    quiz_type: str
    quiz: str
    verify_result: dict
    review_result: dict
    final_quiz: str

# LangGraph 工作流
builder = StateGraph(QuizState)
builder.add_node("generate", generate_quiz_node)      # Agent 1: 出题
builder.add_node("verify", verify_quiz_node)          # Agent 2: 验证答案
builder.add_node("review", review_quiz_node)            # Agent 3: 审核可行性
builder.add_node("supervise", supervise_quiz_node)    # Agent 4: 最终把关
```

### Q6: 遇到高并发瓶颈怎么办？

| 瓶颈 | 解决方案 |
|------|----------|
| 内存记忆 | 迁移到 Redis 集群 |
| ChromaDB | 换用 Milvus/Qdrant 网络版 |
| FastAPI 单进程 | Nginx + Gunicorn 多 Worker |
| Token 费用 | 接入缓存 + 模型路由 |

---

## 六、API 接口文档

### 对话 API

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/chat/stream` | SSE 流式对话 |
| GET | `/api/chat/sessions` | 获取所有会话 |
| POST | `/api/chat/session` | 创建新会话 |
| PUT | `/api/chat/session/{id}` | 重命名会话 |
| DELETE | `/api/chat/session/{id}` | 销毁会话 |
| GET | `/api/chat/tokens` | 算力统计 |

### 知识库 API

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/knowledge/upload` | 上传文件 |
| GET | `/api/knowledge/list` | 文件列表 |
| POST | `/api/knowledge/sample/upload` | 上传样卷 |
| GET | `/api/knowledge/sample` | 获取样卷 |

---

## 七、快速开始

### 1. 环境配置

```bash
# Python 3.11+
python --version

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境变量

```bash
# .env
DASHSCOPE_API_KEY=sk-your-aliyun-key
```

### 3. 启动服务

```bash
# 后端 (端口 8000)
uvicorn api.main:app --reload

# 前端 (端口 3000)
cd react && npm run dev

# 访问 http://127.0.0.1:8000
```

---

## 八、技术亮点总结

| 亮点 | 技术实现 | 面试价值 |
|------|----------|----------|
| 🧠 Agent 架构 | LangChain ReAct + Tool Calling | 熟悉 Agent 范式 |
| 🔍 混合检索 | BM25 + 向量 RRF 融合 | RAG 调优经验 |
| 📡 SSE 流式 | FastAPI + async generator | 高并发处理能力 |
| 🔄 会话隔离 | ContextVar + Collection | 异步编程理解 |
| 🎯 多 Agent | LangGraph 状态机 | 复杂系统设计 |
| 📊 算力监控 | AOP 装饰器 | 工程化意识 |
| 🗂️ 记忆系统 | 短期滑窗 + 图谱提纯 | 成本控制思维 |

---

## 九、扩展方向

- [ ] 接入 Redis 实现分布式 Session
- [ ] 换用 Milvus 支持更大规模向量
- [ ] 添加 MCP Server 对接外部工具
- [ ] 实现多模态 OCR 识别图片内容
- [ ] 添加 Web Search 作为补充知识源

---

**面试话术建议**：

> "这个项目是我从零构建的 RAG + Agent 系统，主要解决了三个工程难点：1) 如何在异步环境下保证 Session 隔离，我用 ContextVar 实现了上下文穿透；2) 如何平衡检索精度和召回率，我采用了 BM25+向量的 RRF 混合检索；3) 如何保证出题质量，我设计了 4 Agent 级联审核流程。项目中用到的技术栈都可以替换，比如换 OpenAI 或本地模型，只需要改 factory.py 一个文件。"

---

*Built with LangChain + FastAPI + ChromaDB | 面试项目首选*
