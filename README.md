# DeepRevision — AI 期末复习引擎

> 基于 RAG + Supervisor 多 Agent + Reflexion 出题 + LangGraph 的智能复习助手

![Python](https://img.shields.io/badge/Python-3.11+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Async%20Backend-009688.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3-FFD700.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Supervisor%20%2B%20Reflexion-orange.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 一、项目定位

**DeepRevision** 是一个面向大学生的 AI 期末复习助手。学生上传自己的课件，系统在课件范围内智能问答、按需出题，同时通过长短期记忆记录学习状态。

### 核心能力

| 能力 | 描述 |
|------|------|
| 📥 多格式课件摄入 | PDF / Word / PPT / TXT / 图片，按页解析，图片内容通过多模态模型并发理解 |
| 🔍 混合检索 | Query Rewrite + BM25 + 向量检索 RRF 融合（支持可选重排） |
| 🤖 Supervisor 多 Agent | LangGraph Supervisor 分析意图 → 路由到专业 SubAgent（RAG / 出题 / 出卷 / 规划 / 历史分析 / 闲聊） |
| 📝 Reflexion 出题 | LangGraph Reflexion 架构：出题 + 推理链 → Critic 质疑推理链 → Revise 修订并交付（出卷默认最多 1 轮修订） |
| 🧠 双轨记忆 | 短期滑窗 + 后台异步图谱提纯，SQLite 本地持久化，重启不丢失 |
| 📄 试卷导出 | 生成完整试卷并导出 Word |
| 🗂️ 多科目隔离 | ContextVar + ChromaDB Collection，每科目独立知识库 |
| 📊 算力监控 | AOP 装饰器实时统计 Token 消耗与调用延迟 |

---

## 二、技术栈

| 层级 | 技术选型 |
|------|----------|
| 语言 | Python 3.11+ |
| Web 框架 | FastAPI + uvicorn（ASGI 异步） |
| LLM 框架 | LangChain 0.3 + LangGraph |
| Agent 编排 | LangGraph StateGraph（Supervisor + Reflexion） |
| 向量库 | ChromaDB（本地持久化） |
| 检索策略 | BM25 + 向量 RRF 混合检索（可选重排） |
| 大模型 | MiniMax / Qwen（可替换为任意 OpenAI 兼容模型） |
| 多模态 | MiniMax/Qwen 视觉模型（图片/PPT 图片理解） |
| 前端 | React + Tailwind CSS |
| 数据库 | SQLite（会话记忆，本地持久化） |

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
        ├── rag_agent      → RAG Service（混合检索 + 片段检索）
        ├── quiz_agent     → Reflexion 出题工作流（3 Agent）
        ├── exam_agent     → Reflexion 出卷工作流（3 Agent，支持分阶段并行）
        ├── planner_agent  → 复习计划生成
        ├── history_agent  → 历史练习分析
        └── chitchat       → 闲聊回复
              │
    ┌────────┴──────────────────────────────┐
    │           RAG Service                  │
    │  Query Rewrite (LLM)                   │
    │  → BM25 + 向量 RRF 混合检索           │
    │  → 结构化上下文拼接                    │
    └─────────────────┬────────────────────┘
                       │
              ChromaDB（per session collection）
```

### Supervisor 路由工作流

```
用户输入
   │
   ▼
Supervisor LLM（分析意图）
   │ 输出 {"route": "rag|quiz|exam|planner|history|chitchat", "params": {...}}
   ▼
add_conditional_edges（条件路由）
   ├── rag      → rag_agent       → END
   ├── quiz     → quiz_agent      → END
   ├── exam     → exam_agent     → END
   ├── planner  → planner_agent   → END
   ├── history  → history_agent  → END
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
   需要修订？（存在可执行问题：数量/编号/高危缺陷等）
   rounds >= 1？（出卷链路强制结束）
   ├── 通过或已达轮次上限 → END（输出最终题目）
   └── 需修订且未到上限 → Agent 3: Revise
               - 针对每条批评逐条修订
               - 输出 revised_quiz / revision_notes / addressed_issues
               └── 直接 END（不再回到 Critic）
```

> Reflexion 核心：推理链（Chain of Thought）显式化，Critic 质疑的是推理过程而非最终输出，Revise 必须逐条回应批评并说明修改依据。

### 出卷分阶段工作流

出卷（exam_agent）采用分阶段并发策略，将试卷按题型分为多个 stage 并行出题：

```
题型分布 plan → stage_plan（choice / fill_judge / essay 三阶段）
   │
   ▼
各 stage 并发执行（各含完整 Reflexion 链路）
   │
   ▼
stage_fast_mode 控制 Critique 路径：
   - True（默认）：格式检查通过则跳过 LLM Critique
   - False：始终启用完整 LLM Critique（含 4 个质量门）
   │
   ▼
合并各 stage 题目 → 格式化交付
```

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
长期结构化摘要记忆
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
│       ├── chat.py             # 对话 SSE（接入 Supervisor 工作流）
│       ├── knowledge.py         # 文件上传 + 样卷管理
│       └── exam_export.py      # 试卷导出 Word
│
├── agent/
│   ├── tools/
│   │   └── agent_tools.py      # 联网搜索、clear_rag_cache 等工具
│   └── multi_agent/
│       ├── supervisor.py        # Supervisor 路由 + 6 个 SubAgent 节点
│       ├── quiz_agent.py        # Reflexion 出题 + 出卷工作流
│       ├── quiz_quality.py      # Critique 质量门（fast path + 4 个质量门）
│       └── quiz_normalization.py # 题目/选项正则化
│
├── rag/
│   ├── rag_service.py          # 混合检索 + Query Rewrite
│   └── vector_store.py         # ChromaDB 封装 + 并发图片处理
│
├── model/
│   └── factory.py              # LLM / Embedding / Vision 模型工厂
│
├── utils/
│   ├── memory_service.py       # 双轨记忆（滑窗 + 异步图谱提纯）
│   ├── session_context.py      # ContextVar 会话隔离
│   ├── logger_handler.py       # AOP 装饰器 + 算力统计
│   ├── file_handler.py         # 多格式文档解析（PDF/PPT/图片/Word）
│   ├── config_handler.py        # YAML 配置读取
│   └── prompt_loader.py        # 提示词加载（懒加载 + 缓存）
│
├── config/
│   ├── chroma.yml              # 向量库 + 分块 + 检索参数
│   ├── rag.yml                 # 模型配置
│   ├── agent.yml               # Agent 行为配置
│   └── prompts.yml             # 提示词路径映射
│
├── prompts/                    # 16 个提示词模板（外置化）
│   ├── quiz_generate_structured.txt
│   ├── quiz_critique.txt
│   ├── quiz_revise.txt
│   ├── exam_generate_structured.txt
│   ├── exam_generate_single_type.txt
│   ├── exam_generate_reasoning.txt
│   ├── exam_critique.txt
│   ├── exam_revise.txt
│   ├── exam_contract.txt
│   └── ...（共16个）
│
├── react/                      # React + Tailwind CSS 前端
└── data/                       # 上传文件（按 session_id 隔离）
    └── {session_id}/
```

---

## 五、工程亮点

### 1. Supervisor 多 Agent 路由（LangGraph）

用一个 LLM 做意图分类和参数提取，输出结构化 JSON，再由 `add_conditional_edges` 分发到专业 SubAgent。Supervisor 同时注入近期对话历史，支持多轮追问的正确路由（"再出 3 道"、"换成判断题"等）。

### 2. Reflexion 出题：推理链显式化

传统质检只看最终输出对不对。Reflexion 要求出题者先写下推理链（每道题的知识点来源、答案依据、干扰项设计逻辑），Critic 逐条核实推理链 vs. 课件原文。

### 3. Session 隔离：ContextVar 上下文穿透

FastAPI async handler 与 LangGraph 节点跨越多个协程栈。用 `ContextVar` 在协程粒度绑定 session_id，从请求入口一直透传到 ChromaDB Collection 选择，无需层层传参，且天然隔离并发用户。

### 4. 混合检索 RRF 融合

BM25 擅长精确关键词匹配，向量检索擅长语义理解，两者互补。RRF 不依赖人工调权重，按名次倒数融合。RAG Service 分为 `retrieve_context`（只检索）和 `rag_summarize`（检索+总结），避免冗余 LLM 调用。

### 5. 提示词外置化

16 个提示词模板全部外置到 `prompts/` 目录，通过 `config/prompts.yml` + `prompt_loader.py` 管理，支持懒加载和缓存复用，避免 LLM 调用时重复读取文件。

### 6. SQLite 持久化 + 异步写入

会话数据（对话历史、图谱节点）持久化到本地 `sessions.db`，重启不丢失。写入通过 `run_in_executor` 放入线程池，不阻塞事件循环。正常追加走单行 `INSERT`，滑动窗口触发时才全量覆写。

### 7. 异步记忆提纯

图谱提纯是重操作（LLM 调用），通过 `asyncio.create_task` fire-and-forget 触发，不阻塞对话响应。

### 8. 分阶段出卷并发

exam_agent 将试卷按题型分为 choice / fill_judge / essay 三个 stage 并行出题，各 stage 独立执行 Reflexion 链路，合并后交付。`exam_fast_mode` 参数控制 Critique 是否启用完整 LLM 质量审查。

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
| DELETE | `/api/chat/message` | 删除单条消息（按 timestamp） |
| DELETE | `/api/chat/messages` | 清空会话消息历史 |
| GET | `/api/chat/practice/history` | 获取练习历史明细 |
| DELETE | `/api/chat/practice/history/item` | 删除单条练习历史 |
| DELETE | `/api/chat/practice/history` | 清空练习历史 |
| GET | `/api/chat/tokens` | 算力消耗统计 |

### 知识库

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/knowledge/upload` | 上传课件（最多5个，后台向量化） |
| GET | `/api/knowledge/list` | 已上传文件列表（含状态） |
| POST | `/api/knowledge/retry-failed` | 重试失败文件 |
| DELETE | `/api/knowledge/file/{filename}` | 删除文件及向量 |
| POST | `/api/knowledge/sample/upload` | 上传样卷 |
| GET | `/api/knowledge/sample` | 获取样卷格式 |
| DELETE | `/api/knowledge/sample` | 删除样卷 |

### 试卷导出

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/exam/format` | 格式化试卷 |
| POST | `/api/exam/export/docx` | 导出试卷 Word |
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

### `config/chroma.yml`

```yaml
chunk_size: 1000        # 分块大小（字符数）
chunk_overlap: 200       # 分块重叠
k: 8                    # 向量检索 top-k
retrieve_top_k: 10      # 混合检索召回数量
rerank_final_k: 8      # 最终返回数量
rrf_k: 60              # RRF 公式参数
mmr_enabled: true      # 是否启用 MMR 多样性检索
mmr_lambda: 0.5        # 0=多样性, 1=相关性
```

### `config/rag.yml`

```yaml
chat_model_name: MiniMax-M2.7        # 主对话模型
light_model_name: MiniMax-M2.7      # 轻量路由模型
embedding_model_name: text-embedding-v3
vision_model_name: qwen-vl-max        # 图片理解模型
max_tokens: 32768
```

### `config/prompts.yml`

提示词路径映射，由 `prompt_loader.py` 在首次调用时懒加载并缓存。

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
# 可选：Qwen API Key（同时配置则支持多模型）
```

> 项目根目录 `.env` 会在启动时自动读取（`model/factory.py`）。使用其他模型只需修改 `model/factory.py` 中的模型工厂，接口兼容 OpenAI 标准。

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

Supervisor prompt 注入 `{recent_history}`（最近 N 条对话格式化为"学生/助手"文本）。追问"再出 3 道"时 Supervisor 能看到上一轮是出题请求，从而正确路由到 quiz 而非 rag。

**Q: RAG 路径有几次 LLM 调用？**

`retrieve_context` 只做 Query Rewrite + 混合检索 + 上下文拼接，不再执行 LLM Rerank；RAG 问答由 SubAgent 再调用一次 LLM 生成最终回答。`rag_summarize` 保留为"检索+总结"接口。

**Q: Reflexion 与普通质检有什么区别？**

普通质检只看输出对不对。Reflexion 要求 Agent 1 先显式写出推理链，Critic 质疑的是推理链声明是否成立，Revise 必须逐条回应批评并给出修改说明。这样使 Agent 的内部推理过程变得可审计、可批评、可改进。

**Q: 怎么知道 Reflexion 是否真的有效？**

`critique_quiz_node` 在首轮（`reflection_rounds==0`）执行后将评分存入 `initial_score`。最终日志输出评分变化，可逐次积累判断 Reflexion 修订有无实质提升。

**Q: Session 隔离怎么实现的？**

`ContextVar` 在协程粒度绑定 session_id，从请求入口一直透传到 ChromaDB Collection 选择，无需层层传参，且天然隔离并发用户。

**Q: 出题/出卷结果如何渲染到前端？**

SubAgent 返回结构化对象（`kind/render_mode/payload/text`），后端通过 `api/message_protocol.py` 组装后在 `complete` 事件下发，前端据 `kind` 渲染为题卡（quiz_set）或试卷画布（exam_paper）。

**Q: exam_fast_mode 是什么？**

控制出卷链路中 Critique 的路径：开启时（默认）仅做格式检查，通过则跳过 LLM Critique；关闭时始终启用完整 LLM Critique（含 4 个质量门：总分偏差、难度梯度、长度方差、简答考点多样性）。

**Q: 如何扩展到更大规模？**

| 瓶颈 | 扩展方案 |
|------|----------|
| 内存记忆 | 迁移 Redis |
| ChromaDB | 换用 Milvus / Qdrant |
| 单进程 FastAPI | Gunicorn 多 Worker + Nginx |
| 模型费用 | 接入语义缓存 + 小模型做 Supervisor 路由 |
