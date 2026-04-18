# DeepRevision

DeepRevision 是一个面向高校课程复习场景的智能问答与出题系统。核心思路是“课件锚定 + 练习闭环”：先用课件构建检索知识库，再把练习记录反哺到后续出题与计划。

> 文档更新时间：2026-04-17

## 当前实现概览

- 多 Agent 路由：`supervisor` 在 `rag / quiz / exam / ops / planner / history / chitchat` 之间分流。
- 课件知识库：支持 `pdf/docx/txt/ppt/pptx/图片` 上传，单次最多 5 个文件，按 MD5 去重，后台向量化并维护状态（`processing/completed/failed`）。
- 失败文件重试：支持对失败或异常状态文件执行重试入库（`/api/knowledge/retry-failed`）。
- RAG 检索：BM25 + 向量检索 + RRF 融合，低召回场景可触发 HyDE，带会话级上下文缓存与语义缓存。
- 智能出题与组卷：基于 Reflexion（Generate -> Critique -> Revise）链路生成题目和试卷，支持按题型与数量控制。
- 出题跨轮去重：同一会话会记录每轮题干签名，默认在新一轮出题前加载最近 15 轮签名，避免生成“一模一样”的重复题。
- 练习闭环：提交作答、错因、相似题推荐、薄弱点统计、知识点回填（`/practice/backfill-kp`）。
- 出题后复盘跟进：当用户追问“怎么样/如何”时，Supervisor 会把最近出题/作答快照作为提示交给 LLM 判定，默认倾向 `history` 复盘；仅在用户明确“继续出题/再来几题”时才走 `quiz/exam`。
- 用户画像：支持用户级助手人格配置（语气、详略、教学风格、称呼等）。
- 导出能力：支持试卷格式化、试卷 Word 导出、答案卷导出与下载。
- 运行观测：提供健康检查、token 统计与运行指标接口。

## 技术栈

- 后端：FastAPI + LangGraph + LangChain
- 前端：Next.js 14（React 18 + Tailwind CSS）
- 向量库：ChromaDB（按会话隔离 collection）
- 持久化：SQLite（会话、消息、练习记录、题库）

## 目录结构

```text
DeepRevision/
├── api/
│   ├── main.py
│   └── routers/
│       ├── auth.py
│       ├── chat.py
│       ├── knowledge.py
│       └── exam_export.py
├── agent/
│   ├── multi_agent/
│   │   ├── supervisor.py
│   │   └── quiz_agent.py
│   └── tools/
├── rag/
│   ├── rag_service.py
│   └── vector_store.py
├── utils/
│   ├── memory_service.py
│   ├── user_profile_service.py
│   └── kb_version.py
├── react/
├── deploy/ecs/docker-compose.prod.yml
├── Dockerfile.backend
├── run.py
└── tests/
```

## 快速开始

### 1) 安装依赖

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
pip install -r requirements.txt

cd react
npm install
cd ..
```

### 2) 配置环境变量

在项目根目录创建 `.env`（`model/factory.py` 会自动读取）：

```bash
# 聊天模型（至少配置一个）
MINIMAX_API_KEY=xxx
# 或
QWEN_API_KEY=xxx

# 默认 embedding/视觉模型依赖 DashScope，建议配置
DASHSCOPE_API_KEY=xxx

# 可选：主模型优先级（minimax | qwen）
PRIMARY_LLM_PROVIDER=minimax
```

### 3) 启动服务

方式 A：一键启动（推荐）

```bash
python run.py
```

- 自动拉起后端（`8001`）与前端（`3000`）
- 自动打开浏览器到前端页面

方式 B：前后端分开启动

```bash
# 后端
uvicorn api.main:app --host 0.0.0.0 --port 8001

# 前端
cd react
npm run dev
```

访问地址：

- 前端：`http://127.0.0.1:3000/chat`
- 健康检查：`http://127.0.0.1:8001/health`
- 后端根路由：`http://127.0.0.1:8001/`（重定向到前端）
- 后端应用页：`http://127.0.0.1:8001/app`（重定向到前端 `/chat`）

## Docker 部署（当前仓库实现）

```bash
# 构建镜像
docker build -f Dockerfile.backend -t deeprevision-backend:latest .
docker build -f react/Dockerfile -t deeprevision-frontend:latest ./react

# 运行（使用 deploy/ecs/docker-compose.prod.yml）
cd deploy/ecs
$env:BACKEND_IMAGE="deeprevision-backend:latest"
$env:FRONTEND_IMAGE="deeprevision-frontend:latest"
docker compose -f docker-compose.prod.yml up -d
```

## 主要 API（按模块）

### 认证

- `POST /api/auth/login`
- `POST /api/auth/register`
- `GET /api/auth/check`

### 对话 / 会话 / 画像

- `POST /api/chat/stream`
- `GET /api/chat/sessions`
- `POST /api/chat/session`
- `PUT /api/chat/session/{session_id}`
- `DELETE /api/chat/session/{session_id}`
- `POST /api/chat/session/cleanup`
- `GET /api/chat/messages`
- `DELETE /api/chat/message`
- `DELETE /api/chat/messages`
- `GET /api/chat/user-profile`
- `PUT /api/chat/user-profile`
- `GET /api/chat/tokens`
- `GET /api/chat/metrics`

### 练习

- `POST /api/chat/practice/submit`
- `POST /api/chat/practice/backfill-kp`
- `POST /api/chat/practice/similar`
- `GET /api/chat/practice/stats`
- `GET /api/chat/mastery`
- `GET /api/chat/practice/history`
- `DELETE /api/chat/practice/history/item`
- `DELETE /api/chat/practice/history`

补充说明（Phase 1）：
- `GET /api/chat/mastery` 返回会话级 mastery 快照与 `priority_review_points`。
- mastery 初版计分公式：`accuracy - streak_penalty - recency_penalty`，分数区间 `[0,1]`。
- `GET /api/chat/practice/stats` 现已附带 `mastery_rows_total` 与 `priority_review_points` 字段（兼容增量）。
- `POST /api/chat/practice/submit` 响应体会附带 `mastery_rows_total` 与 `priority_review_points`，可直接用于前端“下一步复习点”提示。

### 知识库

- `POST /api/knowledge/upload`
- `GET /api/knowledge/list`
- `POST /api/knowledge/retry-failed`
- `DELETE /api/knowledge/file/{filename}`
- `POST /api/knowledge/sample/upload`
- `GET /api/knowledge/sample`
- `DELETE /api/knowledge/sample`

### 试卷导出

- `POST /api/exam/api/exam/format`
- `POST /api/exam/api/exam/export/docx`
- `POST /api/exam/api/exam/answersheet`
- `GET /api/exam/api/exam/download/{filename}`

说明：
- 当前代码中 `api/main.py` 与 `api/routers/exam_export.py` 都配置了 `/api/exam` 前缀，所以实际路由是双前缀形式。
- `export/docx` 与 `answersheet` 返回体中的 `download_url` 当前仍是 `/api/exam/download/{filename}` 字符串（与实际路由存在前缀差异），这是现状实现。

## 测试

当前仓库内已有的基础测试：

```bash
pytest tests/test_history_action_parser.py tests/test_kb_version.py tests/test_user_profile_service.py -q
```

## 说明

- 当前前端首页会直接跳转到 `/chat`（开发阶段跳过登录页）。
- 认证模块是轻量本地实现（用户信息存储在 `data/users`），未接入完整会话态鉴权。
- 本 README 只描述仓库当前实现；若后续功能变更，请同步更新本文件。
