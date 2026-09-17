# DeepRevision

DeepRevision 是一个面向高校课程复习的 Agent Demo。它把分散课件转成可检索知识库，并串起“提问、出题、作答、掌握度更新、薄弱点复习、组卷导出”的完整流程。

项目重点不是堆叠模型或框架，而是让每一步都有明确职责、失败条件和可检查结果。

## 演示主线

1. 创建课程会话并上传课件。
2. 后台解析文件，建立结构化父块与可检索子块，并把子块写入该会话独立的 Chroma collection。
3. 提问课件内容，Supervisor 路由到 RAG Agent，并返回带来源信息的答案。
4. 请求某知识点练习，Quiz Agent 生成题目并通过确定性质量门检查。
5. 提交答案，SQLite 记录练习并更新知识点 mastery。
6. 查看薄弱点，继续生成相似题或复习计划。
7. 请求整卷，Exam Agent 按题型分阶段生成，通过质量检查后导出 DOCX。

## 架构

```text
Next.js UI
    |
    | REST / SSE
    v
FastAPI
    |
    v
Supervisor
    |-- RAG Agent -------- Child BM25 + Vector -> RRF -> Rerank -> Parent
    |-- Quiz Agent ------- Generate -> Check -> Revise
    |-- Exam Agent ------- Staged generation -> Quality gate
    |-- Planner/History -- Practice and mastery data
    |-- Ops Agent -------- Tool calls + destructive-action confirmation
    `-- Chitchat

ChromaDB: session-isolated courseware vectors
SQLite: sessions, messages, practice, mastery, memory and question history
```

这是“Supervisor 编排的分层多 Agent”架构，同时混合了两类控制方式：

- 高确定性请求先走规则，例如问候、答题跟进和危险操作确认，减少 LLM 路由抖动。
- 开放语义请求由 Supervisor 输出结构化的 `route + params + reason`，再交给专业 Agent。

详细设计见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 关键设计与取舍

### 1. 为什么做会话隔离

课程课件、对话、错题和掌握度都与课程会话绑定。Chroma collection 和 SQLite 查询均带 `session_id`，避免不同课程的证据和学习记录串用。

### 2. 为什么同时用 BM25 和向量检索

- BM25 擅长课程术语、缩写和精确关键词。
- 向量检索擅长语义相近但措辞不同的问题。
- RRF 只使用两路结果的排名进行融合，不要求不同检索器的分数处于同一尺度。

RRF 后使用百炼 `qwen3.7-text-rerank` 对小候选集做语义重排；调用失败时保留 RRF 顺序继续执行。
检索和重排针对细粒度子块，最终通过 `parent_id` 回溯完整父块并去重，兼顾定位精度与生成上下文完整性。

### 3. fast / full 是什么

- `fast`：先召回 8 个子块，重排后最多回溯 4 个父块，优先降低常规问答延迟。
- `full`：召回 16 个子块，重排后最多回溯 6 个父块。
- 当 fast 命中文档少于 3 条或来源少于 2 个时，自动升级 full。
- full 后仍低召回时，才用 HyDE 生成假设性教材片段补充向量检索。

因此 fast/full 不是两个模型，而是同一检索链路的两个候选预算档位。

### 4. 为什么需要查询改写和保护

用户常说“这个怎么理解”或“随机出几题”，原句缺少检索词。轻量模型只生成检索关键词；改写保护会把原问题中的课程实体补回，避免关键词被“改飞”。模型失败时直接回退原问题。

### 5. 为什么生成后还要质量门

模型能生成通顺文本，但不保证题量、编号、选项、答案解析或总分正确。Quiz/Exam 先做确定性结构检查，只在需要时进入 Critique/Revise，限制修订轮数并记录降级原因，避免无限自我反思。

### 6. 为什么保留 SQLite

这是单机面试 Demo，数据规模和并发都有限。SQLite 能同时支撑事务、索引、WAL 和可检查的数据表，比引入独立数据库服务更容易复现。若演进到多实例部署，再迁移 PostgreSQL 和外部任务队列。

## 已删除的多余路径

- 删除未接入主流程的登录页和 Auth API。
- 删除独立 MCP 服务与根目录 Node 脚手架。
- 删除未被调用的 RAG 总结链、答案语义缓存和旧 LLM rerank 占位逻辑；当前使用可观测的百炼文本重排服务。
- 删除启动课件预热：旧实现写入的缓存没有任何消费方，只增加启动检索成本。
- 删除旧版 `.doc/.ppt` 的伪支持；当前明确支持 `.txt/.pdf/.docx/.pptx` 和常见图片。
- 删除会导致 pytest 收集时直接退出的调试文件和已提交的 `.pyc`。
- 前端移除登录动画、未用 UI 包和在线字体构建依赖。

## 技术栈

- API：FastAPI、SSE
- Agent 编排：LangGraph、LangChain Core
- 模型：DeepSeek V4、MiniMax 或 Qwen 聊天模型；DashScope/Qwen Embedding
- 检索：父子索引、ChromaDB、BM25、RRF、Qwen Rerank、低召回 HyDE
- 数据：SQLite
- Web：Next.js 16、React 19、Tailwind CSS

## 快速开始

要求：Python 3.11+、Node.js 20+。

### 1. 配置

在项目根目录创建 `.env`：

```dotenv
# 课件上传和 RAG 必填：普通百炼/业务空间 Key，不使用 Token Plan Key
DASHSCOPE_API_KEY=your_key

# 可选聊天模型凭证
TOKEN_PLAN_API_KEY=your_sk_sp_key
DEEPSEEK_API_KEY=your_key
# MINIMAX_API_KEY=your_key
# QWEN_API_KEY=your_key

# 可选：token_plan | deepseek | minimax | qwen
PRIMARY_LLM_PROVIDER=token_plan
TOKEN_PLAN_BASE_URL=https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
TOKEN_PLAN_CHAT_MODEL=qwen3.8-max
TOKEN_PLAN_LIGHT_MODEL=qwen3.8-flash
DEEPSEEK_CHAT_MODEL=deepseek-v4-pro
DEEPSEEK_LIGHT_MODEL=deepseek-v4-flash
```

Token Plan Key 仅用于文本问答；课件上传和 RAG 仍使用独立的
DashScope/百炼 Embedding Key。配置多个 Provider 时，系统按主模型配置排序，
并把下一个可用 Provider 作为故障备用。

### 2. 安装

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd react
npm install
cd ..
```

Windows 激活命令为 `venv\Scripts\activate`。

### 3. 启动

```bash
python run.py
```

- 前端：http://127.0.0.1:3000/chat
- API 文档：http://127.0.0.1:8001/docs
- 健康检查：http://127.0.0.1:8001/health

如端口已占用，可设置 `FRONTEND_PORT`、`BACKEND_PORT`；前后端分开部署时设置 `BACKEND_API_BASE` 和 `FRONTEND_URL`。

### Docker

```bash
docker build -f Dockerfile.backend -t deeprevision-backend .
docker build -f react/Dockerfile -t deeprevision-frontend react
```

生产编排示例见 [deploy/ecs/docker-compose.prod.yml](deploy/ecs/docker-compose.prod.yml)。

## 验证

```bash
pip install -r requirements-dev.txt
pytest -q

cd react
npm run build
npm audit --omit=dev
```

## 主要 API

- `POST /api/knowledge/upload`：上传并后台入库
- `GET /api/knowledge/list`：查看文件及处理状态
- `POST /api/chat/stream`：Supervisor 路由和 SSE 响应
- `POST /api/chat/practice/submit`：保存作答并更新 mastery
- `GET /api/chat/mastery`：获取优先复习点
- `POST /api/exam/export/docx`：导出试卷
- `GET /api/exam/download/{filename}`：下载导出文件

完整接口以运行时 `/docs` 为准。

## Demo 边界

- 这是单用户、单机优先的演示项目，没有认证和租户权限体系。
- 上传向量化使用 FastAPI 后台任务；进程重启后任务不会恢复。
- SQLite 和本地 Chroma 适合 Demo，不代表多实例生产方案。
- 检索和生成效果依赖课件质量与外部模型服务，仓库未提供性能、准确率或并发 SLA。
