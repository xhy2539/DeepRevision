# DeepRevision

DeepRevision 是一个面向高校课程复习场景的智能问答与出题系统。核心思路是“课件锚定 + 练习闭环”：先用课件构建检索知识库，再把练习记录反哺到后续出题与计划。

> 文档更新时间：2026-04-29

## 比赛版定位

DeepRevision 是一个基于课件证据和错题 mastery 的个性化期末冲刺 Agent。比赛演示主线不是普通聊天，而是完整学习闭环：`课件证据 -> 生成练习 -> 作答判分 -> mastery 更新 -> 薄弱点复习计划 -> 再练习纠偏`。

- 比赛定位：[docs/competition/PITCH.md](docs/competition/PITCH.md)
- 演示脚本：[docs/competition/DEMO_SCRIPT.md](docs/competition/DEMO_SCRIPT.md)
- 安全说明：[docs/competition/SAFETY.md](docs/competition/SAFETY.md)
- 生产测评：[output/agent_eval_20260429/production_eval/production_eval_summary.md](output/agent_eval_20260429/production_eval/production_eval_summary.md)

## 当前实现概览

- 多 Agent 路由：`supervisor` 在 `rag / quiz / exam / ops / planner / history / chitchat` 之间分流。
- 课件知识库：支持 `pdf/docx/txt/ppt/pptx/图片` 上传，单次最多 5 个文件，按 MD5 去重，后台向量化并维护状态（`processing/completed/failed`）。
- 失败文件重试：支持对失败或异常状态文件执行重试入库（`/api/knowledge/retry-failed`）。
- RAG 检索：BM25 + 向量检索 + RRF 融合，低召回场景可触发 HyDE，带会话级上下文缓存与语义缓存。
- 路由纠偏：`每个课件讲了什么` 这类“课件内容理解”问题优先走 `rag`，不再误入 `ops` 管理链路。
- 智能出题与组卷：基于 Reflexion（Generate -> Critique -> Revise）链路生成题目和试卷，支持按题型与数量控制。
- 出题跨轮去重：同一会话会记录每轮题干签名，默认在新一轮出题前加载最近 15 轮签名，避免生成“一模一样”的重复题。
- 练习闭环：提交作答、错因、相似题推荐、薄弱点统计、知识点回填（`/practice/backfill-kp`）。
- Mastery 学习画像（Phase 1）：按知识点维护 `attempt/correct/连错/最近作答时间/mastery_score`，支持优先复习点排序。
- 双轨记忆系统：`recent`（短期滑窗）+ `trash`（待提纯缓冲）+ `graph`（长期图谱），并支持父子会话继承记忆。
- 计划与复习建议接入 mastery：Planner 与练习提交响应已接入 `priority_review_points` 作为优先复习兜底。
- 出题后复盘跟进：当用户追问“怎么样/如何”时，Supervisor 会把最近出题/作答快照作为提示交给 LLM 判定，默认倾向 `history` 复盘；仅在用户明确“继续出题/再来几题”时才走 `quiz/exam`。
- Quiz 多轮判分：`我选A`、`答案是A`、`第2题选C`、`判断对错` 等跟进会进入 `quiz.answer_check`，只读取上一轮题目 payload 判分，不会重新出题。
- 上下文指代跟进：`第二个是什么意思`、`这个呢`、`上面那个` 等低信息追问会沿用最近任务路由，并显式说明理解的指代对象。
- 课件检索提速：`search_courseware` 使用限时 `retrieve_context` + 片段提炼，避免走慢总结链路导致 60s 级超时。
- 前端流式体验优化：仅当视图接近底部时自动跟随滚动；用户上滑阅读后不再强制拉回底部，流式分区默认折叠显示。
- 对话内工具直调：支持在 `/api/chat/stream` 里使用 `/tool` 显式调用已注册工具（跳过 Supervisor 路由）。
- 用户画像：支持用户级助手人格配置（语气、详略、教学风格、称呼等）。
- 导出能力：支持试卷格式化、试卷 Word 导出、答案卷导出与下载。
- 运行观测：提供健康检查、token 统计与运行指标接口。

## Agent 关键机制（实现细节）

### 1) Supervisor 路由层

- 意图路由输出结构化决策：`rewritten_query + route + params + reason`。
- 内置预检规则：问候直达 `chitchat`；“导出试卷/答案卷”优先强制 `ops`，避免误进 `exam`。
- 内置确定性跟进识别：作答/判分表达会设置 `route=quiz` 与 `route_params.action=answer_check`；上下文指代表达会优先沿用最近的 `quiz/exam/planner/history/ops` 路由。
- 支持路由失败兜底：LLM 异常时按关键词回退到 `rag/quiz/exam/ops/planner/history/chitchat`。
- 支持复用预路由：`supervisor_precomputed=true` 时跳过重复判路，降低时延。

### 2) Ops Agent（工具编排与安全）

- `ops` 走 ReAct 工具决策，覆盖会话/消息/练习/课件等管理操作。
- 对高风险操作启用二次确认口令（如删除会话、清空历史、删课件）。
- 对历史动作支持确定性解析：如“删除第N条错题/消息”可直接映射具体工具调用。
- 对会话列表、练习历史、危险清空学习记录、导出刚才试卷 Word 等高确定性请求优先直调工具或安全闸，避免 ReAct 口头宣告成功。
- 支持显式工具指令：在 `/api/chat/stream` 通过 `/tool ...` 直接调已注册工具。

### 3) Planner / History Agent（数据驱动）

- `planner` 先汇总练习画像（正确率、薄弱点、近期表现）再生成结构化复习计划。
- 计划输出有规范化与兜底逻辑（字段补全、天数/时长限制、弱点注入）。
- 用户要求基于“最近错题/薄弱点/练习记录”制定计划时，输出会显式包含 `依据：最近错题/薄弱点`；`3天` 等周期约束会保留在标题和正文中。
- `history` 支持“列出/删除/清空/分析”动作直达执行，不命中时再走 LLM 解释。

### 4) Quiz / Exam Agent（Reflexion + 质量门）

- Quiz 与 Exam 均采用 Reflexion：`Generate -> Critique -> Revise`。
- Quiz 的 `answer_check` 跟进分支只做判分和解析；找不到上一轮题目时返回“未找到上一题”，不会调用出题链路。
- 具备质量门校验：题量、题型、编号、重复、选项完整、总分平衡、术语风格等。
- 允许受控降级交付（`delivery_mode`/`degrade_reason`），并保留失败原因用于前端展示。
- Exam 支持分阶段严格生成、阶段重试、失败段单独重跑（`rerun_stage`）。

### 5) 课件预热与随机出题增强

- 启动时对活跃会话执行课件预热，构建考点池缓存，提升“随机练习”首题命中。
- 泛化请求会融合薄弱点与课件考点池，避免固定模板导致题目重复。
- 无本地样卷时可联网补充格式参考（仅作辅助，不替代课件锚点）。

### 6) 流式链路（SSE v1/v2）

- `/api/chat/stream` 支持长连接心跳、终止原因统计、TTFT/首 token 时延采样。
- 支持 `stream_v2` 灰度（按 session hash 命中），默认优先覆盖 `rag/chitchat` 路由，并兼容旧版 `delta` 事件镜像。
- `stream_v2` 可输出分区信息：`reasoning / actions / tool_calls / citations / progress`。

### 7) 练习闭环与 Mastery

- `practice/submit` 在保存记录后会同步更新 mastery，并返回：
  `mastery_rows_total + priority_review_points`。
- `mastery` 独立快照接口会返回：
  `mastery[] + priority_review_points[]`（用于复习优先级排序）。
- Planner 已接入 mastery 优先点作为弱项兜底来源；题目生成前也会结合近期练习表现做方向约束。

### 8) 知识库入库状态机

- 上传流程：校验后缀 -> MD5 去重 -> 入库队列 -> 后台向量化 -> 状态回写。
- 状态字段：`processing / completed / failed`，并带 `detail` 与 `updated_at`。
- 自愈逻辑：`list` 时会校对“状态 vs 实际向量存在性”，自动修复伪完成/伪失败。
- 失败重试：`/api/knowledge/retry-failed` 可重提失败文件；删除课件会触发知识库版本号递增（`kb_version`）。

### 9) Ops 安全闸（高风险操作）

- 危险操作（删会话/清空历史/删课件等）必须二次确认口令匹配后才执行。
- Ops ReAct 默认最多 2 步收敛；若模型“未调工具就宣告成功”，会被守卫拦截并返回明确提示。

### 10) 记忆系统（短期 + 长期 + 画像）

- 双轨结构：短期记忆存在 `messages(bucket=recent)`，长期记忆存在 `graph_nodes`，中间用 `messages(bucket=trash)` 做待提纯缓冲。
- 短期窗口：`max_recent_turns=5`（默认约 10 条消息）；超窗时按“最早 2 条”迁移到 `trash`，并限制 `trash` 最大 50 条。
- 异步提纯触发：当 `trash` 达到 `trash_threshold*2`（默认 `4*2=8` 条）时，后台触发 LLM 提纯为 `subject/relation/object` 节点。
- 提纯安全性：仅接收结构完整节点；若提纯失败会把本次 `trash` 回滚并落库，避免记忆丢失。
- 图谱容量控制：`max_graph_nodes=100`；超限时优先写入本轮新节点，必要时会替换旧节点，避免无限增长。
- 会话继承：`sessions.parent_id` 支持父子会话；`get_memory_context` 会递归合并祖先图谱（带环检测），按 `subject+relation` 去重。
- 持久化机制：SQLite 开启 WAL；启动只加载会话索引（懒加载会话详情）；支持从 `data/sessions.json` 自动迁移到 `data/sessions.db`。
- 练习耦合画像：`practice_records` 写入时同步增量更新 `knowledge_mastery`；删除/清空/回填（非 dry-run）练习后会按历史记录重建 mastery。
- 出题去重记忆：每轮题目会写入 `quiz_round_questions` 签名；生成新题前默认读取最近 15 轮签名做跨轮禁重（主链路当前保留 180 轮，底层默认值为 120 轮）。
- 题库记忆链路：相似题检索优先向量库（Chroma），不足时回退 `question_bank`，再回退 `practice_records`，避免推荐为空。

## Agent 工程化重点（面试可讲）

- 路由可解释：Supervisor 输出 `rewritten_query + route + params + reason`，不是黑盒分类。
- 预检优先级：对“问候/导出试卷/课件内容理解”走规则直达，降低误路由与延迟抖动。
- 质量门与交付策略：Quiz/Exam 内部有结构校验，支持 `delivery_mode/degrade_reason`，并可开启严格模式拒绝降级结果。
- 安全闸设计：Ops 对删除类高风险操作要求二次确认口令，防止误删数据。
- 记忆闭环：短期对话 + 长期图谱 + 练习画像（mastery）联动，计划与复盘都可消费薄弱点排序。
- 可观测性：流式链路与出题链路都沉淀指标（路由分布、成功率、TTFT、事件计数、P95），方便线上定位问题。

## 技术栈

- 后端：FastAPI + LangGraph + LangChain
- 前端：Next.js 14（React 18 + Tailwind CSS）
- 向量库：ChromaDB（按会话隔离 collection）
- 持久化：SQLite（会话、消息、图谱记忆、练习记录、mastery、出题签名、题库）

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

## SSE 事件约定（联调用）

`/api/chat/stream` 会按场景输出以下事件（并保留 `data: [DONE]` 兼容标记）：

- 基础事件：`start`、`delta`、`complete`、`error`、`done`、`heartbeat`
- `stream_v2` 扩展：`progress`、`action`、`action_result`、`citation`、`reasoning_delta`、`text_delta`
- 兼容镜像：开启 `stream_v2_legacy_delta_mirror` 时，`text_delta` 会同步镜像为 `delta`

常见语义：

- `start`：消息开始（含消息元信息）
- `delta`：正文增量
- `complete`：整条消息结构化落地
- `done`：本轮流结束状态（success/timeout/error/client_cancelled）
