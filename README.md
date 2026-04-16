# DeepRevision

DeepRevision 是一个面向大学课程复习场景的智能出题与练习系统。用户上传课件后，系统基于课件检索进行问答、出题、组卷、练习记录与薄弱点分析。

## 项目目标

- 课件内可追溯：回答与题目尽量锚定已上传资料。
- 练习可闭环：做题结果进入历史统计，反向影响后续出题。
- 多会话隔离：每个 session 独立知识库、独立练习历史。

## 当前能力

- 课件上传与向量化：支持 PDF / Word / PPT / TXT / 图片。
- RAG 问答：Supervisor 路由到检索子代理生成回答。
- 智能出题（quiz）：支持按题型和数量生成练习题。
- 综合组卷（exam）：按题型分布生成整套试卷并可导出 Word。
- 练习历史管理：保存作答记录、错因、相似题推荐。
- 薄弱点统计：基于练习记录聚合知识点正确率。
- 随机练习策略：40% 薄弱点 + 60% 课件随机考点。

## 核心架构

- 后端：FastAPI + LangGraph + LangChain
- 前端：React + Tailwind CSS
- 向量库：ChromaDB（按 session 隔离）
- 存储：SQLite（会话消息、练习记录、题库）

主流程：

1. 前端请求 `/api/chat/stream`（SSE）。
2. Supervisor 判别意图并路由到 rag/quiz/exam/planner/history/chitchat。
3. 子代理调用检索、出题或组卷链路。
4. 结构化结果回传前端渲染（题卡/试卷画布/普通消息）。
5. 对话与练习记录落库，供后续复习策略使用。

## 出题与组卷策略（当前实现）

- Quiz：Reflexion 流程（Generate -> Critique -> 可选 Revise）。
- Exam：分阶段生成并合并交付，修订链路默认最多 1 轮。
- 泛化随机请求：基于课件预热池随机抽文件和考点，不使用固定抽象锚点词。
- 质量控制：数量、编号、重复、基础结构校验；优先可交付。

## 历史与统计

- 会话消息历史：`messages`（通过 `/api/chat/messages` 获取）。
- 练习历史面板：当前前端“历史”按钮管理的是练习记录。
- 练习统计：按知识点聚合正确率并识别 weak points。

## 目录结构

```text
DeepRevision/
├── api/
│   ├── main.py
│   └── routers/
│       ├── chat.py
│       ├── knowledge.py
│       └── exam_export.py
├── agent/
│   ├── tools/
│   └── multi_agent/
│       ├── supervisor.py
│       ├── quiz_agent.py
│       ├── quiz_quality.py
│       └── quiz_normalization.py
├── rag/
│   ├── rag_service.py
│   └── vector_store.py
├── model/
│   └── factory.py
├── utils/
│   ├── memory_service.py
│   ├── file_handler.py
│   └── session_context.py
├── react/
├── config/
└── data/
```

## 快速启动

### 1) 安装依赖

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
```

### 2) 配置环境变量

在项目根目录创建 `.env`，配置模型密钥（按你的模型工厂配置）：

```bash
MINIMAX_API_KEY=xxx
# 可选：QWEN_API_KEY=xxx
```

### 3) 启动服务

```bash
# 后端
python run.py
# 或
uvicorn api.main:app --reload --port 8001

# 前端
cd react
npm install
npm run dev
```

访问：`http://127.0.0.1:3000`

## 主要 API

### 会话与对话

- `POST /api/chat/stream`：SSE 对话入口（Supervisor 路由）
- `GET /api/chat/sessions`：会话列表
- `POST /api/chat/session`：创建会话
- `PUT /api/chat/session/{id}`：重命名会话
- `DELETE /api/chat/session/{id}`：删除会话及相关数据
- `GET /api/chat/messages`：获取会话消息历史
- `DELETE /api/chat/message`：删除单条消息
- `DELETE /api/chat/messages`：清空会话消息

### 练习与分析

- `POST /api/chat/practice/submit`：提交练习记录
- `POST /api/chat/practice/similar`：批量获取相似题
- `GET /api/chat/practice/stats`：获取练习统计与薄弱点
- `GET /api/chat/practice/history`：获取练习历史
- `DELETE /api/chat/practice/history/item`：删除单条练习历史
- `DELETE /api/chat/practice/history`：清空练习历史

### 知识库与导出

- `POST /api/knowledge/upload`：上传课件
- `GET /api/knowledge/list`：课件列表
- `DELETE /api/knowledge/file/{filename}`：删除课件
- `POST /api/knowledge/sample/upload`：上传样卷
- `GET /api/knowledge/sample`：获取样卷
- `POST /api/exam/export/docx`：导出试卷 Word
- `POST /api/exam/answersheet`：导出答案卷 Word

## 说明

README 只描述当前项目实现与运行方式。设计细节、实验对比或历史演进建议放到单独文档（如 `docs/`）维护。
