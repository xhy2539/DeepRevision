"""
Supervisor 多 Agent 路由架构

Supervisor Agent 分析用户意图，路由到对应 SubAgent：
  - rag_agent    : 知识问答、概念解释、复习某知识点
  - quiz_agent   : 出单道或少量题目（调用 Reflexion 出题工作流）
  - exam_agent   : 生成完整试卷（调用 Reflexion 出卷工作流）
- ops_agent    : 管理操作（会话/历史 CRUD，ReAct 工具决策）
- planner_agent: 制定复习计划、推荐学习顺序
- history_agent: 查看练习历史、答题记录、错题分析
- learning_loop: 自主学习闭环（画像→课件→计划→练习建议→复盘）
- chitchat     : 闲聊、感谢、无关话题

工作流：supervisor → (路由判断) → SubAgent → END
"""
import json
import time
import re
import ast
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from model.factory import chat_model, backup_chat_model, light_chat_model, backup_light_chat_model
from agent.tools.agent_tools import get_rag_service, tools as registered_tools
from agent.tools.history_action_parser import parse_history_action
from agent.multi_agent.tool_orchestrator import (
    build_tool_plan,
    execute_tool_plan,
    is_explicit_web_search_intent,
    run_react_orchestrator,
)
from utils.logger_handler import logger
from utils.session_context import current_session_id
from utils.rag_metrics import rag_inc
from utils.memory_service import memory_manager


# ==================== 状态定义 ====================

class SupervisorState(TypedDict):
    input: str              # 当前用户输入
    chat_history: List      # 历史消息（LangChain BaseMessage 列表）
    memory_context: str     # 长期图谱记忆（注入 system prompt）
    tool_context: str       # 近期工具结果/生成产物（按需注入）
    session_id: str         # 当前科目会话
    client_action: str      # 前端显式动作，如 generate_exam
    exam_stage_plan: bool   # 是否启用分段出卷
    exam_rerun_stage: str   # 失败后仅重跑某一段
    exam_partial_questions: list  # 已成功段题目（重跑失败段时回传）
    exam_fast_mode: bool   # True=快速路径(格式检查), False=完整路径(LLM Critique)
    quiz_force_llm_critic: bool  # True=quiz强制走LLM Critic，不走本地快速质检短路
    # Supervisor 决策
    route: str              # "rag" | "quiz" | "exam" | "ops" | "planner" | "history" | "learning_loop" | "chitchat"
    route_reason: str       # 路由原因（用于日志）
    route_params: dict      # 从用户输入中提取的参数
    supervisor_precomputed: bool  # 外层已完成路由判定时跳过重复 supervisor 调用
    # SubAgent 结果
    subagent_result: str    # SubAgent 原始输出
    final_answer: str       # 最终回答（透传给调用方）


def _safe_int(value, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    """将 LLM 提取参数安全转为 int，兼容空串/None/脏值。"""
    try:
        if isinstance(value, bool):
            result = int(value)
        elif value is None:
            result = default
        else:
            text = str(value).strip()
            if text == "":
                result = default
            else:
                result = int(text)
    except Exception:
        result = default

    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _extract_quiz_question_texts(quiz_result: Any) -> List[str]:
    """从 quiz payload 中提取题干列表，供跨轮去重落库。"""
    if not isinstance(quiz_result, dict):
        return []
    payload = quiz_result.get("payload")
    if not isinstance(payload, dict):
        return []
    questions = payload.get("questions")
    if not isinstance(questions, list):
        return []

    stems: List[str] = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        stem = str(item.get("question") or item.get("content") or "").strip()
        if stem:
            stems.append(stem)
    return stems


# ==================== 提示词 ====================

SUPERVISOR_PROMPT = """你是【复习助手】的智能调度中心。

## 职责
分析用户输入，判断意图，选择最合适的路由Agent。

## 路由类型（必须严格按此判断）：
- rag：知识点问答、概念讲解、解释名词、复习某个知识点 → 用"rag"
- quiz：用户明确要求"出题"、"做题"、"练习"、"给我几道题" → 用"quiz"
- exam：用户明确要求"出试卷"、"生成试卷"、"期末考试"、"给我出一份试卷" → 用"exam"
- ops：用户要执行管理/工具操作（会话与历史 CRUD、课件文件管理、诊断统计、样卷/相似题查询、试卷导出等）→ 用"ops"
- planner：用户说"复习计划"、"学习计划"、"怎么复习" → 用"planner"
- history：用户说"历史"、"错题"、"练习记录" → 用"history"
- learning_loop：用户明确要求“学习闭环/自主复习/开始复习循环/今天该复习什么并给题/根据错题安排并出题” → 用"learning_loop"
- chitchat：问候（你好/hi/hello）、感谢、闲聊、无关话题、无法分类 → 用"chitchat"

## 关键词触发规则（优先级从高到低）：
1. "导出/下载" 且包含 "试卷/答案卷/答题卡/docx/word" → ops（高优先级，覆盖 exam）
2. "试卷"、"考试" → exam
3. "出题"、"做题"、"练习题" → quiz
4. "创建会话"、"重命名会话"、"删除会话"、"清空历史记录"、"删除第N条错题/消息/记录"、"会话列表"、"系统诊断"、"删除课件"、"查看样卷" → ops
5. "学习闭环"、"自主复习"、"开始复习循环"、"今天该复习什么"且包含"题/练习"、"根据错题安排并出题" → learning_loop
6. "复习计划"、"学习计划" → planner
7. "历史"、"错题" → history
8. "解释"、"什么是"、"概念"、"知识点" → rag
9. 其他 → chitchat

补充：像“每个课件都讲了什么/这份课件主要内容是什么/总结课件要点”属于内容理解，优先走 rag，不走 ops。

## 菜单选择理解（重要）：
- 如果用户输入只是单个数字（1-9）或简单回复（如"第一个"、"第二个"）
- 且历史对话中助手刚提供了带编号的选项菜单
- 则应将用户回复理解为选择该菜单项，并按对应意图路由
- 例如：历史中助手提供了"1.复习 2.做题 3.聊天"，用户回复"2" → 应路由到 quiz

## 低信息跟进理解（重要）：
- 当用户输入是"怎么样/如何/我答得咋样"等低信息短句时，必须结合近期上下文判断，不要机械按关键词改写
- 若近期刚完成出题/作答，默认优先 route=history（复盘、正确率、薄弱点）
- 仅当用户明确说“继续出题/再来几题/重新出卷”时，才 route=quiz/exam

## 参数提取规则（仅对 quiz/exam/ops 路由）：
- quiz_params: {{"topic": "从用户输入中提取的知识点/科目", "quiz_type": "题型", "num": 数量}}
- exam_params: {{"topics": ["知识点列表"], "quiz_types": ["题型列表"], "total_questions": 总题数, "quantity_dist": {{"choice": 数量, "fill": 数量, "judge": 数量, "essay": 数量}}}}
- ops_params: {{"target":"practice|messages|session|knowledge","action":"list|create|delete|clear|rename","limit":20}}
- 如果用户没有指定具体知识点，topic/topics 设为空字符串/空列表

## rewritten_query 规则：
- 如果用户只是打招呼（你好/hi/hello），rewritten_query = 原输入不变
- 如果用户要求出题/出试卷，rewritten_query 可以精简/明确化，但**禁止添加原输入没有的内容**
- 如果用户在回应菜单选项（如"2"、"选1"、"第三个"），rewritten_query 应改写为对应的选项内容（如"做练习题"）
- 禁止把"你好"改写成"出一道题"，这是错误的重写

## 【最高优先级】用户当前输入：
{input}

## 【参考信息】历史对话：
- 如果用户在回应菜单，历史中包含选项列表，应结合理解
- 其他情况下：仅供参考，不要被历史带偏
{recent_history}

## 【参考信息】长期记忆：
{memory_context}

返回JSON（只返回JSON，不要其他内容）：
{{"rewritten_query": "...", "route": "...", "reason": "...", "params": {{}}}}"""


RAG_SUBAGENT_PROMPT = """你是知识问答专家。你必须基于检索到的课件资料回答，并提供可核验依据（仅供系统内部校验）。

【长期记忆摘要】
{memory_context}

【近期对话上下文】
{recent_history}

【检索到的课件资料】
{context}

【学生问题】
{input}

## 任务要求：

1. **禁止输出思考过程**：不要写"让我想想"、"根据我的分析"、"这个问题是..."等思考过程
2. **优先使用课件内容**：引用原文时用"根据课件："开头
3. **如果检索到相关内容**：详细解释、举例说明
4. **如果检索内容不足**：
   - 先说明"根据检索到的资料，..."
   - 再结合你的知识补充解释
   - 不要只说"没找到"
5. **如果完全没有检索结果**：
   - 友好提示："我目前没有检索到相关课件内容"
   - 可以基于常识回答，并建议用户上传相关课件
6. **如果用户要求"讲讲"、"解释一下"**：给出详细、通俗的解释，不仅罗列定义

## 输出格式（必须严格为 JSON，对象格式）：
{{
  "answer": "给用户展示的最终答案（自然语言）",
  "evidence": [
    {{"source": "参考资料1", "quote": "必须是从【检索到的课件资料】中逐字可定位的片段"}},
    {{"source": "参考资料2", "quote": "同上"}}
  ]
}}

约束：
- 必须至少提供 1 条 evidence。
- evidence.quote 必须可在【检索到的课件资料】文本中直接匹配到。
- 不要输出 JSON 之外的任何文字。"""


PLANNER_SUBAGENT_PROMPT = """你是学习规划专家，请根据学生需求输出结构化复习计划（JSON）。

【历史记忆摘要】
{memory_context}

【练习数据快照】
{practice_snapshot}

【计划约束】
{planner_constraints}

【学生需求】
{input}

输出要求：
1) 仅输出 JSON，不要解释文字。
2) 字段必须齐全：
{{
  "title": "计划标题",
  "goal": "本轮目标",
  "cycle_days": 7,
  "daily_plan": [
    {{"day": 1, "focus": "主题", "tasks": ["任务1", "任务2"], "duration_min": 60}}
  ],
  "milestones": ["里程碑1", "里程碑2"],
  "review_strategy": "复盘方式",
  "next_action": "用户现在就可以执行的一步"
}}
3) cycle_days 范围 1-30；daily_plan 的 day 从 1 开始连续。
4) 必须优先覆盖薄弱知识点，且每天都要包含“复习+练习+复盘”三类动作。
5) 任务必须可执行且简洁，不输出思考过程。"""


CHITCHAT_PROMPT = """你是期末复习助手的智能闲聊助手。

【当前会话信息】
{context}

【学生消息】
{input}

请直接回复学生，友好自然地回答。如果学生询问会话信息或课件内容，请根据上述信息回答。"""


HISTORY_SUBAGENT_PROMPT = """你是练习历史分析专家。根据用户的请求，从练习记录中提取相关信息回答用户。

【用户请求】
{input}

【练习统计数据】
{stats}

【练习历史记录】
{history}

## 回答要求（严格遵守）：

- **禁止输出思考过程**：不要写"根据你的记录"、"我来帮你分析"等思考过程
- 直接输出数据内容：如果是统计就列出数据，如果是历史就列出记录
- 如果用户询问统计/正确率/薄弱点，基于统计数据回答
- 如果用户询问历史记录，列出最近的练习
- 如果用户询问错因分析，引导用户提供具体题目
- 如果没有数据，说明暂无记录，建议先练习
- 回答要清晰、有条理

**直接输出内容，不要输出思考过程。**"""


OPS_REACT_DECISION_PROMPT = """你是复习助手的管理操作代理（ReAct）。

目标：在会话/历史管理任务中，自动决定是否调用工具并给出最终结果。

可用工具：
{tool_specs}

当前用户请求：
{input}

近期对话（可选）：
{recent_history}

已执行轨迹：
{trace}

请输出严格 JSON（不要任何额外文本）：
{{
  "done": true/false,
  "thought": "一句话说明当前判断",
  "tool_name": "当 done=false 时填写工具名，否则空字符串",
  "tool_args": {{}},
  "final_answer": "当 done=true 时填写给用户的最终回复，否则空字符串"
}}

规则：
1) 最多执行一个工具后就应尝试收敛为最终回答，避免无限循环。
2) 删除/清空类操作必须先确认对象明确（例如“第N条/ID”或“全部”）。
3) 若请求不属于管理操作，done=true 并建议用户改用问答/出题指令。
"""


# ==================== 工具函数 ====================

INCOMPLETE_PATTERNS = ['：', '...', '了"']

RETRYABLE_ERROR_PATTERNS = [
    'choices',
    '529',
    '520',
    '500',
    'server_error',
    'unknown error',
    'overloaded',
    'null value',
    'connection error',
    'timed out',
    'timeout',
    'temporarily unavailable',
    'server disconnected',
    'remote protocol error',
]


def _is_retryable_llm_error(error: Exception) -> bool:
    """判断 LLM 调用异常是否值得自动重试。"""
    if isinstance(error, TimeoutError):
        return True
    error_str = str(error).lower()
    return any(pattern in error_str for pattern in RETRYABLE_ERROR_PATTERNS)


def _build_provider_chain(primary, backup):
    """构建去重后的主备模型序列，避免同一实例重复调用。"""
    providers = []
    seen = set()
    for name, model in (("primary", primary), ("backup", backup)):
        if model is None:
            continue
        model_id = id(model)
        if model_id in seen:
            continue
        seen.add(model_id)
        providers.append((name, model))
    return providers


OPS_TOOL_EXCLUDELIST = {
    # 预留排除位；默认空表示 ops 可使用全部已注册工具。
}

OPS_DANGEROUS_TOOLS = {
    "delete_practice_record_tool",
    "clear_practice_history_tool",
    "delete_chat_message_tool",
    "clear_chat_history_tool",
    "delete_session_tool",
    "delete_knowledge_file_tool",
    "delete_duplicate_knowledge_files_tool",
}


def _tool_accepts_argument(tool_obj: Any, arg_name: str) -> bool:
    """判断工具是否声明了指定参数（兼容 pydantic v1/v2 args_schema）。"""
    try:
        schema = getattr(tool_obj, "args_schema", None)
        if schema is None:
            return False
        fields = getattr(schema, "model_fields", None)  # pydantic v2
        if isinstance(fields, dict):
            return arg_name in fields
        fields_v1 = getattr(schema, "__fields__", None)  # pydantic v1
        if isinstance(fields_v1, dict):
            return arg_name in fields_v1
    except Exception:
        return False
    return False


def _build_ops_tool_map() -> Dict[str, Any]:
    """构建 ops 子代理可用工具映射（默认覆盖全部注册工具）。"""
    tool_map: Dict[str, Any] = {}
    for tool_obj in registered_tools:
        name = str(getattr(tool_obj, "name", "") or "").strip()
        if not name or name in OPS_TOOL_EXCLUDELIST:
            continue
        tool_map[name] = tool_obj
    return tool_map


def _format_ops_tool_specs(tool_map: Dict[str, Any]) -> str:
    """将工具清单格式化为 prompt 文本。"""
    lines: List[str] = []
    for name in sorted(tool_map.keys()):
        tool_obj = tool_map[name]
        desc = str(getattr(tool_obj, "description", "") or "").strip()
        try:
            schema = getattr(tool_obj, "args_schema", None)
            if schema is not None:
                fields = getattr(schema, "model_fields", None) or getattr(schema, "__fields__", None) or {}
                args = ", ".join(str(k) for k in fields.keys()) if isinstance(fields, dict) else ""
            else:
                args = ""
        except Exception:
            args = ""
        lines.append(f"- {name}({args})：{desc}")
    return "\n".join(lines) if lines else "- （无可用工具）"


def _safe_json_dumps(value: Any) -> str:
    """安全序列化对象，避免日志或提示词构建阶段抛错。"""
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _ops_required_confirmation_phrase(tool_name: str, args: Dict[str, Any]) -> tuple[str, str]:
    """生成危险操作确认短语；返回 (required_phrase, error_message)。"""
    if tool_name == "delete_session_tool":
        sid = str(args.get("session_id", "") or "").strip()
        if not sid:
            return "", "删除会话需要显式提供 session_id。"
        return f"确认删除会话:{sid}", ""
    if tool_name == "clear_practice_history_tool":
        sid = str(args.get("session_id", "") or "").strip()
        if not sid:
            return "", "清空练习记录需要显式提供 session_id。"
        return f"确认清空练习记录:{sid}", ""
    if tool_name == "clear_chat_history_tool":
        sid = str(args.get("session_id", "") or "").strip()
        if not sid:
            return "", "清空对话记录需要显式提供 session_id。"
        return f"确认清空对话记录:{sid}", ""
    if tool_name == "delete_practice_record_tool":
        rid = str(args.get("record_id", "") or "").strip()
        if not rid.isdigit():
            return "", "删除错题需要显式提供 record_id。"
        return f"确认删除错题:{rid}", ""
    if tool_name == "delete_chat_message_tool":
        ts = str(args.get("timestamp", "") or "").strip()
        if not ts.isdigit():
            return "", "删除消息需要显式提供 timestamp。"
        return f"确认删除消息:{ts}", ""
    if tool_name == "delete_knowledge_file_tool":
        filename = str(args.get("filename", "") or "").strip()
        if not filename:
            return "", "删除课件需要显式提供 filename。"
        return f"确认删除课件:{filename}", ""
    if tool_name == "delete_duplicate_knowledge_files_tool":
        keep_filename = str(args.get("keep_filename", "") or "").strip()
        if not keep_filename:
            return "", "删除重复课件需要显式提供 keep_filename。"
        return f"确认删除重复课件:{keep_filename}", ""
    return "", ""


def _is_ops_confirmation_match(user_input: str, required_phrase: str, tool_name: str) -> bool:
    """确认口令严格匹配：避免“引用口令提问”被误判为已确认。"""
    required = str(required_phrase or "").strip().replace("：", ":").rstrip("。.!！")
    if not required:
        return False
    user_text = str(user_input or "").strip().replace("：", ":").rstrip("。.!！")
    if not user_text:
        return False
    # 删除会话允许附加“连同知识库”，其余危险工具必须严格等于确认口令。
    if tool_name == "delete_session_tool":
        return bool(re.fullmatch(rf"{re.escape(required)}(?:\s*连同知识库)?", user_text))
    return user_text == required


def _build_ops_guard_reply(
    *,
    trace: List[Dict[str, Any]],
    tool_name: str,
    pending_args: Dict[str, Any],
    required_phrase: str,
    reason: str,
) -> Dict[str, Any]:
    """构造 ops 安全闸回执（结构化 payload/meta）。"""
    text = f"{reason}\n请在安全确认卡中选择是否继续。"
    payload = {
        "ops_trace": trace,
        "ops_confirmation_required": True,
        "pending_tool": tool_name,
        "pending_args": pending_args,
        "required_confirmation_phrase": required_phrase,
    }
    meta = {
        "ops_react": True,
        "ops_confirmation_required": True,
        "danger_confirmation_required": True,
        "pending_tool": tool_name,
        "pending_args": pending_args,
        "required_confirmation_phrase": required_phrase,
    }
    return {
        "kind": "chat",
        "render_mode": "markdown",
        "text": text,
        "payload": payload,
        "meta": meta,
    }


def _resolve_ops_history_tool_call(
    *,
    action: Dict[str, Any],
    session_id: str,
    limit: int,
) -> tuple[str, Dict[str, Any], str]:
    """把 history 解析动作映射为 ops 工具调用。"""
    action_name = str(action.get("action", "none") or "none")
    safe_limit = max(1, min(int(limit or 20), 200))

    if action_name == "list_practice":
        return "get_practice_history", {"session_id": session_id, "limit": safe_limit}, ""
    if action_name == "clear_practice":
        return "clear_practice_history_tool", {"session_id": session_id}, ""
    if action_name == "list_messages":
        return "get_chat_history_tool", {"session_id": session_id, "limit": min(safe_limit, 50)}, ""
    if action_name == "clear_messages":
        return "clear_chat_history_tool", {"session_id": session_id}, ""

    if action_name == "delete_practice":
        record_id = action.get("record_id")
        if record_id is None and action.get("record_index") is not None:
            from utils.memory_service import memory_manager
            history = memory_manager.get_practice_history(session_id, 200)
            record_id = _resolve_practice_id_by_index(history, int(action.get("record_index")))
        if record_id is None:
            return "", {}, "请提供要删除的错题 ID，或说“删除第N条错题”。"
        return "delete_practice_record_tool", {"record_id": int(record_id), "session_id": session_id}, ""

    if action_name == "delete_message":
        timestamp = action.get("timestamp")
        if timestamp is None and action.get("message_index") is not None:
            from utils.memory_service import memory_manager
            messages = memory_manager.get_messages(session_id)
            _, ordered_messages = _format_message_rows(messages, limit=200)
            timestamp = _resolve_message_ts_by_index(ordered_messages, int(action.get("message_index")))
        if timestamp is None:
            return "", {}, "请提供要删除的消息 timestamp，或说“删除第N条消息”。"
        return "delete_chat_message_tool", {"timestamp": int(timestamp), "session_id": session_id}, ""

    return "", {}, ""


def _ops_query_requires_tool(query: str) -> bool:
    """判断当前 ops 文本是否属于应当调用工具的执行型请求。"""
    text = str(query or "").strip()
    if not text:
        return False
    return bool(
        re.search(r"(清空|清除|清理|删除|移除)\s*(错题|练习记录|历史记录|历史对话|聊天记录|消息|会话)", text)
        or re.search(r"(查看|列出|展示|显示)\s*(会话|历史|记录|错题|消息)", text)
        or re.search(r"(会话列表|查看会话|列出会话)", text)
        or re.search(r"(创建|新建|重命名)\s*会话", text)
        or re.search(r"删除第\s*\d+\s*条", text)
        or re.search(r"(查看|获取|查询|分析|统计|搜索)\s*(系统诊断|token|课件|课件文件|知识库|样卷|薄弱点|相似题)", text)
        or _is_courseware_management_intent(text)
        or re.search(r"(删除|移除)\s*(课件|文件)", text)
        or re.search(r"(导出|下载|保存为|另存为)\s*(试卷|答案卷|答题卡|word|docx)", text, flags=re.IGNORECASE)
    )


def _extract_courseware_filename(text: str) -> str:
    """从文本中提取一个常见课件文件名。"""
    match = re.search(
        r"([\w\u4e00-\u9fa5 .+\-&()（）]+?\.(?:pdf|docx?|pptx?|txt|png|jpe?g|webp|gif|bmp))",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return str(match.group(1) or "").strip(" ，,。；;：:")


def _infer_duplicate_keep_filename(query: str, session_id: str) -> str:
    """
    推断重复课件清理时应保留的文件。
    优先使用当前/近期对话里的“保留 xxx.pdf”，否则从唯一重复组里选非副本命名。
    """
    text = str(query or "")
    explicit = re.search(r"保留\s*([^\s，,。；;]+?\.(?:pdf|docx?|pptx?|txt|png|jpe?g|webp|gif|bmp))", text, flags=re.IGNORECASE)
    if explicit:
        return str(explicit.group(1)).strip()
    explicit_filename = _extract_courseware_filename(text)
    if explicit_filename:
        return explicit_filename

    for msg in reversed(_get_recent_session_messages(session_id, limit=8)):
        content = str(msg.get("content") or "")
        match = re.search(r"保留\s*([^\s，,。；;]+?\.(?:pdf|docx?|pptx?|txt|png|jpe?g|webp|gif|bmp))", content, flags=re.IGNORECASE)
        if match:
            return str(match.group(1)).strip()

    try:
        import os
        import hashlib
        from utils.config_handler import chroma_conf
        from utils.path_tool import get_abs_path

        session_dir = os.path.join(get_abs_path(chroma_conf["data_path"]), session_id)
        if not os.path.isdir(session_dir):
            return ""
        allowed = {".txt", ".pdf", ".docx", ".doc", ".ppt", ".pptx", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
        groups: Dict[str, List[str]] = {}
        for fname in os.listdir(session_dir):
            if fname.startswith(".") or os.path.splitext(fname)[1].lower() not in allowed:
                continue
            fpath = os.path.join(session_dir, fname)
            if not os.path.isfile(fpath):
                continue
            with open(fpath, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            groups.setdefault(md5, []).append(fname)
        duplicate_groups = [sorted(names) for names in groups.values() if len(names) > 1]
        if len(duplicate_groups) != 1:
            return ""
        names = duplicate_groups[0]

        def score(name: str) -> tuple[int, int, str]:
            lower = name.lower()
            duplicate_markers = ("-dup", "-quad", "-trip", "copy", "副本", "重复")
            marker_penalty = 1 if any(marker in lower for marker in duplicate_markers) else 0
            return (marker_penalty, len(name), name)

        return sorted(names, key=score)[0]
    except Exception:
        return ""


def _is_duplicate_delete_intent(query: str) -> bool:
    """判断是否是删除重复课件/其余重复文件的请求。"""
    text = str(query or "").strip()
    if not text:
        return False
    return bool(
        re.search(r"(删除|移除|清理).*(其余|剩余|重复).*(课件|文件|资料)", text)
        or re.search(r"(删除|移除|清理).*(课件|文件|资料).*(重复|副本)", text)
        or re.fullmatch(r"确认删除重复课件:.+", text)
    )


def _is_knowledge_delete_confirmation(query: str) -> bool:
    """判断是否是课件删除确认短语，避免被课件列表预检误拦截。"""
    text = str(query or "").strip().replace("：", ":")
    return bool(re.fullmatch(r"确认删除(?:重复)?课件:.+", text))


def _resolve_ops_deterministic_tool(query: str, session_id: str, limit: int) -> tuple[str, Dict[str, Any]]:
    """识别无需 ReAct 的高确定性 ops 请求。"""
    text = str(query or "").strip()
    safe_limit = max(1, min(int(limit or 20), 200))
    if is_explicit_web_search_intent(text):
        plan = build_tool_plan(text, session_id=session_id, route_hint="ops")
        calls = plan.get("tool_calls") if isinstance(plan.get("tool_calls"), list) else []
        if calls:
            first_call = calls[0] if isinstance(calls[0], dict) else {}
            return str(first_call.get("tool_name") or ""), first_call.get("args") if isinstance(first_call.get("args"), dict) else {}
    if re.search(r"(会话列表|查看会话|列出会话|显示会话|展示会话)", text):
        return "list_sessions_tool", {}
    if _is_duplicate_delete_intent(text):
        keep_filename = _infer_duplicate_keep_filename(text, session_id)
        if keep_filename:
            return "delete_duplicate_knowledge_files_tool", {"keep_filename": keep_filename, "session_id": session_id}
    if re.fullmatch(r"确认删除课件:.+", text.replace("：", ":")):
        filename = text.replace("：", ":").split(":", 1)[1].strip()
        if filename:
            return "delete_knowledge_file_tool", {"filename": filename, "session_id": session_id}
    if _is_courseware_management_intent(text):
        return "list_knowledge_files_tool", {"session_id": session_id}
    if re.search(r"(查看|列出|显示|展示|最近\d{0,3}条)\s*(练习历史|练习记录|练习|错题|做题记录)", text):
        return "get_practice_history", {"session_id": session_id, "limit": safe_limit}
    if re.search(r"(立刻|马上|现在)?\s*(删除|清空|清除|清理).*(所有)?(学习记录|练习记录|错题)", text):
        return "clear_practice_history_tool", {"session_id": session_id}
    return "", {}


def _get_recent_exam_or_quiz_export_content(session_id: str) -> str:
    """从最近 exam/quiz 消息提取可导出的试卷文本。"""
    for msg in reversed(_get_recent_session_messages(session_id, limit=20)):
        if str(msg.get("role", "")).lower() not in {"ai", "assistant"}:
            continue
        kind = str(msg.get("kind") or "").strip().lower()
        meta = msg.get("meta") if isinstance(msg.get("meta"), dict) else {}
        route = str(meta.get("route") or "").strip().lower()
        if kind in {"exam_paper", "quiz_set"} or route in {"exam", "quiz"}:
            content = str(msg.get("content") or "").strip()
            if content:
                return content
    return ""


def _is_incomplete_response(text: str) -> bool:
    """检测响应是否不完整（LLM 想说但被截断）"""
    if not text:
        return True
    text = text.strip()
    if not text:
        return True
    # 以冒号/省略号/"了"结尾（未说完）
    if any(text.endswith(p) for p in INCOMPLETE_PATTERNS):
        return True
    # 以"了"开头的短句（思考内容混入）
    if text.startswith('了') and len(text) < 50 and '。' not in text:
        return True
    # 回答过短且以逗号/顿号结尾（被截断）
    if len(text) < 100 and any(text.endswith(p) for p in '，、'):
        return True
    # 纯思考内容被截断留下的碎片
    if len(text) < 30 and ('，' not in text and '。' not in text):
        if text.startswith('我') or text.startswith('让'):
            return True
    return False


def _sanitize_llm_text(text: str) -> str:
    """清理模型输出中的思考标签与噪音，避免泄露到用户侧。"""
    if text is None:
        return ""
    cleaned = str(text).strip()
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    return cleaned.strip()


async def _call_llm(prompt_template: str, **kwargs) -> str:
    """使用主模型（用于内容生成：rag/quiz/exam/planner），带重试"""
    import asyncio
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate.from_template(prompt_template)
    llm_timeout = 35.0
    max_retries = 1
    providers = _build_provider_chain(chat_model, backup_chat_model)
    if not providers:
        raise RuntimeError("未配置可用的主模型")

    last_error = None
    for provider_name, provider_model in providers:
        chain = prompt | provider_model | StrOutputParser()
        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.wait_for(chain.ainvoke(kwargs), timeout=llm_timeout)
                logger.info(f"[_call_llm] provider={provider_name}, 返回内容(第{attempt+1}次): {result}")
                result = _sanitize_llm_text(result)
                if _is_incomplete_response(result):
                    logger.warning(f"[_call_llm] provider={provider_name} 响应不完整: {str(result)[:100]}...")
                    if attempt < max_retries:
                        await asyncio.sleep(0.8)
                        continue
                return result
            except Exception as e:
                last_error = e
                is_api_error = _is_retryable_llm_error(e)
                logger.warning(f"[_call_llm] provider={provider_name} 调用失败（第{attempt+1}次）: {e}")
                if attempt >= max_retries or not is_api_error:
                    break
                await asyncio.sleep(1.2)
        if not _is_retryable_llm_error(last_error or Exception("unknown")):
            break
        if provider_name == "primary" and backup_chat_model:
            logger.warning("[_call_llm] 主模型异常，切换备用模型重试")
    if last_error:
        raise last_error
    return ""


async def _call_llm_light(prompt_template: str, **kwargs) -> str:
    """使用轻量模型（用于意图识别、简单分类），带超时保护"""
    import asyncio
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate.from_template(prompt_template)
    providers = _build_provider_chain(light_chat_model, backup_light_chat_model)
    if not providers:
        raise RuntimeError("未配置可用的轻量模型")

    last_error = None
    for provider_name, provider_model in providers:
        chain = prompt | provider_model | StrOutputParser()
        try:
            return await asyncio.wait_for(chain.ainvoke(kwargs), timeout=8.0)
        except Exception as e:
            last_error = e
            logger.warning(f"[Supervisor] 轻量模型调用失败 provider={provider_name}: {e}")
            if not _is_retryable_llm_error(e):
                break
    if last_error:
        raise last_error


async def _call_llm_for_supervisor(prompt_template: str, **kwargs) -> str:
    """使用主模型进行意图识别（更好的理解能力），带重试"""
    import asyncio
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate.from_template(prompt_template)
    llm_timeout = 18.0
    max_retries = 1
    providers = _build_provider_chain(chat_model, backup_chat_model)
    if not providers:
        raise RuntimeError("未配置可用的主模型")

    last_error = None
    for provider_name, provider_model in providers:
        chain = prompt | provider_model | StrOutputParser()
        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.wait_for(chain.ainvoke(kwargs), timeout=llm_timeout)
                logger.info(f"[_call_llm_for_supervisor] provider={provider_name} 返回内容(第{attempt+1}次): {result}")
                result = _sanitize_llm_text(result)
                if _is_incomplete_response(result):
                    logger.warning(f"[_call_llm_for_supervisor] provider={provider_name} 响应不完整: {str(result)[:100]}...")
                    if attempt < max_retries:
                        await asyncio.sleep(0.8)
                        continue
                return result
            except (asyncio.TimeoutError, Exception) as e:
                last_error = e
                is_api_error = _is_retryable_llm_error(e)
                logger.warning(f"[Supervisor] provider={provider_name} 调用失败（第{attempt+1}次）: {e}")
                if attempt >= max_retries or not is_api_error:
                    break
                await asyncio.sleep(1.2)
        if not _is_retryable_llm_error(last_error or Exception("unknown")):
            break
        if provider_name == "primary" and backup_chat_model:
            logger.warning("[Supervisor] 主模型异常，切换备用模型继续意图识别")
    if last_error:
        raise last_error


def _extract_json(text: str) -> dict:
    import re
    if not text or not text.strip():
        return {}
    text = text.strip()

    # 去除 <think>...</think> 思考标签内容
    text = re.sub(r'<think>[\s\S]*?<\/think>', '', text, flags=re.DOTALL).strip()
    if text.startswith('result=') or '<think>' in text:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            text = match.group(0)

    # 如果去除后为空，返回空
    if not text:
        return {}

    # 先按原文直接解析，避免 evidence 内部代码块干扰。
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 仅当整段文本是 fenced block 时才去掉外层围栏，避免误匹配内部 ```bash / ```text。
    fenced_match = re.match(r'^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$', text)
    if fenced_match:
        fenced_body = fenced_match.group(1).strip()
        try:
            return json.loads(fenced_body)
        except json.JSONDecodeError:
            pass

    # 最后回退：提取首个 JSON 对象片段再解析。
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _extract_grounded_evidence(parsed: dict, context: str) -> list[dict]:
    """
    从结构化结果中提取并校验证据：
    - 证据格式必须包含 source + quote
    - quote 必须能在检索 context 中直接命中
    """
    if not isinstance(parsed, dict):
        return []
    evidence = parsed.get("evidence", [])
    if not isinstance(evidence, list):
        return []

    context_text = str(context or "")
    context_norm = re.sub(r'[\s\W_]+', '', context_text).lower()

    def _is_grounded(quote: str) -> bool:
        quote = str(quote or "").strip()
        if not quote:
            return False
        if quote in context_text:
            return True
        # 宽松匹配：忽略空白和标点，降低“引用轻微改写”导致的误杀
        quote_norm = re.sub(r'[\s\W_]+', '', quote).lower()
        if len(quote_norm) < 8:
            return False
        return quote_norm in context_norm

    grounded: list[dict] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        quote = str(item.get("quote", "")).strip()
        if not source or not quote:
            continue
        if _is_grounded(quote):
            page = item.get("page")
            card = {"source": source, "quote": quote}
            if page not in (None, ""):
                card["page"] = page
            grounded.append(card)
    return grounded


def _build_rag_structured_result(answer: str, evidence: list[dict], reason: str = "") -> dict:
    """把 RAG 输出包装为前端可展示的证据卡协议。"""
    evidence_cards = []
    for item in evidence or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        quote = str(item.get("quote", "")).strip()
        if not source and not quote:
            continue
        card = {
            "source": source,
            "quote": quote,
            "grounded": True,
        }
        if item.get("page") not in (None, ""):
            card["page"] = item.get("page")
        evidence_cards.append(card)

    evidence_status = "grounded" if evidence_cards else "none"
    if reason == "evidence_partial" and not evidence_cards:
        evidence_status = "partial"

    payload = {"evidence_cards": evidence_cards}
    meta = {
        "evidence_status": evidence_status,
        "evidence_count": len(evidence_cards),
    }
    if evidence_cards:
        meta["evidence_source"] = f"课件证据 {len(evidence_cards)} 条"
    return {
        "kind": "chat",
        "render_mode": "markdown",
        "text": answer,
        "payload": payload,
        "meta": meta,
        "answer": answer,
        "evidence": evidence,
        **({"reason": reason} if reason else {}),
    }


def _validate_supervisor_decision(
    decision: dict,
    original_query: str,
    *,
    has_recent_quiz_context: bool = False,
) -> dict:
    """校验 Supervisor 的结构化路由结果，避免静默误路由。"""
    if not isinstance(decision, dict) or not decision:
        raise ValueError("empty supervisor decision")

    valid_routes = {'rag', 'quiz', 'exam', 'ops', 'planner', 'history', 'learning_loop', 'chitchat'}
    route = str(decision.get('route', '')).strip().lower()
    if route not in valid_routes:
        raise ValueError(f"invalid supervisor route: {route or 'empty'}")

    params = decision.get('params', {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("supervisor params must be a dict")
    # 统一参数语义：空字符串归一，避免后续 int('') 等异常
    if route == "quiz":
        normalized_params = {
            "topic": str(params.get("topic", "") or "").strip(),
            "quiz_type": str(params.get("quiz_type", "") or "选择题").strip() or "选择题",
            "num": _safe_int(params.get("num", 5), default=5, minimum=1, maximum=20),
        }
        # 保留确定性跟进参数，避免 answer_check 被标准化过程吞掉。
        for key in ("action", "answer", "question_number", "deterministic_followup", "quiz_action"):
            if key in params:
                normalized_params[key] = params.get(key)
        params = normalized_params
    elif route == "exam":
        raw_types = params.get("quiz_types", [])
        quiz_types = raw_types if isinstance(raw_types, list) else []
        type_map = {"choice": "选择题", "fill": "填空题", "judge": "判断题", "essay": "简答题"}
        quiz_types = [type_map.get(str(t).strip().lower(), str(t).strip()) for t in quiz_types if str(t).strip()]
        params = {
            "topics": [str(t).strip() for t in (params.get("topics", []) if isinstance(params.get("topics", []), list) else []) if str(t).strip()],
            "quiz_types": quiz_types or ["选择题", "填空题", "判断题", "简答题"],
            "total_questions": _safe_int(params.get("total_questions", 23), default=23, minimum=1, maximum=60),
            "quantity_dist": params.get("quantity_dist", {}) if isinstance(params.get("quantity_dist", {}), dict) else {},
        }
    elif route == "ops":
        params = {
            "target": str(params.get("target", "") or "").strip().lower(),
            "action": str(params.get("action", "") or "").strip().lower(),
            "limit": _safe_int(params.get("limit", 20), default=20, minimum=1, maximum=200),
        }

    rewritten_query = _validate_rewritten_query(
        original_query,
        str(decision.get('rewritten_query', '') or '')
    )
    reason = str(decision.get('reason', '') or '').strip()

    # 纠偏：课件内容理解请求不应落在 ops；优先改走 rag。
    if route == "ops" and _is_courseware_content_intent(original_query):
        route = "rag"
        params = {"topic": original_query}
        reason = "纠偏：课件内容理解请求改走rag"

    # 语义漂移保护：低信息短句优先保持原语义，不让改写硬注入“出题/出卷”触发词。
    if route in {"quiz", "exam"} and not _is_menu_selection_reply(original_query):
        original_has_trigger = _contains_route_trigger(original_query, route)
        rewritten_has_trigger = _contains_route_trigger(rewritten_query, route)
        if (not original_has_trigger) and rewritten_has_trigger:
            logger.warning(
                f"[Supervisor] 语义漂移保护触发: original='{original_query}' rewritten='{rewritten_query}' route={route}"
            )
            rewritten_query = original_query
            if _is_low_info_followup(original_query):
                if has_recent_quiz_context and _is_quiz_review_followup(original_query):
                    route = "history"
                    reason = "低信息跟进，结合近期练习语境转为复盘分析"
                    params = {"limit": 120}
                else:
                    reason = reason or "低信息短句，已阻断语义漂移改写"

    return {
        "rewritten_query": rewritten_query,
        "route": route,
        "reason": reason,
        "params": params,
    }


# ==================== 辅助函数 ====================

def _format_history_with_limit(chat_history: list, max_chars: int = 800) -> str:
    """按句子边界截断历史，保留语义完整性"""
    if not chat_history:
        return "（无近期对话）"

    lines = []
    total_chars = 0

    for msg in reversed(chat_history):
        role = "学生" if isinstance(msg, HumanMessage) else "助手"
        content = msg.content if isinstance(msg, (HumanMessage, str)) else getattr(msg, 'content', str(msg))
        msg_text = f"{role}: {content}"

        # 如果加上这条就超限，先试试按句子截断
        if total_chars + len(msg_text) > max_chars:
            # 尝试保留开头部分（通常是更重要的上下文）
            available = max_chars - total_chars - len(f"{role}: ")
            if available > 50:  # 至少保留50字
                truncated = content[:available] + "..."
                lines.insert(0, f"{role}: {truncated}")
                total_chars = max_chars  # 已满
                break
        else:
            lines.insert(0, msg_text)
            total_chars += len(msg_text)

        if total_chars >= max_chars:
            break

    return "\n".join(lines) if lines else "（无近期对话）"


def _validate_rewritten_query(original: str, rewritten: str) -> str:
    """验证重写结果，异常时回退到原query"""
    if not rewritten:
        logger.warning("[Supervisor] 重写结果为空，回退到原query")
        return original

    # 如果重写结果异常短或像乱码
    if len(rewritten) < 2:
        logger.warning(f"[Supervisor] 重写结果过短 '{rewritten}'，回退到原query")
        return original

    # 如果是明显的 JSON 解析错误
    if rewritten in ['{}', '[]', '{"rewritten_query":}', '{"route":}']:
        logger.warning(f"[Supervisor] 重写结果格式异常 '{rewritten}'，回退到原query")
        return original

    # 如果重写结果只是空白字符
    if not rewritten.strip():
        logger.warning("[Supervisor] 重写结果为空字符串，回退到原query")
        return original

    # 语义校验：打招呼类输入不应该被改成完全不同语义的内容
    greeting_keywords = ['你好', 'hi', 'hello', '嗨', '您好']
    original_lower = original.lower().strip()
    rewritten_lower = rewritten.lower()
    if original_lower in greeting_keywords or original_lower.startswith('你好，'):
        # 如果原输入是打招呼，但重写结果包含"出题"等意图词，回退
        if any(kw in rewritten_lower for kw in ['出题', '做题', '练习', '考试', '试卷', '复习']):
            logger.warning(f"[Supervisor] 检测到语义异常：打招呼被改写成'{rewritten}'，回退到原query")
            return original

    return rewritten


def _is_menu_selection_reply(text: str) -> bool:
    """判断是否是菜单选择式回复（如 1/选2/第二个）。"""
    q = str(text or "").strip()
    if not q:
        return False
    if re.fullmatch(r"\d{1,2}", q):
        return True
    if re.fullmatch(r"(选|第)?\s*\d{1,2}\s*(个|项|题|道)?", q):
        return True
    if re.fullmatch(r"第[一二三四五六七八九十两]+(个|项|题|道)?", q):
        return True
    return False


def _contains_route_trigger(text: str, route: str) -> bool:
    """判断文本是否包含某路由的显式触发词。"""
    q = str(text or "")
    if route == "quiz":
        return any(k in q for k in ["出题", "做题", "练习", "刷题", "练习题"])
    if route == "exam":
        return any(k in q for k in ["试卷", "考试", "出卷", "期末"])
    return False


def _is_low_info_followup(text: str) -> bool:
    """判断是否为低信息短句（容易被模型过度改写）。"""
    q = re.sub(r"\s+", "", str(text or "").strip().lower())
    if not q:
        return True
    low_info_set = {
        "怎么样", "咋样", "如何", "然后呢", "还有吗", "继续", "继续吧", "行吗", "可以吗",
        "行", "可以", "好", "好的", "嗯", "哦", "ok", "okay",
    }
    return q in low_info_set


def _is_quiz_review_followup(text: str) -> bool:
    """判断是否是“评价本轮作答表现”的跟进问句。"""
    q = re.sub(r"\s+", "", str(text or "").strip().lower())
    if not q:
        return False
    exact_matches = {
        "怎么样",
        "咋样",
        "如何",
        "我做得怎么样",
        "我答得怎么样",
        "我答得如何",
        "做得怎么样",
        "答得怎么样",
        "表现如何",
    }
    if q in exact_matches:
        return True
    return (
        any(token in q for token in ["评价", "评估", "复盘", "分析", "总结"])
        and any(token in q for token in ["答题", "作答", "练习", "这次", "刚才", "本轮"])
    )


def _get_recent_session_messages(session_id: str, limit: int = 12) -> List[Dict[str, Any]]:
    """读取会话最近消息，用于路由前的轻量上下文判定。"""
    sid = str(session_id or "").strip() or "default"
    try:
        from utils.memory_service import memory_manager

        messages = memory_manager.get_messages(sid)
    except Exception as e:
        logger.warning(f"[Supervisor] 读取会话消息失败 sid={sid}: {e}")
        return []
    if not messages:
        return []
    safe_limit = max(1, min(int(limit or 12), 40))
    return list(messages[-safe_limit:])


def _has_recent_quiz_turn(session_id: str) -> bool:
    """判断最近一轮助手回复是否来自 quiz 路由。"""
    recent_messages = _get_recent_session_messages(session_id, limit=12)
    for msg in reversed(recent_messages):
        if str(msg.get("role", "")).lower() not in {"ai", "assistant"}:
            continue
        kind = str(msg.get("kind") or "").strip().lower()
        meta = msg.get("meta") if isinstance(msg.get("meta"), dict) else {}
        route = str(meta.get("route") or "").strip().lower()
        return route == "quiz" or kind == "quiz_set"
    return False


def _parse_cn_ordinal(text: str) -> Optional[int]:
    """解析常见中文序号，供跟进问题定位题号/条目。"""
    raw = str(text or "").strip()
    if not raw:
        return None
    digit_match = re.search(r"\d{1,2}", raw)
    if digit_match:
        return _safe_int(digit_match.group(0), default=0, minimum=0, maximum=99) or None
    cn_map = {
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }
    if raw in cn_map:
        return cn_map[raw]
    if raw.startswith("十") and len(raw) == 2 and raw[1] in cn_map:
        return 10 + cn_map[raw[1]]
    if raw.endswith("十") and len(raw) == 2 and raw[0] in cn_map:
        return cn_map[raw[0]] * 10
    if len(raw) == 3 and raw[1] == "十" and raw[0] in cn_map and raw[2] in cn_map:
        return cn_map[raw[0]] * 10 + cn_map[raw[2]]
    return None


def _parse_quiz_answer_attempt(text: str) -> Dict[str, Any]:
    """识别用户是否在回答上一轮 quiz，而不是请求重新出题。"""
    q = re.sub(r"\s+", "", str(text or "").strip())
    if not q:
        return {"is_answer_check": False}

    answer = ""
    question_number: Optional[int] = None
    number_match = re.search(r"第(\d{1,2}|[一二两三四五六七八九十]{1,3})(?:题|个|道)", q)
    if number_match:
        question_number = _parse_cn_ordinal(number_match.group(1))

    answer_patterns = [
        r"(?:我选|选|答案是|答案为|选项是|选择|第(?:\d{1,2}|[一二两三四五六七八九十]{1,3})题选)([A-DＡ-Ｄ])",
        r"^([A-DＡ-Ｄ])$",
        r"^([A-DＡ-Ｄ])[。.!！]?$",
    ]
    for pattern in answer_patterns:
        m = re.search(pattern, q, flags=re.IGNORECASE)
        if m:
            answer = m.group(1).upper()
            answer = answer.translate(str.maketrans("ＡＢＣＤ", "ABCD"))
            break

    wants_check = bool(re.search(r"(判断对错|判分|批改|对不对|是否正确|解释原因|解析一下|请解释|为什么)", q))
    explicit_answer = bool(answer)
    return {
        "is_answer_check": explicit_answer or wants_check,
        "question_number": question_number,
        "answer": answer,
        "wants_check": wants_check,
    }


def _is_quiz_generation_followup(text: str) -> bool:
    """识别明确继续出题请求，避免被 answer_check 分支拦截。"""
    q = re.sub(r"\s+", "", str(text or "").strip())
    if not q:
        return False
    return bool(
        re.search(r"(再来|继续|重新|换个题型|同类型|出)\d{0,2}(道|个|题)?", q)
        and any(token in q for token in ["题", "出题", "练习", "同类型", "换个题型"])
    )


def _get_recent_quiz_payload(session_id: str) -> Optional[Dict[str, Any]]:
    """读取最近 quiz_set payload，用于确定性判分。"""
    for msg in reversed(_get_recent_session_messages(session_id, limit=20)):
        if str(msg.get("role", "")).lower() not in {"ai", "assistant"}:
            continue
        payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else {}
        questions = payload.get("questions") if isinstance(payload.get("questions"), list) else []
        meta = msg.get("meta") if isinstance(msg.get("meta"), dict) else {}
        kind = str(msg.get("kind") or "").strip().lower()
        route = str(meta.get("route") or "").strip().lower()
        if questions and (kind == "quiz_set" or route == "quiz"):
            return payload
    return None


def _normalize_answer_letters(value: Any) -> str:
    """归一化选择题答案字母，兼容“答案：A、C”等写法。"""
    text = str(value or "").upper().translate(str.maketrans("ＡＢＣＤ", "ABCD"))
    letters = re.findall(r"[A-D]", text)
    return "".join(dict.fromkeys(letters))


def _build_quiz_answer_check_result(session_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """基于最近 quiz payload 判分；缺上下文时只提示补充题目，不重新出题。"""
    payload = _get_recent_quiz_payload(session_id)
    questions = payload.get("questions") if isinstance(payload, dict) and isinstance(payload.get("questions"), list) else []
    if not questions:
        text = "未找到上一题，请贴出题目或选项后我再帮你判断对错；本次不会重新出题。"
        return {
            "kind": "chat",
            "render_mode": "markdown",
            "text": text,
            "payload": {},
            "meta": {"deterministic_followup": True, "quiz_action": "answer_check"},
        }

    qn = params.get("question_number")
    question_number = _safe_int(qn, default=1, minimum=1, maximum=len(questions)) if qn else 1
    question = questions[question_number - 1] if 0 < question_number <= len(questions) else questions[0]
    user_answer = _normalize_answer_letters(params.get("answer"))
    correct_answer = _normalize_answer_letters(question.get("answer") or question.get("correct_answer"))
    stem = str(question.get("question") or question.get("content") or "").strip()
    analysis = str(question.get("analysis") or question.get("explanation") or "暂无解析。").strip()

    if not user_answer:
        verdict = "无法判断：我还没识别到你的选项。"
    elif correct_answer and user_answer == correct_answer:
        verdict = "正确。"
    elif correct_answer:
        verdict = "错误。"
    else:
        verdict = "无法判断：上一题 payload 中没有标准答案。"

    lines = [
        f"我理解你在回答第 {question_number} 题。",
        f"- 你的答案：{user_answer or '未识别'}",
        f"- 正确答案：{correct_answer or '未提供'}",
        f"- 判定：{verdict}",
    ]
    if stem:
        lines.append(f"- 题目：{stem}")
    lines.append(f"- 解析：{analysis}")
    return {
        "kind": "chat",
        "render_mode": "markdown",
        "text": "\n".join(lines),
        "payload": {
            "answer_check": {
                "question_number": question_number,
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "is_correct": bool(user_answer and correct_answer and user_answer == correct_answer),
            }
        },
        "meta": {"deterministic_followup": True, "quiz_action": "answer_check"},
    }


def _is_context_reference_followup(text: str) -> bool:
    """识别“第二个/这个/上面那个”等依赖上一轮内容的低信息指代。"""
    q = re.sub(r"\s+", "", str(text or "").strip())
    if not q:
        return False
    patterns = [
        r"第(\d{1,2}|[一二两三四五六七八九十]{1,3})(个|点|条|题).*(什么意思|怎么理解|解释|继续讲)",
        r"(这个|那个|上面那个|刚才那个|它)(呢|是什么意思|怎么理解|解释一下)?",
        r"继续讲第(\d{1,2}|[一二两三四五六七八九十]{1,3})(个|点|条|题)",
    ]
    return any(re.search(pattern, q) for pattern in patterns)


def _get_recent_assistant_route(session_id: str) -> Dict[str, Any]:
    """取最近助手消息的 route/kind/content，供上下文指代沿用任务路由。"""
    for msg in reversed(_get_recent_session_messages(session_id, limit=20)):
        if str(msg.get("role", "")).lower() not in {"ai", "assistant"}:
            continue
        meta = msg.get("meta") if isinstance(msg.get("meta"), dict) else {}
        return {
            "route": str(meta.get("route") or "").strip().lower(),
            "kind": str(msg.get("kind") or "").strip().lower(),
            "content": str(msg.get("content") or "").strip(),
        }
    return {}


def _extract_context_reference_text(user_input: str, recent_content: str) -> tuple[str, str]:
    """从上一条回复中粗略定位用户指代的编号项。"""
    q = str(user_input or "")
    ordinal: Optional[int] = None
    m = re.search(r"第(\d{1,2}|[一二两三四五六七八九十]{1,3})(个|点|条|题)", q)
    if m:
        ordinal = _parse_cn_ordinal(m.group(1))
    label = f"第 {ordinal} 个" if ordinal else "刚才提到的内容"

    content = str(recent_content or "").strip()
    if not content:
        return label, ""
    if ordinal:
        item_patterns = [
            rf"^\s*(?:{ordinal}|{ordinal}[\.、\)]|Day\s*{ordinal}|第\s*{ordinal}\s*[天点条题])\s*[:：.\)、-]?\s*(.+)$",
            rf"^\s*[-*]\s*(?:{ordinal}[\.、\)]|第\s*{ordinal}\s*[点条题])\s*(.+)$",
        ]
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        for pattern in item_patterns:
            for line in lines:
                found = re.search(pattern, line, flags=re.IGNORECASE)
                if found:
                    return label, found.group(1).strip()[:300]
        if 0 < ordinal <= len(lines):
            return label, lines[ordinal - 1][:300]
    return label, content[:300]


def _build_context_reference_reply(session_id: str, user_input: str) -> Dict[str, Any]:
    """为低信息指代提供确定性澄清，避免误判成闲聊。"""
    recent = _get_recent_assistant_route(session_id)
    label, snippet = _extract_context_reference_text(user_input, recent.get("content", ""))
    if snippet:
        text = f"我理解你说的{label}是：{snippet}\n\n你可以继续问“展开讲”或补充具体疑问，我会按这个上下文解释。"
    else:
        text = "我理解你在追问上一条回复中的内容，但当前会话里没有足够上下文。请贴出对应条目，我再继续解释。"
    return {
        "kind": "chat",
        "render_mode": "markdown",
        "text": text,
        "payload": {},
        "meta": {"deterministic_followup": True, "context_reference_resolved": True},
    }


def _build_recent_practice_snapshot(session_id: str, limit: int = 6) -> str:
    """构造近期作答摘要，补充给 Supervisor 作为判定参考。"""
    sid = str(session_id or "").strip() or "default"
    safe_limit = max(3, min(int(limit or 6), 20))
    try:
        from utils.memory_service import memory_manager

        history = memory_manager.get_practice_history(sid, safe_limit)
        stats = memory_manager.get_knowledge_point_stats(sid)
    except Exception as e:
        logger.warning(f"[Supervisor] 读取练习快照失败 sid={sid}: {e}")
        return ""

    if not history:
        return ""

    total = len(history)
    correct = sum(1 for row in history if bool(row.get("is_correct")))
    accuracy = round((correct / total) * 100.0, 1) if total > 0 else 0.0
    weak_points = [kp for kp, data in stats.items() if bool(data.get("weak"))][:3] if stats else []
    latest_items = []
    for row in history[:3]:
        status = "正确" if bool(row.get("is_correct")) else "错误"
        kp = str(row.get("knowledge_point") or "未标注").strip()
        latest_items.append(f"{status}:{kp}")

    lines = [
        f"- 最近{total}题正确率：{correct}/{total}（{accuracy}%）",
        f"- 薄弱点：{', '.join(weak_points) if weak_points else '暂无'}",
        f"- 最近作答：{' / '.join(latest_items) if latest_items else '暂无'}",
    ]
    return "\n".join(lines)


def _build_followup_intent_hint(user_input: str, has_recent_quiz_context: bool) -> str:
    """构造低信息跟进的软提示，让 LLM 结合上下文自主判定。"""
    if not _is_quiz_review_followup(user_input):
        return ""
    if has_recent_quiz_context:
        return (
            "- 当前用户提问是低信息跟进（如“怎么样/如何”），且上一轮为出题或作答场景。\n"
            "- 请优先理解为“复盘/评价最近作答表现”，默认倾向 route=history。\n"
            "- 仅当用户明确要求“继续出题/再来几题”时，才路由到 quiz。"
        )
    return (
        "- 当前用户提问是低信息跟进（如“怎么样/如何”），请先结合近期上下文判断语义。\n"
        "- 若未出现明确出题关键词，不要把该句强改写成“出题/考试”请求。"
    )


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数量：中文约1.5 tokens/字符"""
    if not text:
        return 0
    return int(len(text) * 1.5)


def _check_token_budget(input_query: str, history: str, memory: str) -> bool:
    """检查 token 预算是否充足，返回 False 表示需要精简模式"""
    # 估算输入 token
    input_tokens = _estimate_tokens(input_query)
    history_tokens = _estimate_tokens(history)
    memory_tokens = _estimate_tokens(memory)

    # 预留 2000 tokens 给输出和系统 prompt
    total_estimate = input_tokens + history_tokens + memory_tokens + 2000
    # MiniMax 通常 32K 上下文，这里用 8000 作为安全线
    MAX_BUDGET = 8000

    if total_estimate > MAX_BUDGET:
        logger.warning(f"[Supervisor] Token 预算超限: 估算{total_estimate} > 限制{MAX_BUDGET}")
        return False
    return True


def _is_ops_export_intent(query: str) -> bool:
    """判断是否属于导出/下载试卷类文件的管理操作。"""
    text = str(query or "").strip()
    if not text:
        return False

    # 普通“生成试卷”应走 exam；只有明确文件产物/下载语义才走 ops。
    has_export_verb = bool(re.search(r"(导出|下载|导出来|保存为|另存为)", text, flags=re.IGNORECASE))
    has_generated_file = bool(re.search(r"生成\s*(word|docx|答案卷|答题卡|答案纸)", text, flags=re.IGNORECASE))
    has_exam_artifact = bool(re.search(r"(试卷|答案卷|答题卡|答案纸|word|docx)", text, flags=re.IGNORECASE))
    return (has_export_verb or has_generated_file) and has_exam_artifact


def _is_exam_generation_intent(query: str) -> bool:
    """识别短句试卷生成请求；复杂自然语言留给 LLM Router 判断。"""
    text = str(query or "").strip()
    if not text or _is_ops_export_intent(text):
        return False
    compact = re.sub(r"\s+", "", text)
    if len(compact) > 80:
        return False
    has_exam_artifact = bool(re.search(r"(试卷|测试卷|考试|出卷|命题)", compact))
    has_generation_signal = bool(
        re.search(r"(生成|出|命题|组卷|编写|设计|来一套|给我一套|做一套)", compact)
    )
    return has_exam_artifact and has_generation_signal


def _is_courseware_management_intent(query: str) -> bool:
    """判断是否是课件管理类请求（增删查状态/文件层操作）。"""
    text = str(query or "").strip()
    if not text:
        return False
    return bool(
        re.search(r"(上传|删除|移除|重试|清理)\s*(课件|文件)", text)
        or re.search(r"(课件|文件).*(列表|清单|状态|向量|进度|数量|个数|多少|几个)", text)
        or re.search(r"(知识库|已上传).*(课件|文件|资料).*(数量|个数|多少|几个|列表|清单|状态)", text)
        or re.search(r"(知识库|已上传).*(数量|个数|多少|几个)", text)
        or re.search(r"已上传.*\d+\s*个", text)
        or re.search(r"(有多少|多少个|几个).*(课件|文件|资料)", text)
        or re.search(r"(查看|列出|展示|显示)\s*(课件文件|文件列表|课件列表)", text)
    )


def _is_courseware_content_intent(query: str) -> bool:
    """判断是否是“询问课件讲了什么”的内容理解请求。"""
    text = str(query or "").strip()
    if not text:
        return False
    if _is_courseware_management_intent(text):
        return False
    if "课件" not in text and "讲义" not in text and "pdf" not in text.lower():
        return False
    return bool(
        re.search(r"(讲了什么|讲的什么|讲了哪些|主要讲|内容是什么|内容概述|总结|概述|要点|大纲|覆盖了什么)", text)
        or re.search(r"(每个|每一|逐个).*(课件|讲义).*(讲|内容|要点|概述)", text)
    )


def _is_learning_loop_intent(query: str) -> bool:
    """识别强自主学习闭环请求，避免被普通 planner/quiz 分流。"""
    text = str(query or "").strip()
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    explicit_loop = bool(re.search(r"(学习闭环|自主复习|复习循环|进入学习闭环|开始自主复习|开始复习循环)", compact))
    plan_and_quiz = bool(
        re.search(r"(错题|薄弱点|练习画像|mastery|掌握情况).*(安排|计划|复习).*(出题|给.*题|练习)", compact, flags=re.IGNORECASE)
        or re.search(r"(今天|现在).*(该|应该).*(复习).*(题|练习)", compact)
    )
    return explicit_loop or plan_and_quiz


def _is_learning_loop_continue_intent(query: str) -> bool:
    """识别继续推进已有学习闭环的请求。"""
    text = re.sub(r"\s+", "", str(query or "").strip())
    if not text:
        return False
    return bool(
        re.search(r"(继续|下一步|推进|执行).*(学习闭环|自主复习|复习循环|Day1|今日闭环|今天任务|练习)", text, flags=re.IGNORECASE)
        or re.search(r"(开始|执行|生成|出).*(Day1|今日).*(练习|题)", text, flags=re.IGNORECASE)
    )


def _is_learning_loop_review_intent(query: str) -> bool:
    """识别学习闭环内的作答复盘请求。"""
    text = re.sub(r"\s+", "", str(query or "").strip())
    if not text:
        return False
    return bool(
        re.search(r"(学习闭环|自主复习|复习循环).*(复盘|分析|错因|本次作答|刚才作答|提交结果)", text)
        or re.search(r"(复盘|分析).*(本次|刚才|这次).*(作答|练习|错题)", text)
    )


def _get_route_fallback(error: str, original_query: str) -> str:
    """根据错误类型返回合适的 fallback 路由"""
    error_lower = error.lower()
    query = str(original_query or "")
    is_ops_export_intent = _is_ops_export_intent(query)
    is_ops_delete_intent = bool(
        re.search(r"删除第\s*\d+\s*条\s*(错题|练习记录|消息|对话|聊天记录)", query)
        or re.search(r"(删除|移除)\s*(错题|练习记录|消息|对话|聊天记录)", query)
        or re.search(r"(清空|清除)\s*(历史记录|练习记录|历史对话|聊天记录|对话记录)", query)
    )

    # 先按 query 关键词兜底，避免模型暂时失败时误路由到 chitchat
    if is_ops_export_intent:
        return 'ops'
    if _is_learning_loop_intent(query) or _is_learning_loop_continue_intent(query) or _is_learning_loop_review_intent(query):
        return 'learning_loop'
    if _is_courseware_content_intent(query):
        return 'rag'
    if any(kw in query for kw in ['出题', '做题', '练习', '刷题']):
        return 'quiz'
    if any(kw in query for kw in ['试卷', '考试', '出卷']):
        return 'exam'
    if any(kw in query for kw in ['创建会话', '删除会话', '重命名会话', '会话列表', '系统诊断', '删除课件', '查看样卷', 'token统计', '薄弱点']) or is_ops_delete_intent:
        return 'ops'
    if any(kw in query for kw in ['计划', '复习计划', '学习计划']):
        return 'planner'
    if any(kw in query for kw in ['错题', '历史', '记录', '统计', '正确率']):
        return 'history'

    # API 相关错误：API 不稳定，尝试让 RAG 直接处理
    if _is_retryable_llm_error(Exception(error)):
        # 否则走 rag，因为 rag 通常能给出有用回答
        return 'rag'

    # 超时错误：走 chitchat 快速响应
    if 'timeout' in error_lower or '超时' in error:
        return 'chitchat'

    # 其他错误：默认 chitchat
    return 'chitchat'


# ==================== Supervisor 节点 ====================

async def supervisor_node(state: SupervisorState) -> SupervisorState:
    """Supervisor: 分析意图，输出路由决策（Query重写+意图识别一次完成）"""
    logger.info("[Supervisor] 分析用户意图...")

    # 允许外层先做一次路由判定，然后在 workflow 内复用，避免 quiz/exam 回退路径重复调用 LLM 路由。
    if bool(state.get("supervisor_precomputed", False)):
        precomputed_route = str(state.get("route", "") or "").strip()
        valid_routes = {"rag", "quiz", "exam", "ops", "planner", "history", "learning_loop", "chitchat"}
        if precomputed_route in valid_routes:
            precomputed_reason = str(state.get("route_reason", "") or "预路由命中")
            precomputed_params = state.get("route_params", {}) if isinstance(state.get("route_params"), dict) else {}
            rewritten_query = str(state.get("input", "") or "").strip() or str(state.get("input", "") or "")
            logger.info(
                f"[Supervisor] 复用预路由结果 route={precomputed_route}, reason={precomputed_reason}"
            )
            return {
                "route": precomputed_route,
                "route_reason": precomputed_reason,
                "route_params": precomputed_params,
                "input": rewritten_query,
            }

    # 预检：打招呼直接走 chitchat，不调用 LLM
    greeting_keywords = ['你好', 'hi', 'hello', '嗨', '您好', 'hi there', 'hey']
    input_lower = state['input'].lower().strip()
    if input_lower in greeting_keywords or input_lower.startswith('你好，') or input_lower.startswith('hi,'):
        logger.info(f"[Supervisor] 检测到打招呼，直接路由 chitchat")
        return {"route": "chitchat", "route_reason": "打招呼问候", "route_params": {}, "input": state['input']}

    # 前端按钮传入的显式动作优先于自然语言 Router，避免按钮 prompt 被关键词误分流。
    if str(state.get("client_action", "") or "").strip() == "generate_exam":
        logger.info("[Supervisor] 命中前端显式生成试卷动作，强制路由 exam")
        return {
            "route": "exam",
            "route_reason": "前端显式动作 generate_exam（强制exam）",
            "route_params": {"topics": [], "quiz_types": ["选择题", "判断题", "填空题", "简答题"], "total_questions": 23},
            "input": state["input"],
        }

    # 预检：强自主学习闭环必须优先于普通 planner/quiz/history。
    if (
        _is_learning_loop_intent(state.get("input", ""))
        or _is_learning_loop_continue_intent(state.get("input", ""))
        or _is_learning_loop_review_intent(state.get("input", ""))
    ):
        logger.info("[Supervisor] 命中学习闭环意图，优先路由 learning_loop")
        return {
            "route": "learning_loop",
            "route_reason": "学习闭环意图（预检强制learning_loop）",
            "route_params": {"mode": "learning_loop"},
            "input": state["input"],
        }

    # 预检：导出试卷/答案卷属于管理操作，优先进入 ops，避免被 exam 关键词误吸走。
    if _is_ops_export_intent(state.get("input", "")):
        logger.info("[Supervisor] 命中导出试卷意图，优先路由 ops")
        return {
            "route": "ops",
            "route_reason": "导出试卷意图（预检强制ops）",
            "route_params": {"target": "exam_export", "action": "export", "limit": 20},
            "input": state["input"],
        }

    # 预检：确认短语必须进入 ops 执行链路，不能被课件列表查询抢走。
    if _is_knowledge_delete_confirmation(state.get("input", "")):
        logger.info("[Supervisor] 命中课件删除确认短语，优先路由 ops")
        return {
            "route": "ops",
            "route_reason": "课件删除确认短语（预检强制ops）",
            "route_params": {},
            "input": state["input"],
        }

    # 预检：用户明确要求联网/网络搜索时必须走工具，不让普通模型口头拒绝。
    if is_explicit_web_search_intent(state.get("input", "")):
        logger.info("[Supervisor] 命中显式联网搜索意图，优先路由 ops")
        return {
            "route": "ops",
            "route_reason": "显式联网搜索意图（预检强制ops）",
            "route_params": {"target": "web_search", "action": "search", "limit": 20},
            "input": state["input"],
        }

    # 预检：生成试卷必须优先于“已上传课件”这类课件元信息关键词。
    if _is_exam_generation_intent(state.get("input", "")):
        logger.info("[Supervisor] 命中试卷生成意图，优先路由 exam")
        return {
            "route": "exam",
            "route_reason": "试卷生成意图（预检强制exam）",
            "route_params": {"topics": [], "quiz_types": ["选择题", "判断题", "填空题", "简答题"], "total_questions": 23},
            "input": state["input"],
        }

    # 预检：课件数量/列表/状态是资源元信息，必须走工具查询，避免 RAG 片段估数。
    if _is_courseware_management_intent(state.get("input", "")):
        logger.info("[Supervisor] 命中课件管理查询，优先路由 ops")
        return {
            "route": "ops",
            "route_reason": "课件管理查询（预检强制ops）",
            "route_params": {"target": "knowledge", "action": "list", "limit": 200},
            "input": state["input"],
        }

    # 预检：课件“内容理解”问题优先走 RAG，避免误入 ops 触发慢工具链。
    if _is_courseware_content_intent(state.get("input", "")):
        logger.info("[Supervisor] 命中课件内容理解意图，优先路由 rag")
        return {
            "route": "rag",
            "route_reason": "课件内容理解（预检强制rag）",
            "route_params": {"topic": state.get("input", "")},
            "input": state["input"],
        }

    # 预检：按句子边界截断历史，保留语义完整性
    recent_history = _format_history_with_limit(state.get('chat_history', []))
    sid = state.get("session_id") or current_session_id.get() or "default"
    has_recent_quiz_context = _has_recent_quiz_turn(str(sid))

    quiz_attempt = _parse_quiz_answer_attempt(state.get("input", ""))
    if quiz_attempt.get("is_answer_check") and not _is_quiz_generation_followup(state.get("input", "")):
        logger.info("[Supervisor] 命中 quiz 判分跟进，强制 route=quiz action=answer_check")
        return {
            "route": "quiz",
            "route_reason": "确定性识别：上一题作答/判分跟进",
            "route_params": {
                "topic": "",
                "quiz_type": "选择题",
                "num": 1,
                "action": "answer_check",
                "answer": quiz_attempt.get("answer", ""),
                "question_number": quiz_attempt.get("question_number"),
                "deterministic_followup": True,
                "quiz_action": "answer_check",
            },
            "input": state["input"],
        }

    if _is_context_reference_followup(state.get("input", "")):
        recent_route = _get_recent_assistant_route(str(sid))
        valid_follow_routes = {"rag", "quiz", "exam", "ops", "planner", "history", "learning_loop"}
        route = recent_route.get("route") if recent_route.get("route") in valid_follow_routes else "rag"
        if recent_route.get("kind") == "quiz_set":
            route = "quiz"
        elif recent_route.get("kind") == "exam_paper":
            route = "exam"
        logger.info(f"[Supervisor] 命中上下文指代跟进，沿用 route={route}")
        return {
            "route": route,
            "route_reason": "确定性识别：上下文指代跟进",
            "route_params": {"action": "context_reference", "context_reference_resolved": True},
            "input": state["input"],
        }

    practice_snapshot = _build_recent_practice_snapshot(str(sid))
    if practice_snapshot:
        recent_history = f"{recent_history}\n\n【最近练习快照】\n{practice_snapshot}"
    followup_hint = _build_followup_intent_hint(state.get("input", ""), has_recent_quiz_context)
    if followup_hint:
        recent_history = f"{recent_history}\n\n【意图判定提示】\n{followup_hint}"

    # 使用主模型进行意图识别
    result = ""
    try:
        result = await _call_llm_for_supervisor(
            SUPERVISOR_PROMPT,
            input=state['input'],
            memory_context=state.get('memory_context', ''),
            recent_history=recent_history,
        )
        decision = _validate_supervisor_decision(
            _extract_json(result),
            state['input'],
            has_recent_quiz_context=has_recent_quiz_context,
        )
        rewritten_query = decision['rewritten_query']
        route = decision['route']
        reason = decision['reason']
        params = decision['params']

        # 优化4: 检查 token 预算
        if not _check_token_budget(state['input'], recent_history, state.get('memory_context', '')):
            logger.warning("[Supervisor] Token 预算超限，启用精简模式")

        logger.info(f"[Supervisor] Query: '{state['input']}' -> '{rewritten_query}'")
        logger.info(f"[Supervisor] 路由: route={route}, reason={reason}, params={params}")
    except Exception as e:
        logger.warning(f"[Supervisor] 意图识别失败: {e}, result={result[:200] if result else 'empty'}")
        # 优化1: 多层 fallback
        route = _get_route_fallback(str(e), state['input'])
        reason = '意图识别失败，使用 fallback'
        params = {}
        rewritten_query = state['input']

    logger.info(f"[Supervisor] 路由 → {route}")
    return {"route": route, "route_reason": reason, "route_params": params, "input": rewritten_query}


async def tool_orchestrator_node(state: SupervisorState) -> SupervisorState:
    """在专家 Agent 前统一规划并执行安全工具调用。"""
    route = str(state.get("route") or "rag")
    client_action = str(state.get("client_action") or "").strip()
    if client_action == "generate_exam" or route == "chitchat":
        return state

    recent_history = _format_history_with_limit(state.get("chat_history", []), max_chars=800)
    tool_context = str(state.get("tool_context") or "").strip()
    if tool_context:
        recent_history = f"{recent_history}\n\n{tool_context}".strip()

    react_result = await run_react_orchestrator(
        query=str(state.get("input") or ""),
        session_id=str(state.get("session_id") or current_session_id.get() or "default"),
        route_hint=route,
        recent_history=recent_history,
    )
    observations = react_result.get("observations") if isinstance(react_result.get("observations"), list) else []
    if observations:
        try:
            memory_manager.append_tool_contexts(
                str(state.get("session_id") or current_session_id.get() or "default"),
                observations,
                turn_query=str(state.get("input") or ""),
                source="react_orchestrator",
            )
        except Exception as e:
            logger.warning(f"[ToolContext] 保存 ReAct observation 失败: {e}")
    params = state.get("route_params", {}) if isinstance(state.get("route_params", {}), dict) else {}
    delegate = react_result.get("delegate") if isinstance(react_result.get("delegate"), dict) else {}
    delegate_params = delegate.get("route_params") if isinstance(delegate.get("route_params"), dict) else {}
    next_route = str(react_result.get("next_agent") or route)
    return {
        **state,
        "route": next_route,
        "route_params": {
            **params,
            **delegate_params,
            "orchestrator": {
                "tool_plan": react_result.get("tool_plan", {}),
                "task_plan": react_result.get("task_plan", {}),
                "plan_execution": react_result.get("plan_execution", []),
                "observations": observations,
                "trace": react_result.get("trace", []),
                "guard_blocked": bool(react_result.get("guard_blocked", False)),
                "react_status": str(react_result.get("react_status") or ""),
                "final_answer": str(react_result.get("final_answer") or ""),
                "route_before_orchestrator": route,
                "route_after_orchestrator": next_route,
                "route_changed_by_orchestrator": next_route != route,
            },
        },
    }


def _get_orchestrator_observations(state: SupervisorState) -> List[Dict[str, Any]]:
    """读取 ReAct observation，供专家 Agent 复用已完成的取证结果。"""
    params = state.get("route_params", {}) if isinstance(state.get("route_params", {}), dict) else {}
    orchestrator = params.get("orchestrator") if isinstance(params.get("orchestrator"), dict) else {}
    observations = orchestrator.get("observations") if isinstance(orchestrator.get("observations"), list) else []
    return [item for item in observations if isinstance(item, dict)]


def _format_orchestrator_observations(
    state: SupervisorState,
    tool_names: set[str] | None = None,
    *,
    title: str = "ReAct工具观察",
) -> str:
    """把指定工具的 observation 拼成可注入 prompt 的上下文文本。"""
    allowed = set(tool_names or [])
    lines: List[str] = []
    for item in _get_orchestrator_observations(state):
        tool_name = str(item.get("tool_name") or "").strip()
        if allowed and tool_name not in allowed:
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"【{tool_name}】\n{content}")
    if not lines:
        return ""
    return f"【{title}】\n" + "\n\n".join(lines)


def _get_orchestrator_observation_content(state: SupervisorState, tool_name: str) -> str:
    """读取指定工具的第一条 observation 原文。"""
    for item in _get_orchestrator_observations(state):
        if str(item.get("tool_name") or "").strip() == tool_name:
            return str(item.get("content") or "").strip()
    return ""


def _attach_question_type_plan(result: Any, question_type_plan: Dict[str, Any]) -> Any:
    """把题型决策写入专家结果，供消息协议和前端展示。"""
    if not isinstance(result, dict):
        return result
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    return {
        **result,
        "payload": {**payload, "question_type_plan": question_type_plan},
        "meta": {**meta, "question_type_plan": question_type_plan},
    }


def _merge_quiz_batch_results(topic: str, results: List[Dict[str, Any]], question_type_plan: Dict[str, Any]) -> Dict[str, Any]:
    """合并多题型小练习批次，保持前端只收到一个 quiz_set。"""
    from agent.multi_agent.quiz_parser import build_quiz_payload, build_quiz_text_from_questions, questions_from_quiz_text

    questions: List[Dict[str, Any]] = []
    batch_meta: List[Dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        batch_questions = payload.get("questions") if isinstance(payload.get("questions"), list) else []
        if not batch_questions:
            batch_questions = questions_from_quiz_text(str(result.get("text") or ""), topic).get("questions", [])
        for item in batch_questions or []:
            if isinstance(item, dict):
                questions.append(dict(item))
        batch_meta.append(result.get("meta") if isinstance(result.get("meta"), dict) else {})

    if not questions:
        return results[0] if results else {
            "kind": "chat",
            "render_mode": "markdown",
            "text": "本次多题型练习生成失败，请稍后重试。",
            "payload": {},
            "meta": {"question_type_plan": question_type_plan},
        }

    for idx, question in enumerate(questions, start=1):
        question["id"] = idx
    payload = build_quiz_payload(topic or "练习题", questions)
    text = build_quiz_text_from_questions(questions)
    return {
        "kind": "quiz_set",
        "render_mode": "interactive_cards",
        "text": text,
        "payload": {**payload, "question_type_plan": question_type_plan},
        "meta": {
            "delivery_mode": "full",
            "question_type_plan": question_type_plan,
            "merged_quiz_batches": len(results),
            "batch_meta": batch_meta,
        },
    }


# ==================== SubAgent 节点 ====================

async def rag_subagent_node(state: SupervisorState) -> SupervisorState:
    """RAG SubAgent: 检索课件原始片段，直接回答（单次 LLM）"""
    logger.info("[RAG SubAgent] 检索课件资料...")
    rag_start = time.time()

    from rag.rag_service import RagSummarizeService
    sid = state.get('session_id') or current_session_id.get() or 'default'
    current_session_id.set(sid)

    params = state.get('route_params', {}) if isinstance(state.get('route_params', {}), dict) else {}
    if params.get("action") == "context_reference":
        structured = _build_context_reference_reply(str(sid), state.get("input", ""))
        return {"subagent_result": structured, "final_answer": structured["text"]}

    topic = params.get('topic') or state['input']

    # 格式化近期对话历史，供 RAG 参考
    history_lines = []
    for msg in state.get('chat_history', []):
        role = "学生" if isinstance(msg, HumanMessage) else "助手"
        history_lines.append(f"{role}: {msg.content[:300]}")
    recent_history = "\n".join(history_lines) if history_lines else "（无近期对话）"

    orchestrator_context = _format_orchestrator_observations(state, {"search_courseware"}, title="ReAct课件检索结果")
    if orchestrator_context:
        context = orchestrator_context
        logger.info("[RAG SubAgent] 复用 ReAct 课件 observation，跳过重复检索")
    else:
        try:
            rag = await get_rag_service()
            # retrieve_context 只检索、不调 LLM，避免冗余的双重 LLM 调用
            retrieve_start = time.time()
            context = await rag.retrieve_context(topic, mode="rag_chat")
            logger.info(f"[Latency] rag_retrieve_ms={int((time.time() - retrieve_start) * 1000)}")
            # 如果检索为空，给出友好提示
            if not context or context.strip() == "":
                logger.warning("[RAG SubAgent] 知识库为空，请先上传课件")
                context = "【提示】当前知识库为空，请先上传课件后再提问。你可以通过点击「上传课件」按钮来添加复习资料。"
        except Exception as e:
            logger.warning(f"[RAG SubAgent] 检索失败: {e}")
            context = "【提示】知识库检索遇到问题，请确保已上传课件。如未上传，请先上传复习资料。"

    llm_start = time.time()
    result_raw = await _call_llm(
        RAG_SUBAGENT_PROMPT,
        memory_context=state.get('memory_context', ''),
        recent_history=recent_history,
        context=context,
        input=state['input'],
    )
    logger.info(f"[Latency] rag_answer_llm_ms={int((time.time() - llm_start) * 1000)}")

    parsed = _extract_json(result_raw)
    answer = str(parsed.get("answer", "")).strip() if isinstance(parsed, dict) else ""
    grounded_evidence = _extract_grounded_evidence(parsed, context)

    if answer and grounded_evidence:
        rag_inc("rag_grounded_pass_count", 1)
        logger.info(f"[RAG SubAgent] 回答依据校验通过，evidence_count={len(grounded_evidence)}")
        logger.info(f"[Latency] rag_total_ms={int((time.time() - rag_start) * 1000)}")
        return {
            "subagent_result": _build_rag_structured_result(answer, grounded_evidence),
            "final_answer": answer,
        }

    # 非严格模式：课件作为主要依据即可，不再因证据条目不足直接拒答
    if answer:
        rag_inc("rag_grounded_partial_count", 1)
        logger.warning("[RAG SubAgent] 回答依据条目不足，按课件优先策略继续返回答案")
        logger.info(f"[Latency] rag_total_ms={int((time.time() - rag_start) * 1000)}")
        return {
            "subagent_result": _build_rag_structured_result(answer, grounded_evidence, "evidence_partial"),
            "final_answer": answer,
        }

    logger.warning("[RAG SubAgent] 无有效回答，触发保守提示")
    rag_inc("rag_grounded_fail_count", 1)
    logger.info(f"[Latency] rag_total_ms={int((time.time() - rag_start) * 1000)}")
    fallback = (
        "根据当前检索结果，我暂时无法提取到可核验的课件依据。"
        "为避免误导，请换个更具体的问题，或补充相关课件后再试。"
    )
    return {
        "subagent_result": _build_rag_structured_result(fallback, grounded_evidence, "no_grounded_evidence"),
        "final_answer": fallback,
    }


async def quiz_subagent_node(state: SupervisorState) -> SupervisorState:
    """Quiz SubAgent: 调用 Reflexion 出题工作流"""
    logger.info("[Quiz SubAgent] 启动 Reflexion 出题流程...")
    logger.info(f"[Quiz SubAgent] state.input={state.get('input', '')[:200]}...")
    logger.info(f"[Quiz SubAgent] route_params={state.get('route_params', {})}")
    logger.info(f"[Quiz SubAgent] session_id={state.get('session_id', 'unknown')}")

    sid = state.get('session_id') or current_session_id.get() or 'default'
    current_session_id.set(sid)
    params = state.get('route_params', {}) if isinstance(state.get('route_params', {}), dict) else {}
    if params.get("action") == "context_reference":
        structured = _build_context_reference_reply(str(sid), state.get("input", ""))
        return {"subagent_result": structured, "final_answer": structured["text"]}
    if params.get("action") == "answer_check":
        logger.info("[Quiz SubAgent] 命中 answer_check，跳过出题链路")
        structured = _build_quiz_answer_check_result(str(sid), params)
        return {"subagent_result": structured, "final_answer": structured["text"]}

    from agent.multi_agent.quiz_agent import run_quiz_agent
    from agent.multi_agent.question_type_resolver import resolve_question_type_plan
    from api.routers.knowledge import get_sample_paper_context
    from utils.memory_service import memory_manager

    sample_ctx = get_sample_paper_context(sid)

    topic = params.get('topic', '')
    question_type_plan = await resolve_question_type_plan(
        query=str(state.get("input") or ""),
        route="quiz",
        session_id=str(sid),
        sample_paper_context=sample_ctx,
        route_params=params,
    )
    quiz_batches = [
        (str(item.get("type") or "选择题"), _safe_int(item.get("count"), default=0, minimum=0, maximum=20))
        for item in question_type_plan.get("question_type_plan", [])
        if isinstance(item, dict) and _safe_int(item.get("count"), default=0, minimum=0, maximum=20) > 0
    ]
    if not quiz_batches:
        quiz_batches = [("选择题", 3)]

    # 如果 topic 为空或只有意图词（题、出题等），视为通用出题请求
    # 不拦截，让 LLM 自己选择合适的知识点出题
    intent_only_keywords = ['题', '出题', '做题', '练习', '考试', '测试', '一道', '几个', '一些', '做一个', '出']
    # topic 为空时回退到当前输入，避免参数提取失败导致上下文丢失。
    topic_stripped = (str(topic or "").strip() or str(state.get("input", "") or "").strip())
    is_generic_request = (
        not topic_stripped or
        topic_stripped in intent_only_keywords
    )

    weak_context = ""
    stats = memory_manager.get_knowledge_point_stats(sid)
    if stats:
        weak_points = [(kp, data) for kp, data in stats.items() if data.get('weak')]
        if weak_points and not is_generic_request:
            # 只有在用户指定了具体知识点时，才优先出薄弱点题目
            weak_lines = [f"- {kp}: 正确率{data['accuracy']:.0f}%（{data['correct']}/{data['total']}）" for kp, data in weak_points]
            weak_context = f"\n\n【薄弱点提醒】以下知识点正确率低于60%，请优先出这些方面的题目：\n" + "\n".join(weak_lines)
            logger.info(f"[Quiz SubAgent] 检测到 {len(weak_points)} 个薄弱点，将优先出相关题目")

    # 通用请求不再注入固定锚点词；交给出题链路按课件池随机抽样检索
    effective_topic = topic_stripped
    recent_signatures = memory_manager.get_recent_quiz_question_signatures(
        sid,
        rounds=15,
        limit=600,
    )
    if recent_signatures:
        logger.info(f"[Quiz SubAgent] 已加载最近15轮签名 {len(recent_signatures)} 条，启用跨轮去重")
    if len(quiz_batches) > 1:
        batch_results: List[Dict[str, Any]] = []
        for quiz_type, num in quiz_batches:
            batch_results.append(await run_quiz_agent(
                effective_topic + weak_context,
                quiz_type,
                num,
                sample_ctx,
                force_llm_critic=bool(state.get("quiz_force_llm_critic", False)),
                forbidden_question_signatures=recent_signatures,
            ))
        quiz_result = _merge_quiz_batch_results(effective_topic, batch_results, question_type_plan)
    else:
        quiz_type, num = quiz_batches[0]
        quiz_result = await run_quiz_agent(
            effective_topic + weak_context,
            quiz_type,
            num,
            sample_ctx,
            force_llm_critic=bool(state.get("quiz_force_llm_critic", False)),
            forbidden_question_signatures=recent_signatures,
        )
        quiz_result = _attach_question_type_plan(quiz_result, question_type_plan)
    quiz_text = quiz_result.get("text", "") if isinstance(quiz_result, dict) else quiz_result
    if not quiz_text:
        quiz_text = "抱歉，出题失败了，请稍后重试。"
        if isinstance(quiz_result, dict):
            quiz_result = {**quiz_result, "text": quiz_text}
        else:
            quiz_result = quiz_text
    if isinstance(quiz_result, dict):
        try:
            round_questions = _extract_quiz_question_texts(quiz_result)
            if round_questions:
                saved_count = memory_manager.record_quiz_round_questions(
                    sid,
                    round_questions,
                    keep_recent_rounds=180,
                )
                logger.info(f"[Quiz SubAgent] 已记录本轮题干签名 {saved_count} 条")
        except Exception as e:
            logger.warning(f"[Quiz SubAgent] 记录本轮题干签名失败: {e}")
    return {"subagent_result": quiz_result, "final_answer": quiz_text}


async def exam_subagent_node(state: SupervisorState) -> SupervisorState:
    """Exam SubAgent: 调用 Reflexion 出卷工作流"""
    logger.info("[Exam SubAgent] 启动 Reflexion 出卷流程...")
    exam_start = time.time()
    logger.info(f"[Exam SubAgent] state.input={state.get('input', '')[:200]}...")
    logger.info(f"[Exam SubAgent] route_params={state.get('route_params', {})}")
    logger.info(f"[Exam SubAgent] session_id={state.get('session_id', 'unknown')}")
    logger.info(f"[Exam SubAgent] stage_plan={state.get('exam_stage_plan', False)}, rerun_stage={state.get('exam_rerun_stage')}")

    from agent.multi_agent.quiz_agent import run_exam_agent
    from agent.multi_agent.question_type_resolver import resolve_question_type_plan
    from api.routers.knowledge import get_sample_paper_context
    from utils.memory_service import memory_manager

    sid = state.get('session_id') or current_session_id.get() or 'default'
    current_session_id.set(sid)

    params = state.get('route_params', {}) if isinstance(state.get('route_params', {}), dict) else {}
    if params.get("action") == "context_reference":
        structured = _build_context_reference_reply(str(sid), state.get("input", ""))
        return {"subagent_result": structured, "final_answer": structured["text"]}

    sample_ctx = get_sample_paper_context(sid)
    search_format_observation = _get_orchestrator_observation_content(state, "search_exam_format")

    topics = params.get('topics', [])
    quiz_types = params.get('quiz_types')
    if not quiz_types or not isinstance(quiz_types, list):
        quiz_types = ['选择题', '填空题', '判断题', '简答题']
    total = _safe_int(params.get('total_questions', 23), default=23, minimum=1, maximum=60)
    if params.get('total_questions', None) in ("", None):
        logger.warning(f"[Exam SubAgent] total_questions 缺失/非法，已回退默认值 total={total}")
    raw_quantity_dist = params.get('quantity_dist', {}) or {}
    quantity_dist = {}
    quantity_dist_is_fallback = False
    if isinstance(raw_quantity_dist, dict):
        for key in ("choice", "fill", "judge", "essay"):
            if key in raw_quantity_dist:
                quantity_dist[key] = _safe_int(raw_quantity_dist.get(key), default=0, minimum=0, maximum=60)

    # LLM 意图识别失败时，route_params 可能为空：从原始输入里做一次保底抽取
    raw_input = str(state.get("input", "") or "")
    if not quantity_dist and raw_input:
        alias_to_key = {
            "选择题": "choice",
            "填空题": "fill",
            "判断题": "judge",
            "简答题": "essay",
            "解答题": "essay",
        }
        for label, key in alias_to_key.items():
            m = re.search(rf"{label}\s*(\d+)\s*[道题]?", raw_input)
            if m:
                quantity_dist[key] = _safe_int(m.group(1), default=0, minimum=0, maximum=60)
        # 常见综合卷默认分布（用户没写数量 or 解析失败）
        if not any(int(quantity_dist.get(k, 0)) > 0 for k in ("choice", "fill", "judge", "essay")) and any(
            w in raw_input for w in ["综合测试卷", "综合试卷", "期末试卷", "出一套试卷", "生成试卷"]
        ):
            quantity_dist = {"choice": 10, "fill": 5, "judge": 5, "essay": 3}
            quantity_dist_is_fallback = True
            logger.warning("[Exam SubAgent] route_params 缺失，已按综合卷默认分布回填 quantity_dist=10/5/5/3")

    resolver_params = params if quantity_dist_is_fallback or not quantity_dist else {**params, "quantity_dist": quantity_dist}
    question_type_plan = await resolve_question_type_plan(
        query=raw_input,
        route="exam",
        session_id=str(sid),
        sample_paper_context=sample_ctx,
        search_observation=search_format_observation,
        route_params=resolver_params,
    )
    quantity_dist = question_type_plan.get("quantity_dist", {}) if isinstance(question_type_plan.get("quantity_dist"), dict) else {}
    quiz_types = question_type_plan.get("quiz_types", []) if isinstance(question_type_plan.get("quiz_types"), list) else quiz_types
    total = _safe_int(question_type_plan.get("total_questions", total), default=total, minimum=1, maximum=60)
    raw_format_reference = str(question_type_plan.get("raw_reference") or "").strip()
    if raw_format_reference and question_type_plan.get("source") == "web_search":
        sample_ctx = (sample_ctx or "") + "\n\n【联网题型参考】\n" + raw_format_reference[:2000]

    if any(int(quantity_dist.get(k, 0)) > 0 for k in ("choice", "fill", "judge", "essay")):
        total = int(sum(int(quantity_dist.get(k, 0) or 0) for k in ("choice", "fill", "judge", "essay")))
        if total > 0:
            quiz_types = []
            if quantity_dist.get("choice", 0) > 0:
                quiz_types.append("选择题")
            if quantity_dist.get("fill", 0) > 0:
                quiz_types.append("填空题")
            if quantity_dist.get("judge", 0) > 0:
                quiz_types.append("判断题")
            if quantity_dist.get("essay", 0) > 0:
                quiz_types.append("简答题")

    # 如果 topics 为空或只有意图词，视为通用出卷请求
    # 不拦截，让 LLM 自己选择合适的知识点出卷
    intent_only_keywords = ['卷', '试卷', '考试', '测试', '题', '出卷', '出题', '综合', '做一个', '来', '出', '一张', '一份']
    primary_topic = topics[0] if topics else ""
    topic_stripped = primary_topic.strip()
    is_generic_request = (
        not topic_stripped or
        topic_stripped in intent_only_keywords
    )

    weak_context = ""
    stats = memory_manager.get_knowledge_point_stats(sid)
    if stats:
        weak_points = [(kp, data) for kp, data in stats.items() if data.get('weak')]
        if weak_points and not is_generic_request:
            # 只有在用户指定了具体知识点时，才优先出薄弱点题目
            weak_lines = [f"- {kp}: 正确率{data['accuracy']:.0f}%（{data['correct']}/{data['total']}）" for kp, data in weak_points]
            weak_context = f"\n\n【薄弱点提醒】以下知识点正确率低于60%，请优先出这些方面的题目：\n" + "\n".join(weak_lines)
            logger.info(f"[Exam SubAgent] 检测到 {len(weak_points)} 个薄弱点，将优先出相关题目")

    # 如果用户没有指定具体知识点，topics 为空，让 LLM 自己选择
    topics_with_weak = [t + weak_context for t in topics] if topics else []

    logger.info(f"[Exam SubAgent] topics={topics}, quiz_types={quiz_types}, total={total}, quantity_dist={quantity_dist}")

    if bool(state.get("exam_stage_plan", False)):
        exam_result = await run_exam_agent(
            topics_with_weak,
            quiz_types,
            total,
            sample_ctx,
            quantity_dist,
            stage_plan=True,
            rerun_stage=str(state.get("exam_rerun_stage") or "").strip() or None,
            partial_questions=state.get("exam_partial_questions") if isinstance(state.get("exam_partial_questions"), list) else None,
            exam_fast_mode=bool(state.get("exam_fast_mode", True)),
        )
    else:
        exam_result = await run_exam_agent(
            topics_with_weak, quiz_types, total, sample_ctx, quantity_dist,
            exam_fast_mode=bool(state.get("exam_fast_mode", True)),
        )
    logger.info(f"[Latency] exam_generate_ms={int((time.time() - exam_start) * 1000)}")
    exam_text = exam_result.get("text", "") if isinstance(exam_result, dict) else exam_result
    if not exam_text or not str(exam_text).strip():
        exam_text = "试卷生成失败：本次为严格模式（禁止降级），请稍后重试或减少题量。"
        if isinstance(exam_result, dict):
            exam_result = {**exam_result, "text": exam_text}
        else:
            exam_result = exam_text
    exam_result = _attach_question_type_plan(exam_result, question_type_plan)
    return {"subagent_result": exam_result, "final_answer": exam_text}


async def ops_subagent_node(state: SupervisorState) -> SupervisorState:
    """Ops SubAgent：使用 ReAct 样式自动决策并执行管理工具。"""
    logger.info("[Ops SubAgent] 启动 ReAct 管理操作流程...")

    sid = state.get('session_id') or current_session_id.get() or 'default'
    current_session_id.set(sid)
    user_input = str(state.get("input", "") or "").strip()
    params = state.get("route_params", {}) if isinstance(state.get("route_params", {}), dict) else {}
    orchestrator = params.get("orchestrator") if isinstance(params.get("orchestrator"), dict) else {}
    observations = orchestrator.get("observations") if isinstance(orchestrator.get("observations"), list) else []
    if bool(orchestrator.get("guard_blocked", False)):
        tool_plan = orchestrator.get("tool_plan") if isinstance(orchestrator.get("tool_plan"), dict) else {}
        calls = tool_plan.get("tool_calls") if isinstance(tool_plan.get("tool_calls"), list) else []
        first_call = calls[0] if calls and isinstance(calls[0], dict) else {}
        tool_name = str(first_call.get("tool_name") or "unknown_tool")
        pending_args = first_call.get("args") if isinstance(first_call.get("args"), dict) else {}
        trace = orchestrator.get("trace") if isinstance(orchestrator.get("trace"), list) else []
        required_phrase, arg_error = _ops_required_confirmation_phrase(tool_name, pending_args)
        reason = arg_error or f"检测到高风险操作 `{tool_name}`，需要二次确认后执行。"
        structured = _build_ops_guard_reply(
            trace=trace,
            tool_name=tool_name,
            pending_args=pending_args,
            required_phrase=required_phrase,
            reason=reason,
        )
        structured["meta"]["orchestrator_used"] = True
        structured["meta"]["agent_trace"] = trace
        structured["payload"]["agent_trace"] = trace
        return {"subagent_result": structured, "final_answer": structured["text"]}

    final_answer = str(orchestrator.get("final_answer") or "").strip()
    if final_answer and not observations:
        trace = orchestrator.get("trace") if isinstance(orchestrator.get("trace"), list) else []
        structured = {
            "kind": "chat",
            "render_mode": "markdown",
            "text": final_answer,
            "payload": {"ops_trace": trace, "agent_trace": trace},
            "meta": {
                "ops_react": True,
                "orchestrator_used": True,
                "agent_trace": trace,
            },
        }
        return {"subagent_result": structured, "final_answer": final_answer}

    if observations and str(observations[0].get("tool_name") or "") in {"web_search", "search_exam_format"}:
        lines = ["## 联网搜索结果", ""]
        for item in observations:
            tool_name = str(item.get("tool_name") or "tool")
            content = str(item.get("content") or "").strip()
            lines.append(f"### {tool_name}")
            lines.append(content or "工具已调用，但没有返回可展示内容。")
            lines.append("")
        text = "\n".join(lines).strip()
        trace = orchestrator.get("trace") if isinstance(orchestrator.get("trace"), list) else []
        structured = {
            "kind": "chat",
            "render_mode": "markdown",
            "text": text,
            "payload": {"ops_trace": trace, "agent_trace": trace},
            "meta": {
                "ops_react": True,
                "orchestrator_used": True,
                "agent_trace": trace,
            },
        }
        return {"subagent_result": structured, "final_answer": text}

    tool_map = _build_ops_tool_map()
    if not tool_map:
        text = "当前未注册可用的管理工具，请稍后重试。"
        return {"subagent_result": text, "final_answer": text}

    tool_specs = _format_ops_tool_specs(tool_map)
    recent_history = _format_history_with_limit(state.get('chat_history', []), max_chars=500)
    trace: List[Dict[str, Any]] = []
    max_steps = 2
    limit = _safe_int(params.get("limit", 20), default=20, minimum=1, maximum=200)

    if params.get("action") == "context_reference":
        structured = _build_context_reference_reply(str(sid), user_input)
        return {"subagent_result": structured, "final_answer": structured["text"]}

    if params.get("target") == "exam_export" and params.get("action") == "export":
        exam_paper = _get_recent_exam_or_quiz_export_content(str(sid))
        if not exam_paper:
            text = "无法导出 Word：当前会话未找到可导出的试卷内容。请先生成一套试卷/小测卷，再说“导出刚才的试卷 Word”。"
            structured = {
                "kind": "chat",
                "render_mode": "markdown",
                "text": text,
                "payload": {"ops_trace": []},
                "meta": {"ops_react": True, "ops_deterministic": True},
            }
            return {"subagent_result": structured, "final_answer": text}
        tool_obj = tool_map.get("export_exam_docx_tool")
        if tool_obj is None:
            text = "导出试卷 Word 失败：export_exam_docx_tool 未注册。"
            return {"subagent_result": text, "final_answer": text}
        try:
            result = await tool_obj.ainvoke({"exam_paper": exam_paper, "course_name": "复习助手小测卷", "include_answers": True})
            result_text = _safe_json_dumps(result) if isinstance(result, (dict, list)) else str(result)
            trace.append({"step": 1, "tool": "export_exam_docx_tool", "args": {"course_name": "复习助手小测卷", "include_answers": True}, "observation": result_text[:1200]})
            structured = {
                "kind": "chat",
                "render_mode": "markdown",
                "text": result_text,
                "payload": {"ops_trace": trace},
                "meta": {"ops_react": True, "ops_steps": 1, "ops_deterministic": True},
            }
            return {"subagent_result": structured, "final_answer": result_text}
        except Exception as e:
            text = f"导出试卷 Word 失败：{e}"
            structured = {
                "kind": "chat",
                "render_mode": "markdown",
                "text": text,
                "payload": {"ops_trace": trace},
                "meta": {"ops_react": True, "ops_steps": len(trace), "ops_deterministic": True},
            }
            return {"subagent_result": structured, "final_answer": text}

    if params.get("target") == "knowledge" and params.get("action") == "list":
        tool_obj = tool_map.get("list_knowledge_files_tool")
        if tool_obj is None:
            text = "课件列表工具未注册，无法读取当前知识库状态。"
            return {"subagent_result": text, "final_answer": text}
        try:
            result = await tool_obj.ainvoke({"session_id": str(sid)})
            result_text = _safe_json_dumps(result) if isinstance(result, (dict, list)) else str(result)
            trace.append({"step": 1, "tool": "list_knowledge_files_tool", "args": {"session_id": str(sid)}, "observation": result_text[:1200]})
            structured = {
                "kind": "chat",
                "render_mode": "markdown",
                "text": result_text,
                "payload": {"ops_trace": trace},
                "meta": {"ops_react": True, "ops_steps": 1, "ops_deterministic": True},
            }
            return {"subagent_result": structured, "final_answer": result_text}
        except Exception as e:
            text = f"读取课件列表失败：{e}"
            structured = {
                "kind": "chat",
                "render_mode": "markdown",
                "text": text,
                "payload": {"ops_trace": trace},
                "meta": {"ops_react": True, "ops_steps": len(trace), "ops_deterministic": True},
            }
            return {"subagent_result": structured, "final_answer": text}

    direct_tool, direct_args = _resolve_ops_deterministic_tool(user_input, str(sid), limit)
    if direct_tool:
        logger.info(f"[Ops SubAgent] 命中确定性工具分支 tool={direct_tool}")
        tool_obj = tool_map.get(direct_tool)
        if tool_obj is None:
            text = f"管理工具未注册：{direct_tool}"
            return {"subagent_result": text, "final_answer": text}
        if direct_tool in OPS_DANGEROUS_TOOLS:
            required_phrase, arg_error = _ops_required_confirmation_phrase(direct_tool, direct_args)
            reason = arg_error or f"检测到高风险操作 `{direct_tool}`，需要二次确认后执行。"
            if arg_error or not _is_ops_confirmation_match(user_input, required_phrase, direct_tool):
                trace.append({"step": 1, "tool": direct_tool, "args": direct_args, "guard_blocked": True, "required_confirmation_phrase": required_phrase})
                structured = _build_ops_guard_reply(
                    trace=trace,
                    tool_name=direct_tool,
                    pending_args=direct_args,
                    required_phrase=required_phrase,
                    reason=reason,
                )
                structured["meta"]["ops_deterministic"] = True
                return {"subagent_result": structured, "final_answer": structured["text"]}
        try:
            result = await tool_obj.ainvoke(direct_args)
            result_text = _safe_json_dumps(result) if isinstance(result, (dict, list)) else str(result)
            trace.append({"step": 1, "tool": direct_tool, "args": direct_args, "observation": result_text[:1200]})
            structured = {
                "kind": "chat",
                "render_mode": "markdown",
                "text": result_text,
                "payload": {"ops_trace": trace},
                "meta": {"ops_react": True, "ops_steps": 1, "ops_deterministic": True},
            }
            return {"subagent_result": structured, "final_answer": result_text}
        except Exception as e:
            text = f"管理操作执行失败：`{direct_tool}`\n原因：{e}"
            structured = {
                "kind": "chat",
                "render_mode": "markdown",
                "text": text,
                "payload": {"ops_trace": trace},
                "meta": {"ops_react": True, "ops_steps": len(trace), "ops_deterministic": True},
            }
            return {"subagent_result": structured, "final_answer": text}

    # 对明确的历史管理动作走确定性工具调用，避免 LLM 口头宣告“已完成”。
    deterministic_action = parse_history_action(user_input)
    deterministic_tool, deterministic_args, deterministic_error = _resolve_ops_history_tool_call(
        action=deterministic_action,
        session_id=sid,
        limit=limit,
    )
    if deterministic_error:
        return {"subagent_result": deterministic_error, "final_answer": deterministic_error}
    if deterministic_tool:
        logger.info(
            f"[Ops SubAgent] 命中确定性历史动作 action={deterministic_action.get('action')} -> tool={deterministic_tool}"
        )
        tool_obj = tool_map.get(deterministic_tool)
        if tool_obj is None:
            text = f"管理工具未注册：{deterministic_tool}"
            return {"subagent_result": text, "final_answer": text}

        call_args = dict(deterministic_args)
        is_dangerous = deterministic_tool in OPS_DANGEROUS_TOOLS
        if is_dangerous:
            required_phrase, arg_error = _ops_required_confirmation_phrase(deterministic_tool, call_args)
            if arg_error:
                trace.append(
                    {
                        "step": 1,
                        "tool": deterministic_tool,
                        "args": call_args,
                        "guard_blocked": True,
                        "reason": arg_error,
                    }
                )
                structured = _build_ops_guard_reply(
                    trace=trace,
                    tool_name=deterministic_tool,
                    pending_args=call_args,
                    required_phrase=required_phrase,
                    reason=arg_error,
                )
                return {"subagent_result": structured, "final_answer": structured["text"]}
            if not _is_ops_confirmation_match(user_input, required_phrase, deterministic_tool):
                reason = f"检测到高风险操作 `{deterministic_tool}`，需要二次确认后执行。"
                trace.append(
                    {
                        "step": 1,
                        "tool": deterministic_tool,
                        "args": call_args,
                        "guard_blocked": True,
                        "required_confirmation_phrase": required_phrase,
                    }
                )
                structured = _build_ops_guard_reply(
                    trace=trace,
                    tool_name=deterministic_tool,
                    pending_args=call_args,
                    required_phrase=required_phrase,
                    reason=reason,
                )
                return {"subagent_result": structured, "final_answer": structured["text"]}

        try:
            result = await tool_obj.ainvoke(call_args)
            result_text = _safe_json_dumps(result) if isinstance(result, (dict, list)) else str(result)
            trace.append(
                {
                    "step": 1,
                    "tool": deterministic_tool,
                    "args": call_args,
                    "observation": result_text[:1200],
                }
            )
            structured = {
                "kind": "chat",
                "render_mode": "markdown",
                "text": result_text,
                "payload": {"ops_trace": trace},
                "meta": {"ops_react": True, "ops_steps": 1, "ops_deterministic": True},
            }
            return {"subagent_result": structured, "final_answer": result_text}
        except Exception as e:
            trace.append(
                {
                    "step": 1,
                    "tool": deterministic_tool,
                    "args": call_args,
                    "error": str(e),
                }
            )
            text = f"管理操作执行失败：`{deterministic_tool}`\n原因：{str(e)}"
            structured = {
                "kind": "chat",
                "render_mode": "markdown",
                "text": text,
                "payload": {"ops_trace": trace},
                "meta": {"ops_react": True, "ops_steps": 1, "ops_deterministic": True},
            }
            return {"subagent_result": structured, "final_answer": text}

    for step in range(max_steps):
        decision_raw = await _call_llm(
            OPS_REACT_DECISION_PROMPT,
            tool_specs=tool_specs,
            input=user_input,
            recent_history=recent_history,
            trace=_safe_json_dumps(trace),
        )
        decision = _extract_json(decision_raw)
        done = bool(decision.get("done", False))
        final_answer = str(decision.get("final_answer", "") or "").strip()
        tool_name = str(decision.get("tool_name", "") or "").strip()
        tool_args = decision.get("tool_args", {}) if isinstance(decision.get("tool_args", {}), dict) else {}

        # 模型有时会同时给 done=true 和 tool_name；此时以工具调用为准。
        if done and tool_name:
            done = False

        if done:
            claimed_success_without_tool = bool(
                final_answer and any(token in final_answer for token in ["已清空", "已删除", "已创建", "已重命名", "已执行"])
            )
            if not trace and (_ops_query_requires_tool(user_input) or claimed_success_without_tool):
                final_answer = "未执行任何管理工具，无法确认管理操作结果。请明确说明要查看/删除/清空的对象后重试。"
                trace.append(
                    {
                        "step": step + 1,
                        "guard_blocked": True,
                        "reason": "模型在未调用工具的情况下提前结束。",
                    }
                )
            if not final_answer:
                final_answer = "管理操作已结束，但未生成明确结果。请重试并给出更具体指令。"
            structured = {
                "kind": "chat",
                "render_mode": "markdown",
                "text": final_answer,
                "payload": {"ops_trace": trace},
                "meta": {"ops_react": True, "ops_steps": len(trace)},
            }
            return {"subagent_result": structured, "final_answer": final_answer}

        if not tool_name:
            trace.append({"step": step + 1, "error": "模型未返回 tool_name"})
            continue

        tool_obj = tool_map.get(tool_name)
        if tool_obj is None:
            trace.append({"step": step + 1, "tool": tool_name, "error": "工具不存在或不在白名单"})
            continue

        call_args = dict(tool_args)
        is_dangerous = tool_name in OPS_DANGEROUS_TOOLS
        # 高风险操作禁止自动补 session_id，避免误删放大。
        if (not is_dangerous) and "session_id" not in call_args and _tool_accepts_argument(tool_obj, "session_id"):
            call_args["session_id"] = sid

        if is_dangerous:
            required_phrase, arg_error = _ops_required_confirmation_phrase(tool_name, call_args)
            if arg_error:
                trace.append(
                    {
                        "step": step + 1,
                        "tool": tool_name,
                        "args": call_args,
                        "guard_blocked": True,
                        "reason": arg_error,
                    }
                )
                structured = _build_ops_guard_reply(
                    trace=trace,
                    tool_name=tool_name,
                    pending_args=call_args,
                    required_phrase=required_phrase,
                    reason=arg_error,
                )
                return {"subagent_result": structured, "final_answer": structured["text"]}
            if not _is_ops_confirmation_match(user_input, required_phrase, tool_name):
                reason = f"检测到高风险操作 `{tool_name}`，需要二次确认后执行。"
                trace.append(
                    {
                        "step": step + 1,
                        "tool": tool_name,
                        "args": call_args,
                        "guard_blocked": True,
                        "required_confirmation_phrase": required_phrase,
                    }
                )
                structured = _build_ops_guard_reply(
                    trace=trace,
                    tool_name=tool_name,
                    pending_args=call_args,
                    required_phrase=required_phrase,
                    reason=reason,
                )
                return {"subagent_result": structured, "final_answer": structured["text"]}

            # 删除会话默认不删知识库，仅在用户明确写“连同知识库”时才清理知识库。
            if tool_name == "delete_session_tool":
                call_args["purge_knowledge_base"] = bool("连同知识库" in user_input)

        try:
            result = await tool_obj.ainvoke(call_args)
            if isinstance(result, (dict, list)):
                result_text = _safe_json_dumps(result)
            else:
                result_text = str(result)
            trace.append(
                {
                    "step": step + 1,
                    "tool": tool_name,
                    "args": call_args,
                    "observation": result_text[:1200],
                }
            )
        except Exception as e:
            trace.append(
                {
                    "step": step + 1,
                    "tool": tool_name,
                    "args": call_args,
                    "error": str(e),
                }
            )

    if trace:
        last = trace[-1]
        if last.get("observation"):
            final_text = f"已执行管理操作：`{last.get('tool', 'unknown')}`\n\n{last.get('observation')}"
        else:
            last_tool = str(last.get("tool", "unknown") or "unknown")
            error_text = str(last.get("error", "未知错误") or "未知错误")
            error_lower = error_text.lower()
            is_timeout = ("timeout" in error_lower) or ("timed out" in error_lower) or ("超时" in error_text)
            if last_tool == "search_courseware" and is_timeout:
                final_text = (
                    "课件内容检索超时（本次已中止），并非“对象不明确”。\n"
                    f"失败工具：`{last_tool}`\n"
                    f"错误：{error_text}\n"
                    "建议改用更聚焦提问，例如：`总结 C0-Overview 的核心知识点`。"
                )
            else:
                final_text = (
                    f"管理操作尝试失败：`{last_tool}`\n"
                    f"原因：{error_text}\n"
                    "请补充更明确的对象（例如第几条/具体会话名）后再试。"
                )
    else:
        final_text = "未能解析出可执行的管理操作。请明确说明是查看、删除还是清空哪类记录。"

    structured = {
        "kind": "chat",
        "render_mode": "markdown",
        "text": final_text,
        "payload": {"ops_trace": trace},
        "meta": {"ops_react": True, "ops_steps": len(trace), "ops_fallback": True},
    }
    return {"subagent_result": structured, "final_answer": final_text}


def _extract_planner_constraints(user_input: str) -> Dict[str, Any]:
    """从自然语言中提取计划约束（周期、每日时长、目标模式）。"""
    text = str(user_input or "").strip()
    cycle_days = 0
    daily_minutes = 0
    target_mode = "balanced"

    day_match = re.search(r"(\d{1,2})\s*(?:天|日)(?:内|计划|冲刺|复习)?", text)
    week_match = re.search(r"(\d{1,2})\s*周(?:内|计划|冲刺|复习)?", text)
    if week_match:
        cycle_days = _safe_int(int(week_match.group(1)) * 7, default=7, minimum=1, maximum=30)
    elif day_match:
        cycle_days = _safe_int(day_match.group(1), default=7, minimum=1, maximum=30)

    min_match = re.search(r"(?:每天|每日)\s*(\d{2,3})\s*分钟", text) or re.search(r"(\d{2,3})\s*分钟\s*(?:每天|每日)", text)
    if min_match:
        daily_minutes = _safe_int(min_match.group(1), default=60, minimum=10, maximum=300)

    if any(token in text for token in ["冲刺", "临考", "考试前", "最后"]):
        target_mode = "sprint"
    elif any(token in text for token in ["查漏", "补缺", "薄弱", "错题"]):
        target_mode = "patch_weakness"

    return {
        "cycle_days": cycle_days,
        "daily_minutes": daily_minutes,
        "target_mode": target_mode,
    }


def _build_planner_practice_snapshot(session_id: str, history_limit: int = 200) -> Dict[str, Any]:
    """汇总当前会话练习画像，供 planner 做数据驱动决策。"""
    from utils.memory_service import memory_manager

    sid = str(session_id or "").strip() or "default"
    history = memory_manager.get_practice_history(sid, max(20, min(int(history_limit or 200), 500)))
    kp_stats = memory_manager.get_knowledge_point_stats(sid)
    mastery_priority = memory_manager.get_priority_review_points(sid, limit=8)

    total = len(history)
    correct = sum(1 for item in history if bool(item.get("is_correct")))
    accuracy = round((correct / total) * 100.0, 2) if total > 0 else 0.0
    recent = history[: min(20, total)]
    recent_correct = sum(1 for item in recent if bool(item.get("is_correct")))
    recent_accuracy = round((recent_correct / len(recent)) * 100.0, 2) if recent else 0.0

    weak_items = [
        (kp, data)
        for kp, data in kp_stats.items()
        if float(data.get("accuracy", 0.0)) < 60.0 and int(data.get("total", 0)) > 0
    ]
    weak_items.sort(key=lambda kv: (float(kv[1].get("accuracy", 0.0)), -int(kv[1].get("total", 0)), kv[0]))
    if mastery_priority:
        weak_points = [str(item.get("knowledge_point", "")).strip() for item in mastery_priority if str(item.get("knowledge_point", "")).strip()]
    else:
        weak_points = [kp for kp, _ in weak_items[:8]]
    weak_points = weak_points[:8]

    reason_counts: Dict[str, int] = {}
    for row in history:
        if bool(row.get("is_correct")):
            continue
        label = _classify_wrong_reason(str(row.get("wrong_reason") or ""))
        reason_counts[label] = int(reason_counts.get(label, 0)) + 1
    top_wrong_reasons = [f"{label}:{count}" for label, count in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]]

    mastery_view = [
        {
            "knowledge_point": item.get("knowledge_point", ""),
            "mastery_score": float(item.get("mastery_score", 0.0)),
            "review_urgency": float(item.get("review_urgency", 0.0)),
            "days_since_last_seen": float(item.get("days_since_last_seen", 0.0)),
        }
        for item in mastery_priority[:8]
    ]

    return {
        "session_id": sid,
        "total_attempts": total,
        "accuracy": accuracy,
        "recent_accuracy": recent_accuracy,
        "weak_points": weak_points,
        "mastery_priority": mastery_view,
        "top_wrong_reasons": top_wrong_reasons,
    }


def _render_planner_snapshot(snapshot: Dict[str, Any]) -> str:
    """将练习画像渲染为紧凑文本，降低模型理解成本。"""
    weak_points = snapshot.get("weak_points", []) if isinstance(snapshot.get("weak_points", []), list) else []
    mastery_priority = snapshot.get("mastery_priority", []) if isinstance(snapshot.get("mastery_priority", []), list) else []
    wrong_reasons = snapshot.get("top_wrong_reasons", []) if isinstance(snapshot.get("top_wrong_reasons", []), list) else []
    mastery_preview = []
    for item in mastery_priority[:3]:
        if not isinstance(item, dict):
            continue
        kp = str(item.get("knowledge_point", "")).strip()
        if not kp:
            continue
        mastery_preview.append(f"{kp}:{float(item.get('mastery_score', 0.0)):.2f}")
    lines = [
        f"- session_id: {snapshot.get('session_id', 'default')}",
        f"- total_attempts: {snapshot.get('total_attempts', 0)}",
        f"- accuracy: {snapshot.get('accuracy', 0.0)}%",
        f"- recent_accuracy: {snapshot.get('recent_accuracy', 0.0)}%",
        f"- weak_points: {', '.join(weak_points) if weak_points else 'none'}",
        f"- mastery_priority: {', '.join(mastery_preview) if mastery_preview else 'none'}",
        f"- top_wrong_reasons: {', '.join(wrong_reasons) if wrong_reasons else 'none'}",
    ]
    return "\n".join(lines)


def _build_default_day_tasks(focus: str) -> List[str]:
    """生成默认日任务，保证包含复习、练习、复盘三动作。"""
    item = str(focus or "").strip() or "核心主题"
    return [
        f"复习「{item}」核心概念并整理 1 页笔记（15 分钟）",
        f"让 DR 出 3 道「{item}」练习题并完成作答（30 分钟）",
        "复盘错因并记录 2 条改进点（10 分钟）",
    ]


def _build_default_planner_payload(user_input: str, snapshot: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
    """在模型输出异常时提供可执行的稳态计划。"""
    weak_points = snapshot.get("weak_points", []) if isinstance(snapshot.get("weak_points", []), list) else []
    hinted_days = _safe_int(constraints.get("cycle_days", 0), default=0, minimum=0, maximum=30)
    if hinted_days > 0:
        cycle_days = hinted_days
    elif len(weak_points) >= 6:
        cycle_days = 14
    elif len(weak_points) >= 3:
        cycle_days = 10
    else:
        cycle_days = 7

    hinted_minutes = _safe_int(constraints.get("daily_minutes", 0), default=0, minimum=0, maximum=300)
    if hinted_minutes > 0:
        daily_minutes = hinted_minutes
    else:
        daily_minutes = 80 if constraints.get("target_mode") == "sprint" else 60

    daily_plan: List[Dict[str, Any]] = []
    for day in range(1, cycle_days + 1):
        if day == 1:
            focus = weak_points[0] if weak_points else "基线诊断与知识框架梳理"
        elif day == cycle_days:
            focus = "综合模拟与错题回归"
        else:
            focus = weak_points[(day - 2) % len(weak_points)] if weak_points else f"核心主题 {day - 1}"
        daily_plan.append(
            {
                "day": day,
                "focus": focus,
                "tasks": _build_default_day_tasks(focus),
                "duration_min": daily_minutes,
            }
        )

    return {
        "title": "数据驱动复习计划",
        "goal": "优先修复薄弱知识点并稳定提升正确率",
        "cycle_days": cycle_days,
        "daily_plan": daily_plan,
        "milestones": [
            "完成 Day1 基线诊断并确认薄弱点优先级",
            f"在周期中段将薄弱点正确率提升到 60%+（当前 {snapshot.get('accuracy', 0.0)}%）",
            "完成综合模拟并输出最终错因清单",
        ],
        "review_strategy": "采用 1-3-7 复盘节奏：当天、3天后、7天后各做一次同类回顾。",
        "next_action": f"先执行 Day1：{daily_plan[0]['tasks'][0]}",
    }


def _inject_weak_points_into_plan(plan: Dict[str, Any], weak_points: List[str]) -> Dict[str, Any]:
    """把高优先级薄弱点注入计划前几天，避免计划泛化。"""
    if not weak_points:
        return plan
    updated = dict(plan)
    raw_daily = updated.get("daily_plan", []) if isinstance(updated.get("daily_plan", []), list) else []
    daily_plan: List[Dict[str, Any]] = []
    for row in raw_daily:
        daily_plan.append(dict(row) if isinstance(row, dict) else {})
    if not daily_plan:
        return updated

    for idx, kp in enumerate(weak_points[: min(3, len(daily_plan))]):
        row = dict(daily_plan[idx])
        row["focus"] = kp
        tasks = row.get("tasks", []) if isinstance(row.get("tasks", []), list) else []
        task_list = [str(item).strip() for item in tasks if str(item).strip()]
        if not any(kp in item for item in task_list):
            task_list = [f"复习「{kp}」核心概念并做 3 道同类题"] + task_list
        row["tasks"] = task_list[:4] if task_list else _build_default_day_tasks(kp)
        daily_plan[idx] = row

    updated["daily_plan"] = daily_plan
    return updated


def _normalize_planner_payload(
    parsed: Dict[str, Any],
    fallback_text: str,
    *,
    cycle_days_hint: int = 0,
    daily_minutes_hint: int = 0,
    weak_points: List[str] | None = None,
) -> Dict[str, Any]:
    """规范化 planner 结构，保证字段完整并限制异常值。"""
    weak_points = [str(item).strip() for item in (weak_points or []) if str(item).strip()]
    title = str(parsed.get("title", "") or "复习计划").strip() or "复习计划"
    goal = str(parsed.get("goal", "") or "完成本轮复习目标").strip() or "完成本轮复习目标"
    hinted_days = _safe_int(cycle_days_hint, default=0, minimum=0, maximum=30)
    cycle_default = hinted_days if hinted_days > 0 else 7
    cycle_days = _safe_int(parsed.get("cycle_days", cycle_default), default=cycle_default, minimum=1, maximum=30)
    hinted_minutes = _safe_int(daily_minutes_hint, default=0, minimum=0, maximum=300)
    duration_default = hinted_minutes if hinted_minutes > 0 else 60

    raw_daily = parsed.get("daily_plan", []) if isinstance(parsed.get("daily_plan", []), list) else []
    day_map: Dict[int, Dict[str, Any]] = {}
    for idx, row in enumerate(raw_daily, start=1):
        if not isinstance(row, dict):
            continue
        day = _safe_int(row.get("day", idx), default=idx, minimum=1, maximum=30)
        if day > cycle_days or day in day_map:
            continue
        focus = str(row.get("focus", "") or "").strip() or f"第{day}天复习"
        raw_tasks = row.get("tasks", [])
        if isinstance(raw_tasks, list):
            tasks = [str(item).strip() for item in raw_tasks if str(item).strip()]
        else:
            tasks = [str(raw_tasks).strip()] if str(raw_tasks).strip() else []
        duration = _safe_int(row.get("duration_min", duration_default), default=duration_default, minimum=10, maximum=300)
        day_map[day] = {"day": day, "focus": focus, "tasks": tasks or _build_default_day_tasks(focus), "duration_min": duration}

    daily_plan: List[Dict[str, Any]] = []
    for day in range(1, cycle_days + 1):
        if day in day_map:
            daily_plan.append(day_map[day])
            continue
        focus = weak_points[(day - 1) % len(weak_points)] if weak_points else f"第{day}天复习"
        daily_plan.append(
            {
                "day": day,
                "focus": focus,
                "tasks": _build_default_day_tasks(focus),
                "duration_min": duration_default,
            }
        )

    milestones_raw = parsed.get("milestones", [])
    milestones = [str(item).strip() for item in milestones_raw if str(item).strip()] if isinstance(milestones_raw, list) else []
    if not milestones:
        milestones = ["完成基线诊断", "中期薄弱点纠偏", "完成综合复盘"]
    review_strategy = str(parsed.get("review_strategy", "") or "每日结束后做 5 分钟错因复盘。").strip()
    next_action = str(parsed.get("next_action", "") or "先完成第 1 天第 1 个任务。").strip()
    if not next_action:
        next_action = "先完成第 1 天第 1 个任务。"

    if not daily_plan:
        daily_plan = [{"day": 1, "focus": "核心概念梳理", "tasks": [fallback_text[:80] or "完成核心知识点复习"], "duration_min": duration_default}]

    return {
        "title": title,
        "goal": goal,
        "cycle_days": cycle_days,
        "daily_plan": daily_plan,
        "milestones": milestones[:6],
        "review_strategy": review_strategy,
        "next_action": next_action,
    }


def _planner_requires_wrong_basis(user_input: str) -> bool:
    """判断计划是否必须显式声明基于错题/薄弱点。"""
    text = str(user_input or "")
    return any(token in text for token in ["最近错题", "错题", "薄弱点", "练习记录", "做题记录"])


def _render_planner_markdown(plan: Dict[str, Any], *, basis_text: str = "") -> str:
    """将结构化计划渲染为用户可读 Markdown。"""
    lines = [
        f"## {plan.get('title', '复习计划')}",
        f"- 目标：{plan.get('goal', '')}",
        f"- 周期：{plan.get('cycle_days', 7)}天",
    ]
    if basis_text:
        lines.append(f"- 依据：{basis_text}")
    lines.extend(["", "### 每日安排"])
    for row in plan.get("daily_plan", []):
        day = int(row.get("day", 0) or 0)
        focus = str(row.get("focus", "") or "")
        duration = int(row.get("duration_min", 60) or 60)
        tasks = row.get("tasks", []) if isinstance(row.get("tasks", []), list) else []
        task_text = "；".join(str(item) for item in tasks if str(item).strip()) or "完成当日练习"
        lines.append(f"- Day {day}（{duration} 分钟）：{focus}；{task_text}")

    milestones = plan.get("milestones", []) if isinstance(plan.get("milestones", []), list) else []
    if milestones:
        lines.extend(["", "### 里程碑"])
        for item in milestones:
            lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "### 复盘策略",
            f"- {plan.get('review_strategy', '')}",
            "",
            "### 现在就做",
            f"- {plan.get('next_action', '')}",
        ]
    )
    return "\n".join(lines).strip()


def _extract_learning_loop_evidence(context: str, limit: int = 3) -> List[Dict[str, Any]]:
    """从 RAG context 中提炼前端证据卡，供学习闭环展示课件依据。"""
    text = str(context or "")
    if not text:
        return []
    cards: List[Dict[str, Any]] = []
    seen = set()
    pattern = r"\[参考资料\d+\]:参考资料:(.*?)\|参考元数据:(\{.*?\})(?:\n|$)"
    for match in re.finditer(pattern, text, flags=re.DOTALL):
        quote = re.sub(r"\s+", " ", str(match.group(1) or "")).strip()[:140]
        metadata_text = str(match.group(2) or "").strip()
        source = "课件片段"
        page = None
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(metadata_text)
            except Exception:
                continue
            if isinstance(parsed, dict):
                source = str(parsed.get("source_filename") or parsed.get("source") or source).strip() or source
                page = parsed.get("page")
                break
        key = f"{source}|{page}|{quote}"
        if not quote or key in seen:
            continue
        seen.add(key)
        card: Dict[str, Any] = {"source": source, "quote": quote, "grounded": True}
        if page not in (None, ""):
            card["page"] = page
        cards.append(card)
        if len(cards) >= max(1, min(int(limit or 3), 8)):
            break
    if cards:
        return cards

    simple_pattern = r"\[source=(?P<source>[^\]\s]+)(?:\s+page=(?P<page>[^\]]+))?\]\s*(?P<quote>.*?)(?=\n\[source=|\Z)"
    for match in re.finditer(simple_pattern, text, flags=re.DOTALL):
        source = str(match.group("source") or "课件片段").strip()
        page = str(match.group("page") or "").strip()
        quote = re.sub(r"\s+", " ", str(match.group("quote") or "")).strip()[:140]
        key = f"{source}|{page}|{quote}"
        if not quote or key in seen:
            continue
        seen.add(key)
        card = {"source": source, "quote": quote, "grounded": True}
        if page:
            card["page"] = page
        cards.append(card)
        if len(cards) >= max(1, min(int(limit or 3), 8)):
            break
    if cards:
        return cards

    for match in re.finditer(r"(?P<source>[^：:\n]{1,80}\.(?:pdf|pptx?|docx?|txt))[:：]\s*(?P<quote>[^\n]{10,})", text, flags=re.IGNORECASE):
        source = str(match.group("source") or "课件片段").strip()
        quote = re.sub(r"\s+", " ", str(match.group("quote") or "")).strip()[:140]
        key = f"{source}|{quote}"
        if key in seen:
            continue
        seen.add(key)
        cards.append({"source": source, "quote": quote, "grounded": True})
        if len(cards) >= max(1, min(int(limit or 3), 8)):
            break
    return cards


def _learning_loop_focus_points(user_input: str, snapshot: Dict[str, Any]) -> List[str]:
    """根据真实画像和用户输入选择闭环优先考点。"""
    priority = snapshot.get("mastery_priority", []) if isinstance(snapshot.get("mastery_priority", []), list) else []
    points = [
        str(item.get("knowledge_point", "")).strip()
        for item in priority
        if isinstance(item, dict) and str(item.get("knowledge_point", "")).strip()
    ]
    if points:
        return list(dict.fromkeys(points))[:5]

    weak_points = snapshot.get("weak_points", []) if isinstance(snapshot.get("weak_points", []), list) else []
    points = [str(item).strip() for item in weak_points if str(item).strip()]
    if points:
        return list(dict.fromkeys(points))[:5]

    text = str(user_input or "")
    topic_matches = re.findall(r"(进程|线程|内存|虚拟内存|死锁|调度|文件系统|I/O|同步|信号量|TLB)", text, flags=re.IGNORECASE)
    normalized = [str(item).strip() for item in topic_matches if str(item).strip()]
    if normalized:
        return list(dict.fromkeys(normalized))[:5]

    return ["基线诊断与知识框架梳理", "课件核心概念复习", "综合练习与错因复盘"]


def _build_learning_loop_plan(
    user_input: str,
    snapshot: Dict[str, Any],
    evidence_cards: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """构建确定性学习闭环计划，保证空画像和每日任务约束稳定。"""
    constraints = _extract_planner_constraints(user_input)
    cycle_days = _safe_int(constraints.get("cycle_days", 0), default=3, minimum=1, maximum=7)
    if not constraints.get("cycle_days"):
        cycle_days = 3
    daily_minutes = _safe_int(constraints.get("daily_minutes", 0), default=60, minimum=20, maximum=180)
    if not constraints.get("daily_minutes"):
        daily_minutes = 60

    total_attempts = int(snapshot.get("total_attempts", 0) or 0)
    has_practice_profile = total_attempts > 0
    focus_points = _learning_loop_focus_points(user_input, snapshot)

    daily_plan: List[Dict[str, Any]] = []
    for day in range(1, cycle_days + 1):
        focus = focus_points[(day - 1) % len(focus_points)]
        daily_plan.append(
            {
                "day": day,
                "focus": focus,
                "tasks": [
                    f"复习「{focus}」对应课件要点，整理 3 条关键词",
                    f"练习「{focus}」选择题 3 道，完成后立即提交作答",
                    "复盘错因：记录错误类型、正确依据和下一次避免策略",
                ],
                "duration_min": daily_minutes,
            }
        )

    day1_focus = daily_plan[0]["focus"]
    basis_text = (
        f"基于真实练习画像（累计 {total_attempts} 题，正确率 {snapshot.get('accuracy', 0.0)}%）"
        if has_practice_profile
        else "暂无错题画像，先用课件结构做基线诊断"
    )
    return {
        "title": "自主学习闭环",
        "basis": basis_text,
        "has_practice_profile": has_practice_profile,
        "cycle_days": cycle_days,
        "daily_plan": daily_plan,
        "day1_quiz_request": {
            "topic": day1_focus,
            "quiz_type": "选择题",
            "num": 3,
            "prompt": f"请针对「{day1_focus}」出3道选择题，我作答后请判分并记录错因。",
        },
        "next_action": f"先执行 Day1：复习「{day1_focus}」10-15 分钟，然后让 DR 出 3 道选择题。",
        "evidence_count": len(evidence_cards),
    }


def _critic_learning_loop_plan(plan: Dict[str, Any], snapshot: Dict[str, Any]) -> tuple[Dict[str, Any], List[str], str]:
    """校验并修复学习闭环计划，返回修复后的计划、问题列表和状态。"""
    findings: List[str] = []
    partial = False
    total_attempts = int(snapshot.get("total_attempts", 0) or 0)
    if total_attempts <= 0 and "最近错题" in str(plan.get("basis", "")):
        findings.append("空画像不能声称基于最近错题。")
        plan = {**plan, "basis": "暂无错题画像，先用课件结构做基线诊断", "has_practice_profile": False}

    fixed_daily: List[Dict[str, Any]] = []
    raw_daily = plan.get("daily_plan", []) if isinstance(plan.get("daily_plan", []), list) else []
    for row in raw_daily:
        item = dict(row) if isinstance(row, dict) else {}
        focus = str(item.get("focus") or "核心主题").strip()
        tasks = [str(task).strip() for task in item.get("tasks", []) if str(task).strip()] if isinstance(item.get("tasks", []), list) else []
        joined = " ".join(tasks)
        required_tasks = {
            "复习": f"复习「{focus}」核心概念并整理关键词",
            "练习": f"练习「{focus}」选择题 3 道并提交作答",
            "复盘": "复盘错因并写下下一步纠偏动作",
        }
        for key, fallback in required_tasks.items():
            if key not in joined:
                findings.append(f"Day {item.get('day', '?')} 缺少{key}任务，已补齐。")
                tasks.append(fallback)
        item["tasks"] = tasks
        fixed_daily.append(item)
    if not fixed_daily:
        partial = True
        findings.append("每日计划为空，已降级为确定性模板。")
        fallback_focus = "基线诊断与知识框架梳理" if total_attempts <= 0 else "最低掌握度考点"
        fixed_daily = [
            {
                "day": 1,
                "focus": fallback_focus,
                "tasks": [
                    f"复习「{fallback_focus}」核心概念并整理关键词",
                    f"练习「{fallback_focus}」选择题 3 道并提交作答",
                    "复盘错因并写下下一步纠偏动作",
                ],
                "duration_min": 60,
            }
        ]
        plan = {
            **plan,
            "title": plan.get("title") or "自主学习闭环",
            "basis": plan.get("basis") or ("暂无错题画像，先用课件结构做基线诊断" if total_attempts <= 0 else "基于真实练习画像"),
            "cycle_days": 1,
            "critic_degrade_reason": "daily_plan_empty",
        }
    plan = {**plan, "daily_plan": fixed_daily}

    day1 = plan.get("day1_quiz_request") if isinstance(plan.get("day1_quiz_request"), dict) else {}
    if not str(day1.get("prompt", "")).strip():
        findings.append("缺少 Day1 练习建议，已补齐。")
        focus = str((fixed_daily[0] if fixed_daily else {}).get("focus") or "核心主题")
        plan["day1_quiz_request"] = {
            "topic": focus,
            "quiz_type": "选择题",
            "num": 3,
            "prompt": f"请针对「{focus}」出3道选择题，我作答后请判分并记录错因。",
        }

    if not str(plan.get("next_action", "")).strip():
        focus = str((fixed_daily[0] if fixed_daily else {}).get("focus") or "核心主题")
        findings.append("缺少下一步动作，已补齐。")
        plan["next_action"] = f"先执行 Day1：复习「{focus}」后完成 3 道练习并复盘。"

    return plan, findings, ("partial" if partial else "repaired" if findings else "pass")


def _render_learning_loop_markdown(plan: Dict[str, Any], critic_findings: List[str], evidence_cards: List[Dict[str, Any]]) -> str:
    """将学习闭环结构渲染为用户可读 Markdown。"""
    lines = [
        f"## {plan.get('title', '自主学习闭环')}",
        f"- 依据：{plan.get('basis', '')}",
        f"- 周期：{plan.get('cycle_days', 3)}天",
    ]
    if evidence_cards:
        sources = []
        for card in evidence_cards[:3]:
            label = str(card.get("source") or "课件片段")
            page = card.get("page")
            sources.append(f"{label}{f' p.{page}' if page not in (None, '') else ''}")
        lines.append(f"- 课件证据：{'；'.join(sources)}")
    else:
        lines.append("- 课件证据：本轮未检索到稳定片段，先按画像/基线任务执行。")

    lines.extend(["", "### Day1 今日闭环"])
    first = (plan.get("daily_plan", [{}]) or [{}])[0]
    first_tasks = first.get("tasks", []) if isinstance(first.get("tasks", []), list) else []
    for task in first_tasks:
        lines.append(f"- {task}")

    lines.extend(["", "### 短期安排"])
    rendered_daily = plan.get("daily_plan", []) if isinstance(plan.get("daily_plan", []), list) else []
    for row in rendered_daily:
        tasks = "；".join(str(task) for task in row.get("tasks", []) if str(task).strip()) if isinstance(row.get("tasks", []), list) else ""
        lines.append(f"- Day {row.get('day')}（{row.get('duration_min', 60)} 分钟）：{row.get('focus')}；{tasks}")

    quiz_req = plan.get("day1_quiz_request", {}) if isinstance(plan.get("day1_quiz_request", {}), dict) else {}
    lines.extend(
        [
            "",
            "### Day1 练习建议",
            f"- {quiz_req.get('prompt', '')}",
            "",
            "### 下一步",
            f"- {plan.get('next_action', '')}",
        ]
    )
    if critic_findings:
        lines.extend(["", "### 系统校验"])
        lines.append("- 已自动修复计划格式，确保每天包含复习、练习、复盘。")
    return "\n".join(lines).strip()


def _build_learning_loop_state(
    session_id: str,
    snapshot: Dict[str, Any],
    focus_points: List[str],
    plan: Dict[str, Any],
    evidence_cards: List[Dict[str, Any]],
    critic_status: str,
) -> Dict[str, Any]:
    """构造可持久化的学习闭环状态。"""
    now_ms = int(time.time() * 1000)
    return {
        "agent": "learning_loop",
        "status": "active",
        "phase": "planned",
        "session_id": str(session_id or "default"),
        "current_day": 1,
        "cycle_days": int(plan.get("cycle_days", 3) or 3),
        "snapshot": snapshot,
        "priority_points": focus_points,
        "daily_plan": plan.get("daily_plan", []),
        "day1_quiz_request": plan.get("day1_quiz_request", {}),
        "next_action": plan.get("next_action", ""),
        "critic_status": critic_status,
        "evidence_count": len(evidence_cards),
        "created_at": now_ms,
        "updated_at": now_ms,
    }


def _learning_loop_quiz_route_params(loop_state: Dict[str, Any]) -> Dict[str, Any]:
    """从闭环状态中提取 QuizAgent 可执行参数。"""
    req = loop_state.get("day1_quiz_request") if isinstance(loop_state.get("day1_quiz_request"), dict) else {}
    topic = str(req.get("topic") or "").strip()
    if not topic:
        daily = loop_state.get("daily_plan") if isinstance(loop_state.get("daily_plan"), list) else []
        topic = str((daily[0] if daily and isinstance(daily[0], dict) else {}).get("focus") or "核心主题").strip()
    return {
        "topic": topic or "核心主题",
        "quiz_type": str(req.get("quiz_type") or "选择题"),
        "num": _safe_int(req.get("num", 3), default=3, minimum=1, maximum=10),
        "source": "learning_loop",
    }


def _learning_loop_next_quiz_plan(
    loop_state: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    """根据复盘结果决定继续闭环时的下一张题卡。"""
    now_ms = int(time.time() * 1000)
    base_params = _learning_loop_quiz_route_params(loop_state)
    summary = loop_state.get("last_practice_summary") if isinstance(loop_state.get("last_practice_summary"), dict) else {}
    wrong_points = [
        str(item).strip()
        for item in summary.get("wrong_points", [])
        if str(item).strip()
    ] if isinstance(summary.get("wrong_points", []), list) else []

    if wrong_points:
        topic = wrong_points[0]
        params = {
            **base_params,
            "topic": topic,
            "num": 1,
            "source": "learning_loop_remediation",
        }
        updated = {
            **loop_state,
            "phase": "remediation_quiz_issued",
            "last_quiz_params": params,
            "next_action": f"完成「{topic}」补救题并提交，系统会继续复盘错因是否消除。",
            "updated_at": now_ms,
        }
        return params, updated, f"生成错因补救题：{topic}"

    current_day = _safe_int(loop_state.get("current_day", 1), default=1, minimum=1, maximum=30)
    cycle_days = _safe_int(loop_state.get("cycle_days", 3), default=3, minimum=1, maximum=30)
    next_day = min(current_day + 1, cycle_days)
    priorities = [
        str(item).strip()
        for item in loop_state.get("priority_points", [])
        if str(item).strip()
    ] if isinstance(loop_state.get("priority_points", []), list) else []
    daily_plan = loop_state.get("daily_plan") if isinstance(loop_state.get("daily_plan"), list) else []
    daily_focus = ""
    if len(daily_plan) >= next_day and isinstance(daily_plan[next_day - 1], dict):
        daily_focus = str(daily_plan[next_day - 1].get("focus") or "").strip()
    topic = daily_focus or (priorities[next_day - 1] if len(priorities) >= next_day else "") or (priorities[0] if priorities else "")
    if not topic:
        mastery = snapshot.get("mastery_priority") if isinstance(snapshot.get("mastery_priority"), list) else []
        if mastery and isinstance(mastery[0], dict):
            topic = str(mastery[0].get("knowledge_point") or "").strip()
    topic = topic or str(base_params.get("topic") or "核心主题").strip()

    params = {
        **base_params,
        "topic": topic,
        "num": max(1, _safe_int(base_params.get("num", 2), default=2, minimum=1, maximum=10)),
        "source": "learning_loop_next_day",
    }
    updated = {
        **loop_state,
        "phase": f"day{next_day}_quiz_issued",
        "current_day": next_day,
        "snapshot": snapshot or loop_state.get("snapshot", {}),
        "last_quiz_params": params,
        "next_action": f"完成 Day{next_day}「{topic}」进阶练习并提交，系统会继续自动复盘。",
        "updated_at": now_ms,
    }
    return params, updated, f"生成 Day{next_day} 进阶练习：{topic}"


def _merge_learning_loop_quiz_result(
    quiz_result: Dict[str, Any],
    loop_state: Dict[str, Any],
    agent_trace: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """把 QuizAgent 题卡包装成学习闭环推进结果。"""
    structured = dict(quiz_result) if isinstance(quiz_result, dict) else {}
    payload = structured.get("payload") if isinstance(structured.get("payload"), dict) else {}
    meta = structured.get("meta") if isinstance(structured.get("meta"), dict) else {}
    payload = {
        **payload,
        "learning_loop_data": {
            "snapshot": loop_state.get("snapshot", {}),
            "priority_points": loop_state.get("priority_points", []),
            "daily_plan": loop_state.get("daily_plan", []),
            "day1_quiz_request": loop_state.get("day1_quiz_request", {}),
            "next_action": "完成这组题并提交作答，系统会刷新画像后继续复盘。",
            "loop_state": loop_state,
        },
        "agent_trace": agent_trace,
    }
    structured["payload"] = payload
    structured["meta"] = {
        **meta,
        "agent_mode": "learning_loop",
        "learning_loop_structured": True,
        "learning_loop_stateful": True,
        "learning_loop_phase": loop_state.get("phase", "day1_quiz_issued"),
        "planner_data_driven": True,
        "critic_status": loop_state.get("critic_status") or "pass",
        "agent_trace": agent_trace,
    }
    return structured


def _render_learning_loop_review_markdown(
    loop_state: Dict[str, Any],
    snapshot: Dict[str, Any],
    agent_trace: List[Dict[str, Any]],
) -> tuple[str, Dict[str, Any]]:
    """渲染作答后的闭环复盘，并返回更新后的状态。"""
    current_phase = str(loop_state.get("phase") or "")
    day_match = re.fullmatch(r"day(\d+)_answered", current_phase)
    review_phase = f"day{day_match.group(1)}_reviewed" if day_match else (
        "remediation_reviewed" if current_phase == "remediation_answered" else "day1_reviewed"
    )
    review_title = f"Day{day_match.group(1)} 作答复盘" if day_match else (
        "补救练习作答复盘" if current_phase == "remediation_answered" else "Day1 作答复盘"
    )
    summary = loop_state.get("last_practice_summary") if isinstance(loop_state.get("last_practice_summary"), dict) else {}
    total = _safe_int(summary.get("total", 0), default=0, minimum=0, maximum=200)
    correct = _safe_int(summary.get("correct", 0), default=0, minimum=0, maximum=200)
    wrong_count = _safe_int(summary.get("wrong_count", max(0, total - correct)), default=0, minimum=0, maximum=200)
    accuracy = float(summary.get("accuracy", round((correct / total) * 100.0, 1) if total else 0.0) or 0.0)
    wrong_points = [str(item).strip() for item in summary.get("wrong_points", []) if str(item).strip()] if isinstance(summary.get("wrong_points", []), list) else []
    wrong_reasons = [str(item).strip() for item in summary.get("wrong_reasons", []) if str(item).strip()] if isinstance(summary.get("wrong_reasons", []), list) else []

    def _clip_reason(reason: str, max_len: int = 180) -> str:
        """压缩复盘错因，避免长解析撑爆学习闭环消息。"""
        compact = re.sub(r"\s+", " ", str(reason or "")).strip()
        if len(compact) <= max_len:
            return compact
        return compact[:max_len].rstrip() + "..."

    if wrong_points:
        next_action = f"先复习「{wrong_points[0]}」的课件要点，再继续闭环生成 1 道补救题。"
        review_decision = "remedial_practice"
        completion_status = "active"
    elif total > 0:
        current_day = _safe_int(loop_state.get("current_day", 1), default=1, minimum=1, maximum=30)
        cycle_days = _safe_int(loop_state.get("cycle_days", 3), default=3, minimum=1, maximum=30)
        if current_day < cycle_days:
            next_action = f"本轮全对，可以进入 Day{current_day + 1} 或提高难度继续练习。"
            completion_status = "active"
        else:
            next_action = "本轮全对，已完成当前周期，可以提高难度继续练习或重新生成新周期。"
            review_phase = "cycle_completed"
            completion_status = "completed"
        review_decision = "advance_or_raise_difficulty"
    else:
        next_action = "还没有检测到本轮作答记录，请先完成题卡并提交答案。"
        review_decision = "await_answer"
        completion_status = "active"

    lines = [
        f"## {review_title}",
        f"- 本次结果：{correct}/{total}，正确率 {accuracy:.1f}%",
        f"- 最新画像：累计 {snapshot.get('total_attempts', 0)} 题，整体正确率 {snapshot.get('accuracy', 0.0)}%",
    ]
    if wrong_points:
        lines.append(f"- 错题考点：{'、'.join(wrong_points[:5])}")
    else:
        lines.append("- 错题考点：本轮未发现错题")
    if wrong_reasons:
        clipped_reasons = [_clip_reason(reason) for reason in wrong_reasons[:3]]
        lines.append(f"- 错因线索：{'；'.join(clipped_reasons)}")
    lines.extend(
        [
            "",
            "### 下一步",
            f"- {next_action}",
        ]
    )

    updated_state = {
        **loop_state,
        "status": completion_status,
        "phase": review_phase,
        "review_decision": review_decision,
        "next_action": next_action,
        "snapshot": snapshot,
        "updated_at": int(time.time() * 1000),
    }
    agent_trace.append(
        {
            "task_id": "critic.review",
            "agent": "CriticAgent",
            "status": "success",
            "summary": f"完成作答复盘：wrong={wrong_count}, decision={review_decision}",
        }
    )
    return "\n".join(lines).strip(), updated_state


async def planner_subagent_node(state: SupervisorState) -> SupervisorState:
    """Planner SubAgent: 制定结构化复习计划并落入消息 payload。"""
    logger.info("[Planner SubAgent] 生成复习计划...")

    sid = state.get("session_id") or current_session_id.get() or "default"
    params = state.get("route_params", {}) if isinstance(state.get("route_params", {}), dict) else {}
    if params.get("action") == "context_reference":
        structured = _build_context_reference_reply(str(sid), state.get("input", ""))
        return {"subagent_result": structured, "final_answer": structured["text"]}

    constraints = _extract_planner_constraints(state.get("input", ""))
    snapshot = _build_planner_practice_snapshot(sid)
    snapshot_text = _render_planner_snapshot(snapshot)
    observation_snapshot = _format_orchestrator_observations(
        state,
        {"get_practice_stats_tool", "get_weak_points_tool", "get_practice_history"},
        title="ReAct练习画像观察",
    )
    if observation_snapshot:
        snapshot_text = f"{snapshot_text}\n\n{observation_snapshot}"
    default_plan = _build_default_planner_payload(state.get("input", ""), snapshot, constraints)

    result_raw = ""
    parsed: Dict[str, Any] = {}
    try:
        result_raw = await _call_llm(
            PLANNER_SUBAGENT_PROMPT,
            memory_context=state.get('memory_context', ''),
            practice_snapshot=snapshot_text,
            planner_constraints=_safe_json_dumps(constraints),
            input=state['input'],
        )
        extracted = _extract_json(result_raw)
        if isinstance(extracted, dict):
            parsed = extracted
    except Exception as e:
        logger.warning(f"[Planner SubAgent] LLM 生成失败，回退默认计划: {e}")
        result_raw = str(e)

    if parsed:
        plan = _normalize_planner_payload(
            parsed,
            result_raw,
            cycle_days_hint=default_plan.get("cycle_days", 7),
            daily_minutes_hint=(default_plan.get("daily_plan", [{}])[0] or {}).get("duration_min", 60),
            weak_points=snapshot.get("weak_points", []),
        )
        plan = _inject_weak_points_into_plan(plan, snapshot.get("weak_points", []))
    else:
        plan = default_plan

    if not str(plan.get("next_action", "")).strip():
        first_tasks = (plan.get("daily_plan", [{}])[0] or {}).get("tasks", [])
        if isinstance(first_tasks, list) and first_tasks:
            plan["next_action"] = str(first_tasks[0])
        else:
            plan["next_action"] = default_plan.get("next_action", "先完成第 1 天第 1 个任务。")

    if constraints.get("cycle_days") == 3 and "3天" not in str(plan.get("title", "")):
        plan["title"] = f"3天复习计划：{str(plan.get('title') or '薄弱点突破')}"
    if "复习" not in str(plan.get("goal", "")):
        plan["goal"] = f"复习并巩固：{str(plan.get('goal') or '薄弱知识点')}"
    basis_text = ""
    if _planner_requires_wrong_basis(state.get("input", "")):
        weak_points = snapshot.get("weak_points", []) if isinstance(snapshot.get("weak_points", []), list) else []
        basis_text = "最近错题/薄弱点"
        if weak_points:
            basis_text = f"最近错题/薄弱点（优先：{'、'.join(str(kp) for kp in weak_points[:3])}）"

    plan_text = _render_planner_markdown(plan, basis_text=basis_text)
    structured = {
        "kind": "chat",
        "render_mode": "markdown",
        "text": plan_text,
        "payload": {
            "planner_data": plan,
            "planner_constraints": constraints,
            "planner_snapshot": snapshot,
        },
        "meta": {"planner_structured": True, "planner_data_driven": True},
    }
    return {"subagent_result": structured, "final_answer": plan_text}


async def learning_loop_subagent_node(state: SupervisorState) -> SupervisorState:
    """LearningLoopAgent: 画像、课件、计划、练习建议和 critic 的轻量闭环。"""
    logger.info("[LearningLoopAgent] 启动自主学习闭环...")
    import asyncio
    from utils.memory_service import memory_manager

    sid = state.get("session_id") or current_session_id.get() or "default"
    current_session_id.set(str(sid))

    agent_trace: List[Dict[str, Any]] = []
    persisted_state = memory_manager.get_agent_state(str(sid), "learning_loop")
    persisted_phase = str(persisted_state.get("phase") or "")
    if (
        persisted_state.get("status") == "active"
        and (re.fullmatch(r"day\d+_answered", persisted_phase) or persisted_phase == "remediation_answered")
        and (_is_learning_loop_review_intent(state.get("input", "")) or _is_learning_loop_continue_intent(state.get("input", "")))
    ):
        agent_trace.append(
            {
                "task_id": "loop.state",
                "agent": "LearningLoopAgent",
                "status": "success",
                "summary": f"读取已提交作答的闭环状态，进入复盘：phase={persisted_phase}",
            }
        )
        snapshot = _build_planner_practice_snapshot(str(sid))
        agent_trace.append(
            {
                "task_id": "history.snapshot",
                "agent": "HistoryAgent",
                "status": "success",
                "summary": f"刷新练习画像：{snapshot.get('total_attempts', 0)} 题，正确率 {snapshot.get('accuracy', 0.0)}%",
            }
        )
        answer, reviewed_state = _render_learning_loop_review_markdown(persisted_state, snapshot, agent_trace)
        memory_manager.set_agent_state(str(sid), "learning_loop", reviewed_state)
        structured = {
            "kind": "chat",
            "render_mode": "markdown",
            "text": answer,
            "payload": {
                "learning_loop_data": {
                    "snapshot": snapshot,
                    "priority_points": reviewed_state.get("priority_points", []),
                    "daily_plan": reviewed_state.get("daily_plan", []),
                    "day1_quiz_request": reviewed_state.get("day1_quiz_request", {}),
                    "next_action": reviewed_state.get("next_action", ""),
                    "loop_state": reviewed_state,
                    "last_practice_summary": reviewed_state.get("last_practice_summary", {}),
                },
                "agent_trace": agent_trace,
            },
            "meta": {
                "agent_mode": "learning_loop",
                "learning_loop_structured": True,
                "learning_loop_stateful": True,
                "learning_loop_phase": reviewed_state.get("phase", "day1_reviewed"),
                "planner_data_driven": True,
                "critic_status": reviewed_state.get("critic_status") or "pass",
                "agent_trace": agent_trace,
            },
        }
        return {"subagent_result": structured, "final_answer": answer}

    if _is_learning_loop_continue_intent(state.get("input", "")) and persisted_state.get("phase") == "cycle_completed":
        agent_trace.append(
            {
                "task_id": "loop.state",
                "agent": "LearningLoopAgent",
                "status": "success",
                "summary": "上一周期已完成，本轮开启新的学习闭环周期",
            }
        )
        persisted_state = {}

    if _is_learning_loop_continue_intent(state.get("input", "")) and persisted_state.get("status") == "active":
        agent_trace.append(
            {
                "task_id": "loop.state",
                "agent": "LearningLoopAgent",
                "status": "success",
                "summary": f"读取已有闭环状态：phase={persisted_state.get('phase', 'planned')}",
            }
        )
        snapshot = _build_planner_practice_snapshot(str(sid))
        if re.fullmatch(r"day\d+_reviewed", str(persisted_state.get("phase") or "")) or str(persisted_state.get("phase") or "") == "remediation_reviewed":
            quiz_params, updated_loop_state, next_summary = _learning_loop_next_quiz_plan(persisted_state, snapshot)
        else:
            quiz_params = _learning_loop_quiz_route_params(persisted_state)
            updated_loop_state = {
                **persisted_state,
                "phase": "day1_quiz_issued",
                "snapshot": snapshot or persisted_state.get("snapshot", {}),
                "last_quiz_params": quiz_params,
                "updated_at": int(time.time() * 1000),
            }
            next_summary = f"执行 Day1 练习：{quiz_params.get('topic')}"
        agent_trace.append(
            {
                "task_id": "quiz.day1",
                "agent": "QuizAgent",
                "status": "running",
                "summary": f"{next_summary}，{quiz_params.get('num')} 道{quiz_params.get('quiz_type')}",
            }
        )
        quiz_state = {
            **state,
            "input": str(quiz_params.get("topic") or state.get("input") or ""),
            "route": "quiz",
            "route_params": quiz_params,
            "session_id": str(sid),
        }
        quiz_result = await quiz_subagent_node(quiz_state)
        memory_manager.set_agent_state(str(sid), "learning_loop", updated_loop_state)
        agent_trace[-1] = {
            **agent_trace[-1],
            "status": "success",
            "summary": f"下一步题卡已生成：{quiz_params.get('topic')}",
        }
        raw_quiz = quiz_result.get("subagent_result", {}) if isinstance(quiz_result, dict) else {}
        if isinstance(raw_quiz, dict):
            structured = _merge_learning_loop_quiz_result(raw_quiz, updated_loop_state, agent_trace)
            return {"subagent_result": structured, "final_answer": structured.get("text", quiz_result.get("final_answer", ""))}
        answer = str(quiz_result.get("final_answer", "") if isinstance(quiz_result, dict) else raw_quiz)
        structured = {
            "kind": "chat",
            "render_mode": "markdown",
            "text": answer or "Day1 练习生成失败，请稍后重试。",
            "payload": {"learning_loop_data": {"loop_state": updated_loop_state}, "agent_trace": agent_trace},
            "meta": {
                "agent_mode": "learning_loop",
                "learning_loop_structured": True,
                "learning_loop_stateful": True,
                "learning_loop_phase": "day1_quiz_issued",
                "agent_trace": agent_trace,
            },
        }
        return {"subagent_result": structured, "final_answer": structured["text"]}
    elif _is_learning_loop_continue_intent(state.get("input", "")):
        agent_trace.append(
            {
                "task_id": "loop.state",
                "agent": "LearningLoopAgent",
                "status": "partial",
                "summary": "未找到可继续的闭环状态，本轮先重新生成学习闭环计划",
            }
        )

    snapshot = _build_planner_practice_snapshot(str(sid))
    agent_trace.append(
        {
            "task_id": "history.snapshot",
            "agent": "HistoryAgent",
            "status": "success",
            "summary": f"读取练习画像：{snapshot.get('total_attempts', 0)} 题，薄弱点 {len(snapshot.get('weak_points', []) or [])} 个",
        }
    )

    focus_points = _learning_loop_focus_points(state.get("input", ""), snapshot)
    evidence_cards: List[Dict[str, Any]] = []
    evidence_query = " ".join(focus_points[:3]) or state.get("input", "")
    orchestrator_context = _format_orchestrator_observations(state, {"search_courseware"}, title="ReAct课件检索结果")
    if orchestrator_context:
        evidence_cards = _extract_learning_loop_evidence(orchestrator_context, limit=3)
        agent_trace.append(
            {
                "task_id": "rag.evidence",
                "agent": "RagAgent",
                "status": "success" if evidence_cards else "partial",
                "summary": f"复用 ReAct 课件证据：命中 {len(evidence_cards)} 条",
            }
        )
    else:
        try:
            rag = await get_rag_service()
            context = await asyncio.wait_for(rag.retrieve_context(evidence_query, mode="rag_chat"), timeout=12.0)
            evidence_cards = _extract_learning_loop_evidence(context, limit=3)
            agent_trace.append(
                {
                    "task_id": "rag.evidence",
                    "agent": "RagAgent",
                    "status": "success" if evidence_cards else "partial",
                    "summary": f"检索课件证据：命中 {len(evidence_cards)} 条",
                }
            )
        except Exception as e:
            logger.warning(f"[LearningLoopAgent] 课件检索失败 sid={sid}: {e}")
            agent_trace.append(
                {
                    "task_id": "rag.evidence",
                    "agent": "RagAgent",
                    "status": "partial",
                    "summary": "课件检索未完成，本轮按画像和基线任务降级生成",
                }
            )

    plan = _build_learning_loop_plan(state.get("input", ""), snapshot, evidence_cards)
    agent_trace.append(
        {
            "task_id": "planner.loop",
            "agent": "PlannerAgent",
            "status": "success",
            "summary": f"生成 {plan.get('cycle_days', 3)} 天闭环计划，Day1 聚焦 {((plan.get('daily_plan') or [{}])[0] or {}).get('focus', '核心主题')}",
        }
    )

    plan, critic_findings, critic_status = _critic_learning_loop_plan(plan, snapshot)
    agent_trace.append(
        {
            "task_id": "critic.learning_loop",
            "agent": "CriticAgent",
            "status": critic_status,
            "summary": "校验通过" if not critic_findings else f"发现并修复 {len(critic_findings)} 个计划问题",
        }
    )

    # 若本轮发现历史考点脏数据，给 trace 留提示，但不主动执行写操作。
    try:
        noisy_count = sum(1 for kp in memory_manager.get_knowledge_point_stats(str(sid)).keys() if _is_noisy_kp_name(kp))
        if noisy_count:
            agent_trace.append(
                {
                    "task_id": "data_quality.notice",
                    "agent": "HistoryAgent",
                    "status": "partial",
                    "summary": f"检测到 {noisy_count} 个疑似题干式考点名，可后续回填归一化",
                }
            )
    except Exception:
        pass

    answer = _render_learning_loop_markdown(plan, critic_findings, evidence_cards)
    loop_state = _build_learning_loop_state(str(sid), snapshot, focus_points, plan, evidence_cards, critic_status)
    memory_manager.set_agent_state(str(sid), "learning_loop", loop_state)
    agent_trace.append(
        {
            "task_id": "loop.state",
            "agent": "LearningLoopAgent",
            "status": "success",
            "summary": "已保存闭环状态，可继续执行 Day1 练习",
        }
    )
    structured = {
        "kind": "chat",
        "render_mode": "markdown",
        "text": answer,
        "payload": {
            "learning_loop_data": {
                "snapshot": snapshot,
                "priority_points": focus_points,
                "daily_plan": plan.get("daily_plan", []),
                "day1_quiz_request": plan.get("day1_quiz_request", {}),
                "next_action": plan.get("next_action", ""),
                "critic_findings": critic_findings,
                "loop_state": loop_state,
            },
            "evidence_cards": evidence_cards,
            "agent_trace": agent_trace,
        },
        "meta": {
            "agent_mode": "learning_loop",
            "learning_loop_structured": True,
            "learning_loop_stateful": True,
            "learning_loop_phase": loop_state.get("phase", "planned"),
            "planner_data_driven": True,
            "critic_status": critic_status,
            "agent_trace": agent_trace,
            "evidence_status": "grounded" if evidence_cards else "none",
            "evidence_count": len(evidence_cards),
            **({"degrade_reason": plan.get("critic_degrade_reason")} if plan.get("critic_degrade_reason") else {}),
            **({"evidence_source": f"课件引用 {len(evidence_cards)} 条"} if evidence_cards else {}),
        },
    }
    return {"subagent_result": structured, "final_answer": answer}


async def chitchat_node(state: SupervisorState) -> SupervisorState:
    """Chitchat: 处理闲聊、问候、感谢、无关话题"""
    logger.info("[Chitchat] 处理闲聊")

    session_id = state.get('session_id') or current_session_id.get() or 'default'
    current_session_id.set(session_id)

    # 获取近期对话历史
    chat_history = state.get('chat_history', [])
    recent_history = _format_history_with_limit(chat_history, max_chars=600)

    # 获取图谱记忆
    memory_context = state.get('memory_context', '')

    context_str = f"【近期对话】\n{recent_history}\n\n【知识点备忘】\n{memory_context}".strip()

    result = await _call_llm(
        CHITCHAT_PROMPT,
        input=state['input'],
        context=context_str,
    )
    return {"subagent_result": result, "final_answer": result}


def _format_practice_history_rows(history: list[dict], limit: int = 20) -> str:
    """格式化错题记录，输出可操作的 ID。"""
    if not history:
        return "暂无练习记录"

    safe_limit = max(1, min(int(limit or 20), 200))
    rows = history[:safe_limit]
    lines = [f"【练习历史（最近 {len(rows)} 条）】"]
    for i, rec in enumerate(rows, 1):
        status = "✅" if rec.get("is_correct") else "❌"
        rec_id = rec.get("id")
        kp = rec.get("knowledge_point") or "未标注"
        q_content = str(rec.get("question_content") or "").strip().replace("\n", " ")
        lines.append(f"{i}. [ID:{rec_id}] {status} [{kp}] {q_content[:80]}")
    return "\n".join(lines)


def _format_message_rows(messages: list[dict], limit: int = 30) -> tuple[str, list[dict]]:
    """格式化对话历史，按“最新在前”展示并返回同序列表。"""
    if not messages:
        return "暂无历史对话记录", []

    safe_limit = max(1, min(int(limit or 30), 200))
    ordered = list(reversed(messages[-safe_limit:]))
    lines = [f"【历史对话（最近 {len(ordered)} 条，最新在前）】"]
    for i, msg in enumerate(ordered, 1):
        role = "学生" if msg.get("role") == "human" else "助手"
        timestamp = msg.get("timestamp", 0)
        content = str(msg.get("content", "")).strip().replace("\n", " ")
        lines.append(f"{i}. [TS:{timestamp}] {role}: {content[:120]}")
    return "\n".join(lines), ordered


def _resolve_practice_id_by_index(history: list[dict], index: int) -> int | None:
    """把“第N条错题”映射为实际 record_id。"""
    if index <= 0:
        return None
    pos = index - 1
    if pos < 0 or pos >= len(history):
        return None
    rec_id = history[pos].get("id")
    return int(rec_id) if rec_id is not None else None


def _resolve_message_ts_by_index(ordered_messages: list[dict], index: int) -> int | None:
    """把“第N条消息”映射为 timestamp。"""
    if index <= 0:
        return None
    pos = index - 1
    if pos < 0 or pos >= len(ordered_messages):
        return None
    ts = ordered_messages[pos].get("timestamp")
    return int(ts) if ts is not None else None


def _classify_wrong_reason(reason: str) -> str:
    """将错因归入可读类别，便于聚合展示。"""
    text = str(reason or "").strip().lower()
    if not text:
        return "未填写错因"
    mapping = [
        ("概念理解偏差", ["概念", "定义", "理解", "混淆", "区分"]),
        ("审题与条件遗漏", ["审题", "条件", "忽略", "漏看", "题意"]),
        ("记忆与背诵不牢", ["记忆", "背诵", "遗忘", "没记住", "不熟"]),
        ("步骤与推理链断裂", ["步骤", "推导", "推理", "过程", "链路"]),
        ("计算与细节失误", ["计算", "符号", "单位", "抄错", "粗心"]),
    ]
    for label, keywords in mapping:
        if any(k in text for k in keywords):
            return label
    return "其他原因"


def _is_noisy_kp_name(kp: str) -> bool:
    """识别像题干句子的考点名，用于数据质量提醒。"""
    text = str(kp or "").strip()
    if not text:
        return True
    return len(text) > 28 or any(tok in text for tok in ["请", "下列", "以下", "？", "?", "。", "；", ";"])


def _build_practice_analysis_report(
    stats: Dict[str, Dict[str, Any]],
    history: List[Dict[str, Any]],
    *,
    mastery_priority: Optional[List[Dict[str, Any]]] = None,
    max_weak: int = 8,
    max_recent_wrong: int = 6,
) -> str:
    """构建结构化练习分析报告，避免仅复述历史列表。"""
    total = len(history)
    correct = sum(1 for item in history if bool(item.get("is_correct")))
    wrong = max(0, total - correct)
    accuracy = (correct / total * 100.0) if total > 0 else 0.0

    recent_window = history[: min(20, total)]
    recent_total = len(recent_window)
    recent_correct = sum(1 for item in recent_window if bool(item.get("is_correct")))
    recent_acc = (recent_correct / recent_total * 100.0) if recent_total > 0 else 0.0

    weak_items = [
        (kp, item)
        for kp, item in stats.items()
        if float(item.get("accuracy", 0.0)) < 60.0 and int(item.get("total", 0)) > 0
    ]
    weak_items.sort(key=lambda kv: (float(kv[1].get("accuracy", 0.0)), -int(kv[1].get("total", 0)), kv[0]))
    safe_mastery_priority = []
    for row in mastery_priority or []:
        if isinstance(row, dict) and str(row.get("knowledge_point", "")).strip():
            safe_mastery_priority.append(row)

    strong_items = [
        (kp, item)
        for kp, item in stats.items()
        if float(item.get("accuracy", 0.0)) >= 80.0 and int(item.get("total", 0)) > 0
    ]
    strong_items.sort(key=lambda kv: (-float(kv[1].get("accuracy", 0.0)), -int(kv[1].get("total", 0)), kv[0]))

    reason_counts: Dict[str, int] = {}
    for row in history:
        if bool(row.get("is_correct")):
            continue
        label = _classify_wrong_reason(str(row.get("wrong_reason") or ""))
        reason_counts[label] = int(reason_counts.get(label, 0)) + 1
    top_reasons = sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]

    noisy_kp_count = sum(1 for kp in stats.keys() if _is_noisy_kp_name(kp))
    recent_wrong_rows = [row for row in history if not bool(row.get("is_correct"))][:max_recent_wrong]

    lines: List[str] = ["【历史学习分析】"]
    lines.append(f"- 累计练习 {total} 题，正确 {correct} 题，错误 {wrong} 题，整体正确率 {accuracy:.1f}%")
    if recent_total > 0:
        lines.append(f"- 最近 {recent_total} 题正确率 {recent_acc:.1f}%")
    if total == 0:
        lines.append("- 暂无可分析练习记录。建议先做 5-10 道基础题再查看分析。")
        return "\n".join(lines)

    lines.append("")
    lines.append("【薄弱知识点优先级】")
    if safe_mastery_priority:
        for i, item in enumerate(safe_mastery_priority[:max_weak], 1):
            kp = str(item.get("knowledge_point") or "未标注")
            mastery_score = float(item.get("mastery_score", 0.0))
            accuracy_pct = float(item.get("accuracy", 0.0))
            correct_cnt = int(item.get("correct_count", 0))
            attempt_cnt = int(item.get("attempt_count", 0))
            urgency = float(item.get("review_urgency", 0.0))
            lines.append(
                f"{i}. {kp}：掌握度 {mastery_score:.2f}，正确率 {accuracy_pct:.1f}%（{correct_cnt}/{attempt_cnt}），紧迫度 {urgency:.2f}"
            )
    elif weak_items:
        for i, (kp, item) in enumerate(weak_items[:max_weak], 1):
            lines.append(
                f"{i}. {kp}：{int(item.get('correct', 0))}/{int(item.get('total', 0))}（{float(item.get('accuracy', 0.0)):.1f}%）"
            )
    else:
        lines.append("- 暂无明显薄弱点（当前正确率均 >= 60%）。")

    lines.append("")
    lines.append("【已掌握知识点】")
    if strong_items:
        for i, (kp, item) in enumerate(strong_items[:5], 1):
            lines.append(
                f"{i}. {kp}：{int(item.get('correct', 0))}/{int(item.get('total', 0))}（{float(item.get('accuracy', 0.0)):.1f}%）"
            )
    else:
        lines.append("- 暂无稳定掌握项（正确率 >= 80%）。")

    lines.append("")
    lines.append("【高频错因模式】")
    if top_reasons:
        for i, (label, cnt) in enumerate(top_reasons, 1):
            lines.append(f"{i}. {label}：{cnt} 次")
    else:
        lines.append("- 暂无可用错因文本，建议做题后补充错因。")

    lines.append("")
    lines.append("【最近错题样本】")
    if recent_wrong_rows:
        for i, row in enumerate(recent_wrong_rows, 1):
            kp = str(row.get("knowledge_point") or "未标注")
            question = str(row.get("question_content") or "").strip().replace("\n", " ")
            wrong_reason = str(row.get("wrong_reason") or "").strip()
            lines.append(f"{i}. [{kp}] {question[:120]}{'...' if len(question) > 120 else ''}")
            if wrong_reason:
                lines.append(f"   错因：{wrong_reason[:80]}{'...' if len(wrong_reason) > 80 else ''}")
    else:
        lines.append("- 最近记录中暂无错题。")

    lines.append("")
    lines.append("【下一步建议】")
    if safe_mastery_priority:
        top_targets = [str(item.get("knowledge_point", "")).strip() for item in safe_mastery_priority if str(item.get("knowledge_point", "")).strip()][:3]
    elif weak_items:
        top_targets = [kp for kp, _ in weak_items[:3]]
    else:
        top_targets = []
    if top_targets:
        for kp in top_targets:
            lines.append(f"- 先复习「{kp}」10 分钟，再做 2 道同考点题并立即对照错因。")
        lines.append("- 若仍连续错 2 次以上，请发“讲解 + 出1道同类题”进行纠偏。")
    else:
        lines.append("- 进入混合拔高训练：薄弱点 40% + 随机综合题 60%。")

    if noisy_kp_count > 0:
        lines.append("")
        lines.append(f"【数据质量提示】检测到 {noisy_kp_count} 个考点名接近题干句式，后续统计可继续归一化。")

    return "\n".join(lines)


async def history_subagent_node(state: SupervisorState) -> SupervisorState:
    """History SubAgent: 查看练习历史、答题记录、错题分析（由LLM理解用户意图并回答）"""
    from utils.memory_service import memory_manager

    logger.info("[History SubAgent] 处理练习历史查询...")

    sid = state.get('session_id') or current_session_id.get() or 'default'
    current_session_id.set(sid)

    params = state.get('route_params', {}) if isinstance(state.get('route_params', {}), dict) else {}
    if params.get("action") == "context_reference":
        structured = _build_context_reference_reply(str(sid), state.get("input", ""))
        return {"subagent_result": structured, "final_answer": structured["text"]}

    limit = params.get('limit', 20)

    # 先执行可确定的管理动作（查/删/清空），避免再走 LLM 推断。
    action = parse_history_action(state.get("input", ""))
    action_name = action.get("action", "none")
    if action_name != "none":
        if action_name == "list_practice":
            history = memory_manager.get_practice_history(sid, max(1, min(int(limit or 20), 200)))
            text = _format_practice_history_rows(history, limit=limit)
            return {"subagent_result": text, "final_answer": text}

        if action_name == "delete_practice":
            history = memory_manager.get_practice_history(sid, 200)
            record_id = action.get("record_id")
            if record_id is None and action.get("record_index") is not None:
                record_id = _resolve_practice_id_by_index(history, int(action.get("record_index")))
            if record_id is None:
                text = "请提供要删除的错题 ID，或说“删除第N条错题”。"
                return {"subagent_result": text, "final_answer": text}
            deleted = memory_manager.delete_practice_record(sid, int(record_id))
            if not deleted:
                text = f"未找到错题记录 ID={record_id}，删除失败。"
                return {"subagent_result": text, "final_answer": text}
            latest = memory_manager.get_practice_history(sid, max(1, min(int(limit or 10), 50)))
            text = f"已删除错题记录 ID={record_id}\n\n{_format_practice_history_rows(latest, limit=min(int(limit or 10), 20))}"
            return {"subagent_result": text, "final_answer": text}

        if action_name == "clear_practice":
            deleted_count = memory_manager.clear_practice_history(sid)
            text = f"已清空错题记录，共删除 {deleted_count} 条。"
            return {"subagent_result": text, "final_answer": text}

        if action_name == "list_messages":
            messages = memory_manager.get_messages(sid)
            text, _ = _format_message_rows(messages, limit=limit)
            return {"subagent_result": text, "final_answer": text}

        if action_name == "delete_message":
            messages = memory_manager.get_messages(sid)
            text_rows, ordered_messages = _format_message_rows(messages, limit=200)
            timestamp = action.get("timestamp")
            if timestamp is None and action.get("message_index") is not None:
                timestamp = _resolve_message_ts_by_index(ordered_messages, int(action.get("message_index")))
            if timestamp is None:
                text = "请提供要删除的消息 timestamp，或说“删除第N条消息”。"
                return {"subagent_result": text, "final_answer": text}
            deleted = memory_manager.delete_message(sid, int(timestamp))
            if not deleted:
                text = f"未找到消息 TS={timestamp}，删除失败。\n\n{text_rows}"
                return {"subagent_result": text, "final_answer": text}
            latest_messages = memory_manager.get_messages(sid)
            latest_text, _ = _format_message_rows(latest_messages, limit=min(int(limit or 20), 50))
            text = f"已删除消息 TS={timestamp}\n\n{latest_text}"
            return {"subagent_result": text, "final_answer": text}

        if action_name == "clear_messages":
            success = memory_manager.clear_messages(sid)
            text = "已清空历史对话记录。" if success else "清空失败：会话不存在。"
            return {"subagent_result": text, "final_answer": text}

        if action_name == "analyze_practice":
            stats = memory_manager.get_knowledge_point_stats(sid)
            history_rows = memory_manager.get_practice_history(sid, max(50, min(int(limit or 120), 200)))
            mastery_priority = memory_manager.get_priority_review_points(sid, limit=8)
            text = _build_practice_analysis_report(
                stats,
                history_rows,
                mastery_priority=mastery_priority,
            )
            return {"subagent_result": text, "final_answer": text}

    # 获取统计数据和历史记录
    stats = memory_manager.get_knowledge_point_stats(sid)
    mastery_priority = memory_manager.get_priority_review_points(
        sid,
        limit=max(8, min(len(stats) + 4, 60)) if stats else 8,
    )
    history = memory_manager.get_practice_history(sid, limit)

    # 格式化统计数据
    if stats:
        priority_order = [str(item.get("knowledge_point", "")).strip() for item in mastery_priority if str(item.get("knowledge_point", "")).strip()]
        ordered_stats: List[tuple[str, Dict[str, Any]]] = []
        seen_kp = set()
        for kp in priority_order:
            if kp in stats and kp not in seen_kp:
                ordered_stats.append((kp, stats[kp]))
                seen_kp.add(kp)
        for kp, data in sorted(stats.items(), key=lambda kv: kv[0]):
            if kp in seen_kp:
                continue
            ordered_stats.append((kp, data))

        stats_lines = []
        for kp, data in ordered_stats:
            weak = "🔴 薄弱" if data.get('weak') else "🟢 掌握"
            stats_lines.append(f"- {kp}: {data['correct']}/{data['total']} ({data['accuracy']:.1f}%) {weak}")
        stats_str = "\n".join(stats_lines)
    else:
        stats_str = "暂无练习统计数据"

    # 格式化历史记录
    if history:
        history_lines = []
        for rec in history:
            status = "✅" if rec.get('is_correct') else "❌"
            kp = rec.get('knowledge_point', '未知')
            q_content = rec.get('question_content', '')[:50]
            history_lines.append(f"{status} [{kp}] {q_content}...")
        history_str = "\n".join(history_lines)
    else:
        history_str = "暂无练习记录"

    # LLM 根据用户请求和实际数据生成回答
    result = await _call_llm(
        HISTORY_SUBAGENT_PROMPT,
        input=state['input'],
        stats=stats_str,
        history=history_str,
    )
    return {"subagent_result": result, "final_answer": result}


# ==================== 路由函数 ====================

def route_to_subagent(state: SupervisorState) -> str:
    """根据 Supervisor 决策路由到对应 SubAgent"""
    route = state.get('route', 'rag')
    valid = {"rag", "quiz", "exam", "ops", "planner", "history", "learning_loop", "chitchat"}
    return route if route in valid else "rag"


# ==================== 构建工作流图 ====================

def build_supervisor_graph() -> StateGraph:
    """
    Supervisor 工作流：
    supervisor → tool_orchestrator → [rag_agent | quiz_agent | exam_agent | ops_agent | planner_agent | history_agent | learning_loop | chitchat] → END
    """
    graph = StateGraph(SupervisorState)

    # 注册节点
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("tool_orchestrator", tool_orchestrator_node)
    graph.add_node("rag_agent", rag_subagent_node)
    graph.add_node("quiz_agent", quiz_subagent_node)
    graph.add_node("exam_agent", exam_subagent_node)
    graph.add_node("ops_agent", ops_subagent_node)
    graph.add_node("planner_agent", planner_subagent_node)
    graph.add_node("history_agent", history_subagent_node)
    graph.add_node("learning_loop", learning_loop_subagent_node)
    graph.add_node("chitchat", chitchat_node)

    # 入口
    graph.set_entry_point("supervisor")

    graph.add_edge("supervisor", "tool_orchestrator")

    # Orchestrator 可能基于工具计划调整目标专家 Agent。
    graph.add_conditional_edges(
        "tool_orchestrator",
        route_to_subagent,
        {
            "rag": "rag_agent",
            "quiz": "quiz_agent",
            "exam": "exam_agent",
            "ops": "ops_agent",
            "planner": "planner_agent",
            "history": "history_agent",
            "learning_loop": "learning_loop",
            "chitchat": "chitchat",
        }
    )

    # 所有 SubAgent 执行完毕后结束
    for node in ("rag_agent", "quiz_agent", "exam_agent", "ops_agent", "planner_agent", "history_agent", "learning_loop", "chitchat"):
        graph.add_edge(node, END)

    return graph.compile()


supervisor_workflow = build_supervisor_graph()
