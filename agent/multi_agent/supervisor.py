"""
Supervisor 多 Agent 路由架构

Supervisor Agent 分析用户意图，路由到对应 SubAgent：
  - rag_agent    : 知识问答、概念解释、复习某知识点
  - quiz_agent   : 出单道或少量题目（调用 Reflexion 出题工作流）
  - exam_agent   : 生成完整试卷（调用 Reflexion 出卷工作流）
  - planner_agent: 制定复习计划、推荐学习顺序
  - chitchat     : 闲聊、感谢、无关话题

工作流：supervisor → (路由判断) → SubAgent → END
"""
import json
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from model.factory import chat_model
from utils.logger_handler import logger
from utils.session_context import current_session_id


# ==================== 状态定义 ====================

class SupervisorState(TypedDict):
    input: str              # 当前用户输入
    chat_history: List      # 历史消息（LangChain BaseMessage 列表）
    memory_context: str     # 长期图谱记忆（注入 system prompt）
    session_id: str         # 当前科目会话
    # Supervisor 决策
    route: str              # "rag" | "quiz" | "exam" | "planner" | "chitchat"
    route_reason: str       # 路由原因（用于日志）
    route_params: dict      # 从用户输入中提取的参数
    # SubAgent 结果
    subagent_result: str    # SubAgent 原始输出
    final_answer: str       # 最终回答（透传给调用方）


# ==================== 提示词 ====================

SUPERVISOR_PROMPT = """你是一个学习助手的调度中心。分析用户意图，决定由哪个专业 Agent 处理。

可用 Agent：
- rag      : 明确的知识问答、解释概念、复习知识点、查找资料、总结章节（需要基于课件内容回答）
- quiz     : 出单道或少量题目（用户明确说"出题"、"出X道"、"给我一道选择题"等）
- exam     : 生成完整试卷（"出试卷"、"模拟考试"、"出一套题"、"帮我出10道题"等）
- planner  : 制定复习计划、规划学习顺序、推荐复习策略
- chitchat : 闲聊、感谢、问候、功能介绍、自我介绍、"能做什么"、"你是谁"、一般性聊天（不需要课件内容）

【历史记忆摘要】
{memory_context}

【近期对话记录】
{recent_history}

【用户输入】
{input}

返回如下 JSON（只返回 JSON，不要其他内容）：
```json
{{
    "route": "rag|quiz|exam|planner|chitchat",
    "reason": "路由原因（一句话）",
    "params": {{
        "topic": "核心考点名词（若能提取）",
        "quiz_type": "选择题|填空题|判断题|简答题（若有）",
        "quiz_types": ["选择题", "填空题", "判断题", "简答题"],
        "num": 3,
        "topics": ["考点1", "考点2"],
        "total_questions": 10,
        "quantity_dist": {{"choice": 10, "fill": 5, "judge": 5, "essay": 3}}
    }}
}}
```

【重要】参数提取规则：
- 如果用户要求"综合测试题"、"出一套题"、"模拟考试"、"出试卷"等，route 必须为 "exam"
- 如果提到"选择题"、"填空题"、"判断题"、"简答题"等多种题型，quiz_types 必须包含所有提到的题型
- 【默认题型分布】如果没有指定数量，默认：选择题10道、判断5道、填空5道、简答3道（共23道）
- 【数量提取】如果用户说"10个选择题"或"判断填空各5个"，必须提取到 quantity_dist
- 【综合卷】用户说"出一套题"时，默认用23道
- total_questions = sum(quantity_dist.values())"""


RAG_SUBAGENT_PROMPT = """你是知识问答专家，根据检索到的课件资料准确回答学生问题。

【长期记忆摘要】
{memory_context}

【课件资料】
{context}

【学生问题】
{input}

请给出清晰、有条理的回答。优先引用课件原文，并说明来源。
如果课件中没有相关内容，如实告知并给出通识性解答。"""


PLANNER_SUBAGENT_PROMPT = """你是学习规划专家，根据学生的复习目标制定具体可执行的学习计划。

【历史记忆摘要】
{memory_context}

【学生需求】
{input}

请制定复习计划，包括：
1. 核心知识点梳理与优先级排序
2. 建议复习顺序（从基础到进阶）
3. 时间分配建议
4. 易混淆点与重点提示"""


CHITCHAT_PROMPT = """你是一个友好的学习助手，正在陪伴学生备考。

【学生消息】
{input}

请用简洁友好的方式回应，必要时引导学生回到学习话题。"""


# ==================== 工具函数 ====================

async def _call_llm(prompt_template: str, **kwargs) -> str:
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | chat_model | StrOutputParser()
    return await chain.ainvoke(kwargs)


def _extract_json(text: str) -> dict:
    import re
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


# ==================== Supervisor 节点 ====================

async def supervisor_node(state: SupervisorState) -> SupervisorState:
    """Supervisor: 分析意图，输出路由决策"""
    logger.info("[Supervisor] 分析用户意图...")

    # 将近期对话历史格式化为可读文本
    history_lines = []
    for msg in state.get('chat_history', []):
        role = "学生" if msg.__class__.__name__ == "HumanMessage" else "助手"
        history_lines.append(f"{role}: {msg.content[:200]}")  # 截断超长消息
    recent_history = "\n".join(history_lines) if history_lines else "（无近期对话）"

    result = await _call_llm(
        SUPERVISOR_PROMPT,
        input=state['input'],
        memory_context=state.get('memory_context', ''),
        recent_history=recent_history,
    )
    try:
        decision = _extract_json(result)
        route = decision.get('route', 'rag')
        reason = decision.get('reason', '')
        params = decision.get('params', {})
        logger.info(f"[Supervisor] 路由决策: route={route}, reason={reason}, params={params}")
    except Exception:
        route = 'rag'
        reason = '路由解析失败，默认走 RAG'
        params = {}

    # 强制路由修正：如果输入包含出题相关关键词，强制走 exam
    input_text = state['input'].lower()
    exam_keywords = ['出题', '试卷', '考试', '测验', '题目', '综合卷', '模拟题', '生成题']
    if any(kw in input_text for kw in exam_keywords):
        route = 'exam'
        reason = '用户输入包含出题相关关键词，强制路由到 exam'
        # 默认参数
        params = params or {}
        if 'quiz_types' not in params:
            params['quiz_types'] = ['选择题', '填空题', '判断题', '简答题']
        if 'quantity_dist' not in params:
            params['quantity_dist'] = {"choice": 10, "fill": 5, "judge": 5, "essay": 3}
        params['total_questions'] = sum(params['quantity_dist'].values())
        logger.info(f"[Supervisor] 强制修正: route={route}, params={params}")
        params = {}
        logger.error(f"[Supervisor] 路由解析失败: {result[:200]}")

    logger.info(f"[Supervisor] 路由 → {route} | 原因: {reason}")
    return {"route": route, "route_reason": reason, "route_params": params}


# ==================== SubAgent 节点 ====================

async def rag_subagent_node(state: SupervisorState) -> SupervisorState:
    """RAG SubAgent: 检索课件原始片段，直接回答（单次 LLM）"""
    logger.info("[RAG SubAgent] 检索课件资料...")

    from rag.rag_service import RagSummarizeService
    sid = state.get('session_id') or current_session_id.get() or 'default'
    current_session_id.set(sid)

    params = state.get('route_params', {})
    topic = params.get('topic') or state['input']

    try:
        rag = RagSummarizeService()
        # retrieve_context 只检索、不调 LLM，避免冗余的双重 LLM 调用
        context = await rag.retrieve_context(topic)
        # 如果检索为空，给出友好提示
        if not context or context.strip() == "":
            logger.warning("[RAG SubAgent] 知识库为空，请先上传课件")
            context = "【提示】当前知识库为空，请先上传课件后再提问。你可以通过点击「上传课件」按钮来添加复习资料。"
    except Exception as e:
        logger.warning(f"[RAG SubAgent] 检索失败: {e}")
        context = "【提示】知识库检索遇到问题，请确保已上传课件。如未上传，请先上传复习资料。"

    result = await _call_llm(
        RAG_SUBAGENT_PROMPT,
        memory_context=state.get('memory_context', ''),
        context=context,
        input=state['input'],
    )
    return {"subagent_result": result, "final_answer": result}


async def quiz_subagent_node(state: SupervisorState) -> SupervisorState:
    """Quiz SubAgent: 调用 Reflexion 出题工作流"""
    logger.info("[Quiz SubAgent] 启动 Reflexion 出题流程...")
    logger.info(f"[Quiz SubAgent] state.input={state.get('input', '')[:200]}...")
    logger.info(f"[Quiz SubAgent] route_params={state.get('route_params', {})}")
    logger.info(f"[Quiz SubAgent] session_id={state.get('session_id', 'unknown')}")

    from agent.multi_agent.quiz_agent import run_quiz_agent
    from api.routers.knowledge import get_sample_paper_context

    sid = state.get('session_id') or current_session_id.get() or 'default'
    current_session_id.set(sid)
    sample_ctx = get_sample_paper_context(sid)

    params = state.get('route_params', {})
    topic = params.get('topic') or state['input']
    quiz_type = params.get('quiz_type', '选择题')
    num = int(params.get('num', 3))

    result = await run_quiz_agent(topic, quiz_type, num, sample_ctx)
    return {"subagent_result": result, "final_answer": result}


async def exam_subagent_node(state: SupervisorState) -> SupervisorState:
    """Exam SubAgent: 调用 Reflexion 出卷工作流"""
    logger.info("[Exam SubAgent] 启动 Reflexion 出卷流程...")
    logger.info(f"[Exam SubAgent] state.input={state.get('input', '')[:200]}...")
    logger.info(f"[Exam SubAgent] route_params={state.get('route_params', {})}")
    logger.info(f"[Exam SubAgent] session_id={state.get('session_id', 'unknown')}")

    from agent.multi_agent.quiz_agent import run_exam_agent
    from api.routers.knowledge import get_sample_paper_context

    sid = state.get('session_id') or current_session_id.get() or 'default'
    current_session_id.set(sid)
    sample_ctx = get_sample_paper_context(sid)

    params = state.get('route_params', {})
    topics = params.get('topics') or [state['input']]
    # 确保 quiz_types 是数组，默认包含多种题型（综合卷）
    quiz_types = params.get('quiz_types')
    if not quiz_types or not isinstance(quiz_types, list):
        quiz_types = ['选择题', '填空题', '判断题', '简答题']
    total = int(params.get('total_questions', 23))
    # 获取用户自定义的题型数量分布
    quantity_dist = params.get('quantity_dist', {})

    logger.info(f"[Exam SubAgent] topics={topics}, quiz_types={quiz_types}, total={total}, quantity_dist={quantity_dist}")

    result = await run_exam_agent(topics, quiz_types, total, sample_ctx, quantity_dist)
    return {"subagent_result": result, "final_answer": result}


async def planner_subagent_node(state: SupervisorState) -> SupervisorState:
    """Planner SubAgent: 制定复习计划"""
    logger.info("[Planner SubAgent] 生成复习计划...")

    result = await _call_llm(
        PLANNER_SUBAGENT_PROMPT,
        memory_context=state.get('memory_context', ''),
        input=state['input'],
    )
    return {"subagent_result": result, "final_answer": result}


async def chitchat_node(state: SupervisorState) -> SupervisorState:
    """Chitchat: 处理闲聊"""
    result = await _call_llm(CHITCHAT_PROMPT, input=state['input'])
    return {"subagent_result": result, "final_answer": result}


# ==================== 路由函数 ====================

def route_to_subagent(state: SupervisorState) -> str:
    """根据 Supervisor 决策路由到对应 SubAgent"""
    route = state.get('route', 'rag')
    valid = {"rag", "quiz", "exam", "planner", "chitchat"}
    return route if route in valid else "rag"


# ==================== 构建工作流图 ====================

def build_supervisor_graph() -> StateGraph:
    """
    Supervisor 工作流：
    supervisor → (路由判断) → [rag_agent | quiz_agent | exam_agent | planner_agent | chitchat] → END
    """
    graph = StateGraph(SupervisorState)

    # 注册节点
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag_agent", rag_subagent_node)
    graph.add_node("quiz_agent", quiz_subagent_node)
    graph.add_node("exam_agent", exam_subagent_node)
    graph.add_node("planner_agent", planner_subagent_node)
    graph.add_node("chitchat", chitchat_node)

    # 入口
    graph.set_entry_point("supervisor")

    # Supervisor 条件路由
    graph.add_conditional_edges(
        "supervisor",
        route_to_subagent,
        {
            "rag": "rag_agent",
            "quiz": "quiz_agent",
            "exam": "exam_agent",
            "planner": "planner_agent",
            "chitchat": "chitchat",
        }
    )

    # 所有 SubAgent 执行完毕后结束
    for node in ("rag_agent", "quiz_agent", "exam_agent", "planner_agent", "chitchat"):
        graph.add_edge(node, END)

    return graph.compile()


supervisor_workflow = build_supervisor_graph()
