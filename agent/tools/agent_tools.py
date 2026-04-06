import json
from typing import Optional

from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.logger_handler import logger, timer_and_token_logger
from utils.session_context import current_session_id
from utils.config_handler import agent_conf

# 联网搜索开关
WEB_SEARCH_ENABLED = agent_conf.get('web_search', {}).get('enabled', True)

# 按 session_id 缓存 RAG 实例，避免全局单例在切换科目时检索旧知识库
# asyncio 是协作式调度，RagSummarizeService() 是同步初始化（无 await），
# 因此 check-then-set 在单 event loop 内是原子的，无需加锁（fix #7）
_rag_cache: dict[str, RagSummarizeService] = {}


def clear_rag_cache(session_id: str):
    """清理指定 session 的 RAG 缓存"""
    if session_id in _rag_cache:
        del _rag_cache[session_id]


async def _get_rag() -> RagSummarizeService:
    """
    按当前 session_id 获取（或新建）RAG 服务实例。
    同一科目复用同一实例，切换科目自动创建新实例。
    """
    sid = current_session_id.get() or "default"
    if sid not in _rag_cache:
        logger.info(f"[RAG] 为 session [{sid}] 初始化新的 RagSummarizeService")
        _rag_cache[sid] = RagSummarizeService()
    return _rag_cache[sid]


async def get_rag_service() -> RagSummarizeService:
    """对外暴露 session 级 RAG 实例获取函数。"""
    return await _get_rag()


# ==================== 工具定义 ====================

@tool(description='从课件知识库中检索知识点和参考资料。传入的query应该是一个具体考点名词。')
@timer_and_token_logger
async def search_courseware(query: str) -> str:
    rag = await _get_rag()
    return await rag.rag_summarize(query)


@tool(description='联网搜索信息。当需要验证答案、查找默认试卷风格、查询考试规范时使用。')
@timer_and_token_logger
async def web_search(query: str) -> str:
    """
    联网搜索工具（用于辅助验证，不能替代课件资料）
    优先使用 DuckDuckGo，失败则尝试 Tavily
    """
    if not WEB_SEARCH_ENABLED:
        return "联网搜索功能已关闭，请在配置文件中启用"

    # 尝试 DuckDuckGo
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        if result and len(result) > 10:
            if len(result) > 2000:
                result = result[:2000] + "..."
            logger.info(f"[联网搜索] {query} -> {len(result)} 字符")
            return result
    except Exception as e:
        logger.warning(f"[联网搜索] DuckDuckGo 失败: {e}")

    # 备用：尝试 Tavily
    try:
        from langchain_community.tools import TavilySearchResults
        search = TavilySearchResults(max_results=3)
        docs = search.run(query)
        if docs:
            result = "\n".join([d.get("content", "")[:500] for d in docs if d.get("content")])
            logger.info(f"[联网搜索] Tavily 备用成功 -> {len(result)} 字符")
            return result
    except Exception as e:
        logger.warning(f"[联网搜索] Tavily 也失败: {e}")

    return "联网搜索暂时不可用"


@tool(description='搜索大学期末考试试卷格式范例。当用户没有上传样卷时，可以搜索该学科的典型试卷格式。')
@timer_and_token_logger
async def search_exam_format(course_name: str) -> str:
    """搜索默认试卷格式"""
    if not WEB_SEARCH_ENABLED:
        return "联网搜索功能已关闭，无法获取在线试卷格式。请上传样卷或使用默认格式。"

    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        query = f"{course_name} 期末考试试卷 格式 题型 结构"
        result = search.run(query)
        if len(result) > 1500:
            result = result[:1500] + "..."
        logger.info(f"[试卷格式搜索] {course_name} -> 获取格式参考")
        return f"【{course_name}试卷格式参考】\n{result}"
    except Exception as e:
        logger.warning(f"[试卷格式搜索] 失败: {e}")
        return "无法获取格式参考，将使用默认格式"


@tool(description='当你需要为学生出题时，调用此工具。必须传入具体的考点名词(topic)、题目类型(quiz_type，如选择题/判断题等)和题目数量(num)。它会自动查资料并生成题目。')
@timer_and_token_logger
async def generate_quiz(topic: str, quiz_type: str = "选择题", num: int = 3) -> str:
    """
    单题生成（fix #3：改为 async tool，彻底消除 _run_async 的 event loop 阻塞）
    流程：自动加载样卷 → LangGraph 多 Agent 出题 → 返回
    """
    # 自动加载样卷上下文（fix #6：之前已修复）
    from api.routers.knowledge import get_sample_paper_context
    sid = current_session_id.get() or "default"
    sample_ctx = get_sample_paper_context(sid)

    from agent.multi_agent.quiz_agent import run_quiz_agent
    logger.info(f"[多Agent出题系统] 使用 LangGraph 执行: {topic}")
    return await run_quiz_agent(topic, quiz_type, num, sample_ctx)


@tool(description='生成完整试卷。如果用户上传了样卷，请传入样卷内容作为参考(样卷格式会作为参考)。参数topics是考点列表，quiz_types是题目类型列表，total_questions是总题数。')
@timer_and_token_logger
async def generate_exam_paper(
    topics: str,
    quiz_types: str = "选择题,判断题,填空题",
    total_questions: int = 10,
    sample_paper_context: str = None,
) -> str:
    topic_list = [t.strip() for t in topics.split(",")]
    quiz_type_list = [q.strip() for q in quiz_types.split(",")]

    from agent.multi_agent.quiz_agent import run_exam_agent
    logger.info("[多Agent出卷系统] 使用 LangGraph 执行")
    return await run_exam_agent(topic_list, quiz_type_list, total_questions, sample_paper_context)


@tool(description='查看学生的练习历史记录，包括之前的答题情况、正确答案和错因分析。用于了解学生对知识点的掌握情况。')
@timer_and_token_logger
async def get_practice_history(session_id: Optional[str] = None, limit: int = 20) -> str:
    from utils.memory_service import memory_manager
    sid = session_id or current_session_id.get() or "default"
    history = memory_manager.get_practice_history(sid, limit)
    if not history:
        return "暂无练习记录"

    lines = ["【练习历史】"]
    for i, h in enumerate(history, 1):
        status = "✓" if h["is_correct"] else "✗"
        kp = h["knowledge_point"] or "未标注"
        lines.append(f"{i}. {status} {kp} | 你的答案:{h['user_answer']} | 正确答案:{h['correct_answer']}")
        if h["wrong_reason"]:
            lines.append(f"   错因: {h['wrong_reason']}")
    return "\n".join(lines)


@tool(description='分析学生错题原因。需要传入题目、学生的答案和正确答案。返回错因分类和详细分析。')
@timer_and_token_logger
async def analyze_wrong_reason(question: str, user_answer: str, correct_answer: str) -> str:
    from model.factory import chat_model
    from langchain_core.messages import HumanMessage

    prompt = f"""分析学生错题原因，返回JSON格式：
{{"reason": "概念不清|计算错误|审题错误|粗心大意|完全不会", "analysis": "详细分析"}}

题目：{question}
学生答案：{user_answer}
正确答案：{correct_answer}

只返回JSON，不要其他内容。"""

    response = await chat_model.ainvoke([HumanMessage(content=prompt)])
    return response.content.strip()


@tool(description='搜相似题目。当学生需要针对某个知识点做更多练习时，搜索题库中相似的题目。')
@timer_and_token_logger
async def search_similar_questions_tool(question: str, knowledge_point: str, limit: int = 3) -> str:
    from utils.memory_service import memory_manager
    from utils.session_context import current_session_id

    sid = current_session_id.get() or "default"
    results = memory_manager.search_similar_questions(question, knowledge_point, limit)

    if not results:
        return "题库中没有找到相似的题目"

    lines = ["【相似题推荐】"]
    for i, q in enumerate(results, 1):
        kp = q.get("knowledge_point", "未标注")
        lines.append(f"{i}. [{kp}] {q['question_content']}")
        if q.get("answer"):
            lines.append(f"   答案: {q['answer']}")
    return "\n".join(lines)


# 动态生成工具列表（根据配置开关）
def _build_tools():
    """根据配置动态生成工具列表"""
    tool_list = [search_courseware, generate_quiz, generate_exam_paper, get_practice_history, analyze_wrong_reason, search_similar_questions_tool]

    if WEB_SEARCH_ENABLED:
        tool_list.append(web_search)
        tool_list.append(search_exam_format)
        logger.info("[工具注册] 联网搜索已启用")
    else:
        logger.info("[工具注册] 联网搜索已禁用")

    return tool_list


tools = _build_tools()
