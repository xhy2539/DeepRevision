import json
import re
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
_VALID_SESSION_ID = re.compile(r'^[\u4e00-\u9fa5a-zA-Z0-9_\-]{1,64}$')


def _normalize_session_id(session_id: Optional[str]) -> str:
    """规范化并校验 session_id。"""
    sid = (session_id or current_session_id.get() or "default").strip()
    if not _VALID_SESSION_ID.match(sid):
        raise ValueError("session_id 格式非法，只允许字母、数字、中文、下划线和连字符（最长64位）")
    return sid


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
    """获取练习历史，并携带 record_id 便于后续管理。"""
    from utils.memory_service import memory_manager
    sid = session_id or current_session_id.get() or "default"
    safe_limit = max(1, min(int(limit or 20), 200))
    history = memory_manager.get_practice_history(sid, safe_limit)
    if not history:
        return "暂无练习记录"

    lines = ["【练习历史】"]
    for i, h in enumerate(history, 1):
        status = "✓" if h["is_correct"] else "✗"
        kp = h["knowledge_point"] or "未标注"
        lines.append(
            f"{i}. [ID:{h.get('id')}] {status} {kp} | 你的答案:{h['user_answer']} | 正确答案:{h['correct_answer']}"
        )
        if h["wrong_reason"]:
            lines.append(f"   错因: {h['wrong_reason']}")
    return "\n".join(lines)


@tool(description='删除一条练习历史记录。建议传入 get_practice_history 返回的 record_id。')
@timer_and_token_logger
async def delete_practice_record_tool(record_id: int, session_id: Optional[str] = None) -> str:
    """按记录 ID 删除单条练习历史。"""
    from utils.memory_service import memory_manager

    sid = session_id or current_session_id.get() or "default"
    deleted = memory_manager.delete_practice_record(sid, int(record_id))
    if deleted:
        return f"已删除练习记录 ID={record_id}"
    return f"未找到练习记录 ID={record_id}"


@tool(description='清空当前会话的全部练习历史记录。')
@timer_and_token_logger
async def clear_practice_history_tool(session_id: Optional[str] = None) -> str:
    """清空会话下全部练习历史。"""
    from utils.memory_service import memory_manager

    sid = session_id or current_session_id.get() or "default"
    deleted_count = memory_manager.clear_practice_history(sid)
    return f"已清空练习历史，共删除 {deleted_count} 条记录"


@tool(description='查看会话历史对话消息，可用于审阅和管理聊天记录。')
@timer_and_token_logger
async def get_chat_history_tool(session_id: Optional[str] = None, limit: int = 30) -> str:
    """获取会话消息历史。"""
    from utils.memory_service import memory_manager

    sid = session_id or current_session_id.get() or "default"
    safe_limit = max(1, min(int(limit or 30), 200))
    messages = memory_manager.get_messages(sid)
    if not messages:
        return "暂无历史对话记录"

    selected = messages[-safe_limit:]
    lines = ["【历史对话】"]
    for i, msg in enumerate(selected, 1):
        role = "学生" if msg.get("role") == "human" else "助手"
        timestamp = msg.get("timestamp", 0)
        content = str(msg.get("content", "")).strip().replace("\n", " ")
        lines.append(f"{i}. [TS:{timestamp}] {role}: {content[:160]}")
    return "\n".join(lines)


@tool(description='删除一条历史对话消息。建议传入 get_chat_history_tool 返回的 timestamp。')
@timer_and_token_logger
async def delete_chat_message_tool(timestamp: int, session_id: Optional[str] = None) -> str:
    """按 timestamp 删除单条历史消息。"""
    from utils.memory_service import memory_manager

    sid = session_id or current_session_id.get() or "default"
    deleted = memory_manager.delete_message(sid, int(timestamp))
    if deleted:
        return f"已删除消息 TS={timestamp}"
    return f"未找到消息 TS={timestamp}"


@tool(description='清空当前会话全部历史对话消息。')
@timer_and_token_logger
async def clear_chat_history_tool(session_id: Optional[str] = None) -> str:
    """清空会话下所有消息。"""
    from utils.memory_service import memory_manager

    sid = session_id or current_session_id.get() or "default"
    success = memory_manager.clear_messages(sid)
    if success:
        return "已清空历史对话记录"
    return "会话不存在，清空失败"


@tool(description='查看全部会话列表，包含会话ID、名称与父会话信息。')
@timer_and_token_logger
async def list_sessions_tool() -> str:
    """列出全部会话。"""
    from utils.memory_service import memory_manager

    sessions = memory_manager.get_all_sessions()
    if not sessions:
        return "暂无会话记录"

    lines = ["【会话列表】"]
    for i, item in enumerate(sessions, 1):
        sid = item.get("id", "")
        name = item.get("name", "")
        parent_id = item.get("parent_id") or "-"
        lines.append(f"{i}. id={sid} | 名称={name} | parent={parent_id}")
    return "\n".join(lines)


@tool(description='创建会话。传入 session_id 和名称（name），可选 parent_id。')
@timer_and_token_logger
async def create_session_tool(session_id: str, name: Optional[str] = None, parent_id: Optional[str] = None) -> str:
    """创建新会话并写入会话索引。"""
    from utils.memory_service import memory_manager

    sid = _normalize_session_id(session_id)
    if parent_id:
        _normalize_session_id(parent_id)
    display_name = (name or sid).strip()
    memory_manager.register_session(sid, display_name, parent_id)
    return f"会话创建成功: id={sid}, name={display_name}"


@tool(description='重命名会话。传入会话ID和新名称。')
@timer_and_token_logger
async def rename_session_tool(session_id: str, new_name: str) -> str:
    """重命名已有会话。"""
    from utils.memory_service import memory_manager

    sid = _normalize_session_id(session_id)
    target_name = (new_name or "").strip()
    if not target_name:
        return "新名称不能为空"
    renamed = memory_manager.rename_session(sid, target_name)
    if not renamed:
        return f"会话不存在: {sid}"
    return f"会话重命名成功: {sid} -> {target_name}"


@tool(description='删除会话。默认会同步删除该会话的知识库与RAG缓存。')
@timer_and_token_logger
async def delete_session_tool(session_id: str, purge_knowledge_base: bool = True) -> str:
    """删除会话，可选是否清理知识库。"""
    from utils.memory_service import memory_manager

    sid = _normalize_session_id(session_id)
    token = current_session_id.set(sid)
    try:
        if purge_knowledge_base:
            from rag.vector_store import VectorStoreService
            vs = VectorStoreService()
            await vs.destroy_knowledge_base()
        clear_rag_cache(sid)
        deleted = memory_manager.clear_session(sid)
    finally:
        current_session_id.reset(token)

    if not deleted:
        return f"会话不存在: {sid}"
    if purge_knowledge_base:
        return f"会话已删除: {sid}（包含知识库）"
    return f"会话已删除: {sid}"


@tool(description='查看当前会话的课件文件列表与向量化状态。')
@timer_and_token_logger
async def list_knowledge_files_tool(session_id: Optional[str] = None) -> str:
    """查询课件列表和处理状态。"""
    sid = _normalize_session_id(session_id)

    from api.routers.knowledge import list_documents
    result = await list_documents(session_id=sid)
    files = result.get("files", []) if isinstance(result, dict) else []
    if not files:
        return f"会话 {sid} 暂无课件文件"

    lines = [f"【课件文件列表】session={sid}"]
    for i, file_info in enumerate(files, 1):
        name = file_info.get("filename", "")
        status = file_info.get("status", "unknown")
        embedded = "是" if file_info.get("embedded") else "否"
        size = int(file_info.get("size", 0) or 0)
        lines.append(f"{i}. {name} | status={status} | embedded={embedded} | size={size}B")
    return "\n".join(lines)


@tool(description='删除指定课件文件（filename）并同步删除向量。')
@timer_and_token_logger
async def delete_knowledge_file_tool(filename: str, session_id: Optional[str] = None) -> str:
    """删除单个课件文件和其向量索引。"""
    sid = _normalize_session_id(session_id)
    target = (filename or "").strip()
    if not target:
        return "filename 不能为空"

    from api.routers.knowledge import delete_file
    try:
        result = await delete_file(filename=target, session_id=sid)
        if isinstance(result, dict):
            return result.get("message", f"文件删除完成: {target}")
        return f"文件删除完成: {target}"
    except Exception as e:
        detail = getattr(e, "detail", None)
        return f"删除课件失败: {detail or str(e)}"


@tool(description='查看当前会话是否已上传样卷，并返回样卷格式与内容预览。')
@timer_and_token_logger
async def get_sample_paper_tool(session_id: Optional[str] = None) -> str:
    """查询样卷信息。"""
    sid = _normalize_session_id(session_id)

    from api.routers.knowledge import get_sample_paper
    result = await get_sample_paper(session_id=sid)
    if not isinstance(result, dict):
        return "读取样卷失败"
    if result.get("code") != 200:
        return result.get("message", "该会话尚未上传样卷")

    data = result.get("data", {}) or {}
    format_info = data.get("format", {}) or {}
    preview = str(data.get("content_preview", "") or "").strip()
    lines = [f"【样卷信息】session={sid}"]
    if format_info:
        lines.append("格式要点:")
        lines.append(json.dumps(format_info, ensure_ascii=False, indent=2)[:1200])
    if preview:
        lines.append("内容预览:")
        lines.append(preview[:1200])
    return "\n".join(lines)


@tool(description='获取练习统计：总作答数、错误数、正确率、重练率和薄弱点。')
@timer_and_token_logger
async def get_practice_stats_tool(session_id: Optional[str] = None, limit: int = 500) -> str:
    """返回会话练习统计摘要。"""
    from utils.memory_service import memory_manager

    sid = _normalize_session_id(session_id)
    safe_limit = max(1, min(int(limit or 500), 2000))
    history = memory_manager.get_practice_history(sid, limit=safe_limit)
    kp_stats = memory_manager.get_knowledge_point_stats(sid)

    total = len(history)
    wrong = sum(1 for item in history if not item.get("is_correct"))
    accuracy = round(((total - wrong) / total) * 100, 2) if total > 0 else 0.0
    retry_rate = round((wrong / total) * 100, 2) if total > 0 else 0.0
    weak_points = [kp for kp, data in kp_stats.items() if data.get("weak")]

    lines = [f"【练习统计】session={sid}"]
    lines.append(f"- 总作答数: {total}")
    lines.append(f"- 错误数: {wrong}")
    lines.append(f"- 正确率: {accuracy}%")
    lines.append(f"- 重练率: {retry_rate}%")
    lines.append(f"- 薄弱点数量: {len(weak_points)}")
    if weak_points:
        lines.append(f"- 薄弱点: {', '.join(weak_points[:10])}")
    return "\n".join(lines)


@tool(description='获取薄弱知识点排行（按正确率从低到高）。')
@timer_and_token_logger
async def get_weak_points_tool(session_id: Optional[str] = None, top_k: int = 8) -> str:
    """按正确率输出薄弱点排行。"""
    from utils.memory_service import memory_manager

    sid = _normalize_session_id(session_id)
    safe_top_k = max(1, min(int(top_k or 8), 20))
    kp_stats = memory_manager.get_knowledge_point_stats(sid)
    if not kp_stats:
        return f"会话 {sid} 暂无练习统计"

    ranked = sorted(
        kp_stats.items(),
        key=lambda x: (x[1].get("accuracy", 100.0), -x[1].get("total", 0))
    )
    picked = ranked[:safe_top_k]
    lines = [f"【薄弱点排行 Top {len(picked)}】session={sid}"]
    for i, (kp, data) in enumerate(picked, 1):
        lines.append(
            f"{i}. {kp} | 正确率={data.get('accuracy', 0)}% | "
            f"次数={data.get('total', 0)} | 正确={data.get('correct', 0)}"
        )
    return "\n".join(lines)


@tool(description='查看系统 token 使用统计。')
@timer_and_token_logger
async def get_token_usage_tool() -> str:
    """返回全局 token 统计快照。"""
    from utils.logger_handler import get_system_stats

    stats = get_system_stats()
    return "【Token统计】\n" + json.dumps(stats, ensure_ascii=False, indent=2)


@tool(description='查看系统诊断信息：token统计、RAG运行指标、会话与当前会话数据规模。')
@timer_and_token_logger
async def get_system_diagnostics_tool(session_id: Optional[str] = None) -> str:
    """汇总系统级健康指标。"""
    from utils.logger_handler import get_system_stats
    from utils.memory_service import memory_manager
    from utils.rag_metrics import rag_get_metrics_snapshot

    sid = _normalize_session_id(session_id)
    token_stats = get_system_stats()
    rag_metrics = rag_get_metrics_snapshot()
    sessions = memory_manager.get_all_sessions()
    message_count = len(memory_manager.get_messages(sid))
    practice_count = len(memory_manager.get_practice_history(sid, limit=2000))

    lines = [f"【系统诊断】session={sid}"]
    lines.append(f"- 会话总数: {len(sessions)}")
    lines.append(f"- 当前会话消息数: {message_count}")
    lines.append(f"- 当前会话练习记录数: {practice_count}")
    lines.append(f"- 当前会话RAG缓存: {'已命中' if sid in _rag_cache else '未命中'}")
    lines.append(
        f"- RAG检索: calls={rag_metrics.get('rag_retrieve_calls', 0)}, "
        f"hit={rag_metrics.get('rag_retrieve_cache_hit', 0)}, "
        f"miss={rag_metrics.get('rag_retrieve_cache_miss', 0)}"
    )
    lines.append(
        f"- Token统计: calls={token_stats.get('total_calls', 0)}, "
        f"tokens={token_stats.get('total_tokens', 0)}, "
        f"prompt={token_stats.get('prompt_tokens', 0)}, "
        f"completion={token_stats.get('completion_tokens', 0)}"
    )
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
    tool_list = [
        search_courseware,
        generate_quiz,
        generate_exam_paper,
        get_practice_history,
        delete_practice_record_tool,
        clear_practice_history_tool,
        get_chat_history_tool,
        delete_chat_message_tool,
        clear_chat_history_tool,
        list_sessions_tool,
        create_session_tool,
        rename_session_tool,
        delete_session_tool,
        list_knowledge_files_tool,
        delete_knowledge_file_tool,
        get_sample_paper_tool,
        get_practice_stats_tool,
        get_weak_points_tool,
        get_token_usage_tool,
        get_system_diagnostics_tool,
        analyze_wrong_reason,
        search_similar_questions_tool,
    ]

    if WEB_SEARCH_ENABLED:
        tool_list.append(web_search)
        tool_list.append(search_exam_format)
        logger.info("[工具注册] 联网搜索已启用")
    else:
        logger.info("[工具注册] 联网搜索已禁用")

    return tool_list


tools = _build_tools()
