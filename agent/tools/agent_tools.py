import json
import re
import os
import asyncio
import ast
import hashlib
from typing import Optional, List, Tuple, Any
from urllib.parse import urlparse

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
_UNSAFE_SEARCH_TERMS = {
    "成人视频",
    "成人内容",
    "色情",
    "黄色",
    "裸聊",
    "4p",
    "操死",
    "鲍鱼粉嫩",
}
_LOW_VALUE_SEARCH_DOMAINS = {
    "douyin.com",
    "iesdouyin.com",
    "tiktok.com",
    "pinterest.com",
    "kuaishou.com",
}
_EXAM_FORMAT_TERMS = (
    "期末",
    "考试",
    "试卷",
    "题型",
    "卷面",
    "选择题",
    "判断题",
    "填空题",
    "简答题",
    "final exam",
)
_EXAM_PAPER_TERMS = (
    "期末考试",
    "试卷",
    "试题",
    "真题",
    "答案",
    "a卷",
    "b卷",
    "闭卷",
    "开卷",
    "课程考试",
    "final exam",
    "exam paper",
)
_EXAM_FILE_HINTS = (".pdf", ".doc", ".docx", ".ppt", ".pptx")
_COURSE_ALIASES = {
    "操作系统": ("操作系统", "operating system", "operating systems"),
    "数据结构": ("数据结构", "data structure", "data structures"),
    "计算机网络": ("计算机网络", "computer network", "computer networking"),
    "数据库": ("数据库", "database", "database systems"),
    "软件工程": ("软件工程", "software engineering"),
}


def _normalize_session_id(session_id: Optional[str]) -> str:
    """规范化并校验 session_id。"""
    sid = (session_id or current_session_id.get() or "default").strip()
    if not _VALID_SESSION_ID.match(sid):
        raise ValueError("session_id 格式非法，只允许字母、数字、中文、下划线和连字符（最长64位）")
    return sid


def _clean_search_text(value: Any, limit: int = 260) -> str:
    """清洗搜索标题/摘要，避免前端展示大段噪声。"""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _source_domain(url: str) -> str:
    """从 URL 提取展示用域名。"""
    parsed = urlparse(str(url or ""))
    domain = (parsed.netloc or parsed.path).split("/")[0].lower()
    return domain[4:] if domain.startswith("www.") else domain


def _normalize_search_docs(raw_docs: Any) -> List[dict]:
    """兼容不同搜索后端的返回结构，统一为 title/link/snippet。"""
    if isinstance(raw_docs, str):
        try:
            raw_docs = json.loads(raw_docs)
        except Exception:
            return [{"title": "搜索摘要", "link": "", "snippet": raw_docs}]
    if isinstance(raw_docs, dict):
        raw_docs = raw_docs.get("results") or raw_docs.get("documents") or [raw_docs]
    if not isinstance(raw_docs, list):
        return []

    docs: List[dict] = []
    for item in raw_docs:
        if not isinstance(item, dict):
            continue
        link = str(item.get("link") or item.get("url") or item.get("href") or "").strip()
        title = _clean_search_text(
            item.get("title") or item.get("name") or item.get("heading") or link or "未命名结果",
            limit=120,
        )
        snippet = _clean_search_text(
            item.get("snippet") or item.get("body") or item.get("content") or item.get("description") or "",
            limit=320,
        )
        if not (title or snippet or link):
            continue
        docs.append({"title": title, "link": link, "snippet": snippet, "domain": _source_domain(link)})
    return docs


def _is_low_value_search_doc(doc: dict) -> bool:
    """过滤成人、短视频、纯娱乐站点等不适合学习场景的结果。"""
    text = f"{doc.get('title', '')} {doc.get('snippet', '')}".lower()
    domain = str(doc.get("domain") or _source_domain(str(doc.get("link") or ""))).lower()
    if any(term.lower() in text for term in _UNSAFE_SEARCH_TERMS):
        return True
    return any(domain == bad or domain.endswith("." + bad) for bad in _LOW_VALUE_SEARCH_DOMAINS)


def _dedupe_search_docs(docs: List[dict], limit: int = 5) -> List[dict]:
    """按链接和标题去重，并限制返回数量。"""
    result: List[dict] = []
    seen = set()
    for doc in _normalize_search_docs(docs):
        if _is_low_value_search_doc(doc):
            continue
        key = str(doc.get("link") or doc.get("title") or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(doc)
        if len(result) >= max(1, min(int(limit or 5), 10)):
            break
    return result


def _format_search_results(query: str, docs: Any, limit: int = 5) -> str:
    """把结构化搜索结果格式化为可追溯 Markdown。"""
    normalized = _dedupe_search_docs(_normalize_search_docs(docs), limit=limit)
    if not normalized:
        return f"未获取到与“{query}”相关的可信联网搜索结果。"

    lines = [
        f"**检索词**：{_clean_search_text(query, limit=160)}",
        "**说明**：联网结果仅作辅助参考，课程结论仍应优先以已上传课件为准。",
        "",
    ]
    for index, doc in enumerate(normalized, 1):
        title = doc.get("title") or "未命名结果"
        link = doc.get("link") or ""
        domain = doc.get("domain") or _source_domain(link) or "未知来源"
        snippet = doc.get("snippet") or "该结果未返回摘要。"
        title_part = f"[{title}]({link})" if link else title
        lines.extend(
            [
                f"{index}. {title_part}",
                f"   - 来源：{domain}",
                f"   - 摘要：{snippet}",
            ]
        )
    return "\n".join(lines)


def _course_terms(course_name: str) -> Tuple[str, ...]:
    """生成课程名及常见英文别名，用于过滤题型搜索噪声。"""
    course = str(course_name or "").strip()
    return _COURSE_ALIASES.get(course, (course,)) if course else tuple()


def _filter_exam_format_docs(course_name: str, docs: Any, limit: int = 5) -> List[dict]:
    """保留同时命中课程名和考试题型语义的搜索结果。"""
    terms = tuple(term.lower() for term in _course_terms(course_name) if term)
    candidates = []
    for doc in _dedupe_search_docs(_normalize_search_docs(docs), limit=20):
        text = f"{doc.get('title', '')} {doc.get('snippet', '')}".lower()
        course_hit = any(term in text for term in terms) if terms else True
        exam_score = sum(1 for term in _EXAM_FORMAT_TERMS if term.lower() in text)
        if not course_hit or exam_score <= 0:
            continue
        candidates.append((exam_score, doc))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in candidates[: max(1, min(int(limit or 5), 10))]]


def _filter_exam_paper_docs(course_name: str, docs: Any, limit: int = 5) -> List[dict]:
    """筛出更像真实试卷/真题文件的联网结果，并降低普通总结页权重。"""
    terms = tuple(term.lower() for term in _course_terms(course_name) if term)
    candidates = []
    for doc in _dedupe_search_docs(_normalize_search_docs(docs), limit=30):
        title = str(doc.get("title") or "")
        link = str(doc.get("link") or "")
        snippet = str(doc.get("snippet") or "")
        domain = str(doc.get("domain") or _source_domain(link)).lower()
        text = f"{title} {snippet} {link}".lower()
        title_link_text = f"{title} {link}".lower()
        course_hit = any(term in text for term in terms) if terms else True
        paper_score = sum(1 for term in _EXAM_PAPER_TERMS if term.lower() in text)
        strong_paper_score = sum(1 for term in _EXAM_PAPER_TERMS if term.lower() in title_link_text)
        file_score = 2 if any(hint in link.lower() or hint in title.lower() for hint in _EXAM_FILE_HINTS) else 0
        edu_score = 1 if domain.endswith("edu.cn") or ".edu." in domain or "edu.cn" in link.lower() else 0
        if not course_hit or paper_score <= 0 or (strong_paper_score <= 0 and file_score <= 0):
            continue
        candidates.append((paper_score + file_score + edu_score, file_score, edu_score, doc))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [doc for *_score, doc in candidates[: max(1, min(int(limit or 5), 10))]]


def _format_exam_paper_reference(course_name: str, query: str, docs: Any, limit: int = 5) -> str:
    """把真实试卷搜索结果整理为出卷风格参考，明确禁止复制原题。"""
    filtered = _filter_exam_paper_docs(course_name, docs, limit=limit)
    if not filtered:
        return ""

    lines = [
        "【联网真实试卷参考】",
        f"课程：{_clean_search_text(course_name or '大学课程', limit=80)}",
        f"检索词：{_clean_search_text(query, limit=160)}",
        "用途：仅参考题型结构、难度分布、题干表达风格和是否包含答案。",
        "约束：禁止照抄或改写真实试卷原题；题目事实依据仍以课件/RAG内容为准。",
        "",
    ]
    for index, doc in enumerate(filtered, 1):
        title = doc.get("title") or "未命名试卷参考"
        link = doc.get("link") or ""
        domain = doc.get("domain") or _source_domain(link) or "未知来源"
        snippet = doc.get("snippet") or "该结果未返回摘要。"
        title_part = f"[{title}]({link})" if link else title
        lines.extend(
            [
                f"{index}. {title_part}",
                f"   - 来源：{domain}",
                f"   - 可参考点：{snippet}",
            ]
        )
    return "\n".join(lines)


async def fetch_exam_paper_reference(course_name: str, topic_hint: str = "", limit: int = 5) -> str:
    """联网查找真实试卷/真题参考；找不到时返回空串，由调用方降级。"""
    if not WEB_SEARCH_ENABLED:
        return ""

    course = str(course_name or topic_hint or "大学课程").strip() or "大学课程"
    aliases = [term for term in _course_terms(course) if term]
    primary = aliases[0] if aliases else course
    queries = [
        f"{primary} 期末考试 试卷 filetype:pdf",
        f"{primary} 期末考试 真题 答案",
        f"{primary} 期末考试 试题 doc",
        f"site:edu.cn {primary} 期末考试 试卷",
    ]
    if len(aliases) > 1:
        queries.append(f"{aliases[1]} final exam paper pdf")

    all_docs: List[dict] = []
    best_query = queries[0]
    for query in queries[:5]:
        try:
            docs = await asyncio.to_thread(_run_duckduckgo_results, query, 6)
            all_docs.extend(_normalize_search_docs(docs))
            if _filter_exam_paper_docs(course, all_docs, limit=limit):
                best_query = query
                break
        except Exception as e:
            logger.warning(f"[真实试卷搜索] query={query} 失败: {e}")
            continue

    filtered = _filter_exam_paper_docs(course, all_docs, limit=limit)
    if not filtered:
        logger.info(f"[真实试卷搜索] {course} -> 未找到高相关真实试卷参考")
        return ""
    logger.info(f"[真实试卷搜索] {course} -> 获取 {len(filtered)} 条真实试卷参考")
    return _format_exam_paper_reference(course, best_query, filtered, limit=limit)


def _run_duckduckgo_results(query: str, num_results: int = 6) -> List[dict]:
    """调用 DuckDuckGo 结构化结果接口，保留标题和链接。"""
    from langchain_community.tools import DuckDuckGoSearchResults

    search = DuckDuckGoSearchResults(num_results=num_results, output_format="list")
    return _normalize_search_docs(search.run(query))


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


def _extract_courseware_hits(context: str, limit: int = 8) -> List[Tuple[str, str]]:
    """从 retrieve_context 文本中提取“来源文件 + 片段”用于快速展示。"""
    text = str(context or "")
    if not text:
        return []
    hits: List[Tuple[str, str]] = []
    seen = set()
    pattern = r"\[参考资料\d+\]:参考资料:(.*?)\|参考元数据:(\{.*?\})(?:\n|$)"
    for match in re.finditer(pattern, text, flags=re.DOTALL):
        raw_snippet = re.sub(r"\s+", " ", str(match.group(1) or "")).strip()
        snippet = raw_snippet[:140]
        meta_text = str(match.group(2) or "").strip()
        source = "未知来源"
        try:
            meta: Any = ast.literal_eval(meta_text)
            if isinstance(meta, dict):
                source = str(meta.get("source_filename") or meta.get("source") or source).strip() or source
        except Exception:
            pass
        key = f"{source}|{snippet}"
        if key in seen or not snippet:
            continue
        seen.add(key)
        hits.append((source, snippet))
        if len(hits) >= max(1, min(int(limit or 8), 20)):
            break
    return hits


# ==================== 工具定义 ====================

@tool(description='从课件知识库中检索知识点和参考资料。传入的query应该是一个具体考点名词。')
@timer_and_token_logger
async def search_courseware(query: str) -> str:
    """快速课件检索：限时检索 + 片段提炼，避免走慢总结链路。"""
    q = str(query or "").strip()
    if not q:
        return "请输入要检索的课件关键词。"

    rag = await _get_rag()
    try:
        # 只走检索链路，不走总结链路，避免 60s 级别超时。
        context = await asyncio.wait_for(rag.retrieve_context(q, mode="rag_chat"), timeout=14.0)
    except asyncio.TimeoutError:
        return (
            "课件检索超时（已中止本次查询）。\n"
            "建议缩小范围后重试，例如：`总结 C0-Overview 的核心知识点`。"
        )
    except Exception as e:
        logger.warning(f"[search_courseware] 检索失败 query={q[:80]} err={e}")
        return f"课件检索失败：{str(e)}"

    hits = _extract_courseware_hits(context, limit=8)
    if not hits:
        return (
            f"未检索到与“{q}”相关的课件片段。\n"
            "建议改用更具体的关键词（章节名/概念名/文件名）。"
        )

    lines = [f"【课件检索结果】关键词：{q}"]
    for idx, (source, snippet) in enumerate(hits, 1):
        lines.append(f"{idx}. {source}：{snippet}")
    lines.append("")
    lines.append("如需我继续，可直接说：`按每个文件各总结3条要点`。")
    return "\n".join(lines)


@tool(description='联网搜索信息。当需要验证答案、查找默认试卷风格、查询考试规范时使用。')
@timer_and_token_logger
async def web_search(query: str) -> str:
    """
    联网搜索工具（用于辅助验证，不能替代课件资料）
    优先返回带 URL 的结构化结果，失败则尝试 Tavily
    """
    if not WEB_SEARCH_ENABLED:
        return "联网搜索功能已关闭，请在配置文件中启用"

    q = str(query or "").strip()
    if not q:
        return "请输入要联网搜索的关键词。"

    try:
        docs = await asyncio.to_thread(_run_duckduckgo_results, q, 6)
        trusted_docs = _dedupe_search_docs(_normalize_search_docs(docs), limit=5)
        if trusted_docs:
            logger.info(f"[联网搜索] {q} -> {len(trusted_docs)} 条可信结构化结果")
            return _format_search_results(q, trusted_docs, limit=5)
        if docs:
            logger.info(f"[联网搜索] {q} -> DuckDuckGo 结果均被过滤，尝试备用搜索")
    except Exception as e:
        logger.warning(f"[联网搜索] DuckDuckGo 失败: {e}")

    try:
        from langchain_community.tools import TavilySearchResults
        search = TavilySearchResults(max_results=5)
        docs = await asyncio.to_thread(search.run, q)
        if docs:
            logger.info(f"[联网搜索] Tavily 备用成功 -> {len(_normalize_search_docs(docs))} 条结果")
            return _format_search_results(q, docs, limit=5)
    except Exception as e:
        logger.warning(f"[联网搜索] Tavily 也失败: {e}")

    return "联网搜索暂时不可用"


@tool(description='搜索大学期末考试试卷格式范例。当用户没有上传样卷时，可以搜索该学科的典型试卷格式。')
@timer_and_token_logger
async def search_exam_format(course_name: str) -> str:
    """搜索默认试卷格式"""
    if not WEB_SEARCH_ENABLED:
        return "联网搜索功能已关闭，无法获取在线试卷格式。请上传样卷或使用默认格式。"

    course = str(course_name or "大学课程").strip()
    query = f"{course} 期末考试 试卷 题型 结构 选择题 判断题 填空题 简答题"
    try:
        docs = await asyncio.to_thread(_run_duckduckgo_results, query, 8)
        filtered = _filter_exam_format_docs(course, docs, limit=5)
        if not filtered:
            logger.info(f"[试卷格式搜索] {course} -> 无高相关结构化结果")
            return (
                f"【{course}试卷格式参考】\n"
                "未获取到同时匹配课程名和考试题型的高相关联网结果。"
                "建议上传样卷，或使用系统默认的选择题/判断题/填空题/简答题结构。"
            )
        logger.info(f"[试卷格式搜索] {course} -> 获取 {len(filtered)} 条高相关格式参考")
        return (
            f"【{course}试卷格式参考】\n"
            "以下结果仅用于参考试卷结构，不替代已上传课件中的知识依据。\n\n"
            + _format_search_results(query, filtered, limit=5)
        )
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


@tool(description='回填历史练习记录的知识点字段。可用 dry_run=true 先预览将更新的条数。')
@timer_and_token_logger
async def backfill_practice_kp_tool(
    session_id: Optional[str] = None,
    limit: int = 2000,
    dry_run: bool = True,
) -> str:
    """清洗并回填历史练习的 knowledge_point。"""
    from utils.memory_service import memory_manager

    sid = session_id or current_session_id.get() or "default"
    summary = memory_manager.backfill_practice_knowledge_points(
        session_id=sid,
        limit=max(1, min(int(limit or 2000), 20000)),
        dry_run=bool(dry_run),
    )
    return (
        f"考点回填完成（session={sid}）："
        f"扫描 {summary.get('scanned', 0)} 条，候选更新 {summary.get('candidate_updates', 0)} 条，"
        f"实际更新 {summary.get('updated', 0)} 条，dry_run={bool(summary.get('dry_run', 0))}"
    )


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

    lines = [f"【课件文件列表】session={sid}, count={len(files)}"]
    for i, file_info in enumerate(files, 1):
        name = file_info.get("filename", "")
        status = file_info.get("status", "unknown")
        embedded = "是" if file_info.get("embedded") else "否"
        size = int(file_info.get("size", 0) or 0)
        lines.append(f"{i}. {name} | status={status} | embedded={embedded} | size={size}B")
    return "\n".join(lines)


@tool(description='列出当前会话向量化失败的课件文件；可通过 retry=true 触发重试。')
@timer_and_token_logger
async def list_failed_uploads_tool(
    session_id: Optional[str] = None,
    retry: bool = False,
    wait_for_retry: bool = False,
) -> str:
    """
    列出失败文件；可选触发重试。
    retry=true 时会把 failed 文件状态改为 processing 并重试。
    """
    sid = _normalize_session_id(session_id)

    from api.routers.knowledge import list_documents, _update_ingest_status, process_document_task

    result = await list_documents(session_id=sid)
    files = result.get("files", []) if isinstance(result, dict) else []
    failed_files = [f for f in files if str(f.get("status", "")).strip() == "failed"]
    if not failed_files:
        return f"会话 {sid} 当前没有向量化失败文件"

    failed_names = [str(f.get("filename", "")).strip() for f in failed_files if str(f.get("filename", "")).strip()]
    lines = [f"【失败文件】session={sid}, count={len(failed_names)}"]
    for i, f in enumerate(failed_files, 1):
        lines.append(f"{i}. {f.get('filename', '')} | detail={f.get('detail', '')}")

    if not retry:
        lines.append("如需重试可调用：retry=true")
        return "\n".join(lines)

    _update_ingest_status(
        sid,
        {name: {"status": "processing", "detail": "工具触发重试中"} for name in failed_names},
    )

    if wait_for_retry:
        # 在后台线程中执行重试，避免阻塞事件循环。
        await asyncio.to_thread(process_document_task, failed_names, sid)
        refreshed = await list_documents(session_id=sid)
        refreshed_files = refreshed.get("files", []) if isinstance(refreshed, dict) else []
        still_failed = [
            str(f.get("filename", "")).strip()
            for f in refreshed_files
            if str(f.get("status", "")).strip() == "failed"
        ]
        lines.append(
            f"重试已完成：提交 {len(failed_names)} 个，"
            f"当前仍失败 {len([n for n in still_failed if n])} 个"
        )
        if still_failed:
            lines.append("仍失败文件：" + ", ".join([n for n in still_failed if n]))
        return "\n".join(lines)

    asyncio.create_task(asyncio.to_thread(process_document_task, failed_names, sid))
    lines.append(f"已提交重试任务：{len(failed_names)} 个文件（后台执行中）")
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


@tool(description='删除与指定保留课件内容完全相同的重复课件。传入 keep_filename，仅删除同 MD5 的其他文件。')
@timer_and_token_logger
async def delete_duplicate_knowledge_files_tool(keep_filename: str, session_id: Optional[str] = None) -> str:
    """按 MD5 删除重复课件，仅保留 keep_filename。"""
    sid = _normalize_session_id(session_id)
    keep = os.path.basename((keep_filename or "").strip())
    if not keep:
        return "keep_filename 不能为空"

    from utils.config_handler import chroma_conf
    from utils.path_tool import get_abs_path
    from api.routers.knowledge import delete_file

    data_root = get_abs_path(chroma_conf["data_path"])
    session_data_dir = os.path.join(data_root, sid)
    keep_path = os.path.join(session_data_dir, keep)
    if not os.path.isfile(keep_path):
        return f"保留文件不存在: {keep}"

    allowed = {".txt", ".pdf", ".docx", ".doc", ".ppt", ".pptx", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    with open(keep_path, "rb") as f:
        keep_md5 = hashlib.md5(f.read()).hexdigest()

    duplicates: List[str] = []
    for fname in os.listdir(session_data_dir):
        if fname == keep or fname.startswith("."):
            continue
        if os.path.splitext(fname)[1].lower() not in allowed:
            continue
        fpath = os.path.join(session_data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "rb") as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
        except Exception:
            continue
        if file_md5 == keep_md5:
            duplicates.append(fname)

    if not duplicates:
        return f"未发现与 {keep} 内容完全相同的重复课件。"

    lines = [f"【重复课件删除】session={sid}", f"保留: {keep}"]
    for fname in sorted(duplicates):
        result = await delete_file(filename=fname, session_id=sid)
        if isinstance(result, dict):
            lines.append(f"删除: {fname} -> {result.get('message', '完成')}")
        else:
            lines.append(f"删除: {fname} -> 完成")
    lines.append(f"共删除 {len(duplicates)} 个重复文件。")
    return "\n".join(lines)


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


@tool(description='导出试卷为 Word 文档（docx），返回下载地址与服务器文件路径。')
@timer_and_token_logger
async def export_exam_docx_tool(
    exam_paper: str,
    course_name: str = "期末考试",
    include_answers: bool = True,
) -> str:
    """导出试卷 docx。"""
    from api.routers.exam_export import export_docx, ExamExportRequest, TEMP_DIR

    if not str(exam_paper or "").strip():
        return "exam_paper 不能为空"

    req = ExamExportRequest(
        exam_paper=str(exam_paper),
        course_name=str(course_name or "期末考试"),
        include_answers=bool(include_answers),
    )
    result = await export_docx(req)
    if not isinstance(result, dict):
        return "导出失败：返回格式异常"
    if not result.get("success"):
        return f"导出失败：{result}"

    filename = str(result.get("filename", "")).strip()
    download_url = str(result.get("download_url", "")).strip()
    abs_path = os.path.realpath(os.path.join(TEMP_DIR, filename)) if filename else ""
    return (
        "导出成功\n"
        f"- 文件名: {filename}\n"
        f"- 下载地址: {download_url}\n"
        f"- 服务器路径: {abs_path}"
    )


@tool(description='导出答案卷为 Word 文档（docx），返回下载地址与服务器文件路径。')
@timer_and_token_logger
async def export_answer_sheet_tool(
    exam_paper: str,
    course_name: str = "期末考试",
) -> str:
    """导出答案卷 docx。"""
    from api.routers.exam_export import generate_answer_sheet, AnswerSheetRequest, TEMP_DIR

    if not str(exam_paper or "").strip():
        return "exam_paper 不能为空"

    req = AnswerSheetRequest(
        exam_paper=str(exam_paper),
        course_name=str(course_name or "期末考试"),
    )
    result = await generate_answer_sheet(req)
    if not isinstance(result, dict):
        return "导出失败：返回格式异常"
    if not result.get("success"):
        return f"导出失败：{result}"

    filename = str(result.get("filename", "")).strip()
    download_url = str(result.get("download_url", "")).strip()
    abs_path = os.path.realpath(os.path.join(TEMP_DIR, filename)) if filename else ""
    return (
        "答案卷导出成功\n"
        f"- 文件名: {filename}\n"
        f"- 下载地址: {download_url}\n"
        f"- 服务器路径: {abs_path}"
    )


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


@tool(description='总结指定课件文件的全部内容。传入 filename（课件文件名）和可选的 focus（关注点，如"重点知识"）。会读取该文件所有 chunk 并做 map-reduce 总结，适合"帮我总结这份课件"类请求。')
@timer_and_token_logger
async def summarize_document(filename: str, focus: str = "核心知识点和重点内容") -> str:
    """获取文件全部 chunk 并 map-reduce 总结。"""
    from model.factory import light_chat_model
    from langchain_core.messages import HumanMessage

    fname = (filename or "").strip()
    if not fname:
        return "请传入课件文件名（filename）。"

    rag = await _get_rag()
    vs = rag.vector_store_service.vector_store

    # 获取该文件所有 chunk
    try:
        result = vs.get(where={"source_filename": fname}, include=["documents", "metadatas"])
    except Exception as e:
        return f"读取课件向量失败：{e}"

    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    if not docs:
        return f"未找到课件“{fname}”的向量数据，请确认文件名是否正确且已完成向量化。"

    # 按 page_number 排序
    paired = sorted(
        zip(docs, metas),
        key=lambda x: int((x[1] or {}).get("page_number") or 0)
    )
    chunks = [d for d, _ in paired]

    # map 阶段：每批 ~3000 字
    BATCH = 3000
    batches, cur, cur_len = [], [], 0
    for c in chunks:
        if cur_len + len(c) > BATCH and cur:
            batches.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(c)
        cur_len += len(c)
    if cur:
        batches.append("\n".join(cur))

    map_prompt = "请提炼以下课件片段的核心知识点（{focus}），用简洁的要点列表输出，不超过300字：\n\n{text}"
    summaries = []
    for batch in batches:
        try:
            resp = await asyncio.wait_for(
                light_chat_model.ainvoke([HumanMessage(content=map_prompt.format(focus=focus, text=batch))]),
                timeout=20.0
            )
            summaries.append(resp.content.strip())
        except Exception as e:
            logger.warning(f"[summarize_document] map batch 失败: {e}")

    if not summaries:
        return "总结失败，请稍后重试。"

    # reduce 阶段
    combined = "\n\n".join(summaries)
    reduce_prompt = (
        f"以下是课件“{fname}”各部分的要点摘要，请整合成一份完整的课件总结（关注：{focus}），"
        f"结构清晰，分章节或主题列出，不超过800字：\n\n{combined}"
    )
    try:
        final = await asyncio.wait_for(
            light_chat_model.ainvoke([HumanMessage(content=reduce_prompt)]),
            timeout=30.0
        )
        return f"【课件总结】{fname}\n\n{final.content.strip()}"
    except Exception as e:
        # reduce 超时时直接返回 map 结果
        return f"【课件总结（分段）】{fname}\n\n" + "\n\n---\n\n".join(summaries)


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
        summarize_document,
        generate_quiz,
        generate_exam_paper,
        get_practice_history,
        delete_practice_record_tool,
        clear_practice_history_tool,
        backfill_practice_kp_tool,
        get_chat_history_tool,
        delete_chat_message_tool,
        clear_chat_history_tool,
        list_sessions_tool,
        create_session_tool,
        rename_session_tool,
        delete_session_tool,
        list_knowledge_files_tool,
        list_failed_uploads_tool,
        delete_knowledge_file_tool,
        delete_duplicate_knowledge_files_tool,
        get_sample_paper_tool,
        export_exam_docx_tool,
        export_answer_sheet_tool,
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
