import inspect
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional


TYPE_ORDER = ["choice", "fill", "judge", "essay"]
TYPE_LABELS = {"choice": "选择题", "fill": "填空题", "judge": "判断题", "essay": "简答题"}
TYPE_ALIASES = {
    "choice": ["选择题", "单选题", "多选题", "客观题"],
    "fill": ["填空题"],
    "judge": ["判断题", "是非题", "正误题"],
    "essay": ["简答题", "问答题", "解答题", "论述题", "分析题"],
}


SearchExamFormatFunc = Callable[[str], Awaitable[str] | str]


def _empty_dist() -> Dict[str, int]:
    return {key: 0 for key in TYPE_ORDER}


def _safe_int(value: Any, default: int = 0, maximum: int = 80) -> int:
    try:
        number = int(value)
    except Exception:
        return default
    return max(0, min(number, maximum))


def _course_name_from_query(query: str, fallback: str) -> str:
    match = re.search(r"(操作系统|数据结构|计算机网络|数据库|软件工程|高等数学|线性代数|大学物理|英语)", str(query or ""))
    return match.group(1) if match else str(fallback or "大学课程")


def _alias_pattern() -> str:
    aliases: List[str] = []
    for names in TYPE_ALIASES.values():
        aliases.extend(names)
    return "|".join(sorted((re.escape(item) for item in aliases), key=len, reverse=True))


def _key_for_label(label: str) -> Optional[str]:
    for key, aliases in TYPE_ALIASES.items():
        if label in aliases:
            return key
    return None


def _parse_distribution(text: str) -> Dict[str, int]:
    """从用户请求、样卷或联网摘要中抽取题型数量。"""
    dist = _empty_dist()
    raw = str(text or "")
    if not raw.strip():
        return dist
    label_re = _alias_pattern()
    patterns = [
        rf"(?P<label>{label_re})[^\n。；;]*?(?:共|合计)\s*(?P<count>\d+)\s*(?:道|题|个)",
        rf"(?P<label>{label_re})\s*[:：]?\s*(?P<count>\d+)\s*(?:道|题|个)?",
        rf"(?P<count>\d+)\s*(?:道|题|个)?\s*(?P<label>{label_re})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, raw):
            key = _key_for_label(match.group("label"))
            if not key:
                continue
            count = _safe_int(match.group("count"))
            if count > 0 and dist[key] == 0:
                dist[key] = count
    return dist


def _route_param_distribution(route_params: Dict[str, Any] | None) -> Dict[str, int]:
    """从 supervisor 已解析参数中读取题型分布。"""
    params = route_params or {}
    raw_dist = params.get("quantity_dist") if isinstance(params.get("quantity_dist"), dict) else {}
    dist = _empty_dist()
    for key in TYPE_ORDER:
        if key in raw_dist:
            dist[key] = _safe_int(raw_dist.get(key))
    if any(dist.values()):
        return dist

    quiz_type = str(params.get("quiz_type") or "").strip()
    num = _safe_int(params.get("num"), default=0, maximum=30)
    if quiz_type and num > 0:
        key = _key_for_label(quiz_type)
        if key:
            dist[key] = num
    return dist


def _has_counts(dist: Dict[str, int]) -> bool:
    return any(int(dist.get(key, 0) or 0) > 0 for key in TYPE_ORDER)


def _types_from_dist(dist: Dict[str, int]) -> List[str]:
    return [TYPE_LABELS[key] for key in TYPE_ORDER if int(dist.get(key, 0) or 0) > 0]


def _plan_items_from_dist(dist: Dict[str, int]) -> List[Dict[str, Any]]:
    return [
        {"key": key, "type": TYPE_LABELS[key], "count": int(dist.get(key, 0) or 0)}
        for key in TYPE_ORDER
        if int(dist.get(key, 0) or 0) > 0
    ]


def _default_dist(route: str) -> Dict[str, int]:
    if route == "quiz":
        return {"choice": 3, "fill": 0, "judge": 0, "essay": 0}
    return {"choice": 10, "fill": 5, "judge": 5, "essay": 3}


def _build_plan(
    *,
    dist: Dict[str, int],
    source: str,
    reason: str,
    raw_reference: str = "",
) -> Dict[str, Any]:
    total = sum(int(dist.get(key, 0) or 0) for key in TYPE_ORDER)
    return {
        "source": source,
        "reason": reason,
        "quiz_types": _types_from_dist(dist),
        "quantity_dist": {key: int(dist.get(key, 0) or 0) for key in TYPE_ORDER},
        "total_questions": total,
        "question_type_plan": _plan_items_from_dist(dist),
        "raw_reference": str(raw_reference or "")[:2000],
    }


def _should_fetch_web(query: str, route: str) -> bool:
    text = re.sub(r"\s+", "", str(query or ""))
    if route != "exam":
        return False
    return bool(
        re.search(r"(网络|联网|网上|互联网).{0,10}(题型|试卷|格式|参考|决定)", text)
        or re.search(r"(题型).{0,10}(参考|决定|自动|你来)", text)
        or re.search(r"(综合测试卷|综合试卷|期末试卷|生成试卷)", text)
    )


async def _maybe_call_search(func: SearchExamFormatFunc, course_name: str) -> str:
    result = func(course_name)
    if inspect.isawaitable(result):
        result = await result
    return str(result or "")


async def _default_search_exam_format(course_name: str) -> str:
    from agent.tools.agent_tools import search_exam_format

    return str(await search_exam_format.ainvoke({"course_name": course_name}) or "")


async def resolve_question_type_plan(
    *,
    query: str,
    route: str,
    session_id: str,
    sample_paper_context: str = "",
    search_observation: str = "",
    route_params: Dict[str, Any] | None = None,
    search_exam_format_func: SearchExamFormatFunc | None = None,
) -> Dict[str, Any]:
    """按 用户显式 > 样卷 > 联网参考 > 默认 的优先级解析题型。"""
    safe_route = str(route or "").strip() or "quiz"
    explicit_dist = _parse_distribution(query)
    if not _has_counts(explicit_dist):
        explicit_dist = _route_param_distribution(route_params)
    if _has_counts(explicit_dist):
        return _build_plan(
            dist=explicit_dist,
            source="user_explicit",
            reason="用户显式指定了题型或数量。",
            raw_reference=query,
        )

    sample_dist = _parse_distribution(sample_paper_context)
    if _has_counts(sample_dist):
        return _build_plan(
            dist=sample_dist,
            source="sample_paper",
            reason="检测到已上传样卷中的题型分布。",
            raw_reference=sample_paper_context,
        )

    web_reference = str(search_observation or "")
    if not web_reference and _should_fetch_web(query, safe_route):
        search_func = search_exam_format_func or _default_search_exam_format
        course_name = _course_name_from_query(query, session_id)
        try:
            web_reference = await _maybe_call_search(search_func, course_name)
        except Exception as exc:
            web_reference = f"联网题型参考获取失败：{exc}"

    web_dist = _parse_distribution(web_reference)
    if _has_counts(web_dist):
        return _build_plan(
            dist=web_dist,
            source="web_search",
            reason="无显式题型和样卷时，使用联网题型参考。",
            raw_reference=web_reference,
        )

    default_dist = _default_dist(safe_route)
    source = "default"
    reason = "未获得用户显式题型、样卷或高质量联网题型参考，使用系统默认分布。"
    return _build_plan(dist=default_dist, source=source, reason=reason, raw_reference=web_reference)
