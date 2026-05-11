import json
import re
from typing import Any, Dict, Iterable, List, Optional


_REFERENCE_PATTERNS = [
    r"刚才",
    r"上次",
    r"前面",
    r"上面",
    r"上一轮",
    r"搜索结果",
    r"联网结果",
    r"查到",
    r"题型分布",
    r"题型参考",
    r"那套卷",
    r"这套卷",
    r"按.*(结果|题型|试卷|卷子)",
    r"基于.*(结果|题型|试卷|卷子)",
    r"参考.*(结果|题型|试卷|卷子)",
]


def should_use_tool_context(query: str) -> bool:
    """判断当前请求是否在引用近期工具结果或生成产物。"""
    text = re.sub(r"\s+", "", str(query or "").strip())
    if not text:
        return False
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _REFERENCE_PATTERNS)


def _safe_json(value: Any, max_chars: int = 500) -> str:
    try:
        text = json.dumps(value if value is not None else {}, ensure_ascii=False, default=str)
    except Exception:
        text = str(value or {})
    return text[:max_chars]


def _clip_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def format_tool_contexts_for_prompt(
    contexts: Iterable[Dict[str, Any]],
    *,
    query: str = "",
    max_items: int = 6,
    max_item_chars: int = 1200,
    max_total_chars: int = 4200,
) -> str:
    """把近期工具上下文格式化为可注入 prompt 的结构化文本。"""
    selected: List[Dict[str, Any]] = [item for item in contexts or [] if isinstance(item, dict)]
    if not selected:
        return ""
    lines: List[str] = ["【近期工具上下文】"]
    used = len(lines[0])
    for idx, item in enumerate(selected[-max_items:], start=1):
        tool_name = str(item.get("tool_name") or "tool").strip()
        summary = str(item.get("summary") or "").strip()
        args = _safe_json(item.get("args") or {})
        content = _clip_text(item.get("content") or "", max_item_chars)
        block = (
            f"{idx}. 工具/产物：{tool_name}\n"
            f"   摘要：{summary or content[:120]}\n"
            f"   参数：{args}\n"
            f"   内容：{content}"
        ).strip()
        if used + len(block) > max_total_chars:
            break
        lines.append(block)
        used += len(block)
    if len(lines) == 1:
        return ""
    if should_use_tool_context(query):
        lines.insert(1, "说明：用户当前请求疑似引用近期工具结果，以下内容优先作为跨轮依据。")
    return "\n".join(lines)
