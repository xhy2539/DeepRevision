import re
from typing import Any, List, Optional


def normalize_text_field(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        return text if text else default
    return str(value).strip() or default


def sanitize_question_text(text: str) -> str:
    normalized = str(text or "")
    normalized = re.sub(r'^\s*\d+\s*[.、]\s*', '', normalized)
    normalized = re.sub(r'^题目\d+[:：]\s*', '', normalized)
    normalized = re.sub(r'^第\s*\d+\s*题[:：]?\s*', '', normalized)
    normalized = re.sub(r'^[-*•]+\s*', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # 移除“中文术语（English explanation）”样式，但保留纯缩写
    normalized = re.sub(r'([\u4e00-\u9fff]{2,})\s*[（(]\s*[A-Za-z][A-Za-z0-9_\- ]{2,}\s*[)）]', r'\1', normalized)
    return normalized


def sanitize_user_visible_text(text: str) -> str:
    cleaned = sanitize_question_text(text)
    cleaned = re.sub(r'根据(课件|资料|参考资料)[，,:：]?', '', cleaned)
    cleaned = re.sub(r'(围绕|依据)(课程)?核心知识点', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def contains_term_style_violation(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]{2,}\s*[（(]\s*[A-Za-z][A-Za-z0-9_\- ]{2,}\s*[)）]', text))


def normalize_int_field(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        if isinstance(value, str):
            value = value.replace("分", "").strip()
        return int(float(value))
    except Exception:
        return default


def normalize_dict_field(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def normalize_list_of_dicts(value: Any) -> List[dict]:
    if value is None:
        return []
    if isinstance(value, list):
        output: List[dict] = []
        for item in value:
            if isinstance(item, dict):
                output.append(item)
            elif item is not None:
                output.append({"text": str(item)})
        return output
    if isinstance(value, dict):
        return [value]
    return [{"text": str(value)}]


def normalize_reasoning_payload(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, list):
        return {"items": value}
    text = str(value).strip()
    return {"summary": text} if text else {}


def format_score(score: Any) -> Optional[str]:
    if score in (None, "", 0, "0"):
        return None
    score_str = str(score).strip()
    return score_str if score_str.endswith("分") else f"{score_str}分"


def normalize_option_text(option: str, index: int) -> str:
    option_text = sanitize_user_visible_text(str(option or "").strip())
    if not option_text:
        return f"{chr(65 + index)}. （无内容）"
    m = re.match(r'^([A-D])[.、．:：)\s]*(.*)$', option_text, flags=re.IGNORECASE)
    if m:
        content = (m.group(2) or "").strip()
        if content:
            return f"{m.group(1).upper()}. {content}"
    return f"{chr(65 + index)}. {option_text}"


def normalize_options(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    options: List[str] = []
    if isinstance(value, dict):
        for index, key in enumerate(["A", "B", "C", "D"]):
            if key in value:
                normalized = normalize_option_text(value.get(key), index)
                if normalized:
                    options.append(normalized)
    elif isinstance(value, list):
        for index, option in enumerate(value):
            normalized = normalize_option_text(option, index)
            if normalized:
                options.append(normalized)
    else:
        text = str(value).strip()
        if text:
            matches = re.findall(r'([A-D])[.、．:：)\s]+([^A-D]+?)(?=(?:\s+[A-D][.、．:：)\s])|$)', text)
            if matches:
                for index, (letter, content) in enumerate(matches):
                    options.append(normalize_option_text(f"{letter}. {content}", index))

    deduped: List[str] = []
    seen = set()
    for opt in options:
        # 先完整 normalize，再取其 content 作为 dedup key（避免 "A. A. xxx" 类 malformed 选项逃过去重）
        normalized_opt = normalize_option_text(opt, 0) if isinstance(opt, str) else str(opt)
        content = re.sub(r'^[A-D][.、．:：)\s]*', '', normalized_opt).strip()
        key = re.sub(r'\s+', ' ', content)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(f"{chr(65 + len(deduped))}. {content}")
        if len(deduped) >= 4:
            break

    return deduped or None


def normalize_question_item(item: Any, fallback_id: int) -> dict:
    if not isinstance(item, dict):
        item = {"question": str(item).strip()}
    return {
        "id": normalize_int_field(item.get("id"), fallback_id),
        "type": normalize_text_field(item.get("type"), "选择题"),
        "question": sanitize_user_visible_text(
            sanitize_question_text(normalize_text_field(item.get("question") or item.get("content"), ""))
        ),
        "options": normalize_options(item.get("options")),
        "answer": sanitize_user_visible_text(normalize_text_field(item.get("answer"), "")),
        "explanation": sanitize_user_visible_text(normalize_text_field(item.get("explanation") or item.get("analysis"), "")),
        "score": format_score(item.get("score")),
        "difficulty": normalize_text_field(item.get("difficulty"), "中等"),
    }


def normalize_questions_payload(value: Any) -> List[dict]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [normalize_question_item(item, index) for index, item in enumerate(items, start=1)]


def sanitize_quiz_questions_for_delivery(questions: Any) -> List[dict]:
    normalized = normalize_questions_payload(questions)
    out: List[dict] = []
    for idx, q in enumerate(normalized, start=1):
        item = dict(q)
        item["id"] = int(item.get("id") or idx)
        item["question"] = sanitize_user_visible_text(normalize_text_field(item.get("question"), ""))
        item["answer"] = sanitize_user_visible_text(normalize_text_field(item.get("answer"), ""))
        item["explanation"] = sanitize_user_visible_text(normalize_text_field(item.get("explanation"), ""))
        qtype = str(item.get("type") or "").strip() or "选择题"
        item["type"] = qtype
        if qtype == "选择题":
            opts = normalize_options(item.get("options")) or []
            item["options"] = [normalize_option_text(opt, i) for i, opt in enumerate(opts[:4])]
        else:
            item["options"] = None
        out.append(item)
    return out


def normalize_exam_question_item(item: Any, fallback_id: int) -> dict:
    if not isinstance(item, dict):
        item = {"content": str(item).strip()}
    return {
        "id": normalize_int_field(item.get("id") or item.get("number"), fallback_id),
        "type": normalize_text_field(item.get("type"), "选择题"),
        "content": sanitize_question_text(normalize_text_field(item.get("content") or item.get("question"), "")),
        "options": normalize_options(item.get("options")),
        "answer": normalize_text_field(item.get("answer"), ""),
        "analysis": sanitize_user_visible_text(normalize_text_field(item.get("analysis") or item.get("explanation"), "")),
        "score": normalize_int_field(item.get("score"), 2),
        "difficulty": normalize_text_field(item.get("difficulty"), "中等"),
    }


def normalize_exam_questions_payload(value: Any) -> List[dict]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [normalize_exam_question_item(item, index) for index, item in enumerate(items, start=1)]


def limit_context_size(text: str, max_chars: int = 12000) -> str:
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars]


def normalize_addressed_issues(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    items = value if isinstance(value, list) else [value]
    normalized: List[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            if text:
                normalized.append(text)
            continue
        if isinstance(item, dict):
            issue_id = str(item.get("issue_id", "")).strip()
            fix = str(item.get("fix", "")).strip()
            issue = str(item.get("issue", "")).strip()
            action = str(item.get("action", "")).strip()
            parts = [part for part in [issue_id or issue, fix or action] if part]
            if parts:
                normalized.append("：".join(parts))
                continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized

