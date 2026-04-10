import json
import re
from typing import Any, Optional


def try_json_load(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_json(text: str) -> dict:
    cleaned = sanitize_structured_output(text)
    parsed = try_json_load(cleaned)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"questions": parsed}

    # 尝试提取第一个 JSON 对象
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        candidate = match.group(0)
        parsed = try_json_load(candidate)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"questions": parsed}

    # 尝试提取第一个 JSON 数组
    match = re.search(r'\[[\s\S]*\]', cleaned)
    if match:
        candidate = match.group(0)
        parsed = try_json_load(candidate)
        if isinstance(parsed, list):
            return {"questions": parsed}
        if isinstance(parsed, dict):
            return parsed

    return {}


def extract_loose_json_fields_for_schema(text: Any, schema: type) -> Optional[dict]:
    raw = str(text or "")
    if not raw.strip():
        return None

    schema_name = getattr(schema, "__name__", "")
    out: dict = {}

    if schema_name in {"QuizGenerateResult", "ExamPaperText"}:
        field = "quiz" if schema_name == "QuizGenerateResult" else "exam_paper"
        m = re.search(rf'"{field}"\s*:\s*"([\s\S]*?)"\s*(?:,\s*"reasoning"|}})', raw)
        if m:
            out[field] = m.group(1)
        r = re.search(r'"reasoning"\s*:\s*(\{{[\s\S]*\}})', raw)
        if r:
            out["reasoning"] = extract_json(r.group(1))
        return out if out else None

    if schema_name == "CritiqueResult":
        score = re.search(r'"overall_score"\s*:\s*(\d+)', raw)
        approved = re.search(r'"approved"\s*:\s*(true|false)', raw, flags=re.IGNORECASE)
        critique = re.search(r'"critique"\s*:\s*"([\s\S]*?)"', raw)
        if score:
            out["overall_score"] = int(score.group(1))
        if approved:
            out["approved"] = approved.group(1).lower() == "true"
        if critique:
            out["critique"] = critique.group(1)
        return out if out else None

    if schema_name in {"ReviseResult", "ExamReviseResult"}:
        field = "revised_quiz" if schema_name == "ReviseResult" else "revised_exam"
        m = re.search(rf'"{field}"\s*:\s*"([\s\S]*?)"\s*(?:,\s*"revision_notes"|}})', raw)
        if m:
            out[field] = m.group(1)
        n = re.search(r'"revision_notes"\s*:\s*"([\s\S]*?)"\s*(?:,\s*"addressed_issues"|}})', raw)
        if n:
            out["revision_notes"] = n.group(1)
        return out if out else None

    if schema_name in {"StructuredQuizSetResult", "ExamPaper"}:
        parsed = extract_json(raw)
        return parsed if parsed else None

    return None


def sanitize_structured_output(text: str) -> str:
    t = str(text or "")
    t = re.sub(r"<think>[\s\S]*?</think>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"```(?:json)?", "", t, flags=re.IGNORECASE).strip()
    return t


def summarize_raw_output(text: Any) -> str:
    raw = str(text or "")
    raw = raw.replace("\n", " ").replace("\r", " ").strip()
    return raw[:260]


def strip_think_tags(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    return cleaned

