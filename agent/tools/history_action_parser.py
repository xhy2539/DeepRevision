import re
from typing import Any, Dict


def _to_int(value: str) -> int | None:
    """把提取到的数字字符串转成 int。"""
    try:
        return int(value)
    except Exception:
        return None


def _extract_first(patterns: list[str], text: str) -> int | None:
    """按顺序尝试多个正则，返回第一个匹配到的整数。"""
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = _to_int(match.group(1))
            if value is not None:
                return value
    return None


def parse_history_action(user_input: str) -> Dict[str, Any]:
    """解析历史管理意图，返回标准动作字典。"""
    text = str(user_input or "").strip()
    if not text:
        return {"action": "none"}

    # 先判定“清空类”动作，避免被“删除”关键词误命中。
    if any(k in text for k in ["清空错题", "清除错题", "清空练习记录", "清除练习记录", "删除全部错题"]):
        return {"action": "clear_practice"}
    if any(k in text for k in ["清空对话", "清除对话", "清空聊天记录", "清除聊天记录", "删除全部消息"]):
        return {"action": "clear_messages"}

    # 删除错题：支持按 ID 或“第N条”。
    if any(k in text for k in ["删除错题", "移除错题", "删除练习记录"]):
        record_id = _extract_first([r"(?:错题|记录)?\s*(?:id|ID)\s*(\d+)", r"记录\s*(\d+)"], text)
        record_index = _extract_first([r"第\s*(\d+)\s*条"], text)
        payload: Dict[str, Any] = {"action": "delete_practice"}
        if record_id is not None:
            payload["record_id"] = record_id
        if record_index is not None:
            payload["record_index"] = record_index
        return payload

    # 删除消息：优先提取长数字作为 timestamp。
    if any(k in text for k in ["删除消息", "删除对话", "移除消息", "删除聊天记录"]):
        timestamp = _extract_first([r"(\d{10,17})"], text)
        message_index = _extract_first([r"第\s*(\d+)\s*条"], text)
        payload = {"action": "delete_message"}
        if timestamp is not None:
            payload["timestamp"] = timestamp
        if message_index is not None:
            payload["message_index"] = message_index
        return payload

    # 查询类动作。
    if any(k in text for k in ["历史对话", "对话记录", "聊天记录", "消息历史"]):
        return {"action": "list_messages"}
    practice_keywords = ["错题", "练习记录", "答题记录", "错题本"]
    list_intent_keywords = ["查看", "列出", "展示", "显示", "查询", "看看", "给我看", "帮我看", "最近", "有哪些"]
    analysis_intent_keywords = ["分析", "总结", "归纳", "薄弱", "弱项", "错因", "原因", "复盘", "建议", "提升"]
    if any(k in text for k in practice_keywords):
        # 含“分析/总结”等意图时交回上游 LLM，不要被列表规则短路。
        if any(k in text for k in analysis_intent_keywords):
            return {"action": "none"}
        if any(k in text for k in list_intent_keywords) or text in {"错题", "错题本", "我的错题", "练习记录", "答题记录"}:
            return {"action": "list_practice"}

    return {"action": "none"}
