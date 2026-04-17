"""用户级人格配置服务：读写用户偏好并构建可注入提示词。"""
import json
import hashlib
import os
from typing import Any, Dict

from utils.path_tool import get_abs_path


PROFILE_DIR = get_abs_path("data/user_profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)

# 默认用户标识（未登录/未传用户名时兜底）
DEFAULT_USERNAME = "default_user"

# 默认人格配置（用户级，不随会话切换）
DEFAULT_USER_PROFILE: Dict[str, str] = {
    "assistant_name": "DeepRevision",
    "tone": "warm",
    "verbosity": "normal",
    "teaching_style": "coach",
    "emoji_level": "light",
    "call_user_name": "",
}

_ENUMS: Dict[str, set[str]] = {
    "tone": {"warm", "professional", "strict", "encouraging"},
    "verbosity": {"concise", "normal", "detailed"},
    "teaching_style": {"coach", "socratic", "step_by_step", "exam_oriented"},
    "emoji_level": {"none", "light", "normal"},
}


def normalize_username(username: Any) -> str:
    """归一化用户名，避免空值导致配置无法定位。"""
    raw = str(username or "").strip()
    return raw[:80] if raw else DEFAULT_USERNAME


def _profile_path(username: str) -> str:
    """基于用户名哈希映射到配置文件路径，避免非法文件名。"""
    sid = normalize_username(username)
    key = hashlib.md5(sid.encode("utf-8")).hexdigest()[:20]
    return os.path.join(PROFILE_DIR, f"{key}.json")


def _sanitize_profile(value: Dict[str, Any]) -> Dict[str, str]:
    """白名单清洗人格配置，只保留允许字段与枚举。"""
    current: Dict[str, str] = dict(DEFAULT_USER_PROFILE)
    if not isinstance(value, dict):
        return current

    # 普通文本字段
    for key, max_len in (("assistant_name", 20), ("call_user_name", 20)):
        if key not in value:
            continue
        text = str(value.get(key) or "").strip()
        current[key] = text[:max_len]

    # 枚举字段
    for key, allowed in _ENUMS.items():
        if key not in value:
            continue
        item = str(value.get(key) or "").strip().lower()
        if item in allowed:
            current[key] = item

    return current


def get_user_profile(username: Any) -> Dict[str, Any]:
    """获取用户级人格配置（不存在则返回默认值）。"""
    user = normalize_username(username)
    path = _profile_path(user)
    if not os.path.exists(path):
        return {"username": user, "profile": dict(DEFAULT_USER_PROFILE)}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        profile = _sanitize_profile(payload.get("profile", {}))
        return {"username": user, "profile": profile}
    except Exception:
        return {"username": user, "profile": dict(DEFAULT_USER_PROFILE)}


def update_user_profile(username: Any, patch: Dict[str, Any]) -> Dict[str, Any]:
    """更新并持久化用户级人格配置，返回最新完整配置。"""
    user = normalize_username(username)
    path = _profile_path(user)
    current = get_user_profile(user).get("profile", dict(DEFAULT_USER_PROFILE))
    merged = dict(current)
    if isinstance(patch, dict):
        merged.update(patch)
    profile = _sanitize_profile(merged)
    payload = {"username": user, "profile": profile}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def build_persona_instructions(profile: Dict[str, Any]) -> str:
    """把用户级人格配置转成简短指令文本，供提示词注入。"""
    p = _sanitize_profile(profile if isinstance(profile, dict) else {})
    assistant_name = str(p.get("assistant_name") or "DeepRevision")
    call_user_name = str(p.get("call_user_name") or "").strip()
    tone = str(p.get("tone") or "warm")
    verbosity = str(p.get("verbosity") or "normal")
    style = str(p.get("teaching_style") or "coach")
    emoji_level = str(p.get("emoji_level") or "light")

    lines = [
        f"- 你的称呼：{assistant_name}",
        f"- 语气：{tone}",
        f"- 回答详略：{verbosity}",
        f"- 讲解风格：{style}",
        f"- emoji 使用强度：{emoji_level}",
    ]
    if call_user_name:
        lines.append(f"- 对用户称呼：{call_user_name}")
    return "\n".join(lines)

