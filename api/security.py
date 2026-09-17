"""Authentication, object-level session authorization and request identity context."""

from __future__ import annotations

import contextvars
import hashlib
import hmac
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from utils.path_tool import get_abs_path


@dataclass(frozen=True)
class Principal:
    principal_id: str
    scopes: FrozenSet[str]


LOCAL_PRINCIPAL = Principal("local", frozenset({"read", "write", "admin"}))
current_principal: contextvars.ContextVar[Principal] = contextvars.ContextVar(
    "current_principal", default=LOCAL_PRINCIPAL
)


def _configured_keys() -> dict[str, tuple[str, FrozenSet[str]]]:
    """Read {principal: {key, scopes}} without ever logging the secret."""
    raw = os.getenv("DEEPREVISION_API_KEYS_JSON", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("DEEPREVISION_API_KEYS_JSON 不是合法 JSON") from exc
    result = {}
    for principal_id, value in parsed.items():
        if not isinstance(value, dict) or not value.get("key"):
            continue
        scopes = frozenset(str(s) for s in value.get("scopes", ["read", "write"]))
        digest = hashlib.sha256(str(value["key"]).encode()).hexdigest()
        result[str(principal_id)] = (digest, scopes)
    return result


def authenticate_header(authorization: str) -> Principal:
    required = os.getenv("DEEPREVISION_AUTH_REQUIRED", "0").lower() in {"1", "true", "yes"}
    if not authorization:
        if required:
            raise HTTPException(status_code=401, detail="缺少 Bearer 凭证")
        return LOCAL_PRINCIPAL
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Authorization 必须使用 Bearer")
    token_digest = hashlib.sha256(token.encode()).hexdigest()
    for principal_id, (expected, scopes) in _configured_keys().items():
        if hmac.compare_digest(token_digest, expected):
            return Principal(principal_id, scopes)
    raise HTTPException(status_code=401, detail="凭证无效")


class IdentityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in {"/", "/app", "/health", "/docs", "/openapi.json"}:
            return await call_next(request)
        try:
            principal = authenticate_header(request.headers.get("authorization", ""))
            needed = "read" if request.method in {"GET", "HEAD", "OPTIONS"} else "write"
            if needed not in principal.scopes and "admin" not in principal.scopes:
                raise HTTPException(status_code=403, detail=f"缺少 {needed} 权限")
        except HTTPException as exc:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        request.state.principal_id = principal.principal_id
        token = current_principal.set(principal)
        try:
            return await call_next(request)
        finally:
            current_principal.reset(token)


def _acl_db() -> str:
    configured = os.getenv("DEEPREVISION_SECURITY_DB", "data/security.sqlite3")
    path = Path(configured if os.path.isabs(configured) else get_abs_path(configured))
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def authorize_session(session_id: str, *, create: bool = False) -> str:
    """Enforce object ownership; local demo identity may lazily adopt legacy sessions."""
    principal = current_principal.get()
    with sqlite3.connect(_acl_db()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_acl (
                session_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
        """)
        row = conn.execute(
            "SELECT owner_id FROM session_acl WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row:
            if row[0] != principal.principal_id and "admin" not in principal.scopes:
                raise HTTPException(status_code=403, detail="无权访问该会话")
            return session_id
        if create or principal.principal_id == "local" or "admin" in principal.scopes:
            conn.execute(
                "INSERT INTO session_acl(session_id, owner_id, created_at) VALUES (?, ?, ?)",
                (session_id, principal.principal_id, int(time.time() * 1000)),
            )
            conn.commit()
            return session_id
    raise HTTPException(status_code=403, detail="会话尚未授权，请先由所有者创建")


def delete_session_acl(session_id: str) -> None:
    principal = current_principal.get()
    authorize_session(session_id)
    with sqlite3.connect(_acl_db()) as conn:
        conn.execute("DELETE FROM session_acl WHERE session_id = ?", (session_id,))
        conn.commit()


def allowed_session_ids() -> set[str] | None:
    """Return owned IDs, or None for administrators/local single-user mode."""
    principal = current_principal.get()
    if "admin" in principal.scopes:
        return None
    with sqlite3.connect(_acl_db()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_acl (
                session_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, created_at INTEGER NOT NULL
            )
        """)
        return {
            str(row[0]) for row in conn.execute(
                "SELECT session_id FROM session_acl WHERE owner_id = ?", (principal.principal_id,)
            )
        }


def begin_idempotent(operation: str, key: str, request_hash: str) -> dict | None:
    """Claim an operation, replay a completed response, or reject key misuse/concurrency."""
    if not key or len(key) > 128:
        raise HTTPException(status_code=400, detail="写操作需要1-128字符的 Idempotency-Key")
    principal = current_principal.get()
    now = int(time.time() * 1000)
    with sqlite3.connect(_acl_db(), timeout=10) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS idempotency_records (
                principal_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                response_json TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (principal_id, operation, idempotency_key)
            )
        """)
        row = conn.execute(
            """SELECT request_hash, status, response_json FROM idempotency_records
               WHERE principal_id=? AND operation=? AND idempotency_key=?""",
            (principal.principal_id, operation, key),
        ).fetchone()
        if row:
            if not hmac.compare_digest(str(row[0]), request_hash):
                raise HTTPException(status_code=409, detail="该 Idempotency-Key 已用于不同请求")
            if row[1] == "completed" and row[2]:
                return json.loads(row[2])
            raise HTTPException(status_code=409, detail="相同幂等请求正在执行或上次未完成")
        conn.execute(
            """INSERT INTO idempotency_records VALUES (?, ?, ?, ?, 'processing', NULL, ?, ?)""",
            (principal.principal_id, operation, key, request_hash, now, now),
        )
        conn.commit()
    return None


def complete_idempotent(operation: str, key: str, response: dict) -> None:
    principal = current_principal.get()
    with sqlite3.connect(_acl_db()) as conn:
        conn.execute(
            """UPDATE idempotency_records SET status='completed', response_json=?, updated_at=?
               WHERE principal_id=? AND operation=? AND idempotency_key=?""",
            (json.dumps(response, ensure_ascii=False), int(time.time() * 1000),
             principal.principal_id, operation, key),
        )
        conn.commit()


def fail_idempotent(operation: str, key: str) -> None:
    """Release a failed claim so a client can safely retry with the same key."""
    principal = current_principal.get()
    with sqlite3.connect(_acl_db()) as conn:
        conn.execute(
            """DELETE FROM idempotency_records
               WHERE principal_id=? AND operation=? AND idempotency_key=? AND status='processing'""",
            (principal.principal_id, operation, key),
        )
        conn.commit()
