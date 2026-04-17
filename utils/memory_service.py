"""
双轨记忆服务
- 短期：滑动窗口（最近 N 轮对话）
- 长期：异步 LLM 图谱提纯 → 结构化知识节点
持久化：SQLite（stdlib sqlite3 + run_in_executor，无额外依赖）
启动时自动从旧 sessions.json 迁移数据
"""
import json
import os
import sqlite3
import asyncio
import time
import hashlib
import re
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from functools import partial

from model.factory import chat_model, embed_model
from utils.logger_handler import logger
from utils.config_handler import chroma_conf
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_chroma import Chroma
from langchain_core.documents import Document


class KnowledgeGraphNode(BaseModel):
    subject: str = Field(description="核心知识点或概念名称")
    relation: str = Field(description="关系或意图，如'定义为'、'需要复习'、'理解错误于'")
    object: str = Field(description="具体的详解或结论")


# ==================== SQLite 辅助（同步，在线程池中运行）====================

def _db_path() -> str:
    path = os.path.join(os.getcwd(), "data", "sessions.db")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _create_tables(conn: sqlite3.Connection):
    # 开启 WAL 模式，提升并发读写性能
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            parent_id  TEXT REFERENCES sessions(session_id)
        );
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT    NOT NULL,
            role       TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            bucket     TEXT    NOT NULL,  -- 'recent' | 'trash'
            timestamp  INTEGER,
            kind       TEXT,
            render_mode TEXT,
            payload    TEXT,
            meta       TEXT
        );
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT    NOT NULL,
            subject    TEXT    NOT NULL,
            relation   TEXT    NOT NULL,
            object     TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS practice_records (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT    NOT NULL,
            question_id     TEXT,
            question_content TEXT    NOT NULL,
            knowledge_point TEXT,
            user_answer     TEXT,
            correct_answer  TEXT,
            is_correct      INTEGER NOT NULL,
            wrong_reason    TEXT,
            created_at      INTEGER NOT NULL
        );
    """)
    # 创建索引，加速按 session_id 查询
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_sid ON messages(session_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_sid ON graph_nodes(session_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_practice_sid ON practice_records(session_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_practice_kp ON practice_records(knowledge_point);")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS question_bank (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id     TEXT    UNIQUE,
            session_id      TEXT    NOT NULL,
            knowledge_point TEXT,
            question_content TEXT   NOT NULL,
            answer          TEXT,
            chroma_id       TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_qbank_sid ON question_bank(session_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_qbank_kp ON question_bank(knowledge_point);")
    conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str):
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column in cols:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    conn.commit()
    logger.info(f"[记忆迁移] 已为 {table} 表添加 {column} 列")


def _migrate_add_parent_id(conn: sqlite3.Connection):
    """迁移：为 sessions 表添加 parent_id 列（如果不存在）"""
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN parent_id TEXT")
        conn.commit()
        logger.info("[记忆迁移] 已为 sessions 表添加 parent_id 列")
    except Exception as e:
        # 列可能已存在
        if "duplicate column name" in str(e).lower():
            pass
        else:
            logger.warning(f"[记忆迁移] parent_id 列添加失败（可能已存在）: {e}")


def _migrate_messages_schema(conn: sqlite3.Connection):
    """迁移：为 messages 表添加结构化协议字段"""
    try:
        _ensure_column(conn, "messages", "timestamp", "INTEGER")
        _ensure_column(conn, "messages", "kind", "TEXT")
        _ensure_column(conn, "messages", "render_mode", "TEXT")
        _ensure_column(conn, "messages", "payload", "TEXT")
        _ensure_column(conn, "messages", "meta", "TEXT")
    except Exception as e:
        logger.warning(f"[记忆迁移] messages 表结构升级失败: {e}")


def _loads_json_or_default(text: Optional[str], default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _clean_knowledge_point_phrase(text: str) -> str:
    s = str(text or "")
    # 去 markdown / emoji-like 噪音
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"[*_#>\[\]\(\)✅❌⚠️•·]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""

    # 常见题干模板抽取（偏名词短语）
    patterns = [
        r"下列关于(.{2,24}?)的",
        r"关于(.{2,24}?)的",
        r"在(.{2,24}?)中",
        r"(.{2,24}?)的核心作用",
        r"(.{2,24}?)机制",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            cand = str(m.group(1) or "").strip("：:，,。.;；!?！？ ")
            if 2 <= len(cand) <= 24:
                s = cand
                break

    # 分隔截断（保留前半句名词性内容）
    for sep in ["；", ";", "。", "?", "？", "!", "！", " - ", "——", "：", ":"]:
        if sep in s:
            s = s.split(sep, 1)[0].strip()
            break

    # 去疑问句前缀
    s = re.sub(r"^(下列|以下|请|试|简述|说明|判断|选择|哪个|哪一项|当|在)\s*", "", s).strip()
    s = s.strip("：:，,。.;；!?！？ ")
    s = re.sub(r"\s+", " ", s).strip()

    # 超长短语优先抽取核心专业词，避免“半截考点”。
    if len(s) > 28:
        keyword_patterns = [
            r"(进程调度(?:三层架构|层次关系|策略|模型|器)?)",
            r"(中期调度(?:器)?)",
            r"(进程状态(?:模型|迁移)?)",
            r"(阻塞(?:状态|队列)?)",
            r"(资源分配图)",
            r"(循环等待)",
            r"(死锁(?:判定|条件)?)",
            r"(分时系统)",
            r"(虚拟内存(?:系统|机制)?)",
            r"(页面置换)",
            r"(文件系统分层结构)",
            r"(逻辑文件系统)",
            r"(系统调用接口)",
            r"(内核(?:硬件)?接口)",
        ]
        for pattern in keyword_patterns:
            match = re.search(pattern, s)
            if match:
                core = str(match.group(1) or "").strip("：:，,。.;；!?！？ ")
                if core:
                    return core
    return s


def _load_session_sync(db: str, session_id: str) -> Dict:
    """按需加载单个 session 的完整数据"""
    conn = sqlite3.connect(db)
    try:
        session = {"name": session_id, "parent_id": None, "recent": [], "trash": [], "graph": []}

        row = conn.execute(
            "SELECT name, parent_id FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row:
            session["name"] = row[0]
            session["parent_id"] = row[1]

        for (role, content, bucket, timestamp, kind, render_mode, payload, meta) in conn.execute(
            "SELECT role, content, bucket, timestamp, kind, render_mode, payload, meta FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,)
        ):
            session[bucket].append({
                "role": role,
                "content": content,
                "timestamp": timestamp or int(time.time() * 1000),
                "kind": kind or ("chat" if role == "ai" else None),
                "render_mode": render_mode or ("markdown" if role == "ai" else None),
                "payload": _loads_json_or_default(payload, None),
                "meta": _loads_json_or_default(meta, None),
            })

        for (subject, relation, obj) in conn.execute(
            "SELECT subject, relation, object FROM graph_nodes WHERE session_id = ? ORDER BY id",
            (session_id,)
        ):
            session["graph"].append({"subject": subject, "relation": relation, "object": obj})

        return session
    finally:
        conn.close()


def _load_all_sync(db: str) -> Dict[str, Dict]:
    """启动时只加载 session 索引（懒加载实际数据）"""
    conn = sqlite3.connect(db)
    _create_tables(conn)
    _migrate_add_parent_id(conn)
    _migrate_messages_schema(conn)
    store: Dict[str, Dict] = {}

    for row in conn.execute("SELECT session_id, name, parent_id FROM sessions"):
        sid, name, parent_id = row
        store[sid] = {"name": name, "parent_id": parent_id, "recent": None, "trash": None, "graph": None}

    conn.close()
    return store


def _ensure_session_loaded(store: Dict[str, Dict], db: str, session_id: str):
    if session_id not in store:
        store[session_id] = {"name": session_id, "parent_id": None, "recent": None, "trash": None, "graph": None}

    if store[session_id].get("recent") is None:
        loaded = _load_session_sync(db, session_id)
        store[session_id] = loaded


def _append_message_sync(
    db: str,
    session_id: str,
    role: str,
    content: str,
    bucket: str,
    timestamp: int,
    kind: Optional[str],
    render_mode: Optional[str],
    payload: Optional[Dict[str, Any]],
    meta: Optional[Dict[str, Any]],
):
    """只追加一条消息，不重写整个 session"""
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, name) VALUES (?, ?)",
            (session_id, session_id)
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, bucket, timestamp, kind, render_mode, payload, meta) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                role,
                content,
                bucket,
                timestamp,
                kind,
                render_mode,
                json.dumps(payload, ensure_ascii=False) if payload is not None else None,
                json.dumps(meta, ensure_ascii=False) if meta is not None else None,
            )
        )
        conn.commit()
    finally:
        conn.close()


def _save_session_sync(db: str, session_id: str, session: Dict):
    """将单个 session 的最新状态写入 SQLite（覆盖旧数据）"""
    conn = sqlite3.connect(db)
    try:
        # session 基本信息
        conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, name, parent_id) VALUES (?, ?, ?)",
            (session_id, session["name"], session.get("parent_id"))
        )
        # 清除旧消息、重写（消息量小，直接覆盖简单可靠）
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        for msg in session["recent"]:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, bucket, timestamp, kind, render_mode, payload, meta) VALUES (?, ?, ?, 'recent', ?, ?, ?, ?, ?)",
                (
                    session_id,
                    msg["role"],
                    msg["content"],
                    msg.get("timestamp"),
                    msg.get("kind"),
                    msg.get("render_mode"),
                    json.dumps(msg.get("payload"), ensure_ascii=False) if msg.get("payload") is not None else None,
                    json.dumps(msg.get("meta"), ensure_ascii=False) if msg.get("meta") is not None else None,
                )
            )
        for msg in session["trash"]:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, bucket, timestamp, kind, render_mode, payload, meta) VALUES (?, ?, ?, 'trash', ?, ?, ?, ?, ?)",
                (
                    session_id,
                    msg["role"],
                    msg["content"],
                    msg.get("timestamp"),
                    msg.get("kind"),
                    msg.get("render_mode"),
                    json.dumps(msg.get("payload"), ensure_ascii=False) if msg.get("payload") is not None else None,
                    json.dumps(msg.get("meta"), ensure_ascii=False) if msg.get("meta") is not None else None,
                )
            )
        # graph 节点：只追加（提纯后 extend，不会重复写旧节点）
        # 策略：全量覆盖保持简单
        conn.execute("DELETE FROM graph_nodes WHERE session_id = ?", (session_id,))
        for node in session["graph"]:
            conn.execute(
                "INSERT INTO graph_nodes (session_id, subject, relation, object) VALUES (?, ?, ?, ?)",
                (session_id, node.get("subject", ""), node.get("relation", ""), node.get("object", ""))
            )
        conn.commit()
    finally:
        conn.close()


def _delete_session_sync(db: str, session_id: str):
    conn = sqlite3.connect(db)
    try:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM graph_nodes WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()


def _migrate_from_json(db: str, json_path: str):
    """将旧 sessions.json 数据迁移到 SQLite（只在 SQLite 为空时执行）"""
    if not os.path.exists(json_path):
        return
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            old_store = json.load(f)
        if not old_store:
            return

        conn = sqlite3.connect(db)
        _create_tables(conn)
        _migrate_add_parent_id(conn)
        _migrate_messages_schema(conn)
        existing = {row[0] for row in conn.execute("SELECT session_id FROM sessions")}

        for sid, session in old_store.items():
            if sid in existing:
                continue  # 已迁移，跳过
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, name, parent_id) VALUES (?, ?, ?)",
                (sid, session.get("name", sid), session.get("parent_id"))
            )
            for bucket in ("recent", "trash"):
                for msg in session.get(bucket, []):
                    conn.execute(
                        "INSERT INTO messages (session_id, role, content, bucket, timestamp, kind, render_mode, payload, meta) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            sid,
                            msg["role"],
                            msg["content"],
                            bucket,
                            msg.get("timestamp", int(time.time() * 1000)),
                            msg.get("kind"),
                            msg.get("render_mode"),
                            json.dumps(msg.get("payload"), ensure_ascii=False) if msg.get("payload") is not None else None,
                            json.dumps(msg.get("meta"), ensure_ascii=False) if msg.get("meta") is not None else None,
                        )
                    )
            for node in session.get("graph", []):
                conn.execute(
                    "INSERT INTO graph_nodes (session_id, subject, relation, object) VALUES (?, ?, ?, ?)",
                    (sid, node.get("subject", ""), node.get("relation", ""), node.get("object", ""))
                )
        conn.commit()
        conn.close()
        logger.info(f"[记忆迁移] 已将 {len(old_store)} 个 session 从 sessions.json 迁移到 SQLite")
    except Exception as e:
        logger.error(f"[记忆迁移] 失败: {e}")


# ==================== SessionMemoryManager ====================

class SessionMemoryManager:
    def __init__(self, max_recent_turns: int = 5, trash_threshold: int = 4, max_graph_nodes: int = 100, max_trash_size: int = 50):
        self.max_recent_turns = max_recent_turns
        self.trash_threshold = trash_threshold
        self.max_graph_nodes = max_graph_nodes
        self.max_trash_size = max_trash_size
        self.db = _db_path()
        self._distilling: set = set()
        self._distilling_lock = asyncio.Lock()

        # 迁移旧 JSON 数据（仅首次）
        json_path = os.path.join(os.getcwd(), "data", "sessions.json")
        _migrate_from_json(self.db, json_path)

        # 同步初始化加载（__init__ 不能是 async）
        self.store: Dict[str, Dict] = _load_all_sync(self.db)
        logger.info(f"[记忆服务] 从 SQLite 加载 {len(self.store)} 个 session")

    # ── 内部工具 ──────────────────────────────────────────────────────────

    async def _persist(self, session_id: str):
        """异步将 session 写入 SQLite（线程池，不阻塞事件循环）"""
        loop = asyncio.get_event_loop()
        session = self.store[session_id]
        await loop.run_in_executor(
            None,
            partial(_save_session_sync, self.db, session_id, session)
        )

    def _init_session(self, session_id: str, name: str = None, parent_id: str = None):
        if session_id not in self.store:
            self.store[session_id] = {"name": name or session_id, "parent_id": parent_id, "recent": None, "trash": None, "graph": None}
        _ensure_session_loaded(self.store, self.db, session_id)
        if name:
            self.store[session_id]["name"] = name
        if parent_id is not None:
            self.store[session_id]["parent_id"] = parent_id

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def register_session(self, session_id: str, name: str, parent_id: str = None):
        """显式注册/创建新会话，可指定父会话 ID（用于子会话/分支）"""
        self._init_session(session_id, name, parent_id)
        self.store[session_id]["name"] = name
        self.store[session_id]["parent_id"] = parent_id
        asyncio.create_task(self._persist(session_id))

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        kind: Optional[str] = None,
        render_mode: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        timestamp: Optional[int] = None,
    ):
        """添加消息；超过阈值时 fire-and-forget 触发图谱提纯"""
        self._init_session(session_id)
        _ensure_session_loaded(self.store, self.db, session_id)
        session = self.store[session_id]

        message_timestamp = timestamp or int(time.time() * 1000)
        session["recent"].append({
            "role": role,
            "content": content,
            "timestamp": message_timestamp,
            "kind": kind,
            "render_mode": render_mode,
            "payload": payload,
            "meta": meta,
        })

        max_messages = self.max_recent_turns * 2
        if len(session["recent"]) > max_messages:
            kicked = session["recent"][:2]
            session["recent"] = session["recent"][2:]
            session["trash"].extend(kicked)

            if len(session["trash"]) > self.max_trash_size:
                session["trash"] = session["trash"][-self.max_trash_size:]

            if len(session["trash"]) >= self.trash_threshold * 2:
                asyncio.create_task(self._trigger_background_distillation(session_id))

            await self._persist(session_id)
        else:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                partial(
                    _append_message_sync,
                    self.db,
                    session_id,
                    role,
                    content,
                    "recent",
                    message_timestamp,
                    kind,
                    render_mode,
                    payload,
                    meta,
                )
            )

    async def _trigger_background_distillation(self, session_id: str):
        async with self._distilling_lock:
            if session_id in self._distilling:
                return
            self._distilling.add(session_id)
        trash_data = list(self.store[session_id]["trash"])
        self.store[session_id]["trash"].clear()
        try:
            await self._distill_to_graph(session_id, trash_data)
        except Exception as e:
            logger.error(f"[记忆提纯失败] {e}")
            # 失败回滚：避免提纯失败导致 trash 记录丢失
            session = self.store.get(session_id)
            if session is not None:
                current_trash = session.get("trash") or []
                session["trash"] = (trash_data + current_trash)[-self.max_trash_size:]
                try:
                    await self._persist(session_id)
                except Exception as persist_err:
                    logger.error(f"[记忆提纯失败] 回滚持久化失败: {persist_err}")
        finally:
            self._distilling.discard(session_id)

    async def _distill_to_graph(self, session_id: str, trash_data: List[Dict]):
        # 使用通用 JSON 解析器，兼容返回数组；后续再逐条校验节点结构
        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template=(
                "你是知识提炼师。请从以下零散的对话历史中，提炼出学生掌握不牢固的知识点或是复习焦点。"
                "如果没有明确有效的知识点，则返回空列表。\n"
                "对话记录：{history}\n"
                "请使用如下JSON格式输出，不要包含Markdown标记：\n{format_instructions}\n"
                "返回类型必须是一个 JSON Array 数组。"
            ),
            input_variables=["history"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        history_text = "\n".join([
            f"{m.get('role', 'unknown')}: {str(m.get('content', ''))}"
            for m in trash_data
        ])

        def strip_think_tags(text):
            """去除 <think> 标签内容，避免 JSON 解析失败"""
            import re
            # 兼容 AIMessage / BaseMessage 对象
            if hasattr(text, "content"):
                text = text.content
            text = str(text or "")
            text = re.sub(r'<think>[\s\S]*?<\/think>', '', text).strip()
            return text

        strip_chain = RunnableLambda(strip_think_tags)
        # 先转为纯文本再做清洗，避免把 AIMessage 直接传给正则
        chain = prompt | chat_model | StrOutputParser() | strip_chain | parser

        logger.info(f"[记忆提纯] 正在提纯 {session_id} 的 {len(trash_data)} 条记录...")
        result = await chain.ainvoke({"history": history_text})

        if result and isinstance(result, dict):
            result = [result]

        if result and isinstance(result, list):
            # 只保留结构完整的节点，避免脏数据污染图谱
            valid_nodes = []
            for item in result:
                if not isinstance(item, dict):
                    continue
                subject = str(item.get("subject", "")).strip()
                relation = str(item.get("relation", "")).strip()
                obj = str(item.get("object", "")).strip()
                if subject and relation and obj:
                    valid_nodes.append({"subject": subject, "relation": relation, "object": obj})

            if not valid_nodes:
                return

            session = self.store[session_id]
            current_size = len(session["graph"])
            result_size = len(valid_nodes)
            if current_size + result_size > self.max_graph_nodes:
                available = self.max_graph_nodes - current_size
                if available > 0:
                    session["graph"].extend(valid_nodes[:available])
                else:
                    session["graph"] = valid_nodes[:self.max_graph_nodes] if valid_nodes else []
                logger.info(f"[记忆提纯] {session_id} 图谱超限，保留最近 {self.max_graph_nodes} 条知识节点")
            else:
                session["graph"].extend(valid_nodes)
                logger.info(f"[记忆提纯] {session_id} 抽取 {len(valid_nodes)} 条知识节点")
            await self._persist(session_id)

    def get_memory_context(self, session_id: str) -> str:
        self._init_session(session_id)
        session = self.store[session_id]

        # 获取自己的图谱
        own_graph = session.get("graph") or []

        # 获取父会话的图谱（递归向上查找）
        parent_graph = []
        if session.get("parent_id"):
            parent_graph = self._get_parent_graph(session["parent_id"])

        # 合并去重（基于 subject + relation 作为 key）
        all_nodes = self._merge_graph_nodes(own_graph, parent_graph)

        if not all_nodes:
            return ""
        context = "【以往复习知识点图谱备忘】\n"
        for node in all_nodes:
            context += (
                f"- 概念[{node.get('subject', '')}] : "
                f"{node.get('relation', '')} -> {node.get('object', '')}\n"
            )
        return context

    def _get_parent_graph(self, parent_id: str, _visited: set = None) -> List[Dict]:
        """递归获取父会话及其祖先的图谱节点（带环检测）"""
        if _visited is None:
            _visited = set()
        if not parent_id or parent_id in _visited:
            return []
        _visited.add(parent_id)
        self._init_session(parent_id)
        parent_session = self.store.get(parent_id, {})
        parent_graph = parent_session.get("graph") or []
        # 递归向上获取祖先图谱
        grandparent_id = parent_session.get("parent_id")
        if grandparent_id:
            parent_graph = parent_graph + self._get_parent_graph(grandparent_id, _visited)
        return parent_graph

    def _merge_graph_nodes(self, own_graph: List[Dict], parent_graph: List[Dict]) -> List[Dict]:
        """合并去重图谱节点，基于 subject+relation 作为唯一键"""
        seen = set()
        merged = []
        for node in own_graph + parent_graph:
            key = (node.get('subject', ''), node.get('relation', ''))
            if key not in seen:
                seen.add(key)
                merged.append(node)
        return merged

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话中的所有消息"""
        self._init_session(session_id)
        _ensure_session_loaded(self.store, self.db, session_id)
        return list(self.store.get(session_id, {}).get("recent", []))

    def delete_message(self, session_id: str, timestamp: int) -> bool:
        """
        删除指定 timestamp 的消息。
        返回 True 表示成功，False 表示未找到。
        """
        self._init_session(session_id)
        _ensure_session_loaded(self.store, self.db, session_id)
        session = self.store.get(session_id)
        if not session:
            return False
        recent = session.get("recent", [])
        original_len = len(recent)
        session["recent"] = [m for m in recent if m.get("timestamp") != timestamp]
        deleted = len(recent) - len(session["recent"])
        if deleted > 0:
            loop = asyncio.get_event_loop()
            loop.create_task(self._persist(session_id))
        return deleted > 0

    def clear_messages(self, session_id: str) -> bool:
        """清空会话中的所有消息"""
        self._init_session(session_id)
        _ensure_session_loaded(self.store, self.db, session_id)
        session = self.store.get(session_id)
        if not session:
            return False
        session["recent"] = []
        loop = asyncio.get_event_loop()
        loop.create_task(self._persist(session_id))
        return True

    def get_all_sessions(self) -> List[Dict[str, str]]:
        return [{"id": k, "name": v["name"], "parent_id": v.get("parent_id")} for k, v in self.store.items()]

    def rename_session(self, session_id: str, new_name: str) -> bool:
        if session_id not in self.store:
            return False
        self.store[session_id]["name"] = new_name
        asyncio.create_task(self._persist(session_id))
        return True

    def clear_session(self, session_id: str) -> bool:
        if session_id not in self.store:
            return False
        del self.store[session_id]
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.run_in_executor(None, partial(_delete_session_sync, self.db, session_id))
        else:
            _delete_session_sync(self.db, session_id)
        return True

    def add_practice_record(self, session_id: str, question_id: str, question_content: str,
                           knowledge_point: str, user_answer: str, correct_answer: str,
                           is_correct: bool, wrong_reason: str = None):
        """记录一次练习答题"""
        task = partial(
            _add_practice_record_sync,
            self.db, session_id, question_id, question_content,
            knowledge_point, user_answer, correct_answer, is_correct, wrong_reason
        )
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, task)
        except RuntimeError:
            # 无运行中事件循环时，退化为同步写入，避免抛错导致接口 500
            task()
        except Exception:
            # 兜底：写入失败时再尝试同步一次
            task()

    def get_practice_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """获取练习历史"""
        conn = sqlite3.connect(self.db)
        try:
            rows = conn.execute("""
                SELECT id, question_content, knowledge_point, user_answer, correct_answer, is_correct, wrong_reason, created_at
                FROM practice_records
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (session_id, limit)).fetchall()
            return [
                {
                    "id": row[0],
                    "question_content": row[1],
                    "knowledge_point": row[2],
                    "user_answer": row[3],
                    "correct_answer": row[4],
                    "is_correct": bool(row[5]),
                    "wrong_reason": row[6],
                    "created_at": row[7]
                }
                for row in rows
            ]
        finally:
            conn.close()

    def delete_practice_record(self, session_id: str, record_id: int) -> bool:
        """删除一条练习记录"""
        conn = sqlite3.connect(self.db)
        try:
            cur = conn.execute(
                "DELETE FROM practice_records WHERE session_id = ? AND id = ?",
                (session_id, int(record_id)),
            )
            conn.commit()
            return int(cur.rowcount or 0) > 0
        finally:
            conn.close()

    def clear_practice_history(self, session_id: str) -> int:
        """清空会话下所有练习记录，返回删除条数"""
        conn = sqlite3.connect(self.db)
        try:
            cur = conn.execute(
                "DELETE FROM practice_records WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
            return int(cur.rowcount or 0)
        finally:
            conn.close()

    def backfill_practice_knowledge_points(
        self,
        session_id: str,
        limit: int = 2000,
        dry_run: bool = True,
    ) -> Dict[str, int]:
        """回填练习记录考点字段，修复历史脏数据（支持 dry-run）。"""
        safe_limit = max(1, min(int(limit or 2000), 20000))
        conn = sqlite3.connect(self.db)
        try:
            rows = conn.execute(
                """
                SELECT id, knowledge_point, question_content
                FROM practice_records
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, safe_limit),
            ).fetchall()
            scanned = len(rows)
            unchanged = 0
            candidates: List[tuple[str, int]] = []
            for rec_id, raw_kp, question_content in rows:
                old_kp = str(raw_kp or "").strip()
                normalized = (
                    _clean_knowledge_point_phrase(old_kp)
                    or _clean_knowledge_point_phrase(str(question_content or ""))
                    or "未标注"
                )
                if normalized == old_kp:
                    unchanged += 1
                    continue
                candidates.append((normalized, int(rec_id)))

            updated = 0
            if not dry_run and candidates:
                conn.executemany(
                    "UPDATE practice_records SET knowledge_point = ? WHERE id = ? AND session_id = ?",
                    [(kp, rec_id, session_id) for kp, rec_id in candidates],
                )
                conn.commit()
                updated = int(len(candidates))

            return {
                "scanned": int(scanned),
                "candidate_updates": int(len(candidates)),
                "updated": int(updated),
                "unchanged": int(unchanged),
                "dry_run": int(bool(dry_run)),
            }
        finally:
            conn.close()

    def get_knowledge_point_stats(self, session_id: str) -> Dict[str, Dict]:
        """获取知识点统计：每个知识点的练习次数、正确率"""
        conn = sqlite3.connect(self.db)
        try:
            rows = conn.execute("""
                SELECT knowledge_point,
                       COUNT(*) as total,
                       SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct
                FROM practice_records
                WHERE session_id = ? AND knowledge_point IS NOT NULL
                GROUP BY knowledge_point
            """, (session_id,)).fetchall()
            # 兼容历史脏数据：按归一后的“名词短语考点”聚合
            agg: Dict[str, Dict[str, int]] = {}
            for row in rows:
                raw_kp = str(row[0] or "").strip()
                kp = _clean_knowledge_point_phrase(raw_kp) or raw_kp or "未标注"
                total = row[1]
                correct = row[2]
                if kp not in agg:
                    agg[kp] = {"total": 0, "correct": 0}
                agg[kp]["total"] += int(total or 0)
                agg[kp]["correct"] += int(correct or 0)
            stats = {}
            for kp, data in agg.items():
                total = int(data.get("total", 0))
                correct = int(data.get("correct", 0))
                stats[kp] = {
                    "total": total,
                    "correct": correct,
                    "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
                    "weak": (correct / total < 0.6) if total > 0 else False
                }
            return stats
        finally:
            conn.close()

    def _get_question_bank_chroma(self) -> Chroma:
        """获取题库 ChromaDB 实例（复用单例）"""
        if not hasattr(self, '_question_bank_chroma'):
            self._question_bank_chroma = Chroma(
                collection_name="question_bank",
                embedding_function=embed_model,
                persist_directory=chroma_conf['persist_directory'],
            )
        return self._question_bank_chroma

    def store_question_to_bank(self, session_id: str, question_id: str, question_content: str,
                                knowledge_point: str, answer: str) -> str:
        """存储题目到题库（用于相似题检索）"""
        # 计算确定性 ID
        q_hash = hashlib.md5(f"{session_id}:{question_id}".encode()).hexdigest()[:16]
        chroma_id = f"qbank_{q_hash}"

        # 构建文档
        doc = Document(
            page_content=question_content,
            metadata={
                "question_id": question_id,
                "session_id": session_id,
                "knowledge_point": knowledge_point or "",
                "answer": answer or "",
                "chroma_id": chroma_id,
            }
        )

        # 存入 ChromaDB
        chroma = self._get_question_bank_chroma()
        chroma.add_documents([doc], ids=[chroma_id])

        # 同时记录到 SQLite（直接调用同步函数，store_question_to_bank 本身是同步方法）
        try:
            _store_question_bank_sync(
                self.db, question_id, session_id, knowledge_point, question_content, answer, chroma_id
            )
        except Exception as e:
            logger.error(f"[题库] SQLite 存储失败: {e}")

        logger.info(f"[题库] 已存储题目 {question_id} 到题库")
        return chroma_id

    def search_similar_questions(
        self,
        question_content: str,
        knowledge_point: str = None,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[Dict]:
        """搜索相似题目（优先向量检索，失败/不足时回退 SQLite 历史练习兜底）"""
        try:
            # 嵌入查询内容
            query_embedding = embed_model.embed_query(question_content)

            # 搜索 ChromaDB
            chroma = self._get_question_bank_chroma()
            search_kwargs = {
                "embedding": query_embedding,
                "k": limit * 2,  # 多取一些，后面过滤
            }
            if session_id:
                search_kwargs["filter"] = {"session_id": session_id}
            results = chroma.similarity_search_by_vector(**search_kwargs)

            similar_questions = []
            seen_content = set()

            for doc in results:
                content = doc.page_content
                # 去重（相似内容可能重复）
                content_key = content[:50]
                if content_key in seen_content:
                    continue

                kp = doc.metadata.get("knowledge_point", "")
                # 如果指定了知识点，优先返回同知识点的
                if knowledge_point and kp != knowledge_point and len(similar_questions) >= limit:
                    continue

                similar_questions.append({
                    "question_content": content,
                    "answer": doc.metadata.get("answer", ""),
                    "knowledge_point": kp,
                    "question_id": doc.metadata.get("question_id", ""),
                })
                seen_content.add(content_key)

                if len(similar_questions) >= limit:
                    break

            # 如果不够，回退到 question_bank 的 SQLite（按 session + 知识点）
            if len(similar_questions) < limit:
                conn = sqlite3.connect(self.db)
                try:
                    needed = limit - len(similar_questions)
                    if session_id:
                        rows = conn.execute("""
                            SELECT question_content, answer, knowledge_point, question_id
                            FROM question_bank
                            WHERE session_id = ? AND knowledge_point = ?
                            ORDER BY RANDOM()
                            LIMIT ?
                        """, (session_id, knowledge_point, needed)).fetchall()
                    else:
                        rows = conn.execute("""
                            SELECT question_content, answer, knowledge_point, question_id
                            FROM question_bank
                            WHERE knowledge_point = ?
                            ORDER BY RANDOM()
                            LIMIT ?
                        """, (knowledge_point, needed)).fetchall()

                    for row in rows:
                        content = row[0]
                        content_key = content[:50]
                        if content_key in seen_content:
                            continue
                        similar_questions.append({
                            "question_content": row[0],
                            "answer": row[1],
                            "knowledge_point": row[2],
                            "question_id": row[3],
                        })
                        seen_content.add(content_key)

                    # 兜底：question_bank 仍不足时，从练习历史中补同知识点题目（避免空推荐）
                    needed = limit - len(similar_questions)
                    if needed > 0:
                        if session_id:
                            history_rows = conn.execute("""
                                SELECT question_content, correct_answer, knowledge_point, question_id
                                FROM practice_records
                                WHERE session_id = ? AND knowledge_point = ?
                                ORDER BY created_at DESC
                                LIMIT ?
                            """, (session_id, knowledge_point, needed * 3)).fetchall()
                        else:
                            history_rows = conn.execute("""
                                SELECT question_content, correct_answer, knowledge_point, question_id
                                FROM practice_records
                                WHERE knowledge_point = ?
                                ORDER BY created_at DESC
                                LIMIT ?
                            """, (knowledge_point, needed * 3)).fetchall()

                        for row in history_rows:
                            content = row[0] or ""
                            if not content:
                                continue
                            content_key = content[:50]
                            # 去重 + 排除当前题干
                            if content_key in seen_content or content.strip() == (question_content or "").strip():
                                continue
                            similar_questions.append({
                                "question_content": content,
                                "answer": row[1] or "",
                                "knowledge_point": row[2] or knowledge_point or "",
                                "question_id": row[3] or "",
                            })
                            seen_content.add(content_key)
                            if len(similar_questions) >= limit:
                                break
                finally:
                    conn.close()

            return similar_questions[:limit]
        except Exception as e:
            logger.error(f"[题库] 搜索相似题目失败: {e}")
            return []


def _store_question_bank_sync(db: str, question_id: str, session_id: str,
                               knowledge_point: str, question_content: str,
                               answer: str, chroma_id: str):
    """同步写入题目到 SQLite"""
    conn = sqlite3.connect(db)
    try:
        conn.execute("""
            INSERT OR REPLACE INTO question_bank
            (question_id, session_id, knowledge_point, question_content, answer, chroma_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (question_id, session_id, knowledge_point, question_content, answer, chroma_id))
        conn.commit()
    finally:
        conn.close()


def _add_practice_record_sync(db: str, session_id: str, question_id: str, question_content: str,
                              knowledge_point: str, user_answer: str, correct_answer: str,
                              is_correct: bool, wrong_reason: str):
    """同步写入一条练习记录"""
    conn = sqlite3.connect(db)
    try:
        conn.execute("""
            INSERT INTO practice_records (session_id, question_id, question_content, knowledge_point,
                                         user_answer, correct_answer, is_correct, wrong_reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, question_id, question_content, knowledge_point,
              user_answer, correct_answer, int(is_correct), wrong_reason, int(time.time())))
        conn.commit()
    finally:
        conn.close()


# 全局单例
memory_manager = SessionMemoryManager()
