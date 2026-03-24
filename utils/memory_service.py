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
from pydantic import BaseModel, Field
from typing import List, Dict
from functools import partial

from model.factory import chat_model
from utils.logger_handler import logger
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


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
            name       TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT    NOT NULL,
            role       TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            bucket     TEXT    NOT NULL   -- 'recent' | 'trash'
        );
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT    NOT NULL,
            subject    TEXT    NOT NULL,
            relation   TEXT    NOT NULL,
            object     TEXT    NOT NULL
        );
    """)
    # 创建索引，加速按 session_id 查询
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_sid ON messages(session_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_sid ON graph_nodes(session_id);")
    conn.commit()


def _load_all_sync(db: str) -> Dict[str, Dict]:
    """启动时一次性把 SQLite 全部数据读入内存 store"""
    conn = sqlite3.connect(db)
    _create_tables(conn)
    store: Dict[str, Dict] = {}

    for row in conn.execute("SELECT session_id, name FROM sessions"):
        sid, name = row
        store[sid] = {"name": name, "recent": [], "trash": [], "graph": []}

    for row in conn.execute(
        "SELECT session_id, role, content, bucket FROM messages ORDER BY id"
    ):
        sid, role, content, bucket = row
        if sid in store:
            store[sid][bucket].append({"role": role, "content": content})

    for row in conn.execute(
        "SELECT session_id, subject, relation, object FROM graph_nodes ORDER BY id"
    ):
        sid, subject, relation, obj = row
        if sid in store:
            store[sid]["graph"].append({"subject": subject, "relation": relation, "object": obj})

    conn.close()
    return store


def _append_message_sync(db: str, session_id: str, role: str, content: str, bucket: str):
    """只追加一条消息，不重写整个 session"""
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, name) VALUES (?, ?)",
            (session_id, session_id)
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, bucket) VALUES (?, ?, ?, ?)",
            (session_id, role, content, bucket)
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
            "INSERT OR REPLACE INTO sessions (session_id, name) VALUES (?, ?)",
            (session_id, session["name"])
        )
        # 清除旧消息、重写（消息量小，直接覆盖简单可靠）
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        for msg in session["recent"]:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, bucket) VALUES (?, ?, ?, 'recent')",
                (session_id, msg["role"], msg["content"])
            )
        for msg in session["trash"]:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, bucket) VALUES (?, ?, ?, 'trash')",
                (session_id, msg["role"], msg["content"])
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
        existing = {row[0] for row in conn.execute("SELECT session_id FROM sessions")}

        for sid, session in old_store.items():
            if sid in existing:
                continue  # 已迁移，跳过
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, name) VALUES (?, ?)",
                (sid, session.get("name", sid))
            )
            for bucket in ("recent", "trash"):
                for msg in session.get(bucket, []):
                    conn.execute(
                        "INSERT INTO messages (session_id, role, content, bucket) VALUES (?, ?, ?, ?)",
                        (sid, msg["role"], msg["content"], bucket)
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
    def __init__(self, max_recent_turns: int = 5, trash_threshold: int = 4):
        self.max_recent_turns = max_recent_turns
        self.trash_threshold = trash_threshold
        self.db = _db_path()

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

    def _init_session(self, session_id: str, name: str = None):
        if session_id not in self.store:
            self.store[session_id] = {
                "name": name or session_id,
                "recent": [],
                "trash": [],
                "graph": [],
            }

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def register_session(self, session_id: str, name: str):
        """显式注册/创建新会话"""
        self._init_session(session_id, name)
        self.store[session_id]["name"] = name
        asyncio.create_task(self._persist(session_id))

    async def add_message(self, session_id: str, role: str, content: str):
        """添加消息；超过阈值时 fire-and-forget 触发图谱提纯"""
        self._init_session(session_id)
        session = self.store[session_id]

        session["recent"].append({"role": role, "content": content})

        max_messages = self.max_recent_turns * 2
        if len(session["recent"]) > max_messages:
            # 滑动窗口挤出：消息状态发生结构性变化，全量覆写保证一致性
            kicked = session["recent"][:2]
            session["recent"] = session["recent"][2:]
            session["trash"].extend(kicked)

            if len(session["trash"]) >= self.trash_threshold * 2:
                asyncio.create_task(self._trigger_background_distillation(session_id))

            await self._persist(session_id)  # 结构变更：全量覆写
        else:
            # 正常追加：只写新消息这一行，不重写整个 session
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                partial(_append_message_sync, self.db, session_id, role, content, "recent")
            )

    async def _trigger_background_distillation(self, session_id: str):
        trash_data = list(self.store[session_id]["trash"])
        self.store[session_id]["trash"].clear()
        try:
            await self._distill_to_graph(session_id, trash_data)
        except Exception as e:
            logger.error(f"[记忆提纯失败] {e}")

    async def _distill_to_graph(self, session_id: str, trash_data: List[Dict]):
        parser = JsonOutputParser(pydantic_object=KnowledgeGraphNode)
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

        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in trash_data])
        chain = prompt | chat_model | parser

        logger.info(f"[记忆提纯] 正在提纯 {session_id} 的 {len(trash_data)} 条记录...")
        result = await chain.ainvoke({"history": history_text})

        if result and isinstance(result, list):
            self.store[session_id]["graph"].extend(result)
            await self._persist(session_id)
            logger.info(f"[记忆提纯] {session_id} 抽取 {len(result)} 条知识节点")

    def get_memory_context(self, session_id: str) -> str:
        self._init_session(session_id)
        session = self.store[session_id]
        if not session["graph"]:
            return ""
        context = "【以往复习知识点图谱备忘】\n"
        for node in session["graph"]:
            context += (
                f"- 概念[{node.get('subject', '')}] : "
                f"{node.get('relation', '')} -> {node.get('object', '')}\n"
            )
        return context

    def get_all_sessions(self) -> List[Dict[str, str]]:
        return [{"id": k, "name": v["name"]} for k, v in self.store.items()]

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


# 全局单例
memory_manager = SessionMemoryManager()
