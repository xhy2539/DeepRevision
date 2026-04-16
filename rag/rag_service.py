from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_community.retrievers import BM25Retriever
from model.factory import chat_model, embed_model, light_chat_model
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from utils.config_handler import chroma_conf
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any, Optional
from utils.logger_handler import logger
from utils.session_context import current_session_id
from utils.rag_metrics import rag_inc, rag_append_sample
import asyncio
import sqlite3
import os
import json
import hashlib
import math
import time
import random
import re

# Cross-Encoder 已禁用（网络问题），使用纯 RRF 检索


class SemanticCache:
    """
    RAG 语义缓存：用 embedding 相似度匹配相同/近似问题，直接返回缓存的 LLM 回复。
    缓存失效：文件上传/删除时按 session_id 清空。
    """

    def __init__(self, similarity_threshold: float = 0.92, max_candidates: int = 200):
        self.sim_threshold = similarity_threshold
        self.max_candidates = max(20, int(max_candidates or 200))
        self.db_path = os.path.join(os.getcwd(), "data", "semantic_cache.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                query_text  TEXT    NOT NULL,
                query_hash  TEXT    NOT NULL,
                embedding   TEXT    NOT NULL,
                response    TEXT    NOT NULL,
                created_at  INTEGER NOT NULL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_session ON cache(session_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_hash ON cache(query_hash);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_session_created ON cache(session_id, created_at DESC);")
        conn.commit()
        conn.close()

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def get(self, query: str, session_id: str) -> Optional[str]:
        """
        查询语义缓存。命中返回缓存的 response，未命中返回 None。
        """
        q_hash = hashlib.md5(query.encode()).hexdigest()
        conn = sqlite3.connect(self.db_path)
        try:
            # 1) 精确命中优先
            exact_row = conn.execute(
                "SELECT response FROM cache WHERE session_id = ? AND query_hash = ? ORDER BY created_at DESC LIMIT 1",
                (session_id, q_hash),
            ).fetchone()
            if exact_row and exact_row[0]:
                logger.info("[语义缓存] 精确命中")
                return str(exact_row[0])

            # 2) 语义命中仅扫描最近窗口，避免全量扫描拖慢
            rows = conn.execute(
                "SELECT embedding, response FROM cache WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, self.max_candidates),
            ).fetchall()
        finally:
            conn.close()

        # 精确命中失败后再计算 embedding，避免 embedding 故障导致精确缓存不可用
        try:
            loop = asyncio.get_event_loop()
            emb = await loop.run_in_executor(None, lambda: embed_model.embed_query(query))
        except Exception as e:
            logger.warning(f"[语义缓存] embedding 失败，仅可使用精确缓存: {e}")
            return None

        for cached_emb_str, cached_resp in rows:
            try:
                cached_emb = json.loads(cached_emb_str)
                sim = self._cosine_sim(emb, cached_emb)
                if sim >= self.sim_threshold:
                    logger.info(f"[语义缓存] 语义命中 (相似度={sim:.3f})")
                    return cached_resp
            except Exception:
                continue
        return None

    def set(self, query: str, response: str, session_id: str):
        """写入缓存"""
        try:
            emb = embed_model.embed_query(query)
        except Exception as e:
            logger.warning(f"[语义缓存] embedding 失败，跳过写入: {e}")
            return

        q_hash = hashlib.md5(query.encode()).hexdigest()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO cache (session_id, query_text, query_hash, embedding, response, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, query, q_hash, json.dumps(emb), response, int(time.time()))
        )
        conn.commit()
        conn.close()
        logger.info("[语义缓存] 已写入")

    def invalidate(self, session_id: str):
        """按 session 清空缓存（知识库变更时调用）"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM cache WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        logger.info(f"[语义缓存] 已失效 session={session_id}")


class RRFRetriever(Runnable):
    """
    RRF (Reciprocal Rank Fusion) 检索器
    公式: score(d) = Σ 1/(k + rank(d))
    k 通常取 60
    """
    def __init__(
        self,
        bm25_retriever=None,
        vector_retriever=None,
        rrf_k=60,
        rerank_top_k=8,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.rrf_k = rrf_k
        self.rerank_top_k = rerank_top_k
        self.bm25_weight = max(0.0, float(bm25_weight or 0.0))
        self.vector_weight = max(0.0, float(vector_weight or 0.0))

    def invoke(self, input, config=None):
        """同步调用"""
        raise NotImplementedError("请使用异步调用")

    async def ainvoke(self, input, config=None):
        """异步调用"""
        # 并行获取两个检索结果
        if self.bm25_retriever:
            bm25_task = self.bm25_retriever.ainvoke(input)
        else:
            async def empty():
                return []
            bm25_task = empty()

        vector_task = self.vector_retriever.ainvoke(input)

        bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task)

        # RRF 融合
        fused_docs = self._rrf_fuse(bm25_results, vector_results)
        return fused_docs[:self.rerank_top_k]

    @staticmethod
    def _doc_signature(doc: Document) -> str:
        meta = doc.metadata or {}
        src = str(meta.get("source") or meta.get("source_filename") or "")
        page = str(meta.get("page") or "")
        # 结合来源+页码+内容 hash，降低仅靠前缀导致的误合并
        content_hash = hashlib.md5(str(doc.page_content or "").encode("utf-8")).hexdigest()[:12]
        return f"{src}|{page}|{content_hash}"

    def _rrf_fuse(self, docs1: List[Document], docs2: List[Document]) -> List[Document]:
        """RRF 融合算法"""
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # 对第一个检索结果打分
        for rank, doc in enumerate(docs1, 1):
            key = self._doc_signature(doc)
            doc_scores[key] = doc_scores.get(key, 0) + (self.bm25_weight / (self.rrf_k + rank))
            doc_map[key] = doc

        # 对第二个检索结果打分
        for rank, doc in enumerate(docs2, 1):
            key = self._doc_signature(doc)
            doc_scores[key] = doc_scores.get(key, 0) + (self.vector_weight / (self.rrf_k + rank))
            doc_map[key] = doc

        # 按分数排序
        sorted_keys = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in sorted_keys]

# Cross-Encoder disabled, using RRF hybrid search

class RagSummarizeService:
    def __init__(self):
        """
        RAG 主服务初始化：
        - 组装向量检索/BM25/RRF
        - 加载缓存与检索参数
        - 准备 Query Rewrite 与最终总结链
        """
        self.vector_store_service = VectorStoreService()
        self.vector_retriever = self.vector_store_service.get_retriever()
        self.bm25_retriever = None
        self.semantic_cache = SemanticCache()

        # ============== 检索配置（从 chroma.yml 读取）==============
        self.rerank_top_k = chroma_conf.get('retrieve_top_k', 10)
        self.final_top_k = chroma_conf.get('rerank_final_k', 8)
        self.rrf_k = chroma_conf.get('rrf_k', 30)
        self.bm25_weight = chroma_conf.get('bm25_weight', 0.4)
        self.vector_weight = chroma_conf.get('vector_weight', 0.6)
        self.mmr_enabled = chroma_conf.get('mmr_enabled', False)
        self.mmr_lambda = chroma_conf.get('mmr_lambda', 0.5)
        self.ingest_wait_seconds = int(chroma_conf.get('ingest_wait_seconds', 8))
        self.ingest_poll_interval = float(chroma_conf.get('ingest_poll_interval', 1.0))
        self.fast_to_full_min_docs = max(1, int(chroma_conf.get('retrieve_fast_to_full_min_docs', 3)))
        # 检索上下文缓存默认调长：出题/组卷链路常常超过几十秒，45s 容易失效导致重复检索
        self.context_cache_ttl_seconds = int(chroma_conf.get('retrieve_context_cache_ttl_seconds', 1200))
        self.context_cache_max_entries = int(chroma_conf.get('retrieve_context_cache_max_entries', 1000))
        self.hyde_enabled = bool(chroma_conf.get('hyde_enabled', True))
        self.hyde_trigger_min_docs = max(1, int(chroma_conf.get('hyde_trigger_min_docs', 3)))
        self.hyde_max_chars = max(80, int(chroma_conf.get('hyde_max_chars', 180)))
        self._context_cache: Dict[str, Dict[str, Any]] = {}
        self.fast_retrieve_top_k = 8
        self.fast_final_top_k = 6
        self.full_retrieve_top_k = 16
        self.full_final_top_k = 10

        logger.info("[RAG] 初始化混合检索系统 (BM25 + 向量 RRF)")

        # ============== 创建混合检索器 ==============
        self.refresh(invalidate_cache=False)

    def refresh(self, invalidate_cache: bool = True):
        """
        刷新检索器状态。当知识库（Chroma）中文档发生变化时，
        调用此方法重新构建 BM25 索引，同时失效旧缓存。
        """
        if invalidate_cache:
            self.semantic_cache.invalidate(self.vector_store_service.session_id)
            self._context_cache.clear()
        # 提取当前库中所有的文档供BM25建立本地索引
        all_docs = self.vector_store_service.vector_store.get()
        if all_docs and isinstance(all_docs, dict) and "documents" in all_docs and len(all_docs["documents"]) > 0:
            bm25_docs = [Document(page_content=text, metadata=meta) for text, meta in zip(all_docs["documents"], all_docs["metadatas"])]
            bm25_fast = BM25Retriever.from_documents(bm25_docs)
            bm25_fast.k = self.fast_retrieve_top_k
            bm25_full = BM25Retriever.from_documents(bm25_docs)
            bm25_full.k = self.full_retrieve_top_k

            # 使用 RRF (Reciprocal Rank Fusion) 算法融合
            # RRF 公式: score(d) = Σ 1/(k + rank(d))
            # 不需要手动设置权重，自动融合多个检索结果
            self.retriever_fast = RRFRetriever(
                bm25_retriever=bm25_fast,
                vector_retriever=self.vector_retriever,
                rrf_k=self.rrf_k,
                rerank_top_k=self.fast_retrieve_top_k,
                bm25_weight=self.bm25_weight,
                vector_weight=self.vector_weight,
            )
            self.retriever_full = RRFRetriever(
                bm25_retriever=bm25_full,
                vector_retriever=self.vector_retriever,
                rrf_k=self.rrf_k,
                rerank_top_k=self.full_retrieve_top_k,
                bm25_weight=self.bm25_weight,
                vector_weight=self.vector_weight,
            )
            self.bm25_retriever = bm25_full
            self.retriever = self.retriever_full
            print(f"[RRF] 已启用 Reciprocal Rank Fusion 混合检索 (k={self.rrf_k})")
        else:
            # 如果库是空的（还未上传任何文件），退回到普通的向量检索防崩
            self.retriever_fast = RRFRetriever(
                bm25_retriever=None,
                vector_retriever=self.vector_retriever,
                rrf_k=self.rrf_k,
                rerank_top_k=self.fast_retrieve_top_k,
                bm25_weight=0.0,
                vector_weight=1.0,
            )
            self.retriever_full = RRFRetriever(
                bm25_retriever=None,
                vector_retriever=self.vector_retriever,
                rrf_k=self.rrf_k,
                rerank_top_k=self.full_retrieve_top_k,
                bm25_weight=0.0,
                vector_weight=1.0,
            )
            self.retriever = self.retriever_full
            print(f"[RRF] 仅使用向量检索（知识库为空）")

        # --- 步骤二：准备 Query Rewrite 的 Chain ---
        rewrite_prompt = PromptTemplate.from_template(
            "你是课件检索查询改写器。请把用户输入改写成“仅用于检索”的关键词串，词与词之间用空格分隔。\n"
            "规则：\n"
            "1) 只输出关键词，不要解释，不要标点，不要句子。\n"
            "2) 优先输出课程术语/机制名词（如 进程 调度 同步 死锁 虚拟内存 页置换 文件系统 I/O 系统调用 内核）。\n"
            "3) 若输入是泛化请求（如“出题/随机出两道/练习一下/做题”）或缺少具体考点，提取可检索的课程实体词，保留用户原词并补充：课件 章节 概念 机制\n"
            "4) 禁止输出抽象词或无检索价值词（如 综合知识应用 随机 一下 那个 这个）。\n"
            "5) 若输入里有明确考点，保留该考点并补充 2-4 个同域术语。\n"
            "原问题: {question}"
        )
        self.rewrite_chain = rewrite_prompt | light_chat_model | StrOutputParser()

        # --- 步骤三：打包装配最终总结的 Chain ---
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        """最终问答总结链：把检索上下文交给主模型生成回答。"""
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    async def retriever_docs(self, query: str, profile: str = "full") -> list[Document]:
        """按检索档位调用 retriever（`fast` 或 `full`）。"""
        retriever = self.retriever_full if profile == "full" else self.retriever_fast
        return await retriever.ainvoke(query)

    def _get_session_ingest_state(self) -> str:
        """
        返回当前会话入库状态：
        - processing: 至少存在一个文件在处理中
        - completed: 有状态文件且均非 processing
        - unknown: 状态文件不存在或不可读
        """
        try:
            status_path = os.path.join(self.vector_store_service.session_data_path, "ingest_status.json")
            if not os.path.exists(status_path):
                return "unknown"
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f) or {}
            if not isinstance(status, dict) or not status:
                return "unknown"
            for meta in status.values():
                if str((meta or {}).get("status", "")).strip() == "processing":
                    return "processing"
            return "completed"
        except Exception:
            return "unknown"

    def _build_courseware_fallback_query(self) -> str:
        """
        基于当前会话文件名构造兜底检索词，避免固定锚点导致偏题。
        """
        try:
            session_dir = self.vector_store_service.session_data_path
            if not os.path.isdir(session_dir):
                return "课件 章节 概念 机制"
            terms: List[str] = []
            for fname in os.listdir(session_dir):
                if fname.startswith("."):
                    continue
                base, ext = os.path.splitext(fname)
                if ext.lower() not in {".pdf", ".docx", ".doc", ".txt", ".ppt", ".pptx", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
                    continue
                cleaned = str(base).replace("_", " ").replace("-", " ")
                for token in cleaned.split():
                    token = token.strip()
                    if len(token) >= 2:
                        terms.append(token)
                if len(terms) >= 12:
                    break
            if terms:
                return " ".join(terms[:12]) + " 课件 章节 概念 机制"
        except Exception:
            pass
        return "课件 章节 概念 机制"

    @staticmethod
    def _is_generic_query(query: str) -> bool:
        """判断是否为泛化提问（出题/随机练习类），用于随机化与兜底策略。"""
        text = str(query or "").strip()
        if not text:
            return True
        generic_signals = (
            "随机", "出题", "练习", "做题", "来几道", "来点题", "测试一下",
            "综合", "重点知识点", "复习一下", "考我", "题目",
        )
        return any(sig in text for sig in generic_signals)

    @staticmethod
    def _normalize_query(query: str) -> str:
        """统一 query 归一化（空白合并 + 小写），用于缓存键稳定。"""
        return re.sub(r"\s+", " ", str(query or "").strip().lower())

    def _build_context_cache_key(self, query: str, mode: str) -> str:
        """构造检索上下文缓存键：`session + mode + normalized_query`。"""
        sid = current_session_id.get() or "default"
        norm = self._normalize_query(query)
        key_src = f"{sid}|{mode}|{norm}"
        return hashlib.md5(key_src.encode("utf-8")).hexdigest()

    def _get_cached_context(self, query: str, mode: str) -> Optional[str]:
        """读取检索上下文缓存（仅 `rag_chat` 模式生效，带 TTL 过期）。"""
        if mode != "rag_chat":
            return None
        key = self._build_context_cache_key(query, mode)
        item = self._context_cache.get(key)
        if not item:
            return None
        now = time.time()
        if now - float(item.get("ts", 0.0)) > max(1, self.context_cache_ttl_seconds):
            self._context_cache.pop(key, None)
            return None
        return str(item.get("context", "") or "")

    def _set_cached_context(self, query: str, mode: str, context: str) -> None:
        """写入检索上下文缓存，并做轻量 LRU 式裁剪。"""
        if mode != "rag_chat":
            return
        if not context or not str(context).strip():
            return
        key = self._build_context_cache_key(query, mode)
        self._context_cache[key] = {"context": str(context), "ts": time.time()}
        # 简单 LRU-ish 清理：删除最早写入的缓存
        if len(self._context_cache) > max(10, self.context_cache_max_entries):
            oldest_key = min(self._context_cache.items(), key=lambda kv: float(kv[1].get("ts", 0.0)))[0]
            self._context_cache.pop(oldest_key, None)

    @staticmethod
    def _extract_rewrite_anchors(query: str) -> List[str]:
        text = str(query or "")
        anchors: List[str] = []
        seen = set()

        def _add(token: str):
            t = str(token or "").strip()
            if not t:
                return
            norm = re.sub(r"\s+", " ", t.lower())
            if norm in seen:
                return
            seen.add(norm)
            anchors.append(t)

        for m in re.findall(r"`([^`]{1,80})`", text):
            _add(m)
        for m in re.findall(r"\bgit\s+[a-zA-Z0-9][a-zA-Z0-9\-]*(?:\s+--?[a-zA-Z0-9\-]+)*", text, flags=re.IGNORECASE):
            _add(m)
        for m in re.findall(r"\b[A-Za-z][A-Za-z0-9_/\-]{1,}\b", text):
            _add(m)
        for m in re.findall(r"[\u4e00-\u9fa5]{2,10}", text):
            _add(m)

        return anchors[:10]

    @staticmethod
    def _looks_invalid_rewrite(text: str) -> bool:
        cleaned = str(text or "").strip()
        if not cleaned or len(cleaned) < 2:
            return True
        generic_only = {"课件", "章节", "概念", "机制", "知识点", "复习", "题目", "做题", "出题"}
        tokens = [t for t in re.split(r"\s+", cleaned) if t]
        if tokens and all(t in generic_only for t in tokens):
            return True
        return False

    def _rewrite_guard(self, query: str, expanded_query: str) -> str:
        """改写保护：若重写后丢失关键锚点，把原 query 关键词补回。"""
        base = str(query or "").strip()
        expanded = str(expanded_query or "").strip()
        if self._looks_invalid_rewrite(expanded):
            return base or expanded

        anchors = self._extract_rewrite_anchors(base)
        if not anchors:
            return expanded

        expanded_norm = self._normalize_query(expanded)
        missing: List[str] = []
        for anchor in anchors:
            a_norm = self._normalize_query(anchor)
            if a_norm and a_norm not in expanded_norm:
                missing.append(anchor)
        if not missing:
            return expanded
        guarded = (expanded + " " + " ".join(missing[:4])).strip()
        return guarded

    def _select_profile(self, query: str) -> str:
        """检索档位选择：当前默认先 `fast`，召回不足再升级 `full`。"""
        # 延迟优先：默认先走 fast，召回不足再在 retrieve_context 中升级 full
        return "fast"

    def _profile_limits(self, profile: str) -> tuple[int, int]:
        """返回档位对应的召回上限与最终保留上限。"""
        if profile == "full":
            return self.full_retrieve_top_k, self.full_final_top_k
        return self.fast_retrieve_top_k, self.fast_final_top_k

    @staticmethod
    def _merge_ranked_doc_lists(
        ranked_lists: List[tuple[List[Document], float]],
        rrf_k: int = 60,
    ) -> List[Document]:
        """多路候选文档按加权 RRF 融合，常用于 HyDE 补召回后重排。"""
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        kk = max(1, int(rrf_k or 60))
        for docs, weight in ranked_lists:
            ww = max(0.0, float(weight or 0.0))
            if ww <= 0:
                continue
            for rank, doc in enumerate(docs or [], 1):
                key = RRFRetriever._doc_signature(doc)
                scores[key] = scores.get(key, 0.0) + (ww / (kk + rank))
                doc_map[key] = doc
        sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[k] for k, _ in sorted_keys]

    async def _generate_hyde_query(self, user_query: str, expanded_query: str) -> str:
        """生成 HyDE 假设性教材片段，仅用于向量检索补召回。"""
        model = light_chat_model or chat_model
        if model is None:
            return ""
        prompt = PromptTemplate.from_template(
            "你是课件检索辅助器。请基于用户问题写一小段“可能出现在课件里的教材式内容”，"
            "用于向量检索补召回。\n"
            "要求：\n"
            "1) 仅输出一段正文，不要解释。\n"
            "2) 不要编造具体数字或案例。\n"
            "3) 长度控制在 {max_chars} 字以内。\n"
            "用户问题: {query}\n"
            "改写关键词: {expanded_query}\n"
        )
        chain = prompt | model | StrOutputParser()
        try:
            text = await asyncio.wait_for(
                chain.ainvoke(
                    {
                        "query": str(user_query or "").strip(),
                        "expanded_query": str(expanded_query or "").strip(),
                        "max_chars": self.hyde_max_chars,
                    }
                ),
                timeout=3.0,
            )
        except Exception:
            return ""
        clean = re.sub(r"\s+", " ", str(text or "").strip())
        if not clean:
            return ""
        if len(clean) > self.hyde_max_chars:
            clean = clean[: self.hyde_max_chars].rstrip()
        return clean

    @staticmethod
    def _get_doc_source(doc: Document) -> str:
        meta = doc.metadata or {}
        return str(meta.get("source_filename") or meta.get("source") or "unknown")

    def _apply_source_cap(self, docs: List[Document], final_limit: int, per_source_cap: int = 2) -> List[Document]:
        """来源均衡：限制单文件命中条数，提升最终上下文的来源多样性。"""
        if not docs:
            return []
        picked: List[Document] = []
        overflow: List[Document] = []
        src_count: Dict[str, int] = {}
        for doc in docs:
            src = self._get_doc_source(doc)
            count = src_count.get(src, 0)
            if count < per_source_cap and len(picked) < final_limit:
                picked.append(doc)
                src_count[src] = count + 1
            else:
                overflow.append(doc)
        if len(picked) < final_limit:
            need = final_limit - len(picked)
            picked.extend(overflow[:need])
        return picked[:final_limit]

    async def _rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """
        使用 LLM 对检索结果重排序（fix #9：改为 async，使用 ainvoke 避免阻塞 event loop）
        529错误时快速失败，避免长时间等待
        """
        if not docs or len(docs) <= 1:
            return docs

        try:
            doc_texts = "\n\n".join([
                f"【文档{i+1}】{doc.page_content[:500]}"
                for i, doc in enumerate(docs)
            ])

            rerank_prompt = f"""请根据用户问题，对以下文档进行相关性排序。

用户问题：{query}

文档列表：
{doc_texts}

请按相关性从高到低排序，返回文档编号列表（格式：1, 2, 3... 只返回编号列表，不需要其他内容）。"""

            # 用 asyncio.wait_for 加 3 秒超时（原10秒太长，10个topic浪费100秒），529 错误快速失败
            try:
                response = await asyncio.wait_for(
                    light_chat_model.ainvoke(rerank_prompt),
                    timeout=3.0
                )
            except asyncio.TimeoutError:
                logger.warning("[LLM Rerank] 超时（3秒），跳过重排")
                return docs[:self.final_top_k]
            except Exception as e:
                # 529 等 API 错误快速失败
                if '529' in str(e) or 'overloaded' in str(e).lower():
                    logger.warning(f"[LLM Rerank] API 过载（529），跳过重排")
                else:
                    logger.warning(f"[LLM Rerank] 调用失败: {e}")
                return docs[:self.final_top_k]

            ranking = response.content.strip()

            try:
                ranks = [int(x.strip()) for x in ranking.split(",") if x.strip().isdigit()]
                if ranks:
                    reranked = []
                    for rank in ranks:
                        if 0 < rank <= len(docs):
                            reranked.append(docs[rank - 1])
                    for doc in docs:
                        if doc not in reranked:
                            reranked.append(doc)
                    print(f"[LLM Rerank] 原始 {len(docs)} 个 -> 重排后 {len(reranked)} 个")
                    return reranked[:self.final_top_k]
            except Exception:  # fix #15：不用裸 except
                pass

            return docs[:self.final_top_k]

        except Exception as e:
            logger.warning(f"[LLM Rerank] 重排序失败: {e}")
            return docs[:self.final_top_k]

    async def retrieve_context(self, query: str, mode: str = "rag_chat") -> str:
        """
        只做检索，返回格式化后的原始课件片段字符串，不调用最终 LLM。
        供 Supervisor RAG SubAgent 使用，避免双重 LLM 调用。
        """
        retrieve_start = time.time()
        rag_inc("rag_retrieve_calls", 1)
        cache_eligible = (mode == "rag_chat")
        # 仅 rag_chat 走上下文缓存：quiz/exam 侧通常希望实时检索，不复用旧上下文。
        if cache_eligible:
            rag_inc("rag_cache_eligible_calls", 1)
        cached_context = self._get_cached_context(query, mode)
        if cached_context is not None:
            if cache_eligible:
                rag_inc("rag_retrieve_cache_hit", 1)
            rag_append_sample("rag_retrieve_latency_samples_ms", int((time.time() - retrieve_start) * 1000))
            logger.info(f"[RAG] retrieve_context 缓存命中 mode={mode}, len={len(cached_context)}")
            return cached_context
        if cache_eligible:
            rag_inc("rag_retrieve_cache_miss", 1)

        try:
            rewritten = await asyncio.wait_for(
                self.rewrite_chain.ainvoke({"question": query}),
                timeout=3.0,
            )
            # 改写后再过 guard，防止关键词“改飞”导致召回偏移。
            expanded_query = self._rewrite_guard(query, rewritten)
            if self._normalize_query(expanded_query) != self._normalize_query(str(rewritten or "")):
                logger.info(f"[RAG] rewrite_guard 生效，补齐关键锚点。query={str(query)[:80]}")
        except Exception:
            expanded_query = query

        profile = self._select_profile(query)
        _, profile_final_k = self._profile_limits(profile)

        context_docs: List[Document] = []
        try:
            context_docs = await asyncio.wait_for(
                self.retriever_docs(expanded_query, profile=profile),
                timeout=8.0,
            )
        except Exception as e:
            logger.warning(f"[RAG] retrieve_context 主查询失败，进入回退检索: {e}")
            fallback_queries: List[str] = []
            if str(query or "").strip() and str(query).strip() != str(expanded_query).strip():
                fallback_queries.append(str(query).strip())
            fallback_queries.append(self._build_courseware_fallback_query())
            for fq in fallback_queries:
                try:
                    context_docs = await asyncio.wait_for(self.retriever_docs(fq, profile=profile), timeout=6.0)
                    if context_docs:
                        logger.info(f"[RAG] retrieve_context 回退检索成功: {fq[:80]}")
                        break
                except Exception:
                    continue

        # 低延迟优先：fast 档命中不足时升级 full 档重试
        if profile == "fast" and len(context_docs) < self.fast_to_full_min_docs:
            rag_inc("fast_to_full_escalations", 1)
            logger.info(
                f"[RAG] fast 档命中不足（docs={len(context_docs)}<{self.fast_to_full_min_docs}），升级 full 档重试"
            )
            try:
                full_docs = await asyncio.wait_for(self.retriever_docs(expanded_query, profile="full"), timeout=6.0)
                if full_docs:
                    context_docs = full_docs
                profile = "full"
                _, profile_final_k = self._profile_limits(profile)
            except Exception:
                pass

        # 上传后立刻提问时，若仍在处理中且本轮召回为空，短暂等待并重试，降低“已上传但查不到”概率
        if not context_docs and self._get_session_ingest_state() == "processing":
            deadline = time.time() + max(1, self.ingest_wait_seconds)
            while time.time() < deadline:
                await asyncio.sleep(max(0.3, self.ingest_poll_interval))
                self.refresh(invalidate_cache=False)
                try:
                    context_docs = await asyncio.wait_for(self.retriever_docs(expanded_query, profile=profile), timeout=6.0)
                    if context_docs:
                        logger.info("[RAG] ingest processing wait-retry 命中结果")
                        break
                except Exception:
                    pass

        if self.hyde_enabled and len(context_docs) < self.hyde_trigger_min_docs:
            # 仅低召回触发 HyDE，避免每次都增加额外延迟。
            rag_inc("hyde_trigger_count", 1)
            hyde_query = await self._generate_hyde_query(query, expanded_query)
            if hyde_query:
                try:
                    hyde_docs = await asyncio.wait_for(self.vector_retriever.ainvoke(hyde_query), timeout=6.0)
                except Exception:
                    hyde_docs = []
                if hyde_docs:
                    context_docs = self._merge_ranked_doc_lists(
                        [(context_docs, 1.0), (hyde_docs, 0.8)],
                        rrf_k=self.rrf_k,
                    )
                    logger.info(f"[RAG] HyDE 补召回命中 docs={len(hyde_docs)}")

        # 泛化提问：保留前部高相关锚点，再对候选尾部做随机化
        if context_docs and self._is_generic_query(query):
            pool_size = min(len(context_docs), max(profile_final_k, profile_final_k * 3))
            pool = list(context_docs[:pool_size])
            anchor_n = min(2, len(pool))
            anchors = pool[:anchor_n]
            tail = pool[anchor_n:]
            if tail:
                rng_seed = f"{current_session_id.get() or 'default'}:{hashlib.md5(str(query).encode('utf-8')).hexdigest()}:{time.time_ns()}"
                rng = random.Random(rng_seed)
                rng.shuffle(tail)
            context_docs = (anchors + tail)[: profile_final_k]
        # 跳过 LLM rerank（rerank 每次都超时 3 秒，10 个 topic 浪费 ~30 秒，且效果不明显）
        # RRF 检索结果已经足够好

        # 来源多样性约束：单来源最多保留 2 条，不足则回填。
        context_docs = self._apply_source_cap(context_docs, final_limit=profile_final_k, per_source_cap=2)

        context = ""
        for i, doc in enumerate(context_docs, 1):
            context += f"[参考资料{i}]:参考资料:{doc.page_content}|参考元数据:{doc.metadata}\n"
        if not context.strip():
            rag_inc("rag_empty_context_count", 1)
        else:
            self._set_cached_context(query, mode, context)

        rag_append_sample("rag_retrieve_latency_samples_ms", int((time.time() - retrieve_start) * 1000))
        logger.info(
            f"[RAG] retrieve_context 完成 mode={mode}, profile={profile}, docs={len(context_docs)}, context_len={len(context)}"
        )
        return context

    async def rag_summarize(self, query: str) -> str:
        """完整 RAG 问答链（检索 + 总结），会写入语义缓存。"""
        session_id = current_session_id.get() or "default"

        cached = await self.semantic_cache.get(query, session_id)
        if cached is not None:
            return cached

        # [高级流机制1] Query Rewrite
        try:
            expanded_query = await self.rewrite_chain.ainvoke({"question": query})
            print(f"[Query Rewrite] 改写前: {query} -> 改写后: {expanded_query}")
        except Exception as e:
            # 兼容：如果大模型临时阻断，依然跑原始query
            print(f"[Query Rewrite error] {e}")
            expanded_query = query

        # [高级流机制2] 带着重写好的搜索词去执行混合检索
        context_docs = await self.retriever_docs(expanded_query)

        # [高级流机制3] Cross-Encoder 重排序，精筛 Top 3
        # 跳过 LLM rerank（rerank 每次都超时 3 秒，10 个 topic 浪费 ~30 秒，且效果不明显）
        # RRF 检索结果已经足够好

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            # 将文档的元数据合并上，这会在最后出处呈现上发挥大作用
            context += f"[参考资料{counter}]:参考资料:{doc.page_content}|参考元数据:{doc.metadata}\n"

        response = await self.chain.ainvoke(
            {"input": query,
             "context": context,
             }
        )
        self.semantic_cache.set(query, response, session_id)
        return response

if __name__ == "__main__":
    rag_service = RagSummarizeService()
    print(rag_service.rag_summarize("帮我复习一下第三章的核心考点是什么"))
