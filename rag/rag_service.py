from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_community.retrievers import BM25Retriever
from model.factory import chat_model, light_chat_model
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any, Optional
from utils.logger_handler import logger
from utils.session_context import current_session_id
from utils.rag_metrics import rag_inc, rag_append_sample
from utils.kb_version import read_kb_version
import asyncio
import os
import json
import hashlib
import time
import random
import re

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
        result_limit=8,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.rrf_k = rrf_k
        self.result_limit = result_limit
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
        return fused_docs[:self.result_limit]

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
        bm25_ranks: Dict[str, int] = {}
        vector_ranks: Dict[str, int] = {}

        # 对第一个检索结果打分
        for rank, doc in enumerate(docs1, 1):
            key = self._doc_signature(doc)
            doc_scores[key] = doc_scores.get(key, 0) + (self.bm25_weight / (self.rrf_k + rank))
            doc_map[key] = doc
            bm25_ranks[key] = rank

        # 对第二个检索结果打分
        for rank, doc in enumerate(docs2, 1):
            key = self._doc_signature(doc)
            doc_scores[key] = doc_scores.get(key, 0) + (self.vector_weight / (self.rrf_k + rank))
            doc_map[key] = doc
            vector_ranks[key] = rank

        # 按分数排序
        sorted_keys = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        fused: List[Document] = []
        for key, score in sorted_keys:
            doc = doc_map[key]
            metadata = dict(doc.metadata or {})
            metadata["retrieval_chunk_id"] = key
            metadata["rrf_score"] = round(float(score), 8)
            if key in bm25_ranks:
                metadata["bm25_rank"] = bm25_ranks[key]
            if key in vector_ranks:
                metadata["vector_rank"] = vector_ranks[key]
            fused.append(Document(page_content=doc.page_content, metadata=metadata))
        return fused

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
        # ============== 检索配置（从 chroma.yml 读取）==============
        self.rrf_k = chroma_conf.get('rrf_k', 30)
        self.bm25_weight = chroma_conf.get('bm25_weight', 0.4)
        self.vector_weight = chroma_conf.get('vector_weight', 0.6)
        self.ingest_wait_seconds = int(chroma_conf.get('ingest_wait_seconds', 8))
        self.ingest_poll_interval = float(chroma_conf.get('ingest_poll_interval', 1.0))
        self.fast_to_full_min_docs = max(1, int(chroma_conf.get('retrieve_fast_to_full_min_docs', 3)))
        self.fast_to_full_min_sources = max(1, int(chroma_conf.get('retrieve_fast_to_full_min_sources', 2)))
        # 检索上下文缓存默认调长：出题/组卷链路常常超过几十秒，45s 容易失效导致重复检索
        self.context_cache_ttl_seconds = int(chroma_conf.get('retrieve_context_cache_ttl_seconds', 1200))
        self.context_cache_max_entries = int(chroma_conf.get('retrieve_context_cache_max_entries', 1000))
        self.hyde_enabled = bool(chroma_conf.get('hyde_enabled', True))
        self.hyde_trigger_min_docs = max(1, int(chroma_conf.get('hyde_trigger_min_docs', 3)))
        self.hyde_max_chars = max(80, int(chroma_conf.get('hyde_max_chars', 180)))
        self.rerank_enabled = bool(chroma_conf.get('rerank_enabled', False))
        self.rerank_model = str(chroma_conf.get('rerank_model', 'qwen3.7-text-rerank'))
        self.rerank_timeout_seconds = float(chroma_conf.get('rerank_timeout_seconds', 8.0))
        self.query_rewrite_timeout_seconds = float(chroma_conf.get("query_rewrite_timeout_seconds", 6.0))
        self._context_cache: Dict[str, Dict[str, Any]] = {}
        self._kb_version = 0
        self._kb_version_check_ts = 0.0
        self._retrieve_config_signature = ""
        self.fast_retrieve_top_k = 8
        self.fast_final_top_k = 6
        self.full_retrieve_top_k = 16
        self.full_final_top_k = 10
        self.parent_context_fast_k = max(1, int(chroma_conf.get("parent_context_fast_k", 4)))
        self.parent_context_full_k = max(1, int(chroma_conf.get("parent_context_full_k", 6)))

        logger.info("[RAG] 初始化混合检索系统 (BM25 + 向量 RRF)")

        # ============== 创建混合检索器 ==============
        self.refresh(invalidate_cache=False)

    def refresh(self, invalidate_cache: bool = True):
        """
        刷新检索器状态。当知识库（Chroma）中文档发生变化时，
        调用此方法重新构建 BM25 索引，同时失效旧缓存。
        """
        self._kb_version = self._get_kb_version(force_reload=True)
        self._retrieve_config_signature = self._build_retrieve_config_signature()
        if invalidate_cache:
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
                result_limit=self.fast_retrieve_top_k,
                bm25_weight=self.bm25_weight,
                vector_weight=self.vector_weight,
            )
            self.retriever_full = RRFRetriever(
                bm25_retriever=bm25_full,
                vector_retriever=self.vector_retriever,
                rrf_k=self.rrf_k,
                result_limit=self.full_retrieve_top_k,
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
                result_limit=self.fast_retrieve_top_k,
                bm25_weight=0.0,
                vector_weight=1.0,
            )
            self.retriever_full = RRFRetriever(
                bm25_retriever=None,
                vector_retriever=self.vector_retriever,
                rrf_k=self.rrf_k,
                result_limit=self.full_retrieve_top_k,
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
                if ext.lower() not in {".pdf", ".docx", ".txt", ".pptx", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
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

    def _get_kb_version(self, force_reload: bool = False) -> int:
        """读取会话知识库版本号；短周期缓存避免每次都读磁盘。"""
        now = time.time()
        if force_reload or (now - self._kb_version_check_ts >= 2.0):
            self._kb_version = read_kb_version(self.vector_store_service.session_data_path)
            self._kb_version_check_ts = now
        return int(self._kb_version or 0)

    def _build_retrieve_config_signature(self) -> str:
        """生成检索配置签名，避免参数变化后复用旧缓存。"""
        payload = {
            "fast_top_k": int(self.fast_retrieve_top_k),
            "fast_final_k": int(self.fast_final_top_k),
            "full_top_k": int(self.full_retrieve_top_k),
            "full_final_k": int(self.full_final_top_k),
            "rerank_enabled": self.rerank_enabled,
            "rerank_model": self.rerank_model,
            "parent_context_fast_k": self.parent_context_fast_k,
            "parent_context_full_k": self.parent_context_full_k,
            "source_cap": 2,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]

    def _build_context_cache_key(self, query: str, mode: str) -> str:
        """构造检索上下文缓存键：`session + mode + normalized_query + kb_version + top_k签名`。"""
        sid = self.vector_store_service.session_id or current_session_id.get() or "default"
        norm = self._normalize_query(query)
        key_src = (
            f"{sid}|{mode}|{norm}"
            f"|kb:{self._get_kb_version()}"
            f"|retr:{self._retrieve_config_signature}"
        )
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

        # 疑问/停用/虚词：剥离后不留下检索价值，避免把「讲了什么/是什么」当锚点
        _cjk_stop = re.compile(
            r"什么|怎么|如何|为什么|为啥|哪些|哪项|哪一|哪个|哪|"
            r"吗|呢|啊|吧|哦|呀|"
            r"讲了|讲什么|讲一下|介绍|介绍一下|解释|说说|说明|简述|描述|"
            r"是什么|有哪些|有什么|有没有|是不是|"
            r"一下|帮我|请问|"
            r"的|了|是|在|有|和|与|及|或|这|那"
        )

        def _clean_cjk(segment: str) -> str:
            return _cjk_stop.sub("", str(segment or "")).strip()

        for m in re.findall(r"`([^`]{1,80})`", text):
            _add(m)
        for m in re.findall(r"\bgit\s+[a-zA-Z0-9][a-zA-Z0-9\-]*(?:\s+--?[a-zA-Z0-9\-]+)*", text, flags=re.IGNORECASE):
            _add(m)
        for m in re.findall(r"\b[A-Za-z][A-Za-z0-9_/\-]{1,}\b", text, flags=re.ASCII):
            _add(m)
        # 结构锚点：第 N 章/节/部分/讲/篇/单元（如「第3章」「第 3 节」）
        for m in re.findall(r"第\s*[0-9零一二三四五六七八九十百]+\s*[章节部分讲篇单元]", text):
            _add(re.sub(r"\s+", "", m))
        # 中文片段：剥离疑问/停用词后仍保留 ≥2 个实词字符才作为锚点
        for m in re.findall(r"[\u4e00-\u9fa5]{2,10}", text):
            cleaned = _clean_cjk(m)
            if len(cleaned) >= 2:
                _add(cleaned)

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

    def _count_unique_sources(self, docs: List[Document]) -> int:
        """统计候选文档的来源数，用于 fast->full 升级判定。"""
        sources = {self._get_doc_source(doc) for doc in docs or [] if doc is not None}
        return len({s for s in sources if str(s).strip()})

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

    async def _rerank_documents(self, query: str, docs: List[Document], top_n: int, trace: List[Dict[str, Any]]) -> List[Document]:
        """Use DashScope rerank on the small RRF candidate set; fail open to RRF order."""
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
        if not self.rerank_enabled or not api_key or not docs:
            trace.append({"stage": "rerank", "status": "skipped", "reason": "disabled_or_missing_key"})
            return docs
        try:
            from dashscope import TextReRank
            texts = [str(doc.page_content or "")[:6000] for doc in docs]
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    TextReRank.call,
                    model=self.rerank_model,
                    query=str(query),
                    documents=texts,
                    top_n=min(max(1, int(top_n)), len(docs)),
                    return_documents=False,
                    api_key=api_key,
                ),
                timeout=self.rerank_timeout_seconds,
            )
            output = getattr(response, "output", None) or {}
            results = output.get("results", []) if isinstance(output, dict) else []
            ranked: List[Document] = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                idx = item.get("index")
                if isinstance(idx, int) and 0 <= idx < len(docs):
                    doc = docs[idx]
                    meta = dict(doc.metadata or {})
                    meta["rerank_score"] = item.get("relevance_score")
                    ranked.append(Document(page_content=doc.page_content, metadata=meta))
            if ranked:
                trace.append({"stage": "rerank", "status": "success", "model": self.rerank_model, "candidate_count": len(docs), "selected_count": len(ranked)})
                return ranked
            trace.append({"stage": "rerank", "status": "empty", "model": self.rerank_model})
        except Exception as exc:
            trace.append({"stage": "rerank", "status": "failed", "model": self.rerank_model, "error": str(exc)[:240]})
            logger.warning(f"[RAG] rerank failed, fallback RRF: {exc}")
        return docs

    async def retrieve_context(
        self,
        query: str,
        mode: str = "rag_chat",
        trace: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        只做检索，返回格式化后的原始课件片段字符串，不调用最终 LLM。
        供 Supervisor RAG SubAgent 使用，避免双重 LLM 调用。
        """
        retrieve_start = time.time()
        trace = trace if trace is not None else []
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
            trace.append({
                "stage": "cache_lookup",
                "status": "hit",
                "mode": mode,
                "duration_ms": int((time.time() - retrieve_start) * 1000),
            })
            trace.append({
                "stage": "finalize",
                "status": "success",
                "profile": "cache",
                "selected_chunks": [],
                "duration_ms": int((time.time() - retrieve_start) * 1000),
            })
            return cached_context
        if cache_eligible:
            rag_inc("rag_retrieve_cache_miss", 1)

        rewrite_start = time.time()
        rewrite_status = "success"
        guard_applied = False
        try:
            rewritten = await asyncio.wait_for(
                self.rewrite_chain.ainvoke({"question": query}),
                timeout=self.query_rewrite_timeout_seconds,
            )
            # 改写后再过 guard，防止关键词“改飞”导致召回偏移。
            expanded_query = self._rewrite_guard(query, rewritten)
            guard_applied = self._normalize_query(expanded_query) != self._normalize_query(str(rewritten or ""))
            if guard_applied:
                logger.info(f"[RAG] rewrite_guard 生效，补齐关键锚点。query={str(query)[:80]}")
        except Exception:
            expanded_query = query
            rewrite_status = "fallback"
        trace.append({
            "stage": "query_rewrite",
            "status": rewrite_status,
            "original_query": str(query),
            "rewritten_query": str(expanded_query),
            "guard_applied": guard_applied,
            "duration_ms": int((time.time() - rewrite_start) * 1000),
        })

        profile = self._select_profile(query)
        _, profile_final_k = self._profile_limits(profile)

        context_docs: List[Document] = []
        retrieval_succeeded = False
        hybrid_start = time.time()
        try:
            context_docs = await asyncio.wait_for(
                self.retriever_docs(expanded_query, profile=profile),
                timeout=8.0,
            )
            retrieval_succeeded = True
        except Exception as e:
            logger.warning(f"[RAG] retrieve_context 主查询失败，进入回退检索: {e}")
            fallback_queries: List[str] = []
            if str(query or "").strip() and str(query).strip() != str(expanded_query).strip():
                fallback_queries.append(str(query).strip())
            fallback_queries.append(self._build_courseware_fallback_query())
            for fq in fallback_queries:
                try:
                    context_docs = await asyncio.wait_for(self.retriever_docs(fq, profile=profile), timeout=6.0)
                    retrieval_succeeded = True
                    if context_docs:
                        logger.info(f"[RAG] retrieve_context 回退检索成功: {fq[:80]}")
                        break
                except Exception:
                    continue
        trace.append({
            "stage": "hybrid_retrieval",
            "status": "success" if retrieval_succeeded else "failed",
            "profile": profile,
            "bm25_candidates": sum(1 for doc in context_docs if (doc.metadata or {}).get("bm25_rank")),
            "vector_candidates": sum(1 for doc in context_docs if (doc.metadata or {}).get("vector_rank")),
            "fused_candidates": len(context_docs),
            "duration_ms": int((time.time() - hybrid_start) * 1000),
        })

        # 低延迟优先：fast 档命中不足或来源过于单一时升级 full 档重试
        unique_sources = self._count_unique_sources(context_docs)
        low_docs = len(context_docs) < self.fast_to_full_min_docs
        low_sources = unique_sources < self.fast_to_full_min_sources
        if profile == "fast" and (low_docs or low_sources):
            rag_inc("fast_to_full_escalations", 1)
            reason = []
            if low_docs:
                reason.append(f"docs={len(context_docs)}<{self.fast_to_full_min_docs}")
            if low_sources:
                reason.append(f"sources={unique_sources}<{self.fast_to_full_min_sources}")
            logger.info(
                f"[RAG] fast 档触发升级（{', '.join(reason)}），升级 full 档重试"
            )
            trace.append({
                "stage": "profile_escalation",
                "status": "triggered",
                "from": "fast",
                "to": "full",
                "reason": reason,
            })
            try:
                full_docs = await asyncio.wait_for(self.retriever_docs(expanded_query, profile="full"), timeout=6.0)
                retrieval_succeeded = True
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
                    retrieval_succeeded = True
                    if context_docs:
                        logger.info("[RAG] ingest processing wait-retry 命中结果")
                        break
                except Exception:
                    pass

        if self.hyde_enabled and len(context_docs) < self.hyde_trigger_min_docs:
            # 仅低召回触发 HyDE，避免每次都增加额外延迟。
            rag_inc("hyde_trigger_count", 1)
            hyde_query = await self._generate_hyde_query(query, expanded_query)
            hyde_added = 0
            if hyde_query:
                try:
                    hyde_docs = await asyncio.wait_for(self.vector_retriever.ainvoke(hyde_query), timeout=6.0)
                except Exception:
                    hyde_docs = []
                if hyde_docs:
                    before_hyde = len(context_docs)
                    context_docs = self._merge_ranked_doc_lists(
                        [(context_docs, 1.0), (hyde_docs, 0.8)],
                        rrf_k=self.rrf_k,
                    )
                    hyde_added = max(0, len(context_docs) - before_hyde)
                    logger.info(f"[RAG] HyDE 补召回命中 docs={len(hyde_docs)}")
            trace.append({
                "stage": "hyde",
                "status": "success" if hyde_added else "empty",
                "triggered": True,
                "added_candidates": hyde_added,
            })

        # RRF 先缩小候选集，再调用重排模型，避免对整库逐条打分。
        if context_docs:
            context_docs = await self._rerank_documents(
                query,
                context_docs,
                top_n=profile_final_k,
                trace=trace,
            )

        # 检索与重排都在细粒度子块上完成，生成前再回溯完整父块。
        if context_docs:
            child_count = len(context_docs)
            parent_limit = self.parent_context_full_k if profile == "full" else self.parent_context_fast_k
            context_docs = self.vector_store_service.expand_parent_documents(
                context_docs,
                limit=parent_limit,
            )
            trace.append({
                "stage": "parent_expansion",
                "status": "success" if context_docs else "empty",
                "child_count": child_count,
                "parent_count": len(context_docs),
                "parent_limit": parent_limit,
            })
            profile_final_k = min(profile_final_k, parent_limit)

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
        selected_chunks = []
        for doc in context_docs:
            meta = doc.metadata or {}
            selected_chunks.append({
                "chunk_id": str(meta.get("chunk_id") or meta.get("retrieval_chunk_id") or ""),
                "parent_id": str(meta.get("parent_id") or ""),
                "matched_child_id": str(meta.get("matched_child_id") or ""),
                "source": self._get_doc_source(doc),
                "page": meta.get("page"),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "chapter": meta.get("chapter"),
                "section": meta.get("section"),
                "content_type": meta.get("content_type"),
                "bm25_rank": meta.get("bm25_rank"),
                "vector_rank": meta.get("vector_rank"),
                "rrf_score": meta.get("rrf_score"),
                "rerank_score": meta.get("rerank_score"),
            })
        trace.append({
            "stage": "finalize",
            "status": "success" if context_docs else ("empty" if retrieval_succeeded else "failed"),
            "profile": profile,
            "selected_chunks": selected_chunks,
            "duration_ms": int((time.time() - retrieve_start) * 1000),
        })
        logger.info(
            f"[RAG] retrieve_context 完成 mode={mode}, profile={profile}, docs={len(context_docs)}, context_len={len(context)}"
        )
        return context
