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
import asyncio
import sqlite3
import os
import json
import hashlib
import math
import time

# Cross-Encoder 已禁用（网络问题），使用纯 RRF 检索


class SemanticCache:
    """
    RAG 语义缓存：用 embedding 相似度匹配相同/近似问题，直接返回缓存的 LLM 回复。
    缓存失效：文件上传/删除时按 session_id 清空。
    """

    def __init__(self, similarity_threshold: float = 0.92):
        self.sim_threshold = similarity_threshold
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
        try:
            loop = asyncio.get_event_loop()
            emb = await loop.run_in_executor(None, lambda: embed_model.embed_query(query))
        except Exception as e:
            logger.warning(f"[语义缓存] embedding 失败: {e}")
            return None

        q_hash = hashlib.md5(query.encode()).hexdigest()
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT query_hash, embedding, response FROM cache WHERE session_id = ?",
            (session_id,)
        ).fetchall()
        conn.close()

        for row in rows:
            cached_hash, cached_emb_str, cached_resp = row
            if cached_hash == q_hash:
                logger.info("[语义缓存] 精确命中")
                return cached_resp
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
    def __init__(self, bm25_retriever=None, vector_retriever=None, rrf_k=60, rerank_top_k=8):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.rrf_k = rrf_k
        self.rerank_top_k = rerank_top_k

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

    def _rrf_fuse(self, docs1: List[Document], docs2: List[Document]) -> List[Document]:
        """RRF 融合算法"""
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # 对第一个检索结果打分
        for rank, doc in enumerate(docs1, 1):
            key = doc.page_content[:100]  # 用内容前100字符作为唯一标识
            doc_scores[key] = doc_scores.get(key, 0) + 1 / (self.rrf_k + rank)
            doc_map[key] = doc

        # 对第二个检索结果打分
        for rank, doc in enumerate(docs2, 1):
            key = doc.page_content[:100]
            doc_scores[key] = doc_scores.get(key, 0) + 1 / (self.rrf_k + rank)
            doc_map[key] = doc

        # 按分数排序
        sorted_keys = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in sorted_keys]

# Cross-Encoder disabled, using RRF hybrid search

class RagSummarizeService:
    def __init__(self):
        self.vector_store_service = VectorStoreService()
        self.vector_retriever = self.vector_store_service.get_retriever()
        self.bm25_retriever = None
        self.semantic_cache = SemanticCache()

        # ============== 检索配置（从 chroma.yml 读取）==============
        self.rerank_top_k = chroma_conf.get('retrieve_top_k', 10)
        self.final_top_k = chroma_conf.get('rerank_final_k', 8)
        self.rrf_k = chroma_conf.get('rrf_k', 30)
        self.mmr_enabled = chroma_conf.get('mmr_enabled', False)
        self.mmr_lambda = chroma_conf.get('mmr_lambda', 0.5)

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
        # 提取当前库中所有的文档供BM25建立本地索引
        all_docs = self.vector_store_service.vector_store.get()
        if all_docs and isinstance(all_docs, dict) and "documents" in all_docs and len(all_docs["documents"]) > 0:
            bm25_docs = [Document(page_content=text, metadata=meta) for text, meta in zip(all_docs["documents"], all_docs["metadatas"])]
            self.bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            self.bm25_retriever.k = self.rerank_top_k

            # 使用 RRF (Reciprocal Rank Fusion) 算法融合
            # RRF 公式: score(d) = Σ 1/(k + rank(d))
            # 不需要手动设置权重，自动融合多个检索结果
            self.retriever = RRFRetriever(
                bm25_retriever=self.bm25_retriever,
                vector_retriever=self.vector_retriever,
                rrf_k=self.rrf_k,
                rerank_top_k=self.rerank_top_k
            )
            print(f"[RRF] 已启用 Reciprocal Rank Fusion 混合检索 (k={self.rrf_k})")
        else:
            # 如果库是空的（还未上传任何文件），退回到普通的向量检索防崩
            self.retriever = RRFRetriever(
                bm25_retriever=None,
                vector_retriever=self.vector_retriever,
                rrf_k=self.rrf_k,
                rerank_top_k=self.rerank_top_k
            )
            print(f"[RRF] 仅使用向量检索（知识库为空）")

        # --- 步骤二：准备 Query Rewrite 的 Chain ---
        rewrite_prompt = PromptTemplate.from_template(
            "你是一个期末考试复习助手。学生的问题通常很短或带有代词。请将下面的问题补充完整并改写为更适合在知识库(课件)中检索的关键词串，词与词之间用空格隔开。如果问题本身已经很明确，只需提取出核心名词即可。\n原问题: {question}"
        )
        self.rewrite_chain = rewrite_prompt | light_chat_model | StrOutputParser()

        # --- 步骤三：打包装配最终总结的 Chain ---
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    async def retriever_docs(self, query: str) -> list[Document]:
        return await self.retriever.ainvoke(query)

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

    async def retrieve_context(self, query: str) -> str:
        """
        只做检索，返回格式化后的原始课件片段字符串，不调用最终 LLM。
        供 Supervisor RAG SubAgent 使用，避免双重 LLM 调用。
        """
        try:
            expanded_query = await asyncio.wait_for(
                self.rewrite_chain.ainvoke({"question": query}),
                timeout=3.0,
            )
        except Exception:
            expanded_query = query

        try:
            context_docs = await asyncio.wait_for(
                self.retriever_docs(expanded_query),
                timeout=8.0,
            )
        except Exception as e:
            logger.warning(f"[RAG] retrieve_context 超时/失败，回退空上下文: {e}")
            context_docs = []
        # 跳过 LLM rerank（rerank 每次都超时 3 秒，10 个 topic 浪费 ~30 秒，且效果不明显）
        # RRF 检索结果已经足够好

        context = ""
        for i, doc in enumerate(context_docs, 1):
            context += f"[参考资料{i}]:参考资料:{doc.page_content}|参考元数据:{doc.metadata}\n"
        return context

    async def rag_summarize(self, query: str) -> str:
        session_id = current_session_id.get()

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
