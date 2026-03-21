from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_community.retrievers import BM25Retriever
from model.factory import chat_model
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any
from utils.logger_handler import logger
import asyncio

# Cross-Encoder 已禁用（网络问题），使用纯 RRF 检索


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

        # ============== 检索配置 (从 chroma.yml 读取) ==============
        self.rerank_top_k = 15    # 混合检索召回数量（增加以获得更多上下文）
        self.final_top_k = 8      # 最终返回数量（增加给 Agent 更多参考）
        self.rrf_k = 60          # RRF 公式中的 k 值
        self.mmr_enabled = False  # MMR 多样性检索
        self.mmr_lambda = 0.5     # MMR 平衡参数

        logger.info("[RAG] 初始化混合检索系统 (BM25 + 向量 RRF)")

        # ============== 创建混合检索器 ==============
        self.refresh()
    def refresh(self):
        """
        刷新检索器状态。当知识库（Chroma）中文档发生变化时，调用此方法重新构建 BM25 索引。
        """
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
            "你是一个期末考试复习助手。学生的问题通常很短或带有代词。请将下面的问题补充完整并改写为更适合在知识库(课件)中检索的关键词串，词与词之间用空格隔开。如果问题本身已经很明确，只需提取出核心名词即可。不超过50个字。\n原问题: {question}"
        )
        self.rewrite_chain = rewrite_prompt | chat_model | StrOutputParser()

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

    def _rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """
        使用 LLM 对检索结果重排序
        """
        if not docs or len(docs) <= 1:
            return docs

        try:
            # 使用 LLM 对文档进行相关性打分和排序
            from langchain_core.output_parsers import StrOutputParser

            # 构建 prompt
            doc_texts = "\n\n".join([
                f"【文档{i+1}】{doc.page_content[:500]}"
                for i, doc in enumerate(docs)
            ])

            rerank_prompt = f"""请根据用户问题，对以下文档进行相关性排序。

用户问题：{query}

文档列表：
{doc_texts}

请按相关性从高到低排序，返回文档编号列表（格式：1, 2, 3... 只返回编号列表，不需要其他内容）。"""

            # 调用 LLM
            response = chat_model.invoke(rerank_prompt)
            ranking = response.content.strip()

            # 解析排序结果
            try:
                # 尝试解析返回的编号
                ranks = [int(x.strip()) for x in ranking.split(",") if x.strip().isdigit()]
                if ranks:
                    # 按排名顺序重排
                    reranked = []
                    for rank in ranks:
                        if 0 < rank <= len(docs):
                            reranked.append(docs[rank - 1])
                    # 添加未收录的文档
                    for doc in docs:
                        if doc not in reranked:
                            reranked.append(doc)
                    print(f"[LLM Rerank] 原始 {len(docs)} 个 -> 重排后 {len(reranked)} 个")
                    return reranked[:self.final_top_k]
            except:
                pass

            # 如果解析失败，直接返回原始结果
            return docs[:self.final_top_k]

        except Exception as e:
            logger.warning(f"[LLM Rerank] 重排序失败: {e}")
            return docs[:self.final_top_k]

    async def rag_summarize(self, query: str) -> str:
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
        if context_docs and len(context_docs) > 1:
            context_docs = self._rerank(expanded_query, context_docs)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            # 将文档的元数据合并上，这会在最后出处呈现上发挥大作用
            context += f"[参考资料{counter}]:参考资料:{doc.page_content}|参考元数据:{doc.metadata}\n"

        return await self.chain.ainvoke(
            {"input": query,
             "context": context,
             }
        )

if __name__ == "__main__":
    rag_service = RagSummarizeService()
    print(rag_service.rag_summarize("帮我复习一下第三章的核心考点是什么"))