import pytest
from langchain_core.documents import Document

from rag.rag_service import RRFRetriever


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    async def ainvoke(self, _query):
        return self.docs


@pytest.mark.asyncio
async def test_rrf_documents_keep_rank_and_score_for_trace():
    shared = Document(page_content="共享片段", metadata={"source": "os.pdf", "page": 1})
    bm25_only = Document(page_content="关键词片段", metadata={"source": "os.pdf", "page": 2})
    vector_only = Document(page_content="语义片段", metadata={"source": "thread.pdf", "page": 3})
    retriever = RRFRetriever(
        bm25_retriever=FakeRetriever([shared, bm25_only]),
        vector_retriever=FakeRetriever([vector_only, shared]),
        rrf_k=30,
        result_limit=3,
        bm25_weight=0.4,
        vector_weight=0.6,
    )

    docs = await retriever.ainvoke("进程线程")
    shared_result = next(doc for doc in docs if doc.page_content == "共享片段")
    assert shared_result.metadata["bm25_rank"] == 1
    assert shared_result.metadata["vector_rank"] == 2
    assert shared_result.metadata["rrf_score"] > 0
    assert shared_result.metadata["retrieval_chunk_id"]
    assert "rrf_score" not in shared.metadata
