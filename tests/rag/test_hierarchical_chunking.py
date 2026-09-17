import json

from langchain_core.documents import Document

from rag.hierarchical_chunker import HierarchicalChunker
from rag.vector_store import VectorStoreService


CONFIG = {
    "separators": ["\n\n", "\n", "。", " ", ""],
    "courseware_chunking": {
        "parent_size": 2600,
        "parent_max_pages": 4,
        "child_size": 120,
        "child_overlap": 20,
    },
    "dense_document_chunking": {
        "parent_size": 240,
        "child_size": 100,
        "child_overlap": 15,
    },
}


def _slide(page: int, text: str) -> Document:
    return Document(
        page_content=f"第 {page} 页\n【文本内容】\n{text}",
        metadata={
            "source": "/tmp/ch01.pdf",
            "page": page,
            "total_pages": 5,
            "type": "pdf",
            "parser": "pymupdf",
        },
    )


def test_courseware_builds_page_parents_and_searchable_children():
    documents = [
        _slide(page, f"平台架构知识点{page}。" + "这是详细解释。" * 20)
        for page in range(1, 6)
    ]
    parents, children = HierarchicalChunker(CONFIG).build(
        documents,
        filename="大型平台分析与设计ch01.pdf",
        document_id="doc_abc",
    )

    assert len(parents) == 2
    assert parents[0].metadata["page_start"] == 1
    assert parents[0].metadata["page_end"] == 4
    assert parents[0].metadata["pages"] == "[1, 2, 3, 4]"
    assert children
    assert all(child.metadata["chunk_type"] == "child" for child in children)
    assert all(child.metadata["parent_id"] for child in children)
    assert all(child.metadata["document_id"] == "doc_abc" for child in children)
    assert all(child.metadata["source_filename"] == "大型平台分析与设计ch01.pdf" for child in children)
    assert all(child.metadata["chapter"] == "第1章" for child in children)
    assert all(isinstance(value, (str, int, float, bool)) for child in children for value in child.metadata.values())


def test_parent_expansion_deduplicates_multiple_child_hits(tmp_path):
    service = VectorStoreService.__new__(VectorStoreService)
    service.session_data_path = str(tmp_path)
    service._parent_store_path = str(tmp_path / ".parent_documents.json")
    parent = Document(
        id="par_a_0",
        page_content="完整父块内容",
        metadata={
            "parent_id": "par_a_0",
            "chunk_id": "par_a_0",
            "source_filename": "course.pdf",
            "chapter": "第一章",
        },
    )
    service._replace_file_parents("course.pdf", [parent])

    hits = [
        Document(page_content="子块一", metadata={"parent_id": "par_a_0", "chunk_id": "child_1", "rrf_score": 0.3}),
        Document(page_content="子块二", metadata={"parent_id": "par_a_0", "chunk_id": "child_2", "rrf_score": 0.2}),
    ]
    expanded = service.expand_parent_documents(hits)

    assert len(expanded) == 1
    assert expanded[0].page_content == "完整父块内容"
    assert expanded[0].metadata["matched_child_id"] == "child_1"
    assert expanded[0].metadata["rrf_score"] == 0.3
    assert expanded[0].metadata["expanded_from_child"] is True
    stored = json.loads((tmp_path / ".parent_documents.json").read_text(encoding="utf-8"))
    assert stored["schema_version"] == 2
