"""Structure-aware parent/child chunking for the courseware knowledge base."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


_CHAPTER_RE = re.compile(r"(?:^|\n)\s*(第[一二三四五六七八九十百0-9]+章[^\n]{0,60})")
_SECTION_RE = re.compile(
    r"(?:^|\n)\s*((?:第[一二三四五六七八九十百0-9]+节|\d+(?:\.\d+){1,3})[^\n]{0,70})"
)
_MARKDOWN_HEADING_RE = re.compile(r"(?:^|\n)\s*#{1,4}\s+([^\n]{1,80})")


def _primitive_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata only accepts scalar values; preserve collections as JSON."""
    result: Dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if key.startswith("_"):
            continue
        if value is None:
            result[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            result[key] = value
        else:
            result[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return result


def _first_heading(text: str) -> str:
    markdown = _MARKDOWN_HEADING_RE.search(text or "")
    if markdown:
        return markdown.group(1).strip()
    ignored = ("=", "第 ", "【文本内容】", "【页眉】", "【页脚】")
    for raw in str(text or "").splitlines():
        line = raw.strip().strip("#").strip()
        if not line or line.startswith(ignored) or len(line) > 90:
            continue
        if re.fullmatch(r"第\s*\d+\s*页.*", line):
            continue
        return line
    return ""


def _chapter_and_section(text: str, fallback: str = "") -> Tuple[str, str]:
    chapter_match = _CHAPTER_RE.search(text or "")
    section_match = _SECTION_RE.search(text or "")
    chapter = chapter_match.group(1).strip() if chapter_match else ""
    section = section_match.group(1).strip() if section_match else _first_heading(text)
    if not chapter:
        file_chapter = re.search(r"(?:ch|chapter)[-_ ]?0*(\d+)", fallback, flags=re.IGNORECASE)
        if file_chapter:
            chapter = f"第{file_chapter.group(1)}章"
    return chapter, section


def _content_type(text: str) -> str:
    body = str(text or "")
    if "【表格内容】" in body or re.search(r"\|[^\n]+\|", body):
        return "table"
    if "【公式】" in body:
        return "formula"
    if re.search(r"案例|示例|例如|例题", body):
        return "example"
    if re.search(r"定义|概念|是指|称为", body):
        return "definition"
    if re.search(r"```|class\s+\w+|def\s+\w+", body):
        return "code"
    return "text"


@dataclass(frozen=True)
class ChunkProfile:
    name: str
    parent_size: int
    child_size: int
    child_overlap: int
    courseware_parent_pages: int = 4


class HierarchicalChunker:
    """Build parent context blocks and fine-grained searchable child blocks."""

    def __init__(self, config: Dict[str, Any]):
        courseware = config.get("courseware_chunking", {}) or {}
        dense = config.get("dense_document_chunking", {}) or {}
        self.courseware = ChunkProfile(
            name="courseware",
            parent_size=int(courseware.get("parent_size", 2600)),
            child_size=int(courseware.get("child_size", 700)),
            child_overlap=int(courseware.get("child_overlap", 100)),
            courseware_parent_pages=int(courseware.get("parent_max_pages", 4)),
        )
        self.dense = ChunkProfile(
            name="dense_document",
            parent_size=int(dense.get("parent_size", 2400)),
            child_size=int(dense.get("child_size", 1000)),
            child_overlap=int(dense.get("child_overlap", 150)),
        )
        self.separators = list(config.get("separators") or ["\n\n", "\n", "。", " ", ""])

    @staticmethod
    def select_profile(filename: str, documents: Iterable[Document]) -> str:
        ext = os.path.splitext(filename)[1].lower()
        docs = list(documents)
        if re.search(r"题库|试题|练习题|question|quiz|exam", filename, flags=re.IGNORECASE):
            return "question_bank"
        if ext in {".ppt", ".pptx"}:
            return "courseware"
        if ext == ".pdf" and (len(docs) >= 3 or any("page" in (d.metadata or {}) for d in docs)):
            return "courseware"
        return "dense_document"

    def build(
        self,
        documents: List[Document],
        filename: str,
        document_id: str,
    ) -> Tuple[List[Document], List[Document]]:
        profile_name = self.select_profile(filename, documents)
        profile = self.courseware if profile_name == "courseware" else self.dense
        parents = self._build_parents(documents, filename, document_id, profile, profile_name)
        children = self._build_children(parents, filename, document_id, profile)
        return parents, children

    def _build_parents(
        self,
        documents: List[Document],
        filename: str,
        document_id: str,
        profile: ChunkProfile,
        profile_name: str,
    ) -> List[Document]:
        if profile_name == "courseware" and len(documents) > 1:
            groups: List[List[Document]] = []
            current: List[Document] = []
            for doc in documents:
                text = str(doc.page_content or "").strip()
                strong_heading = bool(_CHAPTER_RE.search(text) or _SECTION_RE.search(text))
                if current and (len(current) >= profile.courseware_parent_pages or strong_heading):
                    groups.append(current)
                    current = []
                if text:
                    current.append(doc)
            if current:
                groups.append(current)
            raw_parents = [self._merge_documents(group) for group in groups]
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=profile.parent_size,
                chunk_overlap=0,
                separators=["\n# ", "\n## ", "\n### "] + self.separators,
                length_function=len,
            )
            raw_parents = splitter.split_documents(documents)

        file_hash = hashlib.md5(filename.encode("utf-8")).hexdigest()[:8]
        parents: List[Document] = []
        for index, doc in enumerate(raw_parents):
            text = str(doc.page_content or "").strip()
            if not text:
                continue
            parent_id = f"par_{file_hash}_{index}"
            metadata = _primitive_metadata(dict(doc.metadata or {}))
            chapter, section = _chapter_and_section(text, filename)
            metadata.update(
                {
                    "document_id": document_id,
                    "parent_id": parent_id,
                    "chunk_id": parent_id,
                    "chunk_type": "parent",
                    "chunk_profile": profile.name,
                    "source_filename": filename,
                    "chapter": chapter,
                    "section": section,
                    "content_type": _content_type(text),
                    "metadata_schema_version": 2,
                }
            )
            parents.append(Document(id=parent_id, page_content=text, metadata=metadata))
        return parents

    @staticmethod
    def _merge_documents(documents: List[Document]) -> Document:
        contents = [str(doc.page_content or "").strip() for doc in documents if str(doc.page_content or "").strip()]
        first_meta = dict((documents[0].metadata if documents else {}) or {})
        pages = [int(doc.metadata.get("page")) for doc in documents if str((doc.metadata or {}).get("page", "")).isdigit()]
        if pages:
            first_meta["page"] = min(pages)
            first_meta["page_start"] = min(pages)
            first_meta["page_end"] = max(pages)
            first_meta["pages"] = pages
        parsers = sorted({str((doc.metadata or {}).get("parser", "")) for doc in documents if (doc.metadata or {}).get("parser")})
        if parsers:
            first_meta["parser"] = ",".join(parsers)
        return Document(page_content="\n\n".join(contents), metadata=first_meta)

    def _build_children(
        self,
        parents: List[Document],
        filename: str,
        document_id: str,
        profile: ChunkProfile,
    ) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=profile.child_size,
            chunk_overlap=profile.child_overlap,
            separators=self.separators,
            length_function=len,
        )
        children: List[Document] = []
        file_hash = hashlib.md5(filename.encode("utf-8")).hexdigest()[:8]
        for parent_index, parent in enumerate(parents):
            pieces = splitter.split_text(parent.page_content)
            for child_index, text in enumerate(pieces):
                clean = str(text or "").strip()
                if not clean:
                    continue
                child_id = f"vec_{file_hash}_{parent_index}_{child_index}"
                metadata = _primitive_metadata(dict(parent.metadata or {}))
                metadata.update(
                    {
                        "document_id": document_id,
                        "parent_id": str(parent.id or metadata.get("parent_id") or ""),
                        "chunk_id": child_id,
                        "chunk_type": "child",
                        "child_index": child_index,
                        "source_filename": filename,
                        "content_type": _content_type(clean),
                        "metadata_schema_version": 2,
                    }
                )
                children.append(Document(id=child_id, page_content=clean, metadata=metadata))
        return children
