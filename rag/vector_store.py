import os.path
import asyncio
import hashlib
import json
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, ppt_loader, image_loader, word_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from utils.session_context import current_session_id
import shutil


class VectorStoreService():
    # 图片并发控制：避免触发 API rate limit
    MAX_CONCURRENT_IMAGES = 5

    def __init__(self):
        self.session_id = current_session_id.get()
        self._image_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_IMAGES)
        _md5 = hashlib.md5(self.session_id.encode("utf-8")).hexdigest()[:16]
        safe_col_name = f"col_{_md5}"

        self.mmr_enabled = chroma_conf.get('mmr_enabled', False)
        self.mmr_lambda = chroma_conf.get('mmr_lambda', 0.7)

        try:
            self.vector_store = Chroma(
                collection_name=safe_col_name,
                embedding_function=embed_model,
                persist_directory=chroma_conf['persist_directory'],
            )
            self.spliter = RecursiveCharacterTextSplitter(
                chunk_size=chroma_conf['chunk_size'],
                chunk_overlap=chroma_conf['chunk_overlap'],
                separators=chroma_conf['separators'],
                length_function=len,
            )
            self.session_data_path = os.path.join(get_abs_path(chroma_conf['data_path']), self.session_id)
            self.session_md5_path = os.path.join(self.session_data_path, chroma_conf['md5_hex_store'])
            self._file_vector_map_path = os.path.join(self.session_data_path, '.file_vector_map.json')
        except Exception as e:
            logger.error(f"初始化向量库失败: {e}")
            raise

    async def destroy_knowledge_base(self):
        """
        彻底销毁当前 Session 的知识库合集。包括向量数据合集和本地物理文件。
        """
        try:
            self.vector_store.delete_collection()
            logger.info(f"成功销毁向量库 Collection: {self.session_id}")
        except Exception as e:
            logger.error(f"销毁向量库集合失败或原本不存在: {e}")
            
        if os.path.exists(self.session_data_path):
            try:
                shutil.rmtree(self.session_data_path)
                logger.info(f"成功删除本地知识库文件目录: {self.session_data_path}")
            except Exception as e:
                logger.error(f"删除物理目录失败: {e}")

    def get_retriever(self, k: int = None, mmr: bool = None, lambda_mult: float = None):
        """
        获取检索器

        Args:
            k: 召回数量，默认从配置读取
            mmr: 是否启用 MMR（默认跟随配置）
            lambda_mult: MMR 参数（默认跟随配置）
        """
        use_mmr = mmr if mmr is not None else self.mmr_enabled
        lambda_val = lambda_mult if lambda_mult is not None else self.mmr_lambda
        search_kwargs = {"k": k or chroma_conf.get('retrieve_top_k', 8)}

        if use_mmr:
            return self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": search_kwargs["k"],
                    "fetch_k": search_kwargs["k"] * 3,
                    "lambda_mult": lambda_val
                }
            )

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def _load_file_vector_map(self) -> Dict[str, List[str]]:
        if not os.path.exists(self._file_vector_map_path):
            return {}
        try:
            with open(self._file_vector_map_path, "r", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            return {}

    def _save_file_vector_map(self, mapping: Dict[str, List[str]]):
        with open(self._file_vector_map_path, "w", encoding="utf-8") as f:
            _json.dump(mapping, f, ensure_ascii=False)

    async def delete_file_vectors(self, filename: str) -> bool:
        mapping = self._load_file_vector_map()
        if filename not in mapping:
            logger.warning(f"[删除向量] 文件 {filename} 不在映射表中，可能未向量化或已删除")
            return False

        doc_ids = mapping[filename]
        try:
            self.vector_store.delete(ids=doc_ids)
            del mapping[filename]
            self._save_file_vector_map(mapping)
            logger.info(f"[删除向量] 成功删除文件 {filename} 的 {len(doc_ids)} 个向量")
            return True
        except Exception as e:
            logger.error(f"[删除向量] 删除文件 {filename} 失败: {e}")
            return False

    async def load_document(self, target_filenames: Optional[List[str]] = None):
        """
        从数据文件夹读取数据转为向量存入向量库。
        图片描述通过 asyncio.gather 并发调用视觉模型，避免串行阻塞。
        """
        import json as _json
        result_map = {}
        target_set = set(target_filenames or [])

        def chunk_md5_hex(md5_for_check: str):
            if not os.path.exists(self.session_md5_path):
                return False
            with open(self.session_md5_path, "r", encoding="utf-8") as f:
                return any(line.strip() == md5_for_check for line in f)

        def save_md5_hex(md5_for_check: str):
            with open(self.session_md5_path, "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def update_file_vector_map(fname: str, doc_ids: List[str]):
            mapping = self._load_file_vector_map()
            mapping[fname] = doc_ids
            self._save_file_vector_map(mapping)

        async def get_file_document(read_path: str):
            if read_path.endswith(".txt"):
                return txt_loader(read_path)
            elif read_path.endswith(".pdf"):
                return pdf_loader(read_path)
            elif read_path.endswith(".docx") or read_path.endswith(".doc"):
                return word_loader(read_path)
            elif read_path.endswith(".pptx") or read_path.endswith(".ppt"):
                return ppt_loader(read_path)
            elif read_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
                return image_loader(read_path)
            return []

        async def _enrich_images(documents: list) -> list:
            """
            对 ppt_loader 返回的 Document 并发处理图片描述。
            图片字节暂存在 metadata['_image_blobs']，处理后追加到 page_content 并清除。
            """
            from utils.file_handler import _summarize_image

            async def _describe_one(blob: bytes, context: str, label: str) -> str:
                # 用 Semaphore 控制并发，避免触发 API rate limit
                async with self._image_semaphore:
                    loop = asyncio.get_event_loop()
                    # _summarize_image 是同步调用，放到线程池避免阻塞事件循环
                    desc = await loop.run_in_executor(None, _summarize_image, blob, context)
                    return label, desc

            tasks = []
            doc_indices = []
            for doc_idx, doc in enumerate(documents):
                blobs = doc.metadata.pop("_image_blobs", [])
                context = doc.metadata.pop("_slide_context", "")
                for img_idx, (_, blob) in enumerate(blobs, 1):
                    label = f"【图片 {img_idx} 描述】"
                    tasks.append(_describe_one(blob, context, label))
                    doc_indices.append(doc_idx)

            if not tasks:
                return documents

            results = await asyncio.gather(*tasks, return_exceptions=True)
            images_by_doc: dict[int, list[str]] = {}
            for doc_idx, result in zip(doc_indices, results):
                if isinstance(result, Exception):
                    logger.warning(f"[图片并发处理] 失败: {result}")
                    continue
                label, desc = result
                if desc and desc != "无有效信息":
                    images_by_doc.setdefault(doc_idx, []).append(f"\n{label}\n{desc}\n")

            for doc_idx, descs in images_by_doc.items():
                documents[doc_idx].page_content += "\n【图片内容】" + "".join(descs)

            logger.info(f"[图片并发处理] 共处理 {len(tasks)} 张图片")
            return documents

        # 确保会话目录存在
        os.makedirs(self.session_data_path, exist_ok=True)

        allowed_files_path = listdir_with_allowed_type(
            self.session_data_path,
            tuple(chroma_conf['allow_knowledge_file_type'])
        )

        for path in allowed_files_path:
            fname = os.path.basename(path)
            if target_set and fname not in target_set:
                continue
            try:
                md5_hex = get_file_md5_hex(path)
                if chunk_md5_hex(md5_hex):
                    logger.info(f"[加载知识库]{path}内容已存在，跳过")
                    result_map[fname] = {"status": "completed", "detail": "内容已存在，跳过重复向量化"}
                    continue

                documents = await get_file_document(path)
                if not documents:
                    logger.info(f"[加载知识库]{path}文件内没有有效文本，跳过")
                    result_map[fname] = {"status": "failed", "detail": "文件内没有可解析文本内容"}
                    continue

                # 并发处理图片（PPT/PDF 中的图片字节暂存在 metadata）
                documents = await _enrich_images(documents)

                split_document = self.spliter.split_documents(documents)
                if not split_document:
                    logger.info(f"[加载知识库]{path}分片内无有效内容，跳过")
                    result_map[fname] = {"status": "failed", "detail": "文档分片后无有效内容"}
                    continue

                # 为每个 chunk 生成确定性ID（包含文件名），便于后续追踪删除
                for idx, doc in enumerate(split_document):
                    doc.id = f"vec_{hashlib.md5(fname.encode()).hexdigest()[:8]}_{idx}"
                    doc.metadata["source_filename"] = fname

                self.vector_store.add_documents(split_document)
                # 记录文件→向量ID映射
                doc_ids = [doc.id for doc in split_document]
                update_file_vector_map(fname, doc_ids)
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path}加载成功，共 {len(split_document)} 个chunk")
                result_map[fname] = {"status": "completed", "detail": f"向量化完成，共{len(split_document)}个片段"}
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败: {e}", exc_info=True)
                result_map[fname] = {"status": "failed", "detail": f"解析失败: {str(e)}"}

        return result_map


if __name__ == '__main__':
    import asyncio
    async def test():
        store = VectorStoreService()
        await store.load_document()
        retriever = store.get_retriever()
        res = retriever.invoke("测试")
        for doc in res:
            print(doc.page_content)
            print("-" * 20)
    asyncio.run(test())
