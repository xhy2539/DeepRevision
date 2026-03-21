import os.path
import hashlib

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
    def __init__(self):
        self.session_id = current_session_id.get()
        _md5 = hashlib.md5(self.session_id.encode("utf-8")).hexdigest()[:16]
        safe_col_name = f"col_{_md5}"
        
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

    def get_retriever(self, k: int = None, mmr: bool = False, lambda_mult: float = 0.5):
        """
        获取检索器

        Args:
            k: 召回数量，默认从配置读取
            mmr: 是否启用 MMR 多样性检索
            lambda_mult: MMR 参数，0=最大多样性，1=最大相关性
        """
        search_kwargs = {"k": k or chroma_conf.get('k', 8)}

        if mmr:
            # MMR 检索：平衡相关性和多样性
            return self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": search_kwargs["k"],
                    "fetch_k": search_kwargs["k"] * 3,  # 获取更多候选
                    "lambda_mult": lambda_mult
                }
            )

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    async def load_document(self):
        """
        从数据文件夹读取数据转为向量存入向量库
        """
        def chunk_md5_hex(md5_for_check: str):
            if not os.path.exists(self.session_md5_path):
                return False
            with open(self.session_md5_path, "r", encoding="utf-8") as f:
                return any(line.strip() == md5_for_check for line in f)

        def save_md5_hex(md5_for_check: str):
            with open(self.session_md5_path, "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        async def get_file_document(read_path: str):
            # 使用优化后的文档加载器，支持 PDF/PPT/图片/文档等所有格式
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

        # 确保会话目录存在
        os.makedirs(self.session_data_path, exist_ok=True)

        allowed_files_path = listdir_with_allowed_type(
            self.session_data_path,
            tuple(chroma_conf['allow_knowledge_file_type'])
        )

        for path in allowed_files_path:
            try:
                md5_hex = get_file_md5_hex(path)
                if chunk_md5_hex(md5_hex):
                    logger.info(f"[加载知识库]{path}内容已存在，跳过")
                    continue

                documents = await get_file_document(path)
                if not documents:
                    logger.info(f"[加载知识库]{path}文件内没有有效文本，跳过")
                    continue

                split_document = self.spliter.split_documents(documents)
                if not split_document:
                    logger.info(f"[加载知识库]{path}分片内无有效内容，跳过")
                    continue

                # Chroma add_documents is typically sync in langchain-chroma 0.1.x
                # but we wrap the call for future-proofing or if it supports async
                self.vector_store.add_documents(split_document)
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path}加载成功")
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败: {e}", exc_info=True)


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