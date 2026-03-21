import os
import hashlib
import shutil
from typing import List

from fastapi import APIRouter, File, UploadFile, BackgroundTasks, Query, HTTPException
from rag.vector_store import VectorStoreService
from utils.session_context import current_session_id
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from agent.tools.agent_tools import _rag_cache

router = APIRouter()

MAX_FILES = 5
ALLOWED_SUFFIX = {".pdf", ".docx", ".txt", ".ppt", ".pptx", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _compute_bytes_md5(data: bytes) -> str:
    """计算内存中字节流的 MD5"""
    return hashlib.md5(data).hexdigest()


def _load_existing_md5s(md5_path: str) -> set:
    """从 md5_hex_store 文件读取已向量化的 MD5 集合"""
    if not os.path.exists(md5_path):
        return set()
    with open(md5_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def process_document_task(filenames: list, session_id: str = "default"):
    """
    后台处理任务：执行向量存入操作（支持多文件批量）
    """
    current_session_id.set(session_id)
    try:
        vs = VectorStoreService()
        vs.load_document()
        
        # 触发 RAG 缓存刷新，确保 BM25 索引同步更新
        if session_id in _rag_cache:
            logger.info(f"[后台任务] 正在刷新 session [{session_id}] 的 RAG 缓存...")
            _rag_cache[session_id].refresh()
            
        logger.info(f"[后台任务] 批次 {filenames} 已成功载入科目 [{session_id}] 知识库，并完成向量化！")
    except Exception as e:
        logger.error(f"[后台任务] 批次 {filenames} 知识库解析失败: {str(e)}")


@router.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    session_id: str = Query(default="default")
):
    """
    接收用户上传的文件（txt/pdf/docx），最多 5 个。
    在保存前先对内存字节做 MD5 去重，避免重复文件入库。
    """
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"每次最多上传 {MAX_FILES} 个文件，当前选择了 {len(files)} 个。")

    # 绑定会话上下文
    current_session_id.set(session_id)

    data_root = get_abs_path(chroma_conf['data_path'])
    session_data_dir = os.path.join(data_root, session_id)
    os.makedirs(session_data_dir, exist_ok=True)

    md5_store_path = os.path.join(session_data_dir, chroma_conf['md5_hex_store'])
    existing_md5s = _load_existing_md5s(md5_store_path)

    results = []
    new_files = []

    for file in files:
        # --- 校验后缀 ---
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_SUFFIX:
            results.append({
                "filename": file.filename,
                "status": "rejected",
                "reason": f"不支持的文件类型 {ext}，仅允许 .pdf / .docx / .txt"
            })
            continue

        # --- 读取内存，计算 MD5 ---
        content = await file.read()
        md5_hex = _compute_bytes_md5(content)

        if md5_hex in existing_md5s:
            results.append({
                "filename": file.filename,
                "status": "duplicate",
                "reason": "文件内容与已有知识库重复，已跳过"
            })
            logger.info(f"[上传去重] {file.filename} MD5={md5_hex} 已存在，跳过")
            continue

        # --- 保存文件 ---
        safe_name = os.path.basename(file.filename)
        file_path = os.path.join(session_data_dir, safe_name)
        with open(file_path, "wb") as buf:
            buf.write(content)

        new_files.append(safe_name)
        results.append({
            "filename": file.filename,
            "status": "accepted",
            "reason": "已接收，排队向量化中"
        })
        logger.info(f"[上传] {file.filename} 已保存至 {file_path}")

    # 只有存在新文件时才触发后台向量化
    if new_files:
        background_tasks.add_task(process_document_task, new_files, session_id)

    accepted = sum(1 for r in results if r["status"] == "accepted")
    duplicate = sum(1 for r in results if r["status"] == "duplicate")
    rejected = sum(1 for r in results if r["status"] == "rejected")

    return {
        "code": 200,
        "message": f"处理完成：{accepted} 个排队向量化，{duplicate} 个内容重复已跳过，{rejected} 个格式不支持。",
        "details": results
    }


# ==================== 样卷上传 ====================
import json

# 样卷存储路径
SAMPLE_PAPER_DIR = os.path.join(get_abs_path(chroma_conf['data_path']), "sample_papers")
os.makedirs(SAMPLE_PAPER_DIR, exist_ok=True)


def _parse_sample_paper_format(content: str) -> dict:
    """
    解析样卷格式
    """
    from model.factory import chat_model
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = """请分析以下试卷的格式特征，包括：

1. 题型分布（如：选择题20题、填空题10题、简答题5题）
2. 每种题型的分值
3. 题目编号风格
4. 答案格式
5. 解析格式（如有）
6. 整体排版风格

试卷内容：
{content}

请按以下JSON格式返回分析结果：
{{
    "quiz_types": ["选择题", "填空题", "判断题", "简答题"],
    "distribution": {{"选择题": 20, "填空题": 10, "简答题": 5}},
    "score_per_type": {{"选择题": 2, "填空题": 3, "简答题": 10}},
    "numbering_style": "1. 2. 3.",
    "answer_format": "答案：[X]",
    "analysis_format": "解析：[...]",
    "style_notes": "其他格式特点"
}}

只返回JSON格式。"""

    try:
        from langchain_core.prompts import PromptTemplate
        prompt_template = PromptTemplate.from_template(prompt)
        chain = prompt_template | chat_model | StrOutputParser()

        result = chain.invoke({"content": content[:3000]})
        format_info = json.loads(result)

        return {
            "success": True,
            "format": format_info
        }
    except Exception as e:
        logger.warning(f"[样卷解析] 失败: {e}")
        return {
            "success": False,
            "format": None,
            "error": str(e)
        }


@router.post("/sample/upload")
async def upload_sample_paper(
    file: UploadFile = File(...),
    session_id: str = Query(default="default")
):
    """
    上传样卷，学习试卷格式
    """
    # 读取文件内容
    content = await file.read()

    # 解析格式
    from utils.file_handler import document_loader
    import tempfile

    # 保存到临时文件进行解析
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 解析文档
        docs = document_loader(tmp_path)
        text_content = "\n\n".join([doc.page_content for doc in docs])

        # 解析格式
        format_result = _parse_sample_paper_format(text_content)

        # 保存样卷内容和格式
        sample_dir = os.path.join(SAMPLE_PAPER_DIR, session_id)
        os.makedirs(sample_dir, exist_ok=True)

        sample_file = os.path.join(sample_dir, "sample_paper.txt")
        format_file = os.path.join(sample_dir, "sample_format.json")

        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(text_content)

        with open(format_file, "w", encoding="utf-8") as f:
            json.dump(format_result, f, ensure_ascii=False, indent=2)

        logger.info(f"[样卷上传] session={session_id}, file={file.filename}")

        return {
            "code": 200,
            "message": "样卷上传成功，格式已学习",
            "data": {
                "filename": file.filename,
                "format": format_result.get("format", {})
            }
        }

    except Exception as e:
        logger.error(f"[样卷上传] 失败: {e}")
        return {
            "code": 500,
            "message": f"样卷上传失败: {str(e)}"
        }
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.get("/sample")
async def get_sample_paper(session_id: str = Query(default="default")):
    """
    获取当前科目的样卷内容和格式
    """
    sample_dir = os.path.join(SAMPLE_PAPER_DIR, session_id)
    sample_file = os.path.join(sample_dir, "sample_paper.txt")
    format_file = os.path.join(sample_dir, "sample_format.json")

    if not os.path.exists(sample_file):
        return {
            "code": 404,
            "message": "该科目尚未上传样卷"
        }

    try:
        # 读取格式
        with open(format_file, "r", encoding="utf-8") as f:
            format_info = json.load(f)

        # 读取内容（限制长度）
        with open(sample_file, "r", encoding="utf-8") as f:
            content = f.read()[:5000]  # 限制返回长度

        return {
            "code": 200,
            "data": {
                "has_sample": True,
                "format": format_info.get("format", {}),
                "content_preview": content
            }
        }
    except Exception as e:
        return {
            "code": 500,
            "message": f"读取样卷失败: {str(e)}"
        }


def get_sample_paper_context(session_id: str) -> str:
    """
    获取样卷格式上下文（供出题时使用）
    """
    sample_dir = os.path.join(SAMPLE_PAPER_DIR, session_id)
    format_file = os.path.join(sample_dir, "sample_format.json")

    if not os.path.exists(format_file):
        return ""  # 无样卷

    try:
        with open(format_file, "r", encoding="utf-8") as f:
            format_info = json.load(f)

        fmt = format_info.get("format", {})
        if not fmt:
            return ""

        # 构建格式描述
        context = "【样卷格式参考】\n"

        if fmt.get("distribution"):
            context += f"题型分布: {fmt['distribution']}\n"

        if fmt.get("score_per_type"):
            context += f"分值: {fmt['score_per_type']}\n"

        if fmt.get("numbering_style"):
            context += f"编号风格: {fmt['numbering_style']}\n"

        if fmt.get("answer_format"):
            context += f"答案格式: {fmt['answer_format']}\n"

        if fmt.get("analysis_format"):
            context += f"解析格式: {fmt['analysis_format']}\n"

        if fmt.get("style_notes"):
            context += f"格式特点: {fmt['style_notes']}\n"

        return context

    except Exception as e:
        logger.warning(f"[获取样卷格式] 失败: {e}")
        return ""


@router.delete("/sample")
async def delete_sample_paper(session_id: str = Query(default="default")):
    """
    删除样卷
    """
    sample_dir = os.path.join(SAMPLE_PAPER_DIR, session_id)

    if not os.path.exists(sample_dir):
        return {"code": 404, "message": "样卷不存在"}

    try:
        shutil.rmtree(sample_dir)
        return {"code": 200, "message": "样卷已删除"}
    except Exception as e:
        return {"code": 500, "message": f"删除失败: {str(e)}"}


@router.get("/list")
async def list_documents(session_id: str = Query(default="default")):
    """
    列出当前 Session 知识库中已完成向量化嵌入的文件。
    判定标准：文件的 MD5 已记录在 md5_hex_store 中，即真正入库完毕。
    """
    data_root = get_abs_path(chroma_conf['data_path'])
    session_data_dir = os.path.join(data_root, session_id)
    md5_store_name = chroma_conf.get('md5_hex_store', '.md5_hex_store')
    md5_store_path = os.path.join(session_data_dir, md5_store_name)

    if not os.path.isdir(session_data_dir):
        return {"code": 200, "files": [], "total": 0}

    # 读取已完成向量化的 MD5 集合
    embedded_md5s = _load_existing_md5s(md5_store_path)

    files = []
    for fname in os.listdir(session_data_dir):
        if fname == md5_store_name or fname.startswith('.'):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ALLOWED_SUFFIX:
            continue

        fpath = os.path.join(session_data_dir, fname)

        # 只显示 MD5 已记录的文件（向量化完成）
        try:
            with open(fpath, "rb") as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
        except Exception:
            continue

        if file_md5 not in embedded_md5s:
            continue  # 尚未完成向量化，跳过

        stat = os.stat(fpath)
        files.append({
            "filename": fname,
            "size": stat.st_size,
            "modified_at": int(stat.st_mtime),
        })

    # 按修改时间倒序（最新在前）
    files.sort(key=lambda x: x["modified_at"], reverse=True)

    return {"code": 200, "files": files, "total": len(files)}
