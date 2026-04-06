import os
import re
import json
import hashlib
import shutil
import time
from typing import List, Dict, Any

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

# session_id 白名单校验（防止路径遍历，允许中文）
_VALID_SESSION_ID = re.compile(r'^[\u4e00-\u9fa5a-zA-Z0-9_\-]{1,64}$')


def _validate_session_id(session_id: str) -> str:
    """校验 session_id，防止路径遍历攻击"""
    if not _VALID_SESSION_ID.match(session_id):
        raise HTTPException(
            status_code=400,
            detail="session_id 格式非法，只允许字母、数字、中文、下划线和连字符（最长64位）",
        )
    return session_id


def _compute_bytes_md5(data: bytes) -> str:
    """计算内存中字节流的 MD5"""
    return hashlib.md5(data).hexdigest()


def _load_existing_md5s(md5_path: str) -> set:
    """从 md5_hex_store 文件读取已向量化的 MD5 集合"""
    if not os.path.exists(md5_path):
        return set()
    with open(md5_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _remove_md5(md5_path: str, target_md5: str):
    if not os.path.exists(md5_path):
        return
    try:
        with open(md5_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        lines = [h for h in lines if h != target_md5]
        with open(md5_path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
    except Exception as e:
        logger.warning(f"[上传去重] 移除旧MD5失败: {e}")


def _load_file_vector_map(session_data_dir: str) -> Dict[str, List[str]]:
    map_path = os.path.join(session_data_dir, ".file_vector_map.json")
    if not os.path.exists(map_path):
        return {}
    try:
        with open(map_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_ingest_status(status_path: str) -> Dict[str, Any]:
    if not os.path.exists(status_path):
        return {}
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_ingest_status(status_path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _update_ingest_status(session_id: str, updates: Dict[str, Dict[str, Any]]):
    data_root = get_abs_path(chroma_conf['data_path'])
    session_data_dir = os.path.join(data_root, session_id)
    status_path = os.path.join(session_data_dir, "ingest_status.json")
    status_data = _load_ingest_status(status_path)
    now = int(time.time())
    for filename, patch in updates.items():
        current = status_data.get(filename, {})
        current.update(patch)
        current["updated_at"] = now
        status_data[filename] = current
    _save_ingest_status(status_path, status_data)


def process_document_task(filenames: list, session_id: str = "default"):
    """
    后台处理任务：执行向量存入操作（支持多文件批量）。
    FastAPI BackgroundTasks 对同步函数在线程池中运行，
    故用 asyncio.run() 驱动内部的异步 load_document（fix：之前未 await 导致向量化从不执行）。
    """
    import asyncio as _asyncio

    async def _run():
        current_session_id.set(session_id)
        vs = VectorStoreService()
        # 标记为处理开始
        _update_ingest_status(
            session_id,
            {name: {"status": "processing", "detail": "正在解析并向量化"} for name in filenames},
        )

        result_map = await vs.load_document(target_filenames=filenames)

        status_patch = {}
        for name in filenames:
            info = result_map.get(name, {"status": "failed", "detail": "后台任务未返回该文件结果"})
            status_patch[name] = {
                "status": info.get("status", "failed"),
                "detail": info.get("detail", ""),
            }
        _update_ingest_status(session_id, status_patch)

        # 触发 RAG 缓存刷新，确保 BM25 索引同步更新
        # fix: 即使 _rag_cache 中没有该 session，也需要刷新或创建新实例
        from agent.tools.agent_tools import _rag_cache as rag_cache_ref
        if session_id in rag_cache_ref:
            logger.info(f"[后台任务] 刷新已有 session [{session_id}] 的 RAG 缓存...")
            rag_cache_ref[session_id].refresh()
        else:
            # 如果缓存中没有，创建新实例（确保 BM25 索引正确加载）
            logger.info(f"[后台任务] 为 session [{session_id}] 创建新的 RAG 实例...")
            from rag.rag_service import RagSummarizeService
            rag_cache_ref[session_id] = RagSummarizeService()

        logger.info(f"[后台任务] 批次 {filenames} 已成功载入科目 [{session_id}] 知识库，并完成向量化！")

    try:
        _asyncio.run(_run())
    except Exception as e:
        logger.error(f"[后台任务] 批次 {filenames} 知识库解析失败: {str(e)}")
        _update_ingest_status(
            session_id,
            {name: {"status": "failed", "detail": f"后台任务异常: {str(e)}"} for name in filenames},
        )


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
    logger.info(f"[上传] 收到请求，session_id={session_id}，files={[f.filename for f in files]}")
    logger.info(f"[上传] 调试 - 接收到的 session_id: '{session_id}'")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"每次最多上传 {MAX_FILES} 个文件，当前选择了 {len(files)} 个。")

    _validate_session_id(session_id)  # fix #2
    current_session_id.set(session_id)

    data_root = get_abs_path(chroma_conf['data_path'])
    session_data_dir = os.path.join(data_root, session_id)
    os.makedirs(session_data_dir, exist_ok=True)

    md5_store_path = os.path.join(session_data_dir, chroma_conf['md5_hex_store'])
    existing_md5s = _load_existing_md5s(md5_store_path)
    status_store_path = os.path.join(session_data_dir, "ingest_status.json")
    ingest_status = _load_ingest_status(status_store_path)
    file_vector_map = _load_file_vector_map(session_data_dir)
    # 需要写入 md5_store 的新 MD5（包含重复文件的 MD5，确保前端状态准确）
    md5s_to_persist = set()

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
            safe_name = os.path.basename(file.filename)
            status_entry = ingest_status.get(safe_name, {})
            is_failed_file = status_entry.get("status") == "failed"
            has_vector_mapping = bool(file_vector_map.get(safe_name))

            # 失败文件允许重试：清理旧 MD5，重新进入向量化流程
            if is_failed_file or not has_vector_mapping:
                logger.info(f"[上传重试] {file.filename} 命中旧MD5但允许重试（failed={is_failed_file}, mapped={has_vector_mapping}）")
                _remove_md5(md5_store_path, md5_hex)
                existing_md5s.discard(md5_hex)
            else:
                results.append({
                    "filename": file.filename,
                    "status": "duplicate",
                    "reason": "文件内容与已有知识库重复，已跳过"
                })
                md5s_to_persist.add(md5_hex)
                logger.info(f"[上传去重] {file.filename} MD5={md5_hex} 已存在，跳过")
                continue

        # --- 保存文件 ---
        safe_name = os.path.basename(file.filename)
        file_path = os.path.join(session_data_dir, safe_name)
        with open(file_path, "wb") as buf:
            buf.write(content)

        md5s_to_persist.add(md5_hex)
        new_files.append(safe_name)
        results.append({
            "filename": file.filename,
            "status": "accepted",
            "reason": "已接收，排队向量化中"
        })
        logger.info(f"[上传] {file.filename} 已保存至 {file_path}")

    # 只有存在新文件时才触发后台向量化
    if new_files:
        _update_ingest_status(
            session_id,
            {name: {"status": "processing", "detail": "已接收，排队向量化中"} for name in new_files},
        )
        background_tasks.add_task(process_document_task, new_files, session_id)
    # duplicate 文件直接标记 completed（因内容已存在）
    dup_updates = {}
    for r in results:
        if r.get("status") == "duplicate" and r.get("filename"):
            dup_updates[r["filename"]] = {"status": "completed", "detail": "内容重复，复用已有向量"}
    if dup_updates:
        _update_ingest_status(session_id, dup_updates)

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
        from langchain_core.prompts import PromptTemplate  # fix #17：删除下方重复 import
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
    """上传样卷，学习试卷格式"""
    _validate_session_id(session_id)  # fix #2

    # fix #12：提前校验文件名，避免 splitext(None) 抛 AttributeError
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    content = await file.read()

    from utils.file_handler import document_loader
    import tempfile

    tmp_path = None  # fix #12：提前初始化，防止 finally 中 NameError
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

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
        # fix #12：tmp_path 可能未赋值（异常发生在 NamedTemporaryFile 之前时）
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.get("/sample")
async def get_sample_paper(session_id: str = Query(default="default")):
    """获取当前科目的样卷内容和格式"""
    _validate_session_id(session_id)  # fix #2
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
    """删除样卷"""
    _validate_session_id(session_id)  # fix #2
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
    """列出当前 Session 知识库中已上传的文件（不管是否完成向量化）。"""
    _validate_session_id(session_id)  # fix #2
    data_root = get_abs_path(chroma_conf['data_path'])
    session_data_dir = os.path.join(data_root, session_id)
    md5_store_name = chroma_conf.get('md5_hex_store', '.md5_hex_store')
    md5_store_path = os.path.join(session_data_dir, md5_store_name)
    status_store_path = os.path.join(session_data_dir, "ingest_status.json")

    if not os.path.isdir(session_data_dir):
        return {"code": 200, "files": [], "total": 0}

    # 读取已完成向量化的 MD5 集合（用于标记状态）
    embedded_md5s = _load_existing_md5s(md5_store_path)
    ingest_status = _load_ingest_status(status_store_path)
    file_vector_map = _load_file_vector_map(session_data_dir)

    files = []
    now = int(time.time())
    processing_timeout_sec = 600  # 10 分钟未完成则判定为失败，避免永久“处理中”
    status_changed = False
    for fname in os.listdir(session_data_dir):
        if fname == md5_store_name or fname.startswith('.'):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ALLOWED_SUFFIX:
            continue

        fpath = os.path.join(session_data_dir, fname)

        try:
            with open(fpath, "rb") as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
        except Exception:
            continue

        stat = os.stat(fpath)
        # 检查是否已完成向量化
        # 优先用文件->向量映射判断是否真正入库；旧数据回退到 md5 标记
        # 注意：历史版本可能没有 file_vector_map，不能因此误判为失败
        md5_embedded = file_md5 in embedded_md5s
        embedded = bool(file_vector_map.get(fname)) or md5_embedded
        state = ingest_status.get(fname, {})
        status = "completed" if embedded else state.get("status", "processing")
        detail = state.get("detail", "")
        updated_at = int(state.get("updated_at", int(stat.st_mtime)))

        # 历史数据修复：若已入库但状态仍为 failed/processing，自动回填 completed
        if embedded and state.get("status") != "completed":
            ingest_status[fname] = {
                "status": "completed",
                "detail": state.get("detail", "") if state.get("status") == "failed" else "向量化已完成",
                "updated_at": now,
            }
            status = "completed"
            status_changed = True

        # 兜底：历史遗留文件无状态文件/状态长期不更新，自动从 processing 转 failed
        if (not embedded) and status == "processing" and (now - updated_at) > processing_timeout_sec:
            status = "failed"
            detail = detail or "处理超时或后台任务中断，请重试上传（建议删除后重新上传）"
            ingest_status[fname] = {
                "status": status,
                "detail": detail,
                "updated_at": now,
            }
            status_changed = True

        files.append({
            "filename": fname,
            "size": stat.st_size,
            "modified_at": int(stat.st_mtime),
            "embedded": embedded,  # 标记是否已完成向量化
            "status": status,
            "detail": detail,
        })

    # 按修改时间倒序（最新在前）
    files.sort(key=lambda x: x["modified_at"], reverse=True)

    if status_changed:
        _save_ingest_status(status_store_path, ingest_status)

    return {"code": 200, "files": files, "total": len(files)}


@router.post("/retry-failed")
async def retry_failed_documents(
    background_tasks: BackgroundTasks,
    session_id: str = Query(default="default"),
):
    """重试当前 session 中标记为 failed 的文件。"""
    _validate_session_id(session_id)
    data_root = get_abs_path(chroma_conf['data_path'])
    session_data_dir = os.path.join(data_root, session_id)
    status_store_path = os.path.join(session_data_dir, "ingest_status.json")

    if not os.path.isdir(session_data_dir):
        return {"code": 404, "message": "会话目录不存在", "retry_count": 0}

    status_data = _load_ingest_status(status_store_path)
    retry_files = []
    for fname, meta in status_data.items():
        if meta.get("status") == "failed" and os.path.exists(os.path.join(session_data_dir, fname)):
            retry_files.append(fname)

    if not retry_files:
        return {"code": 200, "message": "没有可重试的失败文件", "retry_count": 0}

    _update_ingest_status(
        session_id,
        {name: {"status": "processing", "detail": "手动重试中"} for name in retry_files},
    )
    background_tasks.add_task(process_document_task, retry_files, session_id)
    logger.info(f"[重试失败文件] session={session_id}, files={retry_files}")
    return {
        "code": 200,
        "message": f"已提交重试任务：{len(retry_files)} 个文件",
        "retry_count": len(retry_files),
        "files": retry_files,
    }


@router.delete("/file/{filename:path}")
async def delete_file(
    filename: str,
    session_id: str = Query(default="default")
):
    """
    删除指定文件名对应的知识库文件及向量。
    filename 支持 URL 编码的中文文件名。
    """
    _validate_session_id(session_id)
    current_session_id.set(session_id)

    # URL 解码文件名
    from urllib.parse import unquote
    decoded_filename = unquote(filename)

    data_root = get_abs_path(chroma_conf['data_path'])
    session_data_dir = os.path.join(data_root, session_id)
    md5_store_path = os.path.join(session_data_dir, chroma_conf['md5_hex_store'])
    status_store_path = os.path.join(session_data_dir, "ingest_status.json")
    file_path = os.path.join(session_data_dir, decoded_filename)

    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {decoded_filename}")

    try:
        # 2. 计算被删文件的 MD5（删除前必须完成）
        with open(file_path, "rb") as f:
            file_md5 = hashlib.md5(f.read()).hexdigest()

        # 3. 删除物理文件
        os.remove(file_path)
        logger.info(f"[删除文件] 已删除物理文件: {file_path}")

        # 4. 从向量库删除对应的向量
        vs = VectorStoreService()
        await vs.delete_file_vectors(decoded_filename)

        # 5. 从 MD5 store 中移除该文件的 MD5
        if os.path.exists(md5_store_path):
            with open(md5_store_path, "r", encoding="utf-8") as f:
                md5_lines = [line.strip() for line in f if line.strip() and line.strip() != file_md5]
            with open(md5_store_path, "w", encoding="utf-8") as f:
                f.write("\n".join(md5_lines) + "\n")
        # 同步删除状态记录
        status_data = _load_ingest_status(status_store_path)
        if decoded_filename in status_data:
            del status_data[decoded_filename]
            _save_ingest_status(status_store_path, status_data)

        # 6. 刷新 RAG 缓存
        from agent.tools.agent_tools import _rag_cache as rag_cache_ref
        if session_id in rag_cache_ref:
            rag_cache_ref[session_id].refresh()

        logger.info(f"[删除文件] 完成: {decoded_filename}")
        return {"code": 200, "message": f"文件已删除: {decoded_filename}"}

    except Exception as e:
        logger.error(f"[删除文件] 失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")
