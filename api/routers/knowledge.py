import os
import re
import json
import hashlib
import shutil
import time
from typing import List, Dict, Any
from urllib.parse import unquote

from fastapi import APIRouter, File, UploadFile, BackgroundTasks, Query, HTTPException
from fastapi.responses import JSONResponse
from rag.vector_store import VectorStoreService
from utils.session_context import current_session_id
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from utils.kb_version import bump_kb_version
from utils.knowledge_ingest_policy import should_treat_batch_duplicate, can_retry_failed_meta
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


def _resolve_session_file_path(session_data_dir: str, raw_filename: str) -> tuple[str, str]:
    """解析并校验会话内文件路径，阻止路径穿越。"""
    decoded_filename = unquote(str(raw_filename or "")).strip()
    if not decoded_filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    normalized = decoded_filename.replace("\\", "/")
    if "/" in normalized or normalized in {".", ".."}:
        raise HTTPException(status_code=400, detail="文件名非法，不支持路径片段")

    base_dir = os.path.abspath(session_data_dir)
    target_path = os.path.abspath(os.path.join(base_dir, decoded_filename))
    try:
        in_scope = os.path.commonpath([base_dir, target_path]) == base_dir
    except ValueError:
        in_scope = False
    if not in_scope:
        raise HTTPException(status_code=400, detail="文件名非法，超出会话目录范围")
    return decoded_filename, target_path


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


def _append_md5_if_missing(md5_path: str, target_md5: str):
    """在 md5_store 里追加 md5（若不存在则追加）。"""
    if not target_md5:
        return
    try:
        os.makedirs(os.path.dirname(md5_path), exist_ok=True)
        existing = _load_existing_md5s(md5_path)
        if target_md5 in existing:
            return
        with open(md5_path, "a", encoding="utf-8") as f:
            f.write(target_md5 + "\n")
    except Exception as e:
        logger.warning(f"[上传去重] 追加MD5失败: {e}")


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


def _save_file_vector_map(session_data_dir: str, mapping: Dict[str, List[str]]):
    """保存文件->向量ID映射。"""
    map_path = os.path.join(session_data_dir, ".file_vector_map.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)


def _collect_vector_presence(session_id: str) -> tuple[Dict[str, List[str]], bool]:
    """
    扫描当前 session 向量库，返回 {source_filename: [doc_id...]}。
    用于校验“文件状态”与“真实向量”是否一致。
    """
    token = current_session_id.set(session_id)
    try:
        vs = VectorStoreService()
        data = vs.vector_store.get(include=["metadatas"])
        ids = data.get("ids", []) if isinstance(data, dict) else []
        metadatas = data.get("metadatas", []) if isinstance(data, dict) else []
        if not isinstance(ids, list) or not isinstance(metadatas, list):
            return {}, False

        presence: Dict[str, List[str]] = {}
        for idx, meta in enumerate(metadatas):
            if not isinstance(meta, dict):
                continue
            source_filename = str(meta.get("source_filename") or "").strip()
            if not source_filename:
                continue
            doc_id = str(ids[idx]) if idx < len(ids) else ""
            if not doc_id:
                continue
            presence.setdefault(source_filename, []).append(doc_id)
        return presence, True
    except Exception as e:
        logger.warning(f"[知识库状态] 读取向量存在性失败（session={session_id}）: {e}")
        return {}, False
    finally:
        current_session_id.reset(token)

def _count_md5_references(session_data_dir: str, target_md5: str, exclude_filename: str = "") -> int:
    """统计会话目录内（允许类型）引用某个 MD5 的文件数量。"""
    if not target_md5 or not os.path.isdir(session_data_dir):
        return 0
    count = 0
    for fname in os.listdir(session_data_dir):
        if fname.startswith(".") or fname == os.path.basename(chroma_conf.get("md5_hex_store", ".md5_hex_store")):
            continue
        if exclude_filename and fname == exclude_filename:
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ALLOWED_SUFFIX:
            continue
        fpath = os.path.join(session_data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "rb") as f:
                if hashlib.md5(f.read()).hexdigest() == target_md5:
                    count += 1
        except Exception:
            continue
    return count

def _has_vectorized_peer_for_md5(
    session_data_dir: str,
    target_md5: str,
    file_vector_map: Dict[str, List[str]],
    vector_presence: Dict[str, List[str]],
    vector_probe_ok: bool,
    exclude_filename: str = "",
) -> bool:
    """
    判断是否存在“同 MD5 且已入库”的其他文件。
    用于避免“同内容不同文件名”被误判为可重试。
    """
    if not target_md5 or not os.path.isdir(session_data_dir):
        return False
    for fname in os.listdir(session_data_dir):
        if fname.startswith(".") or fname == os.path.basename(chroma_conf.get("md5_hex_store", ".md5_hex_store")):
            continue
        if exclude_filename and fname == exclude_filename:
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ALLOWED_SUFFIX:
            continue
        fpath = os.path.join(session_data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "rb") as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
            if file_md5 != target_md5:
                continue
        except Exception:
            continue
        has_live_vectors = bool(vector_presence.get(fname))
        if has_live_vectors:
            return True
        # 向量探测失败时，退化为映射兜底；探测成功时不再信任“仅映射”。
        if (not vector_probe_ok) and bool(file_vector_map.get(fname)):
            return True
    return False


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
        task_start = time.time()
        current_session_id.set(session_id)
        data_root = get_abs_path(chroma_conf['data_path'])
        session_data_dir = os.path.join(data_root, session_id)
        vs = VectorStoreService()
        # 标记为处理开始
        _update_ingest_status(
            session_id,
            {name: {"status": "processing", "detail": "正在解析并向量化"} for name in filenames},
        )

        parse_start = time.time()
        result_map = await vs.load_document(target_filenames=filenames)
        logger.info(
            f"[Latency] upload_vectorize_ms={int((time.time() - parse_start) * 1000)}, "
            f"session={session_id}, files={len(filenames)}"
        )

        status_patch = {}
        for name in filenames:
            info = result_map.get(name, {"status": "failed", "detail": "后台任务未返回该文件结果"})
            status_patch[name] = {
                "status": info.get("status", "failed"),
                "detail": info.get("detail", ""),
            }
        _update_ingest_status(session_id, status_patch)
        completed_files = [name for name, meta in status_patch.items() if str(meta.get("status", "")).strip() == "completed"]
        if completed_files:
            try:
                version = bump_kb_version(
                    session_data_dir,
                    reason=f"vectorize:{','.join(completed_files[:3])}",
                )
                logger.info(f"[知识库版本] session={session_id}, version={version}, completed={len(completed_files)}")
            except Exception as e:
                # 版本号写入失败不影响主任务结果，避免误把已完成文件标记为失败。
                logger.warning(f"[知识库版本] bump失败（session={session_id}）: {e}")

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
        logger.info(f"[Latency] upload_task_total_ms={int((time.time() - task_start) * 1000)}")

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
    vector_presence, vector_probe_ok = _collect_vector_presence(session_id)
    # 需要写入 md5_store 的新 MD5（包含重复文件的 MD5，确保前端状态准确）
    md5s_to_persist = set()

    results = []
    new_files = []

    for file in files:
        # --- 校验后缀 ---
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_SUFFIX:
            allow_hint = " / ".join(sorted(ALLOWED_SUFFIX))
            results.append({
                "filename": file.filename,
                "status": "rejected",
                "reason": f"不支持的文件类型 {ext}，仅允许 {allow_hint}"
            })
            continue

        # --- 读取内存，计算 MD5 ---
        content = await file.read()
        md5_hex = _compute_bytes_md5(content)

        # 批内去重：同一请求已出现过该 MD5，则直接视为重复，避免重复向量化。
        if should_treat_batch_duplicate(md5_hex, md5s_to_persist):
            results.append({
                "filename": file.filename,
                "status": "duplicate",
                "reason": "文件内容与当前批次已上传文件重复，已跳过"
            })
            logger.info(f"[上传去重] {file.filename} MD5={md5_hex} 与本批次重复，跳过")
            continue

        if md5_hex in existing_md5s:
            safe_name = os.path.basename(file.filename)
            status_entry = ingest_status.get(safe_name, {})
            is_failed_file = status_entry.get("status") == "failed"
            has_live_vectors = bool(vector_presence.get(safe_name))
            has_vector_mapping = bool(file_vector_map.get(safe_name))
            # 仅在探测成功时，才把“仅映射无向量”视为异常。
            if vector_probe_ok and has_vector_mapping and not has_live_vectors:
                logger.warning(f"[上传重试] {safe_name} 映射存在但未检测到真实向量，允许重建")
                has_vector_mapping = False
            peer_vectorized = _has_vectorized_peer_for_md5(
                session_data_dir=session_data_dir,
                target_md5=md5_hex,
                file_vector_map=file_vector_map,
                vector_presence=vector_presence,
                vector_probe_ok=vector_probe_ok,
                exclude_filename=safe_name,
            )

            # 失败文件允许重试：清理旧 MD5，重新进入向量化流程
            # 仅当“同名文件失败”或“同名映射缺失且不存在其他已入库同MD5文件”时允许重试
            if is_failed_file or (not has_vector_mapping and not peer_vectorized):
                logger.info(
                    f"[上传重试] {file.filename} 命中旧MD5但允许重试（failed={is_failed_file}, "
                    f"mapped={has_vector_mapping}, peer_vectorized={peer_vectorized}）"
                )
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
            existing_path = os.path.join(session_data_dir, os.path.basename(r["filename"]))
            # 仅对真实存在的本地文件回填状态，避免“幽灵状态”污染 ingest_status
            if os.path.exists(existing_path):
                dup_updates[r["filename"]] = {"status": "completed", "detail": "内容重复，复用已有向量"}
    if dup_updates:
        _update_ingest_status(session_id, dup_updates)

    # 将本批次涉及的 MD5 落盘，后续请求可正确命中去重。
    for md5_hex in md5s_to_persist:
        _append_md5_if_missing(md5_store_path, md5_hex)

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
    status_store_path = os.path.join(session_data_dir, "ingest_status.json")

    if not os.path.isdir(session_data_dir):
        return {"code": 200, "files": [], "total": 0}

    ingest_status = _load_ingest_status(status_store_path)
    file_vector_map = _load_file_vector_map(session_data_dir)
    vector_presence, vector_probe_ok = _collect_vector_presence(session_id)

    raw_files: List[Dict[str, Any]] = []
    md5_to_files: Dict[str, List[str]] = {}
    for fname in os.listdir(session_data_dir):
        if fname == md5_store_name or fname.startswith('.'):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ALLOWED_SUFFIX:
            continue
        fpath = os.path.join(session_data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "rb") as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
        except Exception:
            continue
        stat = os.stat(fpath)
        raw_files.append(
            {
                "filename": fname,
                "size": int(stat.st_size),
                "modified_at": int(stat.st_mtime),
                "md5": file_md5,
            }
        )
        md5_to_files.setdefault(file_md5, []).append(fname)

    files = []
    now = int(time.time())
    processing_timeout_sec = 600  # 10 分钟未完成则判定为失败，避免永久“处理中”
    status_changed = False
    map_changed = False

    for item in raw_files:
        fname = str(item["filename"])
        file_md5 = str(item["md5"])
        modified_at = int(item["modified_at"])

        state = ingest_status.get(fname, {}) if isinstance(ingest_status.get(fname, {}), dict) else {}
        status = str(state.get("status", "processing") or "processing")
        detail = str(state.get("detail", "") or "")
        updated_at = int(state.get("updated_at", modified_at))

        # 映射与真实向量双校验，避免“状态显示完成但实际无向量”。
        mapped_ids = list(file_vector_map.get(fname) or [])
        live_ids = list(vector_presence.get(fname) or [])
        has_mapping = bool(mapped_ids)
        has_live_vectors = bool(live_ids)

        # 探测成功时，主动清理“仅映射无向量”的陈旧映射。
        if vector_probe_ok and has_mapping and not has_live_vectors:
            file_vector_map.pop(fname, None)
            mapped_ids = []
            has_mapping = False
            map_changed = True

        # 自愈：若向量库存在但映射缺失，回填映射，避免后续误判。
        if has_live_vectors and not has_mapping:
            file_vector_map[fname] = live_ids
            mapped_ids = live_ids
            has_mapping = True
            map_changed = True

        peers = md5_to_files.get(file_md5, [])
        if vector_probe_ok:
            peer_vectorized = any(
                peer != fname and bool(vector_presence.get(peer))
                for peer in peers
            )
            embedded = has_live_vectors or peer_vectorized
        else:
            # 探测失败时降级使用映射，避免误把全部文件打成 failed。
            peer_vectorized = any(
                peer != fname and (bool(file_vector_map.get(peer)) or bool(vector_presence.get(peer)))
                for peer in peers
            )
            embedded = has_mapping or has_live_vectors or peer_vectorized

        # 历史数据修复：确实有向量但状态非 completed -> 自动修复 completed。
        if embedded and status != "completed":
            if peer_vectorized and not has_live_vectors:
                detail = detail or "内容重复，复用已有向量"
            else:
                chunk_count = len(mapped_ids)
                detail = detail or (f"向量化已完成，共{chunk_count}个片段" if chunk_count > 0 else "向量化已完成")
            ingest_status[fname] = {
                "status": "completed",
                "detail": detail,
                "updated_at": now,
            }
            status = "completed"
            status_changed = True

        # 状态矫正：显示 completed 但未检测到向量，回退为 failed，允许“重试失败”补齐。
        if (not embedded) and status == "completed":
            status = "failed"
            detail = detail or "状态异常：未检测到有效向量，请点击“重试失败”重新入库。"
            ingest_status[fname] = {
                "status": status,
                "detail": detail,
                "updated_at": now,
            }
            status_changed = True

        # 兜底：processing 长时间不更新 -> failed
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
            "size": int(item["size"]),
            "modified_at": modified_at,
            "embedded": embedded,
            "status": status,
            "detail": detail,
            "vector_chunks": len(live_ids) if has_live_vectors else 0,
        })

    # 按修改时间倒序（最新在前）
    files.sort(key=lambda x: x["modified_at"], reverse=True)

    if map_changed:
        _save_file_vector_map(session_data_dir, file_vector_map)
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
        return JSONResponse(
            status_code=404,
            content={"code": 404, "message": "会话目录不存在", "retry_count": 0},
        )

    status_data = _load_ingest_status(status_store_path)
    vector_presence, vector_probe_ok = _collect_vector_presence(session_id)

    md5_by_file: Dict[str, str] = {}
    md5_to_files: Dict[str, List[str]] = {}
    for fname in os.listdir(session_data_dir):
        if fname.startswith(".") or fname == os.path.basename(chroma_conf.get("md5_hex_store", ".md5_hex_store")):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ALLOWED_SUFFIX:
            continue
        fpath = os.path.join(session_data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "rb") as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
        except Exception:
            continue
        md5_by_file[fname] = file_md5
        md5_to_files.setdefault(file_md5, []).append(fname)

    retry_files = []
    for fname, meta in status_data.items():
        if can_retry_failed_meta(meta) and os.path.exists(os.path.join(session_data_dir, fname)):
            retry_files.append(fname)

    status_changed = False
    if vector_probe_ok:
        # 将“completed 但无真实向量”的伪完成文件纳入重试，避免界面误导。
        for fname, file_md5 in md5_by_file.items():
            meta = status_data.get(fname, {}) if isinstance(status_data.get(fname, {}), dict) else {}
            status = str(meta.get("status", "processing") or "processing")
            if status != "completed":
                continue

            has_live_vectors = bool(vector_presence.get(fname))
            if has_live_vectors:
                continue

            peers = md5_to_files.get(file_md5, [])
            peer_vectorized = any(
                peer != fname and bool(vector_presence.get(peer))
                for peer in peers
            )
            if peer_vectorized:
                continue

            retry_files.append(fname)
            status_data[fname] = {
                "status": "failed",
                "detail": "状态异常：未检测到有效向量，已自动加入重试队列",
                "updated_at": int(time.time()),
            }
            status_changed = True

    retry_files = list(dict.fromkeys(retry_files))
    if status_changed:
        _save_ingest_status(status_store_path, status_data)

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

    data_root = get_abs_path(chroma_conf['data_path'])
    session_data_dir = os.path.join(data_root, session_id)
    md5_store_path = os.path.join(session_data_dir, chroma_conf['md5_hex_store'])
    status_store_path = os.path.join(session_data_dir, "ingest_status.json")
    decoded_filename, file_path = _resolve_session_file_path(session_data_dir, filename)

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
            # 若仍有其他文件引用该 MD5，则保留 md5 标记，避免误判未入库
            ref_count = _count_md5_references(session_data_dir, file_md5, exclude_filename=decoded_filename)
            if ref_count <= 0:
                with open(md5_store_path, "r", encoding="utf-8") as f:
                    md5_lines = [line.strip() for line in f if line.strip() and line.strip() != file_md5]
                with open(md5_store_path, "w", encoding="utf-8") as f:
                    if md5_lines:
                        f.write("\n".join(md5_lines) + "\n")
        # 同步删除状态记录
        status_data = _load_ingest_status(status_store_path)
        if decoded_filename in status_data:
            del status_data[decoded_filename]
            _save_ingest_status(status_store_path, status_data)

        try:
            version = bump_kb_version(session_data_dir, reason=f"delete:{decoded_filename}")
            logger.info(f"[知识库版本] session={session_id}, version={version}, action=delete_file")
        except Exception as e:
            # 删除流程已完成时，不应因版本号写入失败而返回 500。
            logger.warning(f"[知识库版本] delete后bump失败（session={session_id}）: {e}")

        # 6. 刷新 RAG 缓存
        from agent.tools.agent_tools import _rag_cache as rag_cache_ref
        if session_id in rag_cache_ref:
            rag_cache_ref[session_id].refresh()

        logger.info(f"[删除文件] 完成: {decoded_filename}")
        return {"code": 200, "message": f"文件已删除: {decoded_filename}"}

    except Exception as e:
        logger.error(f"[删除文件] 失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")
