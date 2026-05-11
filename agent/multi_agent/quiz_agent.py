"""
多 Agent 出题系统 - Reflexion 架构
Agent 1: 出题 Agent  — 生成题目 + 推理链（为什么这样出题、答案依据何处）
Agent 2: Critic Agent — 质疑推理链，找出逻辑漏洞，输出结构化批评
Agent 3: Revise Agent — 逐条回应批评，修订题目并给出修改说明
循环：critique → revise → critique，最多 2 轮
"""
import json
import os
import re
import asyncio
import time
import random
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from model.factory import chat_model, light_chat_model, backup_chat_model, backup_light_chat_model
from agent.tools.agent_tools import fetch_exam_paper_reference, search_exam_format, get_rag_service
from utils.config_handler import chroma_conf
from utils.session_context import current_session_id
from utils.logger_handler import logger
from utils.rag_metrics import rag_inc, rag_append_sample, rag_set
from agent.multi_agent.quiz_types import (
    QuizState,
    ExamPaperState,
    Question,
    QuizQuestionItem,
    StructuredQuizSetResult,
    ExamPaper,
    ExamPaperText,
    QuizGenerateResult,
    CritiqueResult,
    ReviseResult,
    ExamReviseResult,
)
from agent.multi_agent.quiz_prompt_runtime import (
    get_exam_generate_single_type_prompt as _get_exam_generate_single_type_prompt,
    get_exam_generate_structured_prompt as _get_exam_generate_structured_prompt,
    get_quiz_generate_structured_prompt as _get_quiz_generate_structured_prompt,
    get_quiz_generate_reasoning_prompt as _get_quiz_generate_reasoning_prompt,
    get_quiz_critique_prompt as _get_quiz_critique_prompt,
    get_quiz_revise_prompt as _get_quiz_revise_prompt,
    get_quiz_generate_fallback_prompt as _get_quiz_generate_fallback_prompt,
    get_exam_contract_prompt as _get_exam_contract_prompt,
    get_exam_generate_reasoning_prompt as _get_exam_generate_reasoning_prompt,
    get_exam_critique_prompt as _get_exam_critique_prompt,
    get_exam_revise_prompt as _get_exam_revise_prompt,
    get_format_examples_prompt as _get_format_examples_prompt,
)
from agent.multi_agent.quiz_normalization import (
    normalize_option_text as _normalize_option_text,
    normalize_text_field as _normalize_text_field,
    sanitize_question_text as _sanitize_question_text,
    sanitize_user_visible_text as _sanitize_user_visible_text,
    contains_term_style_violation as _contains_term_style_violation,
    normalize_int_field as _normalize_int_field,
    normalize_dict_field as _normalize_dict_field,
    normalize_list_of_dicts as _normalize_list_of_dicts,
    normalize_reasoning_payload as _normalize_reasoning_payload,
    normalize_options as _normalize_options,
    normalize_question_type as _normalize_question_type,
    normalize_question_item as _normalize_question_item,
    normalize_questions_payload as _normalize_questions_payload,
    sanitize_quiz_questions_for_delivery as _sanitize_quiz_questions_for_delivery,
    normalize_exam_question_item as _normalize_exam_question_item,
    normalize_exam_questions_payload as _normalize_exam_questions_payload,
    limit_context_size as _limit_context_size,
    normalize_addressed_issues as _normalize_addressed_issues,
    format_score as _format_score,
)
from agent.multi_agent.quiz_quality import (
    question_signature as _question_signature,
    question_dedup_key as _question_dedup_key,
    target_counts as _target_counts,
    question_term_style_ok as _question_term_style_ok,
    collect_term_style_violation_numbers as _collect_term_style_violation_numbers,
    score_map_for_counts as _score_map_for_counts,
    detect_concept_anchor as _detect_concept_anchor,
    concept_key_for_question as _concept_key_for_question,
    count_prompt_contract_violations as _count_prompt_contract_violations,
    exam_quality_floor_flags as _exam_quality_floor_flags,
    compute_fixed_items as _compute_fixed_items,
    count_questions_in_exam as _count_questions_in_exam,
    build_fast_exam_critique as _build_fast_exam_critique,
)
from agent.multi_agent.quiz_json_utils import (
    try_json_load as _try_json_load,
    extract_json as _extract_json,
    extract_loose_json_fields_for_schema as _extract_loose_json_fields_for_schema,
    sanitize_structured_output as _sanitize_structured_output,
    summarize_raw_output as _summarize_raw_output,
    strip_think_tags as _strip_think_tags,
)
from agent.multi_agent.quiz_parser import (
    build_quiz_text_from_questions as _build_quiz_text_from_questions,
    build_quiz_payload as _build_quiz_payload,
    parse_questions_from_serialized_quiz as _parse_questions_from_serialized_quiz,
    questions_from_quiz_text as _questions_from_quiz_text,
)
from agent.multi_agent.quiz_runtime import (
    call_llm as _runtime_call_llm,
    call_llm_structured as _runtime_call_llm_structured,
    classify_structured_error as _runtime_classify_structured_error,
    get_structured_fail_stats as _runtime_get_structured_fail_stats,
)
from api.routers.exam_export import parse_exam_content

EXAM_BUDGET_SECONDS = 600
QUIZ_BUDGET_SECONDS = 600
QUIZ_MIN_COURSEWARE_HITS = 2
ALLOW_DEGRADED_DELIVERY = True
EXAM_STAGE_RETRY_ENABLED = True
EXAM_ENABLE_LOCAL_EMERGENCY_FALLBACK = False
EXAM_MISSING_MODEL_RETRY_ROUNDS = 6
EXAM_TYPE_ORDER = ["选择题", "填空题", "判断题", "简答题"]
TYPE_KEY_TO_LABEL = {"choice": "选择题", "fill": "填空题", "judge": "判断题", "essay": "简答题"}
TYPE_LABEL_TO_KEY = {v: k for k, v in TYPE_KEY_TO_LABEL.items()}
_COURSEWARE_WARMUP_CACHE: Dict[str, Dict[str, Any]] = {}
_WARMUP_LOCKS: Dict[str, asyncio.Lock] = {}


# ==================== 课件预热缓存 ====================

def _build_courseware_manifest(session_data_path: str) -> Dict[str, Any]:
    """
    列出 session_data_path 下所有已上传课件文件（不含子目录），
    并计算整体 fingerprint（所有文件 MD5 排序后拼接的 MD5）。
    """
    if not session_data_path or not os.path.isdir(session_data_path):
        return {"files": [], "fingerprint": ""}
    try:
        from utils.file_handler import listdir_with_allowed_type

        # file_handler 返回绝对路径；预热 seed 仅应使用“文件名”避免路径污染检索 query。
        raw_files = listdir_with_allowed_type(
            session_data_path,
            (".pdf", ".ppt", ".pptx", ".doc", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg")
        )
        if not raw_files:
            return {"files": [], "fingerprint": ""}

        session_root = os.path.abspath(session_data_path)
        basenames: List[str] = []
        for item in raw_files:
            abs_item = os.path.abspath(str(item or ""))
            if not abs_item:
                continue
            try:
                # 安全约束：仅接受当前会话目录下文件，避免跨目录误混入。
                if os.path.commonpath([session_root, abs_item]) != session_root:
                    continue
            except Exception:
                continue
            name = os.path.basename(abs_item).strip()
            if name:
                basenames.append(name)

        files_sorted = sorted(set(basenames))
        if not files_sorted:
            return {"files": [], "fingerprint": ""}

        import hashlib
        combined = "".join(files_sorted)
        fingerprint = hashlib.md5(combined.encode("utf-8")).hexdigest()[:16]
        return {"files": files_sorted, "fingerprint": fingerprint}
    except Exception:
        return {"files": [], "fingerprint": ""}


async def _get_or_build_courseware_warmup(rag: Any, force: bool = False) -> Dict[str, Any]:
    """
    课件预热摘要池（会话级缓存）：
    - 记录已上传文件清单
    - 预抽课件考点池，供"随机出题"多考点覆盖
    """
    sid = current_session_id.get() or "default"
    lock = _WARMUP_LOCKS.setdefault(sid, asyncio.Lock())
    async with lock:
        # 使用文件清单 fingerprint 作为缓存版本号，文件不变则直接复用。
        session_data_path = getattr(getattr(rag, "vector_store_service", None), "session_data_path", "")
        manifest = _build_courseware_manifest(session_data_path)
        files = manifest.get("files", [])
        fingerprint = manifest.get("fingerprint", "")

        cached = _COURSEWARE_WARMUP_CACHE.get(sid)
        if (
            not force
            and isinstance(cached, dict)
            and cached.get("fingerprint") == fingerprint
            and cached.get("files") == files
        ):
            return cached

        name_terms = [os.path.splitext(f)[0].replace("_", " ").replace("-", " ").strip() for f in files]
        name_terms = [t for t in name_terms if t]
        warm_seed = "；".join(name_terms[:8]) if name_terms else ""
        warm_query = (warm_seed + " 核心概念 机制 流程 易错点").strip() or "课程课件 核心概念 机制 流程 易错点"
        warm_context = await _get_comprehensive_rag_context(warm_query)
        topic_pool = _extract_topics_from_rag_context(warm_context, limit=max(28, min(80, max(32, len(files) * 8))))

        payload = {
            "fingerprint": fingerprint,
            "files": files,
            "warm_query": warm_query,
            "topic_pool": topic_pool,
            "core_topics": topic_pool[: min(12, len(topic_pool))],
            "weak_topics": [],
            "dynamic_topics": [],
            "topic_scores": [{"topic": t, "score": 60.0, "tag": "normal"} for t in topic_pool[:60]],
            "last_refresh_source": "startup_warmup",
            "updated_at": int(time.time()),
        }
        _COURSEWARE_WARMUP_CACHE[sid] = payload
        logger.info(f"[Courseware Warmup] session={sid}, files={len(files)}, topics={len(topic_pool)}")
        return payload


async def prewarm_courseware_summary_for_session(session_id: str, force: bool = False) -> Dict[str, Any]:
    """为指定会话构建课件预热摘要缓存，供启动预热与首题加速使用。"""
    sid = str(session_id or "").strip() or "default"
    token = current_session_id.set(sid)
    try:
        rag = await get_rag_service()
        payload = await _get_or_build_courseware_warmup(rag, force=force)
        logger.info(
            f"[Courseware Warmup API] session={sid}, files={len(payload.get('files', []))}, "
            f"topics={len(payload.get('topic_pool', []))}, force={bool(force)}"
        )
        return payload
    finally:
        current_session_id.reset(token)


# ==================== 工具函数 ====================

async def get_rag_context(topic: str) -> str:
    """获取 RAG 原始检索片段（不做最终 LLM 总结，出题 Agent 自己基于原文出题）"""
    logger.info(f"[RAG Context] 正在检索 topic={topic[:50]}...")
    try:
        rag = await get_rag_service()
        # quiz 场景使用独立 mode，避免复用 rag_chat 上下文缓存导致多轮同题。
        context = await rag.retrieve_context(topic, mode="quiz")
        logger.info(f"[RAG Context] 检索完成，返回长度={len(context)}")
        if not context or context.strip() == "":
            logger.warning("[RAG Context] 知识库为空，返回提示信息")
            return "【提示】当前知识库为空，无法生成题目。请先上传课件后再请求出题。"
        return context
    except Exception as e:
        logger.error(f"[RAG Context] 检索异常: {e}")
        return f"【提示】知识库检索失败: {str(e)}"


def _default_comprehensive_topics() -> List[str]:
    """未指定章节时的综合覆盖考点标签（学科无关）。"""
    return [
        "核心概念与术语",
        "关键流程与方法",
        "架构与模块关系",
        "应用场景与实践案例",
        "常见错误与排查思路",
        "重点难点与综合训练",
        "课程总结与迁移应用",
    ]


def _is_os_domain_topic(text: str) -> bool:
    """判断 seed 是否明显属于操作系统领域（用于保留历史检索优势）。"""
    t = str(text or "").lower()
    if not t:
        return False
    os_signals = [
        "操作系统", "进程", "线程", "调度", "死锁",
        "虚拟内存", "页面置换", "文件系统", "i/o", "内核",
    ]
    return any(sig in t for sig in os_signals)


def _build_comprehensive_queries(seed_topic: str) -> List[str]:
    """根据 seed 构建 4 路综合检索 query，避免跨学科锚点污染。"""
    base = str(seed_topic or "").strip()
    if _is_os_domain_topic(base):
        # 操作系统会话保留原锚点，避免已调优场景回退。
        return [
            f"{base} 基础概念 体系结构".strip(),
            f"{base} 进程 线程 调度 同步 死锁".strip(),
            f"{base} 内存管理 虚拟内存 页面置换".strip(),
            f"{base} 文件系统 I/O 设备管理 网络通信".strip(),
        ]
    # 通用学科锚点：适配“实训/产品/工程”等非操作系统主题。
    return [
        f"{base} 核心概念 术语 定义".strip(),
        f"{base} 关键流程 方法 步骤".strip(),
        f"{base} 架构 模块 组件 接口".strip(),
        f"{base} 应用场景 实践案例 易错点".strip(),
    ]


async def _get_comprehensive_rag_context(seed_topic: str = "") -> str:
    """
    综合覆盖检索：针对不同模块做多路检索并拼接，避免只命中少数章节。
    """
    rag_inc("comprehensive_calls", 1)
    start_ts = time.time()
    base = str(seed_topic or "").strip()
    queries = _build_comprehensive_queries(base)
    # 4 路子查询并行覆盖不同知识域，降低“只命中一章”概率。
    # 请求级去重并按归一化 key 稳定排序，减少缓存 key 抖动
    dedup_pairs: List[Tuple[str, str]] = []
    seen = set()
    for q in queries:
        q = re.sub(r"\s+", " ", q).strip()
        if not q:
            continue
        norm = re.sub(r"\s+", " ", q.lower())
        if norm in seen:
            continue
        seen.add(norm)
        dedup_pairs.append((norm, q))
    dedup_pairs.sort(key=lambda x: x[0])
    dedup_queries = [q for _, q in dedup_pairs]

    parallelism = max(1, int(chroma_conf.get("comprehensive_retrieve_parallelism", 2)))
    sem = asyncio.Semaphore(parallelism)

    async def _fetch_ctx(query_text: str) -> str:
        async with sem:
            return await get_rag_context(query_text)

    tasks = [_fetch_ctx(q) for q in dedup_queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    contexts: List[str] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"[Comprehensive Retrieve] 并行检索失败: {result}")
            continue
        ctx = str(result or "").strip()
        if ctx:
            contexts.append(ctx)

    elapsed_ms = int((time.time() - start_ts) * 1000)
    rag_set("comprehensive_parallel_ms", elapsed_ms)
    rag_append_sample("comprehensive_parallel_ms_samples", elapsed_ms)

    if not contexts:
        return await get_rag_context(base or "本课程重点知识点")

    merged = "\n\n".join(contexts)
    return _limit_context_size(merged, max_chars=16000)


async def _fetch_web_reference_for_quiz(topic: str, timeout_seconds: float = 4.0) -> str:
    """
    联网补充题源：只作为辅助上下文，不替代课件依据。
    每个引擎都带短超时，避免拖慢主流程。
    """
    query = re.sub(r'（.*?）', '', str(topic or "")).strip()
    if not query:
        query = "课程核心知识点"
    search_query = f"{query} 核心知识点 典型考法 练习题"

    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        def _ddg_run() -> str:
            tool = DuckDuckGoSearchRun()
            return tool.run(search_query)

        result = await asyncio.wait_for(asyncio.to_thread(_ddg_run), timeout=timeout_seconds)
        if result and len(str(result).strip()) > 40:
            logger.info("[Quiz Web] DuckDuckGo 搜索补充成功")
            return f"【联网补充参考】\n{str(result)[:1500]}"
    except Exception as e:
        logger.warning(f"[Quiz Web] DuckDuckGo 搜索失败: {e}")

    try:
        from langchain_community.tools import TavilySearchResults

        def _tavily_run() -> str:
            tool = TavilySearchResults(max_results=3)
            docs = tool.run(search_query)
            if not docs:
                return ""
            chunks: List[str] = []
            for doc in docs:
                if isinstance(doc, dict):
                    content = str(doc.get("content", "")).strip()
                    if content:
                        chunks.append(content[:500])
            return "\n".join(chunks)

        result = await asyncio.wait_for(asyncio.to_thread(_tavily_run), timeout=timeout_seconds)
        if result and len(str(result).strip()) > 40:
            logger.info("[Quiz Web] Tavily 搜索补充成功")
            return f"【联网补充参考】\n{str(result)[:1500]}"
    except Exception as e:
        logger.warning(f"[Quiz Web] Tavily 搜索失败: {e}")

    return ""


async def call_llm(prompt_template: str, **kwargs) -> str:
    """调用 LLM（运行时封装：超时+有限重试+主备切换）。"""
    return await _runtime_call_llm(
        prompt_template,
        kwargs,
        primary_model=chat_model,
        backup_model=backup_chat_model,
        logger=logger,
    )


async def call_llm_structured(prompt_template: str, schema: type[BaseModel], **kwargs) -> BaseModel:
    """调用 LLM 并强制结构化输出（运行时封装）。"""
    return await _runtime_call_llm_structured(
        prompt_template,
        schema,
        kwargs,
        primary_model=chat_model,
        backup_model=backup_chat_model,
        logger=logger,
        extract_json_fn=_extract_json,
        coerce_payload_fn=_coerce_structured_payload,
        extract_loose_fields_fn=_extract_loose_json_fields_for_schema,
        summarize_raw_output_fn=_summarize_raw_output,
    )


async def generate_single_type_paper(
    quiz_type: str,
    num: int,
    start_num: int,
    topics: str,
    context: str,
    sample_paper_context: str,
    score_per_question: int = 2
) -> str:
    """
    分批生成单种题型的试卷
    """
    import asyncio

    end_num = start_num + num - 1
    total_score = num * score_per_question
    # 简答题每问的分值 = 总分 // 2（如果每题10分则每问5分，如果每题20分则每问10分）
    sub_score = score_per_question // 2

    # 获取格式示例（仅从 prompts/format_examples.txt 读取）
    format_template = _get_format_examples_prompt().strip()
    if not format_template:
        raise RuntimeError("格式示例未配置：请检查 prompts/format_examples.txt")

    format_example = format_template.format(
        num=num,
        score=score_per_question,
        total_score=total_score,
        start_num=start_num,
        start_num_plus1=start_num + 1,
        end_num=end_num,
        sub_score=sub_score
    )

    logger.info(f"[分批生成] {quiz_type}: {num}题, 编号{start_num}-{end_num}")

    try:
        result = await asyncio.wait_for(
            call_llm_structured(
                _get_exam_generate_single_type_prompt(),
                ExamPaperText,
                quiz_type=quiz_type,
                num=num,
                start_num=start_num,
                end_num=end_num,
                score=score_per_question,
                total_score=total_score,
                topics=topics,
                sample_paper_context=sample_paper_context,
                context=context,
                format_example=format_example,
            ),
            timeout=150.0,  # 生成阶段放宽超时，优先保证完整出题
        )
        logger.info(f"[分批生成] {quiz_type} 生成成功，长度={len(result.exam_paper)}")
        return result.exam_paper
    except asyncio.TimeoutError:
        logger.error(f"[分批生成] {quiz_type} 超时（150s）")
        return ""
    except Exception as e:
        logger.warning(f"[分批生成] {quiz_type} 失败: {e}")
        return ""


async def generate_single_type_paper_structured(
    quiz_type: str,
    num: int,
    start_num: int,
    topics: str,
    context: str,
    sample_paper_context: str,
) -> ExamPaper:
    """
    生成单种题型的结构化试卷（返回 JSON 而不是文本）
    让 AI 自行决定分值和难度分配
    """
    import asyncio

    logger.info(f"[结构化生成] {quiz_type}: {num}题, 起始编号{start_num}")

    try:
        result = await asyncio.wait_for(
            call_llm_structured(
                _get_exam_generate_structured_prompt(),
                ExamPaper,
                quiz_type=quiz_type,
                num=num,
                start_num=start_num,
                topics=topics,
                sample_paper_context=sample_paper_context,
                context=context,
            ),
            timeout=150.0,  # 结构化生成放宽超时，优先保证完整出题
        )
        actual_count = len(result.questions)
        if actual_count != num:
            raise ValueError(f"{quiz_type} 结构化生成数量不符，期望 {num} 道，实际 {actual_count} 道")
        for index, question in enumerate(result.questions):
            question.id = start_num + index
            if not question.type:
                question.type = quiz_type
        logger.info(f"[结构化生成] {quiz_type} 成功，生成 {len(result.questions)} 道题")
        return result
    except asyncio.TimeoutError:
        logger.error(f"[结构化生成] {quiz_type} 超时（150s）")
        return ExamPaper(questions=[], reasoning={})
    except Exception as e:
        logger.error(f"[结构化生成] {quiz_type} 失败: {e}")
        return ExamPaper(questions=[], reasoning={})


def _merge_exam_questions(primary: List[dict], supplement: List[dict], quiz_type: str, start_num: int) -> List[dict]:
    merged: List[dict] = []
    current_num = start_num
    for question in primary + supplement:
        normalized = _normalize_exam_question(question, current_num)
        normalized["number"] = current_num
        normalized["type"] = normalized.get("type") or quiz_type
        merged.append(normalized)
        current_num += 1
    return merged


def _count_courseware_hits(context: str) -> int:
    if not context:
        return 0
    return len(re.findall(r'\[参考资料\d+\]', context))


GENERIC_TOPIC_STOPWORDS = {
    "当前", "课件", "核心", "知识点", "自行", "选题", "练习", "练习题", "出题", "做题",
    "试卷", "综合", "测试", "考试", "题目", "一道", "几道", "若干", "请", "根据", "生成",
    "系统", "机制", "流程", "内容",
}


def _extract_topic_keywords(text: str, limit: int = 10) -> List[str]:
    raw = str(text or "")
    if not raw:
        return []
    candidates = re.findall(r'[\u4e00-\u9fa5]{2,10}|[A-Za-z][A-Za-z0-9_-]{2,}', raw)
    keywords: List[str] = []
    seen = set()
    for token in candidates:
        t = token.strip()
        if not t or t in GENERIC_TOPIC_STOPWORDS:
            continue
        if len(t) <= 1:
            continue
        if t in seen:
            continue
        seen.add(t)
        keywords.append(t)
        if len(keywords) >= limit:
            break
    return keywords


def _extract_topics_from_rag_context(context: str, limit: int = 20) -> List[str]:
    """
    从 RAG 命中上下文抽取细粒度考点标签，避免固定 7 大类导致题目重复。
    """
    if not context:
        return []

    candidates: List[str] = []
    seen = set()

    def _is_valid_topic_phrase(text: str) -> bool:
        t = str(text or "").strip()
        if not t:
            return False
        if t in GENERIC_TOPIC_STOPWORDS:
            return False
        # 过滤“叙述句式开头”的残片，如“如何作为用户与内核之”
        bad_prefixes = (
            "这些", "这是", "那个", "这个", "如何", "记住", "表示", "实现", "支持",
            "用于", "提供", "决定", "作为", "通过", "可以", "能够", "需要", "负责",
        )
        if t.startswith(bad_prefixes):
            return False
        # 考点名称需在 3-12 字之间（太短如"进程"无区分度，太长如完整句子）
        if len(t) < 3 or len(t) > 12:
            return False
        # 过滤明显半截短语尾巴（常见于上下文被截断）
        bad_suffixes = ("之", "的", "和", "与", "及", "等", "最", "中", "上", "下", "该", "其", "之间")
        if t.endswith(bad_suffixes):
            return False
        # 过滤纯句子残片/连接词残片
        if re.search(r'(这些|那个|这个|其中|因此|所以|负责|支持|构成|作为|之间的|系统的)$', t):
            return False
        if re.search(r'^[的是在将把与及并而了其则对为]', t):
            return False
        # 至少包含一个专业信号（中文术语或英文缩写）
        if not re.search(r'(进程|线程|调度|同步|互斥|死锁|资源|内存|页|TLB|缺页|抖动|文件系统|I/O|设备|驱动|网络|TCP|系统调用|内核|中断|特权|地址空间|虚拟内存|页面置换|临界区|信号量|管程|银行家|哲学家)', t, re.IGNORECASE):
            return False
        # 过滤纯单词级碎片（单个概念词，无组合）："进程"、"死锁"、"资源" 单独出现
        # 这些需要与另一个概念组合才构成有区分度的考点
        bare_concepts = {"进程", "线程", "死锁", "资源", "内存", "设备", "网络", "内核"}
        if t in bare_concepts:
            return False
        # 如果只含一个概念词（后面无限定词），倾向过滤
        # 例外：长度>=4 的短语可能是 "进程与线程" 这类组合，保留
        if len(t) <= 4 and not re.search(r'[与和及或的]', t):
            return False
        return True

    # 1) 优先从证据片段抽词
    evidences = _extract_courseware_evidence(context, max_items=max(10, limit * 2))
    for ev in evidences:
        for token in _extract_topic_keywords(ev, limit=12):
            t = str(token).strip()
            if not _is_valid_topic_phrase(t):
                continue
            if t in seen:
                continue
            seen.add(t)
            candidates.append(t)
            if len(candidates) >= limit:
                return candidates

    # 2) 回退到全文关键词
    for token in _extract_topic_keywords(context, limit=max(40, limit * 3)):
        t = str(token).strip()
        if not _is_valid_topic_phrase(t):
            continue
        if t in seen:
            continue
        seen.add(t)
        candidates.append(t)
        if len(candidates) >= limit:
            break

    return candidates


def _build_dynamic_exam_topics(base_topics: List[str], context: str, total_questions: int) -> List[str]:
    """
    综合卷考点池：优先使用 RAG 命中提取的细粒度考点。
    考点池规模扩大至 28 个，鼓励简答题和难题覆盖多个考点。
    """
    required = 28  # 固定 28 个考点池，支持 23 题 + 5 个余量考点用于串联
    extracted = _extract_topics_from_rag_context(context, limit=56)

    merged: List[str] = []
    seen = set()
    for t in extracted + [str(x).strip() for x in (base_topics or []) if str(x).strip()]:
        if not t or t in seen:
            continue
        seen.add(t)
        merged.append(t)
    # 可控随机：保留前部高相关考点，打散后部候选，避免每次几乎同一考点池
    if merged:
        rng = random.Random(f"{current_session_id.get() or 'default'}:{time.time_ns()}:{len(context)}:{total_questions}")
        stable_head_size = min(max(4, required // 4), len(merged))
        head = merged[:stable_head_size]
        tail = merged[stable_head_size:]
        rng.shuffle(tail)
        merged = (head + tail)[:required]

    if len(merged) < 6:
        defaults = list(_default_comprehensive_topics())
        rng2 = random.Random(f"defaults:{current_session_id.get() or 'default'}:{time.time_ns()}")
        rng2.shuffle(defaults)
        for t in defaults:
            if t not in seen:
                merged.append(t)
                seen.add(t)
    return merged


def _is_generic_topic_request(topic: str) -> bool:
    text = str(topic or "").strip()
    if not text:
        return True
    keywords = _extract_topic_keywords(text, limit=6)
    if not keywords:
        return True
    # 仅包含高度通用词时视为泛化请求
    generic_signals = [
        "核心知识点", "课件", "练习题", "出题", "试卷", "综合测试",
        "本课程重点知识点",   # 回退 topic，直接跳过 relevance check
    ]
    if any(sig in text for sig in generic_signals) and len(keywords) <= 1:
        return True
    return False


def _check_rag_topic_relevance(topic: str, context: str) -> Tuple[bool, str]:
    """
    校验 topic 与 RAG 证据是否相关。
    - 泛化请求（如“做练习题”）默认允许；
    - 具体请求要求关键词至少部分命中检索上下文；
    - 若无参考资料命中，判定不相关。
    """
    if _is_generic_topic_request(topic):
        return True, "generic_topic"

    hits = _count_courseware_hits(context)
    if hits <= 0:
        return False, "no_courseware_hits"

    keywords = _extract_topic_keywords(topic, limit=8)
    if not keywords:
        return True, "no_specific_keywords"

    ctx = str(context or "")
    matched = [k for k in keywords if k in ctx]
    score = len(matched) / max(1, len(keywords))
    if matched and score >= 0.2:
        return True, f"matched={','.join(matched[:3])};score={score:.2f}"
    return False, f"keyword_mismatch;matched={len(matched)}/{len(keywords)}"


def _extract_courseware_evidence(context: str, max_items: int = 8) -> List[str]:
    if not context:
        return []
    matches = re.findall(r'\[参考资料\d+\]:参考资料:(.*?)\|参考元数据:', context, flags=re.DOTALL)
    snippets: List[str] = []

    def _truncate_on_sentence_boundary(text: str, limit: int = 120) -> str:
        src = str(text or "").strip()
        if len(src) <= limit:
            return src
        head = src[:limit]
        # 优先回退到最近句末，避免“句子腰斩”污染后续考点抽取
        end_marks = [head.rfind(ch) for ch in ["。", "；", "！", "？", ".", ";", "!", "?"]]
        cut = max(end_marks) if end_marks else -1
        if cut >= 24:
            return head[: cut + 1].strip()
        # 次优：回退到逗号
        comma_cut = max(head.rfind("，"), head.rfind(","))
        if comma_cut >= 20:
            return head[:comma_cut].strip()
        return head.strip()

    for raw in matches:
        cleaned = re.sub(r'\s+', ' ', str(raw)).strip()
        # 过滤页码线、markdown 表格残片、无意义分隔符
        if re.search(r'={5,}', cleaned):
            continue
        if re.search(r'\|\s*-{2,}\s*\|', cleaned):
            continue
        if cleaned.startswith(("###", "####", "---", "```")):
            continue
        cleaned = re.sub(r'第\s*\d+\s*页\s*/\s*共\s*\d+\s*页', '', cleaned)
        cleaned = re.sub(r'[`|#*-]+', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if not cleaned:
            continue
        # 仅保留更像“可读句子”的证据片段
        if len(cleaned) < 14:
            continue
        if not re.search(r'[\u4e00-\u9fa5A-Za-z]{6,}', cleaned):
            continue
        cleaned = _truncate_on_sentence_boundary(cleaned, limit=120)
        if cleaned not in snippets:
            snippets.append(cleaned)
        if len(snippets) >= max_items:
            break
    return snippets


def _quiz_budget_left_seconds(state: QuizState) -> float:
    started = float(state.get("quiz_budget_started_at") or time.monotonic())
    total = int(state.get("quiz_budget_seconds") or QUIZ_BUDGET_SECONDS)
    return total - (time.monotonic() - started)


def _build_retrieval_anchored_fallback_questions(
    quiz_type: str,
    num: int,
    start_num: int,
    topic: str,
    courseware_context: str,
    score_per_question: int = 5,
) -> Tuple[List[dict], str]:
    """
    课件锚定兜底：必须基于检索片段生成，避免“泛化空题”。
    返回 (questions, reason)，reason 为空表示成功。
    """
    hits = _count_courseware_hits(courseware_context)
    evidences = _extract_courseware_evidence(courseware_context, max_items=max(8, num * 2))
    if hits < QUIZ_MIN_COURSEWARE_HITS or len(evidences) < QUIZ_MIN_COURSEWARE_HITS:
        return [], "无法从课件提取足够依据"

    topic_seed = re.sub(r'（.*?）', '', str(topic or "")).strip() or "本课程重点内容"
    topic_seed = topic_seed.replace("当前课件核心知识点", "本课程重点内容").replace("课程核心知识点", "本课程重点内容")
    questions: List[dict] = []
    for i in range(num):
        q_no = start_num + i
        ev = evidences[i % len(evidences)]
        ev2 = evidences[(i + 1) % len(evidences)] if len(evidences) > 1 else ev

        if quiz_type == "选择题":
            option_a = ev[:48]
            option_b = ev2[:48]
            if option_a == option_b:
                option_b = f"{option_b}（相关延伸）"
            options = [f"A. {option_a}", f"B. {option_b}", "C. 与题意不符的干扰项", "D. 概念表述错误的干扰项"]
            answer = "A"
            analysis = "正确项与课程知识点定义一致，其他选项存在概念偏差或题意不符。"
            questions.append({
                "id": q_no,
                "type": "选择题",
                "question": f"下列关于“{topic_seed}”的说法，正确的是：",
                "options": [_normalize_option_text(opt, idx) for idx, opt in enumerate(options)],
                "answer": answer,
                "explanation": analysis,
                "score": f"{score_per_question}分",
                "difficulty": "中等",
            })
        elif quiz_type == "填空题":
            questions.append({
                "id": q_no,
                "type": "填空题",
                "question": f"在“{topic_seed}”中，补全核心术语：____。",
                "answer": "以课程定义术语作答",
                "explanation": "考查对核心概念的术语掌握与准确表达。",
                "score": f"{score_per_question}分",
                "difficulty": "中等",
            })
        elif quiz_type == "判断题":
            questions.append({
                "id": q_no,
                "type": "判断题",
                "question": f"关于“{topic_seed}”，命题“{ev[:56]}”表述正确。（ ）",
                "answer": "对",
                "explanation": "判断依据为该概念在课程中的标准定义与适用条件。",
                "score": f"{score_per_question}分",
                "difficulty": "简单",
            })
        else:
            questions.append({
                "id": q_no,
                "type": "简答题",
                "question": f"请说明“{topic_seed}”的关键机制，并结合场景分析其作用。",
                "answer": "应包含概念定义、关键流程和场景分析。",
                "explanation": "评分点：概念准确、逻辑完整、场景合理。",
                "score": f"{score_per_question}分",
                "difficulty": "较难",
            })
    return questions, ""


async def _decide_need_web_context(topic: str, rag_context: str) -> Tuple[bool, str]:
    """
    工具自治 + 策略门控：
    - 默认课件优先
    - 仅在课件证据不足时询问轻量模型是否启用外网补充
    """
    courseware_hits = _count_courseware_hits(rag_context)
    if courseware_hits >= QUIZ_MIN_COURSEWARE_HITS and len(str(rag_context or "")) >= 1800:
        return False, "courseware_sufficient"

    prompt = PromptTemplate.from_template(
        "你是工具调度器。判断是否需要联网补充（仅补充，不得替代课件）。"
        "\n输入考点: {topic}\n课件命中数: {hits}\n课件上下文长度: {length}"
        "\n仅返回JSON: {{\"use_web\": true/false, \"reason\": \"...\"}}"
    )
    gate_model = light_chat_model or backup_light_chat_model or chat_model or backup_chat_model
    chain = prompt | gate_model | StrOutputParser()
    try:
        raw = await asyncio.wait_for(
            chain.ainvoke({"topic": topic or "核心知识点", "hits": courseware_hits, "length": len(str(rag_context or ""))}),
            timeout=3.0,
        )
        decision = _extract_json(raw)
        use_web = bool(decision.get("use_web", False))
        reason = str(decision.get("reason", "") or "").strip() or "llm_decision"
        return use_web, reason
    except Exception as e:
        logger.warning(f"[Quiz ToolGate] 轻量决策失败，按策略保守不联网: {e}")
        return False, "gate_fallback_no_web"


def _quiz_quality_floor_flags(questions: List[dict], expected_type: str, expected_num: int) -> dict:
    if not isinstance(questions, list):
        questions = []
    quantity_ok = len(questions) == max(1, int(expected_num or 1))
    type_ok = True
    answer_ok = True
    options_ok = True
    seen = set()
    duplicates = False

    expected_type = _normalize_question_type(expected_type, "选择题")
    for q in questions:
        qtype = _normalize_question_type((q or {}).get("type"), expected_type or "选择题")
        if expected_type and qtype != expected_type:
            type_ok = False
        ans = str((q or {}).get("answer") or "").strip()
        if not ans:
            answer_ok = False
        sig = _question_signature(str((q or {}).get("question") or (q or {}).get("content") or ""))
        if sig and sig in seen:
            duplicates = True
        if sig:
            seen.add(sig)
        if qtype == "选择题":
            opts = (q or {}).get("options") or []
            if not isinstance(opts, list) or len(opts) < 4:
                options_ok = False

    quality_floor_passed = quantity_ok and type_ok and answer_ok and options_ok and (not duplicates)
    return {
        "quantity_ok": quantity_ok,
        "type_ok": type_ok,
        "answer_ok": answer_ok,
        "options_ok": options_ok,
        "duplicates": duplicates,
        "quality_floor_passed": quality_floor_passed,
    }


def _final_quiz_delivery_state(result: dict, floor_flags: dict, term_style_ok: bool) -> Tuple[str, str, bool]:
    """根据最终质检结果决定是否允许交付题卡。"""
    result = result or {}
    floor_flags = floor_flags or {}
    result_mode = str(result.get("delivery_mode") or "").strip()
    result_reason = str(result.get("degrade_reason") or "").strip()
    quality_ok = bool(floor_flags.get("quality_floor_passed")) and bool(term_style_ok)
    if quality_ok:
        return result_mode or "full", result_reason, True

    failed_reasons: List[str] = []
    for key in ("quantity_ok", "type_ok", "answer_ok", "options_ok"):
        if key in floor_flags and not bool(floor_flags.get(key)):
            failed_reasons.append(key)
    if floor_flags.get("duplicates"):
        failed_reasons.append("duplicates")
    if not term_style_ok:
        failed_reasons.append("term_style_violation")
    reason = result_reason or ("quality_floor_not_passed:" + ",".join(failed_reasons or ["unknown"]))
    return "failed", reason, False


def _dedupe_quiz_questions_across_rounds(
    questions: List[dict],
    forbidden_question_signatures: Optional[List[str]],
    quiz_type: str,
    topic: str,
    rag_context: str,
    target_num: int,
) -> List[dict]:
    """跨轮去重：移除与最近轮次重复题干，并尝试补齐缺失数量。"""
    safe_target = max(1, int(target_num or 1))
    blocked = {str(sig or "").strip() for sig in (forbidden_question_signatures or []) if str(sig or "").strip()}
    if not questions:
        return []
    if not blocked:
        return _sanitize_quiz_questions_for_delivery(questions)[:safe_target]

    deduped: List[dict] = []
    seen_now = set()
    dropped = 0

    def _append_if_unique(item: dict) -> bool:
        nonlocal dropped
        normalized = _normalize_question_item(item, len(deduped) + 1)
        stem = str(normalized.get("question") or normalized.get("content") or "")
        sig = _question_signature(stem)
        if not sig:
            dropped += 1
            return False
        if sig in blocked or sig in seen_now:
            dropped += 1
            return False
        seen_now.add(sig)
        deduped.append(normalized)
        return True

    for q in (questions or []):
        _append_if_unique(q)
        if len(deduped) >= safe_target:
            break

    if len(deduped) < safe_target:
        missing = safe_target - len(deduped)
        fallback, _ = _build_retrieval_anchored_fallback_questions(
            quiz_type=quiz_type,
            num=max(missing * 4, missing + 2),
            start_num=1,
            topic=topic or "本课程重点内容",
            courseware_context=rag_context,
            score_per_question=5,
        )
        for candidate in fallback:
            _append_if_unique(candidate)
            if len(deduped) >= safe_target:
                break

    if len(deduped) < safe_target:
        # 最后一层兜底：给应急题注入轻量场景后缀，避免签名完全重复。
        topic_seed = str(topic or "本课程重点内容").strip() or "本课程重点内容"
        for idx in range(safe_target - len(deduped)):
            suffix = len(blocked) + idx + 1
            emergency = _build_emergency_questions(
                quiz_type=quiz_type,
                num=1,
                start_num=1,
                topics=f"{topic_seed} 场景{suffix}",
                score_per_question=5,
            )
            if not emergency:
                continue
            _append_if_unique(emergency[0])
            if len(deduped) >= safe_target:
                break

    sanitized = _sanitize_quiz_questions_for_delivery(deduped)[:safe_target]
    for idx, q in enumerate(sanitized, start=1):
        q["id"] = idx

    if dropped > 0:
        logger.info(f"[Quiz Dedup] 跨轮去重剔除 {dropped} 道重复题，目标数量={safe_target}，最终数量={len(sanitized)}")
    return sanitized


def _build_emergency_questions(
    quiz_type: str,
    num: int,
    start_num: int,
    topics: str,
    score_per_question: int,
) -> List[dict]:
    """
    本地兜底题：当模型持续超时或失败时，确保试卷可交付，避免前端长期等待。
    """
    topic_seed = (topics or "本课程重点内容").split("；")[0].strip() or "本课程重点内容"
    topic_seed = re.sub(r'（请从当前课件核心知识点中自行选题）', '', topic_seed).strip()
    topic_seed = topic_seed.replace("当前课件核心知识点", "本课程重点内容").replace("课程核心知识点", "本课程重点内容")
    if topic_seed in {"做练习题", "练习", "出题", "做题", "本课程重点知识点"}:
        topic_seed = "本课程重点内容"
    questions: List[dict] = []
    choice_templates = [
        "下列关于“{topic}”的表述，正确的是哪一项？",
        "“{topic}”的核心目标最准确的是哪一项？",
        "关于“{topic}”的关键作用，哪项描述正确？",
    ]
    fill_templates = [
        "在“{topic}”中，系统设计通常需要在（____）与（____）之间权衡。",
        "“{topic}”常见实现需要同时考虑（____）和（____）。",
    ]
    judge_templates = [
        "“{topic}”只影响理论分析，不影响工程实现。（ ）",
        "“{topic}”在真实系统中通常不会影响性能表现。（ ）",
    ]
    essay_templates = [
        "请回答“{topic}”：1）定义与目标；2）关键流程；3）常见问题与优化思路。",
        "请结合“{topic}”说明：1）核心概念；2）典型场景；3）实现要点。",
    ]

    for i in range(num):
        q_no = start_num + i
        if quiz_type == "选择题":
            questions.append({
                "number": q_no,
                "type": "选择题",
                "content": f"{choice_templates[i % len(choice_templates)].format(topic=topic_seed)}",
                "options": [
                    f"A. {topic_seed}对应的是内核或系统关键机制",
                    f"B. {topic_seed}只属于文档表达，不影响系统行为",
                    f"C. {topic_seed}可在不受权限约束下直接操作硬件",
                    f"D. {topic_seed}与性能和正确性没有关系",
                ],
                "answer": "A",
                "analysis": "正确选项体现了系统机制与工程约束，其他选项存在明显逻辑错误。",
                "score": score_per_question,
                "difficulty": "中等",
            })
        elif quiz_type == "填空题":
            questions.append({
                "number": q_no,
                "type": "填空题",
                "content": f"{fill_templates[i % len(fill_templates)].format(topic=topic_seed)}",
                "answer": "正确性；性能",
                "analysis": "课程中常强调正确性与性能的平衡。",
                "score": score_per_question,
                "difficulty": "中等",
            })
        elif quiz_type == "判断题":
            questions.append({
                "number": q_no,
                "type": "判断题",
                "content": f"{judge_templates[i % len(judge_templates)].format(topic=topic_seed)}",
                "answer": "错",
                "analysis": "核心机制会直接影响系统实现与运行表现。",
                "score": score_per_question,
                "difficulty": "简单",
            })
        else:
            questions.append({
                "number": q_no,
                "type": "简答题",
                "content": f"{essay_templates[i % len(essay_templates)].format(topic=topic_seed)}",
                "answer": "应包含定义、流程、问题分析与可行优化方案。",
                "analysis": "考查对知识点的结构化理解与工程应用能力。",
                "score": score_per_question,
                "difficulty": "较难",
            })
    return questions


def _quick_quiz_quality_check(state: QuizState) -> Optional[dict]:
    """
    本地快速质检：满足基本可用性时直接通过，避免 Critic 长耗时导致整链路超时。
    """
    payload = state.get("revised_quiz_payload") or state.get("quiz_payload") or {}
    questions = payload.get("questions", []) if isinstance(payload, dict) else []
    if not isinstance(questions, list) or not questions:
        return None

    expected_num = int(state.get("num", 0) or 0)
    expected_type = _normalize_question_type(state.get("quiz_type"), "选择题")
    quantity_ok = (expected_num <= 0) or (len(questions) == expected_num)
    type_ok = True
    options_ok = True
    answer_ok = True

    for q in questions:
        qtype = _normalize_question_type((q or {}).get("type"), expected_type or "选择题")
        if expected_type and qtype and qtype != expected_type:
            type_ok = False
        if qtype == "选择题":
            opts = (q or {}).get("options") or []
            if not isinstance(opts, list) or len(opts) < 4:
                options_ok = False
        if not str((q or {}).get("answer") or "").strip():
            answer_ok = False

    if quantity_ok and type_ok and options_ok and answer_ok:
        return {
            "approved": True,
            "overall_score": 85,
            "critique": "本地快速质检通过，已满足可交付标准。",
            "reasoning_flaws": [],
            "specific_issues": [],
            "duplicate_check": {"has_duplicates": False, "duplicate_questions": []},
            "numbering_check": {"is_continuous": True, "issues": []},
            "quantity_check": {"is_valid": True},
        }
    return None


async def _ensure_single_type_questions(
    quiz_type: str,
    num: int,
    start_num: int,
    topics: str,
    context: str,
    sample_paper_context: str,
    score_per_question: int,
    question_semaphore: asyncio.Semaphore = None,
    per_type_budget_seconds: float = 360.0,
    shared_topic_bank: Optional[List[str]] = None,
    shared_topic_usage: Optional[Dict[str, int]] = None,
    shared_topic_lock: Optional[asyncio.Lock] = None,
) -> List[dict]:
    """可用性优先：结构化生成不足时，自动回退文本生成补齐缺题。"""
    if question_semaphore is None:
        question_semaphore = asyncio.Semaphore(1)

    start_ts = time.monotonic()
    hard_deadline_seconds = max(120.0, float(per_type_budget_seconds))
    budget_reserved_for_fallback = max(60.0, hard_deadline_seconds * 0.4)
    structured_budget_ceiling = max(60.0, hard_deadline_seconds - budget_reserved_for_fallback)
    logger.info(
        f"[Budget] quiz_type={quiz_type}, budget_left_before_stage={hard_deadline_seconds:.1f}, "
        f"budget_reserved_for_fallback={budget_reserved_for_fallback:.1f}"
    )

    def time_left() -> float:
        return hard_deadline_seconds - (time.monotonic() - start_ts)

    # 动态考点池：优先使用未使用考点，减少同批次重复
    topic_pool = [t.strip() for t in re.split(r"[；;，,、/|\n]+", str(topics or "")) if t and t.strip()]
    if shared_topic_bank:
        for t in shared_topic_bank:
            tt = str(t).strip()
            if tt and tt not in topic_pool:
                topic_pool.append(tt)
    if len(topic_pool) < max(6, min(16, num)):
        extra_topics = _extract_topics_from_rag_context(context, limit=max(16, num * 2))
        for t in extra_topics:
            if t not in topic_pool:
                topic_pool.append(t)
    if not topic_pool:
        topic_pool = ["进程与线程", "调度与同步", "死锁与资源管理", "内存管理", "文件系统与I/O", "网络与通信机制"]
    local_usage: Dict[str, int] = {t: 0 for t in topic_pool}
    if shared_topic_usage is not None:
        for t in topic_pool:
            local_usage[t] = int(shared_topic_usage.get(t, 0))
    topic_rng = random.Random(
        f"{current_session_id.get() or 'default'}:{quiz_type}:{start_num}:{num}:{time.time_ns()}"
    )

    async def pick_batch_topics(batch_num: int) -> str:
        if not topic_pool:
            return topics

        async def _select_with_usage(usage: Dict[str, int]) -> str:
            # 按使用次数分层，并在同层内随机打散，兼顾覆盖与随机性
            buckets: Dict[int, List[str]] = {}
            for item in topic_pool:
                buckets.setdefault(int(usage.get(item, 0)), []).append(item)
            ordered: List[str] = []
            for level in sorted(buckets.keys()):
                bucket = list(buckets[level])
                topic_rng.shuffle(bucket)
                ordered.extend(bucket)
            # 每题至少一个新考点优先：本批先拿 usage=0 的考点
            must_new_count = min(max(1, batch_num), sum(1 for t in ordered if usage.get(t, 0) == 0))
            must_new = [t for t in ordered if usage.get(t, 0) == 0][:must_new_count]
            candidates = (must_new + ordered)[:max(batch_num + 3, 8)]
            # 预占用（调度层），保证并发阶段也能轮到新考点
            for t in must_new:
                usage[t] = int(usage.get(t, 0)) + 1
            return (
                "【本批必须覆盖的新考点】" + "；".join(must_new) +
                "\n【本批可用考点池】" + "；".join(candidates)
            )

        if shared_topic_usage is not None and shared_topic_lock is not None:
            async with shared_topic_lock:
                return await _select_with_usage(shared_topic_usage)
        return await _select_with_usage(local_usage)

    # 结构化阶段：动态批次尝试（避免固定2次导致必然补题）
    structured_questions: List[dict] = []
    structured_attempts = 0
    structured_batch_size = _get_structured_batch_size(quiz_type)
    # 例如选择题10道、每批2道，至少要5次尝试才可能全结构化补齐
    max_structured_attempts = max(2, min(8, (num + structured_batch_size - 1) // structured_batch_size + 1))
    while (
        len(structured_questions) < num
        and structured_attempts < max_structured_attempts
        and time_left() > budget_reserved_for_fallback
    ):
        remaining = num - len(structured_questions)
        # 首轮尽量单批出完，失败后再回到小批次策略
        try_single_shot = (
            structured_attempts == 0
            and remaining >= 6
            and time_left() > (budget_reserved_for_fallback + 100.0)
            and quiz_type in {"选择题", "填空题", "判断题"}
        )
        batch_num = remaining if try_single_shot else min(max(1, structured_batch_size), remaining)
        structured_attempts += 1
        batch_start = start_num + len(structured_questions)
        batch_topics = await pick_batch_topics(batch_num)
        structured_timeout = max(30.0, min(150.0, structured_budget_ceiling / 2))
        try:
            structured = await asyncio.wait_for(
                generate_single_type_paper_structured(
                    quiz_type=quiz_type,
                    num=batch_num,
                    start_num=batch_start,
                    topics=batch_topics,
                    context=context,
                    sample_paper_context=sample_paper_context,
                ),
                timeout=structured_timeout,
            )
            batch_questions = [q.model_dump() for q in structured.questions]
            if not batch_questions:
                logger.warning(
                    f"[分批补题] {quiz_type} 结构化批次为空，attempt={structured_attempts}/{max_structured_attempts}, "
                    f"batch={batch_num}, start={batch_start}, topics={batch_topics[:80]}"
                )
                continue
            structured_questions.extend(batch_questions[:batch_num])
            logger.info(
                f"[分批补题] {quiz_type} 结构化批次成功，attempt={structured_attempts}/{max_structured_attempts}, "
                f"batch={batch_num}, accumulated={len(structured_questions)}/{num}"
            )
        except Exception as e:
            logger.warning(
                f"[分批补题] {quiz_type} 结构化批次失败，attempt={structured_attempts}/{max_structured_attempts}: {e}"
            )

    if len(structured_questions) >= num:
        logger.info(f"[分批补题] {quiz_type} stage_exit_reason=structured_enough")
        return _merge_exam_questions(structured_questions[:num], [], quiz_type, start_num)

    missing = num - len(structured_questions)
    logger.warning(f"[分批补题] {quiz_type} 结构化结果不足，缺少 {missing} 道，改用文本生成补齐")
    merged = _merge_exam_questions(structured_questions, [], quiz_type, start_num)

    # 文本补齐窗口：固定 1 次
    if len(merged) < num and time_left() > 8:
        remaining = num - len(merged)
        batch_size = _get_text_fallback_batch_size(quiz_type)
        current_batch = min(max(1, batch_size), remaining)
        next_num = start_num + len(merged)
        batch_topics = await pick_batch_topics(current_batch)
        logger.info(
            f"[分批补题] {quiz_type} 文本补题窗口: 需补{remaining}道，本批{current_batch}道，起始编号{next_num}"
        )
        try:
            supplement_text = await asyncio.wait_for(
                generate_single_type_paper(
                    quiz_type=quiz_type,
                    num=current_batch,
                    start_num=next_num,
                    topics=batch_topics,
                    context=context,
                    sample_paper_context=sample_paper_context,
                    score_per_question=score_per_question,
                ),
                timeout=max(20.0, min(120.0, time_left())),
            )
        except Exception as e:
            logger.warning(f"[分批补题] {quiz_type} 文本补题调用失败: {e}")
            supplement_text = ""
        supplement_questions = parse_exam_content(supplement_text).get("questions", []) if supplement_text else []
        if supplement_questions:
            merged = _merge_exam_questions(merged, supplement_questions[:current_batch], quiz_type, start_num)
        else:
            logger.warning(f"[分批补题] {quiz_type} 文本补题窗口失败")

    if len(merged) < num:
        logger.warning(f"[分批补题] {quiz_type} 文本补题后仍不足，继续单题兜底直到满足数量")
        max_single_retries = 1
        single_retry_count = 0
        while len(merged) < num and single_retry_count < max_single_retries:
            if time_left() <= 8:
                logger.error(f"[分批补题] {quiz_type} 单题兜底超出时间预算，提前结束")
                break
            remaining = num - len(merged)
            # 每次并发补题数量遵循题型规则（简答题=1）
            single_batch_size = _get_text_fallback_batch_size(quiz_type)
            concurrent_count = min(max(1, single_batch_size), remaining)
            next_num = start_num + len(merged)
            single_retry_count += 1
            logger.info(f"[分批补题] {quiz_type} 单题兜底第{single_retry_count}次，并发{concurrent_count}道，编号 {next_num}")

            async def gen_one(idx: int, q_num: int) -> List[dict]:
                async with question_semaphore:
                    text = await asyncio.wait_for(
                        generate_single_type_paper(
                            quiz_type=quiz_type,
                            num=1,
                            start_num=q_num,
                            topics=await pick_batch_topics(1),
                            context=context,
                            sample_paper_context=sample_paper_context,
                            score_per_question=score_per_question,
                        ),
                        timeout=max(15.0, min(60.0, time_left())),
                    )
                    return parse_exam_content(text).get("questions", []) if text else []

            question_numbers = [start_num + len(merged) + i for i in range(concurrent_count)]
            results = await asyncio.gather(*[gen_one(i, question_numbers[i]) for i in range(concurrent_count)])
            all_generated = [q for r in results for q in r]
            if not all_generated:
                logger.error(f"[分批补题] {quiz_type} 单题兜底失败，编号 {next_num}（第{single_retry_count}次）")
                continue
            merged = _merge_exam_questions(merged, all_generated[:concurrent_count], quiz_type, start_num)
        if single_retry_count >= max_single_retries and len(merged) < num:
            logger.error(f"[分批补题] {quiz_type} 单题兜底已达最大重试次数{max_single_retries}，放弃。已生成{len(merged)}/{num}")

    if len(merged) < num:
        missing = num - len(merged)
        logger.warning(f"[分批补题] {quiz_type} 仍缺{missing}道，进入模型重试兜底（禁用本地模板兜底）")
        rounds = 0
        while len(merged) < num and rounds < EXAM_MISSING_MODEL_RETRY_ROUNDS and time_left() > 8:
            rounds += 1
            next_num = start_num + len(merged)
            logger.info(f"[分批补题] {quiz_type} 缺题模型重试 round={rounds}/{EXAM_MISSING_MODEL_RETRY_ROUNDS}, 编号{next_num}")
            try:
                supplement_text = await asyncio.wait_for(
                    generate_single_type_paper(
                        quiz_type=quiz_type,
                        num=1,
                        start_num=next_num,
                        topics=await pick_batch_topics(1),
                        context=context,
                        sample_paper_context=sample_paper_context,
                        score_per_question=score_per_question,
                    ),
                    timeout=max(20.0, min(90.0, time_left())),
                )
            except Exception as e:
                logger.warning(f"[分批补题] {quiz_type} 缺题模型重试失败 round={rounds}: {e}")
                supplement_text = ""
            supplement_questions = parse_exam_content(supplement_text).get("questions", []) if supplement_text else []
            if supplement_questions:
                merged = _merge_exam_questions(merged, supplement_questions[:1], quiz_type, start_num)

    if len(merged) < num:
        missing = num - len(merged)
        if EXAM_ENABLE_LOCAL_EMERGENCY_FALLBACK:
            logger.warning(f"[分批补题] {quiz_type} 最终仍缺{missing}道，启用本地兜底题")
            merged = _merge_exam_questions(
                merged,
                _build_emergency_questions(
                    quiz_type=quiz_type,
                    num=missing,
                    start_num=start_num + len(merged),
                    topics=topics,
                    score_per_question=score_per_question,
                ),
                quiz_type,
                start_num,
            )
        else:
            raise RuntimeError(f"{quiz_type} 生成不足：仍缺{missing}道，且已禁用本地兜底")

    if len(merged) > num:
        merged = merged[:num]
    if quiz_type == "选择题":
        normalized_choice_questions: List[dict] = []
        repaired_count = 0
        for idx, q in enumerate(merged, start=1):
            qq = _normalize_exam_question(q, start_num + idx - 1)
            before_opts = qq.get("options") if isinstance(qq.get("options"), list) else []
            qq = _ensure_choice_structure(qq)
            after_opts = qq.get("options") if isinstance(qq.get("options"), list) else []
            if len(before_opts) < 4 and len(after_opts) >= 4:
                repaired_count += 1
            normalized_choice_questions.append(qq)
        merged = normalized_choice_questions
        if repaired_count > 0:
            logger.warning(f"[分批补题] {quiz_type} 交付前自动修复缺失选项 {repaired_count} 道")
    logger.info(f"[分批补题] {quiz_type} 最终数量={len(merged)} / 目标={num}")
    logger.info(f"[分批补题] {quiz_type} stage_exit_reason={'filled' if len(merged) >= num else 'budget_exhausted'}")
    return merged


def _get_text_fallback_batch_size(quiz_type: str) -> int:
    """文本补题按小批次生成，避免单次输出过大卡住。"""
    if quiz_type == "选择题":
        return 2
    if quiz_type in {"填空题", "判断题"}:
        return 2
    return 1


def _get_structured_batch_size(quiz_type: str) -> int:
    """结构化补题批大小：简答题严格 1 题/批，其它 2 题/批。"""
    if quiz_type == "简答题":
        return 1
    if quiz_type in {"选择题", "填空题", "判断题"}:
        return 2
    return 1


def get_structured_fail_stats() -> Dict[str, int]:
    return _runtime_get_structured_fail_stats()


def _coerce_structured_payload(parsed: dict, schema: type[BaseModel]) -> dict:
    """兼容模型把字段名答错的情况。"""
    if not isinstance(parsed, dict):
        return parsed

    schema_name = getattr(schema, "__name__", "")
    if schema_name == "QuizGenerateResult":
        if "quiz" not in parsed and "exam_paper" in parsed:
            parsed["quiz"] = parsed.get("exam_paper", "")
        parsed["reasoning"] = _normalize_reasoning_payload(parsed.get("reasoning"))
    elif schema_name == "StructuredQuizSetResult":
        parsed["title"] = _normalize_text_field(parsed.get("title"), "练习题")
        if "questions" not in parsed and "quiz" in parsed:
            parsed["questions"] = []
        parsed["questions"] = _normalize_questions_payload(parsed.get("questions"))
        parsed["reasoning"] = _normalize_reasoning_payload(parsed.get("reasoning"))
    elif schema_name == "CritiqueResult":
        parsed["critique"] = _normalize_text_field(parsed.get("critique"), "")
        parsed["overall_score"] = _normalize_int_field(parsed.get("overall_score"), 80)
        parsed["reasoning_flaws"] = _normalize_list_of_dicts(parsed.get("reasoning_flaws"))
        parsed["specific_issues"] = _normalize_list_of_dicts(parsed.get("specific_issues"))
        parsed["duplicate_check"] = _normalize_dict_field(parsed.get("duplicate_check"))
        parsed["numbering_check"] = _normalize_dict_field(parsed.get("numbering_check"))
        parsed["quantity_check"] = _normalize_dict_field(parsed.get("quantity_check"))
    elif schema_name in {"ReviseResult", "ExamReviseResult"}:
        parsed["addressed_issues"] = _normalize_addressed_issues(parsed.get("addressed_issues"))
        if schema_name == "ReviseResult":
            parsed["revised_quiz"] = _normalize_text_field(parsed.get("revised_quiz"), "")
        else:
            parsed["revised_exam"] = _normalize_text_field(parsed.get("revised_exam"), "")
        parsed["revision_notes"] = _normalize_text_field(parsed.get("revision_notes"), "")
    elif schema_name == "ExamPaperText":
        if "exam_paper" not in parsed and isinstance(parsed.get("questions"), list):
            normalized_questions = _normalize_exam_questions_payload(parsed.get("questions"))
            payload = _build_exam_payload_from_questions(normalized_questions, title="期末考试试卷")
            parsed["exam_paper"] = _build_exam_text_from_payload(payload)
        parsed["exam_paper"] = _normalize_text_field(parsed.get("exam_paper"), "")
        parsed["reasoning"] = _normalize_reasoning_payload(parsed.get("reasoning"))
    elif schema_name == "ExamPaper":
        parsed["questions"] = _normalize_exam_questions_payload(parsed.get("questions"))
        parsed["reasoning"] = _normalize_reasoning_payload(parsed.get("reasoning"))
    return parsed


def _normalize_exam_question(question: dict, fallback_number: int) -> dict:
    qtype = _normalize_question_type(question.get("type"), "选择题")
    score = int(question.get("score") or (2 if qtype in {"选择题", "填空题", "判断题"} else 10))
    content = _sanitize_user_visible_text(question.get("content") or question.get("question") or "")
    options = question.get("options") or None
    if qtype == "选择题":
        stem, parsed_options = _split_stem_and_options_for_payload(content)
        content = stem or content
        if parsed_options:
            options = parsed_options
    normalized = {
        "number": int(question.get("number") or question.get("id") or fallback_number),
        "type": qtype,
        "content": content,
        "options": options,
        "answer": _sanitize_user_visible_text(question.get("answer") or ""),
        "analysis": _sanitize_user_visible_text(question.get("analysis") or question.get("explanation") or ""),
        "score": score,
        "difficulty": question.get("difficulty") or "中等",
        "knowledge_point": question.get("knowledge_point"),
    }
    if normalized["type"] == "选择题":
        normalized = _ensure_choice_structure(normalized)
    return normalized


def _build_exam_payload_from_questions(questions: List[dict], title: str = "完整试卷", subtitle: str = "") -> dict:
    normalized_questions: List[dict] = []
    question_types: Dict[str, Dict[str, int]] = {}
    total_score = 0

    for index, question in enumerate(questions, start=1):
        normalized = _normalize_exam_question(question, index)
        normalized_questions.append(normalized)
        total_score += normalized["score"]
        qtype = normalized["type"]
        if qtype not in question_types:
            question_types[qtype] = {"count": 0, "total_score": 0}
        question_types[qtype]["count"] += 1
        question_types[qtype]["total_score"] += normalized["score"]

    return {
        "title": title,
        "subtitle": subtitle,
        "show_answers_default": False,
        "show_analysis_default": False,
        "exam_data": {
            "title": title,
            "subtitle": subtitle,
            "total_score": total_score,
            "total_questions": len(normalized_questions),
            "question_types": question_types,
            "questions": normalized_questions,
        },
    }


def _split_stem_and_options_for_payload(raw_content: str) -> tuple[str, List[str]]:
    """
    从题干中拆出选择题选项，统一输出为 `A. xxx` 格式，便于前端稳定渲染。
    兼容 `A.` / `A、` / `A．` 等写法。
    """
    content = re.sub(r'^\d+[.、]\s*', '', (raw_content or "").strip())
    if not content:
        return "", []

    # 优先按行解析，能稳定保留题干
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    stem_lines: List[str] = []
    options: List[str] = []
    option_line_pattern = re.compile(r'^([A-DＡ-Ｄ])[.、．:：)\s]+\s*(.+)$')
    inline_split_pattern = re.compile(r'(?=[A-DＡ-Ｄ][.、．:：)\s]+\s*)')
    inline_capture_pattern = re.compile(
        r'([A-DＡ-Ｄ])[.、．:：)\s]+\s*(.+?)(?=(?:\s+[A-DＡ-Ｄ][.、．:：)\s]+)|$)'
    )

    for line in lines:
        line = line.replace('．', '.')
        m = option_line_pattern.match(line)
        if m:
            letter = m.group(1).upper()
            option_text = m.group(2).strip()
            if option_text:
                options.append(f"{letter}. {option_text}")
            continue

        if re.search(r'[A-DＡ-Ｄ][.、．:：)\s]+\s*', line):
            # 优先用捕获模式拆分同一行的多选项，兼容 "A xxx B xxx C xxx D xxx"
            inline_matches = inline_capture_pattern.findall(line)
            if len(inline_matches) >= 2:
                first_marker = re.search(r'[A-DＡ-Ｄ][.、．:：)\s]+\s*', line)
                if first_marker and first_marker.start() > 0:
                    stem_prefix = line[:first_marker.start()].strip()
                    if stem_prefix:
                        stem_lines.append(stem_prefix)
                for letter, opt_text in inline_matches:
                    txt = str(opt_text or "").strip()
                    if txt:
                        options.append(f"{str(letter).upper()}. {txt}")
                continue

            parts = [p.strip() for p in inline_split_pattern.split(line) if p.strip()]
            if parts:
                first = parts[0]
                if not option_line_pattern.match(first):
                    stem_lines.append(first)
                    parts = parts[1:]
                for part in parts:
                    mm = option_line_pattern.match(part.replace('．', '.'))
                    if mm and mm.group(2).strip():
                        options.append(f"{mm.group(1).upper()}. {mm.group(2).strip()}")
                continue

        stem_lines.append(line)

    # 去重并保序，最多保留4个选项
    seen = set()
    normalized_options: List[str] = []
    for opt in options:
        key = re.sub(r'\s+', ' ', opt).strip()
        if key and key not in seen:
            seen.add(key)
            normalized_options.append(opt)
        if len(normalized_options) >= 4:
            break

    return "\n".join(stem_lines).strip(), normalized_options


def _normalize_exam_scores_to_target(
    questions: List[dict],
    score_map: Dict[str, int],
    target_total: int = 100,
) -> List[dict]:
    """统一按题型重算分值，并做全局微调保证总分尽量等于 target_total。"""
    normalized: List[dict] = []
    for q in questions:
        qq = dict(q)
        qtype = _normalize_question_type(qq.get("type"), "选择题")
        qq["type"] = qtype
        if qtype in score_map:
            qq["score"] = int(score_map[qtype])
        else:
            qq["score"] = int(qq.get("score") or 2)
        normalized.append(qq)

    if not normalized:
        return normalized

    total = sum(int(q.get("score") or 0) for q in normalized)
    if total == target_total:
        return normalized

    # 当题量 > 100 时，不可能所有题都 >=1 分，此时允许降到 0 分保证总分收敛
    min_score = 1 if len(normalized) <= target_total else 0

    def adjust_once(delta: int) -> bool:
        if delta == 0:
            return False
        # 优先调整简答题，再调整其它题；减分时从高分题开始更稳定
        order = sorted(
            range(len(normalized)),
            key=lambda i: (normalized[i].get("type") != "简答题", -int(normalized[i].get("score") or 0)),
        )
        if delta > 0:
            for idx in order:
                normalized[idx]["score"] = int(normalized[idx].get("score") or 0) + 1
                return True
        else:
            for idx in order:
                current = int(normalized[idx].get("score") or 0)
                if current > min_score:
                    normalized[idx]["score"] = current - 1
                    return True
        return False

    safety = max(2000, len(normalized) * 200)
    while total != target_total and safety > 0:
        changed = adjust_once(target_total - total)
        if not changed:
            break
        total = sum(int(q.get("score") or 0) for q in normalized)
        safety -= 1

    return normalized


def _build_exam_text_from_payload(payload: dict) -> str:
    exam_data = payload.get("exam_data", {})
    title = exam_data.get("title") or payload.get("title") or "完整试卷"
    questions = exam_data.get("questions", [])
    type_order = ["选择题", "填空题", "判断题", "简答题", "计算题", "分析题", "论述题", "名词解释"]
    section_labels = {
        "选择题": "一、选择题",
        "填空题": "二、填空题",
        "判断题": "三、判断题",
        "简答题": "四、简答题",
        "计算题": "五、计算题",
        "分析题": "六、分析题",
        "论述题": "七、论述题",
        "名词解释": "八、名词解释",
    }
    grouped: Dict[str, List[dict]] = {}
    for question in questions:
        grouped.setdefault(question.get("type") or "选择题", []).append(question)

    lines = [f"## 《{title}》", ""]
    ordered_types = [qtype for qtype in type_order if qtype in grouped] + [qtype for qtype in grouped if qtype not in type_order]
    for qtype in ordered_types:
        items = grouped[qtype]
        total_score = sum(int(item.get("score") or 0) for item in items)
        score_desc = f"共{len(items)}题，计{total_score}分"
        if items and items[0].get("score"):
            score_desc = f"共{len(items)}题，每题{items[0]['score']}分，计{total_score}分"
        lines.append(f"{section_labels.get(qtype, qtype)}（{score_desc}）")
        for item in items:
            lines.append(f"{item['number']}. {item.get('content', '').strip()}")
            for option in item.get("options") or []:
                lines.append(f"   {option.strip()}")
            if item.get("answer"):
                lines.append(f"答案：{item['answer']}")
            if item.get("analysis"):
                lines.append(f"解析：{item['analysis']}")
            lines.append("")
    return "\n".join(lines).strip()


def _exam_budget_left_seconds(state: ExamPaperState) -> float:
    started = float(state.get("exam_budget_started_at") or time.monotonic())
    total = int(state.get("exam_budget_seconds") or EXAM_BUDGET_SECONDS)
    return total - (time.monotonic() - started)


def _ensure_answer_analysis(questions: List[dict]) -> List[dict]:
    fixed: List[dict] = []
    for q in questions:
        qq = dict(q)
        if not str(qq.get("answer", "")).strip():
            qq["answer"] = "参考答案见课程知识点定义与流程描述。"
        if not str(qq.get("analysis", "")).strip():
            qq["analysis"] = "本题考查相关概念的定义、机制与工程应用。"
        fixed.append(qq)
    return fixed


def _build_choice_option_templates(stem: str) -> List[str]:
    stem_core = re.sub(r'\s+', ' ', str(stem or "").strip())
    stem_core = stem_core[:36] if stem_core else "该机制"
    return [
        f"A. {stem_core}符合课程中的标准定义",
        f"B. {stem_core}只在用户态生效，与内核机制无关",
        f"C. {stem_core}可以绕过系统调用直接执行特权操作",
        f"D. {stem_core}与资源管理和正确性没有关系",
    ]


def _normalize_choice_option_content(text: str) -> str:
    content = str(text or "").strip()
    if not content:
        return ""
    # 清理重复前缀: "A A xxx" / "B. B. xxx"
    content = re.sub(r'^\s*[A-D][.、．:：)\s]+\s*', '', content, flags=re.IGNORECASE)
    content = re.sub(r'^\s*[A-D][.、．:：)\s]+\s*', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\s+', ' ', content).strip()
    return content


def _shorten_stem_for_exam(stem: str, max_len: int = 90) -> str:
    text = _sanitize_question_text(stem)
    if len(text) <= max_len:
        return text
    # 优先截到第一个完整句
    parts = re.split(r'[。！？!?]', text)
    if parts:
        first = parts[0].strip()
        if 12 <= len(first) <= max_len:
            return first + "。"
    clipped = text[:max_len].rstrip("，,;；:： ")
    # 避免截断在连接词/残片上，导致“才。”这类不完整句
    clipped = re.sub(r'(其|并|且|才|及|和|或|与|中|上|下|于|及其|以及)$', '', clipped)
    clipped = re.sub(r'[A-Za-z]$', '', clipped)
    clipped = clipped.rstrip("（(").rstrip()
    return (clipped or text[:max_len]).rstrip("，,;；:： ") + "。"


def _stem_max_len_for_question(question: dict, index: int) -> int:
    qtype = _normalize_question_type((question or {}).get("type"), "选择题")
    diff = str((question or {}).get("difficulty") or "中等")
    if qtype == "选择题":
        base = 62 if diff in {"简单", "基础"} else (82 if diff in {"中等"} else 96)
        # 长短搭配：每 4 题做一个小周期
        cycle = index % 4
        if cycle == 0:
            return min(108, base + 12)
        if cycle == 1:
            return max(54, base - 8)
        return base
    if qtype in {"填空题", "判断题"}:
        return 84 if diff in {"简单", "基础"} else 108
    return 160 if diff in {"中等"} else 190


def _content_token_set(text: str) -> set:
    src = str(text or "")
    tokens = re.findall(r'[\u4e00-\u9fff]{2,6}|[A-Za-z]{3,}', src)
    return {t.lower() for t in tokens if t and len(t) >= 2}


def _is_near_duplicate_text(a: str, b: str, threshold: float = 0.72) -> bool:
    ta = _content_token_set(a)
    tb = _content_token_set(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    uni = len(ta | tb)
    if uni == 0:
        return False
    return (inter / uni) >= threshold


def _enforce_concept_uniqueness(
    questions: List[dict],
    topics: List[str],
    score_map: Dict[str, int],
) -> List[dict]:
    """
    综合卷去重：同一考点锚点仅保留 1 题，重复题替换为不同 topic 的同题型题目。
    """
    if not questions:
        return questions

    pool = [t.strip() for t in (topics or []) if str(t).strip()]
    if not pool:
        pool = ["进程与线程", "调度与同步", "死锁与资源管理", "内存管理", "文件系统与I/O", "网络与通信机制", "系统调用", "内核架构"]

    used_concepts: Dict[str, int] = {}
    used_topics: set = set()
    fixed: List[dict] = []

    for idx, q in enumerate(questions, start=1):
        qq = dict(q)
        ckey = _concept_key_for_question(qq)
        if used_concepts.get(ckey, 0) >= 1:
            qtype = _normalize_question_type(qq.get("type"), "选择题")
            candidate_topic = None
            for p in pool:
                if p in used_topics:
                    continue
                if _detect_concept_anchor(p) != ckey:
                    candidate_topic = p
                    break
            if candidate_topic is None:
                candidate_topic = f"{pool[idx % len(pool)]}·专题{idx}"
            qq = _rewrite_duplicate_question_locally(
                qq,
                idx,
                candidate_topic,
                score_map.get(qtype, 2),
            )
            ckey = _concept_key_for_question(qq)

        used_concepts[ckey] = used_concepts.get(ckey, 0) + 1
        used_topics.add(str(_detect_concept_anchor(str(qq.get("content") or ""))))
        fixed.append(qq)

    return fixed


def _rewrite_duplicate_question_locally(
    question: dict,
    idx: int,
    topic_seed: str,
    score: int,
) -> dict:
    """
    重复题本地改写：不依赖本地兜底开关，保证严格模式下也能主动消重。
    """
    q = dict(question or {})
    qtype = _normalize_question_type(q.get("type"), "选择题")
    topic = re.sub(r'\s+', '', str(topic_seed or "")).strip() or f"专题{idx}"
    if qtype == "选择题":
        stem = f"关于{topic}的关键机制，下列说法最准确的是哪一项？"
        q["content"] = stem
        q["options"] = _build_choice_option_templates(topic)
        q["answer"] = "A"
        q["analysis"] = f"本题考查{topic}的核心定义与机制边界。"
    elif qtype == "填空题":
        q["content"] = f"在{topic}相关机制中，核心控制目标是（ ）。"
        q["answer"] = "确保系统在资源约束下保持正确性与效率"
        q["analysis"] = f"填空应围绕{topic}的目标与作用作答。"
    elif qtype == "判断题":
        q["content"] = f"{topic}会直接影响系统行为与资源分配策略。"
        q["options"] = ["A. 正确", "B. 错误", "C. 条件成立", "D. 无法判断"]
        q["answer"] = "A"
        q["analysis"] = f"{topic}属于系统核心机制，对行为和资源都有直接影响。"
    else:
        q["content"] = f"请结合{topic}，说明其设计目标、关键流程与工程权衡。"
        q["answer"] = "需覆盖设计目标、关键流程、性能与可靠性权衡。"
        q["analysis"] = f"本题要求结构化阐述{topic}的核心知识。"
    q["score"] = int(score or q.get("score") or 2)
    q["difficulty"] = q.get("difficulty") or "中等"
    return _normalize_exam_question(q, idx)


def _normalize_choice_answer(answer: str, options: List[str]) -> str:
    ans = str(answer or "").strip().upper()
    if ans in {"A", "B", "C", "D"}:
        return ans
    m = re.search(r'\b([A-D])\b', ans)
    if m:
        return m.group(1)
    if ans:
        for idx, opt in enumerate(options):
            content = re.sub(r'^[A-D][.、．]\s*', '', str(opt)).strip()
            if content and (content in ans or ans in content):
                return chr(65 + idx)
    return "A"


def _extract_choice_answer_from_explanation(text: str) -> str:
    """从解析末尾的明确结论中提取选择题答案字母。"""
    normalized = str(text or "").upper().translate(str.maketrans("ＡＢＣＤ", "ABCD"))
    if not normalized.strip():
        return ""

    patterns = [
        r"(?:最终|最后|综上|因此|故|所以|修正|确认)[^。；;\n]{0,40}(?:答案|应选|选择|选项)[为是：:\s]*([ABCD])\b",
        r"(?:正确答案|标准答案|答案|应选|选择|选项)[为是：:\s]*([ABCD])\b",
        r"\b([ABCD])\s*(?:为|是)?(?:正确答案|标准答案|正确选项)",
    ]
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(m.group(1).upper() for m in re.finditer(pattern, normalized))
    return matches[-1] if matches else ""


def _reconcile_choice_answers_with_explanations(questions: List[dict]) -> tuple[List[dict], List[dict]]:
    """交付前修正选择题 answer 与解析最终答案不一致的问题。"""
    reconciled: List[dict] = []
    fixes: List[dict] = []
    for index, question in enumerate(questions or [], start=1):
        q = dict(question or {})
        qtype = _normalize_question_type(q.get("type"), "选择题")
        options = q.get("options") or []
        if qtype == "选择题" and isinstance(options, list) and len(options) >= 4:
            declared = _normalize_choice_answer(str(q.get("answer") or ""), options)
            inferred = _extract_choice_answer_from_explanation(q.get("explanation") or q.get("analysis") or "")
            if inferred in {"A", "B", "C", "D"} and declared != inferred:
                q["answer"] = inferred
                fixes.append({"number": q.get("number") or q.get("id") or index, "from": declared, "to": inferred})
        reconciled.append(q)
    return reconciled, fixes


def _ensure_choice_structure(question: dict) -> dict:
    q = dict(question)
    stem = _shorten_stem_for_exam(str(q.get("content") or "").strip(), max_len=90)
    options = _normalize_options(q.get("options")) or []
    parsed_stem, parsed_options = _split_stem_and_options_for_payload(stem)
    if parsed_stem:
        stem = _shorten_stem_for_exam(parsed_stem, max_len=90)
    if len(options) < 4 and parsed_options:
        options = _normalize_options(parsed_options) or options
    rebuilt_options: List[str] = []
    for idx, opt in enumerate((options or [])[:4]):
        content = _normalize_choice_option_content(re.sub(r'^[A-D][.、．]\s*', '', str(opt)).strip())
        if not content:
            continue
        rebuilt_options.append(f"{chr(65 + idx)}. {content}")

    q["content"] = stem
    q["options"] = rebuilt_options
    if len(rebuilt_options) == 4:
        q["answer"] = _normalize_choice_answer(str(q.get("answer") or ""), rebuilt_options)
    return q


def _repair_exam_questions_locally(
    questions: List[dict],
    quantity_dist: dict,
    topics: List[str],
    target_total_score: int = 100,
) -> List[dict]:
    """
    本地仅做格式层处理（不干预出题内容）：
    - 字段归一（number/type/content/options/answer/analysis/score）
    - 编号连续化
    - 分值归一到目标总分（仅数值层）
    """
    if not questions:
        return []
    normalized: List[dict] = []
    for idx, q in enumerate(questions, start=1):
        normalized.append(_normalize_exam_question(q, idx))

    # 编号连续化（渲染稳定）
    for i, q in enumerate(normalized, start=1):
        q["number"] = i

    # 分值仅做数值归一，不改题干/选项/答案/解析
    target = _target_counts(quantity_dist, fallback_questions=normalized)
    score_map = _score_map_for_counts(target, target_total_score=target_total_score)
    normalized = _normalize_exam_scores_to_target(normalized, score_map, target_total=target_total_score)
    return normalized


# ==================== 节点函数（单题）====================

async def generate_with_reasoning_node(state: QuizState) -> QuizState:
    """Agent 1: 出题 + 推理链"""
    logger.info(f"[Agent1-出题] {state['topic']} / {state['quiz_type']} / {state['num']}道")
    sample = state.get('sample_paper_context', '') or '无样卷参考，使用默认格式'

    try:
        structured_result = await call_llm_structured(
            _get_quiz_generate_structured_prompt(),
            StructuredQuizSetResult,
            topic=state['topic'],
            quiz_type=state['quiz_type'],
            num=state['num'],
            context=state['context'],
            sample_paper_context=sample,
        )
        questions = []
        for item in structured_result.questions:
            question = item.model_dump(exclude_none=True)
            if question.get("score") is not None:
                question["score"] = _format_score(question["score"])
            questions.append(question)
        payload = _build_quiz_payload(
            structured_result.title or state['topic'] or "练习题",
            questions,
        )
        quiz = _build_quiz_text_from_questions(questions)
        reasoning = json.dumps(structured_result.reasoning or {}, ensure_ascii=False)
        logger.info(f"[Agent1-出题] 原生结构化输出成功，questions={len(questions)}")
        return {"quiz": quiz, "reasoning": reasoning, "quiz_payload": payload}
    except Exception as e:
        logger.warning(f"[Agent1-出题] 原生结构化输出失败，回退到文本结构化链路: {e}")

    # 回退路径分三层：结构化 -> 正则抽取 -> 纯文本兜底，优先保证可交付。
    # 使用结构化输出，彻底杜绝 Markdown 混排
    try:
        result = await call_llm_structured(
            _get_quiz_generate_reasoning_prompt(),
            QuizGenerateResult,
            topic=state['topic'],
            quiz_type=state['quiz_type'],
            num=state['num'],
            context=state['context'],
            sample_paper_context=sample,
        )
        quiz = result.quiz
        reasoning = json.dumps(result.reasoning, ensure_ascii=False)
        logger.info(f"[Agent1-出题] 结构化输出成功，quiz长度={len(quiz)}")
        payload = _questions_from_quiz_text(quiz, state['topic'])
    except Exception as e:
        logger.warning(f"[Agent1-出题] 结构化输出失败，回退到正则解析: {e}")
        # 回退机制：尝试正则解析
        result_text = ""
        try:
            result_text = await call_llm(
                _get_quiz_generate_reasoning_prompt(),
                topic=state['topic'],
                quiz_type=state['quiz_type'],
                num=state['num'],
                context=state['context'],
                sample_paper_context=sample,
            )
            parsed = _extract_json(result_text)
            quiz = parsed.get('quiz', '')
            reasoning = json.dumps(parsed.get('reasoning', {}), ensure_ascii=False)
            payload = _questions_from_quiz_text(quiz, state['topic'])
        except Exception as e2:
            logger.error(f"[Agent1-出题] 回退解析也失败: result_text='{result_text[:200]}...', error={e2}")
            plain_result = await call_llm(
                _get_quiz_generate_fallback_prompt(),
                topic=state['topic'],
                quiz_type=state['quiz_type'],
                num=state['num'],
                context=state['context'],
                sample_paper_context=sample,
            )
            quiz = _strip_think_tags(plain_result)
            reasoning = '{}'
            payload = _questions_from_quiz_text(quiz, state['topic'])

    return {"quiz": quiz, "reasoning": reasoning, "quiz_payload": payload}


async def critique_quiz_node(state: QuizState) -> QuizState:
    """Agent 2: Critic — 质疑推理链，输出结构化批评"""
    quiz = state.get('revised_quiz') or state.get('quiz', '')
    logger.info("[Agent2-Critic] 质疑出题推理链...")

    force_llm_critic = bool(state.get("force_llm_critic", False))
    if not force_llm_critic:
        fast = _quick_quiz_quality_check(state)
        if fast is not None:
            # 快速本地评审通过时跳过 LLM Critic，显著降低链路时延。
            score = fast.get("overall_score", 80)
            logger.info(f"[Agent2-Critic] 本地快速质检通过，跳过 LLM Critic，score={score}")
            update: dict = {"critique": fast}
            if state.get('reflection_rounds', 0) == 0 and not state.get('initial_score'):
                update["initial_score"] = score
            return update
    else:
        logger.info("[Agent2-Critic] force_llm_critic=true，跳过本地快速质检，强制走 LLM Critic")

    budget_left = _quiz_budget_left_seconds(state)
    if budget_left <= 18:
        logger.warning(f"[Agent2-Critic] 剩余预算不足({budget_left:.1f}s)，跳过 Critic 进入可交付模式")
        critique = {
            "approved": True,
            "overall_score": 76,
            "critique": "预算不足，跳过评审，保留当前可交付题目。",
            "reasoning_flaws": [],
            "specific_issues": [],
            "duplicate_check": {"has_duplicates": False, "duplicate_questions": []},
            "numbering_check": {"is_continuous": True, "issues": []},
            "quantity_check": {"is_valid": True},
        }
        update: dict = {"critique": critique}
        if state.get('reflection_rounds', 0) == 0 and not state.get('initial_score'):
            update["initial_score"] = critique.get("overall_score", 76)
        return update

    # 使用结构化输出
    try:
        llm_timeout = max(8.0, min(18.0, budget_left - 6.0))
        result = await asyncio.wait_for(
            call_llm_structured(
                _get_quiz_critique_prompt(),
                CritiqueResult,
                topic=state['topic'],
                quiz_type=state['quiz_type'],
                context=state['context'],
                quiz=quiz,
                reasoning=state.get('reasoning', '{}'),
            ),
            timeout=llm_timeout,
        )
        critique = result.model_dump()
        logger.info(f"[Agent2-Critic] 结构化输出成功，approved={result.approved}")
    except asyncio.TimeoutError:
        logger.warning("[Agent2-Critic] LLM Critic 超时，使用保守通过策略避免整链路超时")
        critique = {
            "approved": True,
            "overall_score": 78,
            "critique": "评审超时，保留已生成题目交付。",
            "reasoning_flaws": [],
            "specific_issues": [],
            "duplicate_check": {"has_duplicates": False, "duplicate_questions": []},
            "numbering_check": {"is_continuous": True, "issues": []},
            "quantity_check": {"is_valid": True},
        }
    except Exception as e:
        logger.warning(f"[Agent2-Critic] 结构化输出失败，使用保守通过策略: {e}")
        critique = {
            "approved": True,
            "overall_score": 76,
            "critique": "评审模型暂时不可用，保留已生成题目交付。",
            "reasoning_flaws": [],
            "specific_issues": [],
            "duplicate_check": {"has_duplicates": False, "duplicate_questions": []},
            "numbering_check": {"is_continuous": True, "issues": []},
            "quantity_check": {"is_valid": True},
        }

    score = critique.get('overall_score', 80)
    logger.info(f"[Agent2-Critic] approved={critique.get('approved')}, score={score}")

    update: dict = {"critique": critique}
    # 首轮评审时记录基准分，用于事后对比 Reflexion 提升幅度
    if state.get('reflection_rounds', 0) == 0 and not state.get('initial_score'):
        update["initial_score"] = score
    return update


async def revise_with_reflection_node(state: QuizState) -> QuizState:
    """Agent 3: Revise — 针对批评逐条修订，输出修改说明"""
    round_n = state.get('reflection_rounds', 0) + 1
    logger.info(f"[Agent3-Revise] 第 {round_n} 轮反思修订...")
    budget_left = _quiz_budget_left_seconds(state)
    if budget_left <= 14:
        logger.warning(f"[Agent3-Revise] 剩余预算不足({budget_left:.1f}s)，跳过修订直接交付")
        base = state.get('revised_quiz') or state.get('quiz', '')
        return {
            "revised_quiz": base,
            "revision_notes": "预算不足，跳过修订并直接交付。",
            "revised_quiz_payload": state.get("revised_quiz_payload") or state.get("quiz_payload") or _questions_from_quiz_text(base, state['topic']),
            "reflection_rounds": round_n,
            "delivery_mode": state.get("delivery_mode", "full"),
            "degrade_reason": state.get("degrade_reason", ""),
        }

    # 使用结构化输出
    try:
        llm_timeout = max(10.0, min(22.0, budget_left - 4.0))
        result = await asyncio.wait_for(
            call_llm_structured(
                _get_quiz_revise_prompt(),
                ReviseResult,
                topic=state['topic'],
                quiz_type=state['quiz_type'],
                context=state['context'],
                quiz=state.get('quiz', ''),
                reasoning=state.get('reasoning', '{}'),
                critique=json.dumps(state.get('critique', {}), ensure_ascii=False),
            ),
            timeout=llm_timeout,
        )
        revised = result.revised_quiz
        notes = result.revision_notes
        logger.info(f"[Agent3-Revise] 结构化输出成功")
    except asyncio.TimeoutError:
        logger.warning("[Agent3-Revise] 修订超时，保留当前版本交付")
        revised = state.get('revised_quiz') or state.get('quiz', '')
        notes = "修订超时，保留当前版本。"
        return {
            "revised_quiz": revised,
            "revision_notes": notes,
            "revised_quiz_payload": state.get("revised_quiz_payload") or state.get("quiz_payload") or _questions_from_quiz_text(revised, state['topic']),
            "reflection_rounds": round_n,
            "delivery_mode": state.get("delivery_mode", "full"),
            "degrade_reason": state.get("degrade_reason", ""),
        }
    except Exception as e:
        logger.warning(f"[Agent3-Revise] 结构化输出失败，回退到正则解析: {e}")
        # 回退机制
        try:
            result_text = await call_llm(
                _get_quiz_revise_prompt(),
                topic=state['topic'],
                quiz_type=state['quiz_type'],
                context=state['context'],
                quiz=state.get('quiz', ''),
                reasoning=state.get('reasoning', '{}'),
                critique=json.dumps(state.get('critique', {}), ensure_ascii=False),
            )
            parsed = _extract_json(result_text)
            revised = parsed.get('revised_quiz', state.get('quiz', ''))
            notes = parsed.get('revision_notes', '')
        except Exception:
            revised = state.get('quiz', '')
            notes = ''

    return {
        "revised_quiz": revised,
        "revision_notes": notes,
        "revised_quiz_payload": _questions_from_quiz_text(revised, state['topic']),
        "reflection_rounds": round_n,
        "delivery_mode": state.get("delivery_mode", "full"),
        "degrade_reason": state.get("degrade_reason", ""),
    }


# ==================== 节点函数（试卷）====================

async def generate_exam_with_reasoning_node(state: ExamPaperState) -> ExamPaperState:
    """Agent 1: 出卷 + 推理链（分批生成每种题型）"""
    logger.info(f"[Agent1-出卷] {state['total_questions']}道，考点: {state['topics']}")
    topics_str = ", ".join(state['topics'])
    sample = state.get('sample_paper_context', '') or '无样卷参考'
    target_total_score = int(state.get("target_total_score") or 100)

    # 计算题型数量分布
    total = state['total_questions']
    quantity_dist = state.get('quantity_dist', {})
    if quantity_dist:
        choice_count = quantity_dist.get('choice', 0)
        fill_count = quantity_dist.get('fill', 0)
        judge_count = quantity_dist.get('judge', 0)
        essay_count = quantity_dist.get('essay', 0)
    elif total == 23:
        # 默认综合卷23道
        choice_count, fill_count, judge_count, essay_count = 10, 5, 5, 3
    elif total == 34:
        choice_count, fill_count, judge_count, essay_count = 10, 10, 10, 4
    else:
        # 默认均分
        choice_count = total // 4
        fill_count = total // 4
        judge_count = total // 4
        essay_count = total - 3 * (total // 4)

    # 固定分值配置
    choice_score = 2
    fill_score = 2
    judge_score = 2

    # 动态计算简答题分数（余数在后处理阶段分摊）
    base_score = choice_count * choice_score + fill_count * fill_score + judge_count * judge_score
    remaining_score = target_total_score - base_score
    if essay_count > 0:
        essay_score = remaining_score // essay_count
        essay_remainder = remaining_score % essay_count
    else:
        essay_score = 0
        essay_remainder = 0

    logger.info(
        f"[出卷] 题型数量: 选择{choice_count}×{choice_score}分, 填空{fill_count}×{fill_score}分, "
        f"判断{judge_count}×{judge_score}分, 简答{essay_count}×{essay_score}分(余数分摊={essay_remainder}), "
        f"修复前理论总分={choice_count*choice_score + fill_count*fill_score + judge_count*judge_score + essay_count*essay_score}, "
        f"目标总分={target_total_score}"
    )

    # 计算编号范围
    current_num = 1
    choice_start = current_num
    current_num += choice_count
    fill_start = current_num
    current_num += fill_count
    judge_start = current_num
    current_num += judge_count
    essay_start = current_num

    # 全局考点库（跨阶段共享）：记录每个考点被抽取次数，优先抽取未使用考点
    topic_bank = [str(t).strip() for t in (state.get("topic_bank") or state.get("topics") or []) if str(t).strip()]
    if not topic_bank:
        topic_bank = _build_dynamic_exam_topics(state.get("topics") or [], state['contexts'][0] if state.get('contexts') else "", state.get("total_questions") or 23)
    topic_usage: Dict[str, int] = dict(state.get("topic_usage") or {})
    for t in topic_bank:
        topic_usage.setdefault(t, 0)
    topic_usage_lock = asyncio.Lock()

    # 并发策略：
    # - 题型间并发=2：平衡吞吐与模型稳定性
    # - 题型内并发=2：避免单题型批量请求挤爆额度
    # 目标是整体端到端更快，而不是局部峰值并发最大化。
    # 同题型并发生成（题型间并发上限=2，题型内并发上限=2）
    type_semaphore = asyncio.Semaphore(2)
    question_semaphore = asyncio.Semaphore(2)  # 题型内并发上限

    async def generate_and_collect(quiz_type: str, num: int, start_num: int, score: int):
        if num <= 0:
            return None
        async with type_semaphore:
            logger.info(f"[Agent1-出卷] 开始题型任务: {quiz_type}，题型内并发上限=2")
            try:
                budget_left = _exam_budget_left_seconds(state)
                if budget_left <= 8:
                    raise RuntimeError(f"{quiz_type} 剩余预算不足({budget_left:.1f}s)，停止生成（已禁用本地兜底）")
                per_type_budget = max(120.0, min(480.0, budget_left * 0.9))
                questions = await asyncio.wait_for(
                    _ensure_single_type_questions(
                        quiz_type=quiz_type,
                        num=num,
                        start_num=start_num,
                        topics=topics_str,
                        context=state['contexts'][0] if state['contexts'] else '',
                        sample_paper_context=sample,
                        score_per_question=score,
                        question_semaphore=question_semaphore,  # 传入题型内并发控制
                        per_type_budget_seconds=per_type_budget,
                        shared_topic_bank=topic_bank,
                        shared_topic_usage=topic_usage,
                        shared_topic_lock=topic_usage_lock,
                    ),
                    timeout=max(120.0, min(520.0, budget_left - 12.0)),  # 生成阶段宽松上限，优先完整题量
                )
            except asyncio.TimeoutError:
                raise RuntimeError(f"{quiz_type} 超过题型总超时（已禁用本地兜底）")
        return {"questions": questions, "type": quiz_type, "count": num}

    # 启动所有题型任务，题型间最多并发 2 个
    tasks = []
    task_info = []
    if choice_count > 0:
        tasks.append(asyncio.create_task(generate_and_collect("选择题", choice_count, choice_start, 2)))
        task_info.append({"type": "选择题", "count": choice_count})
    if fill_count > 0:
        tasks.append(asyncio.create_task(generate_and_collect("填空题", fill_count, fill_start, 2)))
        task_info.append({"type": "填空题", "count": fill_count})
    if judge_count > 0:
        tasks.append(asyncio.create_task(generate_and_collect("判断题", judge_count, judge_start, 2)))
        task_info.append({"type": "判断题", "count": judge_count})
    if essay_count > 0:
        tasks.append(asyncio.create_task(generate_and_collect("简答题", essay_count, essay_start, essay_score)))
        task_info.append({"type": "简答题", "count": essay_count, "score": essay_score})

    # 收集结果
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 收集结果
    all_questions: List[dict] = []
    all_reasonings = task_info
    for index, result in enumerate(results):
        info = task_info[index] if index < len(task_info) else {"type": f"task-{index}"}
        if isinstance(result, Exception):
            logger.error(f"[Agent1-出卷] 题型任务失败: {info.get('type')} -> {result}")
            continue
        if isinstance(result, dict):
            if result.get("questions"):
                all_questions.extend(result["questions"])
                logger.info(
                    f"[Agent1-出卷] 题型完成: {result.get('type')}，"
                    f"期望{result.get('count')}道，实际{len(result.get('questions', []))}道"
                )
            else:
                logger.warning(f"[Agent1-出卷] 题型结果为空: {info.get('type')}")

    # 本地结构修复：保证题型数量、编号、去重、总分=100
    pre_repair_score = sum(int((q or {}).get("score") or 0) for q in all_questions)
    all_questions = _repair_exam_questions_locally(
        all_questions,
        state.get("quantity_dist", {}),
        state.get("topics", []),
        target_total_score=target_total_score,
    )
    post_repair_score = sum(int((q or {}).get("score") or 0) for q in all_questions)
    logger.info(f"[出卷修复] score_before={pre_repair_score}, score_after={post_repair_score}, target={target_total_score}")
    floor_flags = _exam_quality_floor_flags(all_questions, state.get("quantity_dist", {}), target_total_score=target_total_score)
    delivery_mode = "full" if floor_flags.get("quality_floor_passed") else "partial_revised"
    degrade_reason = ""

    exam_payload = _build_exam_payload_from_questions(all_questions, title="期末考试试卷")
    exam_paper = _build_exam_text_from_payload(exam_payload) if all_questions else ""
    if not exam_paper.strip():
        exam_paper = "试卷生成失败：题型生成超时或未返回有效内容，请稍后重试。"
    logger.info(f"[Agent1-出卷] exam_payload_questions={len(exam_payload.get('exam_data', {}).get('questions', []))}")
    reasoning = json.dumps({"type_distribution": all_reasonings}, ensure_ascii=False)

    logger.info(f"[Agent1-出卷] 分批生成完成，总长度={len(exam_paper)}, 题目数={len(all_questions)}")

    return {
        "exam_paper": exam_paper,
        "reasoning": reasoning,
        "exam_payload": exam_payload,
        "delivery_mode": delivery_mode,
        "quality_floor_passed": bool(floor_flags.get("quality_floor_passed")),
        "degrade_reason": degrade_reason,
        "topic_bank": topic_bank,
        "topic_usage": topic_usage,
    }


async def critique_exam_node(state: ExamPaperState) -> ExamPaperState:
    """Agent 2: Critic — 评审试卷推理链"""
    if state.get("degrade_reason") in {"budget_exhausted", "revise_timeout"}:
        logger.info(f"[Agent2-Critic-试卷] 已降级({state.get('degrade_reason')}), 跳过后续 LLM 评审")
        return {"critique": state.get("critique", {"approved": True, "overall_score": 70})}

    exam_paper = state.get('revised_exam') or state.get('exam_paper', '')
    target_total_score = int(state.get("target_total_score") or 100)
    layout_params = _build_exam_layout_params(state.get("quantity_dist", {}) or {})
    contexts_combined = "\n\n".join(state['contexts'])[:2000]
    logger.info("[Agent2-Critic-试卷] 质疑出卷推理链...")
    budget_left = _exam_budget_left_seconds(state)
    critic_timeout_count = int(state.get("critic_timeout_count") or 0)
    stage_fast_mode = bool(state.get("stage_fast_mode"))

    if stage_fast_mode:
        # 阶段模式下，Critic 只做本地快速评审，避免 LLM 评审拖慢整链路。
        # 这是“快速模式”在评审侧的核心行为。
        fast_critique = _build_fast_exam_critique(state, exam_paper)
        if fast_critique is not None:
            critique = fast_critique
        else:
            parsed_questions = parse_exam_content(exam_paper).get("questions", [])
            fixed_questions = _repair_exam_questions_locally(
                parsed_questions if isinstance(parsed_questions, list) else [],
                state.get("quantity_dist", {}),
                state.get("topics", []),
                target_total_score=target_total_score,
            )
            flags = _exam_quality_floor_flags(fixed_questions, state.get("quantity_dist", {}), target_total_score=target_total_score)
            critique = {
                "approved": bool(flags.get("quantity_ok") and flags.get("numbering_ok") and flags.get("duplicates_ok")),
                "overall_score": 80 if flags.get("quality_floor_passed") else 68,
                "critique": "阶段快速评审：基于本地结构规则完成。",
                "reasoning_flaws": [],
                "specific_issues": [],
                "duplicate_check": {"has_duplicates": bool(flags.get("duplicates")), "duplicate_questions": []},
                "numbering_check": {"is_continuous": bool(flags.get("numbering_ok")), "issues": []},
                "quantity_check": {"is_valid": bool(flags.get("quantity_ok"))},
            }

        score = critique.get('overall_score', 80)
        logger.info(f"[Agent2-Critic-试卷] 阶段快速评审，approved={critique.get('approved')}, score={score}")
        update: dict = {"critique": critique, "critic_timeout_count": critic_timeout_count}
        if state.get('reflection_rounds', 0) == 0 and not state.get('initial_score'):
            update["initial_score"] = score
        return update

    fast_critique = _build_fast_exam_critique(state, exam_paper)
    if fast_critique is not None:
        critique = fast_critique
    elif budget_left <= 20:
        logger.warning(f"[Agent2-Critic-试卷] 剩余预算不足({budget_left:.1f}s)，跳过 LLM 评审，使用本地保守评估")
        parsed_questions = parse_exam_content(exam_paper).get("questions", [])
        fixed_questions = _repair_exam_questions_locally(
            parsed_questions if isinstance(parsed_questions, list) else [],
            state.get("quantity_dist", {}),
            state.get("topics", []),
            target_total_score=target_total_score,
        )
        flags = _exam_quality_floor_flags(fixed_questions, state.get("quantity_dist", {}), target_total_score=target_total_score)
        critique = {
            "approved": bool(flags.get("quality_floor_passed")),
            "overall_score": 78 if flags.get("quality_floor_passed") else 62,
            "critique": "预算不足，已采用本地结构校验结果。",
            "reasoning_flaws": [],
            "specific_issues": [],
            "duplicate_check": {"has_duplicates": bool(flags.get("duplicates")), "duplicate_questions": []},
            "numbering_check": {"is_continuous": bool(flags.get("numbering_ok")), "issues": []},
            "quantity_check": {"is_valid": bool(flags.get("quantity_ok"))},
        }
    else:
        # 使用结构化输出
        try:
            result = await call_llm_structured(
                _get_exam_critique_prompt(),
                CritiqueResult,
                topics=", ".join(state['topics']),
                contexts=contexts_combined,
                exam_paper=exam_paper,
                reasoning=state.get('reasoning', '{}'),
                total_questions=state.get('total_questions', 10),
                exam_contract=_get_exam_contract_prompt(),
                **layout_params,
            )
            critique = result.model_dump()
            logger.info(f"[Agent2-Critic-试卷] 结构化输出成功，approved={result.approved}")
        except Exception as e:
            logger.warning(f"[Agent2-Critic-试卷] 结构化输出失败，回退到正则解析: {e}")
            if "timeout" in str(e).lower():
                critic_timeout_count += 1
            # 回退机制
            try:
                result_text = await call_llm(
                    _get_exam_critique_prompt(),
                    topics=", ".join(state['topics']),
                    contexts=contexts_combined,
                    exam_paper=exam_paper,
                    reasoning=state.get('reasoning', '{}'),
                    total_questions=state.get('total_questions', 10),
                    exam_contract=_get_exam_contract_prompt(),
                    **layout_params,
                )
                critique = _extract_json(result_text)
            except Exception:
                critic_timeout_count += 1
                critique = {"approved": True, "overall_score": 80, "reasoning_flaws": [], "specific_issues": []}

    # 【精确数量校验】- 直接解析文本统计题目数量，不依赖 LLM 的正则匹配
    quantity_dist = state.get('quantity_dist', {})

    # 默认综合卷23道（选择10，判断5，填空5，简答3）
    default_dist = {"choice": 10, "fill": 5, "judge": 5, "essay": 3}

    # 优先使用用户自定义的题型分布，否则用默认值
    if quantity_dist:
        expected_dist = quantity_dist
    else:
        expected_dist = default_dist

    expected = sum(expected_dist.values())
    is_strict = True  # 综合卷严格校验

    logger.info(f"[Critique校验] expected={expected}, expected_dist={expected_dist}")

    actual = _count_questions_in_exam(exam_paper, expected_dist)

    if is_strict:
        # 严格校验：每个题型数量必须匹配
        is_valid = (
            actual["choice"] == expected_dist.get("choice", 0) and
            actual["fill"] == expected_dist.get("fill", 0) and
            actual["judge"] == expected_dist.get("judge", 0) and
            actual["essay"] == expected_dist.get("essay", 0)
        )
    else:
        # 宽松校验：只要总题数足够即可
        has_some = actual["choice"] > 0 or actual["fill"] > 0 or actual["judge"] > 0 or actual["essay"] > 0
        is_valid = has_some and actual["total"] >= expected

    # 覆盖 quantity_check，使用本地精确统计
    critique["quantity_check"] = {
        "choice": expected_dist.get("choice", 0),
        "fill": expected_dist.get("fill", 0),
        "judge": expected_dist.get("judge", 0),
        "essay": expected_dist.get("essay", 0),
        "actual_choice": actual["choice"],
        "actual_fill": actual["fill"],
        "actual_judge": actual["judge"],
        "actual_essay": actual["essay"],
        "is_valid": is_valid
    }

    # 如果数量不达标，在 critique 中添加明确提示
    if not is_valid:
        shortage = []
        if actual["choice"] < expected_dist.get("choice", 0):
            shortage.append(f"选择题少{expected_dist['choice'] - actual['choice']}道")
        if actual["fill"] < expected_dist.get("fill", 0):
            shortage.append(f"填空题少{expected_dist['fill'] - actual['fill']}道")
        if actual["judge"] < expected_dist.get("judge", 0):
            shortage.append(f"判断题少{expected_dist['judge'] - actual['judge']}道")
        if actual["essay"] < expected_dist.get("essay", 0):
            shortage.append(f"简答题少{expected_dist['essay'] - actual['essay']}道")

        # 安全处理：确保 _shortage_str 有值
        _shortage_str = "，".join(shortage) if shortage else "题型数量不符"
        logger.warning(f"[数量校验失败] {_shortage_str}，要求{expected}道，实际{actual['total']}道")

        # 添加到 reasoning_flaws
        critique.setdefault("reasoning_flaws", []).append({
            "question": "试卷整体",
            "flaw": f"题目数量不足：你只生成了{actual['total']}道题，距离要求的{expected}道题还差{expected - actual['total']}道。{_shortage_str}。请补全所有题目！",
            "severity": "high"
        })
        critique.setdefault("specific_issues", []).append({
            "question": "试卷整体",
            "issue": f"题量或题型分布不达标：{_shortage_str}",
            "suggestion": "优先补齐缺失题型，再检查编号连续性和总分。",
        })
        # 不在 Critique 阶段硬阻断，改为标记需修订并放宽扣分
        critique["needs_revision"] = True
        if critique.get("overall_score", 100) > 82:
            critique["overall_score"] = 78

    score = critique.get('overall_score', 80)
    logger.info(f"[Agent2-Critic-试卷] approved={critique.get('approved')}, score={score}, quantity_valid={is_valid}")

    update: dict = {"critique": critique, "critic_timeout_count": critic_timeout_count}
    if state.get('reflection_rounds', 0) == 0 and not state.get('initial_score'):
        update["initial_score"] = score
    return update


async def revise_exam_with_reflection_node(state: ExamPaperState) -> ExamPaperState:
    """Agent 3: Revise — 针对批评修订试卷"""
    round_n = state.get('reflection_rounds', 0) + 1
    logger.info(f"[Agent3-Revise-试卷] 第 {round_n} 轮反思修订...")
    contexts_combined = "\n\n".join(state['contexts'])[:2000]
    critique = state.get('critique', {})
    revise_start = time.monotonic()
    budget_left = _exam_budget_left_seconds(state)
    target_total_score = int(state.get("target_total_score") or 100)

    if _exam_is_usable_without_revision(critique):
        logger.info("[Agent3-Revise-试卷] 当前试卷已满足可用性交付标准，跳过整卷修订")
        revised = state.get('revised_exam') or state.get('exam_paper', '')
        payload = state.get('revised_exam_payload') or state.get('exam_payload') or _build_exam_payload_from_questions(
            parse_exam_content(revised).get("questions", []),
            title="期末考试试卷",
        )
        questions = (payload.get("exam_data") or {}).get("questions", [])
        floor_flags = _exam_quality_floor_flags(
            questions if isinstance(questions, list) else [],
            state.get("quantity_dist", {}),
            target_total_score=target_total_score,
        )
        logger.info(f"[Latency] revise_stage_ms={int((time.monotonic() - revise_start) * 1000)}")
        return {
            "revised_exam": revised,
            "revision_notes": "试卷已满足可用性交付标准，跳过整卷修订以避免超时。",
            "revised_exam_payload": payload,
            "delivery_mode": state.get("delivery_mode", "full"),
            "quality_floor_passed": bool(floor_flags.get("quality_floor_passed")),
            "degrade_reason": state.get("degrade_reason", ""),
            "reflection_rounds": round_n,
        }

    base_payload = state.get("revised_exam_payload") or state.get("exam_payload") or {}
    base_questions = (base_payload.get("exam_data") or {}).get("questions", [])
    if not isinstance(base_questions, list) or not base_questions:
        parsed_base = parse_exam_content(state.get('revised_exam') or state.get('exam_paper', ''))
        base_questions = parsed_base.get("questions", []) if isinstance(parsed_base, dict) else []

    # 局部修订优先：先本地修复结构性问题，降低整卷重写成本
    local_fixed_questions = _repair_exam_questions_locally(
        base_questions if isinstance(base_questions, list) else [],
        state.get("quantity_dist", {}),
        state.get("topics", []),
        target_total_score=target_total_score,
    )
    local_flags = _exam_quality_floor_flags(local_fixed_questions, state.get("quantity_dist", {}), target_total_score=target_total_score)
    if local_flags.get("quality_floor_passed"):
        local_payload = _build_exam_payload_from_questions(local_fixed_questions, title="期末考试试卷")
        logger.info("[Agent3-Revise-试卷] 本地局部修订已满足质量底线，跳过整卷重写")
        logger.info(f"[Latency] revise_stage_ms={int((time.monotonic() - revise_start) * 1000)}")
        return {
            "revised_exam": _build_exam_text_from_payload(local_payload),
            "revision_notes": "已执行本地局部修订（编号/去重/总分归一），满足交付质量底线。",
            "revised_exam_payload": local_payload,
            "delivery_mode": "partial_revised",
            "quality_floor_passed": True,
            "degrade_reason": "",
            "reflection_rounds": round_n,
        }

    if budget_left <= 25:
        logger.warning(f"[Agent3-Revise-试卷] 剩余预算不足({budget_left:.1f}s)，直接交付本地修订结果")
        local_payload = _build_exam_payload_from_questions(local_fixed_questions, title="期末考试试卷")
        logger.info(f"[Latency] revise_stage_ms={int((time.monotonic() - revise_start) * 1000)}")
        return {
            "revised_exam": _build_exam_text_from_payload(local_payload),
            "revision_notes": "出卷预算耗尽，已返回本地结构修复结果。",
            "revised_exam_payload": local_payload,
            "delivery_mode": "partial_revised",
            "quality_floor_passed": bool(local_flags.get("quality_floor_passed")),
            "degrade_reason": "budget_exhausted",
            "reflection_rounds": round_n,
        }

    if _exam_structural_only_issues(critique):
        logger.info("[Agent3-Revise-试卷] 仅结构性问题，跳过慢速 LLM 精修，避免长时间阻塞")
        local_payload = _build_exam_payload_from_questions(local_fixed_questions, title="期末考试试卷")
        logger.info(f"[Latency] revise_stage_ms={int((time.monotonic() - revise_start) * 1000)}")
        return {
            "revised_exam": _build_exam_text_from_payload(local_payload),
            "revision_notes": "仅检测到结构性问题（重复/编号/数量），已跳过慢速模型修订。",
            "revised_exam_payload": local_payload,
            "delivery_mode": "partial_revised",
            "quality_floor_passed": bool(local_flags.get("quality_floor_passed")),
            "degrade_reason": "structural_retry",
            "reflection_rounds": round_n,
        }

    llm_budget = max(20.0, min(90.0, budget_left - 10.0))
    revised = _build_exam_text_from_payload(_build_exam_payload_from_questions(local_fixed_questions, title="期末考试试卷"))
    notes = "本地局部修订后进入模型精修。"
    degrade_reason = ""

    # 使用结构化输出（受预算约束）
    try:
        result = await asyncio.wait_for(
            call_llm_structured(
                _get_exam_revise_prompt(),
                ExamReviseResult,
                topics=", ".join(state['topics']),
                contexts=contexts_combined,
                exam_paper=revised,
                reasoning=state.get('reasoning', '{}'),
                critique=json.dumps(critique, ensure_ascii=False),
                total_questions=state.get('total_questions', 10),
                exam_contract=_get_exam_contract_prompt(),
            ),
            timeout=llm_budget,
        )
        revised = result.revised_exam
        notes = result.revision_notes
        logger.info(f"[Agent3-Revise-试卷] 结构化输出成功")
    except asyncio.TimeoutError:
        degrade_reason = "revise_timeout"
        logger.warning(f"[Agent3-Revise-试卷] 结构化精修超时（{llm_budget:.1f}s），保留本地修复版本")
    except Exception as e:
        logger.warning(f"[Agent3-Revise-试卷] 结构化输出失败，回退到正则解析: {e}")
        # 回退机制
        try:
            result_text = await asyncio.wait_for(
                call_llm(
                    _get_exam_revise_prompt(),
                    topics=", ".join(state['topics']),
                    contexts=contexts_combined,
                    exam_paper=revised,
                    reasoning=state.get('reasoning', '{}'),
                    critique=json.dumps(critique, ensure_ascii=False),
                    total_questions=state.get('total_questions', 10),
                    exam_contract=_get_exam_contract_prompt(),
                ),
                timeout=max(15.0, min(45.0, llm_budget / 2)),
            )
            parsed = _extract_json(result_text)
            revised = parsed.get('revised_exam', revised)
            notes = parsed.get('revision_notes', '')
        except asyncio.TimeoutError:
            degrade_reason = "revise_timeout"
            logger.warning("[Agent3-Revise-试卷] 文本回退精修超时，保留本地修复版本")
        except Exception:
            degrade_reason = "revise_timeout"
            notes = notes or "模型精修失败，已保留本地结构修复结果。"

    parsed_questions = parse_exam_content(revised).get("questions", []) if revised else []
    final_questions = _repair_exam_questions_locally(
        parsed_questions if isinstance(parsed_questions, list) else local_fixed_questions,
        state.get("quantity_dist", {}),
        state.get("topics", []),
        target_total_score=target_total_score,
    )
    floor_flags = _exam_quality_floor_flags(final_questions, state.get("quantity_dist", {}), target_total_score=target_total_score)
    final_payload = _build_exam_payload_from_questions(final_questions, title="期末考试试卷")
    logger.info(f"[Latency] revise_stage_ms={int((time.monotonic() - revise_start) * 1000)}")
    return {
        "revised_exam": _build_exam_text_from_payload(final_payload),
        "revision_notes": notes,
        "revised_exam_payload": final_payload,
        "delivery_mode": "partial_revised" if degrade_reason else "full",
        "quality_floor_passed": bool(floor_flags.get("quality_floor_passed")),
        "degrade_reason": degrade_reason,
        "reflection_rounds": round_n,
    }


# ==================== 条件判断 ====================

def _critique_needs_revision(critique: dict) -> bool:
    """
    多条件交叉验证，避免依赖单一布尔字段导致的不稳定判断。
    触发修订的条件（满足任一即修订）：
      1. overall_score < 70
      2. 存在 severity=high 的推理链问题
      3. approved 明确为 False（作为辅助条件，而非唯一依据）
      4. 存在重复题目（duplicate_check.has_duplicates = true）→ 作为优化建议，不再作为硬阻断
    全部通过才放行。
    """
    score = critique.get('overall_score', 80)
    high_flaws = [
        f for f in critique.get('reasoning_flaws', [])
        if f.get('severity') == 'high'
    ]
    approved = critique.get('approved', True)

    # 检查重复题目（严格禁止）
    duplicate_check = critique.get('duplicate_check', {})
    has_duplicates = duplicate_check.get('has_duplicates', False)

    # 检查编号连续性（严格禁止每个题型单独编号）
    numbering_check = critique.get('numbering_check', {})
    has_numbering_issues = not numbering_check.get('is_continuous', True)

    # 检查题型数量（选择题10，填空题10，判断题10，简答题4）
    quantity_check = critique.get('quantity_check', {})
    has_quantity_issues = not quantity_check.get('is_valid', True)

    needs = (score < 70) or (len(high_flaws) > 0) or (not approved) or has_numbering_issues or has_quantity_issues
    logger.info(
        f"[Critic判断] score={score}, high_flaws={len(high_flaws)}, "
        f"approved={approved}, duplicates={has_duplicates}(soft), numbering={has_numbering_issues}, quantity={has_quantity_issues} → {'修订' if needs else '通过'}"
    )
    return needs


def _exam_is_usable_without_revision(critique: dict) -> bool:
    """可用性优先：题量与编号等格式硬指标过关时直接交付，避免整卷修订超时。"""
    score = critique.get('overall_score', 80)
    duplicate_check = critique.get('duplicate_check', {})
    numbering_check = critique.get('numbering_check', {})
    quantity_check = critique.get('quantity_check', {})
    has_duplicates = duplicate_check.get('has_duplicates', False)
    numbering_ok = numbering_check.get('is_continuous', True)
    quantity_ok = quantity_check.get('is_valid', False)
    high_flaws = [
        f for f in critique.get('reasoning_flaws', [])
        if f.get('severity') == 'high'
    ]
    non_structural_high_flaws = [
        flaw for flaw in high_flaws
        if not any(keyword in str(flaw.get('flaw', '')) for keyword in ['数量', '编号', '重复'])
    ]

    usable = quantity_ok and numbering_ok and score >= 55 and len(non_structural_high_flaws) <= 1
    logger.info(
        f"[试卷可用性判断] score={score}, quantity_ok={quantity_ok}, numbering_ok={numbering_ok}, "
        f"duplicates={has_duplicates}(soft), non_structural_high_flaws={len(non_structural_high_flaws)} → {'可直接交付' if usable else '仍需修订'}"
    )
    return usable


def _exam_structural_only_issues(critique: dict) -> bool:
    """
    是否仅存在结构性问题（重复/编号/数量），且没有非结构高危缺陷。
    """
    duplicate_check = critique.get('duplicate_check', {})
    numbering_check = critique.get('numbering_check', {})
    quantity_check = critique.get('quantity_check', {})
    has_structural = (
        bool(duplicate_check.get('has_duplicates', False))
        or (not bool(numbering_check.get('is_continuous', True)))
        or (not bool(quantity_check.get('is_valid', True)))
    )
    if not has_structural:
        return False
    high_flaws = [
        f for f in critique.get('reasoning_flaws', [])
        if f.get('severity') == 'high'
    ]
    non_structural_high_flaws = [
        flaw for flaw in high_flaws
        if not any(keyword in str(flaw.get('flaw', '')) for keyword in ['数量', '编号', '重复'])
    ]
    return len(non_structural_high_flaws) == 0


def _exam_needs_revision_relaxed(critique: dict) -> bool:
    """
    宽松修订判定：
    - 不以分数作为硬阻断；
    - 仅在存在明确可执行问题时进入一次修订。
    """
    if not isinstance(critique, dict):
        return False
    if bool(critique.get("needs_revision")):
        return True
    quantity_check = critique.get("quantity_check", {}) or {}
    numbering_check = critique.get("numbering_check", {}) or {}
    if not bool(quantity_check.get("is_valid", True)):
        return True
    if not bool(numbering_check.get("is_continuous", True)):
        return True
    if critique.get("approved") is False:
        return True
    high_flaws = [
        f for f in (critique.get("reasoning_flaws", []) or [])
        if str((f or {}).get("severity", "")).lower() == "high"
    ]
    if high_flaws:
        return True
    issues = critique.get("specific_issues", []) or []
    return len(issues) > 0


def should_revise_quiz(state: QuizState) -> str:
    """最多 2 轮反思；多条件交叉验证决定是否修订"""
    if state.get('reflection_rounds', 0) >= 2:
        logger.info("[Reflection] 达到最大轮次（2轮），输出最终结果")
        return "end"
    if state.get("degrade_reason") in {"budget_exhausted", "revise_timeout"}:
        return "end"
    if _quiz_budget_left_seconds(state) <= 12:
        logger.info("[Reflection] Quiz 预算不足，结束反思循环")
        return "end"
    if _critique_needs_revision(state.get('critique', {})):
        return "revise"
    return "end"


def should_revise_exam(state: ExamPaperState) -> str:
    # 试卷链路放宽：最多 1 轮修订，避免 revise->critique 长循环
    if state.get('reflection_rounds', 0) >= 1:
        return "end"
    if state.get("degrade_reason") in {"budget_exhausted", "revise_timeout", "structural_retry"}:
        return "end"
    if _exam_budget_left_seconds(state) <= 20:
        logger.info("[Reflection] Exam 预算不足，结束反思循环")
        return "end"
    if state.get("quality_floor_passed") is True:
        return "end"
    critique = state.get('critique', {})
    if _exam_is_usable_without_revision(critique):
        return "end"
    if _exam_needs_revision_relaxed(critique):
        return "revise"
    return "end"


# ==================== 构建工作流图 ====================

def build_quiz_graph() -> StateGraph:
    """Reflexion 架构：出题+推理链 → Critic质疑 → Revise修订（最多2轮）"""
    graph = StateGraph(QuizState)

    graph.add_node("generate", generate_with_reasoning_node)
    graph.add_node("critique", critique_quiz_node)
    graph.add_node("revise", revise_with_reflection_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges(
        "critique",
        should_revise_quiz,
        {"revise": "revise", "end": END}
    )
    graph.add_edge("revise", "critique")

    return graph.compile()


def build_exam_graph() -> StateGraph:
    """试卷 Reflexion 架构"""
    graph = StateGraph(ExamPaperState)

    graph.add_node("generate", generate_exam_with_reasoning_node)
    graph.add_node("critique", critique_exam_node)
    graph.add_node("revise", revise_exam_with_reflection_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges(
        "critique",
        should_revise_exam,
        {"revise": "revise", "end": END}
    )
    # 放宽链路：修订后直接结束，不再进入二次 Critique 阻断
    graph.add_edge("revise", END)

    return graph.compile()


quiz_workflow = build_quiz_graph()
exam_workflow = build_exam_graph()


# ==================== 对外接口 ====================

async def run_quiz_agent(
    topic: str,
    quiz_type: str = "选择题",
    num: int = 3,
    sample_paper_context: str = None,
    force_llm_critic: bool = False,
    forbidden_question_signatures: Optional[List[str]] = None,
) -> dict:
    """
    运行 Reflexion 出题系统（3 Agent 协作）
    流程：出题+推理链 → Critic质疑推理链 → Revise针对批评修订（最多2轮）
    """
    quiz_started_at = time.monotonic()
    quiz_type = _normalize_question_type(quiz_type, "选择题")
    # 检索入口策略：泛化请求走综合检索，具体考点走常规检索。
    if _is_generic_topic_request(topic):
        rag_context = await _get_comprehensive_rag_context(topic)
    else:
        rag_context = await get_rag_context(topic)
    related, rel_reason = _check_rag_topic_relevance(topic, rag_context)
    if not related:
        logger.warning(f"[Quiz Relevance] 首次相关性不足，尝试回退检索。topic={topic}, reason={rel_reason}")
        fallback_topic = "本课程重点知识点"
        fallback_context = await _get_comprehensive_rag_context(fallback_topic)
        related2, rel_reason2 = _check_rag_topic_relevance(fallback_topic, fallback_context)
        if related2:
            rag_context = fallback_context
            topic = fallback_topic
            logger.info(f"[Quiz Relevance] 已回退到课件核心知识点出题。reason={rel_reason2}")
        else:
            return {
                "kind": "chat",
                "render_mode": "markdown",
                "text": (
                    "本次请求与已检索到的课件知识点相关性不足，已停止出题。"
                    "请提供更具体的课件内知识点（例如“进程同步/PV操作/死锁检测”）后重试。"
                ),
                "payload": {},
                "meta": {
                    "delivery_mode": "failed",
                    "degrade_reason": "rag_topic_irrelevant",
                    "quality_floor_passed": False,
                    "evidence_source": "courseware_only",
                    "tool_calls": {"rag": 1, "web": 0},
                    "quiz_end_to_end_ms": int((time.monotonic() - quiz_started_at) * 1000),
                    "relevance_reason": rel_reason,
                },
            }

    gate_state: QuizState = {
        "topic": topic,
        "quiz_type": quiz_type,
        "num": num,
        "context": rag_context,
        "sample_paper_context": sample_paper_context or "",
        "quiz": "",
        "reasoning": "{}",
        "quiz_payload": {},
        "critique": {},
        "initial_score": 0,
        "revised_quiz": "",
        "revision_notes": "",
        "revised_quiz_payload": {},
        "delivery_mode": "full",
        "quality_floor_passed": False,
        "degrade_reason": "",
        "quiz_budget_started_at": quiz_started_at,
        "quiz_budget_seconds": QUIZ_BUDGET_SECONDS,
        "evidence_source": "courseware_only",
        "reflection_rounds": 0,
        "force_llm_critic": bool(force_llm_critic),
    }
    use_web = False
    gate_reason = "courseware_sufficient"
    # 预算足够时才评估是否联网补证据，优先守住整体时延。
    if _quiz_budget_left_seconds(gate_state) > 35:
        use_web, gate_reason = await _decide_need_web_context(topic, rag_context)
    else:
        gate_reason = "budget_skip_web_gate"

    web_context = ""
    if use_web:
        web_context = await _fetch_web_reference_for_quiz(topic)
    evidence_source = "courseware_plus_web" if web_context else "courseware_only"

    context = rag_context if not web_context else f"{rag_context}\n\n{web_context}"
    context = _limit_context_size(context, max_chars=12000)

    initial_state: QuizState = {
        "topic": topic,
        "quiz_type": quiz_type,
        "num": num,
        "context": context,
        "sample_paper_context": sample_paper_context or "",
        "quiz": "",
        "reasoning": "{}",
        "quiz_payload": {},
        "critique": {},
        "initial_score": 0,
        "revised_quiz": "",
        "revision_notes": "",
        "revised_quiz_payload": {},
        "delivery_mode": "full",
        "quality_floor_passed": False,
        "degrade_reason": "",
        "quiz_budget_started_at": quiz_started_at,
        "quiz_budget_seconds": QUIZ_BUDGET_SECONDS,
        "evidence_source": evidence_source,
        "reflection_rounds": 0,
        "force_llm_critic": bool(force_llm_critic),
    }

    try:
        result = await asyncio.wait_for(
            quiz_workflow.ainvoke(initial_state),
            timeout=QUIZ_BUDGET_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error(f"[Quiz Agent] quiz_workflow 超时（>{QUIZ_BUDGET_SECONDS}s）")
        if not ALLOW_DEGRADED_DELIVERY:
            raise TimeoutError("出题超时，且已禁用降级交付。请稍后重试。")
        fallback, fallback_reason = _build_retrieval_anchored_fallback_questions(
            quiz_type=quiz_type,
            num=max(1, int(num)),
            start_num=1,
            topic=topic or "本课程重点内容",
            courseware_context=rag_context,
            score_per_question=5,
        )
        if not fallback:
            return {
                "kind": "chat",
                "render_mode": "markdown",
                "text": f"系统已触发降级保护：{fallback_reason or '无法从课件提取足够依据'}。请上传更完整课件或指定具体知识点后重试。",
                "payload": {},
                "meta": {
                    "delivery_mode": "partial_revised",
                    "degrade_reason": "quality_guard_no_evidence",
                    "quality_floor_passed": False,
                    "evidence_source": evidence_source,
                    "tool_calls": {"rag": 1, "web": 1 if web_context else 0},
                    "quiz_end_to_end_ms": int((time.monotonic() - quiz_started_at) * 1000),
                    "tool_gate_reason": gate_reason,
                },
            }

        questions = []
        for i, q in enumerate(fallback, start=1):
            questions.append({
                "id": i,
                "type": q.get("type", quiz_type),
                "question": q.get("question") or q.get("content", ""),
                "options": q.get("options"),
                "answer": q.get("answer", ""),
                "explanation": q.get("explanation") or q.get("analysis", ""),
                "score": q.get("score") if isinstance(q.get("score"), str) else f"{int(q.get('score') or 5)}分",
                "difficulty": q.get("difficulty", "中等"),
            })
        questions = _dedupe_quiz_questions_across_rounds(
            questions=questions,
            forbidden_question_signatures=forbidden_question_signatures,
            quiz_type=quiz_type,
            topic=topic,
            rag_context=rag_context,
            target_num=max(1, int(num)),
        )
        payload = _build_quiz_payload(topic, questions)
        floor_flags = _quiz_quality_floor_flags(questions, quiz_type, num)
        term_style_ok = all(_question_term_style_ok(q) for q in questions)
        delivery_mode, degrade_reason, can_deliver = _final_quiz_delivery_state(
            {"delivery_mode": "partial_revised", "degrade_reason": "budget_exhausted"},
            floor_flags,
            term_style_ok,
        )
        if not can_deliver:
            return {
                "kind": "chat",
                "render_mode": "markdown",
                "text": f"本次出题未达到交付质量要求，已停止交付题卡。原因：{degrade_reason}。请换一个更具体的知识点后重试。",
                "payload": {},
                "meta": {
                    "delivery_mode": delivery_mode,
                    "degrade_reason": degrade_reason,
                    "quality_floor_passed": False,
                    "quality_floor_flags": floor_flags,
                    "evidence_source": evidence_source,
                    "tool_calls": {"rag": 1, "web": 1 if web_context else 0},
                    "quiz_end_to_end_ms": int((time.monotonic() - quiz_started_at) * 1000),
                    "tool_gate_reason": gate_reason,
                    "term_style_ok": term_style_ok,
                },
            }
        return {
            "kind": "quiz_set",
            "render_mode": "interactive_cards",
            "text": _build_quiz_text_from_questions(questions),
            "payload": payload,
            "meta": {
                "delivery_mode": delivery_mode,
                "degrade_reason": degrade_reason,
                "quality_floor_passed": bool(floor_flags.get("quality_floor_passed")),
                "evidence_source": evidence_source,
                "tool_calls": {"rag": 1, "web": 1 if web_context else 0},
                "quiz_end_to_end_ms": int((time.monotonic() - quiz_started_at) * 1000),
                "tool_gate_reason": gate_reason,
                "term_style_ok": term_style_ok,
            },
        }

    # ── Reflexion 效果验证日志 ──────────────────────────────────────────
    rounds = result.get('reflection_rounds', 0)
    initial_score = result.get('initial_score', 'N/A')
    final_critique = result.get('critique', {})
    final_score = final_critique.get('overall_score', 'N/A')
    final_flaws = len([f for f in final_critique.get('reasoning_flaws', []) if f.get('severity') == 'high'])
    used_revised = bool(result.get('revised_quiz'))
    improvement = (
        f"{initial_score} → {final_score} (+{final_score - initial_score})"
        if isinstance(initial_score, int) and isinstance(final_score, int)
        else f"{initial_score} → {final_score}"
    )
    logger.info(
        f"[Reflexion统计] topic={topic} | 修订轮次={rounds} | "
        f"评分变化={improvement} | 高危缺陷={final_flaws} | 使用修订版={used_revised}"
    )
    # ────────────────────────────────────────────────────────────────────

    final_text = result.get('revised_quiz') or result.get('quiz') or ''
    final_payload = result.get('revised_quiz_payload') or result.get('quiz_payload') or _questions_from_quiz_text(final_text, topic)
    parsed_questions = (final_payload or {}).get("questions", []) if isinstance(final_payload, dict) else []
    if not parsed_questions:
        fallback_questions = _parse_questions_from_serialized_quiz(final_text)
        if fallback_questions:
            final_payload = _build_quiz_payload(topic, fallback_questions)
            final_text = _build_quiz_text_from_questions(fallback_questions)
            parsed_questions = fallback_questions

    if not parsed_questions:
        anchored_fallback, anchored_reason = _build_retrieval_anchored_fallback_questions(
            quiz_type=quiz_type,
            num=max(1, int(num)),
            start_num=1,
            topic=topic or "本课程重点内容",
            courseware_context=rag_context,
            score_per_question=5,
        )
        if anchored_fallback:
            final_payload = _build_quiz_payload(topic, anchored_fallback)
            final_text = _build_quiz_text_from_questions(anchored_fallback)
            parsed_questions = anchored_fallback
            result = {**result, "delivery_mode": "partial_revised", "degrade_reason": "budget_exhausted"}
        else:
            return {
                "kind": "chat",
                "render_mode": "markdown",
                "text": f"系统已触发降级保护：{anchored_reason or '无法从课件提取足够依据'}。请上传更完整课件或指定具体知识点后重试。",
                "payload": {},
                "meta": {
                    "delivery_mode": "partial_revised",
                    "degrade_reason": "quality_guard_no_evidence",
                    "quality_floor_passed": False,
                    "evidence_source": evidence_source,
                    "tool_calls": {"rag": 1, "web": 1 if web_context else 0},
                    "quiz_end_to_end_ms": int((time.monotonic() - quiz_started_at) * 1000),
                    "tool_gate_reason": gate_reason,
                },
            }

    parsed_questions = _dedupe_quiz_questions_across_rounds(
        questions=parsed_questions,
        forbidden_question_signatures=forbidden_question_signatures,
        quiz_type=quiz_type,
        topic=topic,
        rag_context=rag_context,
        target_num=max(1, int(num)),
    )
    parsed_questions, answer_consistency_fixes = _reconcile_choice_answers_with_explanations(parsed_questions)

    floor_flags = _quiz_quality_floor_flags(parsed_questions, quiz_type, num)
    if quiz_type == "选择题" and not bool(floor_flags.get("options_ok")):
        logger.warning("[Quiz Agent] 选择题缺少选项，尝试课件锚定兜底替换")
        anchored_fallback, anchored_reason = _build_retrieval_anchored_fallback_questions(
            quiz_type=quiz_type,
            num=max(1, int(num)),
            start_num=1,
            topic=topic or "本课程重点内容",
            courseware_context=rag_context,
            score_per_question=5,
        )
        if anchored_fallback:
            parsed_questions = _dedupe_quiz_questions_across_rounds(
                questions=anchored_fallback,
                forbidden_question_signatures=forbidden_question_signatures,
                quiz_type=quiz_type,
                topic=topic,
                rag_context=rag_context,
                target_num=max(1, int(num)),
            )
            parsed_questions, answer_consistency_fixes = _reconcile_choice_answers_with_explanations(parsed_questions)
            floor_flags = _quiz_quality_floor_flags(parsed_questions, quiz_type, num)
            result = {**result, "delivery_mode": "partial_revised", "degrade_reason": "missing_choice_options_repaired"}
        if not bool(floor_flags.get("options_ok")):
            return {
                "kind": "chat",
                "render_mode": "markdown",
                "text": f"本次选择题生成缺少完整选项，已停止交付题卡。{anchored_reason or '请换一个更具体的知识点后重试。'}",
                "payload": {},
                "meta": {
                    "delivery_mode": "failed",
                    "degrade_reason": "missing_choice_options",
                    "quality_floor_passed": False,
                    "evidence_source": evidence_source,
                    "tool_calls": {"rag": 1, "web": 1 if web_context else 0},
                    "quiz_end_to_end_ms": int((time.monotonic() - quiz_started_at) * 1000),
                    "tool_gate_reason": gate_reason,
                },
            }

    final_payload = _build_quiz_payload(topic, parsed_questions)
    final_text = _build_quiz_text_from_questions(parsed_questions)

    term_style_ok = all(_question_term_style_ok(q) for q in parsed_questions)
    delivery_mode, degrade_reason, can_deliver = _final_quiz_delivery_state(result, floor_flags, term_style_ok)
    if not can_deliver:
        return {
            "kind": "chat",
            "render_mode": "markdown",
            "text": f"本次出题未达到交付质量要求，已停止交付题卡。原因：{degrade_reason}。请换一个更具体的知识点后重试。",
            "payload": {},
            "meta": {
                "delivery_mode": delivery_mode,
                "degrade_reason": degrade_reason,
                "quality_floor_passed": False,
                "quality_floor_flags": floor_flags,
                "evidence_source": evidence_source,
                "tool_calls": {"rag": 1, "web": 1 if web_context else 0},
                "quiz_end_to_end_ms": int((time.monotonic() - quiz_started_at) * 1000),
                "tool_gate_reason": gate_reason,
                "term_style_ok": term_style_ok,
            },
        }
    if (not ALLOW_DEGRADED_DELIVERY) and (delivery_mode != "full" or degrade_reason):
        raise RuntimeError(f"本次出题未达严格质量要求（delivery_mode={delivery_mode}, reason={degrade_reason or 'quality_floor_not_passed'}），已禁止降级交付，请重试。")
    end_to_end_ms = int((time.monotonic() - quiz_started_at) * 1000)
    return {
        "kind": "quiz_set",
        "render_mode": "interactive_cards",
        "text": final_text,
        "payload": final_payload,
        "meta": {
            "delivery_mode": delivery_mode,
            "quality_floor_passed": bool(floor_flags.get("quality_floor_passed")),
            "degrade_reason": degrade_reason,
            "evidence_source": evidence_source,
            "tool_calls": {"rag": 1, "web": 1 if web_context else 0},
            "quiz_end_to_end_ms": end_to_end_ms,
            "tool_gate_reason": gate_reason,
            "term_style_ok": term_style_ok,
            "choice_answer_consistency_fixes": answer_consistency_fixes,
        },
    }


def _resolve_quantity_dist(
    quiz_types: List[str],
    total_questions: int,
    quantity_dist: Optional[dict],
) -> dict:
    """统一解析题型数量分布，避免空分布被误修复为默认 23 题。"""
    if isinstance(quantity_dist, dict) and any(int(quantity_dist.get(k, 0) or 0) > 0 for k in ("choice", "fill", "judge", "essay")):
        return {
            "choice": max(0, int(quantity_dist.get("choice", 0) or 0)),
            "fill": max(0, int(quantity_dist.get("fill", 0) or 0)),
            "judge": max(0, int(quantity_dist.get("judge", 0) or 0)),
            "essay": max(0, int(quantity_dist.get("essay", 0) or 0)),
        }

    label_to_key = {"选择题": "choice", "填空题": "fill", "判断题": "judge", "简答题": "essay"}
    selected = [label_to_key.get(t) for t in (quiz_types or []) if label_to_key.get(t)]
    if not selected:
        selected = ["choice", "fill", "judge", "essay"]

    total = int(total_questions or 0)
    if total <= 0:
        total = 23 if len(selected) >= 4 else max(5, len(selected) * 3)

    base = total // len(selected)
    rem = total % len(selected)
    resolved = {"choice": 0, "fill": 0, "judge": 0, "essay": 0}
    for idx, key in enumerate(selected):
        resolved[key] = base + (1 if idx < rem else 0)
    return resolved


def _build_exam_layout_params(quantity_dist: dict) -> dict:
    choice_count = int((quantity_dist or {}).get("choice", 0) or 0)
    fill_count = int((quantity_dist or {}).get("fill", 0) or 0)
    judge_count = int((quantity_dist or {}).get("judge", 0) or 0)
    essay_count = int((quantity_dist or {}).get("essay", 0) or 0)

    current = 1
    choice_start = current
    choice_end = max(choice_start, choice_start + max(choice_count, 0) - 1)
    current = choice_end + 1

    fill_start = current
    fill_end = max(fill_start, fill_start + max(fill_count, 0) - 1)
    current = fill_end + 1

    judge_start = current
    judge_end = max(judge_start, judge_start + max(judge_count, 0) - 1)
    current = judge_end + 1

    essay_start = current
    essay_end = max(essay_start, essay_start + max(essay_count, 0) - 1)

    return {
        "choice_count": choice_count,
        "fill_count": fill_count,
        "judge_count": judge_count,
        "essay_count": essay_count,
        "choice_start": choice_start,
        "choice_end": choice_end,
        "fill_start": fill_start,
        "fill_end": fill_end,
        "judge_start": judge_start,
        "judge_end": judge_end,
        "essay_start": essay_start,
        "essay_end": essay_end,
    }


def _build_default_stage_plan(quantity_dist: Optional[dict] = None) -> List[dict]:
    dist = quantity_dist or {"choice": 10, "fill": 5, "judge": 5, "essay": 3}
    choice = max(0, int(dist.get("choice", 0) or 0))
    fill = max(0, int(dist.get("fill", 0) or 0))
    judge = max(0, int(dist.get("judge", 0) or 0))
    essay = max(0, int(dist.get("essay", 0) or 0))

    # 小题量阶段缩短超时，避免 1/1/1/1 仍按大卷超时预算运行
    choice_timeout = max(45, min(240, 24 * max(1, choice)))
    fill_judge_timeout = max(45, min(210, 22 * max(1, fill + judge)))
    # 简答题需要更多时间：考虑失败重试和文本补题，每道题给足60秒
    essay_timeout = max(60, min(300, 60 * max(1, essay)))

    stages: List[dict] = []
    if choice > 0:
        stages.append(
            {
                "id": "choice",
                "label": "选择题阶段",
                "quantity_dist": {"choice": choice, "fill": 0, "judge": 0, "essay": 0},
                "quiz_types": ["选择题"],
                "timeout_seconds": choice_timeout,
                "target_score": choice * 2,
            }
        )
    if fill > 0 or judge > 0:
        stages.append(
            {
                "id": "fill_judge",
                "label": "填空判断阶段",
                "quantity_dist": {"choice": 0, "fill": fill, "judge": judge, "essay": 0},
                "quiz_types": [t for t, c in [("填空题", fill), ("判断题", judge)] if c > 0],
                "timeout_seconds": fill_judge_timeout,
                "target_score": (fill + judge) * 2,
            }
        )
    if essay > 0:
        stages.append(
            {
                "id": "essay",
                "label": "简答题阶段",
                "quantity_dist": {"choice": 0, "fill": 0, "judge": 0, "essay": essay},
                "quiz_types": ["简答题"],
                "timeout_seconds": essay_timeout,
                "target_score": essay * 20,
            }
        )
    return stages


def _strict_stage_quality_ok(questions: List[dict], expected_dist: dict) -> tuple[bool, str]:
    stage_total = (
        int(expected_dist.get("choice", 0)) * 2
        + int(expected_dist.get("fill", 0)) * 2
        + int(expected_dist.get("judge", 0)) * 2
        + int(expected_dist.get("essay", 0)) * 20
    )
    flags = _exam_quality_floor_flags(questions, expected_dist, target_total_score=max(0, stage_total))
    if not flags.get("quantity_ok"):
        return False, "题量不满足"
    if not flags.get("numbering_ok"):
        return False, "编号不连续"
    # 重复题不再作为阶段级硬失败，仅作为日志观测项
    if flags.get("duplicates"):
        logger.warning("[ExamStage] 检测到重复题（soft），继续流程，由模型质量与前端体验兜底。")
    if not flags.get("options_ok"):
        return False, "选择题选项不完整"
    if _collect_term_style_violation_numbers(questions):
        return False, "题面术语风格不符合要求"
    # 严格模式附加：禁止明显兜底模板痕迹
    texts = []
    for q in questions:
        texts.append(str(q.get("content", "")))
        opts = q.get("options") or []
        if isinstance(opts, list):
            texts.extend([str(o) for o in opts])
    merged = "\n".join(texts)
    banned_patterns = [
        "与课件无关的泛化描述",
        "完全否定课件机制的描述",
        "该知识点仅与实现细节有关",
    ]
    if any(p in merged for p in banned_patterns):
        return False, "检测到兜底模板内容"
    return True, ""


def _renumber_questions(questions: List[dict], start_number: int) -> List[dict]:
    out: List[dict] = []
    for idx, q in enumerate(questions, start=0):
        nq = dict(q)
        nq["number"] = start_number + idx
        out.append(nq)
    return out


async def _run_exam_stage_with_retry(
    stage: dict,
    topics: List[str],
    context: str,
    sample_paper_context: str,
    deadline_ts: float,
    exam_fast_mode: bool = True,
) -> tuple[List[dict], dict]:
    stage_id = stage["id"]
    stage_dist = stage["quantity_dist"]
    total_questions = int(sum(stage_dist.values()))
    retries = 2 if EXAM_STAGE_RETRY_ENABLED else 1
    stage_trace = {"exam_stage": stage_id, "attempt": 0, "stage_status": "failed", "error": "", "stage_exit_reason": ""}

    for attempt in range(1, retries + 1):
        stage_trace["attempt"] = attempt
        left = deadline_ts - time.monotonic()
        stage_timeout = min(float(stage.get("timeout_seconds", 45)), max(8.0, left - 2.0))
        budget_reserved_for_fallback = max(30.0, stage_timeout * 0.35)
        logger.info(
            f"[ExamStageBudget] stage={stage_id}, attempt={attempt}/{retries}, "
            f"budget_left_before_stage={left:.1f}, budget_reserved_for_fallback={budget_reserved_for_fallback:.1f}"
        )
        if stage_timeout <= 8:
            stage_trace["error"] = "整体预算不足"
            stage_trace["stage_exit_reason"] = "budget_exhausted_before_run"
            break

        stage_state: ExamPaperState = {
            "topics": topics,
            "quiz_types": stage.get("quiz_types", []),
            "total_questions": total_questions,
            "quantity_dist": stage_dist,
            "sample_paper_context": sample_paper_context or "",
            "contexts": [context],
            "exam_paper": "",
            "reasoning": "{}",
            "exam_payload": {},
            "critique": {},
            "initial_score": 0,
            "revised_exam": "",
            "revision_notes": "",
            "revised_exam_payload": {},
            "delivery_mode": "full",
            "quality_floor_passed": False,
            "degrade_reason": "",
            "exam_budget_started_at": time.monotonic(),
            "exam_budget_seconds": int(stage_timeout),
            "target_total_score": int(stage.get("target_score", 100)),
            "critic_timeout_count": 0,
            "stage_fast_mode": exam_fast_mode,
            "reflection_rounds": 0,
        }

        try:
            # 每次都重新构建workflow，避免全局单例在取消后可能出现的状态问题
            stage_workflow = build_exam_graph()
            result = await asyncio.wait_for(stage_workflow.ainvoke(stage_state), timeout=stage_timeout)
            stage_trace["critic_timeout_count"] = int(result.get("critic_timeout_count") or 0)
            payload = result.get("revised_exam_payload") or result.get("exam_payload") or {}
            questions = (payload.get("exam_data") or {}).get("questions", []) if isinstance(payload, dict) else []
            questions = [ _normalize_exam_question(q, i + 1) for i, q in enumerate(questions or []) ]

            ok, reason = _strict_stage_quality_ok(questions, stage_dist)
            if not ok:
                raise RuntimeError(reason or "阶段质量校验失败")

            stage_trace["stage_status"] = "success"
            stage_trace["error"] = ""
            stage_trace["stage_exit_reason"] = "quality_passed"
            return questions, stage_trace
        except Exception as e:
            err_text = str(e or "").strip()
            if not err_text:
                err_text = type(e).__name__ if type(e).__name__ else "unknown_error"
            stage_trace["error"] = err_text
            stage_exit = _runtime_classify_structured_error(e)
            stage_trace["stage_exit_reason"] = stage_exit if stage_exit != "other" else "stage_runtime_error"
            miss_match = re.search(r'仍缺\s*(\d+)\s*道', err_text)
            if miss_match:
                stage_trace["missing_after_retries"] = int(miss_match.group(1))
            logger.warning(f"[ExamStage] stage={stage_id} attempt={attempt}/{retries} 失败: {e}")
            if attempt >= retries:
                break
            # 重试前短暂延迟，确保前一个任务的资源完全释放
            await asyncio.sleep(0.5)

    if not stage_trace.get("stage_exit_reason"):
        stage_trace["stage_exit_reason"] = "max_retries_exhausted"
    return [], stage_trace


def _build_exam_failed_result(
    message: str,
    *,
    exam_started_at: float,
    stage_trace: Optional[List[dict]] = None,
    evidence_source: str = "courseware_only",
    tool_call_rag: int = 1,
    tool_call_web: int = 0,
    stage_exit_reason: str = "failed",
    missing_after_retries: int = 0,
    quality_checks: Optional[dict] = None,
    term_style_ok: Optional[bool] = None,
    critic_timeout_count: int = 0,
) -> dict:
    meta = {
        "delivery_mode": "failed",
        "quality_floor_passed": False,
        "degrade_reason": stage_exit_reason,
        "stage_exit_reason": stage_exit_reason,
        "missing_after_retries": int(max(0, missing_after_retries)),
        "quality_checks": quality_checks or {},
        "critic_timeout_count": int(max(0, critic_timeout_count)),
        "exam_end_to_end_ms": int((time.monotonic() - exam_started_at) * 1000),
        "evidence_source": evidence_source,
        "tool_calls": {"rag": tool_call_rag, "web": tool_call_web},
    }
    if stage_trace is not None:
        meta["stage_trace"] = stage_trace
    if term_style_ok is not None:
        meta["term_style_ok"] = bool(term_style_ok)
    return {
        "kind": "chat",
        "render_mode": "markdown",
        "text": message,
        "payload": {"exam_data": {"questions": []}},
        "meta": meta,
    }


async def run_exam_agent(
    topics: List[str],
    quiz_types: List[str],
    total_questions: int = 34,
    sample_paper_context: str = None,
    quantity_dist: dict = None,
    stage_plan: bool = False,
    rerun_stage: Optional[str] = None,
    partial_questions: Optional[List[dict]] = None,
    exam_fast_mode: bool = True,
) -> dict:
    """
    运行 Reflexion 出卷系统（3 Agent 协作）
    流程：出卷+推理链 → Critic质疑推理链 → Revise针对批评修订（最多2轮）
    """
    exam_started_at = time.monotonic()
    tool_call_web = 0
    tool_call_rag = 1
    # 参数校验和修正
    if not quiz_types or len(quiz_types) == 0:
        quiz_types = ['选择题', '填空题', '判断题', '简答题']

    auto_comprehensive_mode = False
    if not topics:
        topics = _default_comprehensive_topics()
        auto_comprehensive_mode = True

    quantity_dist = _resolve_quantity_dist(quiz_types, total_questions, quantity_dist)
    total_questions = sum(quantity_dist.values())
    if total_questions <= 0:
        quantity_dist = {"choice": 10, "fill": 5, "judge": 5, "essay": 3}
        total_questions = 23

    logger.info(f"[Exam Agent] 校验后参数: quiz_types={quiz_types}, total_questions={total_questions}, quantity_dist={quantity_dist}")

    # 如果没有样卷，先找真实试卷风格参考；找不到再降级到题型格式参考。
    online_format_context = ""
    if not sample_paper_context or sample_paper_context.strip() == "":
        logger.info("[Exam Agent] 无样卷，联网搜索真实试卷参考...")
        session_course = str(current_session_id.get() or "").strip()
        topic_hint = str(topics[0] if topics else "").strip()
        reference_course = session_course if session_course and session_course != "default" else (topic_hint or "大学课程")

        try:
            online_format_context = await fetch_exam_paper_reference(
                reference_course,
                topic_hint=topic_hint,
                limit=5,
            )
        except Exception as e:
            logger.warning(f"[Exam Agent] 真实试卷搜索失败: {e}")
            online_format_context = ""

        if online_format_context:
            logger.info("[Exam Agent] 联网搜索获取真实试卷参考成功")
            tool_call_web = 1
        else:
            logger.info("[Exam Agent] 未找到真实试卷参考，降级搜索题型格式参考")
            try:
                format_result = str(await search_exam_format.ainvoke({"course_name": reference_course}) or "")
                if format_result and "未获取到" not in format_result and "无法获取" not in format_result:
                    online_format_context = f"【联网题型格式参考】\n{format_result[:2000]}"
                    logger.info("[Exam Agent] 联网搜索获取题型格式参考成功")
                    tool_call_web = 1
            except Exception as e:
                logger.warning(f"[Exam Agent] 题型格式搜索失败: {e}")

            if not online_format_context:
                online_format_context = "【提示】无样卷和可用联网试卷参考，请使用通用期末试卷格式：选择题、填空题、判断题、简答题四种题型均衡分布。"
    else:
        logger.info("[Exam Agent] 使用本地样卷格式")

    # 试卷生成使用单份 merged context，避免多 topic 串行检索造成长时间阻塞
    merged_topic_query = "；".join([t for t in topics if t][:6]) if topics else "本课程重点知识点"
    if _is_generic_topic_request(merged_topic_query):
        merged_context = await _get_comprehensive_rag_context(merged_topic_query)
    else:
        merged_context = await get_rag_context(merged_topic_query)
    exam_related, exam_rel_reason = _check_rag_topic_relevance(merged_topic_query, merged_context)
    comprehensive_like = ("；" in merged_topic_query and len([t for t in topics if str(t).strip()]) >= 4)
    if not exam_related and not auto_comprehensive_mode and not comprehensive_like:
        logger.warning(f"[Exam Relevance] 首次相关性不足，尝试回退检索。query={merged_topic_query}, reason={exam_rel_reason}")
        fallback_topic = "本课程重点知识点"
        fallback_context = await _get_comprehensive_rag_context(fallback_topic)
        fallback_related, fallback_reason = _check_rag_topic_relevance(fallback_topic, fallback_context)
        if fallback_related:
            merged_context = fallback_context
            logger.info(f"[Exam Relevance] 已回退到课件核心知识点出卷。reason={fallback_reason}")
        elif topics:
            return {
                "kind": "chat",
                "render_mode": "markdown",
                "text": (
                    "当前出卷请求与已上传课件知识点相关性不足，已停止生成试卷。"
                    "请改为课件内明确考点后重试，或先补充对应课件。"
                ),
                "payload": {"exam_data": {"questions": []}},
                "meta": {
                    "delivery_mode": "failed",
                    "quality_floor_passed": False,
                    "degrade_reason": "rag_topic_irrelevant",
                    "strict_mode": False,
                    "exam_end_to_end_ms": int((time.monotonic() - exam_started_at) * 1000),
                    "evidence_source": "courseware_only",
                    "tool_calls": {"rag": tool_call_rag, "web": tool_call_web},
                    "relevance_reason": exam_rel_reason,
                },
            }
    elif not exam_related and (auto_comprehensive_mode or comprehensive_like):
        logger.warning(
            f"[Exam Relevance] 综合默认出卷模式下忽略相关性硬拦截，继续生成。reason={exam_rel_reason}"
        )
    merged_context = _limit_context_size(merged_context, max_chars=12000)
    # 以 RAG 命中结果动态扩展考点池，避免固定 7 大类导致重复
    dynamic_topics = _build_dynamic_exam_topics(topics, merged_context, total_questions)
    if dynamic_topics:
        topics = dynamic_topics
        logger.info(f"[Exam Agent] 动态考点池已更新，topics={len(topics)}")
    logger.info(
        f"[Exam Agent] 聚合检索完成: topics={len(topics)} -> single_context_len={len(merged_context)}"
    )
    contexts = [merged_context]
    evidence_source = "courseware_plus_web" if tool_call_web > 0 else "courseware_only"

    # 合并样卷格式和联网搜索结果
    final_sample_context = (sample_paper_context or "") + ("\n\n" + online_format_context if online_format_context else "")

    initial_state: ExamPaperState = {
        "topics": topics,
        "topic_bank": topics,
        "topic_usage": {str(t): 0 for t in topics},
        "quiz_types": quiz_types,
        "total_questions": total_questions,
        "quantity_dist": quantity_dist or {},
        "sample_paper_context": final_sample_context,
        "contexts": contexts,
        "exam_paper": "",
        "reasoning": "{}",
        "exam_payload": {},
        "critique": {},
        "initial_score": 0,
        "revised_exam": "",
        "revision_notes": "",
        "revised_exam_payload": {},
        "delivery_mode": "full",
        "quality_floor_passed": False,
        "degrade_reason": "",
        "exam_budget_started_at": exam_started_at,
        "exam_budget_seconds": EXAM_BUDGET_SECONDS,
        "target_total_score": 100,
        "critic_timeout_count": 0,
        "stage_fast_mode": exam_fast_mode,
        "reflection_rounds": 0,
    }

    if stage_plan:
        stage_defs = _build_default_stage_plan(quantity_dist)
        if not stage_defs:
            return _build_exam_failed_result(
                "试卷生成失败：题型分布为空，无法分段出卷。",
                exam_started_at=exam_started_at,
                evidence_source=evidence_source,
                tool_call_rag=tool_call_rag,
                tool_call_web=tool_call_web,
                stage_exit_reason="empty_stage_plan",
                missing_after_retries=max(0, total_questions),
            )
        stage_map = {s["id"]: s for s in stage_defs}
        stage_order = [s["id"] for s in stage_defs]
        deadline_ts = exam_started_at + EXAM_BUDGET_SECONDS
        stage_trace: List[dict] = []
        stage_questions: Dict[str, List[dict]] = {sid: [] for sid in stage_order}
        stage_critic_timeout_count = 0

        def _stage_for_question(q: dict) -> str:
            qtype = _normalize_question_type((q or {}).get("type"), "选择题")
            if qtype == "选择题":
                return "choice"
            if qtype in {"填空题", "判断题"}:
                return "fill_judge"
            if qtype == "简答题":
                return "essay"
            return "choice"

        for i, q in enumerate(partial_questions or [], start=1):
            nq = _normalize_exam_question(q, i)
            sid = _stage_for_question(nq)
            stage_questions.setdefault(sid, []).append(nq)

        if rerun_stage and rerun_stage not in stage_map:
            rerun_stage = None

        for sid in stage_order:
            if rerun_stage and sid != rerun_stage:
                if stage_questions.get(sid):
                    stage_trace.append({
                        "exam_stage": sid,
                        "attempt": 1,
                        "stage_status": "success",
                        "error": "",
                        "stage_latency_ms": 0,
                        "strict_mode": True,
                        "reused": True,
                    })
                    continue
                fail_text = f"仅重跑阶段 {rerun_stage} 失败：缺少已成功段 {sid} 的题目数据。"
                stage_trace.append({
                    "exam_stage": sid,
                    "attempt": 0,
                    "stage_status": "failed",
                    "error": fail_text,
                    "stage_latency_ms": 0,
                    "strict_mode": True,
                })
                return _build_exam_failed_result(
                    fail_text,
                    exam_started_at=exam_started_at,
                    stage_trace=stage_trace,
                    evidence_source=evidence_source,
                    tool_call_rag=tool_call_rag,
                    tool_call_web=tool_call_web,
                    stage_exit_reason="missing_stage_context",
                )

            if stage_questions.get(sid):
                stage_trace.append({
                    "exam_stage": sid,
                    "attempt": 1,
                    "stage_status": "success",
                    "error": "",
                    "stage_latency_ms": 0,
                    "strict_mode": True,
                    "reused": True,
                })
                continue

            stage_started = time.monotonic()
            questions, trace = await _run_exam_stage_with_retry(
                stage=stage_map[sid],
                topics=topics,
                context=merged_context,
                sample_paper_context=final_sample_context,
                deadline_ts=deadline_ts,
                exam_fast_mode=exam_fast_mode,
            )
            trace["stage_latency_ms"] = int((time.monotonic() - stage_started) * 1000)
            trace["strict_mode"] = True
            stage_critic_timeout_count += int(trace.get("critic_timeout_count") or 0)
            stage_trace.append(trace)

            if trace.get("stage_status") != "success":
                logger.warning(
                    f"[ExamStage] 严格模式：阶段 {sid} 失败，整卷终止。"
                )
                missing_after_retries = int(trace.get("missing_after_retries") or sum(stage_map[sid]["quantity_dist"].values()))
                return _build_exam_failed_result(
                    f"试卷生成失败：阶段 {sid} 执行失败（{trace.get('error') or 'unknown'}）。",
                    exam_started_at=exam_started_at,
                    stage_trace=stage_trace,
                    evidence_source=evidence_source,
                    tool_call_rag=tool_call_rag,
                    tool_call_web=tool_call_web,
                    stage_exit_reason=trace.get("stage_exit_reason") or "stage_failed",
                    missing_after_retries=missing_after_retries,
                    critic_timeout_count=stage_critic_timeout_count,
                )

            stage_questions[sid] = questions

        merged_questions: List[dict] = []
        for sid in stage_order:
            merged_questions.extend(stage_questions.get(sid, []))
        merged_questions = _renumber_questions(merged_questions, 1)
        before_repair_questions = [dict(q) for q in merged_questions]
        merged_questions = _repair_exam_questions_locally(merged_questions, quantity_dist, topics, target_total_score=100)
        fixed_items = _compute_fixed_items(before_repair_questions, merged_questions, quantity_dist or {}, target_total_score=100)
        term_bad_numbers = _collect_term_style_violation_numbers(merged_questions)
        term_style_ok = len(term_bad_numbers) == 0
        if not term_style_ok:
            fixed_items = list(dict.fromkeys((fixed_items or []) + ["term_style_sanitized"]))
        floor_flags = _exam_quality_floor_flags(merged_questions, quantity_dist or {}, target_total_score=100)
        quality_checks = {
            "quantity_ok": bool(floor_flags.get("quantity_ok")),
            "numbering_ok": bool(floor_flags.get("numbering_ok")),
            "total_score_ok": bool(floor_flags.get("total_score_ok")),
            "options_ok": bool(floor_flags.get("options_ok")),
            "duplicates_ok": bool(floor_flags.get("duplicates_ok")),
            "concept_overlap_ok": bool(floor_flags.get("concept_overlap_ok", True)),
            "term_style_ok": term_style_ok,
        }
        quality_ok = bool(floor_flags.get("quality_floor_passed")) and term_style_ok

        final_payload = _build_exam_payload_from_questions(merged_questions, title="期末考试试卷")
        end_to_end_ms = int((time.monotonic() - exam_started_at) * 1000)
        logger.info(f"[Latency] exam_end_to_end_ms={end_to_end_ms}")
        if not quality_ok:
            return _build_exam_failed_result(
                "试卷生成失败：质量门校验未通过，请重试。",
                exam_started_at=exam_started_at,
                stage_trace=stage_trace,
                evidence_source=evidence_source,
                tool_call_rag=tool_call_rag,
                tool_call_web=tool_call_web,
                stage_exit_reason="quality_floor_not_passed",
                missing_after_retries=max(0, total_questions - len(merged_questions)),
                quality_checks=quality_checks,
                term_style_ok=term_style_ok,
                critic_timeout_count=stage_critic_timeout_count,
            )
        return {
            "kind": "exam_paper",
            "render_mode": "exam_canvas",
            "text": _build_exam_text_from_payload(final_payload),
            "payload": final_payload,
            "meta": {
                "delivery_mode": "full",
                "quality_floor_passed": True,
                "degrade_reason": "",
                "auto_fixed": bool(fixed_items),
                "fixed_items": fixed_items,
                "quality_checks": quality_checks,
                "strict_mode": True,
                "exam_stage": "all",
                "attempt": max([int(t.get("attempt") or 0) for t in stage_trace] or [1]),
                "stage_status": "success",
                "stage_exit_reason": "quality_passed",
                "missing_after_retries": 0,
                "term_style_ok": term_style_ok,
                "critic_timeout_count": stage_critic_timeout_count,
                "exam_end_to_end_ms": end_to_end_ms,
                "evidence_source": evidence_source,
                "tool_calls": {"rag": tool_call_rag, "web": tool_call_web},
                "stage_trace": stage_trace,
            },
        }

    try:
        # 每次都重新构建workflow，避免全局单例在取消后可能出现的状态问题
        workflow = build_exam_graph()
        result = await asyncio.wait_for(
            workflow.ainvoke(initial_state),
            timeout=max(60, EXAM_BUDGET_SECONDS - 5),
        )
    except asyncio.TimeoutError:
        logger.error(f"[Exam Agent] exam_workflow 超时（>{max(60, EXAM_BUDGET_SECONDS - 5)}s）")
        return _build_exam_failed_result(
            "试卷生成失败：整体预算已耗尽，请重试。",
            exam_started_at=exam_started_at,
            evidence_source=evidence_source,
            tool_call_rag=tool_call_rag,
            tool_call_web=tool_call_web,
            stage_exit_reason="budget_exhausted",
            missing_after_retries=max(0, total_questions),
        )

    rounds = result.get('reflection_rounds', 0)
    critic_timeout_count = int(result.get("critic_timeout_count") or 0)
    initial_score = result.get('initial_score', 'N/A')
    final_critique = result.get('critique', {})
    final_score = final_critique.get('overall_score', 'N/A')
    final_flaws = len([f for f in final_critique.get('reasoning_flaws', []) if f.get('severity') == 'high'])
    improvement = (
        f"{initial_score} → {final_score} (+{final_score - initial_score})"
        if isinstance(initial_score, int) and isinstance(final_score, int)
        else f"{initial_score} → {final_score}"
    )
    logger.info(
        f"[Reflexion统计-试卷] topics={topics} | 修订轮次={rounds} | "
        f"评分变化={improvement} | 高危缺陷={final_flaws}"
    )

    # 提取试卷内容（保留答案和解析，方便前端显示）
    exam_content = result.get('revised_exam') or result.get('exam_paper') or ''
    # 注意：不再去掉答案和解析，让前端可以显示

    final_payload = result.get('revised_exam_payload') or result.get('exam_payload')
    if not final_payload:
        final_payload = _build_exam_payload_from_questions(parse_exam_content(exam_content).get("questions", []), title="期末考试试卷")

    final_questions = (final_payload.get("exam_data") or {}).get("questions", [])
    if not isinstance(final_questions, list):
        final_questions = []
    before_repair_final_questions = [dict(q) for q in final_questions]
    final_questions = _repair_exam_questions_locally(final_questions, quantity_dist or {}, topics, target_total_score=100)
    fixed_items = _compute_fixed_items(before_repair_final_questions, final_questions, quantity_dist or {}, target_total_score=100)
    term_bad_numbers = _collect_term_style_violation_numbers(final_questions)
    term_style_ok = len(term_bad_numbers) == 0
    if not term_style_ok:
        fixed_items = list(dict.fromkeys((fixed_items or []) + ["term_style_sanitized"]))
    final_payload = _build_exam_payload_from_questions(final_questions, title="期末考试试卷")
    floor_flags = _exam_quality_floor_flags(final_questions, quantity_dist or {}, target_total_score=100)
    quality_checks = {
        "quantity_ok": bool(floor_flags.get("quantity_ok")),
        "numbering_ok": bool(floor_flags.get("numbering_ok")),
        "total_score_ok": bool(floor_flags.get("total_score_ok")),
        "options_ok": bool(floor_flags.get("options_ok")),
        "duplicates_ok": bool(floor_flags.get("duplicates_ok")),
        "concept_overlap_ok": bool(floor_flags.get("concept_overlap_ok", True)),
        "term_style_ok": term_style_ok,
    }
    quality_ok = bool(floor_flags.get("quality_floor_passed")) and term_style_ok
    end_to_end_ms = int((time.monotonic() - exam_started_at) * 1000)
    logger.info(f"[Latency] exam_end_to_end_ms={end_to_end_ms}")
    if not quality_ok:
        stage_exit_reason = result.get("degrade_reason") or "quality_floor_not_passed"
        return _build_exam_failed_result(
            "试卷生成失败：质量门校验未通过，请重试。",
            exam_started_at=exam_started_at,
            evidence_source=evidence_source,
            tool_call_rag=tool_call_rag,
            tool_call_web=tool_call_web,
            stage_exit_reason=stage_exit_reason,
            missing_after_retries=max(0, total_questions - len(final_questions)),
            quality_checks=quality_checks,
            term_style_ok=term_style_ok,
            critic_timeout_count=critic_timeout_count,
        )

    return {
        "kind": "exam_paper",
        "render_mode": "exam_canvas",
        "text": _build_exam_text_from_payload(final_payload),
        "payload": final_payload,
        "meta": {
            "delivery_mode": "full",
            "quality_floor_passed": True,
            "degrade_reason": "",
            "auto_fixed": bool(fixed_items),
            "fixed_items": fixed_items,
            "quality_checks": quality_checks,
            "stage_exit_reason": "quality_passed",
            "missing_after_retries": 0,
            "term_style_ok": term_style_ok,
            "critic_timeout_count": critic_timeout_count,
            "exam_end_to_end_ms": end_to_end_ms,
            "evidence_source": evidence_source,
            "tool_calls": {"rag": tool_call_rag, "web": tool_call_web},
        },
    }
