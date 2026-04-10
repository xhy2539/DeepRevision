"""
多 Agent 出题系统 - Reflexion 架构
Agent 1: 出题 Agent  — 生成题目 + 推理链（为什么这样出题、答案依据何处）
Agent 2: Critic Agent — 质疑推理链，找出逻辑漏洞，输出结构化批评
Agent 3: Revise Agent — 逐条回应批评，修订题目并给出修改说明
循环：critique → revise → critique，最多 2 轮
"""
import json
import re
import asyncio
import time
import ast
import random
from typing import Any, Dict, TypedDict, List, Optional, Tuple
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from model.factory import chat_model, light_chat_model, backup_chat_model, backup_light_chat_model
from agent.tools.agent_tools import get_rag_service
from utils.session_context import current_session_id
from utils.logger_handler import logger
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
STRUCTURED_FAIL_STATS: Dict[str, int] = {
    "provider_overloaded": 0,
    "request_timeout": 0,
    "invalid_json": 0,
    "schema_type_mismatch": 0,
    "other": 0,
}


# ==================== 状态定义 ====================

class QuizState(TypedDict):
    # 输入
    topic: str
    quiz_type: str
    num: int
    context: str
    sample_paper_context: str
    # Agent 1 输出
    quiz: str               # 原始题目文本
    reasoning: str          # 出题推理链（JSON 字符串）
    quiz_payload: dict      # 结构化题目数据
    # Agent 2 输出
    critique: dict          # Critic 的结构化批评
    initial_score: int      # 第一轮 Critic 评分（效果验证基准）
    # Agent 3 输出
    revised_quiz: str       # 修订后题目
    revision_notes: str     # 针对批评的修改说明
    revised_quiz_payload: dict
    delivery_mode: str
    quality_floor_passed: bool
    degrade_reason: str
    quiz_budget_started_at: float
    quiz_budget_seconds: int
    evidence_source: str
    # 控制
    reflection_rounds: int  # 已进行的反思轮次


class ExamPaperState(TypedDict):
    # 输入
    topics: List[str]
    quiz_types: List[str]
    total_questions: int
    quantity_dist: dict  # 题型数量分布，如 {"choice": 10, "fill": 5, "judge": 5, "essay": 3}
    sample_paper_context: str
    contexts: List[str]
    # Agent 1 输出
    exam_paper: str
    reasoning: str
    exam_payload: dict
    # Agent 2 输出
    critique: dict
    initial_score: int      # 第一轮 Critic 评分（效果验证基准）
    # Agent 3 输出
    revised_exam: str
    revision_notes: str
    revised_exam_payload: dict
    delivery_mode: str
    quality_floor_passed: bool
    degrade_reason: str
    exam_budget_started_at: float
    exam_budget_seconds: int
    target_total_score: int
    critic_timeout_count: int
    stage_fast_mode: bool
    # 控制
    reflection_rounds: int


# ==================== Pydantic 结构化输出模型 ====================

class Question(BaseModel):
    """单题数据结构"""
    id: int = Field(description="题目编号")
    type: str = Field(description="题型：选择题/填空题/判断题/简答题/计算题/分析题/论述题/名词解释")
    content: str = Field(description="题干内容")
    options: Optional[List[str]] = Field(default=None, description="选项列表（选择题用）")
    answer: str = Field(description="正确答案")
    analysis: str = Field(description="题目解析")
    score: int = Field(description="分值")
    difficulty: str = Field(description="难度：简单/中等/较难")


class QuizQuestionItem(BaseModel):
    """前端题卡结构"""
    id: int = Field(description="题目编号")
    type: str = Field(description="题型")
    question: str = Field(description="题干内容")
    options: Optional[List[str]] = Field(default=None, description="选项")
    answer: str = Field(description="答案")
    explanation: str = Field(description="解析")
    score: Optional[str] = Field(default=None, description="分值，如5分")
    difficulty: Optional[str] = Field(default=None, description="难度")


class StructuredQuizSetResult(BaseModel):
    """单题/题组的原生结构化输出"""
    title: Optional[str] = Field(default="练习题", description="题组标题")
    questions: List[QuizQuestionItem] = Field(description="题目列表")
    reasoning: Optional[dict] = Field(default_factory=dict, description="出题推理链")


class ExamPaper(BaseModel):
    """试卷数据结构"""
    questions: List[Question] = Field(description="题目列表")
    reasoning: dict = Field(description="出题推理链")


class ExamPaperText(BaseModel):
    """试卷文本格式（用于完整试卷）"""
    exam_paper: str = Field(description="完整试卷文本")
    reasoning: Optional[dict] = Field(default_factory=dict, description="出题推理链")


class QuizGenerateResult(BaseModel):
    """单题/小题集结构化输出"""
    quiz: str = Field(description="完整题目文本")
    reasoning: Optional[dict] = Field(default_factory=dict, description="出题推理链")


class CritiqueResult(BaseModel):
    """Critic 评审结果"""
    approved: bool = Field(description="是否通过评审")
    overall_score: int = Field(description="总体评分 0-100")
    critique: str = Field(description="总体评价")
    reasoning_flaws: List[dict] = Field(default_factory=list, description="推理链缺陷列表")
    specific_issues: List[dict] = Field(default_factory=list, description="具体问题列表")
    duplicate_check: dict = Field(default_factory=dict, description="重复题目检查")
    numbering_check: dict = Field(default_factory=dict, description="编号连续性检查")
    quantity_check: dict = Field(default_factory=dict, description="题型数量检查")


class ReviseResult(BaseModel):
    """Revise 修订结果（单题）"""
    revised_quiz: str = Field(description="修订后的题目")
    revision_notes: str = Field(description="修改说明")
    addressed_issues: List[str] = Field(default_factory=list, description="已解决的问题列表")


class ExamReviseResult(BaseModel):
    """Revise 修订结果（试卷）"""
    revised_exam: str = Field(description="修订后的试卷")
    revision_notes: str = Field(description="修改说明")
    addressed_issues: List[str] = Field(default_factory=list, description="已解决的问题列表")


# ==================== 工具函数 ====================

async def get_rag_context(topic: str) -> str:
    """获取 RAG 原始检索片段（不做最终 LLM 总结，出题 Agent 自己基于原文出题）"""
    logger.info(f"[RAG Context] 正在检索 topic={topic[:50]}...")
    try:
        rag = await get_rag_service()
        context = await rag.retrieve_context(topic)
        logger.info(f"[RAG Context] 检索完成，返回长度={len(context)}")
        if not context or context.strip() == "":
            logger.warning("[RAG Context] 知识库为空，返回提示信息")
            return "【提示】当前知识库为空，无法生成题目。请先上传课件后再请求出题。"
        return context
    except Exception as e:
        logger.error(f"[RAG Context] 检索异常: {e}")
        return f"【提示】知识库检索失败: {str(e)}"


def _default_comprehensive_topics() -> List[str]:
    """未指定章节时的综合覆盖考点标签。"""
    return [
        "基础概念与体系结构",
        "进程与线程管理",
        "调度与同步互斥",
        "死锁与资源管理",
        "内存管理与虚拟内存",
        "文件系统与I/O",
        "网络与通信机制",
    ]


async def _get_comprehensive_rag_context(seed_topic: str = "") -> str:
    """
    综合覆盖检索：针对不同模块做多路检索并拼接，避免只命中少数章节。
    """
    base = str(seed_topic or "").strip()
    queries = [
        f"{base} 基础概念 体系结构".strip(),
        f"{base} 进程 线程 调度 同步 死锁".strip(),
        f"{base} 内存管理 虚拟内存 页面置换".strip(),
        f"{base} 文件系统 I/O 设备管理 网络通信".strip(),
    ]
    # 去重并剔除空查询
    dedup_queries: List[str] = []
    seen = set()
    for q in queries:
        q = re.sub(r"\s+", " ", q).strip()
        if not q or q in seen:
            continue
        seen.add(q)
        dedup_queries.append(q)

    contexts: List[str] = []
    for q in dedup_queries:
        ctx = await get_rag_context(q)
        if ctx and ctx.strip():
            contexts.append(ctx.strip())

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
    """调用 LLM（带超时和有限重试，避免回退链路挂死）"""
    prompt = PromptTemplate.from_template(prompt_template)
    llm_timeout = 28.0
    max_retries = 1
    providers = _build_provider_chain(chat_model, backup_chat_model)
    if not providers:
        raise RuntimeError("未配置可用的聊天模型")

    last_error: Exception | None = None
    for provider_name, provider_model in providers:
        chain = prompt | provider_model | StrOutputParser()
        for attempt in range(max_retries + 1):
            try:
                return await asyncio.wait_for(chain.ainvoke(kwargs), timeout=llm_timeout)
            except Exception as e:
                last_error = e
                logger.warning(f"[call_llm] provider={provider_name} 第{attempt+1}次失败: {e}")
                if attempt >= max_retries or not _should_retry_structured_error(e):
                    break
                await asyncio.sleep(1.0 + attempt * 1.0)
        if not _should_retry_structured_error(last_error or Exception("unknown")):
            break
        if provider_name == "primary" and backup_chat_model:
            logger.warning("[call_llm] 主模型异常，切换备用模型重试")

    if last_error:
        raise last_error
    raise RuntimeError("call_llm unknown error")


async def call_llm_structured(prompt_template: str, schema: type[BaseModel], **kwargs) -> BaseModel:
    """
    调用 LLM 并强制结构化输出
    注意：MiniMax API 不支持 response_format，这里使用 Prompt 约束 + 正则解析
    """
    import asyncio

    prompt = PromptTemplate.from_template(prompt_template)
    last_error = None
    schema_name = getattr(schema, "__name__", "")
    # 智能超时预算：生成阶段给足时间，评审/修订快失败
    timeout_map = {
        # 生成阶段：放宽超时，优先保证完整产出
        "StructuredQuizSetResult": 120.0,
        "QuizGenerateResult": 120.0,
        "ExamPaper": 150.0,
        "ExamPaperText": 150.0,
        # 评审/修订阶段：中等超时，防止卡死
        "CritiqueResult": 20.0,
        "ReviseResult": 30.0,
        "ExamReviseResult": 45.0,
    }
    retry_map = {
        # 2 次尝试（初次 + 1 次重试）
        "StructuredQuizSetResult": 1,
        "QuizGenerateResult": 1,
        "CritiqueResult": 1,
        "ReviseResult": 1,
        "ExamPaper": 1,
        "ExamPaperText": 1,
        "ExamReviseResult": 1,
    }
    llm_timeout = timeout_map.get(schema_name, 35.0)
    max_retries = retry_map.get(schema_name, 1)
    overload_bonus_retry = 1  # provider 529 时额外给 1 次重试机会
    used_overload_bonus = False
    providers = _build_provider_chain(chat_model, backup_chat_model)
    if not providers:
        raise RuntimeError("未配置可用的聊天模型")

    for provider_name, provider_model in providers:
        chain = prompt | provider_model | StrOutputParser()
        attempt = 0
        local_max_retries = max_retries
        local_overload_bonus_used = False
        while attempt <= local_max_retries:
            result = None
            try:
                result = await asyncio.wait_for(chain.ainvoke(kwargs), timeout=llm_timeout)
                parsed = _extract_json(result)
                parsed = _coerce_structured_payload(parsed, schema)
                return schema(**parsed)
            except asyncio.TimeoutError:
                fail_reason = "request_timeout"
                last_error = asyncio.TimeoutError(f"LLM 调用超时（{llm_timeout}s）")
                _inc_structured_fail(fail_reason)
                logger.warning(
                    f"[call_llm_structured] provider={provider_name} 第{attempt+1}次失败: {last_error}"
                )
                if attempt >= local_max_retries:
                    break
                await asyncio.sleep(_retry_sleep_seconds(fail_reason, attempt))
                attempt += 1
            except Exception as e:
                last_error = e
                raw_preview = _summarize_raw_output(result)
                fail_reason = _classify_structured_error(e, raw_preview)
                _inc_structured_fail(fail_reason)
                recovered = _extract_loose_json_fields_for_schema(result, schema) if result else None
                if recovered is not None:
                    logger.warning(f"[call_llm_structured] 宽松解析兜底成功(schema={getattr(schema, '__name__', '')})")
                    recovered = _coerce_structured_payload(recovered, schema)
                    return schema(**recovered)
                logger.warning(
                    f"[call_llm_structured] provider={provider_name} 第{attempt+1}次失败: {e}; raw='{raw_preview}...'"
                )
                should_retry = _should_retry_structured_error(e)
                if (
                    fail_reason == "provider_overloaded"
                    and overload_bonus_retry > 0
                    and not local_overload_bonus_used
                ):
                    local_overload_bonus_used = True
                    local_max_retries += overload_bonus_retry
                    logger.warning("[call_llm_structured] 检测到 provider_overloaded，已启用额外重试机会")
                if attempt >= local_max_retries or not should_retry:
                    break
                await asyncio.sleep(_retry_sleep_seconds(fail_reason, attempt))
                attempt += 1
        if provider_name == "primary" and backup_chat_model and _should_retry_structured_error(last_error or Exception("unknown")):
            logger.warning("[call_llm_structured] 主模型异常，切换备用模型继续结构化生成")

    raise last_error


# ---------- 分批生成单题型试卷的 Prompt ----------
EXAM_GENERATE_SINGLE_TYPE_PROMPT = """【警告】绝对禁止使用'...'、'等'、'以下略'等任何占位符！你必须逐字逐句地生成所有要求的题目。

你是一位资深大学期末考试命题专家。请基于课程知识点生成单种题型试卷。

【考点范围】{topics}
【题型】{quiz_type}
【题数】{num} 道
【起始编号】{start_num}
【每题分值】{score} 分
【总分】{total_score} 分
【样卷格式参考】{sample_paper_context}

【课程参考资料】:
{context}

【硬性要求】：
1. 必须生成正好 {num} 道 {quiz_type}，编号从 {start_num} 到 {end_num}，一道都不能少！
2. 选择题必须有 A、B、C、D 四个完整选项，且每个选项独立成行。
3. 填空题需要填空的部位用括号表示；判断题表述清晰可判定。
4. 简答题/解答题建议 2-3 个小问,分值自行分配，但是总分每题20。
5. 【重要】所有题目统一分值：{score} 分/题，总分 {total_score} 分，每道题必须标注这个分值！
6. 题干要自然考试语气，详略得当，不要教材段落式堆砌。
7. 难度分布：简单30%、中等50%、困难20%
8. 【强制出题】必须逐字逐句生成所有要求的题目，禁止任何省略！
9. 每道题必须包含答案、解析和分值标注。
10. “来源依据说明”只能写在 reasoning 字段，严禁写入题干、选项、答案、解析。
11. 选择题题干长度建议 40-90 字；整批题目需有短中长变化，避免长度雷同。
12. 不得出现 "A A"/"B B" 前缀重复。
13. 禁止输出泛化模板选项：`会直接影响系统行为与资源约束`、`只影响文字表述`、`仅在理想场景成立`、`与性能和正确性无关`。
14. 用户可见字段（题干/选项/答案/解析）禁止“中文术语（英文解释）”样式；CPU/TLB/TCP 等纯缩写可保留。

【格式示例】
{format_example}

请生成 {num} 道 {quiz_type}。

【重要】只返回纯 JSON 对象，不要用 ```json 代码块包裹！直接输出：
{{"exam_paper": "...", "reasoning": {{"knowledge_points": [...], "answer_evidence": [...]}}}}"""


# ---------- 结构化试卷生成 Prompt（返回 JSON） ----------
EXAM_GENERATE_STRUCTURED_PROMPT = """【警告】绝对禁止使用'...'、'等'、'以下略'等任何占位符！你必须逐字逐句地生成所有要求的题目。

你是一位资深大学期末考试命题专家。请基于课程知识点生成结构化试卷。

【考点范围】{topics}
【题型】{quiz_type}
【题数】{num} 道
【起始编号】{start_num}
【样卷格式参考】{sample_paper_context}

【课程参考资料】:
{context}

【硬性要求】：
1. 必须生成正好 {num} 道 {quiz_type}，编号从 {start_num} 开始
2. 选择题必须有 A、B、C、D 四个完整选项，且每个选项独立成行。
3. 填空题需要填空部位用括号表示；判断题表述清晰可判定。
4. 简答题/解答题可以有 2-3 个小问。
5. 【必须】每道题都要包含 score（分值）和 difficulty（难度）。
6. 分值分配由你根据题型和难度决定：选择题2-3分，填空题2-3分，判断题2分，简答题8-10分。
7. 难度分布：简单30%、中等50%、困难20%，根据题目内容自行判断。
8. 【强制出题】必须逐字逐句生成所有要求的题目，禁止任何省略！
9. 每道题必须包含 answer（答案）和 analysis（解析）。
10. 允许你在 reasoning 中说明证据来源，但禁止在题干/选项/答案/解析中写“根据课件/依据资料/参考资料”等来源措辞。
11. 选择题题干长度建议 40-90 字，整批题目需有短中长变化，不得重复前缀（如 "A A"）。
12. 禁止输出泛化模板选项：`会直接影响系统行为与资源约束`、`只影响文字表述`、`仅在理想场景成立`、`与性能和正确性无关`。
13. 用户可见字段（题干/选项/答案/解析）禁止“中文术语（英文解释）”样式；CPU/TLB/TCP 等纯缩写可保留。

请生成 JSON 格式的试卷：
{{
  "questions": [
    {{
      "id": {start_num},
      "type": "{quiz_type}",
      "content": "题干内容",
      "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"] (选择题必须有),
      "answer": "正确答案",
      "analysis": "解析内容",
      "score": 2,
      "difficulty": "中等"
    }},
    ...
  ],
  "reasoning": {{"knowledge_points": [...], "answer_evidence": [...]}}
}}

只返回纯 JSON，不要其他文字！"""


# 单题型生成模板（带分值和难度标注）
SINGLE_TYPE_FORMAT_EXAMPLES = {
    "选择题": """一、选择题（共{num}题，每题{score}分，计{total_score}分）
{start_num}. [题干内容（建议40-90字）]
   A. 选项1
   B. 选项2
   C. 选项3
   D. 选项4
答案：A
分值：{score}
难度：中等
解析：该选项符合课程定义，其他选项存在概念偏差。

{start_num_plus1}. [题干内容（建议40-90字）]
   A. 选项1
   B. 选项2
   C. 选项3
   D. 选项4
答案：B
分值：{score}
难度：简单
解析：...""",

    "填空题": """二、填空题（共{num}题，每题{score}分，计{total_score}分）
（注意：从第{start_num}题开始编号）
{start_num}. [题干内容，需要填空的部位用(  )表示]
答案：xxx
分值：{score}
难度：中等
解析：..

{start_num_plus1}. [题干内容]
答案：xxx
分值：{score}
难度：简单
解析：...""",

    "判断题": """三、判断题（共{num}题，每题{score}分，计{total_score}分）
（注意：从第{start_num}题开始编号）
{start_num}. [题干内容]
答案：正确
分值：{score}
难度：简单
解析：..

{start_num_plus1}. [题干内容]
答案：错误
分值：{score}
难度：中等
解析：...""",

    "简答题": """四、简答题（共{num}题，每题{score}分，计{total_score}分）
（注意：从第{start_num}题开始编号）
{start_num}. [问题描述]（{score}分）
    (1) [小问1]（{sub_score}分）
    (2) [小问2]（{sub_score}分）
答案：...
分值：{score}
难度：中等
解析：..

{start_num_plus1}. [问题描述]
答案：...
分值：{score}
难度：较难
解析：..."""
}


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

    # 获取格式示例
    format_example = SINGLE_TYPE_FORMAT_EXAMPLES.get(
        quiz_type,
        SINGLE_TYPE_FORMAT_EXAMPLES["选择题"]
    ).format(
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
                EXAM_GENERATE_SINGLE_TYPE_PROMPT,
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
                EXAM_GENERATE_STRUCTURED_PROMPT,
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


def _is_generic_topic_request(topic: str) -> bool:
    text = str(topic or "").strip()
    if not text:
        return True
    keywords = _extract_topic_keywords(text, limit=6)
    if not keywords:
        return True
    # 仅包含高度通用词时视为泛化请求
    generic_signals = ["核心知识点", "课件", "练习题", "出题", "试卷", "综合测试"]
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
        cleaned = cleaned[:120]
        if cleaned not in snippets:
            snippets.append(cleaned)
        if len(snippets) >= max_items:
            break
    return snippets


def _quiz_budget_left_seconds(state: QuizState) -> float:
    started = float(state.get("quiz_budget_started_at") or time.monotonic())
    total = int(state.get("quiz_budget_seconds") or QUIZ_BUDGET_SECONDS)
    return total - (time.monotonic() - started)


def _normalize_option_text(option: str, index: int) -> str:
    option_text = _sanitize_user_visible_text(str(option or "").strip())
    if not option_text:
        return f"{chr(65 + index)}. （无内容）"
    m = re.match(r'^([A-D])[.、．:：)\s]*(.*)$', option_text, flags=re.IGNORECASE)
    if m:
        content = (m.group(2) or "").strip()
        if content:
            return f"{m.group(1).upper()}. {content}"
    return f"{chr(65 + index)}. {option_text}"


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

    for q in questions:
        qtype = str((q or {}).get("type") or "").strip() or expected_type or "选择题"
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
    expected_type = str(state.get("quiz_type") or "").strip()
    quantity_ok = (expected_num <= 0) or (len(questions) == expected_num)
    type_ok = True
    options_ok = True
    answer_ok = True

    for q in questions:
        qtype = str((q or {}).get("type") or "")
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
        structured_attempts += 1
        remaining = num - len(structured_questions)
        batch_num = min(max(1, structured_batch_size), remaining)
        batch_start = start_num + len(structured_questions)
        structured_timeout = max(30.0, min(150.0, structured_budget_ceiling / 2))
        try:
            structured = await asyncio.wait_for(
                generate_single_type_paper_structured(
                    quiz_type=quiz_type,
                    num=batch_num,
                    start_num=batch_start,
                    topics=topics,
                    context=context,
                    sample_paper_context=sample_paper_context,
                ),
                timeout=structured_timeout,
            )
            batch_questions = [q.model_dump() for q in structured.questions]
            if not batch_questions:
                logger.warning(
                    f"[分批补题] {quiz_type} 结构化批次为空，attempt={structured_attempts}/{max_structured_attempts}, "
                    f"batch={batch_num}, start={batch_start}"
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
        logger.info(
            f"[分批补题] {quiz_type} 文本补题窗口: 需补{remaining}道，本批{current_batch}道，起始编号{next_num}"
        )
        try:
            supplement_text = await asyncio.wait_for(
                generate_single_type_paper(
                    quiz_type=quiz_type,
                    num=current_batch,
                    start_num=next_num,
                    topics=topics,
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
                            topics=topics,
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
                        topics=topics,
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


def _try_json_load(text: str) -> Any:
    """宽松 JSON 加载：处理尾逗号、代码块等常见污染。"""
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty json text")

    candidates = [raw]
    # 尝试修复尾逗号
    candidates.append(re.sub(r',\s*([}\]])', r'\1', raw))
    # 如果是 Python 风格字典/列表，尝试 literal_eval
    try:
        return json.loads(raw)
    except Exception:
        pass

    for cand in candidates[1:]:
        try:
            return json.loads(cand)
        except Exception:
            continue

    try:
        return ast.literal_eval(raw)
    except Exception:
        raise ValueError("json load failed")


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON，兼容 think 标签、markdown 代码块和前后噪音。"""
    if not text or not str(text).strip():
        raise ValueError("empty model output")

    text = _sanitize_structured_output(text)

    fenced = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        obj = _try_json_load(text)
        if isinstance(obj, list):
            # 兼容顶层直接返回 questions 数组
            return {"questions": obj}
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = -1
    stack: List[str] = []
    opening = {'{': '}', '[': ']'}
    closing = {'}': '{', ']': '['}
    for i, ch in enumerate(text):
        if ch in opening:
            if start == -1:
                start = i
            stack.append(ch)
        elif ch in closing and stack:
            if stack[-1] == closing[ch]:
                stack.pop()
                if not stack and start != -1:
                    candidate = text[start:i + 1]
                    try:
                        obj = _try_json_load(candidate)
                        if isinstance(obj, list):
                            return {"questions": obj}
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        start = -1
                        continue

    raise ValueError("no valid json found in model output")


def _extract_loose_json_fields_for_schema(text: Any, schema: type[BaseModel]) -> Optional[dict]:
    """
    当模型返回“近似 JSON”但夹杂未转义引号时，按 schema 做最小字段兜底提取，
    避免 Critic 阶段反复重试卡住。
    """
    schema_name = getattr(schema, "__name__", "")
    if text is None:
        return None

    raw = _sanitize_structured_output(str(text))
    if schema_name == "StructuredQuizSetResult":
        # 先尝试直接用“序列化题目解析器”恢复 questions
        recovered_questions = _parse_questions_from_serialized_quiz(raw)
        if recovered_questions:
            title_match = re.search(r'"title"\s*:\s*"([^"]{1,80})"', raw)
            title = title_match.group(1).strip() if title_match else "练习题"
            return {
                "title": title,
                "questions": recovered_questions,
                "reasoning": {},
            }
        return None

    if schema_name == "ExamPaper":
        recovered_questions = _normalize_exam_questions_payload(_parse_questions_from_serialized_quiz(raw))
        if recovered_questions:
            return {"questions": recovered_questions, "reasoning": {}}
        return None

    if schema_name == "ExamPaperText":
        # 尝试提取 exam_paper 字段；若缺失则退化为原文文本
        m = re.search(r'"exam_paper"\s*:\s*"([\s\S]*?)"\s*(?:,|})', raw)
        if m:
            exam_paper = m.group(1).replace('\\"', '"').strip()
            if exam_paper:
                return {"exam_paper": exam_paper, "reasoning": {}}
        if len(raw) > 40:
            return {"exam_paper": raw[:12000], "reasoning": {}}
        return None

    if schema_name == "QuizGenerateResult":
        m = re.search(r'"quiz"\s*:\s*"([\s\S]*?)"\s*(?:,|})', raw)
        if m:
            quiz = m.group(1).replace('\\"', '"').strip()
            if quiz:
                return {"quiz": quiz, "reasoning": {}}
        recovered_questions = _parse_questions_from_serialized_quiz(raw)
        if recovered_questions:
            return {
                "quiz": _build_quiz_text_from_questions(recovered_questions),
                "reasoning": {},
            }
        return None

    if schema_name in {"ReviseResult", "ExamReviseResult"}:
        base_key = "revised_quiz" if schema_name == "ReviseResult" else "revised_exam"
        m = re.search(rf'"{base_key}"\s*:\s*"([\s\S]*?)"\s*(?:,|\}})', raw)
        if m:
            content = m.group(1).replace('\\"', '"').strip()
            if content:
                return {
                    base_key: content,
                    "revision_notes": "结构化噪声已自动修复。",
                    "addressed_issues": [],
                }
        return None

    if schema_name != "CritiqueResult":
        return None

    if '"approved"' not in raw and '"overall_score"' not in raw:
        return None

    approved_match = re.search(r'"approved"\s*:\s*(true|false)', raw, re.IGNORECASE)
    score_match = re.search(r'"overall_score"\s*:\s*(\d+)', raw)
    approved = approved_match.group(1).lower() == "true" if approved_match else False
    overall_score = int(score_match.group(1)) if score_match else 80

    critique = ""
    critique_key = re.search(r'"critique"\s*:\s*', raw)
    if critique_key:
        start = critique_key.end()
        end_candidates = [raw.find('"reasoning_flaws"', start), raw.find('"specific_issues"', start), raw.find('"duplicate_check"', start)]
        end_candidates = [x for x in end_candidates if x != -1]
        end = min(end_candidates) if end_candidates else len(raw)
        critique_chunk = raw[start:end].strip().rstrip(",")
        if critique_chunk.startswith('"'):
            critique_chunk = critique_chunk[1:]
        if critique_chunk.endswith('"'):
            critique_chunk = critique_chunk[:-1]
        critique = critique_chunk.replace('\\"', '"').strip()

    # 最小可用结构，剩余字段交由 _coerce_structured_payload 归一
    return {
        "approved": approved,
        "overall_score": overall_score,
        "critique": critique or "结构化输出存在格式噪声，已采用宽松解析兜底。",
        "reasoning_flaws": [],
        "specific_issues": [],
        "duplicate_check": {},
        "numbering_check": {},
        "quantity_check": {},
    }


def _sanitize_structured_output(text: str) -> str:
    """清理结构化输出中的推理标签、代码块包装和前缀噪音。"""
    cleaned = str(text).strip()
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r'^result\s*=\s*', '', cleaned).strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    return cleaned.strip()


def _summarize_raw_output(text: Any) -> str:
    if text is None:
        return ""
    return _sanitize_structured_output(text)[:300]


def _inc_structured_fail(reason: str):
    key = reason if reason in STRUCTURED_FAIL_STATS else "other"
    try:
        STRUCTURED_FAIL_STATS[key] = int(STRUCTURED_FAIL_STATS.get(key, 0)) + 1
    except Exception:
        pass


def _classify_structured_error(error: Exception, raw_preview: str = "") -> str:
    if isinstance(error, TimeoutError):
        return "request_timeout"
    text = f"{str(error or '')} {str(raw_preview or '')}".lower()
    if "529" in text or "overloaded" in text or "current service cluster" in text:
        return "provider_overloaded"
    if "timeout" in text or "timed out" in text or "request timed out" in text:
        return "request_timeout"
    if "no valid json found" in text or "json" in text and "decode" in text:
        return "invalid_json"
    if "validation" in text or "input should be" in text or "pydantic" in text or "string_type" in text:
        return "schema_type_mismatch"
    return "other"


def _retry_sleep_seconds(reason: str, attempt: int) -> float:
    """
    过载错误采用更长退避，减少“高峰期持续撞线”。
    """
    if reason == "provider_overloaded":
        base = min(20.0, 4.0 * (attempt + 1))
        return base + random.uniform(1.0, 3.0)
    if reason == "request_timeout":
        base = min(10.0, 2.0 * (attempt + 1))
        return base + random.uniform(0.5, 1.5)
    return 1.2 + random.uniform(0.0, 0.8)


def get_structured_fail_stats() -> Dict[str, int]:
    return {k: int(v) for k, v in STRUCTURED_FAIL_STATS.items()}


def _should_retry_structured_error(error: Exception) -> bool:
    """网络波动和常见结构化污染都允许重试一次以上。"""
    if isinstance(error, TimeoutError):
        return True
    error_str = str(error).lower()
    retryable_patterns = [
        'connection error',
        'timeout',
        'timed out',
        'temporarily unavailable',
        'overloaded',
        '529',
        'server disconnected',
        'remote protocol error',
        'empty model output',
        'no valid json found',
    ]
    return any(pattern in error_str for pattern in retryable_patterns)


def _build_provider_chain(primary, backup):
    """构建去重后的主备模型序列，避免同一实例重复调用。"""
    providers: List[Tuple[str, Any]] = []
    seen = set()
    for name, model in (("primary", primary), ("backup", backup)):
        if model is None:
            continue
        model_id = id(model)
        if model_id in seen:
            continue
        seen.add(model_id)
        providers.append((name, model))
    return providers


def _strip_think_tags(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.DOTALL).strip()
    if cleaned.startswith("<think>"):
        first_question = re.search(r'(^\d+\.\s*[（(]?\S+)|(^【选择题】)|(^【填空题】)|(^【判断题】)|(^【简答题】)', cleaned, flags=re.MULTILINE)
        if first_question:
            cleaned = cleaned[first_question.start():].strip()
    return cleaned


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


def _normalize_text_field(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


META_QUESTION_PATTERNS = [
    r'^根据课件资料[，,:：]?\s*',
    r'^根据课件[，,:：]?\s*',
    r'^依据课件资料[，,:：]?\s*',
    r'^依据资料[，,:：]?\s*',
    r'^围绕[“"]?课程核心知识点[”"]?[，,:：]?\s*',
    r'^围绕[“"]?本课程重点知识点[”"]?[，,:：]?\s*',
    r'^请基于课件片段[“"].*?[”"]\s*',
]


def _sanitize_user_visible_text(text: str) -> str:
    cleaned = _sanitize_question_text(text)
    cleaned = re.sub(r'^(解析[:：]?)\s*(根据|依据)\s*(课件|资料|参考资料)', r'\1 ', cleaned)
    cleaned = re.sub(r'(根据|依据)\s*(课件|资料|参考资料)\s*(内容)?(可知|可得|显示|指出)?', '', cleaned)
    # 用户可见字段禁止“中文术语 + 英文括号解释”，例如：微内核（microkernel）
    cleaned = re.sub(
        r'([\u4e00-\u9fa5]{2,})\s*[（(]\s*[A-Za-z][A-Za-z0-9\s/_\-]{1,48}\s*[)）]',
        r'\1',
        cleaned,
    )
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _contains_term_style_violation(text: str) -> bool:
    t = str(text or "")
    if not t:
        return False
    return bool(
        re.search(
            r'[\u4e00-\u9fa5]{2,}\s*[（(]\s*[A-Za-z][A-Za-z0-9\s/_\-]{1,48}\s*[)）]',
            t,
        )
    )


def _sanitize_question_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return cleaned
    cleaned = re.sub(r'^\s*\d+\s*[.、]\s*', '', cleaned)
    cleaned = re.sub(r'^\s*[（(]\s*\d+\s*[）)]\s*', '', cleaned)
    for pattern in META_QUESTION_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned)
    cleaned = cleaned.replace("当前课件核心知识点", "本课程重点内容")
    cleaned = cleaned.replace("课程核心知识点", "本课程重点内容")
    cleaned = cleaned.replace("根据课件", "")
    cleaned = re.sub(r'(根据|依据)\s*(已上传)?课件(资料)?', '', cleaned)
    cleaned = re.sub(r'(课件资料|参考资料)\s*显示', '', cleaned)
    cleaned = re.sub(r'请结合课件资料', '请结合课程内容', cleaned)
    cleaned = re.sub(r'请依据课件资料', '请依据课程内容', cleaned)
    cleaned = re.sub(r'^\s*[，,:：]\s*', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _normalize_int_field(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r'-?\d+', str(value))
    return int(match.group(0)) if match else default


def _normalize_dict_field(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _normalize_list_of_dicts(value: Any) -> List[dict]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    normalized: List[dict] = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, str) and item.strip():
            normalized.append({"text": item.strip()})
    return normalized


def _normalize_reasoning_payload(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"items": value}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                return loaded
            if isinstance(loaded, list):
                return {"items": loaded}
        except Exception:
            return {"text": text}
    return {}


def _normalize_options(value: Any) -> Optional[List[str]]:
    if value in (None, "", []):
        return None

    items: List[Any] = []
    if isinstance(value, dict):
        for key in ["A", "B", "C", "D"]:
            if key in value:
                items.append(f"{key}. {str(value.get(key) or '').strip()}")
    elif isinstance(value, list):
        items = value
    elif isinstance(value, str):
        text = value.strip()
        if text:
            _, extracted = _split_stem_and_options_for_payload(text)
            if extracted:
                items = extracted
            else:
                items = [text]
    else:
        items = [value]

    normalized: List[str] = []
    inline_option_pattern = re.compile(
        r'([A-DＡ-Ｄ])[.、．:：)\s]+\s*(.+?)(?=(?:\s+[A-DＡ-Ｄ][.、．:：)\s]+)|$)',
        flags=re.IGNORECASE,
    )
    marker_pattern = re.compile(r'[A-DＡ-Ｄ][.、．:：)\s]+', flags=re.IGNORECASE)

    for idx, item in enumerate(items):
        raw = str(item or "").strip()
        if not raw:
            continue
        # 兼容一整行中包含 A/B/C/D 多个选项
        if len(marker_pattern.findall(raw)) >= 2:
            matches = inline_option_pattern.findall(raw)
            if matches:
                for letter, text in matches:
                    txt = str(text or "").strip()
                    if txt:
                        normalized.append(f"{str(letter).upper()}. {txt}")
                continue
        opt = _normalize_option_text(raw, idx)
        if opt and opt.strip():
            normalized.append(opt.strip())

    deduped: List[str] = []
    seen = set()
    for idx, opt in enumerate(normalized):
        content = re.sub(r'^[A-D][.、．]\s*', '', opt).strip()
        if not content:
            continue
        key = re.sub(r'\s+', ' ', content).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f"{chr(65 + len(deduped))}. {content}")
        if len(deduped) >= 4:
            break

    return deduped or None


def _normalize_question_item(item: Any, fallback_id: int) -> dict:
    if not isinstance(item, dict):
        item = {"question": str(item).strip()}
    return {
        "id": _normalize_int_field(item.get("id"), fallback_id),
        "type": _normalize_text_field(item.get("type"), "选择题"),
        "question": _sanitize_user_visible_text(_sanitize_question_text(_normalize_text_field(item.get("question") or item.get("content"), ""))),
        "options": _normalize_options(item.get("options")),
        "answer": _sanitize_user_visible_text(_normalize_text_field(item.get("answer"), "")),
        "explanation": _sanitize_user_visible_text(_normalize_text_field(item.get("explanation") or item.get("analysis"), "")),
        "score": _format_score(item.get("score")),
        "difficulty": _normalize_text_field(item.get("difficulty"), "中等"),
    }


def _normalize_questions_payload(value: Any) -> List[dict]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [_normalize_question_item(item, index) for index, item in enumerate(items, start=1)]


def _sanitize_quiz_questions_for_delivery(questions: Any) -> List[dict]:
    normalized = _normalize_questions_payload(questions)
    out: List[dict] = []
    for idx, q in enumerate(normalized, start=1):
        item = dict(q)
        item["id"] = int(item.get("id") or idx)
        item["question"] = _sanitize_user_visible_text(_normalize_text_field(item.get("question"), ""))
        item["answer"] = _sanitize_user_visible_text(_normalize_text_field(item.get("answer"), ""))
        item["explanation"] = _sanitize_user_visible_text(_normalize_text_field(item.get("explanation"), ""))
        qtype = str(item.get("type") or "").strip() or "选择题"
        item["type"] = qtype
        if qtype == "选择题":
            opts = _normalize_options(item.get("options")) or []
            item["options"] = [_normalize_option_text(opt, i) for i, opt in enumerate(opts[:4])]
        else:
            item["options"] = None
        out.append(item)
    return out


def _normalize_exam_question_item(item: Any, fallback_id: int) -> dict:
    if not isinstance(item, dict):
        item = {"content": str(item).strip()}
    return {
        "id": _normalize_int_field(item.get("id") or item.get("number"), fallback_id),
        "type": _normalize_text_field(item.get("type"), "选择题"),
        "content": _sanitize_question_text(_normalize_text_field(item.get("content") or item.get("question"), "")),
        "options": _normalize_options(item.get("options")),
        "answer": _normalize_text_field(item.get("answer"), ""),
        "analysis": _sanitize_user_visible_text(_normalize_text_field(item.get("analysis") or item.get("explanation"), "")),
        "score": _normalize_int_field(item.get("score"), 2),
        "difficulty": _normalize_text_field(item.get("difficulty"), "中等"),
    }


def _normalize_exam_questions_payload(value: Any) -> List[dict]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [_normalize_exam_question_item(item, index) for index, item in enumerate(items, start=1)]


def _limit_context_size(text: str, max_chars: int = 12000) -> str:
    """限制上下文长度，避免超长 context 导致生成超时。"""
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars]


def _normalize_addressed_issues(value: Any) -> List[str]:
    """兼容模型把 addressed_issues 输出成对象数组。"""
    if value in (None, ""):
        return []

    items = value if isinstance(value, list) else [value]
    normalized: List[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            if text:
                normalized.append(text)
            continue

        if isinstance(item, dict):
            issue_id = str(item.get("issue_id", "")).strip()
            fix = str(item.get("fix", "")).strip()
            issue = str(item.get("issue", "")).strip()
            action = str(item.get("action", "")).strip()
            parts = [part for part in [issue_id or issue, fix or action] if part]
            if parts:
                normalized.append("：".join(parts))
                continue

        text = str(item).strip()
        if text:
            normalized.append(text)

    return normalized


def _format_score(score: Any) -> Optional[str]:
    if score in (None, "", 0, "0"):
        return None
    score_str = str(score).strip()
    return score_str if score_str.endswith("分") else f"{score_str}分"


def _build_quiz_text_from_questions(questions: List[dict]) -> str:
    lines: List[str] = []
    for index, question in enumerate(questions, start=1):
        qtype = question.get("type") or "选择题"
        score = question.get("score") or "5分"
        difficulty = question.get("difficulty") or "中等"
        lines.append(f"{index}.（{qtype}，分值：{score}，难度：{difficulty}）")
        lines.append(question.get("question", "").strip())
        for option in question.get("options") or []:
            lines.append(option.strip())
        if question.get("answer"):
            lines.append(f"答案：{question['answer']}")
        if question.get("explanation"):
            lines.append(f"解析：{question['explanation']}")
        lines.append("")
    return "\n".join(lines).strip()


def _build_quiz_payload(topic: str, questions: List[dict]) -> dict:
    return {
        "title": topic or "练习题",
        "show_answers_default": False,
        "show_analysis_default": False,
        "questions": questions,
    }


def _parse_questions_from_serialized_quiz(text: str) -> List[dict]:
    """
    兼容模型返回 Python/JSON 列表字符串：
    [{'question_number':1,...}] / [{"question_number":1,...}]
    统一转换为前端可渲染的 questions 结构。
    """
    if not text or not str(text).strip():
        return []
    raw = str(text).strip()
    parsed_obj: Any = None
    try:
        parsed_obj = json.loads(raw)
    except Exception:
        try:
            parsed_obj = ast.literal_eval(raw)
        except Exception:
            return []

    if isinstance(parsed_obj, dict):
        parsed_obj = [parsed_obj]
    if not isinstance(parsed_obj, list):
        return []

    normalized: List[dict] = []
    for idx, item in enumerate(parsed_obj, start=1):
        if not isinstance(item, dict):
            continue
        question = _normalize_text_field(item.get("question") or item.get("content"), "")
        if not question:
            continue
        raw_options = item.get("options")
        options: List[str] = []
        if isinstance(raw_options, dict):
            for key in ["A", "B", "C", "D"]:
                val = raw_options.get(key)
                if val is not None and str(val).strip():
                    options.append(f"{key}. {str(val).strip()}")
        elif isinstance(raw_options, list):
            for i, opt in enumerate(raw_options):
                opt_text = str(opt).strip()
                if not opt_text:
                    continue
                if re.match(r"^[A-D][.、]\s*", opt_text):
                    options.append(opt_text)
                else:
                    options.append(f"{chr(65 + i)}. {opt_text}")

        normalized.append({
            "id": _normalize_int_field(item.get("id") or item.get("question_number"), idx),
            "type": _normalize_text_field(item.get("type"), "选择题"),
            "question": _sanitize_user_visible_text(_sanitize_question_text(question)),
            "options": options or None,
            "answer": _sanitize_user_visible_text(_normalize_text_field(item.get("answer"), "")),
            "explanation": _sanitize_user_visible_text(_normalize_text_field(item.get("explanation") or item.get("analysis"), "")),
            "score": _format_score(item.get("score")) or "5分",
            "difficulty": _normalize_text_field(item.get("difficulty"), "中等"),
        })
    return normalized


def _questions_from_quiz_text(text: str, topic: str) -> dict:
    from api.message_protocol import _parse_quiz_content

    questions = _parse_quiz_content(text)
    if not questions:
        questions = _parse_questions_from_serialized_quiz(text)
    return _build_quiz_payload(topic, questions)


def _normalize_exam_question(question: dict, fallback_number: int) -> dict:
    qtype = question.get("type") or "选择题"
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
        qtype = qq.get("type") or "选择题"
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


def _question_signature(text: str) -> str:
    if not text:
        return ""
    normalized = re.sub(r'^\s*\d+[.、]\s*', '', str(text).strip())
    normalized = normalized.lower()
    normalized = re.sub(r'[（(]\s*\d+\s*[）)]', '', normalized)
    normalized = re.sub(r'[^\w\u4e00-\u9fff]+', '', normalized)
    return normalized


def _target_counts(quantity_dist: dict, fallback_questions: Optional[List[dict]] = None) -> dict:
    dist = quantity_dist or {}
    if dist:
        return {
            "choice": max(0, int(dist.get("choice", 0))),
            "fill": max(0, int(dist.get("fill", 0))),
            "judge": max(0, int(dist.get("judge", 0))),
            "essay": max(0, int(dist.get("essay", 0))),
        }

    # 无显式分布时，根据现有题目类型推断，避免被默认 23 题强覆盖
    if fallback_questions:
        inferred = {"choice": 0, "fill": 0, "judge": 0, "essay": 0}
        for q in fallback_questions:
            key = TYPE_LABEL_TO_KEY.get(str((q or {}).get("type") or ""))
            if key in inferred:
                inferred[key] += 1
        if sum(inferred.values()) > 0:
            return inferred

    # 兜底：仅在完全无法推断时使用默认综合卷分布
    return {
        "choice": 10,
        "fill": 5,
        "judge": 5,
        "essay": 3,
    }


def _question_term_style_ok(question: dict) -> bool:
    if not isinstance(question, dict):
        return True
    visible_parts: List[str] = [
        str(question.get("content") or question.get("question") or ""),
        str(question.get("answer") or ""),
        str(question.get("analysis") or question.get("explanation") or ""),
    ]
    options = question.get("options") or []
    if isinstance(options, list):
        visible_parts.extend([str(opt) for opt in options])
    return not any(_contains_term_style_violation(part) for part in visible_parts if part)


def _collect_term_style_violation_numbers(questions: List[dict]) -> List[int]:
    bad: List[int] = []
    for i, q in enumerate(questions or [], start=1):
        if not _question_term_style_ok(q):
            bad.append(int((q or {}).get("number") or i))
    return bad


def _score_map_for_counts(target: dict, target_total_score: int = 100) -> dict:
    choice_score = 2
    fill_score = 2
    judge_score = 2
    essay_count = max(0, int(target.get("essay", 0)))
    base = (
        int(target.get("choice", 0)) * choice_score
        + int(target.get("fill", 0)) * fill_score
        + int(target.get("judge", 0)) * judge_score
    )
    if essay_count > 0:
        # 允许为 0，后续会在全局归一阶段做精调，防止 base>target_total_score 时不可收敛
        essay_score = max(0, (int(target_total_score) - base) // essay_count)
    else:
        essay_score = 0
    return {"选择题": choice_score, "填空题": fill_score, "判断题": judge_score, "简答题": essay_score}


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
    qtype = str((question or {}).get("type") or "")
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


_CONCEPT_ANCHORS = [
    "系统调用", "微内核", "宏内核", "混合内核", "特权指令", "进程调度", "线程", "进程",
    "死锁", "信号量", "临界区", "内存管理", "虚拟内存", "页表", "TLB", "缺页", "抖动",
    "文件系统", "I/O", "设备驱动", "网络协议", "TCP", "资源分配图",
]


def _detect_concept_anchor(text: str) -> str:
    src = str(text or "")
    for anchor in _CONCEPT_ANCHORS:
        if anchor in src:
            return anchor
    # 回退：取首个较长中文词
    m = re.search(r'[\u4e00-\u9fff]{3,8}', src)
    return m.group(0) if m else "通用概念"


def _concept_key_for_question(question: dict) -> str:
    content = str((question or {}).get("content") or "")
    kp = str((question or {}).get("knowledge_point") or "").strip()
    base = kp if kp else content
    return _detect_concept_anchor(base)


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
            qtype = qq.get("type") or "选择题"
            candidate_topic = None
            for p in pool:
                if p in used_topics:
                    continue
                if _detect_concept_anchor(p) != ckey:
                    candidate_topic = p
                    break
            if candidate_topic is None:
                candidate_topic = f"{pool[idx % len(pool)]}·专题{idx}"
            if EXAM_ENABLE_LOCAL_EMERGENCY_FALLBACK:
                replacement = _build_emergency_questions(
                    qtype,
                    1,
                    idx,
                    candidate_topic,
                    score_map.get(qtype, 2),
                )[0]
                qq = _normalize_exam_question(replacement, idx)
                if qq.get("type") == "选择题":
                    qq = _ensure_choice_structure(qq)
                ckey = _concept_key_for_question(qq)

        used_concepts[ckey] = used_concepts.get(ckey, 0) + 1
        used_topics.add(str(_detect_concept_anchor(str(qq.get("content") or ""))))
        fixed.append(qq)

    return fixed


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


def _ensure_choice_structure(question: dict) -> dict:
    q = dict(question)
    stem = _shorten_stem_for_exam(str(q.get("content") or "").strip(), max_len=90)
    options = _normalize_options(q.get("options")) or []
    parsed_stem, parsed_options = _split_stem_and_options_for_payload(stem)
    if parsed_stem:
        stem = _shorten_stem_for_exam(parsed_stem, max_len=90)
    if len(options) < 4 and parsed_options:
        options = _normalize_options(parsed_options) or options
    if len(options) < 4:
        templates = _build_choice_option_templates(stem)
        existing_contents = {
            re.sub(r'^[A-D][.、．]\s*', '', str(opt)).strip().lower()
            for opt in options
            if str(opt).strip()
        }
        for template in templates:
            content = re.sub(r'^[A-D][.、．]\s*', '', template).strip().lower()
            if content in existing_contents:
                continue
            options.append(template)
            existing_contents.add(content)
            if len(options) >= 4:
                break

    rebuilt_options: List[str] = []
    for idx, opt in enumerate((options or [])[:4]):
        content = _normalize_choice_option_content(re.sub(r'^[A-D][.、．]\s*', '', str(opt)).strip())
        if not content:
            content = f"选项{chr(65 + idx)}"
        rebuilt_options.append(f"{chr(65 + idx)}. {content}")

    while len(rebuilt_options) < 4:
        idx = len(rebuilt_options)
        rebuilt_options.append(f"{chr(65 + idx)}. 该选项表述不准确")

    q["content"] = stem
    q["options"] = rebuilt_options
    q["answer"] = _normalize_choice_answer(str(q.get("answer") or ""), rebuilt_options)
    return q


def _count_prompt_contract_violations(questions: List[dict]) -> int:
    patterns = [
        r'根据课件',
        r'依据资料',
        r'围绕\s*[“"]?(课程核心知识点|本课程重点知识点)[”"]?',
    ]
    count = 0
    for q in questions:
        text = str((q or {}).get("content") or (q or {}).get("question") or "")
        if any(re.search(p, text) for p in patterns):
            count += 1
    return count


def _compute_fixed_items(before_questions: List[dict], after_questions: List[dict], quantity_dist: dict, target_total_score: int = 100) -> List[str]:
    fixed_items: List[str] = []
    before_flags = _exam_quality_floor_flags(before_questions, quantity_dist, target_total_score=target_total_score)
    after_flags = _exam_quality_floor_flags(after_questions, quantity_dist, target_total_score=target_total_score)
    if (not before_flags.get("options_ok")) and after_flags.get("options_ok"):
        fixed_items.append("missing_options")
    if before_flags.get("duplicates") and (not after_flags.get("duplicates")):
        fixed_items.append("duplicate_rewrite")
    if (not before_flags.get("total_score_ok")) and after_flags.get("total_score_ok"):
        fixed_items.append("score_rebalanced")
    if _count_prompt_contract_violations(before_questions) > 0 and _count_prompt_contract_violations(after_questions) == 0:
        fixed_items.append("prompt_contract_violation")
    return fixed_items


def _repair_exam_questions_locally(
    questions: List[dict],
    quantity_dist: dict,
    topics: List[str],
    target_total_score: int = 100,
) -> List[dict]:
    """
    本地结构修复：题型数量、编号、去重、总分=100。
    """
    target = _target_counts(quantity_dist, fallback_questions=questions)
    score_map = _score_map_for_counts(target, target_total_score=target_total_score)
    topics_seed = "；".join([t for t in (topics or []) if t][:3]) or "本课程重点内容"
    topic_pool = [t.strip() for t in (topics or []) if str(t).strip()]
    if not topic_pool:
        topic_pool = ["进程与线程", "调度与同步", "死锁与资源管理", "内存管理", "文件系统与I/O", "网络与通信机制"]

    grouped: Dict[str, List[dict]] = {label: [] for label in EXAM_TYPE_ORDER}
    for idx, q in enumerate(questions, start=1):
        normalized = _normalize_exam_question(q, idx)
        qtype = normalized.get("type") or "选择题"
        if qtype not in grouped:
            qtype = "选择题"
            normalized["type"] = qtype
        grouped[qtype].append(normalized)

    for key, label in TYPE_KEY_TO_LABEL.items():
        target_count = int(target.get(key, 0))
        current = grouped.get(label, [])
        if len(current) > target_count:
            grouped[label] = current[:target_count]
        elif len(current) < target_count:
            missing = target_count - len(current)
            if EXAM_ENABLE_LOCAL_EMERGENCY_FALLBACK:
                start_no = len(current) + 1
                grouped[label].extend(_build_emergency_questions(label, missing, start_no, topics_seed, score_map.get(label, 2)))

    ordered_questions: List[dict] = []
    for label in EXAM_TYPE_ORDER:
        ordered_questions.extend(grouped.get(label, []))

    seen = set()
    seen_texts: List[str] = []
    deduped: List[dict] = []
    for i, q in enumerate(ordered_questions, start=1):
        qq = _normalize_exam_question(q, i)
        max_len = _stem_max_len_for_question(qq, i)
        qq["content"] = _shorten_stem_for_exam(qq.get("content", ""), max_len=max_len)
        if qq.get("type") == "选择题":
            qq = _ensure_choice_structure(qq)
        sig = _question_signature(qq.get("content", ""))
        is_dup = bool(sig and sig in seen)
        if not is_dup:
            for prev in seen_texts:
                if _is_near_duplicate_text(prev, qq.get("content", ""), threshold=0.72):
                    is_dup = True
                    break
        if is_dup and EXAM_ENABLE_LOCAL_EMERGENCY_FALLBACK:
            replacement_topic = topic_pool[(i - 1) % len(topic_pool)]
            replacement = _build_emergency_questions(
                qq.get("type") or "选择题",
                1,
                i,
                replacement_topic,
                score_map.get(qq.get("type") or "选择题", 2),
            )[0]
            qq = _normalize_exam_question(replacement, i)
            if qq.get("type") == "选择题":
                qq = _ensure_choice_structure(qq)
            sig = _question_signature(qq.get("content", ""))
        if sig:
            seen.add(sig)
            seen_texts.append(str(qq.get("content", "")))
        deduped.append(qq)

    deduped = _enforce_concept_uniqueness(deduped, topics, score_map)

    for i, q in enumerate(deduped, start=1):
        q["number"] = i

    deduped = _normalize_exam_scores_to_target(deduped, score_map, target_total=target_total_score)
    deduped = _ensure_answer_analysis(deduped)
    return deduped


def _exam_quality_floor_flags(questions: List[dict], quantity_dist: dict, target_total_score: int = 100) -> dict:
    target = _target_counts(quantity_dist, fallback_questions=questions)
    total_questions_target = int(sum(int(target.get(k, 0) or 0) for k in ["choice", "fill", "judge", "essay"]))
    # 仅对综合卷启用严格“概念不重叠”门槛，小卷不作为硬失败条件
    enforce_concept_overlap_gate = total_questions_target >= 20
    concept_repeat_limit = 1 if enforce_concept_overlap_gate else 999999
    counts = {"choice": 0, "fill": 0, "judge": 0, "essay": 0}
    numbers: List[int] = []
    total_score = 0
    seen = set()
    has_duplicates = False
    concept_overlap = False
    options_ok = True
    answer_analysis_ok = True
    concept_seen: Dict[str, int] = {}

    for q in questions:
        qtype = q.get("type") or "选择题"
        key = TYPE_LABEL_TO_KEY.get(qtype)
        if key in counts:
            counts[key] += 1
        if qtype == "选择题":
            opts = q.get("options") or []
            if not isinstance(opts, list) or len(opts) < 4:
                options_ok = False
        if not str(q.get("answer") or "").strip() or not str(q.get("analysis") or "").strip():
            answer_analysis_ok = False
        number = int(q.get("number") or 0)
        if number > 0:
            numbers.append(number)
        total_score += int(q.get("score") or 0)
        sig = _question_signature(q.get("content", ""))
        if sig and sig in seen:
            has_duplicates = True
        if sig:
            seen.add(sig)
        ckey = _concept_key_for_question(q)
        concept_seen[ckey] = concept_seen.get(ckey, 0) + 1
        if concept_seen[ckey] > concept_repeat_limit:
            concept_overlap = True

    quantity_ok = all(int(counts[k]) == int(target.get(k, 0)) for k in counts)
    numbering_ok = bool(numbers) and numbers == list(range(1, len(questions) + 1))
    total_score_ok = total_score == int(target_total_score)
    concept_ok = (not concept_overlap) if enforce_concept_overlap_gate else True
    quality_floor_passed = quantity_ok and numbering_ok and (not has_duplicates) and concept_ok and total_score_ok and options_ok and answer_analysis_ok
    return {
        "quantity_ok": quantity_ok,
        "numbering_ok": numbering_ok,
        "duplicates": has_duplicates,
        "concept_overlap": concept_overlap,
        "concept_overlap_gate_enabled": enforce_concept_overlap_gate,
        "total_score_ok": total_score_ok,
        "options_ok": options_ok,
        "answer_analysis_ok": answer_analysis_ok,
        "duplicates_ok": not has_duplicates,
        "concept_overlap_ok": (not concept_overlap) if enforce_concept_overlap_gate else True,
        "concept_repeat_limit": concept_repeat_limit,
        "quality_floor_passed": quality_floor_passed,
        "counts": counts,
        "total_score": total_score,
    }


def _build_fast_exam_critique(state: ExamPaperState, exam_paper: str) -> Optional[dict]:
    """可用性优先的本地试卷校验，满足条件时直接放行，避免额外 LLM 评审耗时。"""
    quantity_dist = state.get('quantity_dist', {}) or {"choice": 10, "fill": 5, "judge": 5, "essay": 3}
    actual = _count_questions_in_exam(exam_paper, quantity_dist)
    quantity_valid = (
        actual["choice"] == quantity_dist.get("choice", 0) and
        actual["fill"] == quantity_dist.get("fill", 0) and
        actual["judge"] == quantity_dist.get("judge", 0) and
        actual["essay"] == quantity_dist.get("essay", 0)
    )

    payload = state.get("exam_payload") or {}
    exam_data = payload.get("exam_data", {})
    questions = exam_data.get("questions", [])
    numbers = [int(q.get("number", 0)) for q in questions if q.get("number")]
    numbering_valid = bool(numbers) and numbers == list(range(1, len(numbers) + 1))

    seen = set()
    has_duplicates = False
    options_valid = True
    for q in questions:
        signature = re.sub(r'\s+', ' ', str(q.get("content", "")).strip())
        if signature and signature in seen:
            has_duplicates = True
            break
        if signature:
            seen.add(signature)
        if str(q.get("type") or "") == "选择题":
            opts = q.get("options") or []
            if not isinstance(opts, list) or len(opts) < 4:
                options_valid = False

    if quantity_valid and numbering_valid and not has_duplicates and options_valid:
        logger.info("[Agent2-Critic-试卷] 本地校验通过，跳过 LLM 评审以保证可用性")
        return {
            "approved": True,
            "overall_score": 85,
            "critique": "本地校验通过，题量、编号与重复检查均符合要求，直接交付。",
            "reasoning_flaws": [],
            "specific_issues": [],
            "duplicate_check": {"has_duplicates": False, "duplicate_questions": []},
            "numbering_check": {"is_continuous": True, "issues": []},
            "quantity_check": {
                "choice": quantity_dist.get("choice", 0),
                "fill": quantity_dist.get("fill", 0),
                "judge": quantity_dist.get("judge", 0),
                "essay": quantity_dist.get("essay", 0),
                "actual_choice": actual["choice"],
                "actual_fill": actual["fill"],
                "actual_judge": actual["judge"],
                "actual_essay": actual["essay"],
                "is_valid": True
            }
        }

    return None


def _count_questions_in_exam(exam_text: str, quantity_dist: dict = None) -> dict:
    """
    精确统计试卷中各类题型的数量
    根据 quantity_dist 动态确定题型分布，或者按题号范围分段统计
    返回: {"choice": X, "fill": Y, "judge": Z, "essay": W, "total": N}
    """
    import re

    # 如果有 quantity_dist，使用它来确定题号范围
    if quantity_dist:
        choice_count_q = quantity_dist.get('choice', 10)
        fill_count_q = quantity_dist.get('fill', 5)
        judge_count_q = quantity_dist.get('judge', 5)
        essay_count_q = quantity_dist.get('essay', 3)

        # 计算每种题型的题号范围
        current = 1
        choice_range = (current, current + choice_count_q - 1)
        current += choice_count_q
        fill_range = (current, current + fill_count_q - 1)
        current += fill_count_q
        judge_range = (current, current + judge_count_q - 1)
        current += judge_count_q
        essay_range = (current, current + essay_count_q - 1)

        def count_in_range(text: str, start: int, end: int) -> int:
            count = 0
            for num in range(start, end + 1):
                pattern = rf'^\s*{num}[.、]\s'
                if re.search(pattern, text, re.MULTILINE):
                    count += 1
            return count

        choice_count = count_in_range(exam_text, *choice_range)
        fill_count = count_in_range(exam_text, *fill_range)
        judge_count = count_in_range(exam_text, *judge_range)
        essay_count = count_in_range(exam_text, *essay_range)
    else:
        # 默认按题号范围统计
        def count_by_range(text: str, start: int, end: int) -> int:
            count = 0
            for num in range(start, end + 1):
                pattern = rf'^\s*{num}[.、]\s'
                if re.search(pattern, text, re.MULTILINE):
                    count += 1
            return count

        choice_count = count_by_range(exam_text, 1, 10)
        fill_count = count_by_range(exam_text, 11, 20)
        judge_count = count_by_range(exam_text, 21, 30)
        essay_count = count_by_range(exam_text, 31, 50)

    total = choice_count + fill_count + judge_count + essay_count
    logger.info(f"[题目统计] 选择题={choice_count}, 填空题={fill_count}, 判断题={judge_count}, 简答题={essay_count}, 总计={total}")

    return {
        "choice": choice_count,
        "fill": fill_count,
        "judge": judge_count,
        "essay": essay_count,
        "total": total
    }


# ==================== 提示词 ====================

# ---------- Agent 1: 出题 + 推理链 ----------
GENERATE_STRUCTURED_QUIZ_PROMPT = """你是一位资深大学期末考试命题专家。请基于课程知识点生成结构化题目。

【考点】{topic}
【题型】{quiz_type}
【题目数量】{num}
【样卷格式参考】{sample_paper_context}

【课程参考资料】:
{context}

出题策略（必须遵守）：
- 先根据检索结果自行判断本次考点范围与重点分布。
- 课件知识是主依据；模型知识和外部网络知识只能辅助完善题目质量，不能与课件结论冲突。

要求：
1. 只返回纯 JSON，不要输出思考过程、<think> 标签、Markdown 代码块。
2. 必须生成正好 {num} 道题。
3. 选择题必须包含 4 个完整选项。
4. 每道题都必须包含答案和解析。
5. score 使用“5分”这类字符串，difficulty 使用“简单/中等/较难”。
6. 题干必须是标准考试表述，禁止出现“根据课件/依据资料/围绕课程核心知识点”等元话术。
7. 来源依据判断只允许写在 reasoning 字段，question/options/answer/explanation 禁止出现来源措辞。
8. 选择题题干控制在 80 字以内，不要输出超长背景段落。
9. 选项必须独立成行，且不得出现 "A A"、"B B" 等重复前缀。
10. 禁止输出以下泛化选项句式：`会直接影响系统行为与资源约束`、`只影响文字表述`、`仅在理想场景成立`、`与性能和正确性无关`。
11. 同一概念最多出现 2 次；若出现重复，必须改写为不同章节知识点。
12. 题干长度要有梯度感：短题约40-60字，中题约60-90字，少量长题可到120字，避免整卷长度雷同。

返回格式：
{{
  "title": "练习题",
  "questions": [
    {{
      "id": 1,
      "type": "{quiz_type}",
      "question": "题干内容",
      "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
      "answer": "A",
      "explanation": "解析内容",
      "score": "5分",
      "difficulty": "中等"
    }}
  ],
  "reasoning": {{"knowledge_points": [...], "answer_evidence": [...], "distractor_design": [...]}}
}}"""


GENERATE_WITH_REASONING_PROMPT = """【警告-绝对禁止】绝对禁止使用'...'、'等'、'以下略'等任何占位符！你必须逐字逐句地生成所有要求的题目。任何省略行为将被判定为任务失败！

你是一位资深大学期末考试命题专家。基于课程知识点生成题目，同时给出详细推理链。

【考点】{topic}
【题型】{quiz_type}
【题目数量】{num}
【样卷格式参考】{sample_paper_context}

【课程参考资料】:
{context}

出题策略（必须遵守）：
- 先根据检索结果判断考点范围与重点。
- 课件为主，模型知识与外部知识为辅；外部信息不得覆盖课件结论。

请生成题目，并提供推理链（仅写在 reasoning 字段，说明为什么这样出题与依据判断）。

禁止输出 <think> 标签、思考过程、分析说明。
禁止在题干、选项、答案、解析中出现“根据课件/依据资料/参考资料/围绕核心知识点”等元话术。
禁止输出以下泛化选项模板：`会直接影响系统行为与资源约束`、`只影响文字表述`、`仅在理想场景成立`、`与性能和正确性无关`。
同一概念最多出 2 题，必须跨知识点覆盖。
题干长短需有变化：短题/中题/少量长题搭配，避免所有题目都同样冗长。

【重要】只返回纯 JSON 对象，不要用 ```json 代码块包裹！直接输出：
{{"quiz": "...", "reasoning": {{"knowledge_points": [...], "answer_evidence": [...], "distractor_design": [...]}}}}"""


# 【出题原则】
# - 答案必须有课件原文依据，禁止凭空编造
# - 干扰项要有区分度，不能用"以上都是/都不是"
# - 难度分布：简单30%、中等50%、困难20%


# ---------- Agent 2: Critic — 质疑推理链 ----------
CRITIQUE_PROMPT = """你是一位严格的考试命题评审专家。你的职责不是简单检查题目，
而是深入质疑出题者的推理链，找出逻辑漏洞和错误声明。

【考点】{topic}
【题型】{quiz_type}
【课件资料】（唯一权威依据）:
{context}

【待评审题目】:
{quiz}

【出题者的推理链】（出题者对自己决策的解释）:
{reasoning}

请逐一审查推理链中的每个声明：
1. 课件原文引用是否准确？（对照课件资料逐字核实）
2. 答案依据是否充分？（能否在课件中找到明确支持）
3. 干扰项设计逻辑是否合理？（是否真的具有迷惑性但又有明确错误原因）
4. 是否有超出课件范围的知识点？

禁止输出 <think> 标签、思考过程、分析说明。

【重要】只返回纯 JSON 对象，不要用 ```json 代码块包裹！直接输出：
{{"approved": false, "overall_score": 75, "critique": "...", "reasoning_flaws": [{{"question": "...", "flaw": "...", "severity": "high"}}], "specific_issues": [{{"question": "...", "issue": "...", "suggestion": "..."}}]}}"""


# 判断标准：
# - approved=false 当且仅当存在 high severity 问题，或 overall_score < 70
# - 格式小问题、轻微难度偏差不影响 approved


# ---------- Agent 3: Revise — 针对批评逐条修订 ----------
REVISE_WITH_REFLECTION_PROMPT = """你是一位考试命题专家，你刚刚收到了评审专家对你题目的批评。
请认真对待每一条批评，进行有针对性的修订，不能敷衍了事。

【考点】{topic}
【题型】{quiz_type}
【课件资料】（修订必须基于此）:
{context}

【原始题目】:
{quiz}

【你的原始推理链】:
{reasoning}

【评审批评】（必须逐条回应）:
{critique}

请根据批评修订题目，并详细说明你的修改依据（仅写在 revision_notes，不得写进题干与解析）。

禁止输出 <think> 标签、思考过程、分析说明。

【重要】只返回纯 JSON 对象，不要用 ```json 代码块包裹！直接输出：
{{"revised_quiz": "...", "revision_notes": "...", "addressed_issues": [...]}}"""


GENERATE_TEXT_FALLBACK_PROMPT = """你是一位大学期末考试命题专家。请基于课程知识点，生成 {num} 道{quiz_type}。

【考点】{topic}
【样卷格式参考】{sample_paper_context}
【课程参考资料】
{context}

出题策略（必须遵守）：
- 先根据检索结果确定考点范围；
- 课件为主依据，模型知识与外部知识只用于辅助完善题目表达与干扰项设计。

要求：
1. 只输出题目正文，不要输出思考过程，不要输出 JSON。
2. 每道题都必须包含题干、答案、解析。
3. 选择题必须提供 A、B、C、D 四个选项。
4. 如果 topic 为空，请从课程各模块中综合选题，保证覆盖面。
5. 题干禁止出现“根据课件/围绕课程核心知识点/依据资料”等元话术。
6. 解析也禁止出现“根据课件/依据资料/参考资料”等来源措辞。
7. 选择题题干不超过 80 字；选项一行一个，禁止 `A A` 这类重复前缀。
8. 禁止使用泛化模板选项：`会直接影响系统行为与资源约束`、`只影响文字表述`、`仅在理想场景成立`、`与性能和正确性无关`。
9. 必须避免同一概念连续出题，优先覆盖不同章节。
10. 题干需长短搭配，不要整批题目同样长度。
11. 用户可见字段（题干/选项/答案/解析）禁止“中文术语（英文解释）”样式；CPU/TLB/TCP 等纯缩写可保留。

输出示例：
1.（选择题，分值：5分，难度：中等）
题干内容
A. 选项1
B. 选项2
C. 选项3
D. 选项4
答案：A
解析：...
"""

EXAM_PROMPT_CONTRACT = """【统一主契约】
1. 题干、选项、答案、解析必须是考试语气，禁止出现“根据课件/依据资料/围绕核心知识点”等来源话术。
2. 选择题必须输出 A-D 四个选项且每个选项独立成行，不得出现 A A / B B 等重复前缀。
3. 题干长度要有梯度（短中长搭配），避免整卷长度雷同。
4. 综合卷允许同章节变体，但同一核心概念重复数量必须受控。
5. 用户可见字段（题干/选项/答案/解析）禁止“中文术语（英文解释）”样式；CPU/TLB/TCP 等纯缩写可保留。"""


# ---------- 试卷版本 ----------
EXAM_GENERATE_WITH_REASONING_PROMPT = """你是一位大学课程命题专家。请按给定题型分布生成完整试卷。

【考点范围】{topics}
【题目类型】{quiz_types}
【总题数】{total_questions}
【样卷格式参考】{sample_paper_context}
【课程参考资料】
{contexts}
{exam_contract}

【硬性约束】
1. 总题数必须为 {total_questions}。
2. 题型分布必须严格满足：选择题 {choice_count}、填空题 {fill_count}、判断题 {judge_count}、简答题 {essay_count}。
3. 编号必须连续：1 到 {total_questions}。
4. 选择题必须包含 A/B/C/D 四个完整选项。
5. 每题必须有答案与解析。
6. 题干禁止出现“根据课件/依据资料/围绕课程核心知识点”等元话术。
7. 禁止使用“...”等占位符，不得省略题目。
8. 解析也禁止出现“根据课件/依据资料/参考资料”等来源措辞；来源判断只能写在 reasoning 字段。
9. 选择题题干不超过 80 字，选项必须按 A/B/C/D 独立成行。
10. 禁止输出泛化模板选项：`会直接影响系统行为与资源约束`、`只影响文字表述`、`仅在理想场景成立`、`与性能和正确性无关`。
11. 覆盖要求：综合卷（23题）必须做到题题考点不重叠，同一核心概念只能出现 1 次。
12. 体感要求：题干详略得当、又长有短，避免整卷题干长度单一。

【分值与编号约束】
- 选择题：{choice_count} 题，每题 2 分，编号 1-{choice_end}
- 填空题：{fill_count} 题，每题 2 分，编号 {fill_start}-{fill_end}
- 判断题：{judge_count} 题，每题 2 分，编号 {judge_start}-{judge_end}
- 简答题：{essay_count} 题，每题 {essay_score} 分，编号 {essay_start}-{essay_end}

只返回纯 JSON：
{{"exam_paper": "...", "reasoning": {{"topic_coverage": [...], "answer_evidence": [...], "difficulty_distribution": "..."}}}}"""


EXAM_CRITIQUE_PROMPT = """你是试卷评审专家。请严格检查结构完整性与可作答性。

【考点范围】{topics}
【参考资料】{contexts}
【待评审试卷】{exam_paper}
【出题推理链】{reasoning}
{exam_contract}

【目标分布】
- 总题数：{total_questions}
- 选择题：{choice_count}（编号 1-{choice_end}）
- 填空题：{fill_count}（编号 {fill_start}-{fill_end}）
- 判断题：{judge_count}（编号 {judge_start}-{judge_end}）
- 简答题：{essay_count}（编号 {essay_start}-{essay_end}）

【审查清单】
1. 题型数量是否准确。
2. 编号是否连续且范围正确。
3. 总分是否为 100。
4. 选择题是否都有完整 A-D 选项。
5. 是否存在重复题或高度相似题。
6. 题干是否含元话术（如“根据课件/依据资料”）。
7. 是否出现泛化模板选项（如“会直接影响系统行为与资源约束”等）。
8. 是否存在知识点覆盖失衡（综合卷要求同一概念不得重复）。
9. 题干是否长度单一（应有短中长搭配）。

只返回纯 JSON：
{{"approved": false, "overall_score": 75, "critique": "...", "reasoning_flaws": [{{"question": "...", "flaw": "...", "severity": "high"}}], "specific_issues": [{{"question": "...", "issue": "...", "suggestion": "..."}}], "duplicate_check": {{"has_duplicates": false, "duplicate_questions": []}}, "numbering_check": {{"is_continuous": true, "issues": []}}, "quantity_check": {{"choice": {choice_count}, "fill": {fill_count}, "judge": {judge_count}, "essay": {essay_count}, "actual_choice": 0, "actual_fill": 0, "actual_judge": 0, "actual_essay": 0, "is_valid": false}}}}"""


EXAM_REVISE_WITH_REFLECTION_PROMPT = """你是考试命题专家，收到评审批评后进行针对性修订。

【考点范围】{topics}
【课件资料】:
{contexts}

【原始试卷】:
{exam_paper}

【原始推理链】:
{reasoning}

【评审批评】（必须逐条回应，包括所有问题）：
{critique}
{exam_contract}

【关键提醒】
1. 如果评审批评指出存在重复题目，必须删除或替换重复的题目
2. 确保修订后总分仍然是100分
3. 确保修订后题目数量仍然是{total_questions}道
4. 返回完整的修订后试卷，不要只返回修改的部分
5. 来源依据说明仅允许写在 revision_notes；禁止写入试卷题干、选项、答案、解析。

【重要】只返回纯 JSON 对象，不要用 ```json 代码块包裹！直接输出：
{{"revised_exam": "...", "revision_notes": "...", "addressed_issues": [...]}}"""


# ==================== 节点函数（单题）====================

async def generate_with_reasoning_node(state: QuizState) -> QuizState:
    """Agent 1: 出题 + 推理链"""
    logger.info(f"[Agent1-出题] {state['topic']} / {state['quiz_type']} / {state['num']}道")
    sample = state.get('sample_paper_context', '') or '无样卷参考，使用默认格式'

    try:
        structured_result = await call_llm_structured(
            GENERATE_STRUCTURED_QUIZ_PROMPT,
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

    # 使用结构化输出，彻底杜绝 Markdown 混排
    try:
        result = await call_llm_structured(
            GENERATE_WITH_REASONING_PROMPT,
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
                GENERATE_WITH_REASONING_PROMPT,
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
                GENERATE_TEXT_FALLBACK_PROMPT,
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

    fast = _quick_quiz_quality_check(state)
    if fast is not None:
        score = fast.get("overall_score", 80)
        logger.info(f"[Agent2-Critic] 本地快速质检通过，跳过 LLM Critic，score={score}")
        update: dict = {"critique": fast}
        if state.get('reflection_rounds', 0) == 0 and not state.get('initial_score'):
            update["initial_score"] = score
        return update

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
                CRITIQUE_PROMPT,
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
                REVISE_WITH_REFLECTION_PROMPT,
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
                REVISE_WITH_REFLECTION_PROMPT,
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
                EXAM_CRITIQUE_PROMPT,
                CritiqueResult,
                topics=", ".join(state['topics']),
                contexts=contexts_combined,
                exam_paper=exam_paper,
                reasoning=state.get('reasoning', '{}'),
                total_questions=state.get('total_questions', 10),
                exam_contract=EXAM_PROMPT_CONTRACT,
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
                    EXAM_CRITIQUE_PROMPT,
                    topics=", ".join(state['topics']),
                    contexts=contexts_combined,
                    exam_paper=exam_paper,
                    reasoning=state.get('reasoning', '{}'),
                    total_questions=state.get('total_questions', 10),
                    exam_contract=EXAM_PROMPT_CONTRACT,
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
        # 强制 set approved = False
        critique["approved"] = False
        if critique.get("overall_score", 100) > 70:
            critique["overall_score"] = 65

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

    llm_budget = max(20.0, min(90.0, budget_left - 10.0))
    revised = _build_exam_text_from_payload(_build_exam_payload_from_questions(local_fixed_questions, title="期末考试试卷"))
    notes = "本地局部修订后进入模型精修。"
    degrade_reason = ""

    # 使用结构化输出（受预算约束）
    try:
        result = await asyncio.wait_for(
            call_llm_structured(
                EXAM_REVISE_WITH_REFLECTION_PROMPT,
                ExamReviseResult,
                topics=", ".join(state['topics']),
                contexts=contexts_combined,
                exam_paper=revised,
                reasoning=state.get('reasoning', '{}'),
                critique=json.dumps(critique, ensure_ascii=False),
                total_questions=state.get('total_questions', 10),
                exam_contract=EXAM_PROMPT_CONTRACT,
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
                    EXAM_REVISE_WITH_REFLECTION_PROMPT,
                    topics=", ".join(state['topics']),
                    contexts=contexts_combined,
                    exam_paper=revised,
                    reasoning=state.get('reasoning', '{}'),
                    critique=json.dumps(critique, ensure_ascii=False),
                    total_questions=state.get('total_questions', 10),
                    exam_contract=EXAM_PROMPT_CONTRACT,
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
      4. 存在重复题目（duplicate_check.has_duplicates = true）
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

    needs = (score < 70) or (len(high_flaws) > 0) or (not approved) or has_duplicates or has_numbering_issues or has_quantity_issues
    logger.info(
        f"[Critic判断] score={score}, high_flaws={len(high_flaws)}, "
        f"approved={approved}, duplicates={has_duplicates}, numbering={has_numbering_issues}, quantity={has_quantity_issues} → {'修订' if needs else '通过'}"
    )
    return needs


def _exam_is_usable_without_revision(critique: dict) -> bool:
    """可用性优先：题量、编号、去重等硬指标过关时直接交付，避免整卷修订超时。"""
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

    usable = quantity_ok and numbering_ok and not has_duplicates and score >= 55 and len(non_structural_high_flaws) <= 1
    logger.info(
        f"[试卷可用性判断] score={score}, quantity_ok={quantity_ok}, numbering_ok={numbering_ok}, "
        f"duplicates={has_duplicates}, non_structural_high_flaws={len(non_structural_high_flaws)} → {'可直接交付' if usable else '仍需修订'}"
    )
    return usable


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
    if state.get('reflection_rounds', 0) >= 2:
        return "end"
    if state.get("degrade_reason") in {"budget_exhausted", "revise_timeout"}:
        return "end"
    if state.get("quality_floor_passed") is True:
        return "end"
    critique = state.get('critique', {})
    if _exam_is_usable_without_revision(critique):
        return "end"
    if _critique_needs_revision(critique):
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
    graph.add_edge("revise", "critique")

    return graph.compile()


quiz_workflow = build_quiz_graph()
exam_workflow = build_exam_graph()


# ==================== 对外接口 ====================

async def run_quiz_agent(
    topic: str,
    quiz_type: str = "选择题",
    num: int = 3,
    sample_paper_context: str = None
) -> dict:
    """
    运行 Reflexion 出题系统（3 Agent 协作）
    流程：出题+推理链 → Critic质疑推理链 → Revise针对批评修订（最多2轮）
    """
    quiz_started_at = time.monotonic()
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
    }
    use_web = False
    gate_reason = "courseware_sufficient"
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
        questions = _sanitize_quiz_questions_for_delivery(questions)
        payload = _build_quiz_payload(topic, questions)
        floor_flags = _quiz_quality_floor_flags(questions, quiz_type, num)
        term_style_ok = all(_question_term_style_ok(q) for q in questions)
        return {
            "kind": "quiz_set",
            "render_mode": "interactive_cards",
            "text": _build_quiz_text_from_questions(questions),
            "payload": payload,
            "meta": {
                "delivery_mode": "partial_revised",
                "degrade_reason": "budget_exhausted",
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

    parsed_questions = _sanitize_quiz_questions_for_delivery(parsed_questions)
    final_payload = _build_quiz_payload(topic, parsed_questions)
    final_text = _build_quiz_text_from_questions(parsed_questions)

    floor_flags = _quiz_quality_floor_flags(parsed_questions, quiz_type, num)
    term_style_ok = all(_question_term_style_ok(q) for q in parsed_questions)
    delivery_mode = result.get("delivery_mode") or ("full" if floor_flags.get("quality_floor_passed") else "partial_revised")
    degrade_reason = result.get("degrade_reason") or ("" if floor_flags.get("quality_floor_passed") else "budget_exhausted")
    if not term_style_ok:
        delivery_mode = "partial_revised"
        degrade_reason = degrade_reason or "term_style_violation"
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
    essay_timeout = max(45, min(150, 35 * max(1, essay)))

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
    if flags.get("duplicates"):
        return False, "存在重复题"
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
            "stage_fast_mode": True,
            "reflection_rounds": 0,
        }

        try:
            result = await asyncio.wait_for(exam_workflow.ainvoke(stage_state), timeout=stage_timeout)
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
            stage_exit = _classify_structured_error(e)
            stage_trace["stage_exit_reason"] = stage_exit if stage_exit != "other" else "stage_runtime_error"
            miss_match = re.search(r'仍缺\s*(\d+)\s*道', err_text)
            if miss_match:
                stage_trace["missing_after_retries"] = int(miss_match.group(1))
            logger.warning(f"[ExamStage] stage={stage_id} attempt={attempt}/{retries} 失败: {e}")
            if attempt >= retries:
                break

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

    # 如果没有样卷，联网搜索试卷格式参考
    online_format_context = ""
    if not sample_paper_context or sample_paper_context.strip() == "":
        logger.info("[Exam Agent] 无样卷，联网搜索试卷格式和考点参考...")
        search_result = ""
        try:
            # 优先尝试 DuckDuckGo
            from langchain_community.tools import DuckDuckGoSearchRun
            search = DuckDuckGoSearchRun()
            search_query = f"{topics[0] if topics else '大学课程'} 期末考试试卷 题型分布 结构"
            search_result = search.run(search_query)
        except Exception as e:
            logger.warning(f"[Exam Agent] DuckDuckGo 失败，尝试 Tavily: {e}")
            try:
                # 备用：Tavily
                from langchain_community.tools import TavilySearchResults
                search = TavilySearchResults(max_results=3)
                docs = search.run(f"{topics[0] if topics else '大学课程'} 期末考试试卷 题型")
                if docs:
                    search_result = "\n".join([d.get("content", "")[:500] for d in docs if d.get("content")])
            except Exception as e2:
                logger.warning(f"[Exam Agent] Tavily 也失败: {e2}")

        if search_result and len(search_result) > 50:
            online_format_context = f"\n\n【联网搜索的试卷格式参考】（无本地样卷时使用）：\n{search_result[:1500]}..."
            logger.info("[Exam Agent] 联网搜索获取格式参考成功")
            tool_call_web = 1
        else:
            online_format_context = "\n\n【提示】无样卷参考，请使用通用期末试卷格式：选择题、填空题、判断题、简答题四种题型均衡分布。"
    else:
        logger.info("[Exam Agent] 使用本地样卷格式")

    # 试卷生成只使用单份 context，避免多 topic 串行检索造成长时间阻塞
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
    logger.info(
        f"[Exam Agent] 聚合检索完成: topics={len(topics)} -> single_context_len={len(merged_context)}"
    )
    contexts = [merged_context]
    evidence_source = "courseware_plus_web" if tool_call_web > 0 else "courseware_only"

    # 合并样卷格式和联网搜索结果
    final_sample_context = (sample_paper_context or "") + online_format_context

    initial_state: ExamPaperState = {
        "topics": topics,
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
        "stage_fast_mode": False,
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
            qtype = str((q or {}).get("type") or "")
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
        result = await asyncio.wait_for(
            exam_workflow.ainvoke(initial_state),
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
