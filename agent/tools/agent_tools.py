import random
import json
import re

from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.logger_handler import logger, timer_and_token_logger
from utils.session_context import current_session_id
from utils.config_handler import agent_conf

# 联网搜索开关
WEB_SEARCH_ENABLED = agent_conf.get('web_search', {}).get('enabled', True)

# 按 session_id 缓存 RAG 实例，避免全局单例在切换科目时检索旧知识库
_rag_cache: dict[str, RagSummarizeService] = {}


def clear_rag_cache(session_id: str):
    """清理指定 session 的 RAG 缓存"""
    if session_id in _rag_cache:
        del _rag_cache[session_id]


async def _get_rag() -> RagSummarizeService:
    """
    按当前 session_id 获取（或新建）RAG 服务实例。
    同一科目复用同一实例，切换科目自动创建新实例。
    """
    sid = current_session_id.get() or "default"
    if sid not in _rag_cache:
        logger.info(f"[RAG] 为 session [{sid}] 初始化新的 RagSummarizeService")
        _rag_cache[sid] = RagSummarizeService()
    return _rag_cache[sid]


async def _get_chat_model():
    """获取聊天模型"""
    from model.factory import chat_model
    return chat_model


# ==================== 多 Agent 出题系统 ====================

@tool(description='从课件知识库中检索知识点和参考资料。传入的query应该是一个具体考点名词。')
@timer_and_token_logger
async def search_courseware(query: str) -> str:
    rag = await _get_rag()
    return await rag.rag_summarize(query)


@tool(description='联网搜索信息。当需要验证答案、查找默认试卷风格、查询考试规范时使用。')
@timer_and_token_logger
async def web_search(query: str) -> str:
    """
    联网搜索工具（用于辅助验证，不能替代课件资料）
    使用 DuckDuckGo 免费搜索，无需 API Key
    """
    if not WEB_SEARCH_ENABLED:
        return "联网搜索功能已关闭，请在配置文件中启用"

    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        search = DuckDuckGoSearchRun()
        result = search.run(query)

        # 限制返回结果长度
        if len(result) > 2000:
            result = result[:2000] + "..."

        logger.info(f"[联网搜索] {query} -> {len(result)} 字符")
        return result

    except Exception as e:
        logger.warning(f"[联网搜索] 失败: {e}")
        return f"联网搜索暂时不可用: {str(e)}"


@tool(description='搜索大学期末考试试卷格式范例。当用户没有上传样卷时，可以搜索该学科的典型试卷格式。')
@timer_and_token_logger
async def search_exam_format(course_name: str) -> str:
    """
    搜索默认试卷格式
    """
    if not WEB_SEARCH_ENABLED:
        return "联网搜索功能已关闭，无法获取在线试卷格式。请上传样卷或使用默认格式。"

    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        search = DuckDuckGoSearchRun()
        query = f"{course_name} 期末考试试卷 格式 题型 结构"
        result = search.run(query)

        # 提取关键格式信息
        if len(result) > 1500:
            result = result[:1500] + "..."

        logger.info(f"[试卷格式搜索] {course_name} -> 获取格式参考")
        return f"【{course_name}试卷格式参考】\n{result}"

    except Exception as e:
        logger.warning(f"[试卷格式搜索] 失败: {e}")
        return f"无法获取格式参考，将使用默认格式"


# ==================== Agent 1: 出题 Agent ====================
async def _generate_questions(topic: str, quiz_type: str, num: int, context: str) -> str:
    """出题 Agent：生成题目"""
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from utils.prompt_loader import load_quiz_prompts

    prompt = PromptTemplate.from_template(load_quiz_prompts())
    model = await _get_chat_model()
    chain = prompt | model | StrOutputParser()

    logger.info(f"[出题Agent] 生成 {num} 道 {quiz_type}")
    result = await chain.ainvoke({"topic": topic, "quiz_type": quiz_type, "num": num, "context": context})
    return result


# ==================== Agent 2: 验证 Agent ====================
async def _verify_quiz(quiz: str, topic: str, context: str) -> dict:
    """
    验证 Agent：检查题目和答案是否正确
    返回: {"valid": bool, "issues": list, "fixed_quiz": str}
    """
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    model = await _get_chat_model()

    verify_prompt = PromptTemplate.from_template(
        """请仔细检查以下题目是否存在问题。

考点：{topic}
参考资料：{context}

题目内容：
{quiz}

请按以下 JSON 格式返回检查结果：
{{
    "valid": true/false,
    "issues": ["问题1描述", "问题2描述"],
    "fix_needed": true/false
}}
只返回 JSON，不要其他内容。"""
    )

    chain = verify_prompt | model | StrOutputParser()

    logger.info("[验证Agent] 验证题目正确性...")
    result = await chain.ainvoke({"topic": topic, "context": context, "quiz": quiz})

    try:
        # 尝试解析 JSON
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]

        return json.loads(result.strip())
    except:
        return {"valid": True, "issues": [], "fix_needed": False}


# ==================== Agent 3: 优化 Agent ====================
async def _fix_quiz(quiz: str, issues: list, topic: str, context: str) -> str:
    """
    优化 Agent：修复有问题的题目
    """
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    model = await _get_chat_model()

    fix_prompt = PromptTemplate.from_template(
        """请修复以下题目中的问题。

考点：{topic}
参考资料：{context}

需要修复的问题：
{issues}

原题目：
{quiz}

请直接输出修复后的完整题目，格式清晰，包含答案和解析。
"""
    )

    chain = fix_prompt | model | StrOutputParser()

    logger.info(f"[优化Agent] 修复 {len(issues)} 个问题")
    result = await chain.ainvoke({
        "topic": topic,
        "context": context,
        "issues": "\n".join([f"- {issue}" for issue in issues]),
        "quiz": quiz
    })
    return result


# ==================== Agent 4: 出卷 Agent ====================
async def _generate_exam_paper(
    topics: list,
    quiz_types: list,
    total_questions: int,
    sample_paper_context: str = None
) -> str:
    """
    出卷 Agent：根据多个考点生成完整试卷
    如果有样卷参考，参考样卷格式
    """
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    model = await _get_chat_model()

    # 收集各考点资料
    rag = await _get_rag()
    contexts = []
    for topic in topics:
        context = await rag.rag_summarize(topic)
        contexts.append(f"【{topic}】\n{context}")

    all_context = "\n\n".join(contexts)

    # 构建提示词
    if sample_paper_context:
        prompt_text = f"""请根据以下考点资料，参考样卷格式，生成一套完整的试卷。

样卷格式参考：
{sample_paper_context}

考点资料：
{all_context}

要求：
1. 共 {total_questions} 道题
2. 题目类型包括：{', '.join(quiz_types)}
3. 各类型题目数量自行分配
4. 包含答案和详细解析
5. 难度适中，覆盖核心考点
"""
    else:
        prompt_text = f"""请根据以下考点资料，生成一套完整的试卷。

考点资料：
{all_context}

要求：
1. 共 {total_questions} 道题
2. 题目类型包括：{', '.join(quiz_types)}
3. 各类型题目数量自行分配
4. 包含答案和详细解析
5. 格式规范，难度适中
"""

    prompt = PromptTemplate.from_template(prompt_text)
    chain = prompt | model | StrOutputParser()

    logger.info(f"[出卷Agent] 生成 {total_questions} 道题的试卷")
    result = await chain.ainvoke({})
    return result


# ==================== Agent 5: 试卷验证 Agent ====================
async def _verify_exam_paper(exam_paper: str, topics: list, context: str) -> dict:
    """
    验证 Agent：检查试卷整体质量
    """
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    model = await _get_chat_model()

    verify_prompt = PromptTemplate.from_template(
        """请检查以下试卷的质量。

考点：{topics}
参考资料：{context}

试卷内容：
{exam_paper}

请按以下 JSON 格式返回检查结果：
{{
    "valid": true/false,
    "issues": ["问题1", "问题2"],
    "score": 85,
    "fix_needed": true/false
}}
只返回 JSON。"""
    )

    chain = verify_prompt | model | StrOutputParser()

    logger.info("[验证Agent] 验证试卷质量...")
    result = await chain.ainvoke({
        "topics": ", ".join(topics),
        "context": context[:2000],  # 限制长度
        "exam_paper": exam_paper
    })

    try:
        result = result.strip()
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0]
        elif "```" in result:
            result = result.split("```")[1].split("```")[0]
        return json.loads(result.strip())
    except:
        return {"valid": True, "issues": [], "fix_needed": False}


# ==================== 同步包装器 ====================
import asyncio

def _run_async(coro):
    """在同步上下文中运行异步函数"""
    try:
        loop = asyncio.get_running_loop()
        # 如果已经有运行中的 loop，创建一个新任务
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # 没有运行中的 loop，可以直接使用 asyncio.run
        return asyncio.run(coro)


# ==================== 对外工具接口 ====================

@tool(description='当你需要为学生出题时，调用此工具。必须传入具体的考点名词(topic)、题目类型(quiz_type，如选择题/判断题等)和题目数量(num)。它会自动查资料并生成题目。')
@timer_and_token_logger
def generate_quiz(topic: str, quiz_type: str = "选择题", num: int = 3) -> str:
    """
    单题生成（带验证优化）
    流程：生成 → 验证 → 优化（如需要）→ 返回
    """
    async def _generate():
        # 自动加载样卷上下文（若用户已上传）
        from api.routers.knowledge import get_sample_paper_context
        sid = current_session_id.get() or "default"
        sample_ctx = get_sample_paper_context(sid)

        # 调用 LangGraph 多 Agent 系统
        from agent.multi_agent.quiz_agent import run_quiz_agent
        logger.info(f"[多Agent出题系统] 使用 LangGraph 执行: {topic}")
        return await run_quiz_agent(topic, quiz_type, num, sample_ctx)

    return _run_async(_generate())


@tool(description='生成完整试卷。如果用户上传了样卷，请传入样卷内容作为参考(样卷格式会作为参考)。参数topics是考点列表，quiz_types是题目类型列表，total_questions是总题数。')
@timer_and_token_logger
def generate_exam_paper(
    topics: str,
    quiz_types: str = "选择题,判断题,填空题",
    total_questions: int = 10,
    sample_paper_context: str = None
) -> str:
    """
    试卷生成（带验证优化）
    流程：生成 → 验证 → 优化（如需要）→ 返回
    """
    async def _generate():
        # 解析参数
        topic_list = [t.strip() for t in topics.split(",")]
        quiz_type_list = [q.strip() for q in quiz_types.split(",")]

        # 调用 LangGraph 多 Agent 出卷系统
        from agent.multi_agent.quiz_agent import run_exam_agent
        logger.info(f"[多Agent出卷系统] 使用 LangGraph 执行")
        return await run_exam_agent(topic_list, quiz_type_list, total_questions, sample_paper_context)

    return _run_async(_generate())


# 动态生成工具列表（根据配置开关）
def _build_tools():
    """根据配置动态生成工具列表"""
    tool_list = [search_courseware, generate_quiz, generate_exam_paper]

    if WEB_SEARCH_ENABLED:
        tool_list.append(web_search)
        tool_list.append(search_exam_format)
        logger.info("[工具注册] 联网搜索已启用")
    else:
        logger.info("[工具注册] 联网搜索已禁用")

    return tool_list


tools = _build_tools()
