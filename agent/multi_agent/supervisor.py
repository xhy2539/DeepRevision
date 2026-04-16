"""
Supervisor 多 Agent 路由架构

Supervisor Agent 分析用户意图，路由到对应 SubAgent：
  - rag_agent    : 知识问答、概念解释、复习某知识点
  - quiz_agent   : 出单道或少量题目（调用 Reflexion 出题工作流）
  - exam_agent   : 生成完整试卷（调用 Reflexion 出卷工作流）
  - planner_agent: 制定复习计划、推荐学习顺序
  - history_agent: 查看练习历史、答题记录、错题分析
  - chitchat     : 闲聊、感谢、无关话题

工作流：supervisor → (路由判断) → SubAgent → END
"""
import json
import time
import re
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from model.factory import chat_model, backup_chat_model, light_chat_model, backup_light_chat_model
from agent.tools.agent_tools import get_rag_service
from agent.tools.history_action_parser import parse_history_action
from utils.logger_handler import logger
from utils.session_context import current_session_id
from utils.rag_metrics import rag_inc


# ==================== 状态定义 ====================

class SupervisorState(TypedDict):
    input: str              # 当前用户输入
    chat_history: List      # 历史消息（LangChain BaseMessage 列表）
    memory_context: str     # 长期图谱记忆（注入 system prompt）
    session_id: str         # 当前科目会话
    exam_stage_plan: bool   # 是否启用分段出卷
    exam_rerun_stage: str   # 失败后仅重跑某一段
    exam_partial_questions: list  # 已成功段题目（重跑失败段时回传）
    exam_fast_mode: bool   # True=快速路径(格式检查), False=完整路径(LLM Critique)
    quiz_force_llm_critic: bool  # True=quiz强制走LLM Critic，不走本地快速质检短路
    # Supervisor 决策
    route: str              # "rag" | "quiz" | "exam" | "planner" | "chitchat"
    route_reason: str       # 路由原因（用于日志）
    route_params: dict      # 从用户输入中提取的参数
    supervisor_precomputed: bool  # 外层已完成路由判定时跳过重复 supervisor 调用
    # SubAgent 结果
    subagent_result: str    # SubAgent 原始输出
    final_answer: str       # 最终回答（透传给调用方）


def _safe_int(value, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    """将 LLM 提取参数安全转为 int，兼容空串/None/脏值。"""
    try:
        if isinstance(value, bool):
            result = int(value)
        elif value is None:
            result = default
        else:
            text = str(value).strip()
            if text == "":
                result = default
            else:
                result = int(text)
    except Exception:
        result = default

    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


# ==================== 提示词 ====================

SUPERVISOR_PROMPT = """你是【复习助手】的智能调度中心。

## 职责
分析用户输入，判断意图，选择最合适的路由Agent。

## 路由类型（必须严格按此判断）：
- rag：知识点问答、概念讲解、解释名词、复习某个知识点 → 用"rag"
- quiz：用户明确要求"出题"、"做题"、"练习"、"给我几道题" → 用"quiz"
- exam：用户明确要求"出试卷"、"生成试卷"、"期末考试"、"给我出一份试卷" → 用"exam"
- planner：用户说"复习计划"、"学习计划"、"怎么复习" → 用"planner"
- history：用户说"历史"、"错题"、"练习记录" → 用"history"
- chitchat：问候（你好/hi/hello）、感谢、闲聊、无关话题、无法分类 → 用"chitchat"

## 关键词触发规则（优先级从高到低）：
1. "试卷"、"考试" → exam
2. "出题"、"做题"、"练习题" → quiz
3. "复习计划"、"学习计划" → planner
4. "历史"、"错题" → history
5. "解释"、"什么是"、"概念"、"知识点" → rag
6. 其他 → chitchat

## 菜单选择理解（重要）：
- 如果用户输入只是单个数字（1-9）或简单回复（如"第一个"、"第二个"）
- 且历史对话中助手刚提供了带编号的选项菜单
- 则应将用户回复理解为选择该菜单项，并按对应意图路由
- 例如：历史中助手提供了"1.复习 2.做题 3.聊天"，用户回复"2" → 应路由到 quiz

## 参数提取规则（仅对 quiz/exam 路由）：
- quiz_params: {{"topic": "从用户输入中提取的知识点/科目", "quiz_type": "题型", "num": 数量}}
- exam_params: {{"topics": ["知识点列表"], "quiz_types": ["题型列表"], "total_questions": 总题数, "quantity_dist": {{"choice": 数量, "fill": 数量, "judge": 数量, "essay": 数量}}}}
- 如果用户没有指定具体知识点，topic/topics 设为空字符串/空列表

## rewritten_query 规则：
- 如果用户只是打招呼（你好/hi/hello），rewritten_query = 原输入不变
- 如果用户要求出题/出试卷，rewritten_query 可以精简/明确化，但**禁止添加原输入没有的内容**
- 如果用户在回应菜单选项（如"2"、"选1"、"第三个"），rewritten_query 应改写为对应的选项内容（如"做练习题"）
- 禁止把"你好"改写成"出一道题"，这是错误的重写

## 【最高优先级】用户当前输入：
{input}

## 【参考信息】历史对话：
- 如果用户在回应菜单，历史中包含选项列表，应结合理解
- 其他情况下：仅供参考，不要被历史带偏
{recent_history}

## 【参考信息】长期记忆：
{memory_context}

返回JSON（只返回JSON，不要其他内容）：
{{"rewritten_query": "...", "route": "...", "reason": "...", "params": {{}}}}"""


RAG_SUBAGENT_PROMPT = """你是知识问答专家。你必须基于检索到的课件资料回答，并提供可核验依据（仅供系统内部校验）。

【长期记忆摘要】
{memory_context}

【近期对话上下文】
{recent_history}

【检索到的课件资料】
{context}

【学生问题】
{input}

## 任务要求：

1. **禁止输出思考过程**：不要写"让我想想"、"根据我的分析"、"这个问题是..."等思考过程
2. **优先使用课件内容**：引用原文时用"根据课件："开头
3. **如果检索到相关内容**：详细解释、举例说明
4. **如果检索内容不足**：
   - 先说明"根据检索到的资料，..."
   - 再结合你的知识补充解释
   - 不要只说"没找到"
5. **如果完全没有检索结果**：
   - 友好提示："我目前没有检索到相关课件内容"
   - 可以基于常识回答，并建议用户上传相关课件
6. **如果用户要求"讲讲"、"解释一下"**：给出详细、通俗的解释，不仅罗列定义

## 输出格式（必须严格为 JSON，对象格式）：
{{
  "answer": "给用户展示的最终答案（自然语言）",
  "evidence": [
    {{"source": "参考资料1", "quote": "必须是从【检索到的课件资料】中逐字可定位的片段"}},
    {{"source": "参考资料2", "quote": "同上"}}
  ]
}}

约束：
- 必须至少提供 1 条 evidence。
- evidence.quote 必须可在【检索到的课件资料】文本中直接匹配到。
- 不要输出 JSON 之外的任何文字。"""


PLANNER_SUBAGENT_PROMPT = """你是学习规划专家，根据学生的复习目标制定具体可执行的学习计划。

【历史记忆摘要】
{memory_context}

【学生需求】
{input}

## 你需要做的：

1. **分析需求**：理解学生想要复习什么、目标是什么、时间紧迫程度
2. **智能规划**：根据内容复杂度推荐复习顺序
3. **优先级排序**：区分哪些必须掌握、哪些了解即可
4. **提供建议**：时间分配、重点难点、学习技巧

## 输出要求（严格遵守）：

- **禁止输出思考过程**：不要写"根据你的需求"、"我来帮你规划"、"首先...然后..."等思考过程
- **直接输出计划内容**：切入正题，给出具体的复习计划
- 计划要具体、可执行
- 给出清晰的学习路径
- 标注每个阶段的重点
- 适当提醒常见误区
- 如果学生没有明确目标，可以主动推荐适合的学习路径

请直接输出复习计划，不要输出任何思考过程。"""


CHITCHAT_PROMPT = """你是期末复习助手的智能闲聊助手。

【当前会话信息】
{context}

【学生消息】
{input}

请直接回复学生，友好自然地回答。如果学生询问会话信息或课件内容，请根据上述信息回答。"""


HISTORY_SUBAGENT_PROMPT = """你是练习历史分析专家。根据用户的请求，从练习记录中提取相关信息回答用户。

【用户请求】
{input}

【练习统计数据】
{stats}

【练习历史记录】
{history}

## 回答要求（严格遵守）：

- **禁止输出思考过程**：不要写"根据你的记录"、"我来帮你分析"等思考过程
- 直接输出数据内容：如果是统计就列出数据，如果是历史就列出记录
- 如果用户询问统计/正确率/薄弱点，基于统计数据回答
- 如果用户询问历史记录，列出最近的练习
- 如果用户询问错因分析，引导用户提供具体题目
- 如果没有数据，说明暂无记录，建议先练习
- 回答要清晰、有条理

**直接输出内容，不要输出思考过程。**"""


# ==================== 工具函数 ====================

INCOMPLETE_PATTERNS = ['：', '...', '了"']

RETRYABLE_ERROR_PATTERNS = [
    'choices',
    '529',
    '520',
    '500',
    'server_error',
    'unknown error',
    'overloaded',
    'null value',
    'connection error',
    'timed out',
    'timeout',
    'temporarily unavailable',
    'server disconnected',
    'remote protocol error',
]


def _is_retryable_llm_error(error: Exception) -> bool:
    """判断 LLM 调用异常是否值得自动重试。"""
    if isinstance(error, TimeoutError):
        return True
    error_str = str(error).lower()
    return any(pattern in error_str for pattern in RETRYABLE_ERROR_PATTERNS)


def _build_provider_chain(primary, backup):
    """构建去重后的主备模型序列，避免同一实例重复调用。"""
    providers = []
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

def _is_incomplete_response(text: str) -> bool:
    """检测响应是否不完整（LLM 想说但被截断）"""
    if not text:
        return True
    text = text.strip()
    if not text:
        return True
    # 以冒号/省略号/"了"结尾（未说完）
    if any(text.endswith(p) for p in INCOMPLETE_PATTERNS):
        return True
    # 以"了"开头的短句（思考内容混入）
    if text.startswith('了') and len(text) < 50 and '。' not in text:
        return True
    # 回答过短且以逗号/顿号结尾（被截断）
    if len(text) < 100 and any(text.endswith(p) for p in '，、'):
        return True
    # 纯思考内容被截断留下的碎片
    if len(text) < 30 and ('，' not in text and '。' not in text):
        if text.startswith('我') or text.startswith('让'):
            return True
    return False


def _sanitize_llm_text(text: str) -> str:
    """清理模型输出中的思考标签与噪音，避免泄露到用户侧。"""
    if text is None:
        return ""
    cleaned = str(text).strip()
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    return cleaned.strip()


async def _call_llm(prompt_template: str, **kwargs) -> str:
    """使用主模型（用于内容生成：rag/quiz/exam/planner），带重试"""
    import asyncio
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate.from_template(prompt_template)
    llm_timeout = 35.0
    max_retries = 1
    providers = _build_provider_chain(chat_model, backup_chat_model)
    if not providers:
        raise RuntimeError("未配置可用的主模型")

    last_error = None
    for provider_name, provider_model in providers:
        chain = prompt | provider_model | StrOutputParser()
        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.wait_for(chain.ainvoke(kwargs), timeout=llm_timeout)
                logger.info(f"[_call_llm] provider={provider_name}, 返回内容(第{attempt+1}次): {result}")
                result = _sanitize_llm_text(result)
                if _is_incomplete_response(result):
                    logger.warning(f"[_call_llm] provider={provider_name} 响应不完整: {str(result)[:100]}...")
                    if attempt < max_retries:
                        await asyncio.sleep(0.8)
                        continue
                return result
            except Exception as e:
                last_error = e
                is_api_error = _is_retryable_llm_error(e)
                logger.warning(f"[_call_llm] provider={provider_name} 调用失败（第{attempt+1}次）: {e}")
                if attempt >= max_retries or not is_api_error:
                    break
                await asyncio.sleep(1.2)
        if not _is_retryable_llm_error(last_error or Exception("unknown")):
            break
        if provider_name == "primary" and backup_chat_model:
            logger.warning("[_call_llm] 主模型异常，切换备用模型重试")
    if last_error:
        raise last_error
    return ""


async def _call_llm_light(prompt_template: str, **kwargs) -> str:
    """使用轻量模型（用于意图识别、简单分类），带超时保护"""
    import asyncio
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate.from_template(prompt_template)
    providers = _build_provider_chain(light_chat_model, backup_light_chat_model)
    if not providers:
        raise RuntimeError("未配置可用的轻量模型")

    last_error = None
    for provider_name, provider_model in providers:
        chain = prompt | provider_model | StrOutputParser()
        try:
            return await asyncio.wait_for(chain.ainvoke(kwargs), timeout=8.0)
        except Exception as e:
            last_error = e
            logger.warning(f"[Supervisor] 轻量模型调用失败 provider={provider_name}: {e}")
            if not _is_retryable_llm_error(e):
                break
    if last_error:
        raise last_error


async def _call_llm_for_supervisor(prompt_template: str, **kwargs) -> str:
    """使用主模型进行意图识别（更好的理解能力），带重试"""
    import asyncio
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate.from_template(prompt_template)
    llm_timeout = 18.0
    max_retries = 1
    providers = _build_provider_chain(chat_model, backup_chat_model)
    if not providers:
        raise RuntimeError("未配置可用的主模型")

    last_error = None
    for provider_name, provider_model in providers:
        chain = prompt | provider_model | StrOutputParser()
        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.wait_for(chain.ainvoke(kwargs), timeout=llm_timeout)
                logger.info(f"[_call_llm_for_supervisor] provider={provider_name} 返回内容(第{attempt+1}次): {result}")
                result = _sanitize_llm_text(result)
                if _is_incomplete_response(result):
                    logger.warning(f"[_call_llm_for_supervisor] provider={provider_name} 响应不完整: {str(result)[:100]}...")
                    if attempt < max_retries:
                        await asyncio.sleep(0.8)
                        continue
                return result
            except (asyncio.TimeoutError, Exception) as e:
                last_error = e
                is_api_error = _is_retryable_llm_error(e)
                logger.warning(f"[Supervisor] provider={provider_name} 调用失败（第{attempt+1}次）: {e}")
                if attempt >= max_retries or not is_api_error:
                    break
                await asyncio.sleep(1.2)
        if not _is_retryable_llm_error(last_error or Exception("unknown")):
            break
        if provider_name == "primary" and backup_chat_model:
            logger.warning("[Supervisor] 主模型异常，切换备用模型继续意图识别")
    if last_error:
        raise last_error


def _extract_json(text: str) -> dict:
    import re
    if not text or not text.strip():
        return {}
    text = text.strip()

    # 去除 <think>...</think> 思考标签内容
    text = re.sub(r'<think>[\s\S]*?<\/think>', '', text, flags=re.DOTALL).strip()
    if text.startswith('result=') or '<think>' in text:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            text = match.group(0)

    # 如果去除后为空，返回空
    if not text:
        return {}

    # 先按原文直接解析，避免 evidence 内部代码块干扰。
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 仅当整段文本是 fenced block 时才去掉外层围栏，避免误匹配内部 ```bash / ```text。
    fenced_match = re.match(r'^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$', text)
    if fenced_match:
        fenced_body = fenced_match.group(1).strip()
        try:
            return json.loads(fenced_body)
        except json.JSONDecodeError:
            pass

    # 最后回退：提取首个 JSON 对象片段再解析。
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _extract_grounded_evidence(parsed: dict, context: str) -> list[dict]:
    """
    从结构化结果中提取并校验证据：
    - 证据格式必须包含 source + quote
    - quote 必须能在检索 context 中直接命中
    """
    if not isinstance(parsed, dict):
        return []
    evidence = parsed.get("evidence", [])
    if not isinstance(evidence, list):
        return []

    context_text = str(context or "")
    context_norm = re.sub(r'[\s\W_]+', '', context_text).lower()

    def _is_grounded(quote: str) -> bool:
        quote = str(quote or "").strip()
        if not quote:
            return False
        if quote in context_text:
            return True
        # 宽松匹配：忽略空白和标点，降低“引用轻微改写”导致的误杀
        quote_norm = re.sub(r'[\s\W_]+', '', quote).lower()
        if len(quote_norm) < 8:
            return False
        return quote_norm in context_norm

    grounded: list[dict] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        quote = str(item.get("quote", "")).strip()
        if not source or not quote:
            continue
        if _is_grounded(quote):
            grounded.append({"source": source, "quote": quote})
    return grounded


def _validate_supervisor_decision(decision: dict, original_query: str) -> dict:
    """校验 Supervisor 的结构化路由结果，避免静默误路由。"""
    if not isinstance(decision, dict) or not decision:
        raise ValueError("empty supervisor decision")

    valid_routes = {'rag', 'quiz', 'exam', 'planner', 'history', 'chitchat'}
    route = str(decision.get('route', '')).strip().lower()
    if route not in valid_routes:
        raise ValueError(f"invalid supervisor route: {route or 'empty'}")

    params = decision.get('params', {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("supervisor params must be a dict")
    # 统一参数语义：空字符串归一，避免后续 int('') 等异常
    if route == "quiz":
        params = {
            "topic": str(params.get("topic", "") or "").strip(),
            "quiz_type": str(params.get("quiz_type", "") or "选择题").strip() or "选择题",
            "num": _safe_int(params.get("num", 5), default=5, minimum=1, maximum=20),
        }
    elif route == "exam":
        raw_types = params.get("quiz_types", [])
        quiz_types = raw_types if isinstance(raw_types, list) else []
        type_map = {"choice": "选择题", "fill": "填空题", "judge": "判断题", "essay": "简答题"}
        quiz_types = [type_map.get(str(t).strip().lower(), str(t).strip()) for t in quiz_types if str(t).strip()]
        params = {
            "topics": [str(t).strip() for t in (params.get("topics", []) if isinstance(params.get("topics", []), list) else []) if str(t).strip()],
            "quiz_types": quiz_types or ["选择题", "填空题", "判断题", "简答题"],
            "total_questions": _safe_int(params.get("total_questions", 23), default=23, minimum=1, maximum=60),
            "quantity_dist": params.get("quantity_dist", {}) if isinstance(params.get("quantity_dist", {}), dict) else {},
        }

    rewritten_query = _validate_rewritten_query(
        original_query,
        str(decision.get('rewritten_query', '') or '')
    )
    reason = str(decision.get('reason', '') or '').strip()
    return {
        "rewritten_query": rewritten_query,
        "route": route,
        "reason": reason,
        "params": params,
    }


# ==================== 辅助函数 ====================

def _format_history_with_limit(chat_history: list, max_chars: int = 800) -> str:
    """按句子边界截断历史，保留语义完整性"""
    if not chat_history:
        return "（无近期对话）"

    lines = []
    total_chars = 0

    for msg in reversed(chat_history):
        role = "学生" if isinstance(msg, HumanMessage) else "助手"
        content = msg.content if isinstance(msg, (HumanMessage, str)) else getattr(msg, 'content', str(msg))
        msg_text = f"{role}: {content}"

        # 如果加上这条就超限，先试试按句子截断
        if total_chars + len(msg_text) > max_chars:
            # 尝试保留开头部分（通常是更重要的上下文）
            available = max_chars - total_chars - len(f"{role}: ")
            if available > 50:  # 至少保留50字
                truncated = content[:available] + "..."
                lines.insert(0, f"{role}: {truncated}")
                total_chars = max_chars  # 已满
                break
        else:
            lines.insert(0, msg_text)
            total_chars += len(msg_text)

        if total_chars >= max_chars:
            break

    return "\n".join(lines) if lines else "（无近期对话）"


def _validate_rewritten_query(original: str, rewritten: str) -> str:
    """验证重写结果，异常时回退到原query"""
    if not rewritten:
        logger.warning("[Supervisor] 重写结果为空，回退到原query")
        return original

    # 如果重写结果异常短或像乱码
    if len(rewritten) < 2:
        logger.warning(f"[Supervisor] 重写结果过短 '{rewritten}'，回退到原query")
        return original

    # 如果是明显的 JSON 解析错误
    if rewritten in ['{}', '[]', '{"rewritten_query":}', '{"route":}']:
        logger.warning(f"[Supervisor] 重写结果格式异常 '{rewritten}'，回退到原query")
        return original

    # 如果重写结果只是空白字符
    if not rewritten.strip():
        logger.warning("[Supervisor] 重写结果为空字符串，回退到原query")
        return original

    # 语义校验：打招呼类输入不应该被改成完全不同语义的内容
    greeting_keywords = ['你好', 'hi', 'hello', '嗨', '您好']
    original_lower = original.lower().strip()
    rewritten_lower = rewritten.lower()
    if original_lower in greeting_keywords or original_lower.startswith('你好，'):
        # 如果原输入是打招呼，但重写结果包含"出题"等意图词，回退
        if any(kw in rewritten_lower for kw in ['出题', '做题', '练习', '考试', '试卷', '复习']):
            logger.warning(f"[Supervisor] 检测到语义异常：打招呼被改写成'{rewritten}'，回退到原query")
            return original

    return rewritten


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数量：中文约1.5 tokens/字符"""
    if not text:
        return 0
    return int(len(text) * 1.5)


def _check_token_budget(input_query: str, history: str, memory: str) -> bool:
    """检查 token 预算是否充足，返回 False 表示需要精简模式"""
    # 估算输入 token
    input_tokens = _estimate_tokens(input_query)
    history_tokens = _estimate_tokens(history)
    memory_tokens = _estimate_tokens(memory)

    # 预留 2000 tokens 给输出和系统 prompt
    total_estimate = input_tokens + history_tokens + memory_tokens + 2000
    # MiniMax 通常 32K 上下文，这里用 8000 作为安全线
    MAX_BUDGET = 8000

    if total_estimate > MAX_BUDGET:
        logger.warning(f"[Supervisor] Token 预算超限: 估算{total_estimate} > 限制{MAX_BUDGET}")
        return False
    return True


def _get_route_fallback(error: str, original_query: str) -> str:
    """根据错误类型返回合适的 fallback 路由"""
    error_lower = error.lower()
    query = str(original_query or "")

    # 先按 query 关键词兜底，避免模型暂时失败时误路由到 chitchat
    if any(kw in query for kw in ['出题', '做题', '练习', '刷题']):
        return 'quiz'
    if any(kw in query for kw in ['试卷', '考试', '出卷']):
        return 'exam'
    if any(kw in query for kw in ['计划', '复习计划', '学习计划']):
        return 'planner'
    if any(kw in query for kw in ['错题', '历史', '记录', '统计', '正确率']):
        return 'history'

    # API 相关错误：API 不稳定，尝试让 RAG 直接处理
    if _is_retryable_llm_error(Exception(error)):
        # 否则走 rag，因为 rag 通常能给出有用回答
        return 'rag'

    # 超时错误：走 chitchat 快速响应
    if 'timeout' in error_lower or '超时' in error:
        return 'chitchat'

    # 其他错误：默认 chitchat
    return 'chitchat'


# ==================== Supervisor 节点 ====================

async def supervisor_node(state: SupervisorState) -> SupervisorState:
    """Supervisor: 分析意图，输出路由决策（Query重写+意图识别一次完成）"""
    logger.info("[Supervisor] 分析用户意图...")

    # 允许外层先做一次路由判定，然后在 workflow 内复用，避免 quiz/exam 回退路径重复调用 LLM 路由。
    if bool(state.get("supervisor_precomputed", False)):
        precomputed_route = str(state.get("route", "") or "").strip()
        valid_routes = {"rag", "quiz", "exam", "planner", "history", "chitchat"}
        if precomputed_route in valid_routes:
            precomputed_reason = str(state.get("route_reason", "") or "预路由命中")
            precomputed_params = state.get("route_params", {}) if isinstance(state.get("route_params"), dict) else {}
            rewritten_query = str(state.get("input", "") or "").strip() or str(state.get("input", "") or "")
            logger.info(
                f"[Supervisor] 复用预路由结果 route={precomputed_route}, reason={precomputed_reason}"
            )
            return {
                "route": precomputed_route,
                "route_reason": precomputed_reason,
                "route_params": precomputed_params,
                "input": rewritten_query,
            }

    # 预检：打招呼直接走 chitchat，不调用 LLM
    greeting_keywords = ['你好', 'hi', 'hello', '嗨', '您好', 'hi there', 'hey']
    input_lower = state['input'].lower().strip()
    if input_lower in greeting_keywords or input_lower.startswith('你好，') or input_lower.startswith('hi,'):
        logger.info(f"[Supervisor] 检测到打招呼，直接路由 chitchat")
        return {"route": "chitchat", "route_reason": "打招呼问候", "route_params": {}, "input": state['input']}

    # 预检：按句子边界截断历史，保留语义完整性
    recent_history = _format_history_with_limit(state.get('chat_history', []))

    # 使用主模型进行意图识别
    result = ""
    try:
        result = await _call_llm_for_supervisor(
            SUPERVISOR_PROMPT,
            input=state['input'],
            memory_context=state.get('memory_context', ''),
            recent_history=recent_history,
        )
        decision = _validate_supervisor_decision(_extract_json(result), state['input'])
        rewritten_query = decision['rewritten_query']
        route = decision['route']
        reason = decision['reason']
        params = decision['params']

        # 优化4: 检查 token 预算
        if not _check_token_budget(state['input'], recent_history, state.get('memory_context', '')):
            logger.warning("[Supervisor] Token 预算超限，启用精简模式")

        logger.info(f"[Supervisor] Query: '{state['input']}' -> '{rewritten_query}'")
        logger.info(f"[Supervisor] 路由: route={route}, reason={reason}, params={params}")
    except Exception as e:
        logger.warning(f"[Supervisor] 意图识别失败: {e}, result={result[:200] if result else 'empty'}")
        # 优化1: 多层 fallback
        route = _get_route_fallback(str(e), state['input'])
        reason = '意图识别失败，使用 fallback'
        params = {}
        rewritten_query = state['input']

    logger.info(f"[Supervisor] 路由 → {route}")
    return {"route": route, "route_reason": reason, "route_params": params, "input": rewritten_query}


# ==================== SubAgent 节点 ====================

async def rag_subagent_node(state: SupervisorState) -> SupervisorState:
    """RAG SubAgent: 检索课件原始片段，直接回答（单次 LLM）"""
    logger.info("[RAG SubAgent] 检索课件资料...")
    rag_start = time.time()

    from rag.rag_service import RagSummarizeService
    sid = state.get('session_id') or current_session_id.get() or 'default'
    current_session_id.set(sid)

    params = state.get('route_params', {})
    topic = params.get('topic') or state['input']

    # 格式化近期对话历史，供 RAG 参考
    history_lines = []
    for msg in state.get('chat_history', []):
        role = "学生" if isinstance(msg, HumanMessage) else "助手"
        history_lines.append(f"{role}: {msg.content[:300]}")
    recent_history = "\n".join(history_lines) if history_lines else "（无近期对话）"

    try:
        rag = await get_rag_service()
        # retrieve_context 只检索、不调 LLM，避免冗余的双重 LLM 调用
        retrieve_start = time.time()
        context = await rag.retrieve_context(topic, mode="rag_chat")
        logger.info(f"[Latency] rag_retrieve_ms={int((time.time() - retrieve_start) * 1000)}")
        # 如果检索为空，给出友好提示
        if not context or context.strip() == "":
            logger.warning("[RAG SubAgent] 知识库为空，请先上传课件")
            context = "【提示】当前知识库为空，请先上传课件后再提问。你可以通过点击「上传课件」按钮来添加复习资料。"
    except Exception as e:
        logger.warning(f"[RAG SubAgent] 检索失败: {e}")
        context = "【提示】知识库检索遇到问题，请确保已上传课件。如未上传，请先上传复习资料。"

    llm_start = time.time()
    result_raw = await _call_llm(
        RAG_SUBAGENT_PROMPT,
        memory_context=state.get('memory_context', ''),
        recent_history=recent_history,
        context=context,
        input=state['input'],
    )
    logger.info(f"[Latency] rag_answer_llm_ms={int((time.time() - llm_start) * 1000)}")

    parsed = _extract_json(result_raw)
    answer = str(parsed.get("answer", "")).strip() if isinstance(parsed, dict) else ""
    grounded_evidence = _extract_grounded_evidence(parsed, context)

    if answer and grounded_evidence:
        rag_inc("rag_grounded_pass_count", 1)
        logger.info(f"[RAG SubAgent] 回答依据校验通过，evidence_count={len(grounded_evidence)}")
        logger.info(f"[Latency] rag_total_ms={int((time.time() - rag_start) * 1000)}")
        return {
            "subagent_result": {
                "answer": answer,
                "evidence": grounded_evidence,
            },
            "final_answer": answer,
        }

    # 非严格模式：课件作为主要依据即可，不再因证据条目不足直接拒答
    if answer:
        rag_inc("rag_grounded_partial_count", 1)
        logger.warning("[RAG SubAgent] 回答依据条目不足，按课件优先策略继续返回答案")
        logger.info(f"[Latency] rag_total_ms={int((time.time() - rag_start) * 1000)}")
        return {
            "subagent_result": {
                "answer": answer,
                "evidence": grounded_evidence,
                "reason": "evidence_partial",
            },
            "final_answer": answer,
        }

    logger.warning("[RAG SubAgent] 无有效回答，触发保守提示")
    rag_inc("rag_grounded_fail_count", 1)
    logger.info(f"[Latency] rag_total_ms={int((time.time() - rag_start) * 1000)}")
    fallback = (
        "根据当前检索结果，我暂时无法提取到可核验的课件依据。"
        "为避免误导，请换个更具体的问题，或补充相关课件后再试。"
    )
    return {
        "subagent_result": {
            "answer": fallback,
            "evidence": grounded_evidence,
            "reason": "no_grounded_evidence",
        },
        "final_answer": fallback,
    }


async def quiz_subagent_node(state: SupervisorState) -> SupervisorState:
    """Quiz SubAgent: 调用 Reflexion 出题工作流"""
    logger.info("[Quiz SubAgent] 启动 Reflexion 出题流程...")
    logger.info(f"[Quiz SubAgent] state.input={state.get('input', '')[:200]}...")
    logger.info(f"[Quiz SubAgent] route_params={state.get('route_params', {})}")
    logger.info(f"[Quiz SubAgent] session_id={state.get('session_id', 'unknown')}")

    from agent.multi_agent.quiz_agent import run_quiz_agent
    from api.routers.knowledge import get_sample_paper_context
    from utils.memory_service import memory_manager

    sid = state.get('session_id') or current_session_id.get() or 'default'
    current_session_id.set(sid)
    sample_ctx = get_sample_paper_context(sid)

    params = state.get('route_params', {})
    topic = params.get('topic', '')
    quiz_type = params.get('quiz_type') or '选择题'
    num = _safe_int(params.get('num', 3), default=3, minimum=1, maximum=20)
    if not params.get('num'):
        logger.warning(f"[Quiz SubAgent] num 参数缺失/非法，已回退默认值 num={num}")

    # 如果 topic 为空或只有意图词（题、出题等），视为通用出题请求
    # 不拦截，让 LLM 自己选择合适的知识点出题
    intent_only_keywords = ['题', '出题', '做题', '练习', '考试', '测试', '一道', '几个', '一些', '做一个', '出']
    topic_stripped = topic.strip()
    is_generic_request = (
        not topic_stripped or
        topic_stripped in intent_only_keywords
    )

    weak_context = ""
    stats = memory_manager.get_knowledge_point_stats(sid)
    if stats:
        weak_points = [(kp, data) for kp, data in stats.items() if data.get('weak')]
        if weak_points and not is_generic_request:
            # 只有在用户指定了具体知识点时，才优先出薄弱点题目
            weak_lines = [f"- {kp}: 正确率{data['accuracy']:.0f}%（{data['correct']}/{data['total']}）" for kp, data in weak_points]
            weak_context = f"\n\n【薄弱点提醒】以下知识点正确率低于60%，请优先出这些方面的题目：\n" + "\n".join(weak_lines)
            logger.info(f"[Quiz SubAgent] 检测到 {len(weak_points)} 个薄弱点，将优先出相关题目")

    # 通用请求不再注入固定锚点词；交给出题链路按课件池随机抽样检索
    effective_topic = topic_stripped if is_generic_request else topic
    quiz_result = await run_quiz_agent(
        effective_topic + weak_context,
        quiz_type,
        num,
        sample_ctx,
        force_llm_critic=bool(state.get("quiz_force_llm_critic", False)),
    )
    quiz_text = quiz_result.get("text", "") if isinstance(quiz_result, dict) else quiz_result
    if not quiz_text:
        quiz_text = "抱歉，出题失败了，请稍后重试。"
        if isinstance(quiz_result, dict):
            quiz_result = {**quiz_result, "text": quiz_text}
        else:
            quiz_result = quiz_text
    return {"subagent_result": quiz_result, "final_answer": quiz_text}


async def exam_subagent_node(state: SupervisorState) -> SupervisorState:
    """Exam SubAgent: 调用 Reflexion 出卷工作流"""
    logger.info("[Exam SubAgent] 启动 Reflexion 出卷流程...")
    exam_start = time.time()
    logger.info(f"[Exam SubAgent] state.input={state.get('input', '')[:200]}...")
    logger.info(f"[Exam SubAgent] route_params={state.get('route_params', {})}")
    logger.info(f"[Exam SubAgent] session_id={state.get('session_id', 'unknown')}")
    logger.info(f"[Exam SubAgent] stage_plan={state.get('exam_stage_plan', False)}, rerun_stage={state.get('exam_rerun_stage')}")

    from agent.multi_agent.quiz_agent import run_exam_agent
    from api.routers.knowledge import get_sample_paper_context
    from utils.memory_service import memory_manager

    sid = state.get('session_id') or current_session_id.get() or 'default'
    current_session_id.set(sid)
    sample_ctx = get_sample_paper_context(sid)

    params = state.get('route_params', {})
    topics = params.get('topics', [])
    quiz_types = params.get('quiz_types')
    if not quiz_types or not isinstance(quiz_types, list):
        quiz_types = ['选择题', '填空题', '判断题', '简答题']
    total = _safe_int(params.get('total_questions', 23), default=23, minimum=1, maximum=60)
    if params.get('total_questions', None) in ("", None):
        logger.warning(f"[Exam SubAgent] total_questions 缺失/非法，已回退默认值 total={total}")
    raw_quantity_dist = params.get('quantity_dist', {}) or {}
    quantity_dist = {}
    if isinstance(raw_quantity_dist, dict):
        for key in ("choice", "fill", "judge", "essay"):
            if key in raw_quantity_dist:
                quantity_dist[key] = _safe_int(raw_quantity_dist.get(key), default=0, minimum=0, maximum=60)

    # LLM 意图识别失败时，route_params 可能为空：从原始输入里做一次保底抽取
    raw_input = str(state.get("input", "") or "")
    if not quantity_dist and raw_input:
        alias_to_key = {
            "选择题": "choice",
            "填空题": "fill",
            "判断题": "judge",
            "简答题": "essay",
            "解答题": "essay",
        }
        for label, key in alias_to_key.items():
            m = re.search(rf"{label}\s*(\d+)\s*[道题]?", raw_input)
            if m:
                quantity_dist[key] = _safe_int(m.group(1), default=0, minimum=0, maximum=60)
        # 常见综合卷默认分布（用户没写数量 or 解析失败）
        if not any(int(quantity_dist.get(k, 0)) > 0 for k in ("choice", "fill", "judge", "essay")) and any(
            w in raw_input for w in ["综合测试卷", "综合试卷", "期末试卷", "出一套试卷", "生成试卷"]
        ):
            quantity_dist = {"choice": 10, "fill": 5, "judge": 5, "essay": 3}
            logger.warning("[Exam SubAgent] route_params 缺失，已按综合卷默认分布回填 quantity_dist=10/5/5/3")

    if any(int(quantity_dist.get(k, 0)) > 0 for k in ("choice", "fill", "judge", "essay")):
        total = int(sum(int(quantity_dist.get(k, 0) or 0) for k in ("choice", "fill", "judge", "essay")))
        if total > 0:
            quiz_types = []
            if quantity_dist.get("choice", 0) > 0:
                quiz_types.append("选择题")
            if quantity_dist.get("fill", 0) > 0:
                quiz_types.append("填空题")
            if quantity_dist.get("judge", 0) > 0:
                quiz_types.append("判断题")
            if quantity_dist.get("essay", 0) > 0:
                quiz_types.append("简答题")

    # 如果 topics 为空或只有意图词，视为通用出卷请求
    # 不拦截，让 LLM 自己选择合适的知识点出卷
    intent_only_keywords = ['卷', '试卷', '考试', '测试', '题', '出卷', '出题', '综合', '做一个', '来', '出', '一张', '一份']
    primary_topic = topics[0] if topics else ""
    topic_stripped = primary_topic.strip()
    is_generic_request = (
        not topic_stripped or
        topic_stripped in intent_only_keywords
    )

    weak_context = ""
    stats = memory_manager.get_knowledge_point_stats(sid)
    if stats:
        weak_points = [(kp, data) for kp, data in stats.items() if data.get('weak')]
        if weak_points and not is_generic_request:
            # 只有在用户指定了具体知识点时，才优先出薄弱点题目
            weak_lines = [f"- {kp}: 正确率{data['accuracy']:.0f}%（{data['correct']}/{data['total']}）" for kp, data in weak_points]
            weak_context = f"\n\n【薄弱点提醒】以下知识点正确率低于60%，请优先出这些方面的题目：\n" + "\n".join(weak_lines)
            logger.info(f"[Exam SubAgent] 检测到 {len(weak_points)} 个薄弱点，将优先出相关题目")

    # 如果用户没有指定具体知识点，topics 为空，让 LLM 自己选择
    topics_with_weak = [t + weak_context for t in topics] if topics else []

    logger.info(f"[Exam SubAgent] topics={topics}, quiz_types={quiz_types}, total={total}, quantity_dist={quantity_dist}")

    if bool(state.get("exam_stage_plan", False)):
        exam_result = await run_exam_agent(
            topics_with_weak,
            quiz_types,
            total,
            sample_ctx,
            quantity_dist,
            stage_plan=True,
            rerun_stage=str(state.get("exam_rerun_stage") or "").strip() or None,
            partial_questions=state.get("exam_partial_questions") if isinstance(state.get("exam_partial_questions"), list) else None,
            exam_fast_mode=bool(state.get("exam_fast_mode", True)),
        )
    else:
        exam_result = await run_exam_agent(
            topics_with_weak, quiz_types, total, sample_ctx, quantity_dist,
            exam_fast_mode=bool(state.get("exam_fast_mode", True)),
        )
    logger.info(f"[Latency] exam_generate_ms={int((time.time() - exam_start) * 1000)}")
    exam_text = exam_result.get("text", "") if isinstance(exam_result, dict) else exam_result
    if not exam_text or not str(exam_text).strip():
        exam_text = "试卷生成失败：本次为严格模式（禁止降级），请稍后重试或减少题量。"
        if isinstance(exam_result, dict):
            exam_result = {**exam_result, "text": exam_text}
        else:
            exam_result = exam_text
    return {"subagent_result": exam_result, "final_answer": exam_text}


async def planner_subagent_node(state: SupervisorState) -> SupervisorState:
    """Planner SubAgent: 制定复习计划"""
    logger.info("[Planner SubAgent] 生成复习计划...")

    result = await _call_llm(
        PLANNER_SUBAGENT_PROMPT,
        memory_context=state.get('memory_context', ''),
        input=state['input'],
    )
    return {"subagent_result": result, "final_answer": result}


async def chitchat_node(state: SupervisorState) -> SupervisorState:
    """Chitchat: 处理闲聊、问候、感谢、无关话题"""
    logger.info("[Chitchat] 处理闲聊")

    session_id = state.get('session_id') or current_session_id.get() or 'default'
    current_session_id.set(session_id)

    # 获取近期对话历史
    chat_history = state.get('chat_history', [])
    recent_history = _format_history_with_limit(chat_history, max_chars=600)

    # 获取图谱记忆
    memory_context = state.get('memory_context', '')

    context_str = f"【近期对话】\n{recent_history}\n\n【知识点备忘】\n{memory_context}".strip()

    result = await _call_llm(
        CHITCHAT_PROMPT,
        input=state['input'],
        context=context_str,
    )
    return {"subagent_result": result, "final_answer": result}


def _format_practice_history_rows(history: list[dict], limit: int = 20) -> str:
    """格式化错题记录，输出可操作的 ID。"""
    if not history:
        return "暂无练习记录"

    safe_limit = max(1, min(int(limit or 20), 200))
    rows = history[:safe_limit]
    lines = [f"【练习历史（最近 {len(rows)} 条）】"]
    for i, rec in enumerate(rows, 1):
        status = "✅" if rec.get("is_correct") else "❌"
        rec_id = rec.get("id")
        kp = rec.get("knowledge_point") or "未标注"
        q_content = str(rec.get("question_content") or "").strip().replace("\n", " ")
        lines.append(f"{i}. [ID:{rec_id}] {status} [{kp}] {q_content[:80]}")
    return "\n".join(lines)


def _format_message_rows(messages: list[dict], limit: int = 30) -> tuple[str, list[dict]]:
    """格式化对话历史，按“最新在前”展示并返回同序列表。"""
    if not messages:
        return "暂无历史对话记录", []

    safe_limit = max(1, min(int(limit or 30), 200))
    ordered = list(reversed(messages[-safe_limit:]))
    lines = [f"【历史对话（最近 {len(ordered)} 条，最新在前）】"]
    for i, msg in enumerate(ordered, 1):
        role = "学生" if msg.get("role") == "human" else "助手"
        timestamp = msg.get("timestamp", 0)
        content = str(msg.get("content", "")).strip().replace("\n", " ")
        lines.append(f"{i}. [TS:{timestamp}] {role}: {content[:120]}")
    return "\n".join(lines), ordered


def _resolve_practice_id_by_index(history: list[dict], index: int) -> int | None:
    """把“第N条错题”映射为实际 record_id。"""
    if index <= 0:
        return None
    pos = index - 1
    if pos < 0 or pos >= len(history):
        return None
    rec_id = history[pos].get("id")
    return int(rec_id) if rec_id is not None else None


def _resolve_message_ts_by_index(ordered_messages: list[dict], index: int) -> int | None:
    """把“第N条消息”映射为 timestamp。"""
    if index <= 0:
        return None
    pos = index - 1
    if pos < 0 or pos >= len(ordered_messages):
        return None
    ts = ordered_messages[pos].get("timestamp")
    return int(ts) if ts is not None else None


async def history_subagent_node(state: SupervisorState) -> SupervisorState:
    """History SubAgent: 查看练习历史、答题记录、错题分析（由LLM理解用户意图并回答）"""
    from utils.memory_service import memory_manager

    logger.info("[History SubAgent] 处理练习历史查询...")

    sid = state.get('session_id') or current_session_id.get() or 'default'
    current_session_id.set(sid)

    params = state.get('route_params', {})
    limit = params.get('limit', 20)

    # 先执行可确定的管理动作（查/删/清空），避免再走 LLM 推断。
    action = parse_history_action(state.get("input", ""))
    action_name = action.get("action", "none")
    if action_name != "none":
        if action_name == "list_practice":
            history = memory_manager.get_practice_history(sid, max(1, min(int(limit or 20), 200)))
            text = _format_practice_history_rows(history, limit=limit)
            return {"subagent_result": text, "final_answer": text}

        if action_name == "delete_practice":
            history = memory_manager.get_practice_history(sid, 200)
            record_id = action.get("record_id")
            if record_id is None and action.get("record_index") is not None:
                record_id = _resolve_practice_id_by_index(history, int(action.get("record_index")))
            if record_id is None:
                text = "请提供要删除的错题 ID，或说“删除第N条错题”。"
                return {"subagent_result": text, "final_answer": text}
            deleted = memory_manager.delete_practice_record(sid, int(record_id))
            if not deleted:
                text = f"未找到错题记录 ID={record_id}，删除失败。"
                return {"subagent_result": text, "final_answer": text}
            latest = memory_manager.get_practice_history(sid, max(1, min(int(limit or 10), 50)))
            text = f"已删除错题记录 ID={record_id}\n\n{_format_practice_history_rows(latest, limit=min(int(limit or 10), 20))}"
            return {"subagent_result": text, "final_answer": text}

        if action_name == "clear_practice":
            deleted_count = memory_manager.clear_practice_history(sid)
            text = f"已清空错题记录，共删除 {deleted_count} 条。"
            return {"subagent_result": text, "final_answer": text}

        if action_name == "list_messages":
            messages = memory_manager.get_messages(sid)
            text, _ = _format_message_rows(messages, limit=limit)
            return {"subagent_result": text, "final_answer": text}

        if action_name == "delete_message":
            messages = memory_manager.get_messages(sid)
            text_rows, ordered_messages = _format_message_rows(messages, limit=200)
            timestamp = action.get("timestamp")
            if timestamp is None and action.get("message_index") is not None:
                timestamp = _resolve_message_ts_by_index(ordered_messages, int(action.get("message_index")))
            if timestamp is None:
                text = "请提供要删除的消息 timestamp，或说“删除第N条消息”。"
                return {"subagent_result": text, "final_answer": text}
            deleted = memory_manager.delete_message(sid, int(timestamp))
            if not deleted:
                text = f"未找到消息 TS={timestamp}，删除失败。\n\n{text_rows}"
                return {"subagent_result": text, "final_answer": text}
            latest_messages = memory_manager.get_messages(sid)
            latest_text, _ = _format_message_rows(latest_messages, limit=min(int(limit or 20), 50))
            text = f"已删除消息 TS={timestamp}\n\n{latest_text}"
            return {"subagent_result": text, "final_answer": text}

        if action_name == "clear_messages":
            success = memory_manager.clear_messages(sid)
            text = "已清空历史对话记录。" if success else "清空失败：会话不存在。"
            return {"subagent_result": text, "final_answer": text}

    # 获取统计数据和历史记录
    stats = memory_manager.get_knowledge_point_stats(sid)
    history = memory_manager.get_practice_history(sid, limit)

    # 格式化统计数据
    if stats:
        stats_lines = []
        for kp, data in stats.items():
            weak = "🔴 薄弱" if data.get('weak') else "🟢 掌握"
            stats_lines.append(f"- {kp}: {data['correct']}/{data['total']} ({data['accuracy']:.1f}%) {weak}")
        stats_str = "\n".join(stats_lines)
    else:
        stats_str = "暂无练习统计数据"

    # 格式化历史记录
    if history:
        history_lines = []
        for rec in history:
            status = "✅" if rec.get('is_correct') else "❌"
            kp = rec.get('knowledge_point', '未知')
            q_content = rec.get('question_content', '')[:50]
            history_lines.append(f"{status} [{kp}] {q_content}...")
        history_str = "\n".join(history_lines)
    else:
        history_str = "暂无练习记录"

    # LLM 根据用户请求和实际数据生成回答
    result = await _call_llm(
        HISTORY_SUBAGENT_PROMPT,
        input=state['input'],
        stats=stats_str,
        history=history_str,
    )
    return {"subagent_result": result, "final_answer": result}


# ==================== 路由函数 ====================

def route_to_subagent(state: SupervisorState) -> str:
    """根据 Supervisor 决策路由到对应 SubAgent"""
    route = state.get('route', 'rag')
    valid = {"rag", "quiz", "exam", "planner", "history", "chitchat"}
    return route if route in valid else "rag"


# ==================== 构建工作流图 ====================

def build_supervisor_graph() -> StateGraph:
    """
    Supervisor 工作流：
    supervisor → (路由判断) → [rag_agent | quiz_agent | exam_agent | planner_agent | history_agent | chitchat] → END
    """
    graph = StateGraph(SupervisorState)

    # 注册节点
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag_agent", rag_subagent_node)
    graph.add_node("quiz_agent", quiz_subagent_node)
    graph.add_node("exam_agent", exam_subagent_node)
    graph.add_node("planner_agent", planner_subagent_node)
    graph.add_node("history_agent", history_subagent_node)
    graph.add_node("chitchat", chitchat_node)

    # 入口
    graph.set_entry_point("supervisor")

    # Supervisor 条件路由
    graph.add_conditional_edges(
        "supervisor",
        route_to_subagent,
        {
            "rag": "rag_agent",
            "quiz": "quiz_agent",
            "exam": "exam_agent",
            "planner": "planner_agent",
            "history": "history_agent",
            "chitchat": "chitchat",
        }
    )

    # 所有 SubAgent 执行完毕后结束
    for node in ("rag_agent", "quiz_agent", "exam_agent", "planner_agent", "history_agent", "chitchat"):
        graph.add_edge(node, END)

    return graph.compile()


supervisor_workflow = build_supervisor_graph()
