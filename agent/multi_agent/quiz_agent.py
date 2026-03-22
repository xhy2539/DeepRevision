"""
多 Agent 出题系统 - Reflexion 架构
Agent 1: 出题 Agent  — 生成题目 + 推理链（为什么这样出题、答案依据何处）
Agent 2: Critic Agent — 质疑推理链，找出逻辑漏洞，输出结构化批评
Agent 3: Revise Agent — 逐条回应批评，修订题目并给出修改说明
循环：critique → revise → critique，最多 2 轮
"""
import json
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from model.factory import chat_model
from rag.rag_service import RagSummarizeService
from utils.session_context import current_session_id
from utils.logger_handler import logger


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
    # Agent 2 输出
    critique: dict          # Critic 的结构化批评
    initial_score: int      # 第一轮 Critic 评分（效果验证基准）
    # Agent 3 输出
    revised_quiz: str       # 修订后题目
    revision_notes: str     # 针对批评的修改说明
    # 控制
    reflection_rounds: int  # 已进行的反思轮次


class ExamPaperState(TypedDict):
    # 输入
    topics: List[str]
    quiz_types: List[str]
    total_questions: int
    sample_paper_context: str
    contexts: List[str]
    # Agent 1 输出
    exam_paper: str
    reasoning: str
    # Agent 2 输出
    critique: dict
    initial_score: int      # 第一轮 Critic 评分（效果验证基准）
    # Agent 3 输出
    revised_exam: str
    revision_notes: str
    # 控制
    reflection_rounds: int


# ==================== 工具函数 ====================

async def get_rag_context(topic: str) -> str:
    """获取 RAG 原始检索片段（不做最终 LLM 总结，出题 Agent 自己基于原文出题）"""
    logger.info(f"[RAG Context] 正在检索 topic={topic[:50]}...")
    try:
        rag = RagSummarizeService()
        context = await rag.retrieve_context(topic)
        logger.info(f"[RAG Context] 检索完成，返回长度={len(context)}")
        if not context or context.strip() == "":
            logger.warning("[RAG Context] 知识库为空，返回提示信息")
            return "【提示】当前知识库为空，无法生成题目。请先上传课件后再请求出题。"
        return context
    except Exception as e:
        logger.error(f"[RAG Context] 检索异常: {e}")
        return f"【提示】知识库检索失败: {str(e)}"


async def call_llm(prompt_template: str, **kwargs) -> str:
    """调用 LLM"""
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | chat_model | StrOutputParser()
    return await chain.ainvoke(kwargs)


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON，自动处理 markdown 代码块"""
    import re
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


# ==================== 提示词 ====================

# ---------- Agent 1: 出题 + 推理链 ----------
GENERATE_WITH_REASONING_PROMPT = """你是一位资深大学期末考试命题专家。根据课件资料生成题目，同时给出详细推理链。

【考点】{topic}
【题型】{quiz_type}
【题目数量】{num}
【样卷格式参考】{sample_paper_context}

【课件资料】（所有题目必须严格基于此）:
{context}

请生成题目，并为每道题提供推理链（说明为什么这样出题、答案依据在课件哪里）。

返回如下 JSON（只返回 JSON，不要其他内容）：
```json
{{
    "quiz": "完整题目文本（包含题干、选项、答案、解析）",
    "reasoning": {{
        "knowledge_points": ["第1题考察知识点X，来源：课件原文'...'", "第2题考察..."],
        "answer_evidence": ["第1题答案A的依据：课件原文'...'", "第2题..."],
        "distractor_design": ["第1题干扰项B的设计逻辑：容易与X混淆，因为...", "第2题..."]
    }}
}}
```

【出题原则】
- 答案必须有课件原文依据，禁止凭空编造
- 干扰项要有区分度，不能用"以上都是/都不是"
- 难度分布：简单30%、中等50%、困难20%"""


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

返回如下 JSON（只返回 JSON）：
```json
{{
    "approved": false,
    "overall_score": 75,
    "critique": "总体评价：出题者的推理链在第X题出现了...",
    "reasoning_flaws": [
        {{
            "question": "第1题",
            "flaw": "推理链称答案依据是课件'...'，但实际课件原文是'...'，两者不符",
            "severity": "high"
        }}
    ],
    "specific_issues": [
        {{
            "question": "第2题",
            "issue": "干扰项C与正确答案区分度不够",
            "suggestion": "将C改为...，因为这样能更有效地区分掌握程度"
        }}
    ]
}}
```

判断标准：
- approved=false 当且仅当存在 high severity 问题，或 overall_score < 70
- 格式小问题、轻微难度偏差不影响 approved"""


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

请根据批评修订题目，并详细说明你的修改依据。

返回如下 JSON（只返回 JSON）：
```json
{{
    "revised_quiz": "修订后的完整题目文本（包含题干、选项、答案、解析）",
    "revision_notes": "修改说明：\\n1. 针对'第1题答案依据不准确'：将答案从B改为C，因为课件原文明确写道'...'\\n2. 针对'第2题干扰项区分度'：将选项C改为'...'，原因是...",
    "addressed_issues": ["第1题答案错误", "第2题干扰项设计"]
}}
```"""


# ---------- 试卷版本 ----------
EXAM_GENERATE_WITH_REASONING_PROMPT = """你是一位资深大学期末考试命题专家。根据课件资料生成完整试卷。

【重要】出题前必须先联网搜索该学科的典型期末试卷风格、常见考点和优秀题目作为参考。

【考点范围】{topics}
【题目类型】{quiz_types}
【总题数】{total_questions}
【总分】100分（必须正好100分，不能多也不能少）
【样卷格式参考】{sample_paper_context}

【课件资料】:
{contexts}

【硬性要求】（必须严格遵守，任何一条不遵守都判定为不合格）：
1. 总题数：必须正好 {total_questions} 道，不能多也不能少
2. 总分：必须正好 100分，不能多也不能少
3. 题型分布（推荐）：选择题 10题×2分=20分，填空题 10题×2分=20分，判断题 10题×2分=20分，简答题 4题×10分=40分
4. 选择题必须有 A、B、C、D 四个完整选项，题干描述要详细（至少30字）
5. 题目必须全部统一编号：1、2、3... 到 {total_questions}，不能每个题型单独编号
6. 答案优先以课件原文为依据，如果没有明确依据可标注"参考答案"
7. 简答题/解答题必须有2-3个小问，每问5分左右
8. 填空题和判断题题干也要详细描述
9. 题目描述要详细，包含足够信息让考生理解题意
10. 难度分布：简单题占30%，中等题占50%，难题占20%
11. 【严禁重复】同一知识点、同一题型、相似问法不得出现超过1次，必须确保每道题知识点不重复
12. 【去重检查】生成完成后必须检查所有题目，确保没有完全相同的题干、选项或考点

请生成完整试卷。

【试卷格式要求】（必须严格遵守）：
```
## 《期末考试试卷》

一、选择题（每题2分，共10题）
1. [详细题干内容，描述要充分]
   A. 选项1  B. 选项2  C. 选项3  D. 选项4

2. [详细题干内容]
   A. 选项1  B. 选项2  C. 选项3  D. 选项4

...

二、填空题（每题2分，共10题）
11. [详细题干内容，需要填空的部位用括号表示]

12. ...

三、判断题（每题2分，共10题）
21. [详细题干内容]

22. ...

四、简答题/解答题（每题10分，共4题，每题2-3问）
31. [问题描述]（10分）
    (1) [小问1]（5分）
    (2) [小问2]（5分）

32. [问题描述]
    (1) ...
    (2) ...
```

返回如下 JSON（只返回 JSON）：
```json
{{
    "exam_paper": "完整试卷文本（包含所有题目、答案、解析，按照上述格式）",
    "reasoning": {{
        "topic_coverage": ["选择题主要考察X考点，来源课件...", "填空题考察..."],
        "answer_evidence": ["第1题答案依据：课件原文'...'", "第2题..."],
        "difficulty_distribution": "简单题X道（第N题...），中等X道，困难X道..."
    }}
}}
```"""


EXAM_CRITIQUE_PROMPT = """你是严格的试卷评审专家。深入质疑出题者的推理链，核实每个声明是否与课件一致。

【考点范围】{topics}
【课件资料】（唯一权威依据）:
{contexts}

【待评审试卷】:
{exam_paper}

【出题推理链】:
{reasoning}

【重点审查项】你必须特别严格检查以下问题：
1. 【重复检查】是否存在完全相同或高度相似的题目？同一知识点是否重复出题？
2. 【总分验证】所有题目分值加起来是否正好100分？
3. 【题数验证】题目数量是否正好{total_questions}道？
4. 【编号验证】题目是否统一编号（1、2、3...），而非每个题型单独编号？

返回如下 JSON（只返回 JSON）：
```json
{{
    "approved": false,
    "overall_score": 75,
    "critique": "总体评价...",
    "reasoning_flaws": [
        {{"question": "第X题", "flaw": "...", "severity": "high|medium|low"}}
    ],
    "specific_issues": [
        {{"question": "第X题", "issue": "...", "suggestion": "..."}}
    ],
    "duplicate_check": {{"has_duplicates": true/false, "duplicate_questions": ["第X题与第Y题重复", ...]}}
}}
```"""


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

【关键提醒】
1. 如果评审批评指出存在重复题目，必须删除或替换重复的题目
2. 确保修订后总分仍然是100分
3. 确保修订后题目数量仍然是{total_questions}道
4. 返回完整的修订后试卷，不要只返回修改的部分

返回如下 JSON（只返回 JSON）：
```json
{{
    "revised_exam": "【完整试卷】（包含所有题目、答案、解析，不是只输出修订的部分）",
    "revision_notes": "修改说明：\\n1. 针对'...'：...",
    "addressed_issues": ["问题1", "问题2"]
}}
```"""


# ==================== 节点函数（单题）====================

async def generate_with_reasoning_node(state: QuizState) -> QuizState:
    """Agent 1: 出题 + 推理链"""
    logger.info(f"[Agent1-出题] {state['topic']} / {state['quiz_type']} / {state['num']}道")
    sample = state.get('sample_paper_context', '') or '无样卷参考，使用默认格式'

    result = await call_llm(
        GENERATE_WITH_REASONING_PROMPT,
        topic=state['topic'],
        quiz_type=state['quiz_type'],
        num=state['num'],
        context=state['context'],
        sample_paper_context=sample,
    )
    try:
        parsed = _extract_json(result)
        quiz = parsed.get('quiz', '')
        reasoning = json.dumps(parsed.get('reasoning', {}), ensure_ascii=False)
    except Exception:
        quiz = result
        reasoning = '{}'

    return {"quiz": quiz, "reasoning": reasoning}


async def critique_quiz_node(state: QuizState) -> QuizState:
    """Agent 2: Critic — 质疑推理链，输出结构化批评"""
    quiz = state.get('revised_quiz') or state.get('quiz', '')
    logger.info("[Agent2-Critic] 质疑出题推理链...")

    result = await call_llm(
        CRITIQUE_PROMPT,
        topic=state['topic'],
        quiz_type=state['quiz_type'],
        context=state['context'],
        quiz=quiz,
        reasoning=state.get('reasoning', '{}'),
    )
    try:
        critique = _extract_json(result)
    except Exception:
        critique = {"approved": True, "overall_score": 80, "reasoning_flaws": [], "specific_issues": []}

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

    result = await call_llm(
        REVISE_WITH_REFLECTION_PROMPT,
        topic=state['topic'],
        quiz_type=state['quiz_type'],
        context=state['context'],
        quiz=state.get('quiz', ''),
        reasoning=state.get('reasoning', '{}'),
        critique=json.dumps(state.get('critique', {}), ensure_ascii=False),
    )
    try:
        parsed = _extract_json(result)
        revised = parsed.get('revised_quiz', state.get('quiz', ''))
        notes = parsed.get('revision_notes', '')
    except Exception:
        revised = state.get('quiz', '')
        notes = ''

    return {
        "revised_quiz": revised,
        "revision_notes": notes,
        "reflection_rounds": round_n,
    }


# ==================== 节点函数（试卷）====================

async def generate_exam_with_reasoning_node(state: ExamPaperState) -> ExamPaperState:
    """Agent 1: 出卷 + 推理链"""
    logger.info(f"[Agent1-出卷] {state['total_questions']}道，考点: {state['topics']}")
    contexts_combined = "\n\n".join([
        f"【{topic}】\n{ctx}"
        for topic, ctx in zip(state['topics'], state['contexts'])
    ])
    sample = state.get('sample_paper_context', '') or '无样卷参考'

    result = await call_llm(
        EXAM_GENERATE_WITH_REASONING_PROMPT,
        topics=", ".join(state['topics']),
        quiz_types=", ".join(state['quiz_types']),
        total_questions=state['total_questions'],
        sample_paper_context=sample,
        contexts=contexts_combined,
    )
    try:
        parsed = _extract_json(result)
        exam_paper = parsed.get('exam_paper', '')
        reasoning = json.dumps(parsed.get('reasoning', {}), ensure_ascii=False)
    except Exception:
        exam_paper = result
        reasoning = '{}'

    return {"exam_paper": exam_paper, "reasoning": reasoning}


async def critique_exam_node(state: ExamPaperState) -> ExamPaperState:
    """Agent 2: Critic — 评审试卷推理链"""
    exam_paper = state.get('revised_exam') or state.get('exam_paper', '')
    contexts_combined = "\n\n".join(state['contexts'])[:2000]
    logger.info("[Agent2-Critic-试卷] 质疑出卷推理链...")

    result = await call_llm(
        EXAM_CRITIQUE_PROMPT,
        topics=", ".join(state['topics']),
        contexts=contexts_combined,
        exam_paper=exam_paper,
        reasoning=state.get('reasoning', '{}'),
        total_questions=state.get('total_questions', 10),
    )
    try:
        critique = _extract_json(result)
    except Exception:
        critique = {"approved": True, "overall_score": 80, "reasoning_flaws": [], "specific_issues": []}

    score = critique.get('overall_score', 80)
    logger.info(f"[Agent2-Critic-试卷] approved={critique.get('approved')}, score={score}")

    update: dict = {"critique": critique}
    if state.get('reflection_rounds', 0) == 0 and not state.get('initial_score'):
        update["initial_score"] = score
    return update


async def revise_exam_with_reflection_node(state: ExamPaperState) -> ExamPaperState:
    """Agent 3: Revise — 针对批评修订试卷"""
    round_n = state.get('reflection_rounds', 0) + 1
    logger.info(f"[Agent3-Revise-试卷] 第 {round_n} 轮反思修订...")
    contexts_combined = "\n\n".join(state['contexts'])[:2000]

    result = await call_llm(
        EXAM_REVISE_WITH_REFLECTION_PROMPT,
        topics=", ".join(state['topics']),
        contexts=contexts_combined,
        exam_paper=state.get('exam_paper', ''),
        reasoning=state.get('reasoning', '{}'),
        critique=json.dumps(state.get('critique', {}), ensure_ascii=False),
        total_questions=state.get('total_questions', 10),
    )
    try:
        parsed = _extract_json(result)
        revised = parsed.get('revised_exam', state.get('exam_paper', ''))
        notes = parsed.get('revision_notes', '')
    except Exception:
        revised = state.get('exam_paper', '')
        notes = ''

    return {
        "revised_exam": revised,
        "revision_notes": notes,
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

    needs = (score < 70) or (len(high_flaws) > 0) or (not approved) or has_duplicates
    logger.info(
        f"[Critic判断] score={score}, high_flaws={len(high_flaws)}, "
        f"approved={approved}, duplicates={has_duplicates} → {'修订' if needs else '通过'}"
    )
    return needs


def should_revise_quiz(state: QuizState) -> str:
    """最多 2 轮反思；多条件交叉验证决定是否修订"""
    if state.get('reflection_rounds', 0) >= 2:
        logger.info("[Reflection] 达到最大轮次（2轮），输出最终结果")
        return "end"
    if _critique_needs_revision(state.get('critique', {})):
        return "revise"
    return "end"


def should_revise_exam(state: ExamPaperState) -> str:
    if state.get('reflection_rounds', 0) >= 2:
        return "end"
    if _critique_needs_revision(state.get('critique', {})):
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
) -> str:
    """
    运行 Reflexion 出题系统（3 Agent 协作）
    流程：出题+推理链 → Critic质疑推理链 → Revise针对批评修订（最多2轮）
    """
    context = await get_rag_context(topic)

    initial_state: QuizState = {
        "topic": topic,
        "quiz_type": quiz_type,
        "num": num,
        "context": context,
        "sample_paper_context": sample_paper_context or "",
        "quiz": "",
        "reasoning": "{}",
        "critique": {},
        "initial_score": 0,
        "revised_quiz": "",
        "revision_notes": "",
        "reflection_rounds": 0,
    }

    result = await quiz_workflow.ainvoke(initial_state)

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

    return result.get('revised_quiz') or result.get('quiz') or ''


async def run_exam_agent(
    topics: List[str],
    quiz_types: List[str],
    total_questions: int = 10,
    sample_paper_context: str = None
) -> str:
    """
    运行 Reflexion 出卷系统（3 Agent 协作）
    流程：出卷+推理链 → Critic质疑推理链 → Revise针对批评修订（最多2轮）
    """
    # 参数校验和修正
    if not quiz_types or len(quiz_types) == 0:
        quiz_types = ['选择题', '填空题', '判断题', '简答题']
    if total_questions <= 0 or total_questions > 50:
        total_questions = 10  # 限制题目数量范围

    logger.info(f"[Exam Agent] 校验后参数: quiz_types={quiz_types}, total_questions={total_questions}")

    # 如果没有样卷，联网搜索试卷格式参考
    online_format_context = ""
    online_knowledge_context = ""
    if not sample_paper_context or sample_paper_context.strip() == "":
        logger.info("[Exam Agent] 无样卷，联网搜索试卷格式和考点参考...")
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            search = DuckDuckGoSearchRun()
            # 搜索相关学科的典型试卷格式
            search_query = f"{topics[0] if topics else '大学课程'} 期末考试试卷 题型分布 结构"
            online_result = search.run(search_query)
            if online_result and len(online_result) > 50:
                online_format_context = f"\n\n【联网搜索的试卷格式参考】（无本地样卷时使用）：\n{online_result[:1500]}..."
                logger.info("[Exam Agent] 联网搜索获取格式参考成功")

            # 额外搜索常见考点
            if topics:
                topic_query = f"{topics[0]} 期末考试 常见考点 重点"
                topic_result = search.run(topic_query)
                if topic_result and len(topic_result) > 50:
                    online_knowledge_context = f"\n\n【联网搜索的考点参考】：\n{topic_result[:800]}..."
        except Exception as e:
            logger.warning(f"[Exam Agent] 联网搜索失败: {e}")
            online_format_context = "\n\n【提示】无样卷参考，请使用通用期末试卷格式：选择题、填空题、判断题、简答题四种题型均衡分布。"
    else:
        logger.info("[Exam Agent] 使用本地样卷格式")

    contexts = []
    for topic in topics:
        ctx = await get_rag_context(topic)
        # 如果有联网搜索的考点，合并到课件资料中
        if online_knowledge_context:
            ctx = ctx + online_knowledge_context
        contexts.append(ctx)

    # 合并样卷格式和联网搜索结果
    final_sample_context = (sample_paper_context or "") + online_format_context

    initial_state: ExamPaperState = {
        "topics": topics,
        "quiz_types": quiz_types,
        "total_questions": total_questions,
        "sample_paper_context": final_sample_context,
        "contexts": contexts,
        "exam_paper": "",
        "reasoning": "{}",
        "critique": {},
        "initial_score": 0,
        "revised_exam": "",
        "revision_notes": "",
        "reflection_rounds": 0,
    }

    result = await exam_workflow.ainvoke(initial_state)

    rounds = result.get('reflection_rounds', 0)
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

    # 提取试卷内容（去掉答案和解析，只保留题目）
    exam_content = result.get('revised_exam') or result.get('exam_paper') or ''
    exam_content = _strip_answers_and_analysis(exam_content)

    return exam_content


def _strip_answers_and_analysis(exam_text: str) -> str:
    """
    去掉试卷中的答案和解析，只保留题目部分
    """
    import re
    lines = exam_text.split('\n')
    cleaned_lines = []
    skip_mode = False

    for line in lines:
        stripped = line.strip()

        # 跳过空白行但保持结构
        if not stripped:
            cleaned_lines.append(line)
            continue

        # 遇到答案行或解析行，跳过后续内容直到遇到新题目
        # 匹配 "答案：" 或 "答案:" 或 "【答案】" 等格式
        if re.match(r'^(\【|\[)?答案(\】|\])?[:：]', stripped):
            skip_mode = True
            continue
        # 匹配 "解析：" 或 "解析:" 或 "【解析】" 等格式
        if re.match(r'^(\【|\[)?解析(\】|\])?[:：]', stripped):
            skip_mode = True
            continue
        # 匹配 "参考答案" 等
        if re.match(r'^(\【|\[)?参考答案', stripped):
            skip_mode = True
            continue

        # 新题型标题，恢复正常模式
        if re.match(r'^(#{1,3}\s*)?[一二三四五六七八九十]+[、.]\s*[\u4e00-\u9fa5]', stripped):
            skip_mode = False
        # 题目编号行（如 "1."、"2."），恢复显示
        if re.match(r'^\d+[.、]\s', stripped):
            skip_mode = False

        if not skip_mode:
            cleaned_lines.append(line)

    # 移除末尾多余的空行（保留最多2个换行）
    while len(cleaned_lines) > 2 and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    return '\n'.join(cleaned_lines)
