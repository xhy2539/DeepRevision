"""
多 Agent 出题系统 - 4 Agent 协作
Agent 1: 出题 Agent - 生成题目和答案
Agent 2: 验证 Agent - 验证答案正确性和题目合理性
Agent 3: 审核 Agent - 验证题目可行性和格式规范
Agent 4: 监督 Agent - 确保知识点与课件一致，监督整个流程
"""
import json
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from model.factory import chat_model
from rag.rag_service import RagSummarizeService
from utils.session_context import current_session_id
from utils.logger_handler import logger


# ==================== 状态定义 ====================
class QuizState(TypedDict):
    """出题系统状态"""
    topic: str                      # 考点
    quiz_type: str                 # 题目类型
    num: int                       # 题目数量
    context: str                   # RAG 检索的资料
    quiz: str                     # 生成的题目
    verify_result: dict            # 验证结果
    review_result: dict            # 审核结果
    supervise_result: dict         # 监督结果
    fixed_quiz: str               # 修复后的题目
    issues: List[str]             # 发现的问题
    final_quiz: str               # 最终题目
    max_retries: int              # 最大重试次数
    sample_paper_context: str      # 样卷格式（可选）


class ExamPaperState(TypedDict):
    """出卷系统状态"""
    topics: List[str]              # 考点列表
    quiz_types: List[str]          # 题目类型列表
    total_questions: int           # 总题数
    sample_paper_context: str      # 样卷内容
    contexts: List[str]           # 各考点资料
    exam_paper: str               # 生成的试卷
    verify_result: dict            # 验证结果
    review_result: dict           # 审核结果
    supervise_result: dict         # 监督结果
    issues: List[str]             # 发现的问题
    final_paper: str             # 最终试卷
    max_retries: int             # 最大重试次数


# ==================== 工具函数 ====================
async def get_rag_context(topic: str) -> str:
    """获取 RAG 资料"""
    sid = current_session_id.get() or "default"
    rag = RagSummarizeService()
    return await rag.rag_summarize(topic)


async def call_llm(prompt_template: str, **kwargs) -> str:
    """调用 LLM"""
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | chat_model | StrOutputParser()
    result = await chain.ainvoke(kwargs)
    return result


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON，自动处理 markdown 代码块"""
    import re
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


# ==================== 4 Agent 提示词 (优化版) ====================

# ---------- Agent 1: 出题 Agent (CoT + Few-Shot 增强) ----------
GENERATE_PROMPT = """<context>
你是一位资深的大学期末考试命题专家，拥有10年以上出题经验。你擅长根据教材和课件设计高质量的考试题目，注重知识点的全面覆盖和难度梯度设计。
</context>

<task>
根据以下课件资料，为大学生生成高质量的期末复习试题。
</task>

<thinking>
1. 首先通读课件资料，识别核心知识点和重要概念
2. 根据题型特点设计题目：
   - 选择题：测试对概念的理解和辨析能力
   - 填空题：测试对关键定义和数值的记忆
   - 判断题：测试对易混淆概念的辨别
   - 简答题：测试对知识点的综合理解
3. 确保答案100%来源于课件，禁止编造
4. 解析要包含考点分析和答题思路
</thinking>

【考点】{topic}
【题型】{quiz_type}
【题目数量】{num}

【课件资料】（必须严格以此为依据，禁止超出范围）:
{context}

【样卷格式参考】（如有，请严格遵循）:
{sample_paper_context}

【重要提示】
- 如果用户没有明确要求"含答案"、"要答案"、"附答案"，则只输出题目，不输出答案和解析
- 如果用户要求查看答案，再输出"【答案】"部分

【题型格式模板】（必须按此格式输出）

【注意】以下格式中"答案"和"解析"部分，只有在用户明确要求时才输出！

## 选择题
1. [题干：清晰描述考察点]
A. [选项A]
B. [选项B]
C. [选项C]
D. [选项D]
答案：[A/B/C/D]  ← 仅用户要求答案时输出
解析：[详细解析]  ← 仅用户要求答案时输出

## 填空题
1. [题干：在关键位置使用_____表示填空]
答案：[答案，多个空用|分隔]  ← 仅用户要求答案时输出

## 判断题
1. [题干：描述一个判断命题]
答案：正确/错误  ← 仅用户要求答案时输出

## 简答题
1. [问题：综合性问题]
参考答案要点：  ← 仅用户要求答案时输出
- 要点1
- 要点2
- 要点3

## 名词解释
1. [名词]
答案：[简明定义，50字以内]  ← 仅用户要求答案时输出

## 计算题
1. [题目条件]
解：
步骤1：[计算过程]
步骤2：[计算过程]
最终答案：[结果]  ← 仅用户要求答案时输出

【出题原则 - 务必遵守】
1. ✓ 核心知识点必须来自课件（60%以上）
2. ✓ 可以适度延申（40%以内），用常见大学课程知识补充
3. ✗ 禁止凭空编造课件完全没有的知识点
4. ✓ 答案必须能在课件中找到原文依据
5. ✗ 答案不能是"以上都对/都不对"
6. ✓ 课件内容不足时，只出能出的题，不要编造

【难度分级】
- 简单（30%）：基础概念记忆
- 中等（50%）：理解应用
- 困难（20%）：综合分析

请直接输出题目内容。如果用户没有要求答案，就只输出题目。"""


# ---------- Agent 2: 验证 Agent ----------
VERIFY_PROMPT = """<context>
你是一位试题审核员。检查试题是否合理。
</context>

【考点】{topic}
【课件资料】: {context}

【待验证试题】: {quiz}

请逐项检查：
1. 答案是否来自课件？
2. 知识点是否在课件范围内？（可以适度延申但不能偏离太远）
3. 题目是否有明显错误？
4. 格式是否符合模板要求？
5. 难度是否适合期末考试水平？

【错误类型定义】
- correctness: 答案错误或与课件不符
- clarity: 题目表述有歧义
- format: 格式不符合要求
- difficulty: 难度不合理

请按以下JSON格式返回验证结果（只返回JSON，不要其他内容）：
```json
{{
    "valid": true,
    "score": 85,
    "issues": [
        {{"type": "correctness|clarity|format|difficulty", "location": "第X题", "desc": "问题描述", "fix": "修复建议"}}
    ],
    "summary": "总体评价",
    "fix_needed": false
}}
```"""


# ---------- Agent 3: 审核 Agent (结构化增强) ----------
REVIEW_PROMPT = """<context>
你是一位资深的考试命题审核专家，负责评估试题是否能用于正式考试。你需要从可行性、区分度、知识点覆盖等多个维度进行审核。
</context>

<task>
对以下试题进行可行性审核，评估是否能用于正式期末考试。
</task>

<thinking>
1. 可行性：题目是否适合作为考试题
2. 区分度：是否能有效区分学生水平
3. 唯一性：答案是否确定唯一
4. 覆盖：知识点是否全面
</thinking>

【考点】{topic}
【题型】{quiz_type}

【待审核试题】:
{quiz}

【审核标准】（逐项检查）

## 1. 可行性评估
- [ ] 题目是否适合作为正式考试题？
- [ ] 区分度是否合理？（不能太简单或太难）
- [ ] 是否有足够的区分度？

## 2. 答案唯一性
- [ ] 答案是否确定唯一？
- [ ] 是否存在多个合理答案的可能性？
- [ ] 选择题是否避免了"以上都对"等投机选项？

## 3. 分数配重
- [ ] 分值分配是否合理？
- [ ] 难题和简单题分值是否匹配？

## 4. 知识点覆盖
- [ ] 知识点是否全面？
- [ ] 是否有遗漏的重要考点？
- [ ] 是否有重复考察同一知识点？

【错误类型定义】
- feasibility: 可行性问题
- uniqueness: 答案不唯一
- weight: 分值不合理
- coverage: 知识点覆盖不全或重复

请按以下JSON格式返回审核结果：
```json
{
    "feasible": true/false,
    "score": 80,
    "issues": [
        {"type": "feasibility|uniqueness|weight|coverage", "location": "第X题", "desc": "问题描述", "fix": "修改建议"}
    ],
    "suggestions": ["建议1", "建议2"],
    "pass": true/false
}
```"""


# ---------- Agent 4: 监督 Agent (Constitutional AI 增强) ----------
SUPERVISE_PROMPT = """<context>
你是本系统的最终监督员，负责确保整个出题流程正确进行。你需要综合验证结果和审核结果，进行最终把关。
</context>

<task>
对以下信息进行最终审核，确保题目知识点与课件内容100%一致，确保最终输出的题目质量合格。
</task>

<thinking>
1. 知识点一致性：所有知识点必须来自课件
2. 答案一致性：答案必须与课件内容完全一致
3. 流程完整性：综合前序所有验证结果
4. 质量判断：给出最终判定
</thinking>

【考点】{topic}
【题型】{quiz_type}

【课件资料】（以此为准，任何不一致都是严重问题）:
{context}

【待审核试题】:
{quiz}

【验证结果】(Agent 2):
{verify_result}

【审核结果】(Agent 3):
{review_result}

【监督检查项】（必须逐项确认）

## 1. 知识点一致性检查
- [ ] 题目涉及的知识点是否100%来自课件？
- [ ] 是否有任何超出课件范围的知识点？
- [ ] 是否有凭空编造的概念？

## 2. 答案一致性检查
- [ ] 答案中的知识点是否与课件完全一致？
- [ ] 是否存在与课件相悖的内容？
- [ ] 解析是否准确？

## 3. 流程完整性检查
- [ ] 前序验证是否通过？
- [ ] 前序审核是否通过？
- [ ] 是否存在遗漏的重要问题？

## 4. 格式规范性检查
- [ ] 格式是否统一规范？
- [ ] 是否符合出题模板要求？

【质量等级】
- A级：可直接使用
- B级： minor issues，可接受
- C级：需要修改
- D级：不合格，需要重新生成

请按以下JSON格式返回监督结果：
```json
{
    "consistent": true/false,
    "issues": [
        {"type": "knowledge|answer|process|format", "location": "第X题", "desc": "问题描述", "fix": "修改建议"}
    ],
    "warnings": [],
    "final_verdict": "通过/需要修改",
    "quality_level": "A/B/C/D",
    "action_required": true/false
}
```"""


# ==================== Agent 节点函数 ====================

async def generate_quiz_node(state: QuizState) -> QuizState:
    """Agent 1: 出题 Agent"""
    logger.info(f"[Agent 1-出题] 为 {state['topic']} 生成 {state['num']} 道 {state['quiz_type']}")

    sample_context = state.get('sample_paper_context', '') or '无样卷参考，使用默认格式'

    quiz = await call_llm(
        GENERATE_PROMPT,
        topic=state['topic'],
        quiz_type=state['quiz_type'],
        num=state['num'],
        context=state['context'],
        sample_paper_context=sample_context
    )

    return {"quiz": quiz}


async def verify_quiz_node(state: QuizState) -> QuizState:
    """Agent 2: 验证 Agent"""
    logger.info("[Agent 2-验证] 验证答案正确性...")

    result = await call_llm(
        VERIFY_PROMPT,
        topic=state['topic'],
        quiz_type=state['quiz_type'],
        context=state['context'],
        quiz=state['quiz']
    )

    try:
        verify_result = _extract_json(result)
    except Exception:
        verify_result = {"valid": True, "score": 100, "issues": [], "fix_needed": False}

    return {
        "verify_result": verify_result,
        "issues": [i.get("desc", "") for i in verify_result.get("issues", [])]
    }


async def review_quiz_node(state: QuizState) -> QuizState:
    """Agent 3: 审核 Agent"""
    logger.info("[Agent 3-审核] 审核题目可行性...")

    result = await call_llm(
        REVIEW_PROMPT,
        topic=state['topic'],
        quiz_type=state['quiz_type'],
        quiz=state['quiz']
    )

    try:
        review_result = _extract_json(result)
    except Exception:
        review_result = {"feasible": True, "score": 100, "issues": [], "pass": True}

    return {"review_result": review_result}


async def supervise_quiz_node(state: QuizState) -> QuizState:
    """Agent 4: 监督 Agent"""
    logger.info("[Agent 4-监督] 最终审核...")

    verify_str = json.dumps(state.get('verify_result', {}), ensure_ascii=False, indent=2)
    review_str = json.dumps(state.get('review_result', {}), ensure_ascii=False, indent=2)

    result = await call_llm(
        SUPERVISE_PROMPT,
        topic=state['topic'],
        quiz_type=state['quiz_type'],
        context=state['context'],
        quiz=state['quiz'],
        verify_result=verify_str,
        review_result=review_str
    )

    try:
        supervise_result = _extract_json(result)
    except Exception:
        supervise_result = {"consistent": True, "final_verdict": "通过", "action_required": False}

    return {"supervise_result": supervise_result}


async def fix_quiz_node(state: QuizState) -> QuizState:
    """修复 Agent: 根据问题修复题目"""
    logger.info(f"[修复Agent] 修复 {len(state['issues'])} 个问题")

    all_issues = []

    # 合并所有问题
    if state.get('verify_result', {}).get('issues'):
        all_issues.extend(state['verify_result']['issues'])
    if state.get('review_result', {}).get('issues'):
        all_issues.extend(state['review_result']['issues'])
    if state.get('supervise_result', {}).get('issues'):
        all_issues.extend(state['supervise_result']['issues'])

    issues_text = "\n".join([
        f"- {i.get('fix', i.get('desc', ''))}"
        for i in all_issues[:5]  # 最多5个问题
    ])

    fixed = await call_llm(
        """你是试题修改专家。请根据审核反馈修复以下试题。

考点：{topic}
题型：{quiz_type}

课件资料（必须严格以此为准）：
{context}

审核发现的问题：
{issues}

原试题：
{quiz}

【修复要求】
1. 严格按照审核意见修改
2. 保持其他正确的部分不变
3. 修复后确保答案100%正确
4. 保持格式规范

请输出修复后的完整试题，包含答案和解析。""",
        topic=state['topic'],
        quiz_type=state['quiz_type'],
        context=state['context'],
        issues=issues_text,
        quiz=state['quiz']
    )

    return {"fixed_quiz": fixed, "max_retries": state.get('max_retries', 0) + 1}

def should_fix_quiz(state: QuizState) -> str:
    """判断是否需要修复"""
    max_retries = state.get('max_retries', 0)

    # 减少重试次数，加快速度（只重试1次）
    if max_retries >= 1:
        logger.warning("[出题系统] 达到最大重试次数，直接返回")
        return "end"

    # 检查验证、审核、监督结果
    verify = state.get('verify_result', {})
    review = state.get('review_result', {})
    supervise = state.get('supervise_result', {})

    needs_fix = (
        verify.get('fix_needed', False) or
        review.get('pass', True) == False or
        supervise.get('action_required', False) == True
    )

    if needs_fix:
        return "fix"

    return "end"


# ==================== 构建工作流图 ====================

def build_quiz_graph() -> StateGraph:
    """构建出题工作流图（4 Agent：出题 → 验证 → 审核 → 监督 → 修复）"""

    graph = StateGraph(QuizState)

    # 添加节点
    graph.add_node("generate", generate_quiz_node)      # Agent 1: 出题
    graph.add_node("verify", verify_quiz_node)          # Agent 2: 验证
    graph.add_node("review", review_quiz_node)          # Agent 3: 审核
    graph.add_node("supervise", supervise_quiz_node)    # Agent 4: 监督
    graph.add_node("fix", fix_quiz_node)                # 修复

    # 设置入口
    graph.set_entry_point("generate")

    # 边：出题 → 验证 → 审核 → 监督
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", "review")
    graph.add_edge("review", "supervise")

    # 条件边：监督通过则结束，否则修复
    graph.add_conditional_edges(
        "supervise",
        should_fix_quiz,
        {
            "fix": "fix",
            "end": END
        }
    )

    # 修复后重新验证
    graph.add_edge("fix", "verify")

    return graph.compile()


def build_exam_graph() -> StateGraph:
    """构建出卷工作流图（4 Agent：出题 → 验证 → 审核 → 监督 → 修复）"""

    graph = StateGraph(ExamPaperState)

    # 添加节点
    graph.add_node("generate", generate_exam_node)      # Agent 1: 出题
    graph.add_node("verify", verify_exam_node)          # Agent 2: 验证
    graph.add_node("review", review_exam_node)          # Agent 3: 审核
    graph.add_node("supervise", supervise_exam_node)    # Agent 4: 监督
    graph.add_node("fix", fix_exam_node)                # 修复

    graph.set_entry_point("generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", "review")
    graph.add_edge("review", "supervise")

    # 条件边：监督通过则结束，否则修复
    graph.add_conditional_edges(
        "supervise",
        should_fix_exam,
        {
            "fix": "fix",
            "end": END
        }
    )

    graph.add_edge("fix", "verify")

    return graph.compile()


# ==================== 试卷生成相关节点 ====================

async def generate_exam_node(state: ExamPaperState) -> ExamPaperState:
    """生成试卷 Agent"""
    logger.info(f"[出卷Agent] 生成 {state['total_questions']} 道题的试卷")

    contexts_combined = "\n\n".join([
        f"【{topic}】\n{ctx}"
        for topic, ctx in zip(state['topics'], state['contexts'])
    ])

    sample = state.get('sample_paper_context', '') or '无样卷参考，使用默认格式'

    prompt = f"""<context>
你是一位资深的大学期末考试命题专家。根据给定的课件资料生成高质量试卷。
</context>

<task>
根据以下课件资料，生成一套完整的期末考试试卷。
</task>

<考点范围>
{', '.join(state['topics'])}
</考点范围>

<题目类型>
{', '.join(state['quiz_types'])}
</题目类型>

<总题数>
{state['total_questions']}
</总题数>

<样卷格式参考>
{sample}
</样卷格式参考>

<课件资料>
{contexts_combined}
</课件资料>

<thinking>
1. 识别课件中的核心知识点
2. 规划题目分布（简单30%+中等50%+困难20%）
3. 逐题编写，确保答案有课件原文依据
4. 检查格式规范性
</thinking>

<output_format>
## 选择题
1. [题干]
A. [选项] B. [选项] C. [选项] D. [选项]

## 填空题
1. [题干，用_____填空]

## 判断题
1. [题干]

## 简答题
1. [问题]

【答案】（用户要求时附加）
## 选择题答案
1. A
## 填空题答案
1. [答案]
</output_format>

<rules>
- 默认只输出题目，不输出答案
- 用户明确要求"含答案"才输出答案
- 60%知识点来自课件，40%可适度延申
- 禁止"以上都对/以上都错"类投机选项
- 禁止凭空编造知识点
</rules>

<example>
【错误示范】
1. 人工智能的核心技术包括？
A. 以上都是
B. 以上都不是
C. 机器学习
D. 深度学习
答案：A  ← 错误！使用了"以上都是"

【正确示范】
1. 人工智能的核心技术主要包括？
A. 机器学习
B. 自然语言处理
C. 计算机视觉
D. 增强现实
答案：ABC  ← 正确！
</example>

请严格按照格式输出试卷。"""

    exam_paper = await call_llm(prompt)
    return {"exam_paper": exam_paper}


async def verify_exam_node(state: ExamPaperState) -> ExamPaperState:
    """验证试卷 Agent"""
    logger.info("[验证Agent] 验证试卷...")

    contexts_combined = "\n\n".join(state['contexts'])[:2000]

    result = await call_llm(
        """<context>
你是严格的试题审核员，负责验证试卷答案是否正确。
</context>

<task>
逐题验证试卷答案是否在课件资料中有原文依据。
</task>

<课件资料>
{contexts}
</课件资料>

<试卷>
{exam_paper}
</试卷>

<verification_steps>
1. 提取试卷中每个答案
2. 在课件资料中搜索该答案的相关内容
3. 判断答案是否正确、是否有原文依据
4. 标记所有问题
</verification_steps>

<output_format>
```json
{
  "pass": true/false,
  "score": 85,
  "issues": [
    {"题号": 1, "问题": "答案错误", "课件依据": "相关原文"},
    {"题号": 3, "问题": "找不到依据", "课件依据": "N/A"}
  ],
  "fix_needed": true/false,
  "summary": "验证总结"
}
```
</output_format>""",
        topics=", ".join(state['topics']),
        quiz_types=", ".join(state['quiz_types']),
        contexts=contexts_combined,
        exam_paper=state['exam_paper']
    )

    try:
        verify_result = _extract_json(result)
    except Exception:
        verify_result = {"valid": True, "issues": [], "fix_needed": False}

    return {"verify_result": verify_result}


async def review_exam_node(state: ExamPaperState) -> ExamPaperState:
    """最终确认 Agent - 做最终质量把关"""
    logger.info("[最终确认] 质量审核...")

    result = await call_llm(
        """请快速审核试卷质量。

【考点】{topics}
【题型】{quiz_types}

【试卷】:
{exam_paper}

请检查：
1. 知识点是否覆盖主要考点？
2. 格式是否规范？
3. 能否直接使用？

返回JSON：
{{
    "pass": true/false,
    "score": 85,
    "issues": [],
    "action_required": false,
    "summary": "总结"
}}""",
        topics=", ".join(state['topics']),
        quiz_types=", ".join(state['quiz_types']),
        exam_paper=state['exam_paper']
    )

    try:
        review_result = _extract_json(result)
    except Exception:
        review_result = {"feasible": True, "pass": True}

    return {"review_result": review_result}


async def supervise_exam_node(state: ExamPaperState) -> ExamPaperState:
    """监督试卷 Agent"""
    logger.info("[监督Agent] 最终审核...")

    verify_str = json.dumps(state.get('verify_result', {}), ensure_ascii=False)
    review_str = json.dumps(state.get('review_result', {}), ensure_ascii=False)

    contexts_combined = "\n\n".join(state['contexts'])[:1500]

    result = await call_llm(
        """请对试卷进行最终监督审核。

考点：{topics}
题型：{quiz_types}

课件资料：
{contexts}

试卷内容：
{exam_paper}

验证结果：{verify_result}
审核结果：{review_result}

请确保：
1. 知识点与课件一致
2. 无与课件相悖的内容
3. 流程完整

返回JSON格式：
{{
    "consistent": true/false,
    "issues": [],
    "final_verdict": "通过/需要修改",
    "action_required": false
}}""",
        topics=", ".join(state['topics']),
        quiz_types=", ".join(state['quiz_types']),
        contexts=contexts_combined,
        exam_paper=state['exam_paper'],
        verify_result=verify_str,
        review_result=review_str
    )

    try:
        supervise_result = _extract_json(result)
    except Exception:
        supervise_result = {"consistent": True, "final_verdict": "通过", "action_required": False}

    return {"supervise_result": supervise_result}


async def fix_exam_node(state: ExamPaperState) -> ExamPaperState:
    """修复试卷 Agent"""
    logger.info("[修复Agent] 修复试卷...")

    all_issues = []
    if state.get('verify_result', {}).get('issues'):
        all_issues.extend(state['verify_result']['issues'])
    if state.get('review_result', {}).get('issues'):
        all_issues.extend(state['review_result']['issues'])
    if state.get('supervise_result', {}).get('issues'):
        all_issues.extend(state['supervise_result']['issues'])

    issues_text = "\n".join([f"- {i.get('fix', '')}" for i in all_issues[:5]])

    contexts_combined = "\n\n".join(state['contexts'])

    fixed = await call_llm(
        """请根据审核意见修复试卷。

考点：{topics}

课件资料：
{contexts}

审核问题：
{issues}

原试卷：
{exam_paper}

请输出修复后的完整试卷。""",
        topics=", ".join(state['topics']),
        contexts=contexts_combined,
        issues=issues_text,
        exam_paper=state['exam_paper']
    )

    return {"final_paper": fixed, "max_retries": state.get('max_retries', 0) + 1}


def should_fix_exam(state: ExamPaperState) -> str:
    """判断是否需要修复试卷"""
    # 只重试1次，加快速度
    if state.get('max_retries', 0) >= 1:
        return "end"

    verify = state.get('verify_result', {})
    review = state.get('review_result', {})

    # 验证或审核发现问题才需要修复
    needs_fix = (
        verify.get('fix_needed', False) or
        review.get('pass', True) == False or
        review.get('action_required', False) == True
    )

    if needs_fix:
        return "fix"

    return "end"


# ==================== 对外接口 ====================

quiz_workflow = build_quiz_graph()
exam_workflow = build_exam_graph()


async def run_quiz_agent(
    topic: str,
    quiz_type: str = "选择题",
    num: int = 3,
    sample_paper_context: str = None
) -> str:
    """
    运行多 Agent 出题系统（4 Agent 协作）
    流程: 出题 -> 验证 -> 审核 -> 监督 -> (可选)修复
    """
    # 1. 获取资料
    context = await get_rag_context(topic)

    # 2. 构建初始状态
    initial_state: QuizState = {
        "topic": topic,
        "quiz_type": quiz_type,
        "num": num,
        "context": context,
        "quiz": "",
        "verify_result": {},
        "review_result": {},
        "supervise_result": {},
        "fixed_quiz": "",
        "issues": [],
        "final_quiz": "",
        "max_retries": 0,
        "sample_paper_context": sample_paper_context or ""
    }

    # 3. 运行工作流
    result = await quiz_workflow.ainvoke(initial_state)

    # 4. 返回最终结果
    final_quiz = result.get('fixed_quiz') or result.get('quiz') or ''

    if not final_quiz:
        supervise = result.get('supervise_result', {})
        if supervise.get('consistent', True) == False:
            final_quiz = result.get('quiz', '') + "\n\n⚠️ 警告：题目未通过最终审核，建议人工检查"
        elif supervise.get('final_verdict') == '需要修改':
            final_quiz = result.get('quiz', '') + "\n\n⚠️ 题目需要修改，请查看审核意见"

    return final_quiz


async def run_exam_agent(
    topics: List[str],
    quiz_types: List[str],
    total_questions: int = 10,
    sample_paper_context: str = None
) -> str:
    """
    运行多 Agent 出卷系统（4 Agent 协作）
    """
    # 1. 获取各考点资料
    contexts = []
    for topic in topics:
        ctx = await get_rag_context(topic)
        contexts.append(ctx)

    # 2. 构建初始状态
    initial_state: ExamPaperState = {
        "topics": topics,
        "quiz_types": quiz_types,
        "total_questions": total_questions,
        "sample_paper_context": sample_paper_context or "",
        "contexts": contexts,
        "exam_paper": "",
        "verify_result": {},
        "review_result": {},
        "supervise_result": {},
        "issues": [],
        "final_paper": "",
        "max_retries": 0
    }

    # 3. 运行工作流
    result = await exam_workflow.ainvoke(initial_state)

    # 4. 返回最终结果
    final_paper = result.get('final_paper') or result.get('exam_paper') or ''

    if not final_paper:
        supervise = result.get('supervise_result', {})
        if supervise.get('consistent', True) == False:
            final_paper = result.get('exam_paper', '') + "\n\n⚠️ 试卷未通过最终审核，建议人工检查"

    return final_paper
