"""
多 Agent 出题系统 - Reflexion 架构
Agent 1: 出题 Agent  — 生成题目 + 推理链（为什么这样出题、答案依据何处）
Agent 2: Critic Agent — 质疑推理链，找出逻辑漏洞，输出结构化批评
Agent 3: Revise Agent — 逐条回应批评，修订题目并给出修改说明
循环：critique → revise → critique，最多 2 轮
"""
import json
from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field

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
    quantity_dist: dict  # 题型数量分布，如 {"choice": 10, "fill": 5, "judge": 5, "essay": 3}
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


class ExamPaper(BaseModel):
    """试卷数据结构"""
    questions: List[Question] = Field(description="题目列表")
    reasoning: dict = Field(description="出题推理链")


class ExamPaperText(BaseModel):
    """试卷文本格式（用于完整试卷）"""
    exam_paper: str = Field(description="完整试卷文本")
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


async def call_llm_structured(prompt_template: str, schema: type[BaseModel], **kwargs) -> BaseModel:
    """
    调用 LLM 并强制结构化输出
    注意：MiniMax API 不支持 response_format，这里使用 Prompt 约束 + 正则解析
    """
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | chat_model | StrOutputParser()
    result = await chain.ainvoke(kwargs)
    # 使用正则提取 JSON 并转换为 Pydantic 模型
    try:
        parsed = _extract_json(result)
        return schema(**parsed)
    except Exception as e:
        logger.warning(f"[call_llm_structured] 解析失败: {e}")
        raise


# ---------- 分批生成单题型试卷的 Prompt ----------
EXAM_GENERATE_SINGLE_TYPE_PROMPT = """【警告】绝对禁止使用'...'、'等'、'以下略'等任何占位符！你必须逐字逐句地生成所有要求的题目。

你是一位资深大学期末考试命题专家。根据课件资料生成单种题型的试卷。

【考点范围】{topics}
【题型】{quiz_type}
【题数】{num} 道
【起始编号】{start_num}
【每题分值】{score} 分
【总分】{total_score} 分
【样卷格式参考】{sample_paper_context}

【课件资料】:
{context}

【硬性要求】：
1. 必须生成正好 {num} 道 {quiz_type}，编号从 {start_num} 到 {end_num}，一道都不能少！
2. 选择题必须有 A、B、C、D 四个完整选项，题干描述要详细（至少30字）
3. 填空题题干要详细，需要填空的部位用括号表示
4. 判断题题干要详细
5. 简答题/解答题必须有2-3个小问
6. 【重要】所有题目统一分值：{score} 分/题，总分 {total_score} 分，每道题必须标注这个分值！
7. 题目描述要详细，包含足够信息
8. 难度分布：简单30%、中等50%、困难20%
9. 【强制出题】必须逐字逐句生成所有要求的题目，禁止任何省略！
10. 每道题必须包含答案、解析和分值标注

【格式示例】
{format_example}

请生成 {num} 道 {quiz_type}。

【重要】只返回纯 JSON 对象，不要用 ```json 代码块包裹！直接输出：
{{"exam_paper": "...", "reasoning": {{"knowledge_points": [...], "answer_evidence": [...]}}}}"""


# ---------- 结构化试卷生成 Prompt（返回 JSON） ----------
EXAM_GENERATE_STRUCTURED_PROMPT = """【警告】绝对禁止使用'...'、'等'、'以下略'等任何占位符！你必须逐字逐句地生成所有要求的题目。

你是一位资深大学期末考试命题专家。根据课件资料生成结构化试卷。

【考点范围】{topics}
【题型】{quiz_type}
【题数】{num} 道
【起始编号】{start_num}
【样卷格式参考】{sample_paper_context}

【课件资料】:
{context}

【硬性要求】：
1. 必须生成正好 {num} 道 {quiz_type}，编号从 {start_num} 开始
2. 选择题必须有 A、B、C、D 四个完整选项，题干描述要详细
3. 填空题题干要详细，需要填空的部位用括号表示
4. 判断题题干要详细
5. 简答题/解答题可以有2-3个小问
6. 【必须】每道题都要包含 score（分值）和 difficulty（难度）
7. 分值分配由你根据题型和难度决定：选择题2-3分，填空题2-3分，判断题2分，简答题8-10分
8. 难度分布：简单30%、中等50%、困难20%，根据题目内容自行判断
9. 【强制出题】必须逐字逐句生成所有要求的题目，禁止任何省略！
10. 每道题必须包含 answer（答案）和 analysis（解析）

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
{start_num}. [详细题干内容，至少30字]
   A. 选项1  B. 选项2  C. 选项3  D. 选项4
答案：A
分值：{score}
难度：中等
解析：根据课件内容...

{start_num_plus1}. [详细题干内容]
   A. 选项1  B. 选项2  C. 选项3  D. 选项4
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
        result = await call_llm_structured(
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
        )
        logger.info(f"[分批生成] {quiz_type} 生成成功，长度={len(result.exam_paper)}")
        return result.exam_paper
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
    logger.info(f"[结构化生成] {quiz_type}: {num}题, 起始编号{start_num}")

    try:
        result = await call_llm_structured(
            EXAM_GENERATE_STRUCTURED_PROMPT,
            ExamPaper,
            quiz_type=quiz_type,
            num=num,
            start_num=start_num,
            topics=topics,
            sample_paper_context=sample_paper_context,
            context=context,
        )
        logger.info(f"[结构化生成] {quiz_type} 成功，生成 {len(result.questions)} 道题")
        return result
    except Exception as e:
        logger.error(f"[结构化生成] {quiz_type} 失败: {e}")
        return ExamPaper(questions=[], reasoning={})


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON，自动处理 markdown 代码块"""
    import re
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


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
GENERATE_WITH_REASONING_PROMPT = """【警告-绝对禁止】绝对禁止使用'...'、'等'、'以下略'等任何占位符！你必须逐字逐句地生成所有要求的题目。任何省略行为将被判定为任务失败！

你是一位资深大学期末考试命题专家。根据课件资料生成题目，同时给出详细推理链。

【考点】{topic}
【题型】{quiz_type}
【题目数量】{num}
【样卷格式参考】{sample_paper_context}

【课件资料】（所有题目必须严格基于此）:
{context}

请生成题目，并为每道题提供推理链（说明为什么这样出题、答案依据在课件哪里）。

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

【重要】只返回纯 JSON 对象，不要用 ```json 代码块包裹！直接输出：
{{"approved": false, "overall_score": 75, "critique": "...", "reasoning_flaws": [{{"question": "...", "flaw": "...", "severity": "high"}}], "specific_issues": [{{"question": "...", "issue": "...", "suggestion": "..."}}]}}}"""


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

请根据批评修订题目，并详细说明你的修改依据。

【重要】只返回纯 JSON 对象，不要用 ```json 代码块包裹！直接输出：
{{"revised_quiz": "...", "revision_notes": "...", "addressed_issues": [...]}}"""


# ---------- 试卷版本 ----------
EXAM_GENERATE_WITH_REASONING_PROMPT = """【警告-绝对禁止】绝对禁止使用'...'、'等'、'以下略'、'必须出满XX题'等任何占位符！你必须逐字逐句地生成所有要求的题目。任何省略行为将被判定为任务失败！

你是一位资深大学期末考试命题专家。根据课件资料生成完整试卷。

【重要】出题前必须先联网搜索该学科的典型期末试卷风格、常见考点和优秀题目作为参考。

【考点范围】{topics}
【题目类型】{quiz_types}
【总题数】{total_questions}
【总分】100分（必须正好100分，不能多也不能少）
【样卷格式参考】{sample_paper_context}

【课件资料】:
{contexts}

【硬性要求】（必须严格遵守，任何一条不遵守都判定为不合格）：
【必须严格遵守】总题数 {total_questions} 道！选择{choice_count}道、填空{fill_count}道、判断{judge_count}道、简答{essay_count}道！一道都不能少！

3. 【题型分布】：
   - 选择题：{choice_count}题 × 2分，编号1到{choice_end}
   - 填空题：{fill_count}题 × 2分，编号{fill_start}到{fill_end}
   - 判断题：{judge_count}题 × 2分，编号{judge_start}到{judge_end}
   - 简答题：{essay_count}题 × {essay_score}分，编号{essay_start}到{essay_end}
4. 选择题必须有 A、B、C、D 四个完整选项，题干描述要详细（至少30字）
5. 题目必须全部统一编号：1→2→3→...→{total_questions}，不能每个题型单独从1开始编号
6. 【灵活出题】答案可以参考课件内容，如果课件提到某知识点但内容不够详细，可以结合联网搜索的参考资料来出题。允许标注"参考答案"。
7. 简答题/解答题必须有2-3个小问，每问5分左右
8. 填空题和判断题题干也要详细描述
9. 题目描述要详细，包含足够信息让考生理解题意
10. 难度分布：简单题占30%，中等题占50%，难题占20%
11. 【严禁重复】同一知识点、同一题型、相似问法不得出现超过1次
12. 【强制出题】必须逐字逐句生成所有要求的题目，禁止使用"..."、"等"、"...(略)"等任何占位符！
13. 【必须出完】每一道题都要完整输出，包括题干、选项、答案、解析

请生成完整试卷。

【试卷格式示例】（按此格式输出）：
```
## 《XX课程期末考试试卷》

一、选择题（共10题，每题2分，计20分）
1. [题干内容...]
   A. 选项1  B. 选项2  C. 选项3  D. 选项4

二、填空题（共10题，每题2分，计20分）
11. [题干内容，需要填空的部位用括号表示]

三、判断题（共10题，每题2分，计20分）
21. [题干内容]

四、简答题（共4题，每题10分，计40分）
31. [问题描述]（10分）
    (1) [小问1]（5分）
    (2) [小问2]（5分）
```

【警告】如果不按要求出够题目数量，将被判定为不合格！

一、选择题（每题2分，共10题，编号1-10）
1. [详细题干内容，描述要充分，至少30字]
   A. 选项1  B. 选项2  C. 选项3  D. 选项4

2. [详细题干内容]
   A. 选项1  B. 选项2  C. 选项3  D. 选项4

...（必须出满10题，编号到10）

二、填空题（每题2分，共10题，编号11-20）
（注意：填空题从第11题开始编号！）
11. [详细题干内容，需要填空的部位用括号表示]

...（必须出满10题，编号到20）

三、判断题（每题2分，共10题，编号21-30）
（注意：判断题从第21题开始编号！）
21. [详细题干内容]

...（必须出满10题，编号到30）

四、简答题（每题10分，共4题，编号31-34）
（注意：简答题从第31题开始编号！）
31. [问题描述]（10分）
    (1) [小问1]（5分）
    (2) [小问2]（5分）

...（必须出满4题，编号到34）

12. ...

三、判断题（每题2分，共10题）
（注意：判断题从第21题开始编号！）
21. [详细题干内容]

22. ...

四、简答题/解答题（每题10分，共4题，每题2-3问）
（注意：简答题从第31题开始编号！）
31. [问题描述]（10分）
    (1) [小问1]（5分）
    (2) [小问2]（5分）

32. [问题描述]
    (1) ...
    (2) ...
```

【重要】只返回纯 JSON 对象，不要用 ```json 代码块包裹！直接输出：
{{"exam_paper": "...", "reasoning": {{"topic_coverage": [...], "answer_evidence": [...], "difficulty_distribution": "..."}}}}"""


EXAM_CRITIQUE_PROMPT = """你是试卷评审专家。审查出题质量，允许结合课件和联网参考资料出题。

【考点范围】{topics}
【参考资料】（课件+联网搜索，可结合使用）:
{contexts}

【待评审试卷】:
{exam_paper}

【出题推理链】:
{reasoning}

【重点审查项】你必须特别严格检查以下问题：
1. 【题型数量验证】选择题必须有 EXACTLY 10 道（编号1-10），填空题必须有 EXACTLY 10 道（编号11-20），判断题必须有 EXACTLY 10 道（编号21-30），简答题必须有 EXACTLY 4 道（编号31-34）。如果任何题型数量不够，判定为不合格！
2. 【总分验证】所有题目分值加起来是否正好100分？
3. 【题数验证】题目总数量是否正好{total_questions}道？
4. 【编号验证】题目是否统一连续编号（1→2→3→...），选择题1-10，填空题11-20，判断题21-30，简答题31-34？严禁每个题型单独从1开始编号！
5. 【重复检查】是否存在完全相同或高度相似的题目？

【重要】只返回纯 JSON 对象，不要用 ```json 代码块包裹！直接输出：
{{"approved": false, "overall_score": 75, "critique": "...", "reasoning_flaws": [{{"question": "...", "flaw": "...", "severity": "high"}}], "specific_issues": [{{"question": "...", "issue": "...", "suggestion": "..."}}], "duplicate_check": {{"has_duplicates": false, "duplicate_questions": []}}, "numbering_check": {{"is_continuous": true, "issues": []}}, "quantity_check": {{"choice": 10, "fill": 10, "judge": 10, "essay": 4, "actual_choice": 0, "actual_fill": 0, "actual_judge": 0, "actual_essay": 0, "is_valid": false}}}}"""


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

【重要】只返回纯 JSON 对象，不要用 ```json 代码块包裹！直接输出：
{{"revised_exam": "...", "revision_notes": "...", "addressed_issues": [...]}}"""


# ==================== 节点函数（单题）====================

async def generate_with_reasoning_node(state: QuizState) -> QuizState:
    """Agent 1: 出题 + 推理链"""
    logger.info(f"[Agent1-出题] {state['topic']} / {state['quiz_type']} / {state['num']}道")
    sample = state.get('sample_paper_context', '') or '无样卷参考，使用默认格式'

    # 使用结构化输出，彻底杜绝 Markdown 混排
    try:
        result = await call_llm_structured(
            GENERATE_WITH_REASONING_PROMPT,
            ExamPaperText,
            topic=state['topic'],
            quiz_type=state['quiz_type'],
            num=state['num'],
            context=state['context'],
            sample_paper_context=sample,
        )
        quiz = result.exam_paper
        reasoning = json.dumps(result.reasoning, ensure_ascii=False)
        logger.info(f"[Agent1-出题] 结构化输出成功，quiz长度={len(quiz)}")
    except Exception as e:
        logger.warning(f"[Agent1-出题] 结构化输出失败，回退到正则解析: {e}")
        # 回退机制：尝试正则解析
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
        except Exception:
            quiz = ""
            reasoning = '{}'

    return {"quiz": quiz, "reasoning": reasoning}


async def critique_quiz_node(state: QuizState) -> QuizState:
    """Agent 2: Critic — 质疑推理链，输出结构化批评"""
    quiz = state.get('revised_quiz') or state.get('quiz', '')
    logger.info("[Agent2-Critic] 质疑出题推理链...")

    # 使用结构化输出
    try:
        result = await call_llm_structured(
            CRITIQUE_PROMPT,
            CritiqueResult,
            topic=state['topic'],
            quiz_type=state['quiz_type'],
            context=state['context'],
            quiz=quiz,
            reasoning=state.get('reasoning', '{}'),
        )
        critique = result.model_dump()
        logger.info(f"[Agent2-Critic] 结构化输出成功，approved={result.approved}")
    except Exception as e:
        logger.warning(f"[Agent2-Critic] 结构化输出失败，回退到正则解析: {e}")
        # 回退机制
        try:
            result_text = await call_llm(
                CRITIQUE_PROMPT,
                topic=state['topic'],
                quiz_type=state['quiz_type'],
                context=state['context'],
                quiz=quiz,
                reasoning=state.get('reasoning', '{}'),
            )
            critique = _extract_json(result_text)
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

    # 使用结构化输出
    try:
        result = await call_llm_structured(
            REVISE_WITH_REFLECTION_PROMPT,
            ReviseResult,
            topic=state['topic'],
            quiz_type=state['quiz_type'],
            context=state['context'],
            quiz=state.get('quiz', ''),
            reasoning=state.get('reasoning', '{}'),
            critique=json.dumps(state.get('critique', {}), ensure_ascii=False),
        )
        revised = result.revised_quiz
        notes = result.revision_notes
        logger.info(f"[Agent3-Revise] 结构化输出成功")
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
        "reflection_rounds": round_n,
    }


# ==================== 节点函数（试卷）====================

async def generate_exam_with_reasoning_node(state: ExamPaperState) -> ExamPaperState:
    """Agent 1: 出卷 + 推理链（分批生成每种题型）"""
    logger.info(f"[Agent1-出卷] {state['total_questions']}道，考点: {state['topics']}")
    topics_str = ", ".join(state['topics'])
    sample = state.get('sample_paper_context', '') or '无样卷参考'

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

    # 动态计算简答题分数，确保总分=100分
    base_score = choice_count * choice_score + fill_count * fill_score + judge_count * judge_score
    remaining_score = 100 - base_score
    if essay_count > 0:
        essay_score = remaining_score // essay_count
    else:
        essay_score = 0

    logger.info(f"[出卷] 题型数量: 选择{choice_count}×{choice_score}分, 填空{fill_count}×{fill_score}分, 判断{judge_count}×{judge_score}分, 简答{essay_count}×{essay_score}分, 总分={choice_count*choice_score + fill_count*fill_score + judge_count*judge_score + essay_count*essay_score}")

    # 计算编号范围
    current_num = 1
    choice_start = current_num
    current_num += choice_count
    fill_start = current_num
    current_num += fill_count
    judge_start = current_num
    current_num += judge_count
    essay_start = current_num

    # 并行生成所有题型（提高速度）
    import asyncio

    async def generate_and_collect(quiz_type: str, num: int, start_num: int, score: int):
        if num <= 0:
            return None
        paper = await generate_single_type_paper(
            quiz_type=quiz_type,
            num=num,
            start_num=start_num,
            topics=topics_str,
            context=state['contexts'][0] if state['contexts'] else '',
            sample_paper_context=sample,
            score_per_question=score
        )
        return {"paper": paper, "type": quiz_type, "count": num}

    # 并行执行所有题型生成
    tasks = []
    task_info = []
    if choice_count > 0:
        tasks.append(generate_and_collect("选择题", choice_count, choice_start, 2))
        task_info.append({"type": "选择题", "count": choice_count})
    if fill_count > 0:
        tasks.append(generate_and_collect("填空题", fill_count, fill_start, 2))
        task_info.append({"type": "填空题", "count": fill_count})
    if judge_count > 0:
        tasks.append(generate_and_collect("判断题", judge_count, judge_start, 2))
        task_info.append({"type": "判断题", "count": judge_count})
    if essay_count > 0:
        tasks.append(generate_and_collect("简答题", essay_count, essay_start, essay_score))
        task_info.append({"type": "简答题", "count": essay_count, "score": essay_score})

    # 并行执行
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 收集结果
    all_papers = []
    all_reasonings = task_info
    for result in results:
        if isinstance(result, dict) and result.get("paper"):
            all_papers.append(result["paper"])

    # 合并所有试卷
    exam_paper = "\n\n".join(all_papers)
    reasoning = json.dumps({"type_distribution": all_reasonings}, ensure_ascii=False)

    logger.info(f"[Agent1-出卷] 分批生成完成，总长度={len(exam_paper)}, 题型数={len(all_papers)}")

    return {"exam_paper": exam_paper, "reasoning": reasoning}


async def critique_exam_node(state: ExamPaperState) -> ExamPaperState:
    """Agent 2: Critic — 评审试卷推理链"""
    exam_paper = state.get('revised_exam') or state.get('exam_paper', '')
    contexts_combined = "\n\n".join(state['contexts'])[:2000]
    logger.info("[Agent2-Critic-试卷] 质疑出卷推理链...")

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
        )
        critique = result.model_dump()
        logger.info(f"[Agent2-Critic-试卷] 结构化输出成功，approved={result.approved}")
    except Exception as e:
        logger.warning(f"[Agent2-Critic-试卷] 结构化输出失败，回退到正则解析: {e}")
        # 回退机制
        try:
            result_text = await call_llm(
                EXAM_CRITIQUE_PROMPT,
                topics=", ".join(state['topics']),
                contexts=contexts_combined,
                exam_paper=exam_paper,
                reasoning=state.get('reasoning', '{}'),
                total_questions=state.get('total_questions', 10),
            )
            critique = _extract_json(result_text)
        except Exception:
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

    update: dict = {"critique": critique}
    if state.get('reflection_rounds', 0) == 0 and not state.get('initial_score'):
        update["initial_score"] = score
    return update


async def revise_exam_with_reflection_node(state: ExamPaperState) -> ExamPaperState:
    """Agent 3: Revise — 针对批评修订试卷"""
    round_n = state.get('reflection_rounds', 0) + 1
    logger.info(f"[Agent3-Revise-试卷] 第 {round_n} 轮反思修订...")
    contexts_combined = "\n\n".join(state['contexts'])[:2000]

    # 使用结构化输出
    try:
        result = await call_llm_structured(
            EXAM_REVISE_WITH_REFLECTION_PROMPT,
            ExamReviseResult,
            topics=", ".join(state['topics']),
            contexts=contexts_combined,
            exam_paper=state.get('exam_paper', ''),
            reasoning=state.get('reasoning', '{}'),
            critique=json.dumps(state.get('critique', {}), ensure_ascii=False),
            total_questions=state.get('total_questions', 10),
        )
        revised = result.revised_exam
        notes = result.revision_notes
        logger.info(f"[Agent3-Revise-试卷] 结构化输出成功")
    except Exception as e:
        logger.warning(f"[Agent3-Revise-试卷] 结构化输出失败，回退到正则解析: {e}")
        # 回退机制
        try:
            result_text = await call_llm(
                EXAM_REVISE_WITH_REFLECTION_PROMPT,
                topics=", ".join(state['topics']),
                contexts=contexts_combined,
                exam_paper=state.get('exam_paper', ''),
                reasoning=state.get('reasoning', '{}'),
                critique=json.dumps(state.get('critique', {}), ensure_ascii=False),
                total_questions=state.get('total_questions', 10),
            )
            parsed = _extract_json(result_text)
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
    total_questions: int = 34,
    sample_paper_context: str = None,
    quantity_dist: dict = None
) -> str:
    """
    运行 Reflexion 出卷系统（3 Agent 协作）
    流程：出卷+推理链 → Critic质疑推理链 → Revise针对批评修订（最多2轮）
    """
    # 参数校验和修正
    if not quiz_types or len(quiz_types) == 0:
        quiz_types = ['选择题', '填空题', '判断题', '简答题']

    # 默认题型分布：选择题10，判断5，填空5，问答3（共23道）
    default_dist = {"choice": 10, "fill": 5, "judge": 5, "essay": 3}

    # 如果有 quantity_dist（用户自定义题型数量），优先使用
    if quantity_dist:
        total_questions = sum(quantity_dist.values())
    elif len(quiz_types) >= 4:
        # 综合卷默认23道（选择10，判断5，填空5，问答3）
        quantity_dist = default_dist
        total_questions = 23
    elif total_questions <= 0 or total_questions > 50:
        total_questions = 10  # 限制题目数量范围

    logger.info(f"[Exam Agent] 校验后参数: quiz_types={quiz_types}, total_questions={total_questions}, quantity_dist={quantity_dist}")

    # 如果没有样卷，联网搜索试卷格式参考
    online_format_context = ""
    online_knowledge_context = ""
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
        else:
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

    # 确保 quantity_dist 有值（综合卷默认23道）
    if not quantity_dist and len(quiz_types) >= 4:
        quantity_dist = {"choice": 10, "fill": 5, "judge": 5, "essay": 3}

    initial_state: ExamPaperState = {
        "topics": topics,
        "quiz_types": quiz_types,
        "total_questions": total_questions,
        "quantity_dist": quantity_dist or {},
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

    # 提取试卷内容（保留答案和解析，方便前端显示）
    exam_content = result.get('revised_exam') or result.get('exam_paper') or ''
    # 注意：不再去掉答案和解析，让前端可以显示

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
