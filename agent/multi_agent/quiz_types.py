from typing import Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class QuizState(TypedDict):
    topic: str
    quiz_type: str
    num: int
    context: str
    sample_paper_context: str
    quiz: str
    reasoning: str
    quiz_payload: dict
    critique: dict
    initial_score: int
    revised_quiz: str
    revision_notes: str
    revised_quiz_payload: dict
    delivery_mode: str
    quality_floor_passed: bool
    degrade_reason: str
    quiz_budget_started_at: float
    quiz_budget_seconds: int
    evidence_source: str
    reflection_rounds: int
    force_llm_critic: bool


class ExamPaperState(TypedDict):
    topics: List[str]
    quiz_types: List[str]
    total_questions: int
    quantity_dist: dict
    sample_paper_context: str
    contexts: List[str]
    exam_paper: str
    reasoning: str
    exam_payload: dict
    critique: dict
    initial_score: int
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
    topic_bank: List[str]
    topic_usage: Dict[str, int]
    reflection_rounds: int


class Question(BaseModel):
    id: int = Field(description="题目编号")
    type: str = Field(description="题型：选择题/填空题/判断题/简答题/计算题/分析题/论述题/名词解释")
    content: str = Field(description="题干内容")
    options: Optional[List[str]] = Field(default=None, description="选项列表（选择题用）")
    answer: str = Field(description="正确答案")
    analysis: str = Field(description="题目解析")
    score: int = Field(description="分值")
    difficulty: str = Field(description="难度：简单/中等/较难")


class QuizQuestionItem(BaseModel):
    id: int = Field(description="题目编号")
    type: str = Field(description="题型")
    question: str = Field(description="题干内容")
    options: Optional[List[str]] = Field(default=None, description="选项")
    answer: str = Field(description="答案")
    explanation: str = Field(description="解析")
    score: Optional[str] = Field(default=None, description="分值，如5分")
    difficulty: Optional[str] = Field(default=None, description="难度")


class StructuredQuizSetResult(BaseModel):
    title: Optional[str] = Field(default="练习题", description="题组标题")
    questions: List[QuizQuestionItem] = Field(description="题目列表")
    reasoning: Optional[dict] = Field(default_factory=dict, description="出题推理链")


class ExamPaper(BaseModel):
    questions: List[Question] = Field(description="题目列表")
    reasoning: dict = Field(description="出题推理链")


class ExamPaperText(BaseModel):
    exam_paper: str = Field(description="完整试卷文本")
    reasoning: Optional[dict] = Field(default_factory=dict, description="出题推理链")


class QuizGenerateResult(BaseModel):
    quiz: str = Field(description="完整题目文本")
    reasoning: Optional[dict] = Field(default_factory=dict, description="出题推理链")


class CritiqueResult(BaseModel):
    approved: bool = Field(description="是否通过评审")
    overall_score: int = Field(description="总体评分 0-100")
    critique: str = Field(description="总体评价")
    reasoning_flaws: List[dict] = Field(default_factory=list, description="推理链缺陷列表")
    specific_issues: List[dict] = Field(default_factory=list, description="具体问题列表")
    duplicate_check: dict = Field(default_factory=dict, description="重复题目检查")
    numbering_check: dict = Field(default_factory=dict, description="编号连续性检查")
    quantity_check: dict = Field(default_factory=dict, description="题型数量检查")


class ReviseResult(BaseModel):
    revised_quiz: str = Field(description="修订后的题目")
    revision_notes: str = Field(description="修改说明")
    addressed_issues: List[str] = Field(default_factory=list, description="已解决的问题列表")


class ExamReviseResult(BaseModel):
    revised_exam: str = Field(description="修订后的试卷")
    revision_notes: str = Field(description="修改说明")
    addressed_issues: List[str] = Field(default_factory=list, description="已解决的问题列表")
