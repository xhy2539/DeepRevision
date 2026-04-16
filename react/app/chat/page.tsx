"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ExamCanvas, { isExamContent } from "@/components/ExamCanvas";
import ToastContainer, { showToast } from "@/components/ui/toast";

// Types
type MessageKind = "chat" | "quiz_set" | "exam_paper";

interface QuizQuestion {
  type: string;
  question: string;
  options?: string[];
  answer: string;
  explanation: string;
  score?: string;
  difficulty?: string;
}

interface StreamSectionAction {
  name?: string;
  status?: string;
  detail?: string;
  cost_ms?: number;
}

interface StreamSectionCitation {
  index?: number;
  source?: string;
  quote?: string;
}

interface StreamSections {
  reasoning: string;
  actions: StreamSectionAction[];
  citations: StreamSectionCitation[];
  progress: string[];
}

interface AssistantPayload {
  title?: string;
  show_answers_default?: boolean;
  show_analysis_default?: boolean;
  questions?: QuizQuestion[];
  exam_data?: unknown;
  messageId?: string;
  stream_sections?: StreamSections;
}

interface QuizExamData {
  title: string;
  subtitle: string;
  total_score: number;
  total_questions: number;
  question_types: Record<string, { count: number; total_score: number }>;
  questions: Array<{
    number: number;
    type: string;
    content: string;
    answer: string;
    analysis: string;
    score: number;
    difficulty: string;
  }>;
}

interface ExamPracticeQuestion {
  number: number;
  type: string;
  content: string;
  answer: string;
  analysis: string;
  score: number;
  difficulty: string;
  knowledge_point?: string;
}

interface PracticeRecordItem {
  question: ExamPracticeQuestion;
  userAnswer: string;
  isCorrect: boolean;
}

interface SimilarQuestion {
  question_content: string;
  answer: string;
  knowledge_point: string;
  question_id: string;
}

interface PracticeHistoryRecord {
  id: number;
  question_content: string;
  knowledge_point?: string;
  user_answer?: string;
  correct_answer?: string;
  is_correct: boolean;
  wrong_reason?: string;
  created_at: number;
}

interface KnowledgePointStat {
  total: number;
  correct: number;
  accuracy: number;
  weak: boolean;
}

interface PracticeStatsData {
  session_id: string;
  total_attempts: number;
  wrong_attempts: number;
  accuracy: number;
  retry_rate: number;
  weak_points: string[];
  knowledge_point_stats: Record<string, KnowledgePointStat>;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  kind?: MessageKind;
  render_mode?: "markdown" | "interactive_cards" | "exam_canvas";
  payload?: AssistantPayload;
  meta?: Record<string, unknown>;
}

interface ExamStageTraceItem {
  exam_stage: "choice" | "fill_judge" | "essay" | string;
  attempt?: number;
  stage_status?: "running" | "success" | "failed" | string;
  error?: string;
  strict_mode?: boolean;
  stage_latency_ms?: number;
}

interface Session {
  id: string;
  name: string;
  parent_id?: string;
}

interface StreamDiagnostics {
  ttft_ms: number | null;
  first_delta_ms: number | null;
  stream_duration_ms: number;
  termination_reason: string;
  heartbeat_count: number;
}

interface RuntimeStreamMetrics {
  stream_ttft_p95_ms: number;
  stream_first_delta_p95_ms: number;
  stream_duration_p95_ms: number;
  stream_client_cancel_total: number;
  stream_timeout_total: number;
  stream_error_total: number;
}

interface RuntimeMetricItem {
  label: string;
  value: string;
  valueClassName?: string;
}

interface SSEFrame {
  event?: string;
  id?: string;
  retry?: number;
  data: string;
}

/**
 * 解析 SSE 缓冲区，支持 event/id/retry 及多行 data 拼接。
 * 返回已完成帧与剩余半包字符串（供下一轮拼接）。
 */
function parseSSEFrames(buffer: string): { frames: SSEFrame[]; rest: string } {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const blocks = normalized.split("\n\n");
  const rest = blocks.pop() || "";
  const frames: SSEFrame[] = [];

  for (const block of blocks) {
    if (!block.trim()) continue;

    const frame: SSEFrame = { data: "" };
    const dataLines: string[] = [];
    const lines = block.split("\n");

    for (const rawLine of lines) {
      if (!rawLine || rawLine.startsWith(":")) continue; // comment/heartbeat 注释行
      const colonIndex = rawLine.indexOf(":");
      const field = colonIndex >= 0 ? rawLine.slice(0, colonIndex) : rawLine;
      let value = colonIndex >= 0 ? rawLine.slice(colonIndex + 1) : "";
      if (value.startsWith(" ")) value = value.slice(1);

      if (field === "event") {
        frame.event = value;
      } else if (field === "id") {
        frame.id = value;
      } else if (field === "retry") {
        const parsed = Number.parseInt(value, 10);
        if (Number.isFinite(parsed)) frame.retry = parsed;
      } else if (field === "data") {
        dataLines.push(value);
      }
    }

    frame.data = dataLines.join("\n");
    if (frame.data || frame.event || frame.id || frame.retry !== undefined) {
      frames.push(frame);
    }
  }

  return { frames, rest };
}

function createEmptyStreamSections(): StreamSections {
  return {
    reasoning: "",
    actions: [],
    citations: [],
    progress: [],
  };
}

function normalizeStreamSections(value: unknown): StreamSections {
  if (!value || typeof value !== "object") return createEmptyStreamSections();
  const raw = value as Record<string, unknown>;
  const reasoning = typeof raw.reasoning === "string" ? raw.reasoning : "";
  const actions = Array.isArray(raw.actions)
    ? raw.actions.filter((item): item is StreamSectionAction => !!item && typeof item === "object")
    : [];
  const citations = Array.isArray(raw.citations)
    ? raw.citations.filter((item): item is StreamSectionCitation => !!item && typeof item === "object")
    : [];
  const progress = Array.isArray(raw.progress)
    ? raw.progress.map((item) => String(item || "")).filter((item) => item.trim().length > 0)
    : [];
  return { reasoning, actions, citations, progress };
}

function hasStreamSections(sections: StreamSections | undefined): boolean {
  if (!sections) return false;
  return Boolean(
    sections.reasoning.trim() ||
    sections.actions.length > 0 ||
    sections.citations.length > 0 ||
    sections.progress.length > 0
  );
}

const SESSION_ID_REGEX = /^[\u4e00-\u9fa5a-zA-Z0-9_-]{1,64}$/;
const isValidSessionId = (sid: string) => SESSION_ID_REGEX.test((sid || "").trim());

interface KnowledgeFile {
  filename: string;
  size: number;
  modified_at: number;
  embedded?: boolean;  // 是否已完成向量化
  status?: "processing" | "completed" | "failed";
  detail?: string;
}

// ============== 设计系统 ==============
// 使用 CSS 变量实现 Editorial Minimalist 风格
// 青色品牌色: #14b8a6 (teal-500)

// ============== 思考链组件 ==============
/** 展示系统思考片段的轻量面板，仅用于可视化反馈。 */
function ThoughtChain({ thoughts }: { thoughts: string[] }) {
  if (thoughts.length === 0) return null;

  return (
    <div className="mb-4 p-4 bg-gradient-to-r from-slate-50 to-slate-100 border-l-4 border-teal-500 rounded-r-xl font-mono text-xs">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-2 h-2 rounded-full bg-teal-500 animate-pulse"></div>
        <span className="text-teal-700 font-semibold">AI 思考中...</span>
      </div>
      {thoughts.map((thought, idx) => (
        <div key={idx} className="flex items-start gap-2 text-slate-600" style={{ animationDelay: `${idx * 0.15}s` }}>
          <span className="text-teal-500 font-bold">&gt;</span>
          <span className="leading-relaxed">{thought}</span>
        </div>
      ))}
    </div>
  );
}

function StreamSectionsPanel({ sections }: { sections: StreamSections }) {
  if (!hasStreamSections(sections)) return null;
  return (
    <div className="mt-3 space-y-2">
      {sections.reasoning.trim() && (
        <details className="rounded-lg border border-teal-200 bg-teal-50/40 px-3 py-2" open>
          <summary className="cursor-pointer text-xs font-semibold text-teal-700">推理摘要</summary>
          <div className="mt-2 text-xs leading-relaxed text-teal-900 whitespace-pre-wrap">{sections.reasoning}</div>
        </details>
      )}
      {sections.actions.length > 0 && (
        <details className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
          <summary className="cursor-pointer text-xs font-semibold text-slate-700">执行轨迹</summary>
          <div className="mt-2 space-y-1 text-xs text-slate-600">
            {sections.actions.map((action, idx) => (
              <div key={`${action.name || "action"}-${idx}`}>
                {`${action.name || "action"} · ${action.status || "running"}${action.detail ? ` · ${action.detail}` : ""}${typeof action.cost_ms === "number" ? ` · ${action.cost_ms}ms` : ""}`}
              </div>
            ))}
          </div>
        </details>
      )}
      {sections.citations.length > 0 && (
        <details className="rounded-lg border border-amber-200 bg-amber-50/40 px-3 py-2">
          <summary className="cursor-pointer text-xs font-semibold text-amber-700">引用依据</summary>
          <div className="mt-2 space-y-2 text-xs text-amber-900">
            {sections.citations.map((citation, idx) => (
              <div key={`${citation.source || "source"}-${idx}`} className="rounded border border-amber-200 bg-white px-2 py-1">
                <div className="font-medium">{citation.source || `参考资料${idx + 1}`}</div>
                {citation.quote ? <div className="mt-1 text-amber-800">{citation.quote}</div> : null}
              </div>
            ))}
          </div>
        </details>
      )}
      {sections.progress.length > 0 && (
        <details className="rounded-lg border border-sky-200 bg-sky-50/40 px-3 py-2">
          <summary className="cursor-pointer text-xs font-semibold text-sky-700">流程进度</summary>
          <div className="mt-2 space-y-1 text-xs text-sky-900">
            {sections.progress.map((item, idx) => (
              <div key={`progress-${idx}`}>{item}</div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

// ============== Markdown 渲染组件 ==============
/**
 * 从聊天文本中提取题目结构（兼容新旧两种题目格式）。
 * 解析失败时返回 isQuiz=false，回退为普通 markdown 渲染。
 */
function parseQuizContent(content: string): {
  isQuiz: boolean;
  questions: QuizQuestion[];
} {
  const questions: QuizQuestion[] = [];

  const parseInlineOptions = (text: string): { question: string; options: string[] } => {
    const match = text.match(/^(.*?)(?=\s+[A-D][.、]\s*)/);
    if (!match) {
      return { question: text.trim(), options: [] };
    }

    const question = match[1].trim();
    const optionPart = text.slice(match[0].length).trim();
    const optionMatches = optionPart.match(/[A-D][.、]\s*.*?(?=(?:\s+[A-D][.、]\s*)|$)/g) || [];
    return {
      question,
      options: optionMatches.map(opt => opt.trim()),
    };
  };

  // 检查是否包含题型标记（支持新旧两种格式）
  // 新格式：1.（选择题，分值：5分，难度：中等）
  // 旧格式：【选择题】
  const hasNewFormat = /\（选择题，|\（填空题，|\（判断题，|\（简答题，/.test(content);
  const hasOldFormat = /【选择题】|【填空题】|【判断题】|【简答题】/.test(content);
  const hasInlineOptions = /[A-D][.、][\s\S]*[A-D][.、][\s\S]*/.test(content);

  if (!hasNewFormat && !hasOldFormat && !hasInlineOptions) {
    return { isQuiz: false, questions: [] };
  }

  // 分割每道题（统一按编号分段）
  const questionBlocks = content.split(/(?=^\d+\.)/gm);

  for (const block of questionBlocks) {
    const trimmed = block.trim();
    if (!trimmed || !/^\d+\./.test(trimmed)) continue;

    const lines = trimmed.split("\n");
    let questionText = "";
    let options: string[] = [];
    let answer = "";
    let explanation = "";
    let score = "";
    let difficulty = "";
    let currentType = "选择题";  // 默认题型
    let inExplanation = false;  // 标记是否已进入解析区域

    for (const line of lines) {
      const trimmedLine = line.trim();
      if (!trimmedLine) continue;

      // 解析元数据：1.（选择题，分值：5分，难度：中等）
      const metaMatch = trimmedLine.match(/^\d+\.\s*（([^）]+)/);
      if (metaMatch) {
        const meta = metaMatch[1];
        // 解析题型
        if (meta.includes("选择题")) currentType = "选择题";
        else if (meta.includes("填空题")) currentType = "填空题";
        else if (meta.includes("判断题")) currentType = "判断题";
        else if (meta.includes("简答题")) currentType = "简答题";
        // 解析分值
        const scoreMatch = meta.match(/分值[：:]?\s*(\d+)分/);
        // 解析难度
        const diffMatch = meta.match(/难度[：:]?\s*([^，,，]+)/);
        if (scoreMatch) score = scoreMatch[1] + "分";
        if (diffMatch) difficulty = diffMatch[1].trim();
        inExplanation = false;
        continue;
      }

      // 选项（A. B. C. D.）仅选择题解析
      if (currentType === "选择题" && /^[A-D][.、]\s+\S+/.test(trimmedLine)) {
        options.push(trimmedLine);
        inExplanation = false;
      }
      // 答案行
      else if (trimmedLine.startsWith("答案：") || trimmedLine.startsWith("答案:")) {
        answer = trimmedLine.replace(/^答案[：:]\s*/, "");
        inExplanation = true;  // 答案之后的内容属于解析
      }
      // 解析行（单独一行）
      else if (trimmedLine.startsWith("解析：") || trimmedLine.startsWith("解析:")) {
        explanation = trimmedLine.replace(/^解析[：:]\s*/, "");
        inExplanation = true;
      }
      // 跳过题型标题（旧格式）
      else if (trimmedLine.includes("【") && trimmedLine.includes("】")) {
        continue;
      }
      // 如果已进入解析区域，后续行追加到解析
      else if (inExplanation && trimmedLine) {
        explanation += (explanation ? "\n" : "") + trimmedLine;
      }
      // 行内选项仅选择题解析，防止简答题正文被误拆
      else if (currentType === "选择题" && /\s+[A-D][.、]\s+/.test(trimmedLine)) {
        const parsed = parseInlineOptions(trimmedLine);
        if (parsed.question) {
          questionText += (questionText ? "\n" : "") + parsed.question;
        }
        if (parsed.options.length > 0) {
          options.push(...parsed.options);
        }
      }
      // 其他行视为题目内容
      else if (trimmedLine) {
        questionText += (questionText ? "\n" : "") + trimmedLine;
      }
    }

    if (questionText) {
      questions.push({
        type: currentType,
        question: questionText,
        options: options.length > 0 ? options : undefined,
        answer,
        explanation,
        score,
        difficulty
      });
    }
  }

  if (questions.length === 0 && hasInlineOptions) {
    const inlineParsed = parseInlineOptions(content.replace(/^\d+[.、]\s*/, "").trim());
    if (inlineParsed.question && inlineParsed.options.length >= 2) {
      questions.push({
        type: "选择题",
        question: inlineParsed.question,
        options: inlineParsed.options,
        answer: "",
        explanation: "",
      });
    }
  }

  return { isQuiz: questions.length > 0, questions };
}

// 题型样式映射
const quizTypeStyles: Record<string, { tag: string; border: string; bg: string }> = {
  "选择题": { tag: "quiz-tag-choice", border: "border-l-blue-500", bg: "bg-blue-50" },
  "填空题": { tag: "quiz-tag-fill", border: "border-l-green-500", bg: "bg-green-50" },
  "判断题": { tag: "quiz-tag-judge", border: "border-l-orange-500", bg: "bg-orange-50" },
  "简答题": { tag: "quiz-tag-short", border: "border-l-purple-500", bg: "bg-purple-50" },
};

/** 清理选项前缀（A./A、 等），用于统一展示与判分。 */
function stripChoiceOptionPrefix(text: string): string {
  let s = String(text || "");
  // 第一遍：去掉开头的前缀标记（字母+分隔符，可重复）
  s = s.replace(/^([A-D][.、．:：)\s]+)+/i, "");
  // 第二遍：处理残留的 "A xxx" 格式（上一个 replace 没处理干净的情况）
  s = s.replace(/^[A-D][.、．:：)\s]+/i, "");
  return s.trim();
}

function QuizCard({
  questions,
  defaultShowAnswers = false,
  onPracticeComplete,
  messageId,
}: {
  questions: QuizQuestion[];
  defaultShowAnswers?: boolean;
  onPracticeComplete?: (
    messageId: string,
    score: number,
    total: number,
    records: PracticeRecordItem[],
    wrongAnswers: ExamPracticeQuestion[],
    userAnswers: Record<number, string>
  ) => void;
  messageId?: string;
}) {
  // 题卡内部维护“练习模式/作答状态”，与外层消息流状态解耦。
  const [showAnswers, setShowAnswers] = useState(defaultShowAnswers);
  const [practiceMode, setPracticeMode] = useState(false);
  const [selectedAnswers, setSelectedAnswers] = useState<Record<number, string>>({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);

  const normalizeAnswer = (a: string) => a.trim().replace(/\s+/g, " ").toUpperCase();

  const startPractice = () => {
    setPracticeMode(true);
    setSelectedAnswers({});
    setSubmitted(false);
    setScore(0);
    setShowAnswers(false);
  };

  const submitPractice = () => {
    let correctCount = 0;
    const wrongAnswers: ExamPracticeQuestion[] = [];
    const records: PracticeRecordItem[] = [];

    questions.forEach((q, idx) => {
      const userAnswer = normalizeAnswer(selectedAnswers[idx] || "");
      const correctAnswer = normalizeAnswer(q.answer);
      const isCorrect = userAnswer === correctAnswer;

      if (isCorrect) {
        correctCount++;
      } else {
        wrongAnswers.push({
          number: idx + 1,
          type: q.type,
          content: q.question,
          answer: q.answer,
          analysis: q.explanation || "",
          score: parseFloat(q.score || "0"),
          difficulty: q.difficulty || "中等",
        });
      }

      records.push({
        question: {
          number: idx + 1,
          type: q.type,
          content: q.question,
          answer: q.answer,
          analysis: q.explanation || "",
          score: parseFloat(q.score || "0"),
          difficulty: q.difficulty || "中等",
        },
        userAnswer: selectedAnswers[idx] || "",
        isCorrect,
      });
    });

    setScore(correctCount);
    setSubmitted(true);
    setShowAnswers(true);

    if (onPracticeComplete && messageId) {
      onPracticeComplete(messageId, correctCount, questions.length, records, wrongAnswers, selectedAnswers);
    }
  };

  return (
    <div className="space-y-4 my-4">
      {/* 工具栏 */}
      <div className="flex justify-between items-center mb-2">
        {practiceMode ? (
          <div className="flex gap-2">
            <button
              onClick={submitPractice}
              disabled={submitted}
              className="text-sm px-4 py-1.5 rounded-lg bg-teal-600 hover:bg-teal-700 disabled:bg-slate-300 text-white transition-colors"
            >
              提交答案
            </button>
            {submitted && (
              <span className="text-sm py-1.5 px-3 rounded-lg bg-slate-100 text-slate-700 font-medium">
                得分: {score}/{questions.length}
              </span>
            )}
          </div>
        ) : (
          <div />
        )}
        <div className="flex gap-2">
          {!practiceMode && !submitted && (
            <button
              onClick={startPractice}
              className="text-sm px-3 py-1.5 rounded-lg border border-teal-400 bg-teal-50 hover:bg-teal-100 text-teal-700 transition-colors"
            >
              开始练习
            </button>
          )}
          {!practiceMode && (
            <button
              onClick={() => setShowAnswers(!showAnswers)}
              className="text-sm px-3 py-1.5 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 text-slate-600 transition-colors flex items-center gap-1"
            >
              <span>{showAnswers ? "隐藏答案" : "查看答案"}</span>
            </button>
          )}
        </div>
      </div>

      {questions.map((q, idx) => {
        const style = quizTypeStyles[q.type] || quizTypeStyles["选择题"];
        const userAnswer = selectedAnswers[idx];

        return (
          <div
            key={idx}
            className="quiz-card border border-slate-200 rounded-xl p-5 bg-white shadow-sm hover:shadow-md transition-shadow"
            style={{ "--card-index": idx } as React.CSSProperties}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className={`text-xs px-2.5 py-1 rounded-full font-medium border ${style.tag}`}>{q.type}</span>
                <span className="text-sm font-medium text-slate-500">第 {idx + 1} 题</span>
              </div>
              <div className="flex items-center gap-3">
                {q.score && (
                  <span className="text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-medium">{q.score}</span>
                )}
                {q.difficulty && (
                  <span className={`text-xs px-2 py-0.5 rounded font-medium ${
                    q.difficulty === '简单' ? 'bg-green-100 text-green-700' :
                    q.difficulty === '中等' ? 'bg-yellow-100 text-yellow-700' :
                    ['较难', '困难'].includes(q.difficulty) ? 'bg-red-100 text-red-700' :
                    'bg-slate-100 text-slate-600'
                  }`}>{q.difficulty}</span>
                )}
              </div>
            </div>

            <div className="text-slate-800 mb-4 whitespace-pre-wrap text-[15px] leading-relaxed font-medium">{q.question}</div>

            {/* 选择题选项 */}
            {q.options && q.options.length > 0 && (
              <div className="space-y-2 mb-4 ml-1">
                {q.options.map((opt, optIdx) => {
                  const optLetter = String.fromCharCode(65 + optIdx);
                  const isSelected = userAnswer === optLetter;
                  const correctOptLetter = normalizeAnswer(q.answer).charAt(0);
                  const isCorrectOpt = submitted && optLetter === correctOptLetter;
                  const isWrongOpt = submitted && isSelected && optLetter !== correctOptLetter;

                  let optClass = "bg-white border-slate-200";
                  if (practiceMode && !submitted) {
                    optClass = isSelected ? "bg-teal-100 border-teal-500" : "bg-white border-slate-200 hover:bg-slate-50";
                  } else if (submitted) {
                    optClass = isCorrectOpt ? "bg-green-100 border-green-500" : isWrongOpt ? "bg-red-100 border-red-500" : "bg-white border-slate-200";
                  }

                  return (
                    <div
                      key={optIdx}
                      onClick={() => {
                        if (practiceMode && !submitted) {
                          setSelectedAnswers(prev => ({ ...prev, [idx]: optLetter }));
                        }
                      }}
                      className={`flex items-start gap-2 text-slate-700 text-sm border-2 rounded-lg p-2 cursor-pointer transition-all ${optClass} ${practiceMode && !submitted ? "cursor-pointer" : ""}`}
                    >
                      <span className={`w-6 h-6 rounded flex items-center justify-center text-xs font-medium flex-shrink-0 ${
                        isCorrectOpt ? "bg-green-500 text-white" :
                        isWrongOpt ? "bg-red-500 text-white" :
                        isSelected ? "bg-teal-500 text-white" :
                        "bg-slate-100 text-slate-500"
                      }`}>
                        {isCorrectOpt ? "✓" : isWrongOpt ? "✗" : optLetter}
                      </span>
                      <span>{stripChoiceOptionPrefix(opt)}</span>
                    </div>
                  );
                })}
              </div>
            )}

            {/* 非选择题 */}
            {(!q.options || q.options.length === 0) && q.type === "判断题" && (
              <div className="flex gap-4 mb-4 ml-1">
                {["正确", "错误"].map((opt, i) => {
                  const optKey = opt;
                  const isSelected = userAnswer === optKey;
                  const isCorrectOpt = submitted && optKey === q.answer.trim();
                  const isWrongOpt = submitted && isSelected && optKey !== q.answer.trim();

                  let optClass = "bg-white border-slate-200";
                  if (practiceMode && !submitted) {
                    optClass = isSelected ? "bg-teal-100 border-teal-500" : "bg-white border-slate-200 hover:bg-slate-50";
                  } else if (submitted) {
                    optClass = isCorrectOpt ? "bg-green-100 border-green-500" : isWrongOpt ? "bg-red-100 border-red-500" : "bg-white border-slate-200";
                  }

                  return (
                    <div
                      key={opt}
                      onClick={() => {
                        if (practiceMode && !submitted) {
                          setSelectedAnswers(prev => ({ ...prev, [idx]: optKey }));
                        }
                      }}
                      className={`px-4 py-2 rounded cursor-pointer transition-all border-2 ${optClass} ${practiceMode && !submitted ? "cursor-pointer" : ""}`}
                    >
                      <span className={`flex-shrink-0 w-6 h-6 inline-flex items-center justify-center font-medium rounded mr-2 ${
                        isCorrectOpt ? "bg-green-500 text-white" :
                        isWrongOpt ? "bg-red-500 text-white" :
                        isSelected ? "bg-teal-500 text-white" :
                        "bg-slate-100 text-slate-600"
                      }`}>
                        {isCorrectOpt ? "✓" : isWrongOpt ? "✗" : "○"}
                      </span>
                      <span className="text-slate-700">{opt}</span>
                    </div>
                  );
                })}
              </div>
            )}

            {(!q.options || q.options.length === 0) && q.type === "填空题" && (
              <div className="ml-1 mb-4">
                <input
                  type="text"
                  value={userAnswer || ""}
                  onChange={(e) => {
                    if (practiceMode && !submitted) {
                      setSelectedAnswers(prev => ({ ...prev, [idx]: e.target.value }));
                    }
                  }}
                  disabled={submitted}
                  placeholder={practiceMode && !submitted ? "请输入答案..." : ""}
                  className={`w-full px-4 py-2 border-2 rounded text-sm ${
                    submitted
                      ? normalizeAnswer(userAnswer || "") === normalizeAnswer(q.answer)
                        ? "bg-green-50 border-green-500 text-green-800"
                        : "bg-red-50 border-red-500 text-red-800"
                      : "bg-white border-slate-200 focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
                  }`}
                />
                {submitted && normalizeAnswer(userAnswer || "") !== normalizeAnswer(q.answer) && (
                  <div className="mt-1 text-xs text-slate-500">正确答案: <span className="font-medium text-green-700">{q.answer}</span></div>
                )}
              </div>
            )}

            {/* 答案（练习提交后自动显示） */}
            {showAnswers && q.answer && (
              <div className={`${style.bg} ${style.border} border-l-4 rounded-r-lg p-3 mb-3`}>
                <div className="flex items-center gap-2 mb-1">
                  <svg className="w-4 h-4 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="font-semibold text-teal-800 text-sm">答案</span>
                </div>
                <span className="text-teal-900 font-medium">{q.answer}</span>
              </div>
            )}

            {/* 解析（练习提交后自动显示） */}
            {showAnswers && q.explanation && (
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <svg className="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="font-semibold text-slate-700 text-sm">解析</span>
                </div>
                <span className="text-slate-600 text-sm leading-relaxed">{q.explanation}</span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/**
 * 统一消息渲染入口：
 * - exam_paper => ExamCanvas
 * - quiz_set  => QuizCard
 * - 其他      => Markdown
 */
function MarkdownContent({
  content,
  kind,
  payload,
  onRequestAnswers,
  similarQuestions,
  onPracticeComplete,
}: {
  content: string;
  kind?: MessageKind;
  payload?: AssistantPayload;
  onRequestAnswers?: () => void;
  similarQuestions?: Record<number, SimilarQuestion[]>;
  // Handles both ExamCanvas (5 args) and QuizCard (6 args with messageId)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onPracticeComplete?: (...args: any[]) => void;
}) {
  const parsedQuiz = parseQuizContent(content);

  if (kind === "exam_paper") {
    return (
      <ExamCanvas
        examContent={content}
        examDataOverride={payload?.exam_data}
        courseName={payload?.title || "期末考试"}
        onRequestAnswers={onRequestAnswers}
        similarQuestions={similarQuestions}
        onPracticeComplete={onPracticeComplete as undefined}
      />
    );
  }

  if (kind === "quiz_set") {
    const questions = payload?.questions && payload.questions.length > 0
      ? payload.questions
      : parsedQuiz.questions;
    if (questions.length > 0) {
      return <QuizCard questions={questions} defaultShowAnswers={false} messageId={payload?.messageId} onPracticeComplete={(msgId, sc, tot, rec, wrong) => onPracticeComplete?.(msgId, sc, tot, rec, wrong, {})} />;
    }
  }

  // 检测是否为试卷格式，使用 Canvas 渲染
  if (isExamContent(content)) {
    // 直接使用 ExamCanvas 渲染，它内部会处理解析失败的情况
    // 解析失败时会渲染原始内容作为后备
    return (
      <ExamCanvas
        examContent={content}
        courseName="期末考试"
        onRequestAnswers={onRequestAnswers}
        similarQuestions={similarQuestions}
        onPracticeComplete={onPracticeComplete}
      />
    );
  }

  if (kind === "chat" && !parsedQuiz.isQuiz) {
    return (
      <div className="prose-custom chat-prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => <h1 className="text-2xl font-bold my-4 text-slate-900">{children}</h1>,
          h2: ({ children }) => <h2 className="text-xl font-bold my-3 text-slate-800">{children}</h2>,
          h3: ({ children }) => <h3 className="text-lg font-semibold my-2 text-slate-800">{children}</h3>,
          p: ({ children }) => <p className="my-3 leading-relaxed text-slate-700">{children}</p>,
          ul: ({ children }) => <ul className="list-disc pl-6 my-3 space-y-1">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal pl-6 my-3 space-y-1">{children}</ol>,
          li: ({ children }) => <li className="text-slate-700">{children}</li>,
          strong: ({ children }) => <strong className="font-semibold text-slate-900">{children}</strong>,
          code: ({ children, className }) => {
            const isInline = !className;
            if (isInline) {
              return <code className="bg-slate-100 px-1.5 py-0.5 rounded text-sm font-mono text-teal-700 border border-slate-200">{children}</code>;
            }
            return (
              <pre className="bg-slate-900 text-slate-100 p-4 rounded-lg overflow-x-auto my-4 text-sm font-mono">
                <code>{children}</code>
              </pre>
            );
          },
          table: ({ children }) => (
            <div className="overflow-x-auto my-4">
              <table className="min-w-full border-collapse border border-slate-300">{children}</table>
            </div>
          ),
          th: ({ children }) => <th className="border border-slate-300 bg-slate-100 px-4 py-2 text-left text-sm font-semibold">{children}</th>,
          td: ({ children }) => <td className="border border-slate-300 px-4 py-2 text-sm">{children}</td>,
          blockquote: ({ children }) => <blockquote className="border-l-4 border-teal-500 pl-4 my-3 text-slate-600 italic">{children}</blockquote>,
          hr: () => <hr className="my-6 border-slate-300" />,
        }}
      >
        {content}
      </ReactMarkdown>
      </div>
    );
  }

  // 尝试用简单方式解析题目
  if (parsedQuiz.isQuiz && parsedQuiz.questions.length > 0) {
    const questions = payload?.questions && payload.questions.length > 0
      ? payload.questions
      : parsedQuiz.questions;
    return <QuizCard questions={questions} defaultShowAnswers={false} messageId={payload?.messageId} onPracticeComplete={(msgId, sc, tot, rec, wrong) => onPracticeComplete?.(msgId, sc, tot, rec, wrong, {})} />;
  }

  return (
    <div className="prose-custom chat-prose">
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h1: ({ children }) => <h1 className="text-2xl font-bold my-4 text-slate-900">{children}</h1>,
        h2: ({ children }) => <h2 className="text-xl font-bold my-3 text-slate-800">{children}</h2>,
        h3: ({ children }) => <h3 className="text-lg font-semibold my-2 text-slate-800">{children}</h3>,
        p: ({ children }) => <p className="my-3 leading-relaxed text-slate-700">{children}</p>,
        ul: ({ children }) => <ul className="list-disc pl-6 my-3 space-y-1">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal pl-6 my-3 space-y-1">{children}</ol>,
        li: ({ children }) => <li className="text-slate-700">{children}</li>,
        strong: ({ children }) => <strong className="font-semibold text-slate-900">{children}</strong>,
        code: ({ children, className }) => {
          const isInline = !className;
          if (isInline) {
            return <code className="bg-slate-100 px-1.5 py-0.5 rounded text-sm font-mono text-teal-700 border border-slate-200">{children}</code>;
          }
          return (
            <pre className="bg-slate-900 text-slate-100 p-4 rounded-lg overflow-x-auto my-4 text-sm font-mono">
              <code>{children}</code>
            </pre>
          );
        },
        table: ({ children }) => (
          <div className="overflow-x-auto my-4">
            <table className="min-w-full border-collapse border border-slate-300">{children}</table>
          </div>
        ),
        th: ({ children }) => <th className="border border-slate-300 bg-slate-100 px-4 py-2 text-left text-sm font-semibold">{children}</th>,
        td: ({ children }) => <td className="border border-slate-300 px-4 py-2 text-sm">{children}</td>,
        blockquote: ({ children }) => <blockquote className="border-l-4 border-teal-500 pl-4 my-3 text-slate-600 italic">{children}</blockquote>,
        hr: () => <hr className="my-6 border-slate-300" />,
      }}
    >
      {content}
    </ReactMarkdown>
    </div>
  );
}

// ============== 知识库面板组件 ==============
/** 课件上传与样卷管理面板。 */
function KnowledgePanel({
  sessionId,
  isOpen,
  onClose
}: {
  sessionId: string;
  isOpen: boolean;
  onClose: () => void;
}) {
  const [activeTab, setActiveTab] = useState<"upload" | "list" | "sample">("upload");
  const [files, setFiles] = useState<KnowledgeFile[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [sampleStatus, setSampleStatus] = useState("");
  const [hasSample, setHasSample] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const sampleInputRef = useRef<HTMLInputElement>(null);

  const fetchFiles = useCallback(async () => {
    if (!sessionId) return;
    try {
      const res = await fetch(`/api/knowledge/list?session_id=${encodeURIComponent(sessionId)}`);
      const data = await res.json();
      if (data.files) {
        setFiles(data.files);
      }
    } catch (error) {
      console.error("Failed to fetch files:", error);
    }
  }, [sessionId]);

  useEffect(() => {
    if (isOpen && activeTab === "list") {
      fetchFiles();
    }
  }, [isOpen, activeTab, fetchFiles]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files);
      setSelectedFiles(prev => [...prev, ...newFiles].slice(0, 5));
    }
  };

  const removeFile = (idx: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== idx));
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0 || !sessionId) return;

    console.log("[上传] sessionId:", sessionId);
    setIsUploading(true);
    setUploadStatus("上传中...");

    const formData = new FormData();
    formData.append("session_id", sessionId);
    console.log("[上传] 表单中的 session_id:", sessionId);
    selectedFiles.forEach(file => formData.append("files", file));

    try {
      const res = await fetch(`/api/knowledge/upload?session_id=${encodeURIComponent(sessionId)}`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      console.log("[上传] 响应:", data);
      if (res.ok) {
        setSelectedFiles([]);
        // 不立即切换 tab，让用户看到进度
        setUploadStatus("上传成功，开始向量化...");

        // 轮询等待向量化完成
        let attempts = 0;
        const maxAttempts = 60; // 最多 2 分钟
        const pollInterval = setInterval(async () => {
          attempts++;
          console.log("[轮询] 第", attempts, "次");
          // 重新获取文件列表
          try {
            const listRes = await fetch(`/api/knowledge/list?session_id=${encodeURIComponent(sessionId)}`);
            const listData = await listRes.json();
            console.log("[轮询] 文件列表:", listData);
            if (listData.files && listData.files.length > 0) {
              const completedCount = listData.files.filter(
                (f: KnowledgeFile) => f.status === "completed" || f.embedded === true
              ).length;
              const failedCount = listData.files.filter((f: KnowledgeFile) => f.status === "failed").length;
              const processingCount = listData.files.filter(
                (f: KnowledgeFile) => f.status === "processing" || (f.status == null && f.embedded === false)
              ).length;
              const totalCount = listData.files.length;
              console.log("[轮询] 完成:", completedCount, "/", totalCount);
              if (processingCount > 0) {
                setUploadStatus(`向量化中... 完成${completedCount}/${totalCount}，失败${failedCount}个`);
              } else {
                setUploadStatus(failedCount > 0 ? `向量化结束：完成${completedCount}，失败${failedCount}` : "向量化完成！");
                clearInterval(pollInterval);
                setActiveTab("list");
                setTimeout(() => setUploadStatus(""), 2000);
              }
            } else {
              setUploadStatus(`向量化中...（等待文件出现）`);
            }
          } catch (e) {
            console.error("轮询失败:", e);
          }

          if (attempts >= maxAttempts) {
            clearInterval(pollInterval);
            setUploadStatus("向量化超时，请刷新页面查看");
            setActiveTab("list");
          }
        }, 2000);
        // 立即执行一次
        fetchFiles();
      } else {
        setUploadStatus(data.detail || "上传失败");
      }
    } catch (error) {
      setUploadStatus("上传失败: " + error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleRetryFailed = async () => {
    if (!sessionId) return;
    setUploadStatus("正在提交失败文件重试任务...");
    try {
      const res = await fetch(`/api/knowledge/retry-failed?session_id=${encodeURIComponent(sessionId)}`, {
        method: "POST",
      });
      const data = await res.json();
      if (!res.ok || data.code !== 200) {
        setUploadStatus(data.message || "重试提交失败");
        return;
      }
      setUploadStatus(data.retry_count > 0 ? `已提交重试任务：${data.retry_count} 个文件` : "没有可重试的失败文件");
      await fetchFiles();
    } catch (error) {
      setUploadStatus("重试提交失败: " + error);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  };

  // 检查样卷是否存在
  useEffect(() => {
    if (!sessionId || !isOpen) return;
    fetch(`/api/knowledge/sample?session_id=${encodeURIComponent(sessionId)}`)
      .then(res => res.json())
      .then(data => {
        setHasSample(data.code === 200 && data.data?.has_sample);
      })
      .catch(console.error);
  }, [sessionId, isOpen]);

  // 上传样卷
  const handleSampleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0] || !sessionId) return;
    const file = e.target.files[0];
    setSampleStatus("正在上传和分析样卷...");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`/api/knowledge/sample/upload?session_id=${encodeURIComponent(sessionId)}`, { method: "POST", body: formData });
      const data = await res.json();
      if (res.ok) {
        setSampleStatus("样卷上传成功，格式已学习！");
        setHasSample(true);
      } else {
        setSampleStatus(data.message || "上传失败");
      }
    } catch (error) {
      setSampleStatus("上传失败: " + error);
    }
  };

  // 删除样卷
  const handleDeleteSample = async () => {
    if (!sessionId || !confirm("确定删除样卷吗？")) return;
    try {
      const res = await fetch(`/api/knowledge/sample?session_id=${encodeURIComponent(sessionId)}`, { method: "DELETE" });
      if (res.ok) { setSampleStatus("样卷已删除"); setHasSample(false); }
    } catch (error) { console.error(error); }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-sm flex items-center justify-center animate-fadeIn">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-md overflow-hidden animate-slideUp">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-100 flex justify-between items-center">
          <div>
            <h3 className="text-sm font-bold text-slate-900">知识库管理</h3>
            <p className="text-xs font-mono text-slate-400 mt-0.5">当前科目：{sessionId}</p>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-900 p-1 rounded hover:bg-slate-100 transition-colors">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-slate-100 px-6">
          <button
            onClick={() => setActiveTab("upload")}
            className={`py-2.5 pr-4 text-xs font-semibold border-b-2 transition-colors ${
              activeTab === "upload"
                ? "border-slate-900 text-slate-900"
                : "border-transparent text-slate-400 hover:text-slate-700"
            }`}
          >
            上传文件
          </button>
          <button
            onClick={() => setActiveTab("list")}
            className={`py-2.5 px-4 text-xs font-semibold border-b-2 transition-colors ${
              activeTab === "list"
                ? "border-slate-900 text-slate-900"
                : "border-transparent text-slate-400 hover:text-slate-700"
            }`}
          >
            已上传
            {files.length > 0 && (
              <span className="ml-1 px-1.5 py-0.5 rounded-full bg-slate-100 text-slate-500 text-[0.6rem] font-mono">
                {files.length}
              </span>
            )}
          </button>
          <button
            onClick={() => setActiveTab("sample")}
            className={`py-2.5 px-4 text-xs font-semibold border-b-2 transition-colors ${
              activeTab === "sample"
                ? "border-slate-900 text-slate-900"
                : "border-transparent text-slate-400 hover:text-slate-700"
            }`}
          >
            样卷上传
            {hasSample && <span className="ml-1 text-teal-600">✓</span>}
          </button>
        </div>

        {/* Upload Tab */}
        {activeTab === "upload" && (
          <div className="p-6 flex flex-col gap-4">
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".pdf,.docx,.txt"
              multiple
              onChange={handleFileSelect}
            />
            <div
              onClick={() => fileInputRef.current?.click()}
              className="w-full border-2 border-dashed border-slate-200 hover:border-teal-500 hover:bg-teal-50 transition-colors rounded-lg p-6 text-center cursor-pointer"
            >
              <svg className="w-7 h-7 text-slate-400 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p className="text-sm font-medium text-slate-700">点击或拖拽文件到此处</p>
              <p className="text-xs font-mono text-slate-400 mt-1">支持 .pdf · .docx · txt | 最多 5 个</p>
            </div>

            {selectedFiles.length > 0 && (
              <ul className="flex flex-col gap-2 max-h-44 overflow-y-auto">
                {selectedFiles.map((file, idx) => (
                  <li key={idx} className="flex items-center gap-3 px-3 py-2 rounded-lg bg-slate-50 border border-slate-200 text-xs font-mono group">
                    <span className="px-1.5 py-0.5 rounded text-[0.6rem] font-bold bg-teal-100 text-teal-700">
                      {file.name.split(".").pop()?.toUpperCase()}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="truncate text-slate-800">{file.name}</p>
                      <p className="text-slate-400">{formatBytes(file.size)}</p>
                    </div>
                    <button onClick={() => removeFile(idx)} className="text-slate-300 hover:text-red-500 transition-colors">
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </li>
                ))}
              </ul>
            )}

            {uploadStatus && (
              <p className={`text-xs text-center font-mono ${uploadStatus.includes("失败") ? "text-red-500" : "text-teal-600"}`}>
                {uploadStatus}
              </p>
            )}

            <div className="flex gap-2">
              <button
                onClick={() => setSelectedFiles([])}
                className="flex-1 text-xs font-medium text-slate-500 bg-slate-100 hover:bg-slate-200 border border-slate-200 py-2.5 rounded-md transition-colors"
              >
                清空列表
              </button>
              <button
                onClick={handleUpload}
                disabled={selectedFiles.length === 0 || isUploading}
                className="flex-[2] bg-slate-900 hover:bg-teal-600 text-white font-medium text-sm py-2.5 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isUploading ? "上传中..." : "执行摄入"}
              </button>
            </div>
          </div>
        )}

        {/* List Tab */}
        {activeTab === "list" && (
          <div className="flex flex-col max-h-80">
            <div className="px-6 pt-4 pb-2 flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400">按上传时间倒序</span>
              <div className="flex items-center gap-3">
                <button onClick={handleRetryFailed} className="text-xs font-mono text-amber-600 hover:underline">重试失败</button>
                <button onClick={fetchFiles} className="text-xs font-mono text-teal-600 hover:underline">刷新</button>
              </div>
            </div>
            <ul className="flex-1 overflow-y-auto divide-y divide-slate-100">
              {files.length === 0 ? (
                <li className="px-6 py-8 text-center text-xs font-mono text-slate-400">暂无已上传文件</li>
              ) : (
                files.map((file, idx) => {
                  const ext = file.filename.split(".").pop()?.toLowerCase();
                  const extColors: Record<string, string> = {
                    pdf: "bg-red-100 text-red-700",
                    docx: "bg-blue-100 text-blue-700",
                    txt: "bg-green-100 text-green-700"
                  };
                  const date = new Date(file.modified_at * 1000);
                  const dateStr = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")} ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;

                  return (
                    <li key={idx} className="flex items-center gap-3 px-6 py-3 hover:bg-slate-50 transition-colors">
                      <span className={`px-1.5 py-0.5 rounded text-[0.6rem] font-bold ${extColors[ext || ""] || "bg-slate-200 text-slate-600"}`}>
                        {ext?.toUpperCase()}
                      </span>
                      <div className="flex-1 min-w-0">
                        <p className="truncate text-xs font-medium text-slate-800">{file.filename}</p>
                        <p className="text-xs font-mono text-slate-400">
                          {formatBytes(file.size)} · {dateStr}
                          {(file.status === "processing" || (file.status == null && file.embedded === false)) && (
                            <span className="text-amber-500 ml-1">（处理中）</span>
                          )}
                          {(file.status === "completed" || file.embedded === true) && (
                            <span className="text-green-500 ml-1">✓</span>
                          )}
                          {file.status === "failed" && (
                            <span className="text-red-500 ml-1">（失败）</span>
                          )}
                        </p>
                        {file.status === "failed" && file.detail && (
                          <p className="text-[11px] text-red-500 mt-0.5 truncate" title={file.detail}>
                            {file.detail}
                          </p>
                        )}
                      </div>
                    </li>
                  );
                })
              )}
            </ul>
          </div>
        )}

        {/* Sample Paper Tab */}
        {activeTab === "sample" && (
          <div className="p-6 flex flex-col gap-4">
            <input
              ref={sampleInputRef}
              type="file"
              className="hidden"
              accept=".pdf,.docx,.txt"
              onChange={handleSampleUpload}
            />

            <div className="text-center py-4">
              <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-amber-100 flex items-center justify-center">
                <svg className="w-6 h-6 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h4 className="text-sm font-semibold text-slate-800 mb-1">上传样卷</h4>
              <p className="text-xs text-slate-500 mb-4">上传期末考试样卷，学习出题格式</p>

              {hasSample ? (
                <div className="space-y-3">
                  <div className="px-4 py-2 bg-teal-50 text-teal-700 rounded-lg text-xs">
                    ✓ 样卷已上传，格式已学习
                  </div>
                  <button
                    onClick={handleDeleteSample}
                    className="text-xs text-red-500 hover:underline"
                  >
                    删除样卷
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => sampleInputRef.current?.click()}
                  className="px-4 py-2 bg-amber-500 text-white rounded-lg text-sm font-medium hover:bg-amber-600 transition-colors"
                >
                  选择文件
                </button>
              )}

              {sampleStatus && (
                <p className={`mt-3 text-xs ${sampleStatus.includes("成功") ? "text-teal-600" : "text-red-500"}`}>
                  {sampleStatus}
                </p>
              )}
            </div>

            <div className="text-xs text-slate-400 border-t pt-4">
              <p className="font-medium mb-1">为什么要上传样卷？</p>
              <p>上传样卷后，系统会学习试卷格式（题型、分值、编号风格等），按照相同格式出题</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ============== 历史记录管理面板 ==============
/** 练习历史与薄弱点统计面板。 */
function SessionHistoryPanel({
  sessionId,
  isOpen,
  onClose,
}: {
  sessionId: string;
  isOpen: boolean;
  onClose: () => void;
}) {
  const [practiceHistory, setPracticeHistory] = useState<PracticeHistoryRecord[]>([]);
  const [practiceStats, setPracticeStats] = useState<PracticeStatsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);

  const loadHistoryAndStats = useCallback(async () => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const [historyRes, statsRes] = await Promise.all([
        fetch(`/api/chat/practice/history?session_id=${encodeURIComponent(sessionId)}&limit=300`),
        fetch(`/api/chat/practice/stats?session_id=${encodeURIComponent(sessionId)}`),
      ]);
      const historyData = await historyRes.json();
      const statsData = await statsRes.json();

      if (historyData?.code === 200 && Array.isArray(historyData.data)) {
        setPracticeHistory(historyData.data);
      } else {
        setPracticeHistory([]);
      }

      if (statsData?.code === 200 && statsData?.data) {
        setPracticeStats(statsData.data as PracticeStatsData);
      } else {
        setPracticeStats(null);
      }
    } catch {
      setPracticeHistory([]);
      setPracticeStats(null);
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  // Fetch practice history and stats when panel opens
  useEffect(() => {
    if (!isOpen || !sessionId) return;
    loadHistoryAndStats();
  }, [isOpen, sessionId, loadHistoryAndStats]);

  const handleDeleteRecord = async (recordId: number) => {
    setDeletingId(String(recordId));
    try {
      const res = await fetch("/api/chat/practice/history/item", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, record_id: recordId }),
      });
      const data = await res.json();
      if (data.code === 200) {
        setPracticeHistory(prev => prev.filter(r => r.id !== recordId));
        loadHistoryAndStats();
      }
    } catch { /* ignore */ }
    setDeletingId(null);
  };

  const handleClearAll = async () => {
    if (!confirm("确定清空当前会话的所有练习历史？此操作不可恢复。")) return;
    setClearing(true);
    try {
      const res = await fetch(`/api/chat/practice/history?session_id=${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
      });
      const data = await res.json();
      if (data.code === 200) {
        setPracticeHistory([]);
        loadHistoryAndStats();
      }
    } catch { /* ignore */ }
    setClearing(false);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-sm flex items-center justify-center animate-fadeIn">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-lg overflow-hidden animate-slideUp flex flex-col" style={{ maxHeight: "80vh" }}>
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-100 flex justify-between items-center bg-slate-50 flex-shrink-0">
          <div>
            <h3 className="text-sm font-bold text-slate-900">练习历史</h3>
            <p className="text-xs text-slate-400 font-mono mt-0.5">{practiceHistory.length} 条记录</p>
          </div>
          <div className="flex items-center gap-2">
            {practiceHistory.length > 0 && (
              <button
                onClick={handleClearAll}
                disabled={clearing}
                className="px-3 py-1.5 text-xs font-medium text-red-600 bg-red-50 hover:bg-red-100 border border-red-200 rounded-md transition-all disabled:opacity-50"
              >
                {clearing ? "清空中..." : "清空全部"}
              </button>
            )}
            <button onClick={onClose} className="text-slate-400 hover:text-slate-900">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Practice history list */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="text-center py-8 text-xs text-slate-400">加载中...</div>
          ) : (
            <div className="space-y-4">
              <div className="rounded-lg border border-teal-100 bg-teal-50/60 p-3">
                <div className="text-xs font-semibold text-teal-700 mb-2">薄弱点分析看板</div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="rounded-md bg-white border border-slate-100 px-2.5 py-2">
                    <div className="text-[11px] text-slate-500">总练习</div>
                    <div className="text-sm font-bold text-slate-800">{practiceStats?.total_attempts ?? 0}</div>
                  </div>
                  <div className="rounded-md bg-white border border-slate-100 px-2.5 py-2">
                    <div className="text-[11px] text-slate-500">正确率</div>
                    <div className="text-sm font-bold text-emerald-700">{(practiceStats?.accuracy ?? 0).toFixed(1)}%</div>
                  </div>
                  <div className="rounded-md bg-white border border-slate-100 px-2.5 py-2">
                    <div className="text-[11px] text-slate-500">错题数</div>
                    <div className="text-sm font-bold text-amber-700">{practiceStats?.wrong_attempts ?? 0}</div>
                  </div>
                  <div className="rounded-md bg-white border border-slate-100 px-2.5 py-2">
                    <div className="text-[11px] text-slate-500">错题率</div>
                    <div className="text-sm font-bold text-amber-700">{(practiceStats?.retry_rate ?? 0).toFixed(1)}%</div>
                  </div>
                </div>
                <div className="mt-3">
                  <div className="text-[11px] text-slate-500 mb-1">薄弱知识点（正确率 &lt; 60%）</div>
                  {(practiceStats?.weak_points?.length ?? 0) > 0 ? (
                    <div className="flex flex-wrap gap-1.5">
                      {(practiceStats?.weak_points ?? []).slice(0, 12).map((kp) => (
                        <span key={kp} className="text-[11px] px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 border border-amber-200">
                          {kp}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <div className="text-[11px] text-emerald-700">暂无薄弱点，继续保持。</div>
                  )}
                </div>
                {practiceStats?.knowledge_point_stats && Object.keys(practiceStats.knowledge_point_stats).length > 0 && (
                  <div className="mt-3 rounded-md border border-slate-200 bg-white overflow-hidden">
                    <div className="px-2.5 py-1.5 text-[11px] font-medium text-slate-600 bg-slate-50 border-b">知识点正确率</div>
                    <div className="max-h-40 overflow-y-auto">
                      {Object.entries(practiceStats.knowledge_point_stats)
                        .sort((a, b) => (a[1]?.accuracy ?? 0) - (b[1]?.accuracy ?? 0))
                        .slice(0, 20)
                        .map(([kp, stat]) => (
                          <div key={kp} className="px-2.5 py-1.5 border-b last:border-b-0 border-slate-100 text-[11px] flex items-center justify-between gap-2">
                            <span className="text-slate-700 truncate">{kp}</span>
                            <span className={`font-mono ${stat.weak ? "text-amber-700" : "text-slate-500"}`}>
                              {stat.correct}/{stat.total} · {stat.accuracy.toFixed(1)}%
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>

              {practiceHistory.length === 0 ? (
            <div className="text-center py-8">
              <div className="w-10 h-10 mx-auto mb-3 rounded-full bg-slate-100 flex items-center justify-center">
                <svg className="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <p className="text-xs text-slate-500">暂无练习记录</p>
            </div>
          ) : (
            <div className="space-y-2">
              {practiceHistory.map((rec) => (
                <div
                  key={rec.id}
                  className={`group relative rounded-lg border px-3 py-2.5 transition-all ${
                    rec.is_correct ? "bg-emerald-50 border-emerald-100" : "bg-amber-50 border-amber-100"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      <span className={`flex-shrink-0 text-xs font-bold px-1.5 py-0.5 rounded ${
                        rec.is_correct ? "bg-emerald-200 text-emerald-800" : "bg-amber-200 text-amber-800"
                      }`}>
                        {rec.is_correct ? "正确" : "错题"}
                      </span>
                      <span className="text-[11px] font-mono text-slate-400 flex-shrink-0">
                        {new Date(rec.created_at * 1000).toLocaleString()}
                      </span>
                      {rec.knowledge_point && (
                        <span className="text-[11px] text-slate-500 truncate max-w-[180px]">{rec.knowledge_point}</span>
                      )}
                    </div>
                    <button
                      onClick={() => handleDeleteRecord(rec.id)}
                      disabled={deletingId === String(rec.id)}
                      className="flex-shrink-0 p-1 text-slate-300 hover:text-red-500 rounded transition-all opacity-0 group-hover:opacity-100 disabled:opacity-50"
                      title="删除此记录"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                  <p className="text-xs text-slate-700 mt-1.5 line-clamp-2 leading-relaxed">
                    {String(rec.question_content || "").slice(0, 200)}
                    {String(rec.question_content || "").length > 200 ? "..." : ""}
                  </p>
                  <div className="mt-1 text-[11px] text-slate-500">
                    你的答案：{rec.user_answer || "未填写"} | 正确答案：{rec.correct_answer || "未知"}
                  </div>
                  {!rec.is_correct && rec.wrong_reason && (
                    <div className="mt-1 text-[11px] text-amber-700">错因：{rec.wrong_reason}</div>
                  )}
                </div>
              ))}
            </div>
          )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ============== 会话管理模态框 ==============
/** 会话创建/重命名/分支化管理面板。 */
function SessionModal({
  isOpen,
  onClose,
  sessions,
  currentSession,
  onSelect,
  onCreate,
  onDelete,
  onRename
}: {
  isOpen: boolean;
  onClose: () => void;
  sessions: Session[];
  currentSession: string;
  onSelect: (id: string) => void;
  onCreate: (name: string, parentId?: string) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, newName: string) => void;
}) {
  const [newName, setNewName] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState("");
  const [isCreatingBranch, setIsCreatingBranch] = useState(false);
  const [branchParentId, setBranchParentId] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleCreate = () => {
    if (newName.trim()) {
      onCreate(newName.trim(), branchParentId || undefined);
      setNewName("");
      setIsCreatingBranch(false);
      setBranchParentId(null);
    }
  };

  const startCreateBranch = (parentId: string) => {
    setIsCreatingBranch(true);
    setBranchParentId(parentId);
    setNewName("");
  };

  const startRename = (session: Session) => {
    setEditingId(session.id);
    setEditingName(session.name);
  };

  const handleRename = () => {
    if (editingId && editingName.trim()) {
      onRename(editingId, editingName.trim());
      setEditingId(null);
      setEditingName("");
    }
  };

  const handleDelete = (id: string) => {
    if (confirm("确定删除该会话？此操作不可恢复。")) {
      onDelete(id);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-sm flex items-center justify-center animate-fadeIn">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-sm overflow-hidden animate-slideUp">
        <div className="px-6 py-4 border-b border-slate-100 flex justify-between items-center bg-slate-50">
          <h3 className="text-sm font-bold text-slate-900">会话控制台</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-900">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="p-4">
          {isCreatingBranch ? (
            <div className="mb-4 p-3 bg-teal-50 border border-teal-200 rounded-lg">
              <p className="text-xs text-teal-600 mb-2 font-medium">正在创建分支...</p>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={newName}
                  onChange={e => setNewName(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === "Enter") handleCreate();
                    if (e.key === "Escape") { setIsCreatingBranch(false); setBranchParentId(null); }
                  }}
                  placeholder="分支名称 (如：进程调度复习)"
                  className="flex-1 text-sm border border-teal-400 rounded-md px-3 py-2 focus:outline-none focus:ring-1 focus:ring-teal-500 font-mono"
                  autoFocus
                />
                <button
                  onClick={handleCreate}
                  className="px-3 py-2 bg-teal-600 text-white text-xs font-bold rounded-md hover:bg-teal-700 transition-colors"
                >
                  创建分支
                </button>
                <button
                  onClick={() => { setIsCreatingBranch(false); setBranchParentId(null); setNewName(""); }}
                  className="px-3 py-2 bg-slate-100 text-slate-600 text-xs font-medium rounded-md hover:bg-slate-200 transition-colors"
                >
                  取消
                </button>
              </div>
            </div>
          ) : (
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                value={newName}
                onChange={e => setNewName(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handleCreate()}
                placeholder="新科目名称 (如：Math-101)"
                className="flex-1 text-sm border border-slate-200 rounded-md px-3 py-2 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 font-mono"
              />
              <button
                onClick={handleCreate}
                className="px-3 py-2 bg-slate-900 text-white text-xs font-bold rounded-md hover:bg-teal-600 transition-colors"
              >
                创建
              </button>
            </div>
          )}
          <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2 px-1">活跃的复习会话</div>
          <ul className="space-y-1 font-mono text-sm max-h-60 overflow-y-auto">
            {sessions.map(session => (
              <li key={session.id} className="group">
                {editingId === session.id ? (
                  <div className="flex gap-1 items-center">
                    <input
                      type="text"
                      value={editingName}
                      onChange={e => setEditingName(e.target.value)}
                      onKeyDown={e => {
                        if (e.key === "Enter") handleRename();
                        if (e.key === "Escape") setEditingId(null);
                      }}
                      className="flex-1 text-sm border border-teal-400 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-teal-500 font-mono"
                      autoFocus
                    />
                    <button onClick={handleRename} className="p-1.5 text-teal-600 hover:bg-teal-50 rounded" title="确认">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
                    </button>
                    <button onClick={() => setEditingId(null)} className="p-1.5 text-slate-400 hover:bg-slate-100 rounded" title="取消">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                    </button>
                  </div>
                ) : (
                  <div className={`flex items-center justify-between px-3 py-2 rounded-md transition-colors ${
                    currentSession === session.id
                      ? "bg-teal-50 text-teal-700 border border-teal-200"
                      : "hover:bg-slate-100 text-slate-700"
                  }`}>
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      {session.parent_id && (
                        <span className="text-xs text-slate-400 font-mono">↳</span>
                      )}
                      <button
                        onClick={() => { onSelect(session.id); onClose(); }}
                        className="flex-1 text-left truncate"
                      >
                        {session.name}
                      </button>
                    </div>
                    <div className="hidden group-hover:flex items-center gap-1 ml-2">
                      <button
                        onClick={(e) => { e.stopPropagation(); startCreateBranch(session.id); }}
                        className="p-1 text-slate-400 hover:text-teal-600 hover:bg-teal-50 rounded"
                        title="创建分支"
                      >
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2h-2M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2" /></svg>
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); startRename(session); }}
                        className="p-1 text-slate-400 hover:text-teal-600 hover:bg-slate-200 rounded"
                        title="重命名"
                      >
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" /></svg>
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDelete(session.id); }}
                        className="p-1 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded"
                        title="删除"
                      >
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                      </button>
                    </div>
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

// ============== 主页面组件 ==============
/** 聊天主页：会话管理 + SSE 流式对话 + 练习闭环入口。 */
export default function ChatPage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSession, setCurrentSession] = useState<string>("default");
  const [currentSessionName, setCurrentSessionName] = useState<string>("默认科目");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [thoughts, setThoughts] = useState<string[]>([]);

  // 统计
  const [totalTokens, setTotalTokens] = useState(0);
  const [responseTime, setResponseTime] = useState(0);
  const [streamDiagnostics, setStreamDiagnostics] = useState<StreamDiagnostics>({
    ttft_ms: null,
    first_delta_ms: null,
    stream_duration_ms: 0,
    termination_reason: "idle",
    heartbeat_count: 0,
  });
  const [runtimeStreamMetrics, setRuntimeStreamMetrics] = useState<RuntimeStreamMetrics | null>(null);

  // 模态框状态
  const [showSessionModal, setShowSessionModal] = useState(false);
  const [showKnowledgePanel, setShowKnowledgePanel] = useState(false);
  const [showHistoryPanel, setShowHistoryPanel] = useState(false);
  const [examFastMode, setExamFastMode] = useState(true);  // True=快速路径, False=完整LLM Critique
  const [isBackendConnected, setIsBackendConnected] = useState(true);
  const [similarQuestionsByMessage, setSimilarQuestionsByMessage] = useState<Record<string, Record<number, SimilarQuestion[]>>>({});
  const [retryingMessageId, setRetryingMessageId] = useState<string | null>(null);
  const [practiceFeedbackByMessage, setPracticeFeedbackByMessage] = useState<
    Record<string, { type: "success" | "warning" | "error"; text: string }>
  >({});

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const streamAbortRef = useRef<AbortController | null>(null);

  // 主动停止按钮：中止当前流式请求（前后端 cancel 链路联动）。
  const stopStreaming = useCallback(() => {
    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
    }
  }, []);

  // 缺省知识点提取：优先后端返回，其次题干首行降级。
  const inferKnowledgePoint = (question: ExamPracticeQuestion): string => {
    if (question.knowledge_point && question.knowledge_point.trim()) {
      return question.knowledge_point.trim();
    }
    const head = (question.content || "").split("\n")[0] || "";
    return head.slice(0, 20) || question.type || "未标注知识点";
  };

  const handlePracticeCompleteForMessage = async (
    messageId: string,
    score: number,
    total: number,
    records: PracticeRecordItem[],
    wrongAnswers: ExamPracticeQuestion[],
    userAnswers: Record<number, string>
  ) => {
    // 练习提交闭环：先存作答记录，再补拉相似题用于二次复练。
    if (!currentSession) return;

    try {
      const payloadRecords = records.map((r) => ({
        question_id: `${messageId}_${r.question.number}`,
        question_number: r.question.number,
        question_content: r.question.content,
        knowledge_point: inferKnowledgePoint(r.question),
        user_answer: r.userAnswer || userAnswers[r.question.number] || "",
        correct_answer: r.question.answer || "",
        is_correct: r.isCorrect,
        wrong_reason: r.isCorrect ? "" : (r.question.analysis || ""),
      }));

      if (payloadRecords.length > 0) {
        const submitRes = await fetch("/api/chat/practice/submit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: currentSession,
            records: payloadRecords,
          }),
        });
        if (!submitRes.ok) {
          throw new Error(`练习记录提交失败（HTTP ${submitRes.status}）`);
        }
        const submitData = await submitRes.json();
        if (submitData?.code !== 200) {
          throw new Error(submitData?.message || "练习记录提交失败");
        }

        if (wrongAnswers.length > 0) {
          const similarRes = await fetch("/api/chat/practice/similar", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              session_id: currentSession,
              limit: 3,
              wrong_questions: wrongAnswers.map((q) => ({
                question_number: q.number,
                question_content: q.content,
                knowledge_point: inferKnowledgePoint(q),
              })),
            }),
          });
          if (!similarRes.ok) {
            throw new Error(`相似题获取失败（HTTP ${similarRes.status}）`);
          }
          const similarData = await similarRes.json();
          if (similarData?.code === 200 && similarData?.similar_questions) {
            setSimilarQuestionsByMessage((prev) => ({
              ...prev,
              [messageId]: similarData.similar_questions,
            }));
            const similarCount = Object.values(similarData.similar_questions as Record<string, SimilarQuestion[]>)
              .reduce((acc, arr) => acc + (Array.isArray(arr) ? arr.length : 0), 0);
            setPracticeFeedbackByMessage((prev) => ({
              ...prev,
              [messageId]: {
                type: "success",
                text: `练习已保存，已生成 ${similarCount} 道复练推荐题。`,
              },
            }));
          }
        } else {
          setPracticeFeedbackByMessage((prev) => ({
            ...prev,
            [messageId]: {
              type: "success",
              text: "本次练习全对，记录已保存。",
            },
          }));
        }
        showToast("练习记录已保存，已更新相似题推荐", "success");
      }
      console.log("[Practice] 完成练习:", { score, total, totalCount: records.length, wrongCount: wrongAnswers.length });
    } catch (error) {
      console.error("[Practice] 保存练习记录失败:", error);
      const msg = error instanceof Error ? error.message : "练习记录保存失败";
      setPracticeFeedbackByMessage((prev) => ({
        ...prev,
        [messageId]: {
          type: "error",
          text: `练习记录保存失败：${msg}`,
        },
      }));
      showToast(`练习记录保存失败：${msg}`, "error");
    }
  };

  // 加载会话列表和消息历史
  useEffect(() => {
    fetch("/api/chat/sessions")
      .then(res => res.json())
      .then(data => {
        setIsBackendConnected(true);
        if (data.code === 200 && data.data && data.data.length > 0) {
          setSessions(data.data);
          setCurrentSession(data.data[0].id);
          setCurrentSessionName(data.data[0].name);
          // 加载第一个会话的历史消息
          loadSessionMessages(data.data[0].id);
        } else {
          // 默认会话
          setSessions([{ id: "default", name: "默认科目" }]);
        }
      })
      .catch(() => {
        // API 连接失败，使用默认会话
        setIsBackendConnected(false);
        setSessions([{ id: "default", name: "默认科目" }]);
      });
  }, []);

  // 加载会话消息历史
  const loadSessionMessages = (sessionId: string) => {
    if (!isValidSessionId(sessionId)) {
      setMessages([]);
      showToast(`会话ID非法（${sessionId}），请先清理该会话`, "warning");
      return;
    }
    fetch(`/api/chat/messages?session_id=${encodeURIComponent(sessionId)}`)
      .then(async res => {
        const data = await res.json();
        return { ok: res.ok, status: res.status, data };
      })
      .then(data => {
        if (!data.ok) {
          if (data.status === 404) {
            showToast("会话不存在，可能已被删除", "warning");
            setMessages([]);
            return;
          }
          if (data.status === 400) {
            showToast("会话ID不合法，请删除后重新创建", "warning");
            setMessages([]);
            return;
          }
        }
        if (data.data.code === 200 && data.data.data) {
          setMessages(data.data.data);
        }
      })
      .catch(err => console.error("加载消息失败:", err));
  };

  // 滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thoughts]);

  useEffect(() => {
    return () => {
      if (streamAbortRef.current) {
        streamAbortRef.current.abort();
      }
    };
  }, []);

  // 发送消息
  const sendMessage = async (overrideText?: string, extraBody?: Record<string, unknown>) => {
    // 入口保护：空输入/非法会话/重复发送均直接返回。
    const outgoingText = (overrideText ?? input).trim();
    if (!outgoingText || !currentSession || isLoading) return;
    if (!isValidSessionId(currentSession)) {
      showToast(`当前会话ID非法（${currentSession}），请先删除该会话`, "error");
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: outgoingText,
      timestamp: Date.now()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setThoughts([]);
    setStreamDiagnostics({
      ttft_ms: null,
      first_delta_ms: null,
      stream_duration_ms: 0,
      termination_reason: "running",
      heartbeat_count: 0,
    });

    const startTime = Date.now();
    const assistantMessageId = (Date.now() + 1).toString();

    setMessages(prev => [...prev, {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      timestamp: Date.now(),
      kind: "chat",
      render_mode: "markdown",
      payload: { stream_sections: createEmptyStreamSections() },
    }]);

    let fullContent = "";
    const abortController = new AbortController();
    streamAbortRef.current = abortController;
    let streamTerminationReason = "success";
    let heartbeatCount = 0;
    let ttftMs: number | null = null;
    let firstDeltaMs: number | null = null;
    let seenNonPendingStart = false;
    let seenTextDeltaEvent = false;
    let shouldDrainRenderQueue = true;
    let renderedContent = "";
    let liveSections = createEmptyStreamSections();
    const renderQueue: string[] = [];
    let renderTimer: ReturnType<typeof setTimeout> | null = null;
    let drainResolver: (() => void) | null = null;

    const appendAssistantContent = (content: string) => {
      setMessages(prev => prev.map(m =>
        m.id === assistantMessageId ? { ...m, content } : m
      ));
    };

    const updateAssistantSections = (sections: StreamSections) => {
      setMessages(prev => prev.map(m => {
        if (m.id !== assistantMessageId) return m;
        const payload = (m.payload || {}) as AssistantPayload;
        return {
          ...m,
          payload: {
            ...payload,
            stream_sections: sections,
          },
        };
      }));
    };

    const markTtft = () => {
      // TTFT 只取首个有效时刻，避免后续事件覆盖。
      if (ttftMs === null) {
        ttftMs = Date.now() - startTime;
      }
    };

    const resolveDrainIfIdle = () => {
      if (renderQueue.length === 0 && renderTimer === null && drainResolver) {
        const resolver = drainResolver;
        drainResolver = null;
        resolver();
      }
    };

    const appendRenderedContent = (delta: string) => {
      if (!delta) return;
      renderedContent += delta;
      appendAssistantContent(renderedContent);
    };

    const forceFlushRenderQueue = (commitRemaining: boolean) => {
      if (renderTimer !== null) {
        clearTimeout(renderTimer);
        renderTimer = null;
      }
      if (commitRemaining && renderQueue.length > 0) {
        const tail = renderQueue.join("");
        renderQueue.length = 0;
        appendRenderedContent(tail);
      } else {
        renderQueue.length = 0;
      }
      resolveDrainIfIdle();
    };

    const scheduleRenderTick = () => {
      if (renderTimer !== null) return;
      renderTimer = setTimeout(() => {
        renderTimer = null;
        if (abortController.signal.aborted) {
          forceFlushRenderQueue(false);
          return;
        }
        if (renderQueue.length === 0) {
          resolveDrainIfIdle();
          return;
        }

        const frameBudget = 32;
        let remainingBudget = frameBudget;
        let chunk = "";
        while (remainingBudget > 0 && renderQueue.length > 0) {
          const head = renderQueue[0];
          const take = Math.min(head.length, remainingBudget);
          chunk += head.slice(0, take);
          remainingBudget -= take;
          if (take === head.length) {
            renderQueue.shift();
          } else {
            renderQueue[0] = head.slice(take);
          }
          if (/[，。！？.!?\n]$/.test(chunk) && chunk.length >= 12) {
            break;
          }
        }

        appendRenderedContent(chunk);
        if (renderQueue.length > 0) {
          scheduleRenderTick();
        } else {
          resolveDrainIfIdle();
        }
      }, 16);
    };

    const queueTextForRender = (text: string) => {
      if (!text) return;
      renderQueue.push(text);
      scheduleRenderTick();
    };

    const drainRenderQueue = async () => {
      if (renderQueue.length === 0 && renderTimer === null) return;
      await new Promise<void>((resolve) => {
        const prevResolver = drainResolver;
        drainResolver = () => {
          if (prevResolver) prevResolver();
          resolve();
        };
        scheduleRenderTick();
      });
    };

    try {
      const extra = (extraBody || {}) as Record<string, unknown>;
      const resolvedExamFastMode =
        typeof extra.exam_fast_mode === "boolean" ? Boolean(extra.exam_fast_mode) : examFastMode;
      const resolvedQuizForceCritic =
        typeof extra.quiz_force_llm_critic === "boolean"
          ? Boolean(extra.quiz_force_llm_critic)
          : !resolvedExamFastMode;
      const requestBody = {
        query: outgoingText,
        session_id: currentSession,
        exam_fast_mode: resolvedExamFastMode, // 快速/完整模式开关
        quiz_force_llm_critic: resolvedQuizForceCritic, // quiz 与模式联动：完整=强制 LLM Critic
        ...extra,
      };
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
        signal: abortController.signal,
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let sseBuffer = "";
      let streamErrored = false;

      const processSSEData = (data: string, eventName?: string) => {
        // 统一处理后端标准事件与代理透传事件。
        if (!data || data === "[DONE]") return;

        try {
          const parsed = JSON.parse(data) as Record<string, unknown>;
          const semanticEvent =
            (typeof parsed.event === "string" ? parsed.event : eventName) || "";
          const errorValue = parsed.error;
          const parsedPayload =
            parsed.payload && typeof parsed.payload === "object"
              ? (parsed.payload as Record<string, unknown>)
              : null;

          // 心跳用于观测链路是否活着，不参与消息渲染。
          if (semanticEvent === "heartbeat") {
            heartbeatCount += 1;
            return;
          }
          if (semanticEvent === "done") {
            const doneStatus =
              typeof parsed.status === "string"
                ? parsed.status
                : typeof parsedPayload?.status === "string"
                  ? parsedPayload.status
                  : "";
            if (doneStatus.trim()) {
              streamTerminationReason = doneStatus.trim();
            }
            return;
          }

          if (semanticEvent === "error" || errorValue !== undefined) {
            streamErrored = true;
            shouldDrainRenderQueue = false;
            const errorText = typeof errorValue === "string" ? errorValue : "服务暂时不可用";
            streamTerminationReason =
              /超时|timeout/i.test(errorText) ? "timeout" : "error";
            markTtft();
            forceFlushRenderQueue(false);
            renderedContent = `[系统提示: ${errorText}]`;
            fullContent = renderedContent;
            setMessages(prev => prev.map(m =>
              m.id === assistantMessageId
                ? { ...m, content: `[系统提示: ${errorText}]`, meta: { error: true, error_message: errorText } }
                : m
            ));
            return;
          }

          const parsedMessage =
            parsed.message && typeof parsed.message === "object"
              ? (parsed.message as Record<string, unknown>)
              : parsedPayload?.message && typeof parsedPayload.message === "object"
                ? (parsedPayload.message as Record<string, unknown>)
                : null;

          if (semanticEvent === "start" && parsedMessage) {
            const parsedMeta =
              parsedMessage.meta && typeof parsedMessage.meta === "object"
                ? (parsedMessage.meta as Record<string, unknown>)
                : {};
            const routeTag = String(parsedMeta.route || "");
            if (routeTag !== "pending") {
              seenNonPendingStart = true;
              markTtft();
            }
            setMessages(prev => prev.map(m =>
              m.id === assistantMessageId
                ? {
                    ...m,
                    kind: (parsedMessage.kind as MessageKind) || "chat",
                    render_mode: (parsedMessage.render_mode as Message["render_mode"]) || "markdown",
                    meta: parsedMeta || m.meta,
                  }
                : m
            ));
            return;
          }

          const parsedText = typeof parsed.text === "string" ? parsed.text : "";
          const parsedDeltaText =
            parsedText ||
            (typeof parsedPayload?.text === "string" ? parsedPayload.text : "");
          if (semanticEvent === "text_delta") {
            seenTextDeltaEvent = true;
          }
          // V2 灰度期后端可能镜像 legacy delta；前端优先消费 text_delta，避免正文重复。
          if (semanticEvent === "delta" && seenTextDeltaEvent) {
            return;
          }
          if ((semanticEvent === "delta" || semanticEvent === "text_delta") && parsedDeltaText) {
            // 只统计“业务正文 delta”的首包延迟，排除 pending 占位。
            if (seenNonPendingStart && firstDeltaMs === null) {
              firstDeltaMs = Date.now() - startTime;
            }
            if (seenNonPendingStart) {
              markTtft();
            }
            fullContent += parsedDeltaText;
            queueTextForRender(parsedDeltaText);
            return;
          }

          if (semanticEvent === "reasoning_delta") {
            const text = typeof parsedPayload?.text === "string" ? parsedPayload.text : "";
            if (text) {
              liveSections = { ...liveSections, reasoning: `${liveSections.reasoning}${text}` };
              updateAssistantSections(liveSections);
            }
            return;
          }

          if (semanticEvent === "progress") {
            const progressText =
              typeof parsedPayload?.message === "string" ? parsedPayload.message : "";
            if (progressText) {
              liveSections = {
                ...liveSections,
                progress: [...liveSections.progress, progressText],
              };
              updateAssistantSections(liveSections);
            }
            return;
          }

          if (semanticEvent === "citation") {
            const citation = parsedPayload || parsed;
            liveSections = {
              ...liveSections,
              citations: [...liveSections.citations, citation as StreamSectionCitation],
            };
            updateAssistantSections(liveSections);
            return;
          }

          if (semanticEvent === "action" || semanticEvent === "action_result") {
            const action = parsedPayload || parsed;
            liveSections = {
              ...liveSections,
              actions: [...liveSections.actions, action as StreamSectionAction],
            };
            updateAssistantSections(liveSections);
            return;
          }

          if (semanticEvent === "complete" && parsedMessage) {
            const messageKind = (parsedMessage.kind as MessageKind) || "chat";
            const messageContent =
              typeof parsedMessage.content === "string" ? parsedMessage.content : fullContent;
            fullContent = messageContent || fullContent;
            const parsedPayloadObj =
              parsedMessage.payload && typeof parsedMessage.payload === "object"
                ? (parsedMessage.payload as AssistantPayload)
                : undefined;
            const completeSections = normalizeStreamSections(parsedPayloadObj?.stream_sections);
            if (hasStreamSections(completeSections)) {
              liveSections = completeSections;
            }
            if (messageKind === "chat") {
              setMessages(prev => prev.map(m =>
                m.id === assistantMessageId
                  ? {
                      ...m,
                      kind: messageKind,
                      render_mode: (parsedMessage.render_mode as Message["render_mode"]) || "markdown",
                      payload: parsedPayloadObj
                        ? {
                            ...parsedPayloadObj,
                            stream_sections: hasStreamSections(completeSections) ? completeSections : liveSections,
                          }
                        : {
                            stream_sections: liveSections,
                          },
                      meta: parsedMessage.meta as Record<string, unknown> | undefined,
                    }
                  : m
              ));

              if (messageContent) {
                if (messageContent.startsWith(renderedContent)) {
                  const remain = messageContent.slice(renderedContent.length);
                  if (remain) queueTextForRender(remain);
                } else {
                  forceFlushRenderQueue(false);
                  renderedContent = "";
                  queueTextForRender(messageContent);
                }
              }
            } else {
              // 结构化消息（quiz/exam）直接整块落地，避免半包解析闪烁。
              forceFlushRenderQueue(false);
              renderedContent = messageContent || fullContent;
              setMessages(prev => prev.map(m =>
                m.id === assistantMessageId
                  ? {
                      ...m,
                      content: renderedContent,
                      kind: messageKind,
                      render_mode: (parsedMessage.render_mode as Message["render_mode"]) || "markdown",
                      payload: parsedPayloadObj,
                      meta: parsedMessage.meta as Record<string, unknown> | undefined,
                    }
                  : m
              ));
            }
            return;
          }

          if (parsedText) {
            // 跳过 think 标签内容
            if (parsedText.trim().startsWith("<think>")) {
              return;
            }
            // 检测系统思考
            if (parsedText.includes("**[系统思考")) {
              const match = parsedText.match(/\*\*(.*?)\*\*/);
              if (match) {
                const thought = match[1].replace("[系统思考：", "").replace("]", "");
                setThoughts(prev => [...prev, thought]);
              }
            } else {
              fullContent += parsedText;
              queueTextForRender(parsedText);
            }
            return;
          }

          // 有 event 但 data 里不是标准对象时忽略，保持前向兼容
        } catch {
          if (eventName === "text_delta") {
            seenTextDeltaEvent = true;
          }
          if (eventName === "delta" && seenTextDeltaEvent) {
            return;
          }
          if ((eventName === "delta" || eventName === "text_delta") && data) {
            // 兼容纯文本 delta
            if (seenNonPendingStart && firstDeltaMs === null) {
              firstDeltaMs = Date.now() - startTime;
            }
            if (seenNonPendingStart) {
              markTtft();
            }
            fullContent += data;
            queueTextForRender(data);
            return;
          }
          if (eventName === "error") {
            streamErrored = true;
            shouldDrainRenderQueue = false;
            const errorText = data || "服务暂时不可用";
            streamTerminationReason =
              /超时|timeout/i.test(errorText) ? "timeout" : "error";
            markTtft();
            forceFlushRenderQueue(false);
            renderedContent = `[系统提示: ${errorText}]`;
            fullContent = renderedContent;
            setMessages(prev => prev.map(m =>
              m.id === assistantMessageId
                ? { ...m, content: `[系统提示: ${errorText}]`, meta: { error: true, error_message: errorText } }
                : m
            ));
          }
        }
      };

      if (reader) {
        while (true) {
          if (abortController.signal.aborted) break;
          const { done, value } = await reader.read();
          if (done) break;

          // 按帧解析，避免“按行解析”在多行 data 场景下丢内容。
          sseBuffer += decoder.decode(value, { stream: true });
          const parsed = parseSSEFrames(sseBuffer);
          sseBuffer = parsed.rest;

          for (const frame of parsed.frames) {
            processSSEData(frame.data, frame.event);
            if (streamErrored) break;
          }
          if (streamErrored) {
            break;
          }
        }

        // 处理最后一个可能未双换行结束的片段
        sseBuffer += decoder.decode();
        if (sseBuffer.trim()) {
          const parsedTail = parseSSEFrames(`${sseBuffer}\n\n`);
          for (const frame of parsedTail.frames) {
            processSSEData(frame.data, frame.event);
            if (streamErrored) break;
          }
        }

        if (!streamErrored && !fullContent.trim()) {
          setMessages(prev => prev.map(m =>
            m.id === assistantMessageId && !String(m.content || "").trim()
              ? { ...m, content: "服务已返回空内容，请重试一次。" }
              : m
          ));
        }

        try {
          await reader.cancel();
        } catch {}
      } else {
        shouldDrainRenderQueue = false;
        forceFlushRenderQueue(false);
        setMessages(prev => prev.map(m =>
          m.id === assistantMessageId ? { ...m, content: "流式连接不可用，请稍后重试。" } : m
        ));
      }

    } catch (error) {
      const aborted =
        abortController.signal.aborted ||
        (error instanceof DOMException && error.name === "AbortError");
      if (aborted) {
        streamTerminationReason = "client_cancelled";
        shouldDrainRenderQueue = false;
        forceFlushRenderQueue(false);
        const visibleContent = renderedContent.trim();
        setMessages(prev => prev.map(m =>
          m.id === assistantMessageId
            ? {
                ...m,
                content: visibleContent || "已停止生成。",
                meta: { ...(m.meta || {}), cancelled: true },
              }
            : m
        ));
        fullContent = visibleContent;
      } else {
        streamTerminationReason = "error";
        shouldDrainRenderQueue = false;
        forceFlushRenderQueue(false);
        console.error("Chat error:", error);
        const errorMessage = error instanceof Error ? error.message : String(error);
        setMessages(prev => prev.map(m =>
          m.id === assistantMessageId
            ? { ...m, content: `请求失败: ${errorMessage}`, meta: { error: true, error_message: errorMessage } }
            : m
        ));
        fullContent = `请求失败: ${errorMessage}`;
      }
    } finally {
      if (shouldDrainRenderQueue) {
        await drainRenderQueue();
      } else {
        forceFlushRenderQueue(false);
      }
      const finalDurationMs = Date.now() - startTime;
      setResponseTime(finalDurationMs / 1000);
      setTotalTokens(prev => prev + Math.ceil(fullContent.length / 4));
      setStreamDiagnostics({
        ttft_ms: ttftMs,
        first_delta_ms: firstDeltaMs,
        stream_duration_ms: finalDurationMs,
        termination_reason: streamTerminationReason,
        heartbeat_count: heartbeatCount,
      });
      if (streamAbortRef.current === abortController) {
        streamAbortRef.current = null;
      }
      setIsLoading(false);
    }
  };

  const retryAssistantMessage = async (assistantIndex: number) => {
    if (isLoading) return;
    const assistantMessage = messages[assistantIndex];
    for (let i = assistantIndex - 1; i >= 0; i -= 1) {
      if (messages[i]?.role === "user" && messages[i]?.content?.trim()) {
        setRetryingMessageId(assistantMessage?.id || null);
        try {
          await sendMessage(messages[i].content);
        } finally {
          setRetryingMessageId(null);
        }
        return;
      }
    }
    showToast("未找到可重试的问题内容", "warning");
  };

  const getExamStageTrace = (meta?: Record<string, unknown>): ExamStageTraceItem[] => {
    if (!meta || !Array.isArray(meta.stage_trace)) return [];
    return meta.stage_trace.filter((item): item is ExamStageTraceItem => !!item && typeof item === "object");
  };

  const retryFailedExamStage = async (assistantIndex: number) => {
    if (isLoading) return;
    const assistantMessage = messages[assistantIndex];
    if (!assistantMessage) return;
    const meta = assistantMessage.meta as Record<string, unknown> | undefined;
    const stageTrace = getExamStageTrace(meta);
    const failedStage = stageTrace.find((s) => s.stage_status === "failed")?.exam_stage;
    if (!failedStage) {
      showToast("没有可重跑的失败阶段", "warning");
      return;
    }

    let previousUserText = "";
    for (let i = assistantIndex - 1; i >= 0; i -= 1) {
      if (messages[i]?.role === "user" && messages[i]?.content?.trim()) {
        previousUserText = messages[i].content;
        break;
      }
    }
    if (!previousUserText) {
      showToast("未找到原始出卷请求，无法重跑失败段", "warning");
      return;
    }

    const questions = ((((assistantMessage.payload || {}) as AssistantPayload).exam_data as QuizExamData | undefined)?.questions || []);
    const stageTypeMap: Record<string, string[]> = {
      choice: ["选择题"],
      fill_judge: ["填空题", "判断题"],
      essay: ["简答题"],
    };
    const failedTypes = new Set(stageTypeMap[failedStage] || []);
    const partialQuestions = questions.filter((q) => !failedTypes.has(String(q.type || "")));

    setRetryingMessageId(assistantMessage.id || null);
    try {
      await sendMessage(previousUserText, {
        exam_stage_plan: true,
        exam_rerun_stage: failedStage,
        exam_partial_questions: partialQuestions,
        exam_fast_mode: examFastMode,
      });
    } finally {
      setRetryingMessageId(null);
    }
  };

  // 获取 Token 统计
  const fetchTokenStats = useCallback(async () => {
    try {
      const res = await fetch("/api/chat/tokens");
      const data = await res.json();
      if (data.code === 200 && data.data) {
        setTotalTokens(data.data.total_tokens || 0);
      }
    } catch (error) {
      console.error("Failed to fetch token stats:", error);
    }
  }, []);

  // 获取流式运行指标（后端聚合）
  const fetchRuntimeStreamMetrics = useCallback(async () => {
    try {
      const res = await fetch("/api/chat/metrics");
      const data = await res.json();
      if (data?.code === 200 && data?.data) {
        const metricData = data.data as Record<string, unknown>;
        setRuntimeStreamMetrics({
          stream_ttft_p95_ms: Number(metricData.stream_ttft_p95_ms || 0),
          stream_first_delta_p95_ms: Number(metricData.stream_first_delta_p95_ms || 0),
          stream_duration_p95_ms: Number(metricData.stream_duration_p95_ms || 0),
          stream_client_cancel_total: Number(metricData.stream_client_cancel_total || 0),
          stream_timeout_total: Number(metricData.stream_timeout_total || 0),
          stream_error_total: Number(metricData.stream_error_total || 0),
        });
      }
    } catch (error) {
      console.error("Failed to fetch runtime metrics:", error);
    }
  }, []);

  // 定期刷新统计
  useEffect(() => {
    fetchTokenStats();
    fetchRuntimeStreamMetrics();
    const interval = setInterval(() => {
      void fetchTokenStats();
      void fetchRuntimeStreamMetrics();
    }, 30000); // 每30秒刷新
    return () => clearInterval(interval);
  }, [fetchTokenStats, fetchRuntimeStreamMetrics]);

  // 创建会话
  const createSession = async (name: string, parentId?: string) => {
    const sessionId = parentId ? `${parentId}_sub_${Date.now()}` : name;
    try {
      const res = await fetch("/api/chat/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, name, parent_id: parentId })
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      if (data.code === 200) {
        const newSession = { id: sessionId, name, parent_id: parentId };
        setSessions(prev => [...prev, newSession]);
        setCurrentSession(sessionId);
        setCurrentSessionName(name);
        setMessages([]);
        setThoughts([]);
        showToast("会话创建成功", "success");
      } else {
        throw new Error(data?.message || "会话创建失败");
      }
    } catch (error) {
      console.error("Create session error:", error);
      const msg = error instanceof Error ? error.message : "请检查后端连接";
      showToast(`会话创建失败：${msg}`, "error");
    }
  };

  // 删除会话
  const deleteSession = async (id: string) => {
    try {
      const res = isValidSessionId(id)
        ? await fetch(`/api/chat/session/${id}`, { method: "DELETE" })
        : await fetch(`/api/chat/session/cleanup`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: id }),
          });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        if (res.status === 404) {
          throw new Error(data?.message || "会话不存在或已删除");
        }
        if (res.status === 400) {
          throw new Error(data?.message || "会话ID非法");
        }
        throw new Error(data?.message || `HTTP ${res.status}`);
      }
      if (data?.code !== 200) {
        throw new Error(data?.message || "删除失败");
      }

      setSessions(prev => prev.filter(s => s.id !== id));
      if (currentSession === id) {
        setCurrentSession("default");
        setCurrentSessionName("默认科目");
        setMessages([]);
        setThoughts([]);
      }
      showToast("会话已删除", "success");
    } catch (error) {
      console.error("Delete session error:", error);
      const msg = error instanceof Error ? error.message : "删除失败";
      showToast(`删除会话失败：${msg}`, "error");
    }
  };

  // 重命名会话
  const renameSession = async (id: string, newName: string) => {
    try {
      if (!isValidSessionId(id)) {
        showToast("该会话ID非法，无法重命名，请先删除该会话", "warning");
        return;
      }
      const res = await fetch(`/api/chat/session/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_name: newName })
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      if (data?.code !== 200) {
        throw new Error(data?.message || "重命名失败");
      }

      setSessions(prev => prev.map(s => s.id === id ? { ...s, name: newName } : s));
      if (currentSession === id) {
        setCurrentSessionName(newName);
      }
      showToast("会话重命名成功", "success");
    } catch (error) {
      console.error("Rename session error:", error);
      const msg = error instanceof Error ? error.message : "重命名失败";
      showToast(`重命名失败：${msg}`, "error");
    }
  };

  // 销毁记忆
  const clearMemory = async () => {
    if (!confirm("确定要销毁当前会话的记忆吗？此操作不可恢复。")) return;
    try {
      const res = isValidSessionId(currentSession)
        ? await fetch(`/api/chat/session/${currentSession}`, { method: "DELETE" })
        : await fetch(`/api/chat/session/cleanup`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: currentSession }),
          });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      if (data?.code !== 200) {
        throw new Error(data?.message || "销毁失败");
      }
      setMessages([]);
      setThoughts([]);
      setTotalTokens(0);
      setResponseTime(0);
      showToast("当前会话记忆已销毁", "success");
    } catch (error) {
      console.error("Clear memory error:", error);
      const msg = error instanceof Error ? error.message : "销毁失败";
      showToast(`销毁记忆失败：${msg}`, "error");
    }
  };

  /** 将毫秒值转为可读文本，空值时输出占位符。 */
  const formatMsMetric = (value: number | null): string => (value !== null ? `${value}ms` : "--");

  /** 数值使用本地化分组，避免长数字难以扫描。 */
  const formatCountMetric = (value: number): string => value.toLocaleString("zh-CN");

  const terminationValueClassName =
    streamDiagnostics.termination_reason === "success"
      ? "text-emerald-600"
      : ["error", "timeout", "client_cancel", "client_cancelled"].includes(streamDiagnostics.termination_reason)
        ? "text-rose-600"
        : "text-amber-600";

  const primaryRuntimeMetrics: RuntimeMetricItem[] = [
    { label: "Tokens", value: formatCountMetric(totalTokens), valueClassName: "text-slate-800" },
    { label: "延迟", value: `${responseTime.toFixed(2)}s`, valueClassName: "text-cyan-700" },
    { label: "TTFT", value: formatMsMetric(streamDiagnostics.ttft_ms), valueClassName: "text-emerald-700" },
    { label: "终止", value: streamDiagnostics.termination_reason, valueClassName: terminationValueClassName },
  ];

  const detailRuntimeMetrics: RuntimeMetricItem[] = [
    { label: "首 Delta", value: formatMsMetric(streamDiagnostics.first_delta_ms), valueClassName: "text-slate-700" },
    { label: "流时长", value: `${streamDiagnostics.stream_duration_ms}ms`, valueClassName: "text-slate-700" },
    { label: "心跳数", value: `${streamDiagnostics.heartbeat_count}`, valueClassName: "text-slate-700" },
    {
      label: "TTFT P95",
      value: runtimeStreamMetrics ? `${runtimeStreamMetrics.stream_ttft_p95_ms}ms` : "--",
      valueClassName: "text-violet-700",
    },
    {
      label: "流时长 P95",
      value: runtimeStreamMetrics ? `${runtimeStreamMetrics.stream_duration_p95_ms}ms` : "--",
      valueClassName: "text-violet-700",
    },
  ];

  /** 切换出题模式并给出轻提示。 */
  const toggleExamMode = () => {
    const wasFast = examFastMode;
    setExamFastMode(prev => !prev);
    showToast(
      wasFast
        ? "已关闭快速模式，将启用完整 LLM Critique"
        : "已开启快速模式，将跳过内容审查",
      "info"
    );
  };

  /** 组装综合测试卷提示词，统一命题口径并提升输出质量。 */
  const buildComprehensiveExamPrompt = (sampleHint: string): string => {
    return `你是资深课程命题教师，请基于已上传课件生成一套“综合测试卷”。

命题目标：
- 覆盖课件核心知识点，检验理解、应用与综合分析能力。
- 题目表述清晰，避免歧义、重复和超纲内容。

试卷结构要求：
1. 试卷标题与考试说明（建议作答时长、总分）。
2. 题型与数量：
   - 选择题 10 道
   - 判断题 5 道
   - 填空题 5 道
   - 简答题 3 道
3. 难度梯度建议：基础 60%，提升 30%，挑战 10%。
4. 每道题都要给出：题干、标准答案、解析（解析点明对应知识点/考点）。

命题约束：
- 术语与结论优先采用课件原有表述。
- 如果课件信息不足，不要编造；用“依据不足”并给出保守表达。
${sampleHint}

输出顺序（必须）：
先输出完整试卷题目，再单独输出“答案与解析”部分，编号一一对应。`;
  };

  return (
    <div className="chat-shell h-screen w-full flex flex-col bg-slate-50 overflow-hidden font-sans">
      {/* 后端未连接警告 */}
      {!isBackendConnected && (
        <div className="bg-amber-500 text-white text-center py-2 text-sm font-medium">
          ⚠️ 后端未连接，请启动 FastAPI: python api/main.py
        </div>
      )}

      {/* ============== 顶部导航栏 ============== */}
      <header className="flex-none px-4 md:px-6 py-3 flex items-center justify-between border-b border-slate-200/80 bg-white/75 backdrop-blur-xl z-20 shadow-[0_10px_30px_rgba(15,23,42,0.04)]">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-slate-900 text-white rounded flex items-center justify-center font-mono font-bold text-sm shadow-sm ring-1 ring-slate-900">
            DR
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight text-slate-900 uppercase">
              Deep<span className="text-teal-600">Revision</span>
            </h1>
          </div>
        </div>

        {/* 算力核心统计 */}
        <div className="hidden xl:flex flex-1 mx-4 max-w-2xl">
          <div className="group relative w-full rounded-xl border border-slate-200/90 bg-white/90 px-3 py-2 shadow-[0_8px_20px_rgba(15,23,42,0.06)] backdrop-blur-sm">
            <div className="flex items-center gap-3">
              <span className="w-1.5 h-1.5 rounded-full bg-teal-500 animate-pulse"></span>
              <div className="grid flex-1 grid-cols-4 gap-x-2">
                {primaryRuntimeMetrics.map(metric => (
                  <div key={metric.label} className="min-w-0">
                    <p className="truncate text-[0.56rem] font-mono uppercase tracking-wide text-slate-400">{metric.label}</p>
                    <p className={`truncate text-[0.72rem] font-semibold font-mono tracking-tight ${metric.valueClassName ?? "text-slate-700"}`}>
                      {metric.value}
                    </p>
                  </div>
                ))}
              </div>
              <button
                type="button"
                aria-label="查看运行指标详情"
                className="h-6 w-6 rounded-md border border-slate-200 text-slate-400 transition-colors hover:text-slate-600 hover:border-slate-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500/40"
              >
                <svg className="mx-auto h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </button>
            </div>
            <div className="pointer-events-none absolute right-0 top-[calc(100%+8px)] z-30 w-[330px] translate-y-1 rounded-xl border border-slate-200 bg-white/95 p-3 opacity-0 shadow-[0_16px_30px_rgba(15,23,42,0.12)] backdrop-blur-sm transition-all duration-150 group-hover:translate-y-0 group-hover:opacity-100 group-focus-within:translate-y-0 group-focus-within:opacity-100">
              <p className="mb-2 text-[0.62rem] font-mono uppercase tracking-[0.14em] text-slate-500">详细指标</p>
              <div className="grid grid-cols-2 gap-x-3 gap-y-2">
                {detailRuntimeMetrics.map(metric => (
                  <div key={metric.label} className="min-w-0">
                    <p className="truncate text-[0.58rem] font-mono uppercase tracking-wide text-slate-400">{metric.label}</p>
                    <p className={`truncate text-[0.7rem] font-semibold font-mono ${metric.valueClassName ?? "text-slate-700"}`}>
                      {metric.value}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Header Actions */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowSessionModal(true)}
            className="group relative px-4 py-1.5 text-xs font-mono font-medium text-slate-600 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md transition-all flex items-center gap-2"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            <span className="truncate max-w-[100px]">{currentSessionName}</span>
          </button>
          <button
            onClick={() => setShowKnowledgePanel(true)}
            className="group relative px-4 py-1.5 text-xs font-medium text-slate-600 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md transition-all ease-out duration-200"
          >
            上传课件
          </button>
          <button
            onClick={() => setShowHistoryPanel(true)}
            title="历史记录管理"
            className="group relative px-3 py-1.5 text-xs font-medium text-slate-600 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md transition-all ease-out duration-200"
          >
            <span className="flex items-center gap-1.5">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              历史
            </span>
          </button>
          {/* 出题按钮组 */}
          <div className="relative group">
            <button
              onClick={async () => {
                if (!currentSession || isLoading) return;
                let sampleInfo = "";
                try {
                  const sampleRes = await fetch(`/api/knowledge/sample?session_id=${encodeURIComponent(currentSession)}`);
                  const sampleData = await sampleRes.json();
                  if (sampleData.code === 200 && sampleData.data?.has_sample) {
                    sampleInfo = "- 参考已上传样卷的题型分布、分值设置与编号风格。";
                  }
                } catch (e) {}
                const prompt = buildComprehensiveExamPrompt(sampleInfo);
                await sendMessage(prompt, { exam_stage_plan: true, exam_fast_mode: examFastMode });
              }}
              className="px-4 py-1.5 text-xs font-medium text-teal-700 bg-teal-50 hover:bg-teal-100 border border-teal-200 rounded-md transition-all flex items-center gap-1"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              综合测试卷
            </button>
          </div>
          <button
            onClick={clearMemory}
            className="px-4 py-1.5 text-xs font-medium text-red-600 bg-red-50 hover:bg-red-100 border border-red-100 hover:border-red-200 rounded-md transition-all focus:outline-none"
          >
            销毁记忆
          </button>
        </div>
      </header>

      {/* ============== 主内容区 ============== */}
      <main className="chat-main flex-1 w-full relative overflow-hidden flex">
        {/* 聊天区域 */}
        <div className="flex-1 flex flex-col overflow-hidden relative z-10">
          {/* 消息列表 */}
          <div className="flex-1 overflow-y-auto p-4 md:p-8 lg:p-10">
            <div className="max-w-5xl mx-auto conversation-lane">
              {messages.length === 0 && (
                <div className="w-full flex items-start gap-4 animate-slideUp">
                  <div className="w-6 h-6 rounded bg-teal-100 text-teal-700 flex items-center justify-center flex-shrink-0 mt-1 border border-teal-200">
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <div className="text-[0.65rem] font-mono text-slate-400 uppercase tracking-wider mb-1">系统就绪</div>
                    <div className="prose-custom text-slate-800">
                      <p>环境已就绪。我是您的 <strong>DeepRevision</strong> 复习引擎。</p>
                      <p>请上传课件作为知识锚点，随后可以直接向我提问，或要求我为您生成定制化测试题。</p>
                    </div>
                  </div>
                </div>
              )}

              {messages.map((message, idx) => (
                <div
                  key={`${message.id}-${idx}`}
                  className={`w-full flex items-start gap-4 message-enter message-row ${message.role === "user" ? "flex-row-reverse" : ""}`}
                  style={{ animationDelay: `${idx * 0.05}s` }}
                >
                  {/* Avatar - 精致圆形头像 */}
                  {message.role === "user" ? (
                    <div className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 mt-1 shadow-sm"
                      style={{
                        background: 'linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%)',
                        border: '2px solid rgba(255,255,255,0.8)'
                      }}>
                      <span className="font-medium text-xs" style={{ color: '#64748b' }}>我</span>
                    </div>
                  ) : (
                    <div className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 mt-1 shadow-md"
                      style={{
                        background: 'linear-gradient(135deg, #0d9488 0%, #0f766e 100%)',
                        border: '2px solid rgba(255,255,255,0.8)'
                      }}>
                      <svg className="w-4 h-4" fill="none" stroke="white" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                    </div>
                  )}

                  <div className={`flex-1 max-w-3xl ${message.role === "user" ? "text-right" : ""}`}>
                    <div className="text-[0.65rem] font-semibold mb-2 flex items-center gap-2" style={{ color: '#a0a0a0', letterSpacing: '0.05em' }}>
                      {message.role === "user" ? (
                        <>
                          <span style={{ color: '#6b7280' }}>你</span>
                          <span style={{ opacity: 0.5 }}>·</span>
                          <span style={{ color: '#a0a0a0' }}>{new Date(message.timestamp).toLocaleTimeString()}</span>
                        </>
                      ) : (
                        <>
                          <span style={{ color: '#0d9488', fontWeight: 600 }}>DeepRevision</span>
                          <span style={{ opacity: 0.5 }}>·</span>
                          <span style={{ color: '#a0a0a0' }}>AI 助手</span>
                        </>
                      )}
                    </div>

                    {message.role === "assistant" && thoughts.length > 0 && (
                      <ThoughtChain thoughts={thoughts} />
                    )}

                    <div className={`inline-block px-5 py-3.5 rounded-2xl shadow-sm message-bubble ${
                      message.role === "user"
                        ? "rounded-tr-md"
                        : "rounded-tl-md border"
                    } ${message.role === "assistant" && (message.kind ?? "chat") === "chat" ? "chat-message" : ""}`} style={{
                      background: message.role === "user"
                        ? '#f1f5f9'
                        : ((message.kind ?? "chat") === "chat"
                          ? 'linear-gradient(145deg, rgba(255,255,255,0.98) 0%, rgba(244,252,251,0.96) 100%)'
                          : 'rgba(255,255,255,0.9)'),
                      color: message.role === "user" ? '#334155' : '#1e293b',
                      borderColor: ((message.kind ?? "chat") === "chat")
                        ? 'rgba(13,148,136,0.18)'
                        : 'rgba(180,170,150,0.2)',
                      boxShadow: ((message.kind ?? "chat") === "chat")
                        ? '0 8px 24px rgba(15,118,110,0.08)'
                        : '0 2px 6px rgba(15,23,42,0.06)'
                    }}>
                      <MarkdownContent
                        content={message.content || (message.role === "assistant" && isLoading ? "正在思考中..." : "")}
                        kind={message.kind}
                        payload={message.payload}
                        similarQuestions={similarQuestionsByMessage[message.id]}
                        onPracticeComplete={(score, total, records, wrongAnswers, userAnswers) =>
                          handlePracticeCompleteForMessage(message.id, score, total, records, wrongAnswers, userAnswers)
                        }
                        onRequestAnswers={() => {
                          void sendMessage("请给出上面试卷的答案和解析");
                        }}
                      />
                      {message.role === "assistant" && (message.kind ?? "chat") === "chat" && (
                        <StreamSectionsPanel
                          sections={normalizeStreamSections((message.payload as AssistantPayload | undefined)?.stream_sections)}
                        />
                      )}
                    </div>
                    {message.role === "assistant" && (
                      <div className="mt-2 flex items-center gap-2">
                        {message.kind !== "exam_paper" && message.meta && (
                          <span className={`text-xs px-2 py-1 rounded border ${
                            (message.meta as Record<string, unknown>).degrade_reason
                              ? "bg-yellow-50 text-yellow-700 border-yellow-200"
                              : "bg-emerald-50 text-emerald-700 border-emerald-200"
                          }`}>
                            {(message.meta as Record<string, unknown>).degrade_reason
                              ? `降级交付：${String((message.meta as Record<string, unknown>).degrade_reason)}`
                              : "完整交付"}
                            {(message.meta as Record<string, unknown>).evidence_source
                              ? ` · ${String((message.meta as Record<string, unknown>).evidence_source)}`
                              : ""}
                          </span>
                        )}
                        {message.kind === "exam_paper" && message.meta && (
                          <span className={`text-xs px-2 py-1 rounded border ${
                            (message.meta as Record<string, unknown>).degrade_reason
                              ? "bg-yellow-50 text-yellow-700 border-yellow-200"
                              : "bg-emerald-50 text-emerald-700 border-emerald-200"
                          }`}>
                            {(message.meta as Record<string, unknown>).degrade_reason
                              ? `降级交付：${String((message.meta as Record<string, unknown>).degrade_reason)}`
                              : "完整交付"}
                            {typeof (message.meta as Record<string, unknown>).exam_end_to_end_ms === "number"
                              ? ` · ${String((message.meta as Record<string, unknown>).exam_end_to_end_ms)}ms`
                              : ""}
                          </span>
                        )}
                        {(() => {
                          const trace = getExamStageTrace(message.meta as Record<string, unknown> | undefined);
                          if (!trace.length) return null;
                          const labelMap: Record<string, string> = { choice: "选择", fill_judge: "填空判断", essay: "简答" };
                          const brief = trace
                            .map((s) => `${labelMap[s.exam_stage] || s.exam_stage}:${s.stage_status === "success" ? "成功" : "失败"}(x${s.attempt || 0})`)
                            .join(" | ");
                          return (
                            <span className="text-xs px-2 py-1 rounded border bg-slate-50 text-slate-700 border-slate-200">
                              {brief}
                            </span>
                          );
                        })()}
                        {(() => {
                          const trace = getExamStageTrace(message.meta as Record<string, unknown> | undefined);
                          const failed = trace.find((s) => s.stage_status === "failed" && s.error);
                          if (!failed) return null;
                          const labelMap: Record<string, string> = { choice: "选择题阶段", fill_judge: "填空判断阶段", essay: "简答题阶段" };
                          return (
                            <span className="text-xs px-2 py-1 rounded border bg-red-50 text-red-700 border-red-200">
                              {`${labelMap[failed.exam_stage] || failed.exam_stage}失败：${String(failed.error)}`}
                            </span>
                          );
                        })()}
                        {(() => {
                          const trace = getExamStageTrace(message.meta as Record<string, unknown> | undefined);
                          const hasFailed = trace.some((s) => s.stage_status === "failed");
                          if (!hasFailed) return null;
                          return (
                            <button
                              onClick={() => retryFailedExamStage(idx)}
                              disabled={isLoading}
                              className="text-xs px-2 py-1 rounded border border-slate-300 text-slate-600 hover:bg-slate-50 disabled:opacity-50"
                            >
                              {retryingMessageId === message.id ? "重跑中..." : "重跑失败段"}
                            </button>
                          );
                        })()}
                        {practiceFeedbackByMessage[message.id] && (
                          <span
                            className={`text-xs px-2 py-1 rounded border ${
                              practiceFeedbackByMessage[message.id].type === "success"
                                ? "bg-green-50 text-green-700 border-green-200"
                                : practiceFeedbackByMessage[message.id].type === "warning"
                                ? "bg-yellow-50 text-yellow-700 border-yellow-200"
                                : "bg-red-50 text-red-700 border-red-200"
                            }`}
                          >
                            {practiceFeedbackByMessage[message.id].text}
                          </span>
                        )}
                        {(message.content.includes("[系统提示:") ||
                          message.content.startsWith("请求失败:") ||
                          (message.meta && (message.meta as Record<string, unknown>).error === true)) && (
                          <button
                            onClick={() => retryAssistantMessage(idx)}
                            disabled={isLoading}
                            className="text-xs px-2 py-1 rounded border border-slate-300 text-slate-600 hover:bg-slate-50 disabled:opacity-50"
                          >
                            {retryingMessageId === message.id ? "重试中..." : "重试本轮"}
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* 输入框 - 精致日式风格 */}
          <div className="flex-none w-full pb-6 pt-4 px-4 composer-wrap">
            <div className="max-w-3xl mx-auto w-full relative">
              <div className="relative flex items-end gap-3 rounded-2xl transition-all duration-300 composer-shell">
                <div className="flex-1 relative">
                  <textarea
                    id="user-input"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
                    rows={1}
                    className="w-full bg-transparent max-h-40 py-3.5 px-4 text-sm resize-none focus:outline-none"
                    style={{
                      color: '#3d3d3d',
                      fontFamily: 'Georgia, "Times New Roman", serif',
                      minHeight: '52px'
                    }}
                    placeholder="在这里输入你想复习的内容..."
                    disabled={isLoading}
                  />
                </div>
                <button
                  id="send-btn"
                  onClick={() => {
                    if (isLoading) {
                      stopStreaming();
                    } else {
                      void sendMessage();
                    }
                  }}
                  disabled={!isLoading && !input.trim()}
                  className="mb-2 mr-2 p-3 rounded-xl transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed"
                  style={{
                    background: isLoading
                      ? 'linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%)'
                      : input.trim()
                        ? 'linear-gradient(135deg, #2d3436 0%, #636e72 100%)'
                        : 'rgba(200,200,200,0.3)',
                    boxShadow: isLoading
                      ? '0 2px 12px rgba(185,28,28,0.25)'
                      : input.trim()
                        ? '0 2px 12px rgba(45,52,54,0.3)'
                      : 'none'
                  }}
                >
                  {isLoading ? (
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24" aria-label="停止生成">
                      <rect x="6" y="6" width="12" height="12" rx="2" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4" fill="none" stroke="white" viewBox="0 0 24 24" style={{ opacity: input.trim() ? 1 : 0.5 }}>
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  )}
                </button>
              </div>
              <div className="mt-3 flex flex-wrap items-center justify-between gap-2 px-1 text-[11px] text-slate-400">
                <button
                  type="button"
                  role="switch"
                  aria-checked={examFastMode}
                  aria-label={examFastMode ? "当前快速模式，点击切换完整模式" : "当前完整模式，点击切换快速模式"}
                  onClick={toggleExamMode}
                  title={examFastMode ? "快速模式：跳过内容审查" : "完整模式：启用 LLM Critique"}
                  className={`relative h-5 w-9 rounded-full border p-0.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500/30 ${
                    examFastMode
                      ? "border-teal-400/70 bg-teal-500/85"
                      : "border-slate-300 bg-slate-200"
                  }`}
                >
                  <span
                    className={`absolute left-0.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 rounded-full bg-white shadow-[0_1px_2px_rgba(15,23,42,0.25)] transition-transform ${
                      examFastMode ? "translate-x-[18px]" : "translate-x-0"
                    }`}
                  />
                </button>
                <div className="flex items-center gap-3">
                  <span>Enter 发送</span>
                  <span className="opacity-50">·</span>
                  <span>Shift + Enter 换行</span>
                  {isLoading && (
                    <>
                      <span className="opacity-50">·</span>
                      <span>点击红色按钮停止</span>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* ============== 模态框 ============== */}
      <SessionModal
        isOpen={showSessionModal}
        onClose={() => setShowSessionModal(false)}
        sessions={sessions}
        currentSession={currentSession}
        onSelect={(id) => {
          const session = sessions.find(s => s.id === id);
          if (session) {
            setCurrentSession(id);
            setCurrentSessionName(session.name);
            setMessages([]);
            setThoughts([]);
            loadSessionMessages(id);  // 加载会话历史消息
          }
        }}
        onCreate={createSession}
        onDelete={deleteSession}
        onRename={renameSession}
      />

      <KnowledgePanel
        sessionId={currentSession}
        isOpen={showKnowledgePanel}
        onClose={() => setShowKnowledgePanel(false)}
      />
      <SessionHistoryPanel
        sessionId={currentSession}
        isOpen={showHistoryPanel}
        onClose={() => setShowHistoryPanel(false)}
      />
      <ToastContainer />

      {/* ============== 全局样式 ============== */}
      <style jsx global>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out forwards;
        }
        .animate-slideUp {
          animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        .border-l-3 {
          border-left-width: 3px;
        }
        ::-webkit-scrollbar {
          width: 4px;
          height: 4px;
        }
        ::-webkit-scrollbar-track {
          background: transparent;
        }
        ::-webkit-scrollbar-thumb {
          background: #e2e8f0;
          border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
          background: #cbd5e1;
        }
      `}</style>
    </div>
  );
}
