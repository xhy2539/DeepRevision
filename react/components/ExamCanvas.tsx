"use client";

import { useState, useRef, useEffect } from "react";

// 题目类型
interface Question {
  number: number;
  type: string;
  content: string;
  options?: string[];
  answer: string;
  analysis: string;
  score: number;
  difficulty: string;
  knowledge_point?: string;
}

// 解析后的试卷数据
interface ExamData {
  title: string;
  subtitle: string;
  total_score: number;
  total_questions: number;
  question_types: Record<string, { count: number; total_score: number }>;
  questions: Question[];
}

interface PracticeRecord {
  question: Question;
  userAnswer: string;
  isCorrect: boolean;
}

function stripLeadingQuestionNumber(text: string): string {
  const raw = (text || "").trim();
  if (!raw) return raw;
  // 兼容: "1. xxx" / "1、xxx" / "（1）xxx" / "(1) xxx"
  return raw
    .replace(/^\s*\d+\s*[.、]\s*/, "")
    .replace(/^\s*[（(]\s*\d+\s*[）)]\s*/, "")
    .trim();
}

function normalizeOptions(raw: unknown): string[] | undefined {
  const splitInlineOptions = (text: string): string[] => {
    const src = String(text || "").trim();
    if (!src) return [];
    const markerCount = (src.match(/[A-D][.、．:：)\s]+/g) || []).length;
    if (markerCount < 2) return [];
    const re = /([A-D])[.、．:：)\s]+\s*(.+?)(?=(?:\s+[A-D][.、．:：)\s]+)|$)/g;
    const out: string[] = [];
    let m: RegExpExecArray | null = null;
    while ((m = re.exec(src)) !== null) {
      const letter = (m[1] || "").toUpperCase();
      const content = (m[2] || "").trim();
      if (letter && content) out.push(`${letter}. ${content}`);
    }
    return out;
  };

  if (!raw) return undefined;
  if (Array.isArray(raw)) {
    const opts: string[] = [];
    for (const v of raw) {
      const text = String(v || "").trim();
      if (!text) continue;
      const inline = splitInlineOptions(text);
      if (inline.length > 0) {
        opts.push(...inline);
        continue;
      }
      if (/^[A-D][.、．:：)]\s*/.test(text)) {
        opts.push(text.replace(/^[A-D][.、．:：)]\s*/, (m) => `${m[0].toUpperCase().replace(/[、．:：)]/, ".")} `));
      } else {
        opts.push(text);
      }
    }
    const normalized = opts
      .map((text, i) => {
        const t = String(text || "").trim();
        if (!t) return "";
        if (/^[A-D][.、．:：)]\s*/.test(t)) {
          const letter = t.charAt(0).toUpperCase();
          const content = t.replace(/^[A-D][.、．:：)]\s*/, "").trim();
          return `${letter}. ${content}`;
        }
        return `${String.fromCharCode(65 + i)}. ${t}`;
      })
      .filter(Boolean);
    return normalized.length ? normalized : undefined;
  }
  if (typeof raw === "object") {
    const obj = raw as Record<string, unknown>;
    const keys = ["A", "B", "C", "D"];
    const opts = keys
      .map((k) => {
        const v = String(obj[k] || "").trim();
        return v ? `${k}. ${v}` : "";
      })
      .filter(Boolean);
    return opts.length ? opts : undefined;
  }
  return undefined;
}

function canonicalizeOptionLines(rawOptions: string[]): string[] {
  const cleaned: string[] = [];
  const seen = new Set<string>();
  for (const raw of rawOptions || []) {
    const text = String(raw || "").trim();
    if (!text) continue;
    const content = text.replace(/^[A-D][.、．:：)\s]+/, "").trim();
    if (!content) continue;
    const sig = content.replace(/\s+/g, " ").toLowerCase();
    if (seen.has(sig)) continue;
    seen.add(sig);
    cleaned.push(content);
    if (cleaned.length >= 4) break;
  }
  return cleaned.map((content, idx) => `${String.fromCharCode(65 + idx)}. ${content}`);
}

function extractOptionKey(optionText: string, fallbackIndex: number): string {
  const text = String(optionText || "").trim();
  const m = text.match(/^\s*([A-D])(?:[.、．:：)\s]|$)/i);
  if (m && m[1]) return m[1].toUpperCase();
  return String.fromCharCode(65 + fallbackIndex);
}

function stripOptionPrefix(optionText: string): string {
  return String(optionText || "")
    // 支持清理重复前缀：如 "A A xxx" / "A\nA xxx" / "A) A. xxx"
    .replace(/^(?:\s*[A-D](?:[.、．:：)\s]+|$))+/i, "")
    .trim();
}

// Props
interface ExamCanvasProps {
  examContent: string;
  examDataOverride?: unknown;
  courseName?: string;
  onExportWord?: (includeAnswers: boolean) => void;
  onExportAnswerSheet?: () => void;
  onRequestAnswers?: () => void;
  onRegenerateQuestion?: (questionNumber: number, questionType: string) => void;
  onRegenerateComplete?: (questionNumber: number) => void;
  onQuestionFeedback?: (questionNumber: number, feedback: 'up' | 'down') => void;
  onWrongAnswer?: (question: Question, userAnswer: string) => void;
  onPracticeComplete?: (
    score: number,
    total: number,
    records: PracticeRecord[],
    wrongAnswers: Question[],
    userAnswers: Record<number, string>
  ) => void;
  similarQuestions?: Record<number, Array<{question_content: string; answer: string; knowledge_point: string; question_id: string}>>;
}

// 尝试解析 JSON 格式的试卷（后端结构化输出）
function extractJSONObject(text: string): string {
  let braceCount = 0;
  let start = -1;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '{') {
      if (start === -1) start = i;
      braceCount++;
    } else if (c === '}') {
      braceCount--;
      if (braceCount === 0 && start !== -1) {
        return text.slice(start, i + 1);
      }
    }
  }
  let bracketCount = 0;
  start = -1;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '[') {
      if (start === -1) start = i;
      bracketCount++;
    } else if (c === ']') {
      bracketCount--;
      if (bracketCount === 0 && start !== -1) {
        return text.slice(start, i + 1);
      }
    }
  }
  return text.trim();
}

function tryParseJSONExam(content: string): ExamData | null {
  try {
    // 预检：如果内容不包含 "{" 或 "```" 很可能不是 JSON，直接返回 null
    const trimmed = content.trim();
    if (!trimmed.includes('{') && !trimmed.includes('```')) {
      return null;
    }

    console.log("[ExamCanvas] 尝试解析JSON，内容前200字:", trimmed.slice(0, 200));

    let jsonStr = trimmed;

    // 1. 优先尝试代码块包裹的 JSON
    let match = jsonStr.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (match) {
      jsonStr = match[1].trim();
    } else {
      jsonStr = extractJSONObject(jsonStr);
    }

    // 预检：提取后仍不包含 "{" 也不是 JSON
    if (!jsonStr.includes('{')) {
      return null;
    }

    const data = JSON.parse(jsonStr);
    console.log("[ExamCanvas] JSON解析成功，exam_paper长度:", data.exam_paper?.length, "questions数:", data.questions?.length);

    // 检查是否是结构化试卷格式
    if (data.questions && Array.isArray(data.questions) && data.questions.length > 0) {
      const result: ExamData = {
        title: data.title || "期末考试",
        subtitle: data.subtitle || "",
        total_score: 0,
        total_questions: 0,
        question_types: {},
        questions: []
      };

      // 转换题目格式
      for (const q of data.questions) {
        const question: Question = {
          number: q.id || q.number || 1,
          type: q.type || "选择题",
          content: stripLeadingQuestionNumber(q.content || ""),
          options: normalizeOptions(q.options),
          answer: q.answer || "",
          analysis: q.analysis || "",
          score: typeof q.score === 'number' ? q.score : (q.score_value || 5),
          difficulty: q.difficulty || q.difficult || "中等",
          knowledge_point: q.knowledge_point || undefined
        };

        result.questions.push(question);
        result.total_score += question.score;
      }

      console.log("[ExamCanvas] 解析成功，总分:", result.total_score);

      // 统计题型
      result.total_questions = result.questions.length;
      for (const q of result.questions) {
        if (!result.question_types[q.type]) {
          result.question_types[q.type] = { count: 0, total_score: 0 };
        }
        result.question_types[q.type].count++;
        result.question_types[q.type].total_score += q.score;
      }

      return result;
    }

    return null;
  } catch (e) {
    console.warn("[ExamCanvas] JSON解析失败:", e);
    return null;
  }
}

// 解析试卷内容
function parseExamContent(content: string): ExamData {
  // 先尝试解析 JSON 格式
  const jsonResult = tryParseJSONExam(content);
  if (jsonResult && jsonResult.questions.length > 0) {
    return jsonResult;
  }

  const result: ExamData = {
    title: "",
    subtitle: "",
    total_score: 0,
    total_questions: 0,
    question_types: {},
    questions: []
  };

  const lines = content.split('\n');
  let currentSection = null;
  let currentQuestion: Question | null = null;
  let globalQuestionNumber = 0;  // 统一编号计数器

  const typePatterns: Record<string, RegExp> = {
    // 选择题：标题包含"选择题"
    '选择题': /(一[、.]\s*)?选择题/,
    // 填空题：标题包含"填空题"
    '填空题': /(二[、.]\s*)?填空题/,
    // 判断题：标题包含"判断题"
    '判断题': /(三[、.]\s*)?判断题/,
    // 简答题：标题包含"简答题"或"解答题"
    '简答题': /(四[、.]\s*)?简答题|(四[、.]\s*)?解答题/,
    // 计算题
    '计算题': /(五[、.]\s*)?计算题/,
    // 分析题
    '分析题': /(六[、.]\s*)?分析题/,
    // 论述题
    '论述题': /(论述题)/,
    // 名词解释
    '名词解释': /(名词解释)/,
  };

  // 提取标题（放宽条件）
  for (let i = 0; i < Math.min(10, lines.length); i++) {
    const line = lines[i].trim();
    if (line && line.length > 2 && !result.title) {
      // 使用第一行作为标题（忽略题型匹配）
      result.title = line;
      break;
    }
  }

  // 尝试从内容中找副标题
  for (const line of lines.slice(0, 10)) {
    const trimmed = line.trim();
    if (trimmed.includes("期末") || trimmed.includes("试卷") || trimmed.includes("考试")) {
      result.subtitle = trimmed;
      break;
    }
  }

  // 用于去重的内容片段集合
  const seenQuestionSignatures = new Set<string>();

  // 解析题目
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    // 检测题型标题（放宽长度限制，支持 "一、选择题（共10题）" 这种格式）
    for (const [qtype, pattern] of Object.entries(typePatterns)) {
      if (pattern.test(trimmed) && trimmed.length < 50) {
        currentSection = qtype;
        currentQuestion = null;
        break;
      }
    }

    // 检测题目编号 - 支持统一编号如 "1." "2." 或 "(1)" 小问
    const qMatch = trimmed.match(/^(\d+)[.、]\s*(.+)/);
    const subQMatch = trimmed.match(/^[\(（](\d+)[\)）]\s*(.+)/);

    // 处理选择题选项（支持三种格式）
    // 格式1: "A. 选项1" 每行一个（标准格式）
    // 格式2: "A. 选项1  B. 选项2  C. 选项3  D. 选项4" 多选项在同一行
    // 格式3: "A" 单独一行 + "选项内容" 在下一行（LLM 错误输出格式）
    if (currentSection === "选择题" && /^[A-D][.、]\s+\S+/.test(trimmed)) {
      // 先尝试拆分成多行选项
      const optionMatches = trimmed.match(/([A-D])[.、]\s+(.+?)(?=\s+[A-D][.、]\s+|$)/g);
      if (optionMatches && optionMatches.length > 1) {
        // 多选项在同一行，拆分成多行
        if (currentQuestion) {
          for (const opt of optionMatches) {
            currentQuestion.content += "\n" + opt.trim();
          }
        }
      } else {
        // 单选项（每行一个）
        if (currentQuestion) {
          currentQuestion.content += "\n" + trimmed;
        }
      }
      continue;
    }

    // 格式3修复：单独一个 "A" 或 "B" 或 "C" 或 "D" 后面跟着选项内容
    // 如果当前行只是单独的 A/B/C/D，且下一行是较长的文本（选项内容），合并它们
    if (currentSection === "选择题" && /^[A-D]$/.test(trimmed) && currentQuestion) {
      // 标记这是一个选项的开始，下一行会合并
      currentQuestion.content += "\n" + trimmed + ".";
      continue;
    }

    // 处理小问 (1) (2) 等
    if (subQMatch && currentQuestion) {
      currentQuestion.content += "\n" + trimmed;
      continue;
    }

    if (qMatch && currentSection) {
      const qNum = parseInt(qMatch[1]);
      const qContent = qMatch[2].trim();

      // 跳过空的题目内容
      if (!qContent || qContent.length < 2) {
        if (currentQuestion) {
          currentQuestion.content += "\n" + trimmed;
        }
        continue;
      }

      // 检查是否是选项行（不是题目）
      if (currentSection === "选择题" && /^[A-D][.、]\s+\S+/.test(qContent)) {
        if (currentQuestion) {
          currentQuestion.content += "\n" + trimmed;
        }
        continue;
      }

      // 【修复】检查题目内容中是否包含选项（选项和题干在同一行）
      // 例如："题干 A. 选项1 B. 选项2 C. 选项3 D. 选项4"
      if (currentSection === "选择题" && /\b[A-D][.、]\s+/.test(qContent)) {
        // 用 split 分割：按选项字母分割字符串
        const segments = qContent.split(/(?=[A-D][.、]\s+)/);
        const questionPart = segments[0].trim();
        const optionsParts = segments.slice(1);
        // 清理每个选项，提取字母和内容
        const cleanOptions = optionsParts
          .map(seg => seg.trim())
          .filter(seg => seg.length > 0 && /^[A-D][.、]\s+/.test(seg))
          .map(seg => {
            // 提取字母和内容
            const letter = seg.charAt(0);
            const content = seg.substring(2).trim();
            return `${letter}. ${content}`;
          });
        // 创建题目，选项作为单独行追加到 content
        currentQuestion = {
          number: qNum,
          type: currentSection,
          content: cleanOptions.length > 0 ? questionPart + "\n" + cleanOptions.join("\n") : questionPart,
          answer: "",
          analysis: "",
          score: 2,
          difficulty: "中等"
        };
        result.questions.push(currentQuestion);
        continue;
      }

      // 去重：基于编号+题型+内容前20字符生成签名
      const signature = `${qNum}-${currentSection}-${qContent.slice(0, 20)}`;
      if (seenQuestionSignatures.has(signature)) {
        // 跳过完全重复的题目，并清除当前题目防止后续选项被追加
        currentQuestion = null;
        continue;
      }
      seenQuestionSignatures.add(signature);

      // 检查是否已存在相同编号的题目（兼容旧逻辑）
      const existingIndex = result.questions.findIndex(q => q.number === qNum);
      if (existingIndex >= 0) {
        // 已有相同编号题目，检查内容是否相似
        const existing = result.questions[existingIndex];
        if (existing.content.slice(0, 30) === qContent.slice(0, 30)) {
          // 内容相似，跳过，并清除当前题目
          currentQuestion = null;
          continue;
        }
        // 内容不同，可能是不同题型，保持两个
      }

      // 新题目 - 使用全局编号
      globalQuestionNumber = qNum;

      // 根据题号自动推断题型（fallback机制）
      let inferredType = currentSection || "选择题";
      if (!currentSection) {
        if (qNum >= 1 && qNum <= 10) {
          inferredType = "选择题";
        } else if (qNum >= 11 && qNum <= 20) {
          inferredType = "填空题";
        } else if (qNum >= 21 && qNum <= 30) {
          inferredType = "判断题";
        } else if (qNum >= 31) {
          inferredType = "简答题";
        }
      }

      // 根据题型设置默认分值
      let defaultScore = 10;
      if (inferredType === "选择题") defaultScore = 2;
      else if (inferredType === "填空题") defaultScore = 2;
      else if (inferredType === "判断题") defaultScore = 2;
      else if (inferredType === "简答题") defaultScore = 10;

      currentQuestion = {
        number: qNum,
        type: inferredType,
        content: stripLeadingQuestionNumber(qContent),
        answer: "",
        analysis: "",
        score: defaultScore,
        difficulty: "中等"
      };
      result.questions.push(currentQuestion);

    } else if (currentQuestion) {
      // 如果已经有当前题目，继续添加内容（选项或小问或答案）
      // 答案行：匹配 "答案：" 或 "答案:" 或行首直接是答案
      if (trimmed.match(/^答案[：:]\s*/)) {
        const answer = trimmed.replace(/^答案[：:]\s*/, "");
        currentQuestion.answer = answer;
      } else if (trimmed.includes("解析") && !trimmed.includes("解析：") && !trimmed.includes("解析:")) {
        // 跳过单独的"解析"文字
      } else if (trimmed.includes("解析：") || trimmed.includes("解析:") || trimmed.includes("解释")) {
        const analysis = trimmed.replace(/^.*(?:解析|解释)[：:]\s*/, "");
        currentQuestion.analysis = analysis;

        // 尝试从解析中提取难度信息
        if (trimmed.includes("简单") || trimmed.includes("基础")) {
          currentQuestion.difficulty = "简单";
        } else if (trimmed.includes("困难") || trimmed.includes("较难") || trimmed.includes("复杂")) {
          currentQuestion.difficulty = "较难";
        } else if (trimmed.includes("中等") || trimmed.includes("理解") || trimmed.includes("应用")) {
          currentQuestion.difficulty = "中等";
        }
      } else if (trimmed.includes("分值：") || trimmed.includes("分值:")) {
        // 提取分值信息，如 "分值：5"
        const scoreMatch = trimmed.match(/分值[：:]\s*(\d+)/);
        if (scoreMatch) {
          currentQuestion.score = parseInt(scoreMatch[1]);
        }
      } else if (trimmed.includes("难度：") || trimmed.includes("难度:")) {
        // 提取难度信息，如 "难度：中等"
        if (trimmed.includes("简单")) {
          currentQuestion.difficulty = "简单";
        } else if (trimmed.includes("较难") || trimmed.includes("困难")) {
          currentQuestion.difficulty = "较难";
        } else if (trimmed.includes("中等")) {
          currentQuestion.difficulty = "中等";
        }
      } else if (trimmed.includes("每题")) {
        // 尝试提取分值信息，如 "每题5分"
        const scoreMatch = trimmed.match(/每题(\d+)\s*分/);
        if (scoreMatch) {
          currentQuestion.score = parseInt(scoreMatch[1]);
        }
      } else if (trimmed.length > 1) {
        // 其他内容添加到题目中（选项、小问等）
        // 避免添加无意义内容
        if (!/^[【\[《<].*[】\]》>]$/.test(trimmed)) {
          currentQuestion.content += "\n" + trimmed;
        }
      }
    }
  }

  // 统计
  result.total_questions = result.questions.length;
  for (const q of result.questions) {
    const len = q.content.length;
    // 如果没有从解析中提取到难度，则根据题目长度判断
    if (!q.difficulty) {
      if (len < 30) {
        q.difficulty = "简单";
      } else if (len < 80) {
        q.difficulty = "中等";
      } else {
        q.difficulty = "较难";
      }
    }
    // 如果没有提取到分值，根据难度默认分配
    if (!q.score || q.score <= 0) {
      if (q.difficulty === "简单") {
        q.score = 5;
      } else if (q.difficulty === "中等") {
        q.score = 10;
      } else {
        q.score = 15;
      }
    }

    result.total_score += q.score;

    if (!result.question_types[q.type]) {
      result.question_types[q.type] = { count: 0, total_score: 0 };
    }
    result.question_types[q.type].count++;
    result.question_types[q.type].total_score += q.score;
  }

  return result;
}

// 难度颜色
function getDifficultyColor(difficulty: string): string {
  const normalized = (difficulty || "").trim();
  if (normalized === "简单" || normalized === "基础") {
    return "bg-emerald-100 text-emerald-800 border border-emerald-300";
  }
  if (normalized === "较难" || normalized === "困难" || normalized === "高难" || normalized === "难") {
    return "bg-rose-100 text-rose-800 border border-rose-300";
  }
  return "bg-amber-100 text-amber-800 border border-amber-300";
}

// 主组件
function isValidExamData(data: any): data is ExamData {
  return !!data && Array.isArray(data.questions) && typeof data.total_questions === "number";
}

export default function ExamCanvas({ examContent, examDataOverride, courseName = "期末考试", onExportWord, onExportAnswerSheet, onRequestAnswers, onRegenerateQuestion, onRegenerateComplete, onQuestionFeedback, onWrongAnswer, onPracticeComplete, similarQuestions }: ExamCanvasProps) {
  const [examData, setExamData] = useState<ExamData | null>(null);
  const [showAnswers, setShowAnswers] = useState(false);
  const [hasAnswers, setHasAnswers] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [collapsedSections, setCollapsedSections] = useState<Record<string, boolean>>({});
  const [expandedQuestions, setExpandedQuestions] = useState<Record<number, boolean>>({});
  const [regeneratingQuestion, setRegeneratingQuestion] = useState<number | null>(null);
  const [practiceMode, setPracticeMode] = useState<'view' | 'practice'>('view');
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [wrongAnswersWithUser, setWrongAnswersWithUser] = useState<Array<{ question: Question; userAnswer: string }>>([]);
  const [similarQuestionsExpanded, setSimilarQuestionsExpanded] = useState<Record<number, boolean>>({});
  const canvasRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (regeneratingQuestion !== null) {
      const qNum = regeneratingQuestion;
      const timeout = setTimeout(() => {
        setRegeneratingQuestion(null);
        onRegenerateComplete?.(qNum);
      }, 3000);
      return () => clearTimeout(timeout);
    }
  }, [regeneratingQuestion, onRegenerateComplete]);

  const toggleSection = (type: string) => {
    setCollapsedSections(prev => ({...prev, [type]: !prev[type]}));
  };

  const expandAll = () => setCollapsedSections({});
  const collapseAll = () => {
    const all: Record<string, boolean> = {};
    Object.keys(questionsByType).forEach(t => all[t] = true);
    setCollapsedSections(all);
  };

  const startPractice = () => {
    setPracticeMode('practice');
    setAnswers({});
    setSubmitted(false);
    setScore(0);
    setWrongAnswersWithUser([]);
    setCollapsedSections({});
  };

  const submitPractice = () => {
    if (!examData) return;
    let totalScore = 0;
    const wrongAnswers: Question[] = [];
    const wrongWithUser: Array<{ question: Question; userAnswer: string }> = [];
    const records: PracticeRecord[] = [];

    for (const q of examData.questions) {
      const normalizeAnswer = (a: string) => a.trim().replace(/\s+/g, " ").toUpperCase();
      const userAnswer = normalizeAnswer(answers[q.number] || "");
      const correctAnswer = normalizeAnswer(q.answer);
      const isCorrect = userAnswer === correctAnswer;

      if (isCorrect) {
        totalScore += q.score;
      } else {
        wrongAnswers.push(q);
        wrongWithUser.push({ question: q, userAnswer: answers[q.number] || "" });
        onWrongAnswer?.(q, answers[q.number] || "");
      }
      records.push({
        question: q,
        userAnswer: answers[q.number] || "",
        isCorrect,
      });
    }

    setScore(totalScore);
    setSubmitted(true);
    setPracticeMode('view');
    setWrongAnswersWithUser(wrongWithUser);
    onPracticeComplete?.(totalScore, examData.total_score, records, wrongAnswers, answers);
  };

  const getQuestionResult = (q: Question) => {
    const userAnswer = (answers[q.number] || "").trim().toUpperCase();
    const correctAnswer = q.answer.trim().toUpperCase();
    return userAnswer === correctAnswer;
  };

  // 解析试卷
  useEffect(() => {
    if (isValidExamData(examDataOverride) && examDataOverride.questions.length > 0) {
      const normalizedQuestions = examDataOverride.questions.map((q) => ({
        ...q,
        content: stripLeadingQuestionNumber(q.content || ""),
        options: q.type === "选择题"
          ? canonicalizeOptionLines(normalizeOptions((q as any).options) || [])
          : normalizeOptions((q as any).options),
      }));
      const normalizedData: ExamData = {
        ...examDataOverride,
        questions: normalizedQuestions,
        title: examDataOverride.title || courseName,
      };
      setExamData(normalizedData);
      setHasAnswers(normalizedData.questions.some(q => q.answer && q.answer.trim()));
      return;
    }

    if (examContent && examContent.length > 50) {
      try {
        const data = parseExamContent(examContent);
        // 只有解析出题目才显示
        if (data.questions && data.questions.length > 0) {
          data.title = data.title || courseName;
          setExamData(data);

          // 检查是否有答案
          const hasAns = data.questions.some(q => q.answer && q.answer.trim());
          setHasAnswers(hasAns);
        } else {
          console.warn("[ExamCanvas] 解析失败，未找到题目:", examContent.slice(0, 200));
          // 解析失败时设置一个标记，但不返回 null
          setExamData(null);
        }
      } catch (err) {
        console.error("[ExamCanvas] 解析异常:", err);
        setExamData(null);
      }
    }
  }, [examContent, examDataOverride, courseName]);

  // 导出 Word（提前定义，供解析失败时使用）
  const handleExportWord = async (includeAnswers: boolean) => {
    if (!onExportWord) {
      setIsExporting(true);
      try {
        const response = await fetch("/api/exam/export/docx", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            exam_paper: examContent,
            course_name: courseName,
            include_answers: includeAnswers
          })
        });
        const data = await response.json();
        if (data.success && data.download_url) {
          window.open(data.download_url, "_blank");
        }
      } catch (e) {
        console.error("Export failed:", e);
        alert("导出失败，请重试");
      } finally {
        setIsExporting(false);
      }
    } else {
      onExportWord(includeAnswers);
    }
  };

  // 导出答案卷
  const handleExportAnswerSheet = async () => {
    if (!onExportAnswerSheet) {
      setIsExporting(true);
      try {
        const response = await fetch("/api/exam/answersheet", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            exam_paper: examContent,
            course_name: courseName
          })
        });
        const data = await response.json();
        if (data.success && data.download_url) {
          window.open(data.download_url, "_blank");
        }
      } catch (e) {
        console.error("Export failed:", e);
        alert("导出失败，请重试");
      } finally {
        setIsExporting(false);
      }
    } else {
      onExportAnswerSheet();
    }
  };

  // 解析失败时仍然显示工具栏（支持导出原始内容）
  if (!examData) {
    return (
      <div className="my-4">
        {/* 即使解析失败也显示工具栏 */}
        <div className="flex flex-wrap gap-2 mb-4 p-3 bg-slate-50 border border-slate-200 rounded-lg">
          {/* 答案开关 - 始终显示 */}
          <button
            onClick={() => setShowAnswers(!showAnswers)}
            className="px-3 py-1.5 text-xs font-medium bg-white text-slate-600 border border-slate-200 hover:bg-slate-50 rounded-md transition-colors"
          >
            {showAnswers ? "👁 隐藏答案" : "👁 查看答案"}
          </button>

          <div className="h-6 w-px bg-slate-300"></div>

          <button
            onClick={() => handleExportWord(true)}
            disabled={isExporting}
            className="px-3 py-1.5 text-xs font-medium bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            📄 导出 Word（含答案）
          </button>

          <button
            onClick={() => handleExportWord(false)}
            disabled={isExporting}
            className="px-3 py-1.5 text-xs font-medium bg-white text-slate-700 border border-slate-200 rounded-md hover:bg-slate-50 transition-colors disabled:opacity-50"
          >
            📄 导出 Word（不含答案）
          </button>

          <button
            onClick={handleExportAnswerSheet}
            disabled={isExporting}
            className="px-3 py-1.5 text-xs font-medium bg-amber-500 text-white rounded-md hover:bg-amber-600 transition-colors disabled:opacity-50"
          >
            📋 导出答案卷
          </button>
        </div>

        {/* 解析失败时显示原始内容 */}
        <div className="bg-white border border-slate-200 rounded-lg p-6">
          <div className="text-sm text-slate-500 mb-4">⚠️ 自动解析失败，显示原始内容</div>
          <pre className="whitespace-pre-wrap text-sm text-slate-700 overflow-x-auto max-h-[500px]">
            {examContent}
          </pre>
        </div>
      </div>
    );
  }

  // 按题型分组，并按全局题号排序
  const questionsByType: Record<string, Question[]> = {};
  for (const q of examData.questions) {
    // 如果题型为空或未知，根据题号自动推断题型
    let finalType = q.type;
    if (!finalType || finalType === "未知") {
      if (q.number >= 1 && q.number <= 10) finalType = "选择题";
      else if (q.number >= 11 && q.number <= 20) finalType = "填空题";
      else if (q.number >= 21 && q.number <= 30) finalType = "判断题";
      else if (q.number >= 31) finalType = "简答题";
      else finalType = "选择题"; // 默认
    }

    if (!questionsByType[finalType]) {
      questionsByType[finalType] = [];
    }
    questionsByType[finalType].push({...q, type: finalType});
  }
  // 确保每个题型内按题号升序排列
  for (const type of Object.keys(questionsByType)) {
    questionsByType[type].sort((a, b) => a.number - b.number);
  }

  return (
    <div className="my-4">
      {/* 工具栏 */}
      <div className="flex flex-wrap gap-2 mb-4 p-3 bg-slate-50 border border-slate-200 rounded-lg">
        {/* 答案显示/隐藏开关 */}
        <button
          onClick={() => setShowAnswers(!showAnswers)}
          disabled={!hasAnswers}
          className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
            showAnswers
              ? "bg-teal-100 text-teal-700 border border-teal-200"
              : hasAnswers
              ? "bg-white text-slate-600 border border-slate-200 hover:bg-slate-50"
              : "bg-slate-100 text-slate-400 border border-slate-200 cursor-not-allowed"
          }`}
        >
          {showAnswers ? "👁 隐藏答案" : hasAnswers ? "👁 查看答案" : "暂无答案"}
        </button>

        <div className="h-6 w-px bg-slate-300"></div>

        <button
          onClick={() => handleExportWord(true)}
          disabled={isExporting}
          className="px-3 py-1.5 text-xs font-medium bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50"
        >
          📄 导出 Word（含答案）
        </button>

        <button
          onClick={() => handleExportWord(false)}
          disabled={isExporting}
          className="px-3 py-1.5 text-xs font-medium bg-white text-slate-700 border border-slate-200 rounded-md hover:bg-slate-50 transition-colors disabled:opacity-50"
        >
          📄 导出 Word（不含答案）
        </button>

        <button
          onClick={handleExportAnswerSheet}
          disabled={isExporting}
          className="px-3 py-1.5 text-xs font-medium bg-amber-500 text-white rounded-md hover:bg-amber-600 transition-colors disabled:opacity-50"
        >
          📋 导出答案卷
        </button>

        <div className="h-6 w-px bg-slate-300"></div>

        {practiceMode === 'view' && !submitted ? (
          <button
            onClick={startPractice}
            className="px-3 py-1.5 text-xs font-medium bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
          >
            ✏️ 开始练习
          </button>
        ) : practiceMode === 'practice' ? (
          <button
            onClick={submitPractice}
            className="px-3 py-1.5 text-xs font-medium bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
          >
            📝 提交答案
          </button>
        ) : null}
      </div>

      {/* 试卷 Canvas */}
      <div
        ref={canvasRef}
        className="bg-white border border-slate-800 rounded-lg shadow-sm overflow-hidden"
        style={{ maxWidth: 800, margin: "0 auto" }}
      >
        <div className="h-4 bg-slate-100"></div>

        <div className="px-8 pt-6 pb-4 border-b-2 border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center">
              <span className="text-white font-bold text-lg">DR</span>
            </div>
            <div className="flex-1 text-center">
              <p className="text-xs text-slate-500 mb-1">考前复习练习卷</p>
              <h1 className="text-xl font-bold text-slate-800 tracking-wide">{courseName || examData.title}</h1>
              <p className="text-sm text-slate-600 mt-1">期末复习专用</p>
            </div>
            <div className="w-16"></div>
          </div>

          <div className="border border-slate-300 rounded">
            <div className="grid grid-cols-4 divide-x divide-slate-300 text-xs">
              <div className="px-3 py-2 text-slate-500 text-center">考生姓名</div>
              <div className="px-3 py-2 text-slate-500 text-center">学号</div>
              <div className="px-3 py-2 text-slate-500 text-center">班级</div>
              <div className="px-3 py-2 text-slate-500 text-center">得分</div>
            </div>
            <div className="grid grid-cols-4 divide-x divide-slate-300 text-xs border-t border-slate-300">
              <div className="px-3 py-2 h-8"></div>
              <div className="px-3 py-2 h-8"></div>
              <div className="px-3 py-2 h-8"></div>
              <div className="px-3 py-2 h-8"></div>
            </div>
          </div>
        </div>

        <div className="px-8 py-3 bg-slate-50 border-b border-slate-200 flex flex-wrap gap-2">
          {Object.entries(examData.question_types).map(([type, stats]) => (
            <span key={type} className="px-3 py-1 bg-white border border-slate-300 text-slate-700 text-xs rounded">
              {type} {stats.count}题/{stats.total_score}分
            </span>
          ))}
          <span className="ml-auto text-xs text-slate-500 flex items-center">
            共{examData.total_questions}题 · 满分{examData.total_score}分 · 建议120分钟
          </span>
        </div>

        {/* 题目内容 - 按标准顺序显示：选择->填空->判断->简答 */}
        <div className="space-y-8">
          {Object.entries(questionsByType)
            .sort(([a], [b]) => {
              const order = ['选择题', '填空题', '判断题', '简答题', '计算题', '名词解释', '论述题'];
              const idxA = order.indexOf(a);
              const idxB = order.indexOf(b);
              return (idxA === -1 ? 99 : idxA) - (idxB === -1 ? 99 : idxB);
            })
            .map(([type, questions]) => {
              const isCollapsed = collapsedSections[type];
              return (
              <div key={type}>
                {/* 题型标题 - 可点击折叠 */}
                <button
                  onClick={() => toggleSection(type)}
                  className="w-full text-left flex items-center justify-between text-lg font-bold text-slate-800 mb-4 pb-2 border-b border-slate-200 hover:bg-slate-50 -mx-2 px-2 py-1 rounded transition-colors"
                >
                  <span>{type}（{questions.length}题）</span>
                  <span className="text-slate-400 text-sm">
                    {isCollapsed ? "▼ 点击展开" : "▲ 点击折叠"}
                  </span>
                </button>

                {/* 题目列表 */}
                {!isCollapsed && (
                  <div className="space-y-6">
                    {questions.map((q, idx) => {
                      const rawLines = q.content.split('\n');
                      const questionLines: string[] = [];
                      const options: string[] = q.type === "选择题" && q.options ? [...q.options] : [];

                      for (const line of rawLines) {
                        if (q.type === "选择题" && q.options && q.options.length >= 4) {
                          if (line.trim()) {
                            questionLines.push(line);
                          }
                          continue;
                        }

                        if (q.type === '选择题' && /^[A-D][.、]\s+\S+/.test(line)) {
                          options.push(line);
                        } else if (q.type === '选择题' && /\b[A-D][.、]\s+/.test(line)) {
                          const optionRegex = /\b([A-D])[.、]\s+/g;
                          const parts: string[] = line.split(optionRegex);
                          if (parts.length >= 3) {
                            const questionPart = parts[0].trim();
                            if (questionPart) {
                              questionLines.push(questionPart);
                            }
                            for (let i = 1; i < parts.length - 1; i += 2) {
                              const letter = parts[i].trim();
                              let content = parts[i + 1] || '';
                              content = content.replace(/\s*[A-D][.、]\s+$/, '').trim();
                              if (letter && content) {
                                const sep = line.match(/\b([A-D])[.、]/)?.[0]?.slice(-1) || '.';
                                options.push(`${letter}${sep} ${content}`);
                              }
                            }
                          }
                        } else {
                          questionLines.push(line);
                        }
                      }
                      const normalizedRenderOptions = q.type === "选择题" ? canonicalizeOptionLines(options) : options;
                      const hasOptionIssue = q.type === "选择题" && normalizedRenderOptions.length < 4;

                      return (
                        <div key={`${type}-${q.number}-${idx}`} className="question-item group p-4 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors">
                          {/* 题目标题 */}
                          <div className="flex items-center gap-3 mb-3">
                            <span className="inline-flex items-center justify-center w-7 h-7 bg-slate-800 text-white text-sm font-bold rounded-full">
                              {q.number}
                            </span>
                            <span className={`px-2 py-0.5 text-xs rounded ${getDifficultyColor(q.difficulty)}`}>
                              {q.difficulty}
                            </span>
                            <span className="text-xs text-slate-400">({q.score}分)</span>
                            {onRegenerateQuestion && (
                              <button
                                onClick={() => onRegenerateQuestion(q.number, q.type)}
                                disabled={regeneratingQuestion === q.number}
                                className="ml-auto opacity-0 group-hover:opacity-100 p-1.5 text-slate-400 hover:text-teal-600 hover:bg-teal-50 rounded transition-all disabled:opacity-50"
                                title="重新生成此题"
                              >
                                {regeneratingQuestion === q.number ? (
                                  <svg className="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                  </svg>
                                ) : (
                                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                  </svg>
                                )}
                              </button>
                            )}
                            {onQuestionFeedback && (
                              <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 ml-2">
                                <button
                                  onClick={() => onQuestionFeedback(q.number, 'up')}
                                  className="p-1 text-slate-400 hover:text-green-600 hover:bg-green-50 rounded transition-all"
                                  title="题目质量好"
                                >
                                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                                  </svg>
                                </button>
                                <button
                                  onClick={() => onQuestionFeedback(q.number, 'down')}
                                  className="p-1 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded transition-all"
                                  title="题目质量差"
                                >
                                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5" />
                                  </svg>
                                </button>
                              </div>
                            )}
                          </div>

                          {/* 题目内容 - 先显示题干 */}
                          <div className="ml-10 text-slate-800 text-sm leading-relaxed mb-3">
                            {questionLines.map((line, i) => (
                              <div key={i}>{line}</div>
                            ))}
                          </div>

                          {hasOptionIssue && (
                            <div className="ml-10 mb-3 text-xs text-rose-600 bg-rose-50 border border-rose-200 rounded px-2 py-1">
                              题目结构异常：该选择题选项不足 4 项，可点击右上角重生此题。
                            </div>
                          )}

                          {/* 选择题选项 - 交互式练习模式 */}
                          {q.type === "选择题" && normalizedRenderOptions.length > 0 && (
                            <div className="ml-10 grid grid-cols-1 sm:grid-cols-2 gap-2 mb-3">
                              {normalizedRenderOptions.map((opt, i) => {
                                const optionKey = extractOptionKey(opt, i);
                                const isSelected = answers[q.number] === optionKey;
                                const isCorrect = submitted && optionKey === q.answer.trim().charAt(0);
                                const isWrong = submitted && isSelected && optionKey !== q.answer.trim().charAt(0);

                                return (
                                  <div
                                    key={i}
                                    onClick={() => {
                                      if (!submitted) {
                                        setAnswers(prev => ({...prev, [q.number]: optionKey}));
                                      }
                                    }}
                                    className={`flex items-center gap-2 p-2 rounded text-sm cursor-pointer transition-all ${
                                      practiceMode === 'practice' && !submitted
                                        ? isSelected
                                          ? 'bg-teal-100 border-2 border-teal-500'
                                          : 'bg-white border border-slate-200 hover:bg-slate-50'
                                        : submitted
                                        ? isCorrect
                                          ? 'bg-green-100 border-2 border-green-500'
                                          : isWrong
                                          ? 'bg-red-100 border-2 border-red-500'
                                          : 'bg-white border border-slate-200'
                                        : 'bg-white border border-slate-200'
                                    }`}
                                  >
                                    <span className={`flex-shrink-0 w-6 h-6 flex items-center justify-center font-medium rounded ${
                                      isCorrect ? 'bg-green-500 text-white' :
                                      isWrong ? 'bg-red-500 text-white' :
                                      isSelected ? 'bg-teal-500 text-white' :
                                      'bg-slate-100 text-slate-600'
                                    }`}>
                                      {isCorrect ? '✓' : isWrong ? '✗' : optionKey}
                                    </span>
                                    <span className="text-slate-700">{stripOptionPrefix(opt)}</span>
                                  </div>
                                );
                              })}
                            </div>
                          )}

                          {/* 非选择题：填空/简答/判断题输入框 */}
                          {normalizedRenderOptions.length === 0 && q.type !== '选择题' && (
                            <div className="ml-10 mb-3">
                              {q.type === '判断题' ? (
                                <div className="flex gap-4">
                                  {['正确', '错误'].map((opt, i) => {
                                    const optionKey = i === 0 ? '正确' : '错误';
                                    const isSelected = answers[q.number] === optionKey;
                                    const isCorrect = submitted && optionKey === q.answer.trim();
                                    const isWrong = submitted && isSelected && optionKey !== q.answer.trim();

                                    return (
                                      <div
                                        key={opt}
                                        onClick={() => {
                                          if (!submitted) {
                                            setAnswers(prev => ({...prev, [q.number]: optionKey}));
                                          }
                                        }}
                                        className={`px-4 py-2 rounded cursor-pointer transition-all ${
                                          practiceMode === 'practice' && !submitted
                                            ? isSelected
                                              ? 'bg-teal-100 border-2 border-teal-500'
                                              : 'bg-white border border-slate-200 hover:bg-slate-50'
                                            : submitted
                                            ? isCorrect
                                              ? 'bg-green-100 border-2 border-green-500'
                                              : isWrong
                                              ? 'bg-red-100 border-2 border-red-500'
                                              : 'bg-white border border-slate-200'
                                            : 'bg-white border border-slate-200'
                                        }`}
                                      >
                                        <span className={`flex-shrink-0 w-6 h-6 inline-flex items-center justify-center font-medium rounded mr-2 ${
                                          isCorrect ? 'bg-green-500 text-white' :
                                          isWrong ? 'bg-red-500 text-white' :
                                          isSelected ? 'bg-teal-500 text-white' :
                                          'bg-slate-100 text-slate-600'
                                        }`}>
                                          {isCorrect ? '✓' : isWrong ? '✗' : '○'}
                                        </span>
                                        <span className="text-slate-700">{opt}</span>
                                      </div>
                                    );
                                  })}
                                </div>
                              ) : (
                                <input
                                  type="text"
                                  value={answers[q.number] || ""}
                                  onChange={(e) => {
                                    if (!submitted) {
                                      setAnswers(prev => ({...prev, [q.number]: e.target.value}));
                                    }
                                  }}
                                  disabled={submitted}
                                  placeholder={practiceMode === 'practice' && !submitted ? "请输入答案..." : ""}
                                  className={`w-full px-4 py-2 border rounded text-sm ${
                                    submitted
                                      ? answers[q.number]?.trim().toUpperCase() === q.answer.trim().toUpperCase()
                                        ? 'bg-green-50 border-green-500 text-green-800'
                                        : 'bg-red-50 border-red-500 text-red-800'
                                      : 'bg-white border-slate-200 focus:border-teal-500 focus:ring-1 focus:ring-teal-500'
                                  }`}
                                />
                              )}
                              {submitted && (
                                <div className="mt-2 text-sm">
                                  <span className="font-medium text-slate-600">你的答案：</span>
                                  <span className={answers[q.number]?.trim().toUpperCase() === q.answer.trim().toUpperCase() ? 'text-green-600' : 'text-red-600'}>
                                    {answers[q.number] || "(未作答)"}
                                  </span>
                                  <span className="mx-2 text-slate-400">|</span>
                                  <span className="font-medium text-slate-600">正确答案：</span>
                                  <span className="text-green-600 font-bold">{q.answer}</span>
                                </div>
                              )}
                            </div>
                          )}

                          {/* 答案 */}
                          {showAnswers && q.answer && (
                            <div className="ml-10 mt-3 p-3 bg-emerald-50 border-l-4 border-emerald-500 rounded-r-lg">
                              <div className="text-sm">
                                <span className="font-medium text-emerald-700">答案：</span>
                                <span className="text-emerald-800 font-bold">{q.answer}</span>
                              </div>
                              {q.analysis && (
                                <div className="mt-2 text-sm">
                                  <span className="font-medium text-blue-700">解析：</span>
                                  <span className="text-blue-800">{q.analysis}</span>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* 相似题推荐区域 */}
        {submitted && wrongAnswersWithUser.length > 0 && similarQuestions && Object.keys(similarQuestions).length > 0 && (
          <div className="mt-6 p-4 bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 rounded-xl">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2 h-2 rounded-full bg-amber-500"></div>
              <h3 className="font-bold text-amber-800">相似题推荐</h3>
              <span className="text-xs text-amber-600">针对错题的系统化练习</span>
            </div>
            <div className="space-y-4">
              {wrongAnswersWithUser
              .filter(({ question }) => {
                const simqs = similarQuestions?.[question.number];
                return simqs && simqs.length > 0;
              })
              .map(({ question }) => {
                const simqs = similarQuestions[question.number];
                const isExpanded = similarQuestionsExpanded[question.number];
                return (
                  <div key={question.number} className="bg-white rounded-lg border border-amber-100 overflow-hidden">
                    <button
                      onClick={() => setSimilarQuestionsExpanded(prev => ({...prev, [question.number]: !prev[question.number]}))}
                      className="w-full px-4 py-3 flex items-center justify-between bg-amber-50 hover:bg-amber-100 transition-colors text-left"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-amber-800">第 {question.number} 题</span>
                        <span className="text-xs text-amber-600">({question.type})</span>
                        <span className="ml-2 px-2 py-0.5 bg-amber-200 text-amber-700 text-xs rounded-full">
                          {simqs.length} 道相似题
                        </span>
                      </div>
                      <svg className={`w-4 h-4 text-amber-500 transition-transform ${isExpanded ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </button>
                    {isExpanded && (
                      <div className="p-4 space-y-3">
                        {simqs.map((sq, idx) => (
                          <div key={idx} className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                            <div className="text-sm text-slate-700 mb-2 whitespace-pre-wrap">{sq.question_content}</div>
                            <div className="flex items-center gap-4 text-xs">
                              {sq.knowledge_point && (
                                <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded">{sq.knowledge_point}</span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// 检测内容是否为完整试卷（多题型混合的大题量测试）
export function isExamContent(content: string): boolean {
  if (!content || content.length < 30) {
    return false;
  }

  // 题型标题检测
  const hasChoiceTitle = /【选择题】|一[、.]\s*选择题/.test(content);
  const hasFillTitle = /【填空题】|二[、.]\s*填空题/.test(content);
  const hasJudgeTitle = /【判断题】|三[、.]\s*判断题/.test(content);
  const hasEssayTitle = /【简答题】|四[、.]\s*简答题/.test(content);

  // 必须有多个题型（至少2种）才算完整试卷
  const typeCount = [hasChoiceTitle, hasFillTitle, hasJudgeTitle, hasEssayTitle].filter(Boolean).length;
  if (typeCount < 2) {
    return false;
  }

  // 有多个题型 + 题目编号（支持新格式：1. （分值：5分，难度：中等））
  const hasNumberedQuestions = /^\d+\.\s*（[^）]+）/.test(content) || /^\d+[.、]\s*.+/m.test(content);
  if (!hasNumberedQuestions) {
    return false;
  }

  // 额外验证：检查是否真的有答案格式（防止误判功能介绍为试卷）
  // 真正试卷会有"答案：X"或"答案:"这样的格式
  const hasAnswerFormat = /答案[：:]\s*[A-Da-d正确错误\/\d]/.test(content);
  if (!hasAnswerFormat) {
    return false;
  }

  // 最终验证：检查是否包含选项标记（选择题必须有A/B/C/D选项）
  if (hasChoiceTitle) {
    const hasOptions = /[A-D][.、]/.test(content);
    if (!hasOptions) {
      return false;
    }
  }

  return true;
}
