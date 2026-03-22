"use client";

import { useState, useRef, useEffect } from "react";

// 题目类型
interface Question {
  number: number;
  type: string;
  content: string;
  answer: string;
  analysis: string;
  score: number;
  difficulty: string;
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

// Props
interface ExamCanvasProps {
  examContent: string;
  courseName?: string;
  onExportWord?: (includeAnswers: boolean) => void;
  onExportAnswerSheet?: () => void;
  onRequestAnswers?: () => void;  // 请求答案的回调
}

// 解析试卷内容
function parseExamContent(content: string): ExamData {
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
    // 选择题：1-10题
    '选择题': /(一[、.]\s*)?选择题|^[1-9]\d{0,1}\.\s.*[A-D][.、]/,
    // 填空题：11-20题
    '填空题': /(二[、.]\s*)?填空题|^1[1-9]\.\s/,
    // 判断题：21-30题
    '判断题': /(三[、.]\s*)?判断题|^2[1-9]\.\s/,
    // 简答题：31-34题
    '简答题': /(四[、.]\s*)?简答题|(四[、.]\s*)?解答题|^3[1-4]\.\s/,
    // 计算题
    '计算题': /(五[、.]\s*)?计算题/,
    // 名词解释
    '名词解释': /(名词解释)/,
    // 论述题
    '论述题': /(论述题)/,
  };

  // 提取标题
  for (let i = 0; i < Math.min(5, lines.length); i++) {
    const line = lines[i].trim();
    if (line && !Object.values(typePatterns).some(p => p.test(line))) {
      if (!result.title) {
        result.title = line;
      } else if (line.includes("期末") || line.includes("试卷") || line.includes("考试")) {
        result.subtitle = line;
      }
    }
  }

  // 用于去重的内容片段集合
  const seenQuestionSignatures = new Set<string>();

  // 解析题目
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    // 检测题型标题
    for (const [qtype, pattern] of Object.entries(typePatterns)) {
      if (pattern.test(trimmed) && trimmed.length < 20) {
        currentSection = qtype;
        currentQuestion = null;
        break;
      }
    }

    // 检测题目编号 - 支持统一编号如 "1." "2." 或 "(1)" 小问
    const qMatch = trimmed.match(/^(\d+)[.、]\s*(.+)/);
    const subQMatch = trimmed.match(/^[\(（](\d+)[\)）]\s*(.+)/);

    // 处理选择题选项（以 A. B. C. D. 开头的行）
    if (/^[A-D][.、]\s*/.test(trimmed)) {
      if (currentQuestion) {
        currentQuestion.content += "\n" + trimmed;
      }
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
      if (/^[A-D][.、]/.test(qContent)) {
        if (currentQuestion) {
          currentQuestion.content += "\n" + trimmed;
        }
        continue;
      }

      // 去重：基于编号+题型+内容前20字符生成签名
      const signature = `${qNum}-${currentSection}-${qContent.slice(0, 20)}`;
      if (seenQuestionSignatures.has(signature)) {
        // 跳过完全重复的题目
        continue;
      }
      seenQuestionSignatures.add(signature);

      // 检查是否已存在相同编号的题目（兼容旧逻辑）
      const existingIndex = result.questions.findIndex(q => q.number === qNum);
      if (existingIndex >= 0) {
        // 已有相同编号题目，检查内容是否相似
        const existing = result.questions[existingIndex];
        if (existing.content.slice(0, 30) === qContent.slice(0, 30)) {
          // 内容相似，跳过
          continue;
        }
        // 内容不同，可能是不同题型，保持两个
      }

      // 新题目 - 使用全局编号
      globalQuestionNumber = qNum;
      currentQuestion = {
        number: qNum,
        type: currentSection,
        content: qContent,  // 只取编号后的内容，不包含编号
        answer: "",
        analysis: "",
        score: 10,
        difficulty: "中等"
      };
      result.questions.push(currentQuestion);

    } else if (currentQuestion) {
      // 如果已经有当前题目，继续添加内容（选项或小问）
      if (trimmed.includes("答案") && !trimmed.includes("答案：") && !trimmed.includes("答案:")) {
        // 跳过单独的"答案"文字
      } else if (trimmed.includes("答案：") || trimmed.includes("答案:")) {
        const answer = trimmed.replace(/^.*答案[：:]\s*/, "");
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
      } else if (trimmed.includes("分值") || trimmed.includes("每题")) {
        // 尝试提取分值信息，如 "每题5分"
        const scoreMatch = trimmed.match(/(\d+)\s*分/);
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
    if (!q.score || q.score === 10) {
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
  switch (difficulty) {
    case "简单": return "bg-green-100 text-green-700";
    case "较难": return "bg-red-100 text-red-700";
    default: return "bg-amber-100 text-amber-700";
  }
}

// 主组件
export default function ExamCanvas({ examContent, courseName = "期末考试", onExportWord, onExportAnswerSheet, onRequestAnswers }: ExamCanvasProps) {
  const [examData, setExamData] = useState<ExamData | null>(null);
  const [showAnswers, setShowAnswers] = useState(false);  // 默认不显示答案
  const [hasAnswers, setHasAnswers] = useState(false);   // 是否有答案
  const [isExporting, setIsExporting] = useState(false);
  const canvasRef = useRef<HTMLDivElement>(null);

  // 解析试卷
  useEffect(() => {
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
  }, [examContent, courseName]);

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
            onClick={() => onRequestAnswers && onRequestAnswers()}
            className="px-3 py-1.5 text-xs font-medium bg-white text-slate-600 border border-slate-200 hover:bg-slate-50 rounded-md transition-colors"
          >
            👁 请求答案
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
    if (!questionsByType[q.type]) {
      questionsByType[q.type] = [];
    }
    questionsByType[q.type].push(q);
  }
  // 确保每个题型内按题号升序排列
  for (const type of Object.keys(questionsByType)) {
    questionsByType[type].sort((a, b) => a.number - b.number);
  }

  return (
    <div className="my-4">
      {/* 工具栏 */}
      <div className="flex flex-wrap gap-2 mb-4 p-3 bg-slate-50 border border-slate-200 rounded-lg">
        {/* 答案显示/隐藏开关 - 始终显示 */}
        <button
          onClick={() => {
            if (hasAnswers) {
              setShowAnswers(!showAnswers);
            } else if (onRequestAnswers) {
              // 没有答案时，点击则请求答案
              onRequestAnswers();
            }
          }}
          className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
            showAnswers
              ? "bg-teal-100 text-teal-700 border border-teal-200"
              : "bg-white text-slate-600 border border-slate-200 hover:bg-slate-50"
          }`}
        >
          {showAnswers ? "👁 隐藏答案" : hasAnswers ? "👁 显示答案" : "👁 请求答案"}
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

      {/* 试卷 Canvas */}
      <div
        ref={canvasRef}
        className="bg-white border border-slate-200 rounded-lg shadow-sm p-8"
      >
        {/* 标题 */}
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-slate-900 mb-2">{examData.title}</h1>
          {examData.subtitle && (
            <p className="text-slate-500">{examData.subtitle}</p>
          )}
        </div>

        {/* 统计信息 */}
        <div className="flex justify-center gap-6 mb-8 text-sm">
          <div className="px-4 py-2 bg-slate-100 rounded-lg">
            <span className="text-slate-500">总分：</span>
            <span className="font-bold text-slate-900">{examData.total_score}分</span>
          </div>
          <div className="px-4 py-2 bg-slate-100 rounded-lg">
            <span className="text-slate-500">题数：</span>
            <span className="font-bold text-slate-900">{examData.total_questions}题</span>
          </div>
          <div className="px-4 py-2 bg-slate-100 rounded-lg">
            <span className="text-slate-500">考试时间：</span>
            <span className="font-bold text-slate-900">120分钟</span>
          </div>
        </div>

        {/* 题型统计 */}
        <div className="flex flex-wrap justify-center gap-3 mb-8">
          {Object.entries(examData.question_types).map(([type, stats]) => (
            <span key={type} className="px-3 py-1 bg-teal-50 text-teal-700 text-xs rounded-full">
              {type} {stats.count}题/{stats.total_score}分
            </span>
          ))}
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
            .map(([type, questions]) => (
            <div key={type}>
              {/* 题型标题 */}
              <h2 className="text-lg font-bold text-slate-800 mb-4 pb-2 border-b border-slate-200">
                {type}
              </h2>

              {/* 题目列表 */}
              <div className="space-y-6">
                {questions.map((q) => (
                  <div key={q.number} className="question-item">
                    {/* 题目标题 */}
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-slate-900">{q.number}.</span>
                        <span className={`px-2 py-0.5 text-xs rounded ${getDifficultyColor(q.difficulty)}`}>
                          {q.difficulty}
                        </span>
                        <span className="text-xs text-slate-400">({q.score}分)</span>
                      </div>
                    </div>

                    {/* 题目内容 */}
                    <div className="ml-6 text-slate-800 whitespace-pre-wrap text-sm leading-relaxed">
                      {q.content.split('\n').map((line, i) => {
                        // 处理选项
                        if (/^[A-D][.、、]/.test(line)) {
                          return (
                            <div key={i} className="ml-4 text-slate-700">
                              {line}
                            </div>
                          );
                        }
                        return <div key={i}>{line}</div>;
                      })}
                    </div>

                    {/* 答案 */}
                    {showAnswers && q.answer && (
                      <div className="ml-6 mt-3 p-3 bg-emerald-50 border-l-4 border-emerald-500 rounded-r-lg">
                        <div className="text-sm">
                          <span className="font-medium text-emerald-700">答案：</span>
                          <span className="text-emerald-800">{q.answer}</span>
                        </div>
                        {showAnswers && q.analysis && (
                          <div className="mt-2 text-sm">
                            <span className="font-medium text-blue-700">解析：</span>
                            <span className="text-blue-800">{q.analysis}</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// 检测内容是否为试卷
export function isExamContent(content: string): boolean {
  if (!content || content.length < 30) return false;

  // 题型标题检测（更宽松）
  const hasTypeTitle = /(一[、.]\s*)?选择题|(二[、.]\s*)?填空题|(三[、.]\s*)?判断题|(四[、.]\s*)?简答题|论述题|计算题|名词解释|解答题/.test(content);

  // 题目编号检测（支持多种格式）
  const hasNumberedQuestions = /^\d+[.、]\s*.{1,}/m.test(content) || /\(\d+\)\s*.{1,}/.test(content);

  // 选择题选项检测
  const hasOptions = /^[A-D][.、]\s*.{1,}/m.test(content);

  // 有题型标题，或者有题目编号+选项，都可能是试卷
  const result = hasTypeTitle || (hasNumberedQuestions && hasOptions);

  // 排除明显非试卷的内容
  if (result) {
    // 如果是纯对话式内容，不是试卷
    if (content.includes('请问') && content.includes('谢谢') && !content.includes('题')) return false;
    // 如果是纯代码块且没有题目内容，不是试卷
    if (content.includes('```') && !hasTypeTitle && !hasNumberedQuestions) return false;
  }

  return result;
}
