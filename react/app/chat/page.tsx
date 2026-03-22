"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ExamCanvas, { isExamContent } from "@/components/ExamCanvas";

// Types
interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

interface Session {
  id: string;
  name: string;
}

interface KnowledgeFile {
  filename: string;
  size: number;
  modified_at: number;
  embedded?: boolean;  // 是否已完成向量化
}

// ============== 设计系统 ==============
// 使用 CSS 变量实现 Editorial Minimalist 风格
// 青色品牌色: #14b8a6 (teal-500)

// ============== 思考链组件 ==============
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

// ============== Markdown 渲染组件 ==============
function parseQuizContent(content: string): {
  isQuiz: boolean;
  questions: Array<{ type: string; question: string; options?: string[]; answer: string; explanation: string }>;
} {
  const questions: Array<{ type: string; question: string; options?: string[]; answer: string; explanation: string }> = [];

  const choiceMatch = content.match(/【选择题】|一、选择题|1\./g);
  const fillMatch = content.match(/【填空题】|二、填空题/g);
  const judgeMatch = content.match(/【判断题】|三、判断题/g);

  if (choiceMatch || fillMatch || judgeMatch) {
    const lines = content.split("\n");
    let currentType = "选择题";
    let currentQuestion = "";
    let options: string[] = [];
    let answer = "";
    let explanation = "";

    for (const line of lines) {
      const trimmed = line.trim();

      if (trimmed.includes("选择题")) {
        currentType = "选择题";
        continue;
      } else if (trimmed.includes("填空题")) {
        currentType = "填空题";
        continue;
      } else if (trimmed.includes("判断题")) {
        currentType = "判断题";
        continue;
      }

      if (/^\d+[.、]/.test(trimmed)) {
        if (currentQuestion) {
          questions.push({ type: currentType, question: currentQuestion, options: options.length > 0 ? options : undefined, answer, explanation });
        }
        currentQuestion = trimmed.replace(/^\d+[.、]\s*/, "");
        options = [];
        answer = "";
        explanation = "";
      } else if (/^[A-D][.、、]/.test(trimmed) && currentType === "选择题") {
        options.push(trimmed);
      } else if (trimmed.includes("答案") || trimmed.includes("答案：")) {
        answer = trimmed.replace(/.*答案[：:]\s*/, "");
      } else if (trimmed.includes("解析") || trimmed.includes("解析：") || trimmed.includes("解释")) {
        explanation = trimmed.replace(/.*解析[：:]\s*/, "");
      } else if (currentQuestion && !trimmed.includes("【")) {
        currentQuestion += "\n" + trimmed;
      }
    }

    if (currentQuestion) {
      questions.push({ type: currentType, question: currentQuestion, options: options.length > 0 ? options : undefined, answer, explanation });
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

function QuizCard({ questions }: { questions: Array<{ type: string; question: string; options?: string[]; answer: string; explanation: string }> }) {
  return (
    <div className="space-y-4 my-4">
      {questions.map((q, idx) => {
        const style = quizTypeStyles[q.type] || quizTypeStyles["选择题"];
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
              <div className="w-2 h-2 rounded-full bg-teal-500"></div>
            </div>

            <div className="text-slate-800 mb-4 whitespace-pre-wrap text-[15px] leading-relaxed font-medium">{q.question}</div>

            {q.options && q.options.length > 0 && (
              <div className="space-y-2 mb-4 ml-1">
                {q.options.map((opt, optIdx) => (
                  <div key={optIdx} className="flex items-start gap-2 text-slate-700 text-sm">
                    <span className="w-5 h-5 rounded bg-slate-100 flex items-center justify-center text-xs font-medium text-slate-500 flex-shrink-0">
                      {String.fromCharCode(65 + optIdx)}
                    </span>
                    <span>{opt.replace(/^[A-D][.、\s]+/, "")}</span>
                  </div>
                ))}
              </div>
            )}

            {q.answer && (
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

            {q.explanation && (
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

function MarkdownContent({ content, onRequestAnswers }: { content: string; onRequestAnswers?: () => void }) {
  // 检测是否为试卷格式，使用 Canvas 渲染
  if (isExamContent(content)) {
    // 直接使用 ExamCanvas 渲染，它内部会处理解析失败的情况
    // 解析失败时会渲染原始内容作为后备
    return <ExamCanvas examContent={content} courseName="期末考试" onRequestAnswers={onRequestAnswers} />;
  }

  // 尝试用简单方式解析题目
  const { isQuiz, questions } = parseQuizContent(content);
  if (isQuiz && questions.length > 0) {
    return <QuizCard questions={questions} />;
  }

  return (
    <div className="prose-custom">
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

    setIsUploading(true);
    setUploadStatus("上传中...");

    const formData = new FormData();
    formData.append("session_id", sessionId);
    selectedFiles.forEach(file => formData.append("files", file));

    try {
      const res = await fetch("/api/knowledge/upload", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        setUploadStatus(`成功上传 ${selectedFiles.length} 个文件，正在向量化...`);
        setSelectedFiles([]);
        setActiveTab("list");

        // 轮询等待向量化完成（最多10次，间隔2秒）
        let attempts = 0;
        const pollInterval = setInterval(() => {
          attempts++;
          fetchFiles();
          if (attempts >= 10) {
            clearInterval(pollInterval);
            setUploadStatus("");
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
    formData.append("session_id", sessionId);
    formData.append("file", file);
    try {
      const res = await fetch("/api/knowledge/sample/upload", { method: "POST", body: formData });
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
              <button onClick={fetchFiles} className="text-xs font-mono text-teal-600 hover:underline">刷新</button>
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
                          {file.embedded === false && <span className="text-amber-500 ml-1">（处理中）</span>}
                          {file.embedded === true && <span className="text-green-500 ml-1">✓</span>}
                        </p>
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

// ============== 会话管理模态框 ==============
function SessionModal({
  isOpen,
  onClose,
  sessions,
  currentSession,
  onSelect,
  onCreate
}: {
  isOpen: boolean;
  onClose: () => void;
  sessions: Session[];
  currentSession: string;
  onSelect: (id: string) => void;
  onCreate: (name: string) => void;
}) {
  const [newName, setNewName] = useState("");

  if (!isOpen) return null;

  const handleCreate = () => {
    if (newName.trim()) {
      onCreate(newName.trim());
      setNewName("");
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
          <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2 px-1">活跃的复习会话</div>
          <ul className="space-y-1 font-mono text-sm max-h-60 overflow-y-auto">
            {sessions.map(session => (
              <li key={session.id}>
                <button
                  onClick={() => { onSelect(session.id); onClose(); }}
                  className={`w-full text-left px-3 py-2 rounded-md transition-colors ${
                    currentSession === session.id
                      ? "bg-teal-50 text-teal-700 border border-teal-200"
                      : "hover:bg-slate-100 text-slate-700"
                  }`}
                >
                  {session.name}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

// ============== 主页面组件 ==============
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

  // 模态框状态
  const [showSessionModal, setShowSessionModal] = useState(false);
  const [showKnowledgePanel, setShowKnowledgePanel] = useState(false);
  const [isBackendConnected, setIsBackendConnected] = useState(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);

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
    fetch(`/api/chat/messages?session_id=${encodeURIComponent(sessionId)}`)
      .then(res => res.json())
      .then(data => {
        if (data.code === 200 && data.data) {
          setMessages(data.data);
        }
      })
      .catch(err => console.error("加载消息失败:", err));
  };

  // 滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thoughts]);

  // 发送消息
  const sendMessage = async () => {
    if (!input.trim() || !currentSession) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: Date.now()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setThoughts([]);

    const startTime = Date.now();
    const assistantMessageId = (Date.now() + 1).toString();

    setMessages(prev => [...prev, {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      timestamp: Date.now()
    }]);

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input, session_id: currentSession })
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullContent = "";

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split("\n");

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const data = line.slice(6);
              if (data === "[DONE]") continue;

              try {
                const parsed = JSON.parse(data);
                if (parsed.text) {
                  // 检测系统思考
                  if (parsed.text.includes("**[系统思考")) {
                    const match = parsed.text.match(/\*\*(.*?)\*\*/);
                    if (match) {
                      const thought = match[1].replace("[系统思考：", "").replace("]", "");
                      setThoughts(prev => [...prev, thought]);
                    }
                  } else {
                    fullContent += parsed.text;
                    setMessages(prev => prev.map(m =>
                      m.id === assistantMessageId ? { ...m, content: fullContent } : m
                    ));
                  }
                }
              } catch {
                // 忽略解析错误
              }
            }
          }
        }
      }

      // 更新统计
      setResponseTime((Date.now() - startTime) / 1000);
      setTotalTokens(prev => prev + Math.ceil(fullContent.length / 4));
    } catch (error) {
      console.error("Chat error:", error);
      setMessages(prev => prev.map(m =>
        m.id === assistantMessageId ? { ...m, content: "请求失败，请稍后重试。" } : m
      ));
    } finally {
      setIsLoading(false);
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

  // 定期刷新统计
  useEffect(() => {
    fetchTokenStats();
    const interval = setInterval(fetchTokenStats, 30000); // 每30秒刷新
    return () => clearInterval(interval);
  }, [fetchTokenStats]);

  // 创建会话
  const createSession = async (name: string) => {
    try {
      const res = await fetch("/api/chat/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: name, name })
      });
      const data = await res.json();
      if (data.code === 200) {
        const newSession = { id: name, name };
        setSessions(prev => [...prev, newSession]);
        setCurrentSession(name);
        setCurrentSessionName(name);
        setMessages([]);
        setThoughts([]);
      }
    } catch (error) {
      console.error("Create session error:", error);
      // 即使 API 失败也创建本地会话
      const newSession = { id: name, name };
      setSessions(prev => [...prev, newSession]);
      setCurrentSession(name);
      setCurrentSessionName(name);
      setMessages([]);
      setThoughts([]);
    }
  };

  // 销毁记忆
  const clearMemory = async () => {
    if (!confirm("确定要销毁当前会话的记忆吗？此操作不可恢复。")) return;
    try {
      await fetch(`/api/chat/session/${currentSession}`, { method: "DELETE" });
      setMessages([]);
      setThoughts([]);
      setTotalTokens(0);
      setResponseTime(0);
    } catch (error) {
      console.error("Clear memory error:", error);
    }
  };

  return (
    <div className="h-screen w-full flex flex-col bg-slate-50 overflow-hidden font-sans">
      {/* 后端未连接警告 */}
      {!isBackendConnected && (
        <div className="bg-amber-500 text-white text-center py-2 text-sm font-medium">
          ⚠️ 后端未连接，请启动 FastAPI: python api/main.py
        </div>
      )}

      {/* ============== 顶部导航栏 ============== */}
      <header className="flex-none px-6 py-3 flex items-center justify-between border-b border-slate-200 bg-white z-20">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-slate-900 text-white rounded flex items-center justify-center font-mono font-bold text-sm shadow-sm ring-1 ring-slate-900">
            DR
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight text-slate-900 uppercase">
              Deep<span className="text-teal-600">Revision</span>
            </h1>
            <p className="text-[0.65rem] font-mono text-slate-400 uppercase tracking-widest">RAG 引擎运行中</p>
          </div>
        </div>

        {/* 算力核心统计 */}
        <div className="hidden md:flex items-center gap-3 px-4 py-1.5 bg-slate-900 rounded-full shadow-sm ring-1 ring-slate-800 transition-all duration-300 hover:shadow-md">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-teal-500 animate-pulse shadow-[0_0_8px_rgba(20,184,166,0.8)]"></span>
            <span className="text-[0.65rem] font-mono text-slate-300 uppercase tracking-widest">算力核心</span>
          </div>
          <div className="h-3 w-px bg-slate-700"></div>
          <div className="flex items-center gap-4 text-[0.7rem] font-mono text-slate-50">
            <div className="flex items-baseline gap-1">
              <span className="text-slate-400">总消耗 Tokens</span>
              <span className="font-bold text-white tracking-tight">{totalTokens}</span>
            </div>
            <div className="flex items-baseline gap-1">
              <span className="text-slate-400">响应延迟</span>
              <span className="font-bold text-teal-400 tracking-tight">{responseTime.toFixed(2)}s</span>
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
                    sampleInfo = "\n\n【重要】请参考已上传的样卷格式出题，包括题型分布、分值、编号风格等。";
                  }
                } catch (e) {}
                const prompt = `请根据已上传的课件内容，生成一份期末复习综合测试卷。考点范围请从课件中提取关键知识点。${sampleInfo}

请生成完整试卷，包含题目、答案和解析。`;
                setInput(prompt);
                setTimeout(() => {
                  const btn = document.getElementById('send-btn');
                  if (btn) btn.click();
                }, 100);
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
      <main className="flex-1 w-full relative overflow-hidden flex">
        {/* 聊天区域 */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* 消息列表 */}
          <div className="flex-1 overflow-y-auto p-4 md:p-8">
            <div className="max-w-4xl mx-auto">
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
                  key={message.id}
                  className={`w-full flex items-start gap-4 message-enter ${message.role === "user" ? "flex-row-reverse" : ""}`}
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

                    <div className={`inline-block px-5 py-3.5 rounded-2xl shadow-sm ${
                      message.role === "user"
                        ? "rounded-tr-md"
                        : "rounded-tl-md border"
                    }`} style={{
                      background: message.role === "user"
                        ? '#f1f5f9'
                        : 'rgba(255,255,255,0.9)',
                      color: message.role === "user" ? '#334155' : '#1e293b',
                      borderColor: 'rgba(180,170,150,0.2)'
                    }}>
                      <MarkdownContent
                        content={message.content || (message.role === "assistant" && isLoading ? "正在思考中..." : "")}
                        onRequestAnswers={() => {
                          // 发送消息请求答案
                          const input = "请给出上面试卷的答案和解析";
                          setMessages(prev => [...prev, {
                            id: Date.now().toString(),
                            role: "user",
                            content: input,
                            timestamp: Date.now()
                          }]);
                          setInput("");
                          setTimeout(() => {
                            const btn = document.getElementById('send-btn');
                            if (btn) btn.click();
                          }, 100);
                        }}
                      />
                    </div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* 输入框 - 精致日式风格 */}
          <div className="flex-none w-full pb-6 pt-4 px-4" style={{ background: 'linear-gradient(to top, rgba(251,250,245,0.95), rgba(251,250,245,0))' }}>
            <div className="max-w-3xl mx-auto w-full relative">
              <div className="relative flex items-end gap-3 rounded-2xl transition-all duration-300"
                style={{
                  background: 'rgba(255,255,255,0.7)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(180,170,150,0.2)',
                  boxShadow: '0 4px 30px rgba(180,170,150,0.1), inset 0 1px 0 rgba(255,255,255,0.8)'
                }}
              >
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
                  onClick={sendMessage}
                  disabled={isLoading || !input.trim()}
                  className="mb-2 mr-2 p-3 rounded-xl transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed"
                  style={{
                    background: input.trim() && !isLoading
                      ? 'linear-gradient(135deg, #2d3436 0%, #636e72 100%)'
                      : 'rgba(200,200,200,0.3)',
                    boxShadow: input.trim() && !isLoading
                      ? '0 2px 12px rgba(45,52,54,0.3)'
                      : 'none'
                  }}
                >
                  {isLoading ? (
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  ) : (
                    <svg className="w-4 h-4" fill="none" stroke="white" viewBox="0 0 24 24" style={{ opacity: input.trim() ? 1 : 0.5 }}>
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  )}
                </button>
              </div>
              <div className="flex items-center justify-center gap-4 mt-3" style={{ color: '#b0b0b0', fontSize: '11px' }}>
                <span>Enter 发送</span>
                <span style={{ opacity: 0.5 }}>·</span>
                <span>Shift + Enter 换行</span>
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
      />

      <KnowledgePanel
        sessionId={currentSession}
        isOpen={showKnowledgePanel}
        onClose={() => setShowKnowledgePanel(false)}
      />

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
