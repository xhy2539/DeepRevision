"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ExamCanvas from "@/components/ExamCanvas";

// Types
type MessageKind = "chat" | "quiz_set" | "exam_paper";

interface QuizQuestion {
  id?: number;
  type: string;
  stem?: string;
  question: string;
  options?: string[];
  answer: string;
  analysis?: string;
  explanation: string;
  score?: string | number;
  difficulty?: string;
  knowledge_points?: string[];
  evidence_snippets?: string[];
}

interface AssistantPayload {
  schema_version?: string;
  kind?: MessageKind;
  render_mode?: "markdown" | "interactive_cards" | "exam_canvas";
  title?: string;
  show_answers_default?: boolean;
  show_analysis_default?: boolean;
  questions?: QuizQuestion[];
  exam_data?: unknown;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  kind?: MessageKind;
  render_mode?: "markdown" | "interactive_cards" | "exam_canvas";
  schema_version?: string;
  payload?: AssistantPayload;
}

interface Session {
  id: string;
  name: string;
  parent_id?: string;
}

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

// 题型样式映射
const quizTypeStyles: Record<string, { tag: string; border: string; bg: string }> = {
  "选择题": { tag: "quiz-tag-choice", border: "border-l-blue-500", bg: "bg-blue-50" },
  "填空题": { tag: "quiz-tag-fill", border: "border-l-green-500", bg: "bg-green-50" },
  "判断题": { tag: "quiz-tag-judge", border: "border-l-orange-500", bg: "bg-orange-50" },
  "简答题": { tag: "quiz-tag-short", border: "border-l-purple-500", bg: "bg-purple-50" },
};

function QuizCard({
  questions,
  defaultShowAnswers = false,
}: {
  questions: QuizQuestion[];
  defaultShowAnswers?: boolean;
}) {
  const [showAnswers, setShowAnswers] = useState(defaultShowAnswers);

  return (
    <div className="space-y-4 my-4">
      {/* 答案显示控制按钮 */}
      <div className="flex justify-end mb-2">
        <button
          onClick={() => setShowAnswers(!showAnswers)}
          className="text-sm px-3 py-1.5 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 text-slate-600 transition-colors flex items-center gap-1"
        >
          <span>{showAnswers ? "👁 隐藏答案" : "👁 查看答案"}</span>
        </button>
      </div>

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
              <div className="flex items-center gap-3">
                {q.score !== undefined && q.score !== null && (
                  <span className="text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-medium">
                    {typeof q.score === "number" ? `${q.score}分` : q.score}
                  </span>
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

            <div className="text-slate-800 mb-4 whitespace-pre-wrap text-[15px] leading-relaxed font-medium">{q.question || q.stem}</div>

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

            {showAnswers && (q.analysis || q.explanation) && (
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <svg className="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="font-semibold text-slate-700 text-sm">解析</span>
                </div>
                <span className="text-slate-600 text-sm leading-relaxed">{q.analysis || q.explanation}</span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function MarkdownContent({
  content,
  kind,
  payload,
  onRequestAnswers,
}: {
  content: string;
  kind?: MessageKind;
  payload?: AssistantPayload;
  onRequestAnswers?: () => void;
}) {
  const schemaKind = payload?.kind || kind;
  const schemaRenderMode = payload?.render_mode;
  const effectiveKind = schemaKind;
  const effectiveRenderMode = schemaRenderMode || (effectiveKind === "quiz_set"
    ? "interactive_cards"
    : effectiveKind === "exam_paper"
    ? "exam_canvas"
    : "markdown");

  const markdownNode = (
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

  if (effectiveKind === "exam_paper" || effectiveRenderMode === "exam_canvas") {
    return (
      <ExamCanvas
        examContent={content}
        examDataOverride={payload?.exam_data || payload?.questions}
        courseName={payload?.title || "期末考试"}
        onRequestAnswers={onRequestAnswers}
      />
    );
  }

  if (effectiveKind === "quiz_set" || effectiveRenderMode === "interactive_cards") {
    const questions = payload?.questions && payload.questions.length > 0 ? payload.questions : [];
    if (questions.length > 0) {
      return <QuizCard questions={questions} defaultShowAnswers={payload?.show_answers_default} />;
    }
  }

  return markdownNode;
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

// ============== 会话管理模态框 ==============
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
      timestamp: Date.now(),
      kind: "chat",
      render_mode: "markdown",
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
                if (parsed.error) {
                  // 后端返回的错误，替换现有内容
                  setMessages(prev => prev.map(m =>
                    m.id === assistantMessageId ? { ...m, content: `[系统提示: ${parsed.error}]` } : m
                  ));
                  break;  // 退出流式读取
                }
                if (parsed.event === "start" && parsed.message) {
                  setMessages(prev => prev.map(m =>
                    m.id === assistantMessageId
                      ? {
                          ...m,
                          kind: parsed.message.kind || "chat",
                          render_mode: parsed.message.render_mode || "markdown",
                        }
                      : m
                  ));
                  continue;
                }
                if (parsed.event === "delta" && parsed.text) {
                  fullContent += parsed.text;
                  setMessages(prev => prev.map(m =>
                    m.id === assistantMessageId ? { ...m, content: fullContent } : m
                  ));
                  continue;
                }
                if (parsed.event === "complete" && parsed.message) {
                  const message = parsed.message;
                  fullContent = message.content || fullContent;
                  setMessages(prev => prev.map(m =>
                    m.id === assistantMessageId
                      ? {
                          ...m,
                          content: message.content || fullContent,
                          kind: message.kind || "chat",
                          render_mode: message.render_mode || "markdown",
                          payload: message.payload,
                        }
                      : m
                  ));
                  continue;
                }
                if (parsed.text) {
                  // 跳过 think 标签内容
                  if (parsed.text.trim().startsWith("<think>")) {
                    continue;
                  }
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
              } catch (e) {
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
      const errorMessage = error instanceof Error ? error.message : String(error);
      setMessages(prev => prev.map(m =>
        m.id === assistantMessageId ? { ...m, content: `请求失败: ${errorMessage}` } : m
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
  const createSession = async (name: string, parentId?: string) => {
    const sessionId = parentId ? `${parentId}_sub_${Date.now()}` : name;
    try {
      const res = await fetch("/api/chat/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, name, parent_id: parentId })
      });
      const data = await res.json();
      if (data.code === 200) {
        const newSession = { id: sessionId, name, parent_id: parentId };
        setSessions(prev => [...prev, newSession]);
        setCurrentSession(sessionId);
        setCurrentSessionName(name);
        setMessages([]);
        setThoughts([]);
      }
    } catch (error) {
      console.error("Create session error:", error);
      const newSession = { id: sessionId, name, parent_id: parentId };
      setSessions(prev => [...prev, newSession]);
      setCurrentSession(sessionId);
      setCurrentSessionName(name);
      setMessages([]);
      setThoughts([]);
    }
  };

  // 删除会话
  const deleteSession = async (id: string) => {
    try {
      await fetch(`/api/chat/session/${id}`, { method: "DELETE" });
    } catch (error) {
      console.error("Delete session error:", error);
    }
    setSessions(prev => prev.filter(s => s.id !== id));
    if (currentSession === id) {
      setCurrentSession("default");
      setCurrentSessionName("默认科目");
      setMessages([]);
      setThoughts([]);
    }
  };

  // 重命名会话
  const renameSession = async (id: string, newName: string) => {
    try {
      await fetch(`/api/chat/session/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_name: newName })
      });
    } catch (error) {
      console.error("Rename session error:", error);
    }
    setSessions(prev => prev.map(s => s.id === id ? { ...s, name: newName } : s));
    if (currentSession === id) {
      setCurrentSessionName(newName);
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
                const prompt = `请出一套综合测试卷题目，选择10道，判断5道，填空5道，简答3道。根据已上传的课件内容生成，考点范围请从课件中提取关键知识点。${sampleInfo}

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
                  key={`${message.id}-${idx}`}
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
                        kind={message.kind}
                        payload={message.payload}
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
        onDelete={deleteSession}
        onRename={renameSession}
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
