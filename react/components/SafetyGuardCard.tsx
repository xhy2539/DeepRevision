"use client";

import { useState } from "react";
import { AlertTriangle, Check, X } from "lucide-react";

type SafetyGuardCardProps = {
  meta?: Record<string, unknown>;
  disabled?: boolean;
  onConfirm?: (phrase: string) => void;
  onCancel?: () => void;
};

// 提取对象字段，避免危险操作说明里直接展示内部确认短语。
function getTextField(value: unknown, key: string): string {
  if (!value || typeof value !== "object") return "";
  const raw = (value as Record<string, unknown>)[key];
  return typeof raw === "string" ? raw : "";
}

// 根据后端 pending_tool 元数据生成用户可读的操作摘要。
function describePendingAction(meta: Record<string, unknown>): string {
  const toolName = typeof meta.pending_tool === "string" ? meta.pending_tool : "";
  const pendingArgs = meta.pending_args;
  const keepFilename = getTextField(pendingArgs, "keep_filename");
  const filename = getTextField(pendingArgs, "filename");

  if (toolName === "delete_knowledge_file_tool") {
    return filename ? `删除课件 ${filename}` : "删除课件";
  }
  if (toolName === "delete_duplicate_knowledge_files_tool") {
    return keepFilename ? `删除重复课件，保留 ${keepFilename}` : "删除重复课件";
  }
  if (toolName.includes("delete")) return "执行删除操作";
  if (toolName.includes("clear")) return "执行清空操作";
  return "执行高风险管理操作";
}

// 渲染危险操作安全确认卡，确认按钮只提交隐藏确认，不要求用户手动输入。
export default function SafetyGuardCard({ meta, disabled = false, onConfirm, onCancel }: SafetyGuardCardProps) {
  const [cancelled, setCancelled] = useState(false);
  const required = meta?.danger_confirmation_required === true || meta?.danger_confirmation_required === "true";
  if (!required) return null;

  const phrase = typeof meta?.required_confirmation_phrase === "string" ? meta.required_confirmation_phrase : "";
  const actionText = describePendingAction(meta || {});

  const handleCancel = () => {
    setCancelled(true);
    onCancel?.();
  };

  return (
    <div className="mt-3 rounded-xl border border-red-200 bg-red-50 px-3 py-3 text-left text-xs text-red-800 shadow-sm">
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 flex-none text-red-600" />
        <div className="min-w-0">
          <div className="font-semibold">需要确认高风险操作</div>
          <div className="mt-1 text-red-700">
            系统尚未执行：{actionText}。请选择是否继续。
          </div>
        </div>
      </div>
      {cancelled ? (
        <div className="mt-3 rounded-lg border border-slate-200 bg-white px-2 py-1 text-slate-600">
          已取消，本次操作未执行。
        </div>
      ) : (
        <div className="mt-3 flex flex-wrap gap-2">
          <button
            type="button"
            disabled={disabled || !phrase}
            onClick={() => onConfirm?.(phrase)}
            className="inline-flex items-center gap-1 rounded-lg border border-red-600 bg-red-600 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Check className="h-3.5 w-3.5" />
            确认执行
          </button>
          <button
            type="button"
            disabled={disabled}
            onClick={handleCancel}
            className="inline-flex items-center gap-1 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <X className="h-3.5 w-3.5" />
            取消
          </button>
        </div>
      )}
    </div>
  );
}
