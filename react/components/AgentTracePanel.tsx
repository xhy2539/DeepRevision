import { getAgentRouteLabel } from "./AgentRouteBadge";

type AgentTracePanelProps = {
  meta?: Record<string, unknown>;
};

function boolValue(meta: Record<string, unknown>, key: string): boolean {
  return meta[key] === true || meta[key] === "true";
}

function textValue(meta: Record<string, unknown>, key: string): string {
  const value = meta[key];
  return typeof value === "string" || typeof value === "number" ? String(value) : "";
}

function buildTraceLines(meta: Record<string, unknown>): string[] {
  const lines: string[] = [];
  const route = textValue(meta, "route");
  if (route && route !== "pending") {
    lines.push(`Supervisor 路由到：${getAgentRouteLabel(meta)}`);
  }

  const quizAction = textValue(meta, "quiz_action");
  if (quizAction === "answer_check") {
    lines.push("识别为作答判分：读取上一轮题目 payload，未重新出题。");
  }
  if (boolValue(meta, "deterministic_followup")) {
    lines.push("命中确定性跟进识别：本轮没有交给 LLM 自由改写意图。");
  }
  if (boolValue(meta, "context_reference_resolved")) {
    lines.push("已解析上下文指代：按最近任务语境继续回答。");
  }
  if (boolValue(meta, "ops_deterministic")) {
    lines.push("命中确定性工具分支：直接调用后端工具，避免口头宣告结果。");
  }
  if (boolValue(meta, "danger_confirmation_required")) {
    lines.push("危险操作已拦截：必须二次确认后才会执行。");
  }
  if (boolValue(meta, "planner_data_driven")) {
    lines.push("计划由练习画像驱动：已读取错题、薄弱点或 mastery 优先级。");
  }
  if (boolValue(meta, "structured")) {
    lines.push("已生成结构化消息：前端可按题卡、试卷或 Markdown 渲染。");
  }

  const pendingTool = textValue(meta, "pending_tool");
  if (pendingTool) {
    lines.push(`待确认工具：${pendingTool}`);
  }
  const requiredPhrase = textValue(meta, "required_confirmation_phrase");
  if (requiredPhrase) {
    lines.push(`确认短语：${requiredPhrase}`);
  }
  const evidenceSource = textValue(meta, "evidence_source");
  if (evidenceSource) {
    lines.push(`证据来源：${evidenceSource}`);
  }

  return lines;
}

export default function AgentTracePanel({ meta }: AgentTracePanelProps) {
  if (!meta) return null;
  const lines = buildTraceLines(meta);
  if (!lines.length) return null;

  return (
    <details className="mt-2 rounded-xl border border-slate-200 bg-white/80 px-3 py-2 text-left text-xs text-slate-600 shadow-sm">
      <summary className="cursor-pointer select-none font-semibold text-slate-700">
        Agent 执行轨迹 ({lines.length})
      </summary>
      <ul className="mt-2 space-y-1.5">
        {lines.map((line) => (
          <li key={line} className="flex gap-2">
            <span className="mt-1 h-1.5 w-1.5 flex-none rounded-full bg-teal-500" />
            <span>{line}</span>
          </li>
        ))}
      </ul>
    </details>
  );
}
