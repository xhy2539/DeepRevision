import { getAgentRouteLabel } from "./AgentRouteBadge";

type AgentTracePanelProps = {
  meta?: Record<string, unknown>;
  payload?: Record<string, unknown>;
};

const stepLabelMap: Record<string, string> = {
  intent: "识别目标",
  plan: "规划工具",
  action: "调用工具",
  observation: "观察结果",
  delegate: "委派专家",
  guard: "安全确认",
  final: "生成答复",
  specialist: "专家 Agent",
  critic: "质量校验",
};

const actionLabelMap: Record<string, string> = {
  tool: "工具",
  delegate: "委派",
  final: "回答",
  clarify: "澄清",
};

const statusLabelMap: Record<string, string> = {
  pending: "待执行",
  completed: "完成",
  delegated: "已委派",
  blocked: "已拦截",
  failed: "失败",
  partial: "部分完成",
  skipped: "跳过",
  success: "完成",
};

function boolValue(meta: Record<string, unknown>, key: string): boolean {
  return meta[key] === true || meta[key] === "true";
}

function textValue(meta: Record<string, unknown>, key: string): string {
  const value = meta[key];
  return typeof value === "string" || typeof value === "number" ? String(value) : "";
}

function agentTraceLines(meta: Record<string, unknown>): string[] {
  const raw = meta.agent_trace;
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((item): item is Record<string, unknown> => !!item && typeof item === "object")
    .map((item) => {
      const step = typeof item.step === "string" ? item.step : "";
      const agent = stepLabelMap[step] || (typeof item.agent === "string" ? item.agent : "Agent");
      const status = typeof item.status === "string" ? item.status : "running";
      const tool = typeof item.tool_name === "string" ? item.tool_name : typeof item.tool === "string" ? item.tool : "";
      const summary = typeof item.summary === "string" ? item.summary : "";
      const detail = [tool, summary].filter(Boolean).join("，");
      return `${agent} [${status}]：${detail || "已完成"}`;
    })
    .filter(Boolean);
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
  if (textValue(meta, "agent_mode") === "learning_loop") {
    lines.push("学习闭环已启动：读取画像、检索课件、生成计划并进行 Critic 校验。");
  }
  const loopPhase = textValue(meta, "learning_loop_phase");
  if (loopPhase) {
    lines.push(`学习闭环阶段：${loopPhase}`);
  }
  if (boolValue(meta, "structured")) {
    lines.push("已生成结构化消息：前端可按题卡、试卷或 Markdown 渲染。");
  }
  lines.push(...agentTraceLines(meta));

  const pendingTool = textValue(meta, "pending_tool");
  if (pendingTool) {
    lines.push(`待确认工具：${pendingTool}`);
  }
  const requiredPhrase = textValue(meta, "required_confirmation_phrase");
  if (requiredPhrase) {
    lines.push("确认方式：点击安全确认卡按钮，不需要输入确认短语。");
  }
  const evidenceSource = textValue(meta, "evidence_source");
  if (evidenceSource) {
    lines.push(`证据来源：${evidenceSource}`);
  }

  return lines;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function asRecordArray(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => !!item && typeof item === "object" && !Array.isArray(item))
    : [];
}

function taskPlanFromPayload(payload?: Record<string, unknown>): Record<string, unknown> | undefined {
  return asRecord(payload?.task_plan);
}

function planExecutionByStep(payload?: Record<string, unknown>): Map<string, Record<string, unknown>> {
  const map = new Map<string, Record<string, unknown>>();
  asRecordArray(payload?.plan_execution).forEach((item) => {
    const stepId = typeof item.step_id === "string" || typeof item.step_id === "number" ? String(item.step_id) : "";
    if (stepId) map.set(stepId, item);
  });
  return map;
}

function textFromRecord(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  return typeof value === "string" || typeof value === "number" ? String(value) : "";
}

function TaskPlanView({ payload }: { payload?: Record<string, unknown> }) {
  const plan = taskPlanFromPayload(payload);
  if (!plan) return null;

  const goal = textFromRecord(plan, "goal");
  const steps = asRecordArray(plan.steps);
  const execution = planExecutionByStep(payload);
  if (!goal && !steps.length) return null;

  return (
    <div className="mt-2 rounded-lg border border-teal-100 bg-teal-50/60 px-3 py-2">
      <div className="text-[0.68rem] font-semibold text-teal-800">任务计划</div>
      {goal && <div className="mt-1 text-xs leading-relaxed text-slate-700">{goal}</div>}
      {steps.length > 0 && (
        <ol className="mt-2 space-y-1.5">
          {steps.map((step, index) => {
            const id = textFromRecord(step, "id") || String(index + 1);
            const exec = execution.get(id);
            const title = textFromRecord(step, "title") || `步骤 ${id}`;
            const action = textFromRecord(step, "action_type");
            const tool = textFromRecord(step, "tool_name");
            const status = textFromRecord(exec || step, "status") || "pending";
            const summary = textFromRecord(exec || step, "summary") || textFromRecord(step, "expected_observation");
            return (
              <li key={`${id}-${title}`} className="rounded-md bg-white/80 px-2 py-1.5 text-xs text-slate-700">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="font-semibold text-slate-800">{id}. {title}</span>
                  {action && <span className="rounded border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[0.65rem] text-slate-500">{actionLabelMap[action] || action}</span>}
                  {tool && <span className="rounded border border-teal-100 bg-teal-50 px-1.5 py-0.5 font-mono text-[0.65rem] text-teal-700">{tool}</span>}
                  <span className="rounded border border-slate-200 bg-white px-1.5 py-0.5 text-[0.65rem] text-slate-500">{statusLabelMap[status] || status}</span>
                </div>
                {summary && <div className="mt-1 line-clamp-2 text-[0.68rem] leading-relaxed text-slate-500">{summary}</div>}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

export default function AgentTracePanel({ meta, payload }: AgentTracePanelProps) {
  if (!meta) return null;
  const lines = buildTraceLines(meta);
  const hasTaskPlan = !!taskPlanFromPayload(payload);
  if (!lines.length && !hasTaskPlan) return null;

  return (
    <details className="mt-2 rounded-xl border border-slate-200 bg-white/80 px-3 py-2 text-left text-xs text-slate-600 shadow-sm">
      <summary className="cursor-pointer select-none font-semibold text-slate-700">
        Agent 执行轨迹 ({lines.length})
      </summary>
      <TaskPlanView payload={payload} />
      {lines.length > 0 && (
        <ul className="mt-2 space-y-1.5">
          {lines.map((line) => (
            <li key={line} className="flex gap-2">
              <span className="mt-1 h-1.5 w-1.5 flex-none rounded-full bg-teal-500" />
              <span>{line}</span>
            </li>
          ))}
        </ul>
      )}
    </details>
  );
}
