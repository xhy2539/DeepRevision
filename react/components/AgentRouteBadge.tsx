type AgentRouteBadgeProps = {
  meta?: Record<string, unknown>;
};

const ROUTE_STYLES: Record<string, { label: string; className: string }> = {
  rag: {
    label: "知识检索 Agent",
    className: "border-sky-200 bg-sky-50 text-sky-700",
  },
  quiz: {
    label: "出题/判分 Agent",
    className: "border-emerald-200 bg-emerald-50 text-emerald-700",
  },
  exam: {
    label: "组卷 Agent",
    className: "border-indigo-200 bg-indigo-50 text-indigo-700",
  },
  ops: {
    label: "工具操作 Agent",
    className: "border-amber-200 bg-amber-50 text-amber-700",
  },
  planner: {
    label: "复习规划 Agent",
    className: "border-teal-200 bg-teal-50 text-teal-700",
  },
  history: {
    label: "学习分析 Agent",
    className: "border-cyan-200 bg-cyan-50 text-cyan-700",
  },
  learning_loop: {
    label: "学习闭环 Agent",
    className: "border-teal-200 bg-teal-50 text-teal-700",
  },
  chitchat: {
    label: "对话 Agent",
    className: "border-slate-200 bg-slate-50 text-slate-600",
  },
};

export function getAgentRouteLabel(meta?: Record<string, unknown>): string {
  const route = String(meta?.route || "").trim().toLowerCase();
  return ROUTE_STYLES[route]?.label || "Agent";
}

export default function AgentRouteBadge({ meta }: AgentRouteBadgeProps) {
  const route = String(meta?.route || "").trim().toLowerCase();
  if (!route || route === "pending") return null;

  const config = ROUTE_STYLES[route] || {
    label: `${route} Agent`,
    className: "border-slate-200 bg-slate-50 text-slate-600",
  };

  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[0.68rem] font-semibold ${config.className}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current opacity-70" />
      {config.label}
    </span>
  );
}
