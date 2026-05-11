type PriorityReviewPoint = {
  knowledge_point?: string;
  mastery_score?: number;
  attempt_count?: number;
  correct_count?: number;
  recent_wrong_streak?: number;
  review_urgency?: number;
};

type PracticeStats = {
  total_attempts?: number;
  wrong_attempts?: number;
  accuracy?: number;
  retry_rate?: number;
  weak_points?: string[];
  mastery_rows_total?: number;
  priority_review_points?: PriorityReviewPoint[];
};

type PracticeSummary = {
  score: number;
  total: number;
  wrongPoints: string[];
  saved: number;
} | null;

type LearningLoopState = {
  phase?: string;
  status?: string;
  current_day?: number;
  next_action?: string;
  review_decision?: string;
};

type LearningLoopPanelProps = {
  stats: PracticeStats | null;
  masteryCount: number;
  lastPractice: PracticeSummary;
  loopState?: LearningLoopState | null;
  disabled?: boolean;
  onSendPrompt: (prompt: string) => void;
};

function fmtPercent(value?: number): string {
  return `${Number(value || 0).toFixed(1)}%`;
}

function fmtScore(value?: number): string {
  return Number.isFinite(value) ? Number(value).toFixed(2) : "0.00";
}

function formatPhaseLabel(phase: string): string {
  const match = phase.match(/^day(\d+)_(quiz_issued|answered|reviewed)$/);
  if (match) {
    const [, day, status] = match;
    if (status === "quiz_issued") return `Day${day} 练习中`;
    if (status === "answered") return `Day${day} 待复盘`;
    return `Day${day} 已复盘`;
  }
  const phaseLabelMap: Record<string, string> = {
    planned: "已规划",
    cycle_completed: "周期完成",
    remediation_quiz_issued: "补救练习中",
    remediation_answered: "补救待复盘",
    remediation_reviewed: "补救已复盘",
    answered: "待复盘",
  };
  return phase ? (phaseLabelMap[phase] || phase) : "未启动";
}

export default function LearningLoopPanel({
  stats,
  masteryCount,
  lastPractice,
  loopState,
  disabled,
  onSendPrompt,
}: LearningLoopPanelProps) {
  const priority = (stats?.priority_review_points || [])
    .filter((item) => String(item?.knowledge_point || "").trim())
    .slice(0, 3);
  const weakPoints = (stats?.weak_points || []).slice(0, 5);
  const firstWeakPoint =
    priority[0]?.knowledge_point || weakPoints[0] || lastPractice?.wrongPoints?.[0] || "";
  const phase = String(loopState?.phase || "").trim();
  const phaseLabel = formatPhaseLabel(phase);
  const nextAction = String(loopState?.next_action || "").trim();
  const continueLabel =
    phase === "cycle_completed"
      ? "开启新周期"
      : /^day\d+_reviewed$/.test(phase) || phase === "remediation_reviewed"
      ? (loopState?.review_decision === "remedial_practice" ? "生成补救题" : "生成下一题")
      : /^day\d+_answered$/.test(phase) || phase === "remediation_answered"
      ? "进入复盘"
      : "继续闭环";

  return (
    <section className="rounded-[1.35rem] border border-slate-200/70 bg-white/85 p-4 shadow-[0_18px_45px_rgba(15,23,42,0.06)] backdrop-blur">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold tracking-[0.16em] text-teal-700">LOOP</div>
          <div className="mt-1 text-sm font-bold text-slate-900">学习闭环</div>
        </div>
        <div className="rounded-full bg-slate-900 px-3 py-1 text-[11px] font-semibold text-white">
          {masteryCount || stats?.mastery_rows_total || 0} 考点
        </div>
      </div>

      <div className="mt-3 rounded-2xl border border-teal-100 bg-teal-50/70 px-3 py-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[11px] font-semibold text-teal-700">当前阶段</span>
          <span className="rounded-full bg-white px-2 py-0.5 text-[11px] font-semibold text-teal-800">
            {phaseLabel}
          </span>
        </div>
        <div className="mt-1.5 line-clamp-2 text-[11px] leading-5 text-teal-900">
          {nextAction || (phase ? "按当前闭环阶段继续推进。" : "点击开始自主复习生成 Day1 任务。")}
        </div>
      </div>

      <div className="mt-4 grid grid-cols-3 divide-x divide-slate-200 rounded-2xl bg-slate-50/80 px-2 py-2">
        <div className="px-2">
          <div className="text-[10px] text-slate-500">练习数</div>
          <div className="mt-0.5 text-base font-bold text-slate-900">{stats?.total_attempts || 0}</div>
        </div>
        <div className="px-2">
          <div className="text-[10px] text-slate-500">正确率</div>
          <div className="mt-0.5 text-base font-bold text-emerald-700">{fmtPercent(stats?.accuracy)}</div>
        </div>
        <div className="px-2">
          <div className="text-[10px] text-slate-500">错题</div>
          <div className="mt-0.5 text-base font-bold text-amber-700">{stats?.wrong_attempts || 0}</div>
        </div>
      </div>

      {lastPractice && (
        <div className="mt-3 rounded-2xl bg-teal-50 px-3 py-2 text-xs font-medium text-teal-800">
          最近 {lastPractice.score}/{lastPractice.total}
          {lastPractice.wrongPoints.length > 0 ? ` · ${lastPractice.wrongPoints[0]}` : " · 全对"}
        </div>
      )}

      <div className="mt-4">
        <div className="flex items-center justify-between">
          <div className="text-[11px] font-semibold text-slate-500">优先复习</div>
          <div className="text-[10px] text-slate-400">mastery</div>
        </div>
        {priority.length > 0 ? (
          <div className="mt-2 space-y-2">
            {priority.slice(0, 2).map((item, idx) => (
              <div key={`${item.knowledge_point}-${idx}`} className="rounded-2xl bg-slate-50 px-3 py-2.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-sm font-semibold text-slate-900">{item.knowledge_point}</span>
                  <span className="font-mono text-xs text-amber-700">{fmtScore(item.mastery_score)}</span>
                </div>
                <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-200">
                  <div
                    className="h-full rounded-full bg-teal-500"
                    style={{ width: `${Math.max(4, Math.min(100, Number(item.mastery_score || 0) * 100))}%` }}
                  />
                </div>
                <div className="mt-1.5 text-[11px] text-slate-500">
                  {item.correct_count || 0}/{item.attempt_count || 0} 正确
                  {item.recent_wrong_streak ? ` · 连错 ${item.recent_wrong_streak}` : ""}
                </div>
              </div>
            ))}
          </div>
        ) : weakPoints.length > 0 ? (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {weakPoints.map((kp) => (
              <span key={kp} className="rounded-full bg-amber-50 px-2 py-0.5 text-[11px] text-amber-800">
                {kp}
              </span>
            ))}
          </div>
        ) : (
          <div className="mt-2 rounded-2xl border border-dashed border-slate-200 px-3 py-3 text-[11px] text-slate-500">
            完成题卡后生成画像。
          </div>
        )}
      </div>

      <div className="mt-4 space-y-2">
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            disabled={disabled}
            onClick={() => onSendPrompt("请进入学习闭环：根据我的练习画像和课件，安排今天的复习任务，并给我 Day1 练习建议。")}
            className="rounded-full bg-teal-600 px-3 py-2 text-center text-xs font-semibold text-white transition hover:bg-teal-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            开始自主复习
          </button>
          <button
            type="button"
            disabled={disabled}
            onClick={() => onSendPrompt("继续学习闭环，进入下一步。")}
            className="rounded-full bg-emerald-100 px-3 py-2 text-center text-xs font-semibold text-emerald-900 transition hover:bg-emerald-200 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {continueLabel}
          </button>
        </div>
        <div className="grid grid-cols-2 gap-2">
        <button
          type="button"
          disabled={disabled}
          onClick={() => onSendPrompt("根据我最近错题制定3天复习计划。")}
          className="rounded-full bg-slate-900 px-3 py-2 text-center text-xs font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          3天计划
        </button>
        <button
          type="button"
          disabled={disabled || !firstWeakPoint}
          onClick={() => onSendPrompt(firstWeakPoint ? `针对${firstWeakPoint}再出1道选择题。` : "针对第一个薄弱点再出1道选择题。")}
          className="rounded-full bg-amber-100 px-3 py-2 text-center text-xs font-semibold text-amber-900 transition hover:bg-amber-200 disabled:cursor-not-allowed disabled:opacity-50"
        >
          再练一题
        </button>
        </div>
      </div>
    </section>
  );
}
