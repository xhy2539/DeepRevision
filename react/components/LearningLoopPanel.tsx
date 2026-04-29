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

type LearningLoopPanelProps = {
  stats: PracticeStats | null;
  masteryCount: number;
  lastPractice: PracticeSummary;
  disabled?: boolean;
  onSendPrompt: (prompt: string) => void;
};

function fmtPercent(value?: number): string {
  return `${Number(value || 0).toFixed(1)}%`;
}

function fmtScore(value?: number): string {
  return Number.isFinite(value) ? Number(value).toFixed(2) : "0.00";
}

export default function LearningLoopPanel({
  stats,
  masteryCount,
  lastPractice,
  disabled,
  onSendPrompt,
}: LearningLoopPanelProps) {
  const priority = (stats?.priority_review_points || [])
    .filter((item) => String(item?.knowledge_point || "").trim())
    .slice(0, 3);
  const weakPoints = (stats?.weak_points || []).slice(0, 5);
  const firstWeakPoint =
    priority[0]?.knowledge_point || weakPoints[0] || lastPractice?.wrongPoints?.[0] || "";

  return (
    <section className="rounded-2xl border border-teal-100 bg-white/90 p-4 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-xs font-semibold text-teal-700">学习闭环</div>
          <div className="mt-1 text-[11px] text-slate-500">
            课件证据 &gt; 练习 &gt; mastery &gt; 计划 &gt; 再练习
          </div>
        </div>
        <div className="rounded-full border border-teal-100 bg-teal-50 px-2.5 py-1 text-[11px] font-semibold text-teal-700">
          {masteryCount || stats?.mastery_rows_total || 0} 个考点
        </div>
      </div>

      <div className="mt-3 grid grid-cols-3 gap-2">
        <div className="rounded-xl border border-slate-100 bg-slate-50 px-3 py-2">
          <div className="text-[10px] text-slate-500">练习数</div>
          <div className="mt-1 text-sm font-bold text-slate-800">{stats?.total_attempts || 0}</div>
        </div>
        <div className="rounded-xl border border-emerald-100 bg-emerald-50 px-3 py-2">
          <div className="text-[10px] text-emerald-700">正确率</div>
          <div className="mt-1 text-sm font-bold text-emerald-800">{fmtPercent(stats?.accuracy)}</div>
        </div>
        <div className="rounded-xl border border-amber-100 bg-amber-50 px-3 py-2">
          <div className="text-[10px] text-amber-700">错题</div>
          <div className="mt-1 text-sm font-bold text-amber-800">{stats?.wrong_attempts || 0}</div>
        </div>
      </div>

      {lastPractice && (
        <div className="mt-3 rounded-xl border border-cyan-100 bg-cyan-50/70 px-3 py-2 text-xs text-cyan-900">
          最近练习：{lastPractice.score}/{lastPractice.total}，已保存 {lastPractice.saved} 条记录
          {lastPractice.wrongPoints.length > 0 ? `，错点：${lastPractice.wrongPoints.slice(0, 2).join("、")}` : "，本轮全对"}
        </div>
      )}

      <div className="mt-4">
        <div className="text-[11px] font-semibold text-slate-600">优先复习点</div>
        {priority.length > 0 ? (
          <div className="mt-2 space-y-2">
            {priority.map((item, idx) => (
              <div key={`${item.knowledge_point}-${idx}`} className="rounded-xl border border-slate-100 bg-slate-50 px-3 py-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-xs font-semibold text-slate-800">{item.knowledge_point}</span>
                  <span className="text-[11px] text-amber-700">mastery {fmtScore(item.mastery_score)}</span>
                </div>
                <div className="mt-1 text-[11px] text-slate-500">
                  {item.correct_count || 0}/{item.attempt_count || 0} 正确
                  {item.recent_wrong_streak ? ` · 连错 ${item.recent_wrong_streak}` : ""}
                </div>
              </div>
            ))}
          </div>
        ) : weakPoints.length > 0 ? (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {weakPoints.map((kp) => (
              <span key={kp} className="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[11px] text-amber-800">
                {kp}
              </span>
            ))}
          </div>
        ) : (
          <div className="mt-2 rounded-xl border border-dashed border-slate-200 bg-slate-50 px-3 py-3 text-[11px] text-slate-500">
            暂无错题画像。完成一次题卡练习后，这里会显示薄弱点和复习优先级。
          </div>
        )}
      </div>

      <div className="mt-4 grid grid-cols-1 gap-2">
        <button
          type="button"
          disabled={disabled}
          onClick={() => onSendPrompt("根据我最近错题制定3天复习计划。")}
          className="rounded-xl border border-teal-200 bg-teal-50 px-3 py-2 text-left text-xs font-semibold text-teal-800 transition hover:bg-teal-100 disabled:cursor-not-allowed disabled:opacity-50"
        >
          根据错题制定3天计划
        </button>
        <button
          type="button"
          disabled={disabled || !firstWeakPoint}
          onClick={() => onSendPrompt(firstWeakPoint ? `针对${firstWeakPoint}再出1道选择题。` : "针对第一个薄弱点再出1道选择题。")}
          className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-left text-xs font-semibold text-amber-800 transition hover:bg-amber-100 disabled:cursor-not-allowed disabled:opacity-50"
        >
          针对第一个薄弱点再出1道题
        </button>
      </div>
    </section>
  );
}
