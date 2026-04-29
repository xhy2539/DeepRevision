type CompetitionHeroProps = {
  sessionName: string;
  totalAttempts: number;
  weakCount: number;
  masteryCount: number;
  backendReady: boolean;
};

export default function CompetitionHero({
  sessionName,
  totalAttempts,
  weakCount,
  masteryCount,
  backendReady,
}: CompetitionHeroProps) {
  const items = [
    { label: "当前课程", value: sessionName || "默认科目" },
    { label: "学习闭环", value: `${totalAttempts} 次练习` },
    { label: "薄弱点", value: `${weakCount} 个` },
    { label: "mastery", value: `${masteryCount} 个考点` },
  ];

  return (
    <section className="border-b border-teal-100 bg-gradient-to-r from-teal-50 via-white to-amber-50 px-4 py-3 md:px-8">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-teal-700">期末冲刺驾驶舱</div>
          <h2 className="mt-1 text-sm font-bold text-slate-900">
            课件证据 &gt; 出题判分 &gt; mastery 更新 &gt; 个性化复习计划
          </h2>
        </div>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
          {items.map((item) => (
            <div key={item.label} className="rounded-xl border border-white/70 bg-white/75 px-3 py-2 shadow-sm">
              <div className="text-[10px] text-slate-500">{item.label}</div>
              <div className="mt-0.5 truncate text-xs font-bold text-slate-800">{item.value}</div>
            </div>
          ))}
          <div className="rounded-xl border border-white/70 bg-white/75 px-3 py-2 shadow-sm">
            <div className="text-[10px] text-slate-500">服务状态</div>
            <div className={`mt-0.5 text-xs font-bold ${backendReady ? "text-emerald-700" : "text-red-600"}`}>
              {backendReady ? "已连接" : "未连接"}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
