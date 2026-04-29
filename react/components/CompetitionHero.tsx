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
    { label: "练习", value: totalAttempts },
    { label: "薄弱", value: weakCount },
    { label: "考点", value: masteryCount },
  ];

  return (
    <section className="border-b border-slate-200/70 bg-white/70 px-4 py-2 backdrop-blur md:px-6">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4">
        <div className="min-w-0">
          <div className="truncate text-xs font-semibold text-slate-800">
            {sessionName || "默认科目"}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-4">
          {items.map((item) => (
            <div key={item.label} className="text-right">
              <div className="text-[10px] text-slate-400">{item.label}</div>
              <div className="text-sm font-bold text-slate-900">{item.value}</div>
            </div>
          ))}
          <div className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${
            backendReady ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-600"
          }`}>
            {backendReady ? "在线" : "离线"}
          </div>
        </div>
      </div>
    </section>
  );
}
