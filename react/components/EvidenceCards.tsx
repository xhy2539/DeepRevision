type EvidenceCard = {
  source?: string;
  page?: string | number;
  quote?: string;
  confidence?: number;
  grounded?: boolean;
};

type EvidenceCardsProps = {
  cards?: EvidenceCard[];
  status?: string;
  route?: string;
};

export default function EvidenceCards({ cards, status, route }: EvidenceCardsProps) {
  const safeCards = (cards || []).filter((card) => card && (card.source || card.quote));
  if (safeCards.length === 0 && route !== "rag") return null;

  if (safeCards.length === 0) {
    if (status && status !== "none") return null;
    return (
      <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-left text-xs text-amber-800">
        未找到可靠课件证据。建议上传或补充相关课件后再追问。
      </div>
    );
  }

  return (
    <div className="mt-3 rounded-xl border border-sky-100 bg-sky-50/70 p-3 text-left">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="text-xs font-semibold text-sky-800">课件证据</div>
        <div className="rounded-full border border-sky-200 bg-white px-2 py-0.5 text-[10px] text-sky-700">
          {status === "grounded" ? "已核验" : "参考依据"}
        </div>
      </div>
      <div className="space-y-2">
        {safeCards.slice(0, 3).map((card, idx) => (
          <div key={`${card.source || "source"}-${idx}`} className="rounded-lg border border-sky-100 bg-white px-3 py-2">
            <div className="flex items-center justify-between gap-2">
              <div className="truncate text-[11px] font-semibold text-slate-700">{card.source || `课件片段 ${idx + 1}`}</div>
              {card.page ? <div className="text-[10px] text-slate-400">p.{card.page}</div> : null}
            </div>
            {card.quote ? (
              <blockquote className="mt-1 border-l-2 border-sky-200 pl-2 text-[11px] leading-relaxed text-slate-600">
                {card.quote}
              </blockquote>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
}
