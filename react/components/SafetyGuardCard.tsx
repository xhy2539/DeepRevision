type SafetyGuardCardProps = {
  meta?: Record<string, unknown>;
};

export default function SafetyGuardCard({ meta }: SafetyGuardCardProps) {
  const required = meta?.danger_confirmation_required === true || meta?.danger_confirmation_required === "true";
  if (!required) return null;

  const phrase = typeof meta?.required_confirmation_phrase === "string" ? meta.required_confirmation_phrase : "";
  return (
    <div className="mt-3 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-left text-xs text-red-800">
      <div className="font-semibold">安全闸已拦截危险操作</div>
      <div className="mt-1 text-red-700">
        系统没有执行删除/清空动作。必须输入指定确认短语后才会继续。
      </div>
      {phrase ? (
        <div className="mt-2 rounded-lg border border-red-200 bg-white px-2 py-1 font-mono text-[11px] text-red-700">
          {phrase}
        </div>
      ) : null}
    </div>
  );
}
