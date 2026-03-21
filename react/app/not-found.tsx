import Link from "next/link";

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50">
      <div className="text-center p-8">
        <h1 className="text-6xl font-bold text-slate-900 mb-4">404</h1>
        <h2 className="text-xl font-semibold text-slate-700 mb-2">页面未找到</h2>
        <p className="text-slate-600 mb-6">您访问的页面不存在</p>
        <Link
          href="/chat"
          className="inline-block px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition-colors"
        >
          返回聊天页面
        </Link>
      </div>
    </div>
  );
}
