import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Deep Revision | AI 期末复习引擎",
  description: "AI 期末复习引擎 - RAG 驱动",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
