import type { Metadata } from "next";
import "./globals.css";
import { Inter, Noto_Serif_SC } from "next/font/google";
import { cn } from "@/lib/utils";

const inter = Inter({subsets:['latin'],variable:'--font-sans'});
const notoSerif = Noto_Serif_SC({
  weight: ['400', '600', '700'],
  subsets: ['latin'],
  variable:'--font-serif'
});

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
    <html lang="zh-CN" className={cn("font-sans", inter.variable, notoSerif.variable)}>
      <body>{children}</body>
    </html>
  );
}
