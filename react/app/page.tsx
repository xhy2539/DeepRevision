import { redirect } from "next/navigation";

export default function Home() {
  // 开发阶段跳过登录，直接跳转聊天页面
  redirect("/chat");
}
