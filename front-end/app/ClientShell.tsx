"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import Sidebar from "./components/Sidebar";
import Topbar from "./components/Topbar";

export default function ClientShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();

  const isAuthPage = pathname === "/auth";

  useEffect(() => {
    if (isAuthPage) return;
    const isAuthed = localStorage.getItem("aidex_auth") === "true";
    if (!isAuthed) router.replace("/auth");
  }, [isAuthPage, router]);

  if (isAuthPage) return <>{children}</>;

  return (
  <div className="flex min-h-screen w-full bg-[#0b1020] text-white">
    {/* Sidebar */}
    <Sidebar />

    {/* Main area */}
    <div className="mx-auto w-full max-w-5xl px-6 py-6">

      <Topbar />

      {/* ✅ Page container like before (not too wide, not too padded) */}
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-6xl px-6 py-6">
          {children}
        </div>
      </main>
    </div>
  </div>
);

}