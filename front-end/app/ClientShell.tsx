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
    <div className="min-h-screen bg-[#0b1020] text-white flex">
      {/* Sidebar */}
      <Sidebar />

      {/* Main */}
      <div className="flex-1 flex flex-col">
        <Topbar />

        <div className="p-10">{children}</div>
      </div>
    </div>
  );
}
