 "use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import Sidebar from "./components/Sidebar";
import Topbar from "./components/Topbar";

export default function ClientShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();

  const isAuthPage = pathname === "/auth";

  // Protect pages except /auth
  useEffect(() => {
    if (isAuthPage) return;
    const isAuthed = localStorage.getItem("aidex_auth") === "true";
    if (!isAuthed) router.replace("/auth");
  }, [isAuthPage, router]);

  // Auth page should NOT show sidebar/topbar
  if (isAuthPage) return <>{children}</>;

  return (
    <div className="aidex-app-shell">
      <Sidebar />
      <div className="aidex-main">
        <Topbar />
        <div className="aidex-content">{children}</div>
      </div>
    </div>
  );
}
