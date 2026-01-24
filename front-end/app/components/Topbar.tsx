"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

export default function Topbar() {
  const router = useRouter();
  const [userEmail, setUserEmail] = useState("");

  useEffect(() => {
    const email = localStorage.getItem("aidex_user") || "";
    setUserEmail(email);
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("aidex_auth");
    localStorage.removeItem("aidex_user");
    router.replace("/auth");
  };

  return (
    <header className="w-full bg-white border-b border-slate-200">
      <div className="h-16 px-6 flex items-center justify-between">
        <h1 className="text-sm font-semibold text-slate-900">Dashboard</h1>

        <div className="flex items-center gap-3">
          <div className="px-4 py-2 rounded-xl border border-slate-200 text-sm text-slate-700 bg-white">
            {userEmail || "User"}
          </div>

          <button
            onClick={handleLogout}
            className="rounded-xl bg-blue-600 text-white px-5 py-2 text-sm font-semibold hover:bg-blue-700 transition"
          >
            Logout
          </button>
        </div>
      </div>
    </header>
  );
}

