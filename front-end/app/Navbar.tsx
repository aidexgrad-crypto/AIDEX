"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

export default function Navbar() {
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
    <header className="sticky top-0 z-50 border-b border-white/10 bg-[#0B1020]/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        {/* Left */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-white/10 flex items-center justify-center">
            <span className="font-bold">A</span>
          </div>
          <div>
            <h1 className="font-bold text-lg leading-none">AIDEX</h1>
            <p className="text-xs text-white/50 leading-none">
              AutoML Dashboard
            </p>
          </div>
        </div>

        {/* Right */}
        <div className="flex items-center gap-3">
          {/* Profile */}
          <button
            onClick={() => router.push("/profile")}
            className="flex items-center gap-2 rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-sm hover:bg-white/10 transition"
          >
            👤
            <span className="hidden sm:inline">
              {userEmail ? userEmail : "Profile"}
            </span>
          </button>

          {/* Logout */}
          <button
            onClick={handleLogout}
            className="rounded-xl bg-white text-black px-4 py-2 text-sm font-semibold hover:bg-white/90 transition"
          >
            Logout
          </button>
        </div>
      </div>
    </header>
  );
}
