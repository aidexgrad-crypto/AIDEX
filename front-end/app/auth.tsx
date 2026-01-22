"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Auth() {
  const router = useRouter();

  const [mode, setMode] = useState<"login" | "signup">("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (mode === "signup" && !name.trim()) {
      alert("Please enter your name ✅");
      return;
    }

    if (!email.trim() || !password.trim()) {
      alert("Please fill in all fields ✅");
      return;
    }

    // ✅ Create session (frontend only)
    localStorage.setItem("aidex_auth", "true");
    localStorage.setItem("aidex_user", email);

    // ✅ Go to main page.tsx
    router.push("/");
  };

  return (
    <main className="min-h-screen bg-[#0B1020] text-white flex items-center justify-center px-6">
      <div className="w-full max-w-md rounded-3xl border border-white/10 bg-white/5 shadow-[0_0_0_1px_rgba(255,255,255,0.05)] backdrop-blur-xl p-6 sm:p-8">
        <h1 className="text-3xl font-bold tracking-tight">AIDEX</h1>
        <p className="text-white/60 text-sm mt-1">
          {mode === "login"
            ? "Login to continue to AIDEX."
            : "Create your account to start using AIDEX."}
        </p>

        {/* Tabs */}
        <div className="mt-6 grid grid-cols-2 rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
          <button
            onClick={() => setMode("login")}
            className={`py-3 text-sm font-semibold transition ${
              mode === "login" ? "bg-white text-black" : "hover:bg-white/10"
            }`}
          >
            Login
          </button>
          <button
            onClick={() => setMode("signup")}
            className={`py-3 text-sm font-semibold transition ${
              mode === "signup" ? "bg-white text-black" : "hover:bg-white/10"
            }`}
          >
            Sign up
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="mt-6 space-y-4">
          {mode === "signup" && (
            <div>
              <label className="text-sm text-white/70">Full Name</label>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Your name"
                className="mt-2 w-full rounded-xl bg-white/5 border border-white/10 px-4 py-3 text-sm outline-none text-white placeholder:text-white/40"
              />
            </div>
          )}

          <div>
            <label className="text-sm text-white/70">Email</label>
            <input
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="example@email.com"
              type="email"
              className="mt-2 w-full rounded-xl bg-white/5 border border-white/10 px-4 py-3 text-sm outline-none text-white placeholder:text-white/40"
            />
          </div>

          <div>
            <label className="text-sm text-white/70">Password</label>
            <input
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              type="password"
              className="mt-2 w-full rounded-xl bg-white/5 border border-white/10 px-4 py-3 text-sm outline-none text-white placeholder:text-white/40"
            />
          </div>

          <button
            type="submit"
            className="w-full rounded-xl bg-white text-black py-3 text-sm font-semibold hover:bg-white/90 transition"
          >
            {mode === "login" ? "Login" : "Create Account"}
          </button>

          <p className="text-xs text-white/50 text-center">
            UI-only auth (backend later)
          </p>
        </form>
      </div>
    </main>
  );
}
