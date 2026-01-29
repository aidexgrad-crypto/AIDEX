"use client";

export default function AuthPage() {
  return (
    <div data-auth-page className="fixed inset-0 flex font-sans">

      {/* LEFT HALF — IMAGE */}
      <div
        className="hidden md:flex w-1/2 flex-shrink-0 bg-cover bg-center"
        style={{
          backgroundImage: "url('/login-bg.jpg')",
        }}
      >
        <div className="w-full h-full bg-black/50" />
      </div>

      {/* RIGHT HALF — LOGIN */}
      <div className="flex w-full md:w-1/2 flex-shrink-0 items-center justify-center bg-[#0B1220] text-white">
        <div className="w-full max-w-md rounded-2xl bg-white/5 backdrop-blur-xl border border-white/10 p-8 shadow-2xl">

          <h1 className="text-3xl font-bold tracking-tight">AIDEX</h1>
          <p className="text-sm text-white/60 mb-6">
            Login to continue to AIDEX
          </p>

          <div className="space-y-4">
            <input
              type="email"
              placeholder="Email"
              className="w-full rounded-xl bg-white/10 border border-white/10 px-4 py-3 text-sm outline-none focus:border-white/30"
            />

            <input
              type="password"
              placeholder="Password"
              className="w-full rounded-xl bg-white/10 border border-white/10 px-4 py-3 text-sm outline-none focus:border-white/30"
            />

            <button className="w-full rounded-xl bg-white text-black py-3 font-medium hover:bg-white/90 transition">
              Login
            </button>
          </div>

          <p className="mt-4 text-center text-xs text-white/40">
            UI-only auth (backend later)
          </p>

        </div>
      </div>
    </div>
  );
}
