"use client";

import { usePathname, useRouter } from "next/navigation";

const navItems = [
  { label: "Home", href: "/" },
  { label: "Data Overview", href: "/overview" },
  { label: "Preprocessing", href: "/preprocessing" },
  { label: "Profile", href: "/profile" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();

  return (
    <aside className="w-[260px] min-h-screen bg-[#123B7A] text-white flex flex-col">
      {/* Logo */}
      <div className="px-6 py-5 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-white/15 flex items-center justify-center font-bold">
            A
          </div>
          <div>
            <p className="font-bold leading-none text-lg">AIDEX</p>
            <p className="text-xs text-white/70 leading-none mt-1">
              Automated & Explainable Data Science
            </p>
          </div>
        </div>
      </div>

      {/* Menu */}
      <nav className="flex-1 px-4 py-4 space-y-2">
        {navItems.map((item) => {
          const active = pathname === item.href;
          return (
            <button
              key={item.href}
              onClick={() => router.push(item.href)}
              className={`w-full text-left rounded-xl px-4 py-3 text-sm font-medium transition ${
                active
                  ? "bg-white/15"
                  : "hover:bg-white/10 text-white/90"
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
