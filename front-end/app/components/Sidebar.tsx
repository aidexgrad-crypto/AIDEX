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
    <aside className="aidex-sidebar2">
      {/* Brand */}
      <div className="aidex-sidebar2-header">
        <div className="aidex-logo2">A</div>
        <div>
          <p className="aidex-brand2">AIDEX</p>
          <p className="aidex-subtitle2">
            Automated & Explainable Data Science
          </p>
        </div>
      </div>

      {/* Nav */}
      <nav className="aidex-nav2">
        {navItems.map((item) => {
          const active = pathname === item.href;

          return (
            <button
              key={item.href}
              onClick={() => router.push(item.href)}
              className={`aidex-nav2-item ${active ? "active" : ""}`}
            >
              <span>{item.label}</span>
              {active && <span className="aidex-active-pill">Active</span>}
            </button>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="aidex-sidebar2-footer">
        <div className="aidex-tip">
          Upload your dataset and AIDEX will detect the type automatically.
        </div>
      </div>
    </aside>
  );
}
