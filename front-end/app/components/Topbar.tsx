"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";

export default function Topbar() {
  const router = useRouter();
  const pathname = usePathname();
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

  const title =
    pathname === "/"
      ? "Dashboard"
      : pathname === "/overview"
      ? "Data Overview"
      : pathname === "/preprocessing"
      ? "Preprocessing"
      : pathname === "/profile"
      ? "Profile"
      : "AIDEX";

  return (
    <header className="aidex-topbar2">
      <div className="aidex-topbar2-inner">
        <div>
          <p className="aidex-top-title">{title}</p>
          <p className="aidex-top-sub">
            Manage datasets, AutoML and explainability.
          </p>
        </div>

        <div className="aidex-top-actions">
          <button
            className="aidex-userpill"
            onClick={() => router.push("/profile")}
            title="Go to profile"
          >
            <span className="aidex-avatar">
              {userEmail ? userEmail[0].toUpperCase() : "U"}
            </span>
            <span className="aidex-email">
              {userEmail || "User"}
            </span>
          </button>

          <button className="aidex-logout" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </div>
    </header>
  );
}
