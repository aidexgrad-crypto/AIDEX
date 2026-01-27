"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

type User = {
  name: string;
  email: string;
  password: string;
};

export default function AuthPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");

  /* =========================
     HELPERS
  ========================= */

  const isValidEmail = (value: string) =>
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);

  const getUsers = (): User[] =>
    JSON.parse(localStorage.getItem("aidex_users") || "[]");

  const saveUsers = (users: User[]) =>
    localStorage.setItem("aidex_users", JSON.stringify(users));

  /* =========================
     REGISTER
  ========================= */

  const handleRegister = () => {
    setError("");

    if (!name || !email || !password || !confirm) {
      setError("All fields are required.");
      return;
    }

    if (!isValidEmail(email)) {
      setError("Please enter a valid email address.");
      return;
    }

    if (password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }

    if (password !== confirm) {
      setError("Passwords do not match.");
      return;
    }

    const users = getUsers();

    if (users.some((u) => u.email === email)) {
      setError("An account with this email already exists.");
      return;
    }

    users.push({ name, email, password });
    saveUsers(users);

    alert("Account created successfully!");
    setMode("login");
    setName("");
    setEmail("");
    setPassword("");
    setConfirm("");
  };

  /* =========================
     LOGIN
  ========================= */

  const handleLogin = () => {
    setError("");

    if (!email || !password) {
      setError("Please enter email and password.");
      return;
    }

    if (!isValidEmail(email)) {
      setError("Invalid email format.");
      return;
    }

    const users = getUsers();
    const user = users.find(
      (u) => u.email === email && u.password === password
    );

    if (!user) {
      setError("Invalid email or password. Please register first.");
      return;
    }

    localStorage.setItem("aidex_auth", "true");
    localStorage.setItem("aidex_user_email", user.email);

    router.push("/");
  };

  return (
    <div className="auth-root">
      <div className="auth-image">
        <div className="auth-image-overlay" />
      </div>

      <div className="auth-login">
        <div className="auth-card">
          <h1 className="auth-title">AIDEX</h1>
          <p className="auth-subtitle">
            {mode === "login"
              ? "Login to continue to AIDEX"
              : "Create your AIDEX account"}
          </p>

          {/* TOGGLE */}
          <div className="auth-toggle">
            <button
              className={mode === "login" ? "active" : ""}
              onClick={() => {
                setMode("login");
                setError("");
              }}
            >
              Login
            </button>
            <button
              className={mode === "register" ? "active" : ""}
              onClick={() => {
                setMode("register");
                setError("");
              }}
            >
              Register
            </button>
          </div>

          {/* REGISTER EXTRA */}
          {mode === "register" && (
            <input
              className="auth-input"
              placeholder="Full name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          )}

          <input
            className="auth-input"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />

          <input
            className="auth-input"
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />

          {mode === "register" && (
            <input
              className="auth-input"
              type="password"
              placeholder="Confirm password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
            />
          )}

          {error && (
            <p style={{ color: "#fca5a5", fontSize: 13, marginBottom: 8 }}>
              {error}
            </p>
          )}

          {mode === "login" ? (
            <button className="auth-button" onClick={handleLogin}>
              Login
            </button>
          ) : (
            <button className="auth-button" onClick={handleRegister}>
              Create account
            </button>
          )}

          <p className="auth-footer">Client-side validation only</p>
        </div>
      </div>
    </div>
  );
}


