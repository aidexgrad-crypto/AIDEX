"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import ModernLayout from "../components/ModernLayout";

export default function DashboardPage() {
  const router = useRouter();
  const [stats, setStats] = useState({
    totalProjects: 0,
    activeModels: 0,
    totalPredictions: 0,
    avgAccuracy: 0,
  });

  useEffect(() => {
    const isAuthed = localStorage.getItem("aidex_auth") === "true";
    if (!isAuthed) router.replace("/auth");

    // Load stats from localStorage or API
    // For now, using mock data
    setStats({
      totalProjects: 12,
      activeModels: 8,
      totalPredictions: 1543,
      avgAccuracy: 94.2,
    });
  }, [router]);

  const quickActions = [
    { icon: "📊", label: "New Project", desc: "Start from scratch", color: "#6366f1", path: "/preprocessing" },
    { icon: "📁", label: "Load Dataset", desc: "Upload your data", color: "#8b5cf6", path: "/preprocessing" },
    { icon: "🤖", label: "Train Model", desc: "AutoML training", color: "#14b8a6", path: "/preprocessing" },
    { icon: "🔮", label: "Make Predictions", desc: "Use trained models", color: "#f59e0b", path: "/preprocessing" },
  ];

  const recentProjects = [
    { name: "Customer Churn", type: "Classification", accuracy: 96.5, date: "2 hours ago", status: "completed" },
    { name: "Sales Forecast", type: "Regression", r2: 0.89, date: "5 hours ago", status: "completed" },
    { name: "Image Recognition", type: "Image Classification", accuracy: 92.3, date: "1 day ago", status: "completed" },
  ];

  return (
    <ModernLayout>
      <div>
        {/* Welcome Banner */}
        <div
          style={{
            padding: 32,
            borderRadius: 20,
            background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            color: "#ffffff",
            marginBottom: 32,
            boxShadow: "0 20px 60px rgba(102, 126, 234, 0.3)",
            position: "relative",
            overflow: "hidden",
          }}
        >
          <div style={{ position: "relative", zIndex: 1 }}>
            <h1 style={{ fontSize: 36, fontWeight: 900, margin: 0, marginBottom: 8 }}>
              Welcome to AIDEX 👋
            </h1>
            <p style={{ fontSize: 16, margin: 0, opacity: 0.9 }}>
              Your intelligent AutoML platform for classification, regression, and image recognition
            </p>
          </div>
          {/* Decorative circles */}
          <div
            style={{
              position: "absolute",
              top: -50,
              right: -50,
              width: 200,
              height: 200,
              borderRadius: "50%",
              background: "rgba(255,255,255,0.1)",
            }}
          />
          <div
            style={{
              position: "absolute",
              bottom: -80,
              right: 100,
              width: 150,
              height: 150,
              borderRadius: "50%",
              background: "rgba(255,255,255,0.05)",
            }}
          />
        </div>

        {/* Stats Grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
            gap: 20,
            marginBottom: 32,
          }}
        >
          {[
            { icon: "📊", label: "Total Projects", value: stats.totalProjects, color: "#6366f1", trend: "+3" },
            { icon: "🤖", label: "Active Models", value: stats.activeModels, color: "#8b5cf6", trend: "+2" },
            { icon: "🔮", label: "Predictions", value: stats.totalPredictions.toLocaleString(), color: "#14b8a6", trend: "+127" },
            { icon: "📈", label: "Avg Accuracy", value: `${stats.avgAccuracy}%`, color: "#f59e0b", trend: "+2.1%" },
          ].map((stat, idx) => (
            <div
              key={idx}
              style={{
                padding: 24,
                borderRadius: 16,
                background: "#ffffff",
                border: "1px solid #e2e8f0",
                boxShadow: "0 4px 12px rgba(0,0,0,0.05)",
                transition: "all 0.3s ease",
                cursor: "pointer",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = "translateY(-4px)";
                e.currentTarget.style.boxShadow = "0 12px 24px rgba(0,0,0,0.1)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.boxShadow = "0 4px 12px rgba(0,0,0,0.05)";
              }}
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                <div
                  style={{
                    width: 48,
                    height: 48,
                    borderRadius: 12,
                    background: `${stat.color}15`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 24,
                  }}
                >
                  {stat.icon}
                </div>
                <div
                  style={{
                    padding: "4px 10px",
                    borderRadius: 8,
                    background: "#10b98115",
                    color: "#10b981",
                    fontSize: 12,
                    fontWeight: 600,
                  }}
                >
                  {stat.trend}
                </div>
              </div>
              <p style={{ fontSize: 13, color: "#64748b", margin: 0, marginBottom: 4 }}>{stat.label}</p>
              <p style={{ fontSize: 28, fontWeight: 900, color: "#0f172a", margin: 0 }}>{stat.value}</p>
            </div>
          ))}
        </div>

        {/* Quick Actions */}
        <div style={{ marginBottom: 32 }}>
          <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 16, color: "#0f172a" }}>⚡ Quick Actions</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 16 }}>
            {quickActions.map((action, idx) => (
              <button
                key={idx}
                onClick={() => router.push(action.path)}
                style={{
                  padding: 20,
                  borderRadius: 14,
                  background: "#ffffff",
                  border: `2px solid ${action.color}20`,
                  cursor: "pointer",
                  transition: "all 0.3s ease",
                  textAlign: "left",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = "translateY(-4px)";
                  e.currentTarget.style.borderColor = action.color;
                  e.currentTarget.style.boxShadow = `0 12px 24px ${action.color}30`;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "translateY(0)";
                  e.currentTarget.style.borderColor = `${action.color}20`;
                  e.currentTarget.style.boxShadow = "none";
                }}
              >
                <div style={{ fontSize: 32, marginBottom: 8 }}>{action.icon}</div>
                <p style={{ fontSize: 16, fontWeight: 700, color: "#0f172a", margin: 0, marginBottom: 4 }}>
                  {action.label}
                </p>
                <p style={{ fontSize: 13, color: "#64748b", margin: 0 }}>{action.desc}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Recent Projects */}
        <div>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
            <h2 style={{ fontSize: 20, fontWeight: 700, margin: 0, color: "#0f172a" }}>📁 Recent Projects</h2>
            <button
              onClick={() => router.push("/projects")}
              style={{
                background: "transparent",
                border: "1px solid #e2e8f0",
                padding: "8px 16px",
                borderRadius: 8,
                fontSize: 13,
                fontWeight: 600,
                color: "#6366f1",
                cursor: "pointer",
                transition: "all 0.2s",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "#6366f105";
                e.currentTarget.style.borderColor = "#6366f1";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "transparent";
                e.currentTarget.style.borderColor = "#e2e8f0";
              }}
            >
              View All →
            </button>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {recentProjects.map((project, idx) => (
              <div
                key={idx}
                style={{
                  padding: 20,
                  borderRadius: 14,
                  background: "#ffffff",
                  border: "1px solid #e2e8f0",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                }}
                onClick={() => router.push("/preprocessing")}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "#6366f1";
                  e.currentTarget.style.boxShadow = "0 4px 12px rgba(99, 102, 241, 0.1)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "#e2e8f0";
                  e.currentTarget.style.boxShadow = "none";
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                  <div
                    style={{
                      width: 48,
                      height: 48,
                      borderRadius: 12,
                      background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 20,
                    }}
                  >
                    {project.type === "Image Classification" ? "🖼️" : project.type === "Regression" ? "📈" : "🎯"}
                  </div>
                  <div>
                    <p style={{ fontSize: 16, fontWeight: 700, color: "#0f172a", margin: 0, marginBottom: 4 }}>
                      {project.name}
                    </p>
                    <p style={{ fontSize: 13, color: "#64748b", margin: 0 }}>
                      {project.type} • {project.date}
                    </p>
                  </div>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                  <div style={{ textAlign: "right" }}>
                    <p style={{ fontSize: 12, color: "#64748b", margin: 0 }}>
                      {project.type === "Regression" ? "R² Score" : "Accuracy"}
                    </p>
                    <p style={{ fontSize: 20, fontWeight: 700, color: "#10b981", margin: 0 }}>
                      {project.type === "Regression" ? project.r2 : `${project.accuracy}%`}
                    </p>
                  </div>
                  <div
                    style={{
                      padding: "6px 12px",
                      borderRadius: 8,
                      background: "#10b98115",
                      color: "#10b981",
                      fontSize: 12,
                      fontWeight: 600,
                    }}
                  >
                    ✓ {project.status}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </ModernLayout>
  );
}
