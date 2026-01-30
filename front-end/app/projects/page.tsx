"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import ModernLayout from "../components/ModernLayoutClean";

type Project = {
  id: string;
  name: string;
  type: "Classification" | "Regression" | "Image Classification";
  created: string;
  lastModified: string;
  status: "completed" | "training" | "failed";
  accuracy?: number;
  r2?: number;
  models: number;
};

export default function ProjectsPage() {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState("");
  const [filterType, setFilterType] = useState<string>("all");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [showNewProjectModal, setShowNewProjectModal] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [projectDescription, setProjectDescription] = useState("");

  useEffect(() => {
    const isAuthed = localStorage.getItem("aidex_auth") === "true";
    if (!isAuthed) router.replace("/auth");
  }, [router]);

  const handleCreateProject = () => {
    if (!projectName.trim()) {
      alert("Project name is required");
      return;
    }

    const projectId = Date.now().toString();
    localStorage.setItem(`project_${projectId}`, JSON.stringify({
      id: projectId,
      name: projectName,
      description: projectDescription,
      created: new Date().toISOString(),
    }));

    setShowNewProjectModal(false);
    setProjectName("");
    setProjectDescription("");
    router.push(`/project/${projectId}`);
  };

  // Mock projects data - replace with API call
  const allProjects: Project[] = [
    {
      id: "1",
      name: "Customer Churn Prediction",
      type: "Classification",
      created: "2026-01-25",
      lastModified: "2 hours ago",
      status: "completed",
      accuracy: 96.5,
      models: 6,
    },
    {
      id: "2",
      name: "Sales Forecast Q1",
      type: "Regression",
      created: "2026-01-24",
      lastModified: "5 hours ago",
      status: "completed",
      r2: 0.89,
      models: 6,
    },
    {
      id: "3",
      name: "Product Image Recognition",
      type: "Image Classification",
      created: "2026-01-23",
      lastModified: "1 day ago",
      status: "completed",
      accuracy: 92.3,
      models: 1,
    },
    {
      id: "4",
      name: "Fraud Detection System",
      type: "Classification",
      created: "2026-01-22",
      lastModified: "2 days ago",
      status: "training",
      models: 4,
    },
  ];

  const filteredProjects = allProjects.filter((p) => {
    const matchesSearch = p.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = filterType === "all" || p.type === filterType;
    return matchesSearch && matchesType;
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return { bg: "#dcfce7", color: "#16a34a", label: "Completed" };
      case "training":
        return { bg: "#dbeafe", color: "#2563eb", label: "Training" };
      case "failed":
        return { bg: "#fee2e2", color: "#dc2626", label: "Failed" };
      default:
        return { bg: "#f3f4f6", color: "#6b7280", label: "Unknown" };
    }
  };

  return (
    <ModernLayout>
      <div>
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <h1 style={{ fontSize: 32, fontWeight: 900, margin: 0, marginBottom: 8, color: "#0f172a" }}>
            Projects
          </h1>
          <p style={{ fontSize: 14, color: "#64748b", margin: 0 }}>
            Manage and explore your AutoML projects
          </p>
        </div>

        {/* Controls Bar */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 16,
            marginBottom: 24,
            flexWrap: "wrap",
          }}
        >
          {/* Search */}
          <div style={{ flex: 1, minWidth: 250 }}>
            <input
              type="text"
              placeholder="Search projects..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                width: "100%",
                padding: "10px 16px",
                borderRadius: 10,
                border: "1px solid #e2e8f0",
                fontSize: 14,
                outline: "none",
                transition: "all 0.2s",
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = "#6366f1";
                e.currentTarget.style.boxShadow = "0 0 0 3px rgba(99, 102, 241, 0.1)";
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = "#e2e8f0";
                e.currentTarget.style.boxShadow = "none";
              }}
            />
          </div>

          {/* Filters */}
          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              style={{
                padding: "10px 16px",
                borderRadius: 10,
                border: "1px solid #e2e8f0",
                fontSize: 14,
                cursor: "pointer",
                background: "#ffffff",
              }}
            >
              <option value="all">All Types</option>
              <option value="Classification">Classification</option>
              <option value="Regression">Regression</option>
              <option value="Image Classification">Image Classification</option>
            </select>

            {/* View Toggle */}
            <div style={{ display: "flex", border: "1px solid #e2e8f0", borderRadius: 10, overflow: "hidden" }}>
              <button
                onClick={() => setViewMode("grid")}
                style={{
                  padding: "10px 16px",
                  border: "none",
                  background: viewMode === "grid" ? "#6366f1" : "#ffffff",
                  color: viewMode === "grid" ? "#ffffff" : "#64748b",
                  cursor: "pointer",
                  fontSize: 14,
                  fontWeight: 600,
                  transition: "all 0.2s",
                }}
              >
                ⊞ Grid
              </button>
              <button
                onClick={() => setViewMode("list")}
                style={{
                  padding: "10px 16px",
                  border: "none",
                  borderLeft: "1px solid #e2e8f0",
                  background: viewMode === "list" ? "#6366f1" : "#ffffff",
                  color: viewMode === "list" ? "#ffffff" : "#64748b",
                  cursor: "pointer",
                  fontSize: 14,
                  fontWeight: 600,
                  transition: "all 0.2s",
                }}
              >
                ☰ List
              </button>
            </div>

            {/* New Project */}
            <button
              onClick={() => router.push("/preprocessing")}
              style={{
                padding: "10px 20px",
                borderRadius: 10,
                border: "none",
                background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
                color: "#111827",
                fontSize: 14,
                fontWeight: 700,
                cursor: "pointer",
                boxShadow: "0 4px 12px rgba(99, 102, 241, 0.3)",
                transition: "all 0.2s",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = "translateY(-2px)";
                e.currentTarget.style.boxShadow = "0 6px 16px rgba(99, 102, 241, 0.4)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.boxShadow = "0 4px 12px rgba(99, 102, 241, 0.3)";
              }}
            >
              New Project
            </button>
          </div>
        </div>

        {/* Projects Display */}
        {viewMode === "grid" ? (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
              gap: 20,
            }}
          >
            {filteredProjects.map((project) => {
              const statusStyle = getStatusColor(project.status);
              return (
                <div
                  key={project.id}
                  style={{
                    padding: 24,
                    borderRadius: 16,
                    background: "#ffffff",
                    border: "1px solid #e2e8f0",
                    cursor: "pointer",
                    transition: "all 0.3s ease",
                  }}
                  onClick={() => router.push("/preprocessing")}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = "translateY(-4px)";
                    e.currentTarget.style.boxShadow = "0 12px 24px rgba(0, 0, 0, 0.1)";
                    e.currentTarget.style.borderColor = "#6366f1";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "translateY(0)";
                    e.currentTarget.style.boxShadow = "none";
                    e.currentTarget.style.borderColor = "#e2e8f0";
                  }}
                >
                  {/* Icon & Status */}
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
                    <div
                      style={{
                        width: 56,
                        height: 56,
                        borderRadius: 8,
                        background: "#2563eb",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 20,
                        fontWeight: 700,
                        color: "#111827",
                      }}
                    >
                      {project.type.charAt(0)}
                    </div>
                    <div
                      style={{
                        padding: "6px 12px",
                        borderRadius: 6,
                        background: statusStyle.bg,
                        color: statusStyle.color,
                        fontSize: 11,
                        fontWeight: 500,
                      }}
                    >
                      {statusStyle.label}
                    </div>
                  </div>

                  {/* Project Info */}
                  <h3 style={{ fontSize: 18, fontWeight: 700, color: "#0f172a", margin: 0, marginBottom: 8 }}>
                    {project.name}
                  </h3>
                  <p style={{ fontSize: 13, color: "#64748b", margin: 0, marginBottom: 16 }}>
                    {project.type} • {project.models} model{project.models !== 1 ? "s" : ""}
                  </p>

                  {/* Performance */}
                  {project.status === "completed" && (
                    <div
                      style={{
                        padding: 12,
                        borderRadius: 10,
                        background: "#f8fafc",
                        marginBottom: 16,
                      }}
                    >
                      <p style={{ fontSize: 12, color: "#64748b", margin: 0, marginBottom: 4 }}>
                        {project.type === "Regression" ? "R² Score" : "Accuracy"}
                      </p>
                      <p style={{ fontSize: 24, fontWeight: 900, color: "#10b981", margin: 0 }}>
                        {project.type === "Regression" ? project.r2?.toFixed(3) : `${project.accuracy}%`}
                      </p>
                    </div>
                  )}

                  {/* Metadata */}
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#94a3b8" }}>
                    <span>Created: {project.created}</span>
                    <span>Modified: {project.lastModified}</span>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {filteredProjects.map((project) => {
              const statusStyle = getStatusColor(project.status);
              return (
                <div
                  key={project.id}
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
                      {getTypeIcon(project.type)}
                    </div>
                    <div>
                      <p style={{ fontSize: 16, fontWeight: 700, color: "#0f172a", margin: 0, marginBottom: 4 }}>
                        {project.name}
                      </p>
                      <p style={{ fontSize: 13, color: "#64748b", margin: 0 }}>
                        {project.type} • {project.models} models • Modified {project.lastModified}
                      </p>
                    </div>
                  </div>

                  <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                    {project.status === "completed" && (
                      <div style={{ textAlign: "right" }}>
                        <p style={{ fontSize: 12, color: "#64748b", margin: 0 }}>
                          {project.type === "Regression" ? "R² Score" : "Accuracy"}
                        </p>
                        <p style={{ fontSize: 20, fontWeight: 700, color: "#10b981", margin: 0 }}>
                          {project.type === "Regression" ? project.r2?.toFixed(3) : `${project.accuracy}%`}
                        </p>
                      </div>
                    )}
                    <div
                      style={{
                        padding: "6px 12px",
                        borderRadius: 8,
                        background: statusStyle.bg,
                        color: statusStyle.color,
                        fontSize: 12,
                        fontWeight: 600,
                      }}
                    >
                      {statusStyle.icon} {project.status}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Empty State */}
        {filteredProjects.length === 0 && (
          <div
            style={{
              padding: 60,
              textAlign: "center",
              background: "#ffffff",
              borderRadius: 16,
              border: "1px solid #e2e8f0",
            }}
          >
            <div style={{ fontSize: 64, marginBottom: 16 }}>📂</div>
            <h3 style={{ fontSize: 20, fontWeight: 700, color: "#0f172a", margin: 0, marginBottom: 8 }}>
              No projects found
            </h3>
            <p style={{ fontSize: 14, color: "#64748b", margin: 0, marginBottom: 24 }}>
              {searchQuery || filterType !== "all"
                ? "Try adjusting your search or filters"
                : "Get started by creating your first project"}
            </p>
            <button
              onClick={() => router.push("/preprocessing")}
              style={{
                padding: "12px 24px",
                borderRadius: 10,
                border: "none",
                background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
                color: "#111827",
                fontSize: 14,
                fontWeight: 700,
                cursor: "pointer",
                boxShadow: "0 4px 12px rgba(99, 102, 241, 0.3)",
              }}
            >
              Create New Project
            </button>
          </div>
        )}
      </div>
    </ModernLayout>
  );
}
