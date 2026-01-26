"use client";


import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";


type Project = {
  id: string;
  name: string;
  createdAt: string;
  datasetType: "CSV" | "Images" | "Unknown";
};

const STORAGE_KEY = "aidex_projects";

export default function ProfilePage() {
  const router = useRouter();

  const [projects, setProjects] = useState<Project[]>([]);

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingValue, setEditingValue] = useState("");

  useEffect(() => {
  const isAuthed = localStorage.getItem("aidex_auth") === "true";
  if (!isAuthed) router.replace("/auth");
}, [router]);


  /* =========================
     LOAD PROJECTS
  ========================= */
  useEffect(() => {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      setProjects([]);
      return;
    }

    try {
      const parsed = JSON.parse(raw);
      setProjects(Array.isArray(parsed) ? parsed : []);
    } catch {
      setProjects([]);
    }
  }, []);

  /* =========================
     SAVE PROJECTS
  ========================= */
  const saveProjects = (next: Project[]) => {
    setProjects(next);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  };

  /* =========================
     DELETE
  ========================= */
  const handleDelete = (id: string) => {
    const filtered = projects.filter((p) => p.id !== id);
    saveProjects(filtered);
  };

  /* =========================
     START EDIT
  ========================= */
  const startEdit = (p: Project) => {
    setEditingId(p.id);
    setEditingValue(p.name);
  };

  /* =========================
     CANCEL EDIT
  ========================= */
  const cancelEdit = () => {
    setEditingId(null);
    setEditingValue("");
  };

  /* =========================
     UPDATE
  ========================= */
  const handleUpdate = () => {
    if (!editingId) return;

    const trimmed = editingValue.trim();
    if (!trimmed) return;

    const updated = projects.map((p) =>
      p.id === editingId ? { ...p, name: trimmed } : p
    );

    saveProjects(updated);
    cancelEdit();
  };

  const totalProjects = useMemo(() => projects.length, [projects]);

  return (
    <main className="min-h-screen bg-[#0B1020] text-white">
      <div className="max-w-6xl mx-auto px-6 py-10 space-y-6">
        {/* Header */}
        <div className="rounded-3xl border border-white/10 bg-white/5 p-6 sm:p-8">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <h1 className="text-2xl sm:text-3xl font-bold">User Profile</h1>
              <p className="text-white/60 mt-2 text-sm">
                View and manage your past projects.
              </p>
            </div>

            {/* ✅ Redirect Create button to app/page.tsx */}
            <button
              onClick={() => router.push("/")}
              className="rounded-xl bg-white text-black px-5 py-3 text-sm font-semibold hover:bg-white/90 transition"
            >
              + Create Project
            </button>
          </div>

          <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs text-white/50">Total Projects</p>
              <p className="text-xl font-bold mt-1">{totalProjects}</p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs text-white/50">Storage</p>
              <p className="text-sm font-semibold mt-1">localStorage</p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs text-white/50">Actions</p>
              <p className="text-sm font-semibold mt-1">Edit / Delete</p>
            </div>
          </div>
        </div>

        {/* Past Projects */}
        <div className="rounded-3xl border border-white/10 bg-white/5 p-6 sm:p-8">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <h2 className="text-xl font-semibold">Past Projects</h2>
              <p className="text-white/60 text-sm mt-1">
                Manage your projects using CRUD operations.
              </p>
            </div>
          </div>

          {projects.length === 0 ? (
            <div className="mt-6 rounded-2xl border border-white/10 bg-white/5 p-6 text-white/70 text-sm">
              No projects yet. Click{" "}
              <span className="font-semibold text-white">+ Create Project</span>{" "}
              to start ✅
            </div>
          ) : (
            <div className="mt-6 space-y-3">
              {projects.map((p) => {
                const isEditing = editingId === p.id;

                return (
                  <div
                    key={p.id}
                    className="rounded-2xl border border-white/10 bg-white/5 p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3"
                  >
                    {/* Left side */}
                    <div className="flex-1">
                      {!isEditing ? (
                        <>
                          <p className="font-semibold text-white">{p.name}</p>
                          <p className="text-xs text-white/50 mt-1">
                            Created: {new Date(p.createdAt).toLocaleString()} •
                            Type: {p.datasetType}
                          </p>
                        </>
                      ) : (
                        <input
                          value={editingValue}
                          onChange={(e) => setEditingValue(e.target.value)}
                          className="w-full rounded-xl bg-white/5 border border-white/10 px-4 py-2 text-sm text-white outline-none"
                        />
                      )}
                    </div>

                    {/* Right side buttons */}
                    <div className="flex items-center gap-2">
                      {!isEditing ? (
                        <>
                          <button
                            onClick={() => startEdit(p)}
                            className="rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-sm hover:bg-white/10 transition"
                          >
                            Edit
                          </button>

                          <button
                            onClick={() => handleDelete(p.id)}
                            className="rounded-xl border border-red-400/30 bg-red-500/10 px-4 py-2 text-sm text-red-200 hover:bg-red-500/20 transition"
                          >
                            Delete
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            onClick={handleUpdate}
                            className="rounded-xl bg-white text-black px-4 py-2 text-sm font-semibold hover:bg-white/90 transition"
                          >
                            Save
                          </button>

                          <button
                            onClick={cancelEdit}
                            className="rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-sm hover:bg-white/10 transition"
                          >
                            Cancel
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
