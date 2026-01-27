"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

/* =========================
   TYPES
========================= */
type Project = {
  id: string;
  name: string;
  createdAt: string;
  datasetType: "CSV" | "Images" | "Unknown";
};

type ProjectMap = {
  [email: string]: Project[];
};

const STORAGE_KEY = "aidex_projects";

/* =========================
   PAGE
========================= */
export default function ProfilePage() {
  const router = useRouter();

  const [projects, setProjects] = useState<Project[]>([]);
  const [userEmail, setUserEmail] = useState("");
  const [userName, setUserName] = useState("");

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingValue, setEditingValue] = useState("");

  /* =========================
     AUTH + USER
  ========================= */
  useEffect(() => {
    const authed = localStorage.getItem("aidex_auth") === "true";
    const email = localStorage.getItem("aidex_user_email");

    if (!authed || !email) {
      router.replace("/auth");
      return;
    }

    setUserEmail(email);

    const users = JSON.parse(localStorage.getItem("aidex_users") || "[]");
    const user = users.find((u: any) => u.email === email);
    setUserName(user?.name || "User");
  }, [router]);

  /* =========================
     LOAD PROJECTS
  ========================= */
  useEffect(() => {
    if (!userEmail) return;

    const raw = localStorage.getItem(STORAGE_KEY);
    const data: ProjectMap = raw ? JSON.parse(raw) : {};
    setProjects(data[userEmail] || []);
  }, [userEmail]);

  /* =========================
     SAVE PROJECTS
  ========================= */
  const saveProjects = (next: Project[]) => {
    const raw = localStorage.getItem(STORAGE_KEY);
    const data: ProjectMap = raw ? JSON.parse(raw) : {};

    data[userEmail] = next;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    setProjects(next);
  };

  /* =========================
     CRUD
  ========================= */
  const handleDelete = (id: string) =>
    saveProjects(projects.filter((p) => p.id !== id));

  const startEdit = (p: Project) => {
    setEditingId(p.id);
    setEditingValue(p.name);
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditingValue("");
  };

  const handleUpdate = () => {
    if (!editingId) return;

    const trimmed = editingValue.trim();
    if (!trimmed) return;

    saveProjects(
      projects.map((p) =>
        p.id === editingId ? { ...p, name: trimmed } : p
      )
    );

    cancelEdit();
  };

  const totalProjects = useMemo(() => projects.length, [projects]);

  /* =========================
     RENDER
  ========================= */
  return (
    <main className="min-h-screen bg-[#0B1020] text-white">
      <div className="max-w-6xl mx-auto px-6 py-10 space-y-8">

        {/* HEADER */}
        <section className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold">User Profile</h1>
            <p className="text-white/60 text-sm">
              {userName} • {userEmail}
            </p>
          </div>

          <button
            onClick={() => router.push("/")}
            className="rounded-xl bg-white text-black px-5 py-3 text-sm font-semibold hover:bg-white/90"
          >
            + Create Project
          </button>
        </section>

        {/* STATS */}
        <section className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          <StatCard title="Total Projects" value={totalProjects} />
          <StatCard title="Storage" value="localStorage" />
          <StatCard title="Actions" value="Edit / Delete" />
        </section>

        {/* PROJECTS */}
        <section className="rounded-3xl border border-white/10 bg-white/5 p-6">
          <h2 className="text-xl font-semibold mb-4">Past Projects</h2>

          {projects.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="space-y-3">
              {projects.map((p) => {
                const editing = editingId === p.id;

                return (
                  <div
                    key={p.id}
                    className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 rounded-2xl border border-white/10 bg-white/5 p-4"
                  >
                    <div className="flex-1">
                      {!editing ? (
                        <>
                          <p className="font-semibold">{p.name}</p>
                          <p className="text-xs text-white/50">
                            {new Date(p.createdAt).toLocaleString()} • {p.datasetType}
                          </p>
                        </>
                      ) : (
                        <input
                          value={editingValue}
                          onChange={(e) => setEditingValue(e.target.value)}
                          className="w-full rounded-xl bg-white/5 border border-white/10 px-4 py-2"
                          autoFocus
                        />
                      )}
                    </div>

                    <div className="flex gap-2">
                      {!editing ? (
                        <>
                          <ActionButton onClick={() => startEdit(p)}>Edit</ActionButton>
                          <DangerButton onClick={() => handleDelete(p.id)}>Delete</DangerButton>
                        </>
                      ) : (
                        <>
                          <PrimaryButton onClick={handleUpdate}>Save</PrimaryButton>
                          <ActionButton onClick={cancelEdit}>Cancel</ActionButton>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

/* =========================
   UI COMPONENTS
========================= */

function StatCard({ title, value }: { title: string; value: any }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
      <p className="text-xs text-white/50">{title}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-6 text-sm text-white/70">
      No projects yet. Click <b>+ Create Project</b> to start ✅
    </div>
  );
}

function ActionButton({ children, onClick }: any) {
  return (
    <button
      onClick={onClick}
      className="rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-sm hover:bg-white/10"
    >
      {children}
    </button>
  );
}

function PrimaryButton({ children, onClick }: any) {
  return (
    <button
      onClick={onClick}
      className="rounded-xl bg-white text-black px-4 py-2 text-sm font-semibold hover:bg-white/90"
    >
      {children}
    </button>
  );
}

function DangerButton({ children, onClick }: any) {
  return (
    <button
      onClick={onClick}
      className="rounded-xl border border-red-400/30 bg-red-500/10 px-4 py-2 text-sm text-red-200 hover:bg-red-500/20"
    >
      {children}
    </button>
  );
}
