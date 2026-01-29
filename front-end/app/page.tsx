"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Papa from "papaparse";
import { useDataset } from "./context/DatasetContext";

export default function Home() {
  const router = useRouter();

  const { state, setDatasetKind, setErrorMsg, setStructured, setImages, resetAll } =
    useDataset();

  const [showUploadMenu, setShowUploadMenu] = useState(false);
  const [csvLoading, setCsvLoading] = useState(false);
  const [folderLoading, setFolderLoading] = useState(false);

  // ✅ modal state
  const [showCreateProjectModal, setShowCreateProjectModal] = useState(false);
  const [tempProjectName, setTempProjectName] = useState("");

  // ✅ project logic
  const [projectName, setProjectName] = useState("");
  const [projectCreated, setProjectCreated] = useState(false);

  /* =========================
     AUTH GUARD
  ========================= */
  useEffect(() => {
    const isAuthed = localStorage.getItem("aidex_auth") === "true";
    if (!isAuthed) router.replace("/auth");
  }, [router]);

  /* =========================
     LOAD PROJECT (FROM LOCALSTORAGE)
  ========================= */
  useEffect(() => {
    const saved = localStorage.getItem("aidex_project_name") || "";
    if (saved.trim()) {
      setProjectName(saved.trim());
      setProjectCreated(true);
    }
  }, []);

  /* =========================
     RESET PAGE
  ========================= */
  const resetPage1 = () => {
    resetAll();
    setShowUploadMenu(false);
    setCsvLoading(false);
    setFolderLoading(false);

    // ✅ remove project
    setProjectName("");
    setProjectCreated(false);

    localStorage.removeItem("aidex_project_name");
    setErrorMsg("");
  };

  /* =========================
     UPLOAD HANDLERS
  ========================= */
  const handleStructuredUpload = (file: File) => {
    if (!projectCreated) {
      setErrorMsg("Please create a project first before uploading.");
      return;
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
      resetAll();
      setDatasetKind("unsupported");
      setErrorMsg("Unsupported file type. Please upload a CSV file.");
      return;
    }

    resetAll();
    setDatasetKind("structured");
    setCsvLoading(true);

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      complete: (results: Papa.ParseResult<any>) => {
        const rows = (results.data as any[]) || [];
        const columns = rows.length > 0 ? Object.keys(rows[0]) : [];

        setStructured({ fileName: file.name, rows, columns });
        setCsvLoading(false);

        router.push("/overview");
      },
      error: () => {
        resetAll();
        setDatasetKind("unsupported");
        setErrorMsg("Failed to parse CSV file.");
        setCsvLoading(false);
      },
    });
  };

  const handleImageFolderUpload = (files: FileList) => {
    if (!projectCreated) {
      setErrorMsg("Please create a project first before uploading.");
      return;
    }

    resetAll();
    setDatasetKind("images");
    setFolderLoading(true);

    const items: { file: File; label: string; relativePath: string }[] = [];

    Array.from(files).forEach((file) => {
      if (!file.type.startsWith("image/")) return;

      const relativePath = (file as any).webkitRelativePath || file.name;
      
      // Extract label from filename (e.g., "dog.101.jpg" -> "dog")
      const fileName = file.name;
      let label = "unknown";
      
      // Try to extract label from filename pattern like "dog.123.jpg" or "cat.5.jpg"
      const match = fileName.match(/^([a-zA-Z_]+)\.\d+\./);
      if (match) {
        label = match[1];
      } else {
        // If no pattern match, check if filename starts with a word followed by underscore or dot
        const simpleMatch = fileName.match(/^([a-zA-Z_]+)[._]/);
        if (simpleMatch) {
          label = simpleMatch[1];
        } else {
          // Try to get label from parent folder name in relative path
          const pathParts = relativePath.split('/');
          if (pathParts.length > 1) {
            label = pathParts[pathParts.length - 2];
          }
        }
      }

      items.push({
        file,
        label,
        relativePath,
      });
    });

    if (items.length === 0) {
      resetAll();
      setDatasetKind("unsupported");
      setErrorMsg("No image files found in the uploaded folder.");
      setFolderLoading(false);
      return;
    }

    setImages(items);
    setFolderLoading(false);

    router.push("/overview");
  };

  return (
    <main className="w-full">
      <div className="w-full max-w-6xl space-y-6 pb-24">
        {/* =========================
            HEADER
        ========================= */}
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
          <h1 className="text-center text-3xl font-medium tracking-tight text-white">
  AIDEX Insight Workspace
</h1>



            {/* ✅ proper space after sentence */}
            <p className="mt-2 text-sm text-white/60 max-w-xl mb-6">
              Create a project name first, then upload your dataset.
            </p>
          </div>

          <button onClick={resetPage1} className="aidex-btn-outline">
            Reset
          </button>
        </div>

        {/* ✅ SPACE between header and step 1 */}
        <div className="h-5" />

        {/* =========================
            STEP 1 — CREATE PROJECT
        ========================= */}
        <div className="aidex-card">
          <p className="aidex-card-title"> Create Project</p>

          <p className="text-sm mt-2" style={{ color: "var(--muted)" }}>
            Create a project to unlock dataset upload.
          </p>

          {/* ✅ nice spacing inside step 1 */}
          <div className="mt-5 flex items-center justify-between gap-4 flex-wrap">
            <div>
              {!projectCreated ? (
                <p className="text-sm text-white/70">
                  No project yet. Click create to continue.
                </p>
              ) : (
                <div
                  style={{
                    borderRadius: 16,
                    border: "1px solid rgba(255,255,255,0.10)",
                    background: "rgba(255,255,255,0.04)",
                    padding: 12,
                  }}
                >
                  <p className="text-xs text-white/50">Current project</p>
                  <p className="text-sm font-extrabold text-white mt-1">
                    {projectName}
                  </p>
                </div>
              )}
            </div>

            <button
              onClick={() => {
                setTempProjectName(projectName || "");
                setShowCreateProjectModal(true);
              }}
              className="aidex-btn-primary"
            >
              {projectCreated ? "Rename Project" : "Create Project"}
            </button>
          </div>
        </div>

        {/* ✅ SPACE between step 1 and step 2 (THIS IS WHAT YOU WANT) */}
        <div className="h-4" />

        {/* =========================
            STEP 2 — UPLOAD DATASET
        ========================= */}
        <div
          className="aidex-card"
          style={{
            opacity: projectCreated ? 1 : 0.5,
            pointerEvents: projectCreated ? "auto" : "none",
          }}
        >
          <p className="aidex-card-title"> Upload Dataset</p>

         
          <div className="mt-6 flex justify-center">
            <button
              onClick={() => setShowUploadMenu((prev) => !prev)}
              className="aidex-btn-primary"
            >
              Upload Dataset
            </button>
          </div>

          {showUploadMenu && (
            <div
              style={{
                marginTop: 18,
                borderRadius: 18,
                border: "1px solid rgba(255,255,255,0.12)",
                background: "rgba(255,255,255,0.04)",
                overflow: "hidden",
              }}
            >
              <label
                style={{
                  display: "block",
                  cursor: "pointer",
                  padding: 14,
                  borderBottom: "1px solid rgba(255,255,255,0.10)",
                }}
              >
                <p style={{ fontWeight: 900, color: "white" }}>
                  Upload CSV Dataset
                </p>
                <p style={{ fontSize: 12, color: "var(--muted)", marginTop: 3 }}>
                  Structured tabular data (.csv)
                </p>

                <input
                  type="file"
                  accept=".csv"
                  hidden
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (!f) return;
                    setShowUploadMenu(false);
                    handleStructuredUpload(f);
                  }}
                />
              </label>

              <label
                style={{
                  display: "block",
                  cursor: "pointer",
                  padding: 14,
                }}
              >
                <p style={{ fontWeight: 900, color: "white" }}>
                  Upload Image Folder
                </p>
                <p style={{ fontSize: 12, color: "var(--muted)", marginTop: 3 }}>
                  Folder containing image files
                </p>

                <input
                  type="file"
                  multiple
                  accept="image/*"
                  // @ts-ignore
                  webkitdirectory="true"
                  hidden
                  onChange={(e) => {
                    const files = e.target.files;
                    if (!files) return;
                    setShowUploadMenu(false);
                    handleImageFolderUpload(files);
                  }}
                />
              </label>
            </div>
          )}

          {(csvLoading || folderLoading) && (
            <p
              className="text-sm mt-4"
              style={{ color: "rgba(129,140,248,0.95)" }}
            >
              {csvLoading ? "Processing dataset..." : "Loading image folder..."}
            </p>
          )}
        </div>

        {/* ✅ WARNING */}
        {!projectCreated && (
          <div
            style={{
              borderRadius: 16,
              border: "1px solid rgba(234,179,8,0.35)",
              background: "rgba(234,179,8,0.10)",
              padding: 12,
              color: "rgba(254,240,138,0.95)",
              fontWeight: 700,
              fontSize: 13,
            }}
          >
            ⚠️ Create a project first to unlock upload.
          </div>
        )}

        {/* ✅ ERROR */}
        {state.errorMsg && (
          <div
            style={{
              borderRadius: 16,
              border: "1px solid rgba(239,68,68,0.35)",
              background: "rgba(239,68,68,0.12)",
              padding: 12,
              color: "rgba(254,202,202,0.95)",
              fontWeight: 700,
              fontSize: 13,
            }}
          >
            {state.errorMsg}
          </div>
        )}
      </div>

      {/* =========================
          ✅ CREATE PROJECT MODAL
      ========================= */}
      {showCreateProjectModal && (
        <div
          className="aidex-modal-overlay"
          onClick={() => setShowCreateProjectModal(false)}
        >
          <div className="aidex-modal" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-lg font-extrabold text-white">
                  {projectCreated ? "Rename Project" : "Create Project"}
                </h3>
                <p className="text-sm mt-1 text-white/60">
                  Enter a project name to save your workspace.
                </p>
              </div>

              <button
                onClick={() => setShowCreateProjectModal(false)}
                className="aidex-btn-outline"
              >
                Close
              </button>
            </div>

            <div className="mt-5">
              <p className="text-xs font-semibold text-white/70 mb-2">
                Project Name
              </p>

              <input
                value={tempProjectName}
                onChange={(e) => setTempProjectName(e.target.value)}
                placeholder="e.g. Cat vs Dog Classification"
                className="aidex-input-modal"
                autoFocus
              />

              <p className="mt-2 text-xs text-white/40">
                Minimum 3 characters.
              </p>
            </div>

            <div className="mt-6 flex justify-end gap-2 flex-wrap">
              <button
                onClick={() => {
                  setShowCreateProjectModal(false);
                }}
                className="aidex-btn-outline"
              >
                Cancel
              </button>

              <button
                onClick={() => {
                  const name = tempProjectName.trim();

                  if (!name) {
                    setErrorMsg("Please enter a project name first.");
                    return;
                  }

                  if (name.length < 3) {
                    setErrorMsg("Project name must be at least 3 characters.");
                    return;
                  }

                  setProjectName(name);
                  setProjectCreated(true);

                  localStorage.setItem("aidex_project_name", name);

                  setErrorMsg("");
                  setShowCreateProjectModal(false);
                }}
                className="aidex-btn-primary"
              >
                Confirm & Save
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
