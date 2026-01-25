"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Papa from "papaparse";
import { useDataset } from "./context/DatasetContext";

export default function Home() {
  const router = useRouter();

  const {
    state,
    setDatasetKind,
    setErrorMsg,
    setStructured,
    setImages,
    resetAll,
  } = useDataset();

  const [showUploadMenu, setShowUploadMenu] = useState(false);
  const [csvLoading, setCsvLoading] = useState(false);
  const [folderLoading, setFolderLoading] = useState(false);

  // ✅ NEW: project logic
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
     LOAD PROJECT FROM STORAGE
  ========================= */
  useEffect(() => {
    if (typeof window === "undefined") return;

    const savedName = localStorage.getItem("aidex_project_name") || "";
    if (savedName.trim()) {
      setProjectName(savedName);
      setProjectCreated(true);
    }
  }, []);

  /* =========================
     PAGE RESET
  ========================= */
  const resetPage1 = () => {
    resetAll();
    setShowUploadMenu(false);
    setCsvLoading(false);
    setFolderLoading(false);

    // ✅ reset project too
    setProjectName("");
    setProjectCreated(false);

    if (typeof window !== "undefined") {
      localStorage.removeItem("aidex_project_name");
    }
  };

  /* =========================
     CREATE PROJECT
  ========================= */
  const handleCreateProject = () => {
    const name = projectName.trim();

    if (!name) {
      setErrorMsg("Please enter a project name first.");
      return;
    }

    if (name.length < 3) {
      setErrorMsg("Project name must be at least 3 characters.");
      return;
    }

    // ✅ Save project name (frontend only)
    localStorage.setItem("aidex_project_name", name);
    setProjectCreated(true);

    // clear any old error
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
      setShowUploadMenu(false);
      setDatasetKind("unsupported");
      setErrorMsg("Unsupported file type. Please upload a CSV file.");
      return;
    }

    resetAll();
    setShowUploadMenu(false);
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
        setShowUploadMenu(false);
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
    setShowUploadMenu(false);
    setDatasetKind("images");
    setFolderLoading(true);

    const items: {
      file: File;
      label: string;
      relativePath: string;
    }[] = [];

    Array.from(files).forEach((file) => {
      if (!file.type.startsWith("image/")) return;
      const relativePath = (file as any).webkitRelativePath || file.name;

      items.push({
        file,
        label: "unknown",
        relativePath,
      });
    });

    if (items.length === 0) {
      resetAll();
      setShowUploadMenu(false);
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
    <main className="w-full flex justify-center">
      <div className="w-full max-w-4xl mt-10">
        {/* OUTER CARD */}
        <div className="rounded-3xl bg-white/5 border border-white/10 shadow-xl p-8">
          {/* HEADER */}
          <div className="flex items-start justify-between mb-6">
            <div>
              <h1 className="text-2xl font-semibold text-white">
                Start Your AIDEX Project
              </h1>
              <p className="mt-2 text-sm text-white/60 max-w-xl">
                First create a project name, then upload your dataset (CSV or
                image folder).
              </p>
            </div>

            <button
              onClick={resetPage1}
              className="rounded-xl border border-white/15 px-4 py-2 text-sm text-white/80 hover:bg-white/10 transition"
            >
              Reset
            </button>
          </div>

          {/* ✅ PROJECT CREATION CARD */}
          <div className="rounded-2xl bg-white/5 border border-white/10 p-6 mb-6">
            <p className="text-sm font-semibold text-white">
              Step 1 — Create Project
            </p>
            <p className="mt-1 text-xs text-white/60">
              Give your project a clear name so you can track it later.
            </p>

            <div className="mt-4 flex flex-col sm:flex-row gap-3">
              <input
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                disabled={projectCreated}
                placeholder="Example: Diabetes Prediction / Cat vs Dog Dataset"
                className="w-full rounded-xl bg-black/20 border border-white/10 px-4 py-3 text-sm text-white placeholder:text-white/40 outline-none focus:border-white/30"
              />

              {!projectCreated ? (
                <button
                  onClick={handleCreateProject}
                  className="rounded-xl bg-white text-black px-5 py-3 text-sm font-semibold hover:bg-gray-200 transition"
                >
                  Create Project
                </button>
              ) : (
                <button
                  className="rounded-xl bg-green-500/20 border border-green-400/40 text-green-200 px-5 py-3 text-sm font-semibold cursor-default"
                >
                  ✅ Project Created
                </button>
              )}
            </div>

            {projectCreated && (
              <p className="mt-3 text-xs text-white/60">
                Current project:{" "}
                <span className="text-white font-semibold">{projectName}</span>
              </p>
            )}
          </div>

          {/* ✅ UPLOAD PANEL */}
          <div
            className={`rounded-2xl bg-white/5 border border-white/10 p-10 flex flex-col items-center text-center gap-4 ${
              !projectCreated ? "opacity-50 pointer-events-none" : ""
            }`}
          >
            <p className="text-sm text-white/70">
              Step 2 — Upload Dataset
              <br />
              Drag & Drop is optional — click below to choose a dataset.
            </p>

            <button
              onClick={() => setShowUploadMenu(true)}
              className="mt-2 rounded-xl bg-white text-black px-6 py-3 text-sm font-semibold hover:bg-gray-200 transition"
            >
              Upload Dataset
            </button>

            {/* UPLOAD MENU */}
            {showUploadMenu && (
              <div className="mt-6 w-full max-w-sm rounded-xl bg-[#0b1020] border border-white/15 p-2 text-left">
                <label className="block cursor-pointer rounded-lg px-4 py-3 hover:bg-white/10 transition">
                  <p className="text-sm font-semibold text-white">
                    Upload CSV Dataset
                  </p>
                  <p className="text-xs text-white/60 mt-1">
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

                <label className="block cursor-pointer rounded-lg px-4 py-3 hover:bg-white/10 transition">
                  <p className="text-sm font-semibold text-white">
                    Upload Image Folder
                  </p>
                  <p className="text-xs text-white/60 mt-1">
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
              <p className="text-sm text-indigo-400 mt-2">
                {csvLoading ? "Processing dataset..." : "Loading image folder..."}
              </p>
            )}
          </div>

          {/* ✅ Friendly warning if user didn’t create project */}
          {!projectCreated && (
            <div className="mt-4 text-xs text-white/60">
              ⚠️ Create a project first to unlock upload.
            </div>
          )}

          {/* STATUS */}
          {state.datasetKind !== "none" && (
            <div className="mt-6 rounded-xl bg-white/5 border border-white/10 p-4">
              <p className="text-sm font-semibold text-white">Dataset Detected</p>
              <p className="text-xs text-white/60 mt-1">
                {state.datasetKind === "structured" && "CSV Dataset"}
                {state.datasetKind === "images" && "Image Folder Dataset"}
                {state.datasetKind === "unsupported" && "Unsupported Dataset"}
              </p>

              {state.errorMsg && (
                <p className="mt-2 text-sm text-red-400">{state.errorMsg}</p>
              )}
            </div>
          )}

          {/* ✅ show error even before datasetKind changes */}
          {state.datasetKind === "none" && state.errorMsg && (
            <div className="mt-6 rounded-xl bg-red-500/10 border border-red-400/20 p-4">
              <p className="text-sm text-red-300">{state.errorMsg}</p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

