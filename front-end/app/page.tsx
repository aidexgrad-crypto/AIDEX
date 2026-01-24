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

  /* =========================
     AUTH GUARD
  ========================= */
  useEffect(() => {
    const isAuthed = localStorage.getItem("aidex_auth") === "true";
    if (!isAuthed) router.replace("/auth");
  }, [router]);

  /* =========================
     PAGE 1 RESET
  ========================= */
  const resetPage1 = () => {
    resetAll();
    setShowUploadMenu(false);
    setCsvLoading(false);
    setFolderLoading(false);
  };

  /* =========================
     UPLOAD HANDLERS
  ========================= */
  const handleStructuredUpload = (file: File) => {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      resetPage1();
      setDatasetKind("unsupported");
      setErrorMsg("Unsupported file type. Please upload a CSV file.");
      return;
    }

    resetPage1();
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
        resetPage1();
        setDatasetKind("unsupported");
        setErrorMsg("Failed to parse CSV file.");
        setCsvLoading(false);
      },
    });
  };

  const handleImageFolderUpload = (files: FileList) => {
    resetPage1();
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
      resetPage1();
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
      <div className="max-w-5xl">
        {/* Header */}
        <div className="mb-6 flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-3xl font-bold">
              Upload Your Data to Get Insights
            </h1>
            <p className="mt-2 text-sm" style={{ color: "var(--muted)" }}>
              Upload a CSV dataset or an image folder. AIDEX will detect the type
              automatically.
            </p>
          </div>

          <button onClick={resetPage1} className="aidex-btn-outline">
            Reset
          </button>
        </div>

        {/* Upload Panel */}
        <div className="aidex-card">
          <div
            style={{
              borderRadius: 16,
              border: "1px solid var(--border)",
              background: "rgba(15, 23, 42, 0.03)",
              padding: 28,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              textAlign: "center",
              gap: 14,
            }}
          >
            <p className="text-sm" style={{ color: "var(--muted)" }}>
              Drag and drop is optional. Use the button below to select your dataset.
            </p>

            {/* Upload Button + Menu */}
            <div style={{ position: "relative" }}>
              <button
                onClick={() => setShowUploadMenu((prev) => !prev)}
                className="aidex-btn-primary"
              >
                Upload Dataset
              </button>

              {showUploadMenu && (
                <div
                  style={{
                    position: "absolute",
                    left: "50%",
                    transform: "translateX(-50%)",
                    marginTop: 12,
                    width: 340,
                    borderRadius: 16,
                    border: "1px solid var(--border)",
                    background: "white",
                    boxShadow: "var(--shadow)",
                    padding: 8,
                    textAlign: "left",
                    zIndex: 50,
                  }}
                >
                  <label
                    style={{
                      display: "block",
                      cursor: "pointer",
                      borderRadius: 14,
                      padding: 14,
                      transition: "0.2s",
                    }}
                  >
                    <p style={{ fontWeight: 700, fontSize: 13 }}>
                      Upload CSV Dataset
                    </p>
                    <p style={{ fontSize: 12, color: "var(--muted)", marginTop: 4 }}>
                      CSV format
                    </p>

                    <input
                      type="file"
                      accept=".csv"
                      style={{ display: "none" }}
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
                      borderRadius: 14,
                      padding: 14,
                      transition: "0.2s",
                    }}
                  >
                    <p style={{ fontWeight: 700, fontSize: 13 }}>
                      Upload Image Folder
                    </p>
                    <p style={{ fontSize: 12, color: "var(--muted)", marginTop: 4 }}>
                      Folder containing labeled images
                    </p>

                    <input
                      type="file"
                      multiple
                      accept="image/*"
                      // @ts-ignore
                      webkitdirectory="true"
                      style={{ display: "none" }}
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
            </div>

            {/* Loading */}
            {(csvLoading || folderLoading) && (
              <p className="text-sm" style={{ color: "var(--primary)" }}>
                {csvLoading ? "Processing dataset..." : "Loading folder..."}
              </p>
            )}

            {/* Dataset detected */}
            {state.datasetKind !== "none" && (
              <div
                style={{
                  width: "100%",
                  borderRadius: 16,
                  border: "1px solid var(--border)",
                  background: "white",
                  padding: 16,
                  textAlign: "left",
                }}
              >
                <p style={{ fontWeight: 700 }}>Dataset Detected</p>
                <p className="text-sm" style={{ color: "var(--muted)", marginTop: 4 }}>
                  {state.datasetKind === "structured" && "CSV Dataset"}
                  {state.datasetKind === "images" && "Image Folder Dataset"}
                  {state.datasetKind === "unsupported" && "Unsupported Dataset"}
                </p>

                {state.errorMsg && (
                  <p className="text-sm" style={{ color: "#b91c1c", marginTop: 8 }}>
                    {state.errorMsg}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
