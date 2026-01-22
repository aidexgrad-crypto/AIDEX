"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Papa from "papaparse";

/* =========================
   TYPES
========================= */

type DatasetKind = "none" | "structured" | "images" | "unsupported";

type StructuredDataset = {
  fileName: string;
  rows: any[];
  columns: string[];
};

type ImageItem = {
  file: File;
  label: string;
  relativePath: string;
};

type ColumnType = "numeric" | "categorical" | "text" | "datetime" | "unknown";

type ColumnAnalysis = {
  name: string;
  type: ColumnType;
  missing: number;
  unique: number;
  exampleValues: string[];
};

type TaskType = "classification" | "regression" | "unknown";

type TargetRecommendation = {
  target: string;
  taskType: TaskType;
  score: number;
  reasons: string[];
};

/* =========================
   HELPERS
========================= */

function getLabelFromRelativePath(relativePath: string, labelDepth: number) {
  const parts = relativePath.split("/").filter(Boolean);
  const idx = labelDepth - 1;
  if (parts.length >= idx + 1) return parts[idx] ?? "unknown";
  return "unknown";
}

function computeClassDistributionFromImages(images: ImageItem[]) {
  const dist: Record<string, number> = {};
  for (const img of images) {
    dist[img.label] = (dist[img.label] || 0) + 1;
  }
  return dist;
}

function sortDistribution(dist: Record<string, number>) {
  return Object.entries(dist).sort((a, b) => b[1] - a[1]);
}

function getBalancedPreview(images: ImageItem[], maxCount: number) {
  const groups: Record<string, ImageItem[]> = {};

  for (const img of images) {
    if (!groups[img.label]) groups[img.label] = [];
    groups[img.label].push(img);
  }

  const labels = Object.keys(groups);
  if (labels.length === 0) return [];

  for (const label of labels) groups[label].sort(() => Math.random() - 0.5);

  const result: ImageItem[] = [];
  let i = 0;

  while (result.length < maxCount) {
    let added = false;
    for (const label of labels) {
      if (groups[label][i]) {
        result.push(groups[label][i]);
        added = true;
        if (result.length >= maxCount) break;
      }
    }
    if (!added) break;
    i++;
  }

  return result;
}

function looksLikeDate(value: any) {
  if (value === null || value === undefined) return false;
  const s = String(value).trim();
  if (!s) return false;

  const dateLike =
    /^\d{4}-\d{2}-\d{2}/.test(s) ||
    /^\d{2}\/\d{2}\/\d{4}/.test(s) ||
    /^\d{4}\/\d{2}\/\d{2}/.test(s);

  if (!dateLike) return false;
  const d = new Date(s);
  return !isNaN(d.getTime());
}

function analyzeColumns(rows: any[], columns: string[]): ColumnAnalysis[] {
  const result: ColumnAnalysis[] = [];

  for (const col of columns) {
    const values = rows.map((r) => r[col]);
    const nonEmpty = values.filter(
      (v) => v !== null && v !== undefined && String(v).trim() !== ""
    );

    const missing = values.length - nonEmpty.length;
    const unique = new Set(nonEmpty.map((v) => String(v))).size;

    const exampleValues = nonEmpty
      .slice(0, 3)
      .map((v) => String(v))
      .filter(Boolean);

    let numericCount = 0;
    let dateCount = 0;
    let longTextCount = 0;

    for (const v of nonEmpty.slice(0, 30)) {
      const str = String(v);
      if (!isNaN(Number(v)) && str.trim() !== "") numericCount++;
      if (looksLikeDate(v)) dateCount++;
      if (str.length > 30) longTextCount++;
    }

    let type: ColumnType = "unknown";

    if (nonEmpty.length === 0) type = "unknown";
    else if (dateCount >= 3) type = "datetime";
    else if (numericCount >= 3 && numericCount >= nonEmpty.length * 0.7)
      type = "numeric";
    else if (longTextCount >= 3) type = "text";
    else {
      type =
        unique <= Math.max(20, Math.floor(rows.length * 0.2))
          ? "categorical"
          : "text";
    }

    result.push({
      name: col,
      type,
      missing,
      unique,
      exampleValues,
    });
  }

  return result;
}

/* =========================
   TARGET RECOMMENDATION
========================= */

function recommendTarget(
  rows: any[],
  columns: string[],
  analysis: ColumnAnalysis[]
): TargetRecommendation | null {
  if (!rows.length || !columns.length) return null;

  const n = rows.length;
  const lower = (s: string) => s.toLowerCase();

  // 🚫 identifiers / non-targets
  const bannedNames = [
    "id",
    "index",
    "uuid",
    "passengerid",
    "customerid",
    "userid",
    "name",
    "ticket",
    "address",
    "phone",
    "email",
  ];

  // ✅ label-like names that usually represent target
  const strongTargetHints = [
    "target",
    "label",
    "class",
    "outcome",
    "result",
    "y",
    "smoker",
    "survived",
    "default",
    "fraud",
    "diagnosis",
    "diabetes",
    "churn",
    "approved",
    "status",
  ];

  // ✅ usually feature-like numeric
  const commonFeatureNames = [
    "age",
    "height",
    "weight",
    "salary",
    "income",
    "fare",
    "price",
    "amount",
  ];

  const scored: TargetRecommendation[] = [];

  for (const col of columns) {
    const info = analysis.find((a) => a.name === col);
    if (!info) continue;

    const colName = lower(col);

    let score = 0;
    const reasons: string[] = [];

    // ❌ avoid IDs
    if (bannedNames.some((b) => colName.includes(b))) {
      score -= 100;
      reasons.push("Looks like an identifier (not a target).");
    }

    // ✅ missing ratio check
    const missingRatio = info.missing / Math.max(n, 1);

    if (missingRatio <= 0.05) {
      score += 20;
      reasons.push("Very low missing values.");
    } else if (missingRatio <= 0.2) {
      score += 5;
      reasons.push("Acceptable missing values.");
    } else {
      score -= 25;
      reasons.push("Too many missing values.");
    }

    // ✅ avoid fully unique columns
    if (info.unique === n) {
      score -= 60;
      reasons.push("Values are unique per row → likely not a target.");
    } else if (info.unique <= 1) {
      score -= 100;
      reasons.push("Only one class/value → not a target.");
    } else {
      score += 10;
      reasons.push("Not fully unique → can be a target.");
    }

    // ✅ strong hint from name
    if (strongTargetHints.some((h) => colName === h || colName.includes(h))) {
      score += 60;
      reasons.push("Column name strongly suggests it is the target.");
    }

    // ✅ penalize common feature columns unless name has strong hint
    if (
      commonFeatureNames.some((f) => colName === f) &&
      !strongTargetHints.some((h) => colName.includes(h))
    ) {
      score -= 25;
      reasons.push("This looks like a feature (not a label/target).");
    }

    // ✅ type logic
    if (info.type === "categorical") {
      // categorical/binary labels are best for classification
      if (info.unique <= 20) {
        score += 35;
        reasons.push("Categorical with limited classes → great for classification.");
      } else {
        score += 5;
        reasons.push("Categorical but many classes → may be harder.");
      }
    }

    if (info.type === "numeric") {
      // numeric usually regression unless very few unique values
      if (info.unique <= 10) {
        score += 10;
        reasons.push("Numeric but few unique values → possible classification.");
      } else {
        score -= 5;
        reasons.push("Numeric continuous target → regression (less common default).");
      }
    }

    if (info.type === "text") {
      // text targets are often messy unless few classes
      if (info.unique <= 30) {
        score += 5;
        reasons.push("Text with limited unique values → possible labels.");
      } else {
        score -= 10;
        reasons.push("Text with too many unique values → likely not target.");
      }
    }

    if (info.type === "datetime") {
      score -= 20;
      reasons.push("Datetime is rarely used as a target.");
    }

    // ✅ decide task type
    let taskType: TaskType = "unknown";

    if (info.type === "categorical") {
      taskType = "classification";
    } else if (info.type === "numeric") {
      taskType = info.unique <= 10 ? "classification" : "regression";
    } else if (info.type === "text") {
      taskType = info.unique <= 30 ? "classification" : "unknown";
    }

    scored.push({ target: col, taskType, score, reasons });
  }

  scored.sort((a, b) => b.score - a.score);

  return scored.length > 0 ? scored[0] : null;
}


/* =========================
   PAGE
========================= */

export default function Home() {
  const router = useRouter();

  /* =========================
     AUTH GUARD
  ========================= */
  useEffect(() => {
    const isAuthed = localStorage.getItem("aidex_auth") === "true";
    if (!isAuthed) router.replace("/auth");
  }, [router]);

  /* =========================
     UI STATES
  ========================= */
  const [showUploadMenu, setShowUploadMenu] = useState(false);
  const [showExplanationModal, setShowExplanationModal] = useState(false);

  /* =========================
     DATASET STATE
  ========================= */
  const [datasetKind, setDatasetKind] = useState<DatasetKind>("none");
  const [errorMsg, setErrorMsg] = useState("");

  /* ---------- structured ---------- */
  const [structured, setStructured] = useState<StructuredDataset | null>(null);
  const [csvLoading, setCsvLoading] = useState(false);
  const [targetColumn, setTargetColumn] = useState("");

  /* ---------- images ---------- */
  const [images, setImages] = useState<ImageItem[]>([]);
  const [folderLoading, setFolderLoading] = useState(false);
  const [labelDepth, setLabelDepth] = useState<number>(2);

  /* =========================
     RESET
  ========================= */
  const resetAll = () => {
    setDatasetKind("none");
    setErrorMsg("");

    setStructured(null);
    setTargetColumn("");
    setCsvLoading(false);

    setImages([]);
    setFolderLoading(false);

    setShowUploadMenu(false);
    setShowExplanationModal(false);
  };

  /* =========================
     UPLOAD HANDLERS
  ========================= */

  const handleStructuredUpload = (file: File) => {
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
      complete: (results) => {
        const rows = (results.data as any[]) || [];
        const columns = rows.length > 0 ? Object.keys(rows[0]) : [];
        setStructured({ fileName: file.name, rows, columns });
        setCsvLoading(false);
      },
      error: () => {
        resetAll();
        setDatasetKind("unsupported");
        setErrorMsg("Failed to parse CSV ❌");
        setCsvLoading(false);
      },
    });
  };

  const handleImageFolderUpload = (files: FileList) => {
    resetAll();
    setDatasetKind("images");
    setFolderLoading(true);

    const items: ImageItem[] = [];

    Array.from(files).forEach((file) => {
      if (!file.type.startsWith("image/")) return;
      const relativePath = (file as any).webkitRelativePath || file.name;
      items.push({ file, label: "unknown", relativePath });
    });

    if (items.length === 0) {
      resetAll();
      setDatasetKind("unsupported");
      setErrorMsg(
        "No image files found in folder. Please upload a folder that contains images."
      );
      setFolderLoading(false);
      return;
    }

    setImages(items);
    setFolderLoading(false);
  };

  /* =========================
     DERIVED: STRUCTURED
  ========================= */
  const structuredPreviewRows = useMemo(() => {
    if (!structured) return [];
    return structured.rows.slice(0, 10);
  }, [structured]);

  const columnAnalysis = useMemo(() => {
    if (!structured) return [];
    return analyzeColumns(structured.rows, structured.columns);
  }, [structured]);

  const targetRecommendation = useMemo(() => {
    if (!structured) return null;
    return recommendTarget(structured.rows, structured.columns, columnAnalysis);
  }, [structured, columnAnalysis]);

  /* =========================
     DERIVED: IMAGES
  ========================= */
  const relabeledImages = useMemo(() => {
    if (images.length === 0) return [];
    return images.map((img) => ({
      ...img,
      label: getLabelFromRelativePath(img.relativePath, labelDepth),
    }));
  }, [images, labelDepth]);

  const imageClassDist = useMemo(() => {
    return computeClassDistributionFromImages(relabeledImages);
  }, [relabeledImages]);

  const imagePreview = useMemo(() => {
    return getBalancedPreview(relabeledImages, 12);
  }, [relabeledImages]);

  /* =========================
     MODAL CLOSE (ESC)
  ========================= */
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setShowUploadMenu(false);
        setShowExplanationModal(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  /* =========================
     UI
  ========================= */
  return (
    <main className="min-h-screen bg-[#0B1020] text-white">
      <div className="max-w-6xl mx-auto px-6 py-10 space-y-6">
        {/* =========================
            STEP 1 — ONE UPLOAD AREA
        ========================= */}
        <div className="rounded-3xl border border-white/10 bg-white/5 p-6 sm:p-8">
          <div className="flex items-start justify-between flex-wrap gap-4">
            <div>
              <h2 className="text-xl sm:text-2xl font-semibold">
                Step 1 — Upload Dataset
              </h2>
              <p className="text-white/60 mt-1 text-sm">
                Upload your data and AIDEX will detect the type automatically.
              </p>
            </div>

            <button
              onClick={resetAll}
              className="rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-sm hover:bg-white/10 transition"
            >
              Reset
            </button>
          </div>

          <div className="mt-6 rounded-2xl border border-white/10 bg-white/5 p-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <p className="font-semibold text-lg">Choose your dataset</p>
              <p className="text-sm text-white/60 mt-1">
                Supported: CSV file or image folder.
              </p>
            </div>

            <div className="relative">
              <button
                onClick={() => setShowUploadMenu((prev) => !prev)}
                className="rounded-xl bg-white text-black px-5 py-2.5 text-sm font-semibold hover:bg-white/90 transition"
              >
                Choose Dataset
              </button>

              {showUploadMenu && (
                <div className="absolute right-0 mt-2 w-72 rounded-2xl border border-white/10 bg-[#0B1020] shadow-xl p-2 z-50">
                  <label className="block cursor-pointer rounded-xl px-4 py-3 hover:bg-white/10 transition">
                    <p className="font-semibold text-sm">
                      📄 Upload Dataset File
                    </p>
                    <p className="text-xs text-white/50 mt-1">CSV format</p>
                    <input
                      type="file"
                      accept=".csv"
                      className="hidden"
                      onChange={(e) => {
                        const f = e.target.files?.[0];
                        if (!f) return;
                        setShowUploadMenu(false);
                        handleStructuredUpload(f);
                      }}
                    />
                  </label>

                  <label className="block cursor-pointer rounded-xl px-4 py-3 hover:bg-white/10 transition">
                    <p className="font-semibold text-sm">
                      🖼️ Upload Dataset Folder
                    </p>
                    <p className="text-xs text-white/50 mt-1">
                      Folder of images
                    </p>
                    <input
                      type="file"
                      multiple
                      accept="image/*"
                      // @ts-ignore
                      webkitdirectory="true"
                      className="hidden"
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
          </div>

          {(csvLoading || folderLoading) && (
            <p className="text-sm text-cyan-300 mt-4">
              {csvLoading ? "Processing dataset..." : "Loading dataset..."}
            </p>
          )}

          {datasetKind !== "none" && (
            <div className="mt-6 rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="font-semibold">Dataset Detected</p>
              <p className="text-sm text-white/70 mt-1">
                {datasetKind === "structured" && "✅ CSV Dataset"}
                {datasetKind === "images" && "✅ Image Folder Dataset"}
                {datasetKind === "unsupported" && "❌ Unsupported Dataset"}
              </p>

              {errorMsg && (
                <p className="text-sm text-yellow-200 mt-2">⚠️ {errorMsg}</p>
              )}
            </div>
          )}
        </div>

        {/* =========================
            STRUCTURED PREVIEW
        ========================= */}
        {datasetKind === "structured" && structured && (
          <div className="rounded-3xl border border-white/10 bg-white/5 p-6 sm:p-8">
            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div>
                <h2 className="text-xl sm:text-2xl font-semibold">
                  Dataset Preview
                </h2>
                <p className="text-white/60 mt-1 text-sm">
                  File: {structured.fileName}
                </p>
              </div>

              <button
                onClick={() => setShowExplanationModal(true)}
                className="rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-sm hover:bg-white/10 transition"
              >
                View more explanation
              </button>
            </div>

            {/* ✅ Recommended target */}
            {targetRecommendation && (
              <div className="mt-6 rounded-2xl border border-white/10 bg-white/5 p-5">
                <div className="flex items-start justify-between gap-4 flex-wrap">
                  <div>
                    <p className="font-semibold text-lg">
                      ✅ Recommended Target
                    </p>

                    <p className="text-white/70 text-sm mt-1">
                      <span className="text-white font-semibold">
                        {targetRecommendation.target}
                      </span>{" "}
                      • Suggested task:{" "}
                      <span className="text-white font-semibold">
                        {targetRecommendation.taskType}
                      </span>
                    </p>
                  </div>

                  <button
                    onClick={() =>
                      setTargetColumn(targetRecommendation.target)
                    }
                    className="rounded-xl bg-white text-black px-4 py-2 text-sm font-semibold hover:bg-white/90 transition"
                  >
                    Use Recommended Target
                  </button>
                </div>

                {/* WHY explanation */}
                <div className="mt-4">
                  <p className="text-sm font-semibold text-white/90">
                    Why this is the best option:
                  </p>
                  <ul className="mt-2 space-y-1 text-sm text-white/70 list-disc pl-5">
                    {targetRecommendation.reasons.slice(0, 5).map((r, idx) => (
                      <li key={idx}>{r}</li>
                    ))}
                  </ul>

                  <p className="text-xs text-white/50 mt-3">
                    You can accept this recommendation or choose manually below.
                  </p>
                </div>
              </div>
            )}

            {/* Table preview */}
            <div className="mt-6 overflow-auto rounded-2xl border border-white/10">
              <table className="min-w-full text-sm">
                <thead className="bg-white/5 sticky top-0">
                  <tr>
                    {structured.columns.map((col) => (
                      <th
                        key={col}
                        className="px-4 py-3 text-left font-semibold text-white/80 whitespace-nowrap"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>

                <tbody>
                  {structuredPreviewRows.map((row, i) => (
                    <tr
                      key={i}
                      className="border-t border-white/10 hover:bg-white/5 transition"
                    >
                      {structured.columns.map((col) => (
                        <td
                          key={col}
                          className="px-4 py-3 text-white/80 whitespace-nowrap"
                        >
                          {row[col] === null || row[col] === undefined
                            ? ""
                            : String(row[col])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Manual select */}
            <div className="mt-6">
              <p className="font-semibold mb-2">
                Select Target Variable (Manual)
              </p>

              <select
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                className="w-full max-w-md rounded-xl bg-white/5 border border-white/10 px-4 py-3 text-sm text-white outline-none"
              >
                <option value="">-- Choose target --</option>
                {structured.columns.map((col) => (
                  <option key={col} value={col} className="text-black">
                    {col}
                  </option>
                ))}
              </select>

              {targetColumn && (
                <p className="text-sm text-white/70 mt-2">
                  ✅ Selected target:{" "}
                  <span className="font-semibold text-white">
                    {targetColumn}
                  </span>
                </p>
              )}
            </div>
          </div>
        )}

        {/* =========================
            IMAGE PREVIEW
        ========================= */}
        {datasetKind === "images" && relabeledImages.length > 0 && (
          <div className="rounded-3xl border border-white/10 bg-white/5 p-6 sm:p-8">
            <h2 className="text-xl sm:text-2xl font-semibold">
              Dataset Preview
            </h2>
            <p className="text-white/60 mt-1 text-sm">
              Labels are inferred from folder names.
            </p>

            <div className="mt-6 rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="font-semibold">Class Distribution</p>
              <div className="mt-2 text-sm text-white/70 space-y-1">
                {sortDistribution(imageClassDist).map(([label, count]) => (
                  <p key={label}>
                    {label}:{" "}
                    <span className="font-semibold text-white">{count}</span>
                  </p>
                ))}
              </div>
            </div>

            <div className="mt-6">
              <p className="font-semibold mb-3">Preview</p>

              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-3">
                {imagePreview.map((img) => (
                  <div
                    key={img.relativePath}
                    className="rounded-2xl border border-white/10 bg-white/5 p-2"
                  >
                    <img
                      src={URL.createObjectURL(img.file)}
                      alt={img.file.name}
                      className="w-full h-24 object-cover rounded-xl"
                    />
                    <p className="text-[10px] text-white/70 mt-2 truncate">
                      {img.label} • {img.file.name}
                    </p>
                  </div>
                ))}
              </div>

              <p className="text-xs text-white/50 mt-3">
                Showing first 12 samples (mixed across classes).
              </p>
            </div>
          </div>
        )}

        {/* =========================
            UNSUPPORTED
        ========================= */}
        {datasetKind === "unsupported" && (
          <div className="rounded-3xl border border-yellow-300/30 bg-yellow-500/10 p-6">
            <p className="font-semibold text-yellow-200">
              ⚠️ Dataset type not supported yet
            </p>
            <p className="text-sm text-yellow-200/80 mt-2">
              Currently supported: CSV file and image folder.
            </p>
          </div>
        )}
      </div>

      {/* =========================
          ✅ EXPLANATION MODAL
      ========================= */}
      {showExplanationModal && datasetKind === "structured" && structured && (
        <div
          className="fixed inset-0 z-[999] flex items-center justify-center bg-black/60 px-4"
          onClick={() => setShowExplanationModal(false)}
        >
          <div
            className="w-full max-w-3xl rounded-3xl border border-white/10 bg-[#0B1020] p-6 sm:p-8 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-xl font-bold">Detected Columns</h3>
                <p className="text-sm text-white/60 mt-1">
                  Column types, missing values, and unique counts.
                </p>
              </div>

              <button
                onClick={() => setShowExplanationModal(false)}
                className="rounded-xl border border-white/15 bg-white/5 px-3 py-2 text-sm hover:bg-white/10 transition"
              >
                ✕ Close
              </button>
            </div>

            <div className="mt-6 max-h-[60vh] overflow-auto rounded-2xl border border-white/10">
              <div className="p-4 space-y-3">
                {columnAnalysis.map((c) => (
                  <div
                    key={c.name}
                    className="rounded-2xl border border-white/10 bg-white/5 p-4"
                  >
                    <div className="flex items-center justify-between gap-4 flex-wrap">
                      <p className="font-semibold">{c.name}</p>
                      <p className="text-sm text-white/70">
                        Type:{" "}
                        <span className="font-semibold text-white">{c.type}</span>{" "}
                        | Missing:{" "}
                        <span className="font-semibold text-white">
                          {c.missing}
                        </span>{" "}
                        | Unique:{" "}
                        <span className="font-semibold text-white">{c.unique}</span>
                      </p>
                    </div>

                    {c.exampleValues.length > 0 && (
                      <p className="text-xs text-white/50 mt-2">
                        Examples:{" "}
                        <span className="text-white/70">
                          {c.exampleValues.join(" | ")}
                        </span>
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <p className="text-xs text-white/50 mt-4">
              Tip: These types are detected automatically and will be used later
              for preprocessing.
            </p>
          </div>
        </div>
      )}
    </main>
  );
}
