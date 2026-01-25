"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { useDataset } from "../context/DatasetContext";

/* =========================
   HELPERS
========================= */

function isMissing(v: any) {
  return v === null || v === undefined || String(v).trim() === "";
}

function safeNumber(v: any): number | null {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function mean(arr: number[]) {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function uniqRowKey(row: any, columns: string[]) {
  return columns.map((c) => String(row[c] ?? "")).join("||");
}

/* =========================
   IMAGE HELPERS
========================= */

async function loadImageFromFile(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();

    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to load image"));
    };

    img.src = url;
  });
}

async function resizeAndNormalizeImage(
  file: File,
  size: number = 224,
  quality: number = 0.92
): Promise<File | null> {
  if (!file.type.startsWith("image/")) return null;

  const img = await loadImageFromFile(file);

  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;

  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  // Fill background white (avoids transparent PNG issues)
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, size, size);

  // --- Keep aspect ratio (center crop) ---
  const srcW = img.width;
  const srcH = img.height;

  const srcAspect = srcW / srcH;
  const dstAspect = 1;

  let cropW = srcW;
  let cropH = srcH;

  if (srcAspect > dstAspect) {
    cropW = srcH;
    cropH = srcH;
  } else {
    cropW = srcW;
    cropH = srcW;
  }

  const sx = Math.floor((srcW - cropW) / 2);
  const sy = Math.floor((srcH - cropH) / 2);

  ctx.drawImage(img, sx, sy, cropW, cropH, 0, 0, size, size);

  const blob: Blob | null = await new Promise((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", quality)
  );

  if (!blob) return null;

  // ✅ FIX: include jpg too
  const newFile = new File(
    [blob],
    file.name.replace(/\.(png|webp|jpeg|jpg)$/i, ".jpg"),
    { type: "image/jpeg" }
  );

  return newFile;
}

/* =========================
   TYPES
========================= */

type DropReason =
  | "Constant column"
  | "Too many missing values"
  | "High correlation (simplified)";

type DroppedColumn = {
  name: string;
  reason: DropReason;
};

type SimplePreprocessSummary = {
  beforeRows: number;
  beforeCols: number;
  afterRows: number;
  afterCols: number;

  removedDuplicates: number;
  filledMissing: number;
  handledOutliers: number;
  scaledNumeric: boolean;
  encodedCategorical: boolean;

  droppedColumns: DroppedColumn[];
  notes: string[];
};

type ImagePreprocessSummary = {
  beforeCount: number;
  afterCount: number;

  resizedCount: number;
  skippedNonImages: number;
  removedCorrupted: number;

  targetSize: number;
  notes: string[];
};

/* =========================
   PAGE
========================= */

export default function PreprocessingPage() {
  const router = useRouter();

  // ✅ IMPORTANT: add setImages
  const { state, setStructured, setImages } = useDataset();

  const structured = state.structured;

  const [isRunning, setIsRunning] = useState(false);
  const [done, setDone] = useState(false);

  // ✅ NEW: store preview for UI
  const [imagePreview, setImagePreview] = useState<any[]>([]);

  // Structured summary
  const [summary, setSummary] = useState<SimplePreprocessSummary | null>(null);

  // Image summary
  const [imageSummary, setImageSummary] = useState<ImagePreprocessSummary | null>(
    null
  );

  const [protectedColumns, setProtectedColumns] = useState<string[]>([]);

  /* =========================
     AUTH GUARD
  ========================= */
  useEffect(() => {
    const isAuthed = localStorage.getItem("aidex_auth") === "true";
    if (!isAuthed) router.replace("/auth");
  }, [router]);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const raw = localStorage.getItem("aidex_protected_cols");
    if (!raw) return;

    try {
      setProtectedColumns(JSON.parse(raw));
    } catch {
      setProtectedColumns([]);
    }
  }, []);

  /* =========================
     MUST HAVE DATASET
  ========================= */
  useEffect(() => {
    if (state.datasetKind === "none") router.replace("/");
  }, [state.datasetKind, router]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    localStorage.setItem(
      "aidex_protected_cols",
      JSON.stringify(protectedColumns)
    );
  }, [protectedColumns]);

  const canRunStructured = useMemo(() => {
    return (
      state.datasetKind === "structured" && !!structured && !!state.targetColumn
    );
  }, [state.datasetKind, structured, state.targetColumn]);

  const canRunImages = useMemo(() => {
    return state.datasetKind === "images" && state.images.length > 0;
  }, [state.datasetKind, state.images.length]);

  /* =========================
     CORE CLEANING (GENERIC) - STRUCTURED
     ✅ UNCHANGED
  ========================= */

  function runGenericPreprocessing() {
    if (!structured) return;

    const target = state.targetColumn;

    let rows = structured.rows.map((r) => ({ ...r }));
    let columns = [...structured.columns];

    const beforeRows = rows.length;
    const beforeCols = columns.length;

    const droppedColumns: DroppedColumn[] = [];

    let removedDuplicates = 0;
    let filledMissing = 0;
    let handledOutliers = 0;

    // 1) Constant columns
    {
      const toDrop: string[] = [];

      for (const col of columns) {
        if (protectedColumns.includes(col)) continue;

        const nonEmpty = rows
          .map((r) => r[col])
          .filter((v) => !isMissing(v))
          .map((v) => String(v));

        const unique = new Set(nonEmpty);
        if (nonEmpty.length > 0 && unique.size <= 1) {
          toDrop.push(col);
        }
      }

      if (toDrop.length) {
        for (const c of toDrop)
          droppedColumns.push({ name: c, reason: "Constant column" });

        columns = columns.filter((c) => !toDrop.includes(c));
        rows = rows.map((r) => {
          const nr = { ...r };
          for (const c of toDrop) delete nr[c];
          return nr;
        });
      }
    }

    // 2) Duplicate rows
    {
      const seen = new Set<string>();
      const nextRows: any[] = [];

      for (const r of rows) {
        const key = uniqRowKey(r, columns);
        if (seen.has(key)) {
          removedDuplicates++;
        } else {
          seen.add(key);
          nextRows.push(r);
        }
      }

      rows = nextRows;
    }

    // 3) Drop columns with 40%+ missing
    {
      const threshold = 0.4;
      const toDrop: string[] = [];

      for (const col of columns) {
        if (protectedColumns.includes(col)) continue;
        if (col === target) continue;

        let missingCount = 0;
        for (const r of rows) {
          if (isMissing(r[col])) missingCount++;
        }

        const ratio = missingCount / Math.max(rows.length, 1);
        if (ratio >= threshold) toDrop.push(col);
      }

      if (toDrop.length) {
        for (const c of toDrop)
          droppedColumns.push({ name: c, reason: "Too many missing values" });

        columns = columns.filter((c) => !toDrop.includes(c));
        rows = rows.map((r) => {
          const nr = { ...r };
          for (const c of toDrop) delete nr[c];
          return nr;
        });
      }
    }

    // numeric / categorical detect
    const numericCols: string[] = [];
    const categoricalCols: string[] = [];

    for (const col of columns) {
      if (col === target) continue;

      let numericHits = 0;
      let sample = 0;

      for (const r of rows) {
        const v = r[col];
        if (isMissing(v)) continue;
        sample++;
        if (safeNumber(v) !== null) numericHits++;
        if (sample >= 25) break;
      }

      if (numericHits >= 5 && numericHits >= sample * 0.7) numericCols.push(col);
      else categoricalCols.push(col);
    }

    // 4) Imputation
    {
      for (const col of numericCols) {
        const nums: number[] = [];
        for (const r of rows) {
          const n = safeNumber(r[col]);
          if (n !== null) nums.push(n);
        }
        const fill = mean(nums);

        for (const r of rows) {
          if (isMissing(r[col])) {
            r[col] = fill;
            filledMissing++;
          }
        }
      }

      for (const col of categoricalCols) {
        const freq: Record<string, number> = {};

        for (const r of rows) {
          const v = r[col];
          if (isMissing(v)) continue;
          const s = String(v);
          freq[s] = (freq[s] || 0) + 1;
        }

        const mode =
          Object.entries(freq).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "Unknown";

        for (const r of rows) {
          if (isMissing(r[col])) {
            r[col] = mode;
            filledMissing++;
          }
        }
      }
    }

    // 5) Outliers cap
    {
      for (const col of numericCols) {
        const nums: number[] = [];
        for (const r of rows) {
          const n = safeNumber(r[col]);
          if (n !== null) nums.push(n);
        }
        if (nums.length < 10) continue;

        const m = mean(nums);
        const variance =
          nums.reduce((sum, x) => sum + (x - m) * (x - m), 0) /
          Math.max(nums.length, 1);
        const s = Math.sqrt(variance) || 1;

        const lower = m - 4 * s;
        const upper = m + 4 * s;

        for (const r of rows) {
          const n = safeNumber(r[col]);
          if (n === null) continue;

          if (n < lower) {
            r[col] = lower;
            handledOutliers++;
          } else if (n > upper) {
            r[col] = upper;
            handledOutliers++;
          }
        }
      }
    }

    // 6) MinMax scaling
    {
      for (const col of numericCols) {
        const nums: number[] = [];
        for (const r of rows) {
          const n = safeNumber(r[col]);
          if (n !== null) nums.push(n);
        }
        if (!nums.length) continue;

        const minVal = Math.min(...nums);
        const maxVal = Math.max(...nums);
        const range = maxVal - minVal || 1;

        for (const r of rows) {
          const n = safeNumber(r[col]);
          if (n === null) continue;
          r[col] = (n - minVal) / range;
        }
      }
    }

    // 8 + 9) One-hot encoding
    let encodedCategorical = false;
    {
      if (categoricalCols.length > 0) {
        encodedCategorical = true;

        const newCols: string[] = [];
        const removeCols: string[] = [];

        for (const col of categoricalCols) {
          const uniq = new Set<string>();
          for (const r of rows) {
            const v = isMissing(r[col]) ? "Unknown" : String(r[col]);
            uniq.add(v);
            if (uniq.size >= 12) break;
          }
          const categories = Array.from(uniq);

          for (const cat of categories) {
            const safe = String(cat).replace(/\s+/g, "_").slice(0, 25);
            newCols.push(`${col}__${safe}`);
          }

          removeCols.push(col);

          for (const r of rows) {
            const v = isMissing(r[col]) ? "Unknown" : String(r[col]);
            for (const cat of categories) {
              const safe = String(cat).replace(/\s+/g, "_").slice(0, 25);
              const newCol = `${col}__${safe}`;
              r[newCol] = v === cat ? 1 : 0;
            }
            delete r[col];
          }
        }

        columns = columns.filter((c) => !removeCols.includes(c));
        columns.push(...newCols);
      }
    }

    const afterRows = rows.length;
    const afterCols = columns.length;

    const notes: string[] = [];
    if (removedDuplicates > 0)
      notes.push("Duplicate records were removed to improve data quality.");
    if (filledMissing > 0) notes.push("Missing values were filled automatically.");
    if (handledOutliers > 0)
      notes.push("Extreme values were adjusted to reduce noise.");
    notes.push("Numerical values were normalized to improve model performance.");
    if (encodedCategorical)
      notes.push("Text categories were converted into machine-readable format.");

    const summary: SimplePreprocessSummary = {
      beforeRows,
      beforeCols,
      afterRows,
      afterCols,
      removedDuplicates,
      filledMissing,
      handledOutliers,
      scaledNumeric: numericCols.length > 0,
      encodedCategorical,
      droppedColumns,
      notes,
    };

    return { rows, columns, summary };
  }

  /* =========================
     GENERIC IMAGE PREPROCESSING ✅ NEW
  ========================= */

  async function runGenericImagePreprocessing() {
    const beforeCount = state.images.length;
    const targetSize = 224;

    let resizedCount = 0;
    let skippedNonImages = 0;
    let removedCorrupted = 0;

    const cleaned: any[] = [];

    for (const item of state.images) {
      try {
        if (!item.file || !item.file.type?.startsWith("image/")) {
          skippedNonImages++;
          continue;
        }

        const newFile = await resizeAndNormalizeImage(item.file, targetSize, 0.92);
        if (!newFile) {
          removedCorrupted++;
          continue;
        }

        resizedCount++;

        cleaned.push({
          ...item,
          file: newFile,
          label: item.label ?? "unknown",
          relativePath: item.relativePath ?? newFile.name,
        });
      } catch {
        removedCorrupted++;
      }
    }

    const afterCount = cleaned.length;

    // ✅ store preview (8 images)
    setImagePreview(cleaned.slice(0, 8));

    const notes: string[] = [];
    notes.push("Images were resized into a consistent format for training.");
    notes.push("Corrupted or unreadable images were removed.");
    notes.push("Images were standardized to improve model stability.");

    const imageSummary: ImagePreprocessSummary = {
      beforeCount,
      afterCount,
      resizedCount,
      skippedNonImages,
      removedCorrupted,
      targetSize,
      notes,
    };

    return { cleaned, imageSummary };
  }

  /* =========================
     START BUTTON
  ========================= */

  const handleStart = async () => {
    setIsRunning(true);
    setDone(false);
    setSummary(null);
    setImageSummary(null);
    setImagePreview([]);

    setTimeout(async () => {
      // ✅ STRUCTURED
      if (state.datasetKind === "structured") {
        if (!canRunStructured || !structured) {
          setIsRunning(false);
          return;
        }

        const result = runGenericPreprocessing();
        if (!result) {
          setIsRunning(false);
          return;
        }

        const { rows, columns, summary } = result;

        setStructured({
          fileName: structured.fileName,
          rows,
          columns,
        });

        setSummary(summary);
        setIsRunning(false);
        setDone(true);
        return;
      }

      // ✅ IMAGES
      if (state.datasetKind === "images") {
        if (!canRunImages) {
          setIsRunning(false);
          return;
        }

        const { cleaned, imageSummary } = await runGenericImagePreprocessing();

        setImages(cleaned);
        setImageSummary(imageSummary);

        setIsRunning(false);
        setDone(true);
        return;
      }

      setIsRunning(false);
    }, 1200);
  };

  const markImportant = (col: string) => {
    setProtectedColumns((prev) => {
      if (prev.includes(col)) return prev;
      return [...prev, col];
    });
  };

  /* =========================
     UI
  ========================= */

  return (
    <main className="w-full">
      <div className="max-w-6xl space-y-5">
        {/* Header */}
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-3xl font-bold">Step 3 — Data Preparation</h1>
            <p className="mt-2 text-sm" style={{ color: "var(--muted)" }}>
              AIDEX will automatically prepare your dataset to improve accuracy and reduce errors.
            </p>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <button onClick={() => router.push("/overview")} className="aidex-btn-outline">
              Back
            </button>
            <button onClick={() => router.push("/")} className="aidex-btn-outline">
              Upload New Dataset
            </button>
          </div>
        </div>

        {/* Guard */}
        {state.datasetKind === "structured" && !canRunStructured && (
          <div className="aidex-card">
            <p className="aidex-card-title">Before you continue</p>
            <p className="text-sm" style={{ color: "var(--muted)", marginTop: 6 }}>
              Please make sure you uploaded a CSV dataset and selected a target column in the Overview step.
            </p>

            <div style={{ marginTop: 12 }}>
              <button onClick={() => router.push("/overview")} className="aidex-btn-primary">
                Go to Overview
              </button>
            </div>
          </div>
        )}

        {state.datasetKind === "images" && !canRunImages && (
          <div className="aidex-card">
            <p className="aidex-card-title">Before you continue</p>
            <p className="text-sm" style={{ color: "var(--muted)", marginTop: 6 }}>
              Please make sure you uploaded an image folder dataset first.
            </p>

            <div style={{ marginTop: 12 }}>
              <button onClick={() => router.push("/")} className="aidex-btn-primary">
                Go to Upload
              </button>
            </div>
          </div>
        )}

        {/* Start Card */}
        {!done && (canRunStructured || canRunImages) && (
          <div className="aidex-card">
            <p className="aidex-card-title">Automatic Data Preparation</p>
            <p className="text-sm" style={{ color: "var(--muted)", marginTop: 6 }}>
              This process will clean and format your data automatically. You do not need to configure anything.
            </p>

            <div style={{ marginTop: 16, display: "flex", gap: 10, flexWrap: "wrap" }}>
              <button
                className="aidex-btn-primary"
                onClick={handleStart}
                disabled={isRunning}
                style={{
                  opacity: isRunning ? 0.6 : 1,
                  cursor: isRunning ? "not-allowed" : "pointer",
                }}
              >
                Start Data Preparation
              </button>
            </div>

            {isRunning && (
              <div style={{ marginTop: 16 }}>
                <p className="text-sm" style={{ color: "var(--muted)" }}>
                  Preparing your dataset. Please wait...
                </p>

                <div
                  style={{
                    marginTop: 10,
                    height: 10,
                    width: "100%",
                    borderRadius: 999,
                    background: "rgba(15, 23, 42, 0.08)",
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: "70%",
                      borderRadius: 999,
                      background: "var(--primary)",
                      animation: "aidexPulse 1.1s ease-in-out infinite",
                    }}
                  />
                </div>

                <style jsx>{`
                  @keyframes aidexPulse {
                    0% {
                      transform: translateX(-40%);
                      opacity: 0.5;
                    }
                    50% {
                      transform: translateX(10%);
                      opacity: 1;
                    }
                    100% {
                      transform: translateX(55%);
                      opacity: 0.5;
                    }
                  }
                `}</style>
              </div>
            )}
          </div>
        )}

        {/* RESULT (STRUCTURED) ✅ SAME AS YOU */}
        {done && summary && state.datasetKind === "structured" && (
          <div className="aidex-card">
            <p className="aidex-card-title">Data Preparation Completed</p>
            <p className="text-sm" style={{ color: "var(--muted)", marginTop: 6 }}>
              Here is a simple summary of what AIDEX improved in your dataset.
            </p>

            <div
              style={{
                marginTop: 14,
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
                gap: 12,
              }}
            >
              <div
                style={{
                  borderRadius: 14,
                  border: "1px solid var(--border)",
                  background: "rgba(15, 23, 42, 0.03)",
                  padding: 14,
                }}
              >
                <p style={{ fontSize: 12, color: "var(--muted)" }}>Rows</p>
                <p style={{ fontSize: 18, fontWeight: 800 }}>
                  {summary.beforeRows} → {summary.afterRows}
                </p>
              </div>

              <div
                style={{
                  borderRadius: 14,
                  border: "1px solid var(--border)",
                  background: "rgba(15, 23, 42, 0.03)",
                  padding: 14,
                }}
              >
                <p style={{ fontSize: 12, color: "var(--muted)" }}>Columns</p>
                <p style={{ fontSize: 18, fontWeight: 800 }}>
                  {summary.beforeCols} → {summary.afterCols}
                </p>
              </div>

              <div
                style={{
                  borderRadius: 14,
                  border: "1px solid var(--border)",
                  background: "rgba(15, 23, 42, 0.03)",
                  padding: 14,
                }}
              >
                <p style={{ fontSize: 12, color: "var(--muted)" }}>Duplicates removed</p>
                <p style={{ fontSize: 18, fontWeight: 800 }}>
                  {summary.removedDuplicates}
                </p>
              </div>

              <div
                style={{
                  borderRadius: 14,
                  border: "1px solid var(--border)",
                  background: "rgba(15, 23, 42, 0.03)",
                  padding: 14,
                }}
              >
                <p style={{ fontSize: 12, color: "var(--muted)" }}>Missing values fixed</p>
                <p style={{ fontSize: 18, fontWeight: 800 }}>
                  {summary.filledMissing}
                </p>
              </div>
            </div>

            <div style={{ marginTop: 16 }}>
              <p style={{ fontWeight: 800 }}>What AIDEX did for you:</p>
              <ul
                style={{
                  marginTop: 8,
                  paddingLeft: 18,
                  color: "var(--muted)",
                  fontSize: 13,
                }}
              >
                {summary.notes.map((n, i) => (
                  <li key={i} style={{ marginBottom: 6 }}>
                    {n}
                  </li>
                ))}
              </ul>
            </div>

            {summary.droppedColumns.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <p style={{ fontWeight: 800 }}>
                  Columns removed to improve quality (review):
                </p>

                <div
                  style={{
                    marginTop: 10,
                    display: "flex",
                    flexDirection: "column",
                    gap: 10,
                  }}
                >
                  {summary.droppedColumns.map((c) => (
                    <div
                      key={c.name}
                      style={{
                        borderRadius: 14,
                        border: "1px solid var(--border)",
                        background: "rgba(15, 23, 42, 0.03)",
                        padding: 12,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        gap: 10,
                        flexWrap: "wrap",
                      }}
                    >
                      <div>
                        <p style={{ fontWeight: 900 }}>{c.name}</p>
                        <p
                          style={{
                            fontSize: 13,
                            color: "var(--muted)",
                            marginTop: 4,
                          }}
                        >
                          Reason: {c.reason}
                        </p>
                      </div>

                      <button
                        className="aidex-btn-outline"
                        onClick={() => markImportant(c.name)}
                        disabled={protectedColumns.includes(c.name)}
                        style={{
                          opacity: protectedColumns.includes(c.name) ? 0.6 : 1,
                          cursor: protectedColumns.includes(c.name)
                            ? "not-allowed"
                            : "pointer",
                        }}
                      >
                        {protectedColumns.includes(c.name)
                          ? "Marked as Important"
                          : "This column is important"}
                      </button>
                    </div>
                  ))}
                </div>

                <p
                  className="text-xs"
                  style={{ color: "var(--muted)", marginTop: 10 }}
                >
                  If you marked a column as important, upload again or re-run preparation later and AIDEX will protect it.
                </p>
              </div>
            )}
          </div>
        )}

        {/* RESULT (IMAGES) ✅ NEW + PREVIEW GRID */}
        {done && imageSummary && state.datasetKind === "images" && (
          <div className="aidex-card">
            <p className="aidex-card-title">Image Preparation Completed</p>
            <p className="text-sm" style={{ color: "var(--muted)", marginTop: 6 }}>
              AIDEX prepared your images so they become consistent and ready for training.
            </p>

            <div
              style={{
                marginTop: 14,
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                gap: 12,
              }}
            >
              <div
                style={{
                  borderRadius: 14,
                  border: "1px solid var(--border)",
                  background: "rgba(15, 23, 42, 0.03)",
                  padding: 14,
                }}
              >
                <p style={{ fontSize: 12, color: "var(--muted)" }}>Images</p>
                <p style={{ fontSize: 18, fontWeight: 800 }}>
                  {imageSummary.beforeCount} → {imageSummary.afterCount}
                </p>
              </div>

              <div
                style={{
                  borderRadius: 14,
                  border: "1px solid var(--border)",
                  background: "rgba(15, 23, 42, 0.03)",
                  padding: 14,
                }}
              >
                <p style={{ fontSize: 12, color: "var(--muted)" }}>Resized</p>
                <p style={{ fontSize: 18, fontWeight: 800 }}>
                  {imageSummary.resizedCount}
                </p>
              </div>

              <div
                style={{
                  borderRadius: 14,
                  border: "1px solid var(--border)",
                  background: "rgba(15, 23, 42, 0.03)",
                  padding: 14,
                }}
              >
                <p style={{ fontSize: 12, color: "var(--muted)" }}>Removed corrupted</p>
                <p style={{ fontSize: 18, fontWeight: 800 }}>
                  {imageSummary.removedCorrupted}
                </p>
              </div>

              <div
                style={{
                  borderRadius: 14,
                  border: "1px solid var(--border)",
                  background: "rgba(15, 23, 42, 0.03)",
                  padding: 14,
                }}
              >
                <p style={{ fontSize: 12, color: "var(--muted)" }}>Image size</p>
                <p style={{ fontSize: 18, fontWeight: 800 }}>
                  {imageSummary.targetSize} × {imageSummary.targetSize}
                </p>
              </div>
            </div>

            <div style={{ marginTop: 16 }}>
              <p style={{ fontWeight: 800 }}>What AIDEX did for you:</p>
              <ul
                style={{
                  marginTop: 8,
                  paddingLeft: 18,
                  color: "var(--muted)",
                  fontSize: 13,
                }}
              >
                {imageSummary.notes.map((n, i) => (
                  <li key={i} style={{ marginBottom: 6 }}>
                    {n}
                  </li>
                ))}
              </ul>
            </div>

            {/* ✅ IMAGE PREVIEW GRID */}
            {imagePreview.length > 0 && (
              <div style={{ marginTop: 18 }}>
                <p style={{ fontWeight: 800 }}>Preview (after preparation):</p>

                <div
                  style={{
                    marginTop: 10,
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
                    gap: 12,
                  }}
                >
                  {imagePreview.map((img, idx) => (
                    <div
                      key={img.relativePath ?? idx}
                      style={{
                        borderRadius: 14,
                        border: "1px solid var(--border)",
                        background: "rgba(15, 23, 42, 0.03)",
                        padding: 10,
                      }}
                    >
                      <img
                        src={URL.createObjectURL(img.file)}
                        alt={img.file?.name ?? "image"}
                        style={{
                          width: "100%",
                          height: 100,
                          objectFit: "cover",
                          borderRadius: 10,
                        }}
                      />

                      <p
                        style={{
                          marginTop: 8,
                          fontSize: 11,
                          color: "var(--muted)",
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                        }}
                      >
                        {img.label ?? "unknown"} • {img.file?.name ?? "image"}
                      </p>
                    </div>
                  ))}
                </div>

                <p className="text-xs" style={{ color: "var(--muted)", marginTop: 10 }}>
                  Showing 8 prepared samples.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
