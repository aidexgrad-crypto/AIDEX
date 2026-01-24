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
  filledMissing: number; // total filled cells
  handledOutliers: number; // total affected cells
  scaledNumeric: boolean;
  encodedCategorical: boolean;

  droppedColumns: DroppedColumn[];
  notes: string[];
};

/* =========================
   PAGE
========================= */

export default function PreprocessingPage() {
  const router = useRouter();
  const { state, setStructured } = useDataset();

  const structured = state.structured;

  const [isRunning, setIsRunning] = useState(false);
  const [done, setDone] = useState(false);

  const [summary, setSummary] = useState<SimplePreprocessSummary | null>(null);

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
  localStorage.setItem("aidex_protected_cols", JSON.stringify(protectedColumns));
}, [protectedColumns]);


  const canRun = useMemo(() => {
    return state.datasetKind === "structured" && !!structured && !!state.targetColumn;
  }, [state.datasetKind, structured, state.targetColumn]);

  /* =========================
     CORE CLEANING (GENERIC)
     Runs automatically with your 10 steps in a "silent way"
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

    // -------------------------
    // 1) Removing Constant Columns
    // -------------------------
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
        for (const c of toDrop) droppedColumns.push({ name: c, reason: "Constant column" });

        columns = columns.filter((c) => !toDrop.includes(c));
        rows = rows.map((r) => {
          const nr = { ...r };
          for (const c of toDrop) delete nr[c];
          return nr;
        });
      }
    }

    // -------------------------
    // 2) Removing Duplicate Rows
    // -------------------------
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

    // -------------------------
    // 3) Drop Columns if +40% NaNs
    // -------------------------
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

    // Detect numeric / categorical columns
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

    // -------------------------
    // 4) NaNs Imputation
    // (numeric -> mean, categorical -> most frequent)
    // -------------------------
    {
      // numeric mean fill
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

      // categorical mode fill
      for (const col of categoricalCols) {
        const freq: Record<string, number> = {};

        for (const r of rows) {
          const v = r[col];
          if (isMissing(v)) continue;
          const s = String(v);
          freq[s] = (freq[s] || 0) + 1;
        }

        const mode = Object.entries(freq).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "Unknown";

        for (const r of rows) {
          if (isMissing(r[col])) {
            r[col] = mode;
            filledMissing++;
          }
        }
      }
    }

    // -------------------------
    // 5) Removing Outliers (simple capping)
    // -------------------------
    {
      // For simplicity: cap values outside mean ± 4*std
      for (const col of numericCols) {
        const nums: number[] = [];
        for (const r of rows) {
          const n = safeNumber(r[col]);
          if (n !== null) nums.push(n);
        }
        if (nums.length < 10) continue;

        const m = mean(nums);
        const variance =
          nums.reduce((sum, x) => sum + (x - m) * (x - m), 0) / Math.max(nums.length, 1);
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

    // -------------------------
    // 6) MinMax scaling (default)
    // -------------------------
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

    // -------------------------
    // 7) Z-score scaling
    // (we will NOT apply this now, because we already applied MinMax)
    // -------------------------

    // -------------------------
    // 8 + 9) Encoding
    // We will do ONE HOT encoding (simple, limited categories)
    // -------------------------
    let encodedCategorical = false;
    {
      if (categoricalCols.length > 0) {
        encodedCategorical = true;

        const newCols: string[] = [];
        const removeCols: string[] = [];

        for (const col of categoricalCols) {
          // safety limit
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

          // apply one-hot
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

    // -------------------------
    // 10) Remove high correlations (simplified)
    // For now: we will NOT aggressively drop columns here
    // (because users may get confused when columns disappear)
    // We'll keep it "gentle" and safe.
    // -------------------------

    const afterRows = rows.length;
    const afterCols = columns.length;

    const notes: string[] = [];
    if (removedDuplicates > 0) notes.push("Duplicate records were removed to improve data quality.");
    if (filledMissing > 0) notes.push("Missing values were filled automatically.");
    if (handledOutliers > 0) notes.push("Extreme values were adjusted to reduce noise.");
    notes.push("Numerical values were normalized to improve model performance.");
    if (encodedCategorical) notes.push("Text categories were converted into machine-readable format.");

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

  const handleStart = async () => {
    if (!canRun || !structured) return;

    setIsRunning(true);
    setDone(false);
    setSummary(null);

    // ✅ show user-friendly "processing" phase
    setTimeout(() => {
      const { rows, columns, summary } = runGenericPreprocessing();

      setStructured({
        fileName: structured.fileName,
        rows,
        columns,
      });

      setSummary(summary);
      setIsRunning(false);
      setDone(true);
    }, 1200);
  };

  const markImportant = (col: string) => {
    setProtectedColumns((prev) => {
      if (prev.includes(col)) return prev;
      return [...prev, col];
    });
  };

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

        {/* Guard message */}
        {!canRun && (
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

        {/* Start Card */}
        {canRun && !done && (
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

        {/* RESULT */}
        {done && summary && (
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

            {/* Explanation (simple text) */}
            <div style={{ marginTop: 16 }}>
              <p style={{ fontWeight: 800 }}>What AIDEX did for you:</p>
              <ul style={{ marginTop: 8, paddingLeft: 18, color: "var(--muted)", fontSize: 13 }}>
                {summary.notes.map((n, i) => (
                  <li key={i} style={{ marginBottom: 6 }}>
                    {n}
                  </li>
                ))}
              </ul>
            </div>

            {/* Dropped Columns */}
            {summary.droppedColumns.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <p style={{ fontWeight: 800 }}>
                  Columns removed to improve quality (review):
                </p>

                <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 10 }}>
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
                        <p style={{ fontSize: 13, color: "var(--muted)", marginTop: 4 }}>
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

                <p className="text-xs" style={{ color: "var(--muted)", marginTop: 10 }}>
                  If you marked a column as important, upload again or re-run preparation later and AIDEX will protect it.
                </p>
              </div>
            )}

          </div>
        )}
      </div>
    </main>
  );
}
