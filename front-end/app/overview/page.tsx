"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { useDataset } from "../context/DatasetContext";

/* =========================
   TYPES
========================= */

type ColumnType = "numeric" | "categorical" | "text" | "datetime" | "unknown";

type ColumnAnalysis = {
  name: string;
  type: ColumnType;
  missing: number;
  unique: number;
  exampleValues: string[];
};

type TargetRecommendation = {
  target: string;
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

function computeClassDistributionFromImages(
  images: { label: string }[]
): Record<string, number> {
  const dist: Record<string, number> = {};
  for (const img of images) dist[img.label] = (dist[img.label] || 0) + 1;
  return dist;
}

function sortDistribution(dist: Record<string, number>) {
  return Object.entries(dist).sort((a, b) => b[1] - a[1]);
}

function getBalancedPreview<T extends { label: string }>(
  images: T[],
  maxCount: number
) {
  const groups: Record<string, T[]> = {};
  for (const img of images) {
    if (!groups[img.label]) groups[img.label] = [];
    groups[img.label].push(img);
  }
  const labels = Object.keys(groups);
  if (labels.length === 0) return [];

  for (const label of labels) groups[label].sort(() => Math.random() - 0.5);

  const result: T[] = [];
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

function recommendTarget(
  rows: any[],
  columns: string[],
  analysis: ColumnAnalysis[]
): TargetRecommendation | null {
  if (!rows.length || !columns.length) return null;

  const n = rows.length;
  const lower = (s: string) => s.toLowerCase();

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

  const scored: TargetRecommendation[] = [];

  for (const col of columns) {
    const info = analysis.find((a) => a.name === col);
    if (!info) continue;

    const colName = lower(col);
    let score = 0;
    const reasons: string[] = [];

    if (bannedNames.some((b) => colName.includes(b))) {
      score -= 100;
      reasons.push("Looks like an identifier (not something we should predict).");
    }

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

    if (info.unique === n) {
      score -= 60;
      reasons.push("Values are unique per row → not a good prediction target.");
    } else if (info.unique <= 1) {
      score -= 100;
      reasons.push("Only one value → cannot be predicted.");
    } else {
      score += 10;
      reasons.push("Not fully unique → can be predicted.");
    }

    if (strongTargetHints.some((h) => colName === h || colName.includes(h))) {
      score += 60;
      reasons.push("Column name strongly suggests it is the output you want.");
    }

    if (info.type === "categorical") {
      score += info.unique <= 20 ? 35 : 5;
      reasons.push(
        info.unique <= 20
          ? "Has a limited number of values → good for predicting categories."
          : "Has many categories → might still work, but harder."
      );
    }

    if (info.type === "numeric") {
      score += info.unique <= 10 ? 5 : 10;
      reasons.push(
        info.unique <= 10
          ? "Numeric but few unique values → could behave like categories."
          : "Numeric target with many values → good for predicting a number."
      );
    }

    if (info.type === "datetime") {
      score -= 20;
      reasons.push("Dates/timestamps are rarely used as a prediction target.");
    }

    scored.push({ target: col, score, reasons });
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.length > 0 ? scored[0] : null;
}

/* =========================
   PAGE
========================= */

export default function OverviewPage() {
  const router = useRouter();
  const { state, setTargetColumn, setTaskChoice, resetAll } = useDataset();

  const [showExplanationModal, setShowExplanationModal] = useState(false);
  const [labelDepth, setLabelDepth] = useState<number>(2);

  useEffect(() => {
    const isAuthed = localStorage.getItem("aidex_auth") === "true";
    if (!isAuthed) router.replace("/auth");
  }, [router]);

  useEffect(() => {
    if (state.datasetKind === "none") {
      router.replace("/");
    }
  }, [state.datasetKind, router]);

  const resetPage2 = () => {
    resetAll();
    setShowExplanationModal(false);
    router.push("/");
  };

  const structured = state.structured;

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

  const relabeledImages = useMemo(() => {
    if (state.images.length === 0) return [];
    return state.images.map((img) => ({
      ...img,
      label: getLabelFromRelativePath(img.relativePath, labelDepth),
    }));
  }, [state.images, labelDepth]);

  const imageClassDist = useMemo(() => {
    return computeClassDistributionFromImages(relabeledImages);
  }, [relabeledImages]);

  const imageStats = useMemo(() => {
    const totalImages = relabeledImages.length;
    const entries = Object.entries(imageClassDist);
    const totalClasses = entries.length;

    let largestClassCount = 0;
    let smallestClassCount = 0;

    if (entries.length > 0) {
      const counts = entries.map(([, c]) => c);
      largestClassCount = Math.max(...counts);
      smallestClassCount = Math.min(...counts);
    }

    const imbalanceRatio =
      smallestClassCount > 0 ? largestClassCount / smallestClassCount : 0;

    const hasImbalanceWarning =
      totalClasses > 1 && (imbalanceRatio >= 3 || smallestClassCount < 5);

    return {
      totalImages,
      totalClasses,
      largestClassCount,
      smallestClassCount,
      imbalanceRatio,
      hasImbalanceWarning,
    };
  }, [relabeledImages, imageClassDist]);

  const imagePreview = useMemo(() => {
    return getBalancedPreview(relabeledImages, 12);
  }, [relabeledImages]);

  const canContinueStructured =
    !!structured && !!state.targetColumn && !!state.taskChoice;

 return (
  <main className="w-full">
    <div className="max-w-6xl w-full space-y-8 pb-32">
      {/* =========================
          HEADER
      ========================= */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-3xl font-bold">Data Overview</h1>
          <p className="mt-2 text-sm" style={{ color: "var(--muted)" }}>
            Preview your dataset and select your prediction goal.
          </p>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={resetPage2} className="aidex-btn-outline">
            Reset & Upload Again
          </button>
        </div>
      </div>

      {/* =========================
          STRUCTURED OVERVIEW
      ========================= */}
      {state.datasetKind === "structured" && structured && (
        <div className="space-y-8">
          {/* Dataset info */}
          <div className="aidex-card" style={{ padding: 18 }}>
            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div>
                <p className="aidex-card-title">Dataset Preview</p>
                <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
                  File: <b style={{ color: "var(--text)" }}>{structured.fileName}</b>{" "}
                  • Rows: <b style={{ color: "var(--text)" }}>{structured.rows.length}</b>{" "}
                  • Columns:{" "}
                  <b style={{ color: "var(--text)" }}>{structured.columns.length}</b>
                </p>
              </div>

              <button
                onClick={() => setShowExplanationModal(true)}
                className="aidex-btn-outline"
              >
                View details
              </button>
            </div>
          </div>

          {/* Recommended target */}
          {targetRecommendation && (
            <div className="aidex-card" style={{ padding: 18 }}>
              <div className="flex items-start justify-between gap-4 flex-wrap">
                <div>
                  <p className="aidex-card-title">Recommended Target</p>
                  <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
                    Suggested column to predict:{" "}
                    <b style={{ color: "var(--text)" }}>
                      {targetRecommendation.target}
                    </b>
                  </p>
                </div>

                <button
                  onClick={() => setTargetColumn(targetRecommendation.target)}
                  className="aidex-btn-primary"
                >
                  Use Recommendation
                </button>
              </div>

              <div className="mt-4">
                <p style={{ fontWeight: 800, fontSize: 13, marginBottom: 10 }}>
                  Why this is a good target:
                </p>

                <ul
                  style={{
                    paddingLeft: 18,
                    color: "var(--muted)",
                    fontSize: 13,
                    display: "flex",
                    flexDirection: "column",
                    gap: 8,
                  }}
                >
                  {targetRecommendation.reasons.slice(0, 5).map((r, idx) => (
                    <li key={idx}>{r}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Target + Task grid */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
              gap: 18,
              alignItems: "start",
            }}
          >
            {/* Choose target */}
            <div className="aidex-card" style={{ padding: 18 }}>
              <p className="aidex-card-title">Choose Target</p>
              <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
                The target is the column AIDEX will predict.
              </p>

              <select
                value={state.targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                style={{
                  marginTop: 14,
                  width: "100%",
                  borderRadius: 14,
                  border: "1px solid rgba(255,255,255,0.14)",
                  padding: "11px 12px",
                  outline: "none",
                  fontSize: 14,
                  background: "rgba(255,255,255,0.08)",
                  color: "white",
                }}
              >
                <option value="">-- Choose target --</option>
                {structured.columns.map((col) => (
                  <option key={col} value={col} style={{ color: "#0b1020" }}>
                    {col}
                  </option>
                ))}
              </select>

              {state.targetColumn && (
                <div
                  style={{
                    marginTop: 14,
                    borderRadius: 14,
                    border: "1px solid rgba(255,255,255,0.14)",
                    background: "rgba(255,255,255,0.06)",
                    padding: 14,
                  }}
                >
                  <p className="text-sm" style={{ color: "var(--muted)" }}>
                    Selected:
                  </p>
                  <p style={{ fontWeight: 900, marginTop: 4 }}>
                    {state.targetColumn}
                  </p>
                </div>
              )}
            </div>

            {/* Task selection */}
            <div className="aidex-card" style={{ padding: 18 }}>
              <p className="aidex-card-title">Select Analysis Type</p>
              <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
                Choose what you want AIDEX to predict.
              </p>

              <div
                style={{
                  marginTop: 14,
                  display: "flex",
                  flexDirection: "column",
                  gap: 14,
                }}
              >
                <button
                  onClick={() => setTaskChoice("predictCategory")}
                  style={{
                    textAlign: "left",
                    borderRadius: 18,
                    border:
                      state.taskChoice === "predictCategory"
                        ? `2px solid var(--primary)`
                        : "1px solid rgba(255,255,255,0.14)",
                    background:
                      state.taskChoice === "predictCategory"
                        ? "rgba(99,102,241,0.15)"
                        : "rgba(255,255,255,0.06)",
                    padding: 16,
                    cursor: "pointer",
                    transition: "0.2s",
                  }}
                >
                  <p style={{ fontWeight: 900, color: "white" }}>
                    Predict a category / label
                  </p>
                  <p style={{ fontSize: 13, color: "var(--muted)", marginTop: 6 }}>
                    Example: Fraud / Not Fraud, Survived / Not Survived
                  </p>
                </button>

                <button
                  onClick={() => setTaskChoice("predictNumber")}
                  style={{
                    textAlign: "left",
                    borderRadius: 18,
                    border:
                      state.taskChoice === "predictNumber"
                        ? `2px solid var(--primary)`
                        : "1px solid rgba(255,255,255,0.14)",
                    background:
                      state.taskChoice === "predictNumber"
                        ? "rgba(99,102,241,0.15)"
                        : "rgba(255,255,255,0.06)",
                    padding: 16,
                    cursor: "pointer",
                    transition: "0.2s",
                  }}
                >
                  <p style={{ fontWeight: 900, color: "white" }}>
                    Predict a number
                  </p>
                  <p style={{ fontSize: 13, color: "var(--muted)", marginTop: 6 }}>
                    Example: Price, Salary, Score
                  </p>
                </button>

                {!canContinueStructured && (
                  <p className="text-xs" style={{ color: "var(--muted)" }}>
                    Select target + prediction type to continue.
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Table */}
          <div className="aidex-card" style={{ padding: 0, overflow: "hidden" }}>
            <div
              style={{
                padding: 18,
                borderBottom: "1px solid rgba(255,255,255,0.12)",
              }}
            >
              <p className="aidex-card-title">Preview Table</p>
              <p className="text-sm" style={{ color: "var(--muted)", marginTop: 4 }}>
                Showing first 10 rows from your CSV.
              </p>
            </div>

            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead style={{ background: "rgba(255,255,255,0.06)" }}>
                  <tr>
                    {structured.columns.map((col) => (
                      <th
                        key={col}
                        style={{
                          textAlign: "left",
                          padding: "12px 14px",
                          fontWeight: 800,
                          color: "white",
                          borderBottom: "1px solid rgba(255,255,255,0.12)",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {structuredPreviewRows.map((row, i) => (
                    <tr key={i}>
                      {structured.columns.map((col) => (
                        <td
                          key={col}
                          style={{
                            padding: "12px 14px",
                            borderBottom: "1px solid rgba(255,255,255,0.08)",
                            whiteSpace: "nowrap",
                            color: "rgba(255,255,255,0.88)",
                          }}
                        >
                          {row[col] === null || row[col] === undefined ? "" : String(row[col])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* ✅ Continue button AFTER TABLE (CENTERED) */}
          <div style={{ display: "flex", justifyContent: "center", marginTop: 12 }}>
            <button
              disabled={!canContinueStructured}
              onClick={() => router.push("/preprocessing")}
              className="aidex-btn-primary"
              style={{
                width: "100%",
                maxWidth: 420,
                padding: "14px 18px",
                borderRadius: 18,
                opacity: canContinueStructured ? 1 : 0.45,
                cursor: canContinueStructured ? "pointer" : "not-allowed",
              }}
            >
              Continue to Preprocessing
            </button>
          </div>

          {/* BIG bottom space ✅ */}
          <div style={{ height: 60 }} />
        </div>
      )}

      {/* =========================
          IMAGE OVERVIEW
      ========================= */}
      {state.datasetKind === "images" && relabeledImages.length > 0 && (
        <div className="space-y-8">
          <div className="aidex-card" style={{ padding: 18 }}>
            <p className="aidex-card-title">Image Dataset Summary</p>

            <div
              style={{
                marginTop: 16,
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                gap: 16,
              }}
            >
              {[
                ["Total images uploaded", imageStats.totalImages],
                ["Total classes", imageStats.totalClasses],
                ["Largest class count", imageStats.largestClassCount],
                ["Smallest class count", imageStats.smallestClassCount],
              ].map(([label, value]) => (
                <div
                  key={String(label)}
                  style={{
                    borderRadius: 16,
                    border: "1px solid rgba(255,255,255,0.14)",
                    background: "rgba(255,255,255,0.06)",
                    padding: 16,
                  }}
                >
                  <p style={{ fontSize: 12, color: "var(--muted)" }}>{label}</p>
                  <p style={{ fontSize: 20, fontWeight: 900, marginTop: 6 }}>{value}</p>
                </div>
              ))}
            </div>

            {imageStats.totalClasses > 1 && (
              <p className="text-sm" style={{ marginTop: 14, color: "var(--muted)" }}>
                Balance ratio: {imageStats.imbalanceRatio.toFixed(2)}x
              </p>
            )}
          </div>

          <div className="aidex-card" style={{ padding: 18 }}>
            <p className="aidex-card-title">Class Distribution</p>
            <div style={{ marginTop: 12, color: "var(--muted)", fontSize: 13 }}>
              {sortDistribution(imageClassDist).map(([label, count]) => (
                <p key={label} style={{ marginBottom: 8 }}>
                  <b style={{ color: "var(--text)" }}>{label}</b>: {count}
                </p>
              ))}
            </div>
          </div>

          <div className="aidex-card" style={{ padding: 18 }}>
            <p className="aidex-card-title">Preview</p>

            <div
              style={{
                marginTop: 16,
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
                gap: 14,
              }}
            >
              {imagePreview.map((img) => (
                <div
                  key={img.relativePath}
                  style={{
                    borderRadius: 16,
                    border: "1px solid rgba(255,255,255,0.14)",
                    background: "rgba(255,255,255,0.06)",
                    padding: 10,
                  }}
                >
                  <img
                    src={URL.createObjectURL((img as any).file)}
                    alt={(img as any).file.name}
                    style={{
                      width: "100%",
                      height: 100,
                      objectFit: "cover",
                      borderRadius: 14,
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
                    {(img as any).label} • {(img as any).file.name}
                  </p>
                </div>
              ))}
            </div>

            <p className="text-xs" style={{ marginTop: 12, color: "var(--muted)" }}>
              Showing 12 samples (balanced preview).
            </p>

            <div style={{ marginTop: 18, display: "flex", justifyContent: "center" }}>
              <button
                onClick={() => router.push("/preprocessing")}
                className="aidex-btn-primary"
                style={{ width: "100%", maxWidth: 420, padding: "14px 18px" }}
              >
                Continue to Preprocessing
              </button>
            </div>
          </div>

          <div style={{ height: 60 }} />
        </div>
      )}

      {/* =========================
          MODAL (VIEW DETAILS)
      ========================= */}
      {showExplanationModal && state.datasetKind === "structured" && structured && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 999,
            background: "rgba(0,0,0,0.70)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 18,
          }}
          onClick={() => setShowExplanationModal(false)}
        >
          <div
            style={{
              width: "100%",
              maxWidth: 980,
              maxHeight: "85vh",
              borderRadius: 22,
              overflow: "hidden",
              border: "1px solid rgba(255,255,255,0.12)",
              background: "#0b1020", // ✅ NOT transparent anymore
              boxShadow: "0 30px 80px rgba(0,0,0,0.65)",
              padding: 18,
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div>
                <h3 style={{ fontSize: 18, fontWeight: 900, color: "white" }}>
                  Detected Columns
                </h3>
                <p className="text-sm mt-1" style={{ color: "rgba(255,255,255,0.65)" }}>
                  Column types, missing values, and unique counts.
                </p>
              </div>

              <button
                onClick={() => setShowExplanationModal(false)}
                className="aidex-btn-outline"
              >
                Close
              </button>
            </div>

            <div
              style={{
                marginTop: 14,
                maxHeight: "62vh",
                overflowY: "auto",
                borderRadius: 16,
                border: "1px solid rgba(255,255,255,0.12)",
                padding: 12,
                background: "rgba(255,255,255,0.04)", // ✅ readable container
              }}
            >
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {columnAnalysis.map((c) => (
                  <div
                    key={c.name}
                    style={{
                      borderRadius: 16,
                      border: "1px solid rgba(255,255,255,0.12)",
                      background: "rgba(255,255,255,0.06)",
                      padding: 14,
                    }}
                  >
                    <div className="flex items-start justify-between gap-4 flex-wrap">
                      <div>
                        <p style={{ fontWeight: 900, color: "white" }}>{c.name}</p>
                        {c.exampleValues.length > 0 && (
                          <p
                            className="text-xs"
                            style={{ color: "rgba(255,255,255,0.65)", marginTop: 6 }}
                          >
                            Examples: {c.exampleValues.join(" | ")}
                          </p>
                        )}
                      </div>

                      <div style={{ textAlign: "right" }}>
                        <p className="text-sm" style={{ color: "rgba(255,255,255,0.65)" }}>
                          Type: <b style={{ color: "white" }}>{c.type}</b>
                        </p>
                        <p className="text-sm" style={{ color: "rgba(255,255,255,0.65)" }}>
                          Missing: <b style={{ color: "white" }}>{c.missing}</b>{" "}
                          • Unique: <b style={{ color: "white" }}>{c.unique}</b>
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <p className="text-xs" style={{ color: "rgba(255,255,255,0.65)", marginTop: 12 }}>
              Tip: These types are detected automatically and will be used later for preprocessing.
            </p>
          </div>
        </div>
      )}
    </div>
  </main>
);


}
