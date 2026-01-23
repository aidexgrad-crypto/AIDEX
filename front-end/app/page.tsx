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
type ValidationIssue = {
  id: string;
  type: "missing" | "outlier" | "duplicate" | "invalid_type" | "image_issue";
  column?: string;
  description: string;
  count?: number;
};

type CleaningAction =
  | { id: string; type: "missing"; column: string; method: "mean" | "median" | "mode" | "drop" }
  | { id: string; type: "outlier"; column: string; method: "zscore_cap"; threshold: number }
  | { id: string; type: "duplicate"; method: "remove" }
  | { id: string; type: "image_invalid"; method: "remove" }
  | { id: string; type: "image_duplicate"; method: "remove" };


type CleaningLogRow = {
  step: string;
  action: string;
  explanation: string;
  before: string;
  after: string;
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
function isMissing(v: any) {
  return v === null || v === undefined || String(v).trim() === "";
}

function safeNumber(v: any) {
  const n = Number(v);
  return isNaN(n) ? null : n;
}

/* =========================
   VALIDATION (STRUCTURED)
========================= */
function validateStructured(
  rows: any[],
  columns: string[],
  analysis: ColumnAnalysis[]
): ValidationIssue[] {
  const issues: ValidationIssue[] = [];

  // missing values per column
  for (const c of analysis) {
    if (c.missing > 0) {
      issues.push({
        id: `missing_${c.name}`,
        type: "missing",
        column: c.name,
        description: `Missing values detected in "${c.name}"`,
        count: c.missing,
      });
    }
  }

  // duplicates (full row duplicates)
  const seen = new Set<string>();
  let dupCount = 0;
  for (const r of rows) {
    const key = JSON.stringify(r);
    if (seen.has(key)) dupCount++;
    else seen.add(key);
  }
  if (dupCount > 0) {
    issues.push({
      id: "duplicates",
      type: "duplicate",
      description: "Duplicate rows detected",
      count: dupCount,
    });
  }

  // outlier check (numeric columns, simple z-score flag)
  for (const c of analysis) {
    if (c.type !== "numeric") continue;

    const nums = rows
      .map((r) => safeNumber(r[c.name]))
      .filter((v): v is number => v !== null);

    if (nums.length < 10) continue;

    const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
    const variance =
      nums.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / nums.length;
    const std = Math.sqrt(variance);

    if (std === 0) continue;

    let outCount = 0;
    for (const x of nums) {
      const z = Math.abs((x - mean) / std);
      if (z > 3) outCount++;
    }

    if (outCount > 0) {
      issues.push({
        id: `outlier_${c.name}`,
        type: "outlier",
        column: c.name,
        description: `Potential outliers detected in "${c.name}" (z-score > 3)`,
        count: outCount,
      });
    }
  }

  return issues;
}

/* =========================
   APPLY CLEANING (STRUCTURED)
========================= */
function applyStructuredCleaning(
  rows: any[],
  columns: string[],
  analysis: ColumnAnalysis[],
  actions: CleaningAction[]
) {
  let newRows = [...rows];
  const logs: CleaningLogRow[] = [];

  // 1) remove duplicates
  const dupAction = actions.find((a) => a.type === "duplicate");
  if (dupAction && dupAction.method === "remove") {
    const beforeCount = newRows.length;
    const seen = new Set<string>();
    newRows = newRows.filter((r) => {
      const key = JSON.stringify(r);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    logs.push({
      step: "Duplicates",
      action: "Removed duplicate rows",
      explanation: "Duplicate rows were removed to avoid bias and repeated samples.",
      before: `${beforeCount} rows`,
      after: `${newRows.length} rows`,
    });
  }

  // 2) missing values per column
  const missingActions = actions.filter((a) => a.type === "missing") as Extract<
    CleaningAction,
    { type: "missing" }
  >[];

  for (const act of missingActions) {
    const col = act.column;
    const info = analysis.find((a) => a.name === col);
    if (!info) continue;

    const beforeMissing = newRows.filter((r) => isMissing(r[col])).length;

    if (beforeMissing === 0) continue;

    if (act.method === "drop") {
      const beforeCount = newRows.length;
      newRows = newRows.filter((r) => !isMissing(r[col]));
      logs.push({
        step: `Missing (${col})`,
        action: `Dropped rows with missing "${col}"`,
        explanation: "Rows with missing values were removed based on your choice.",
        before: `${beforeCount} rows`,
        after: `${newRows.length} rows`,
      });
      continue;
    }

    // Fill value based on method
    let fillValue: any = "";

    const nonMissingVals = newRows
      .map((r) => r[col])
      .filter((v) => !isMissing(v));

    if (info.type === "numeric") {
      const nums = nonMissingVals
        .map((v) => safeNumber(v))
        .filter((v): v is number => v !== null);

      if (nums.length === 0) continue;

      if (act.method === "mean") {
        fillValue = nums.reduce((a, b) => a + b, 0) / nums.length;
      } else if (act.method === "median") {
        const sorted = [...nums].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        fillValue =
          sorted.length % 2 === 0
            ? (sorted[mid - 1] + sorted[mid]) / 2
            : sorted[mid];
      } else {
        // mode fallback for numeric
        const freq: Record<string, number> = {};
        for (const v of nums) freq[String(v)] = (freq[String(v)] || 0) + 1;
        fillValue = Number(
          Object.entries(freq).sort((a, b) => b[1] - a[1])[0][0]
        );
      }
    } else {
      // categorical/text mode fill
      const freq: Record<string, number> = {};
      for (const v of nonMissingVals) {
        const key = String(v);
        freq[key] = (freq[key] || 0) + 1;
      }
      fillValue = Object.entries(freq).sort((a, b) => b[1] - a[1])[0][0];
    }

    newRows = newRows.map((r) => {
      if (!isMissing(r[col])) return r;
      return { ...r, [col]: fillValue };
    });

    const afterMissing = newRows.filter((r) => isMissing(r[col])).length;

    logs.push({
      step: `Missing (${col})`,
      action: `Filled missing values using ${act.method}`,
      explanation:
        act.method === "mean"
          ? "Mean is used for numeric data when values are stable."
          : act.method === "median"
          ? "Median is safer when the data has outliers."
          : act.method === "mode"
          ? "Mode is best for categorical fields (most frequent value)."
          : "Missing values were handled based on your choice.",
      before: `${beforeMissing} missing`,
      after: `${afterMissing} missing`,
    });
  }

  // 3) outliers (z-score cap)
  const outlierActions = actions.filter((a) => a.type === "outlier") as Extract<
    CleaningAction,
    { type: "outlier" }
  >[];

  for (const act of outlierActions) {
    const col = act.column;

    const nums = newRows
      .map((r) => safeNumber(r[col]))
      .filter((v): v is number => v !== null);

    if (nums.length < 10) continue;

    const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
    const variance =
      nums.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / nums.length;
    const std = Math.sqrt(variance);

    if (std === 0) continue;

    const beforeOut = nums.filter((x) => Math.abs((x - mean) / std) > act.threshold)
      .length;

    if (beforeOut === 0) continue;

    // cap values at threshold
    const upper = mean + act.threshold * std;
    const lower = mean - act.threshold * std;

    newRows = newRows.map((r) => {
      const v = safeNumber(r[col]);
      if (v === null) return r;
      if (v > upper) return { ...r, [col]: upper };
      if (v < lower) return { ...r, [col]: lower };
      return r;
    });

    const afterNums = newRows
      .map((r) => safeNumber(r[col]))
      .filter((v): v is number => v !== null);

    const afterOut = afterNums.filter(
      (x) => Math.abs((x - mean) / std) > act.threshold
    ).length;

    logs.push({
      step: `Outliers (${col})`,
      action: `Capped outliers using z-score threshold = ${act.threshold}`,
      explanation:
        "Outliers were capped instead of removed to keep dataset size stable.",
      before: `${beforeOut} outliers`,
      after: `${afterOut} outliers`,
    });
  }

  return { newRows, logs };
}

/* =========================
   VALIDATION + CLEANING (IMAGES)
========================= */
function validateImages(items: ImageItem[]) {
  const invalid = items.filter((x) => !x.file.type.startsWith("image/"));
  const issues: ValidationIssue[] = [];

  if (invalid.length > 0) {
    issues.push({
      id: "invalid_images",
      type: "image_issue",
      description: "Some files are not valid images",
      count: invalid.length,
    });
  }

  return issues;
}

async function hashFileSHA256(file: File) {
  const buf = await file.arrayBuffer();
  const hash = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(hash))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function applyImageCleaning(items: ImageItem[], actions: CleaningAction[]) {
  let newItems = [...items];
  const logs: CleaningLogRow[] = [];

  // 1) Remove invalid image files
  const invalidAct = actions.find((a) => a.type === "image_invalid");
  if (invalidAct && invalidAct.method === "remove") {
    const before = newItems.length;
    newItems = newItems.filter((x) => x.file.type.startsWith("image/"));

    logs.push({
      step: "Image Validation",
      action: "Removed invalid image files",
      explanation: "Non-image files were removed from the uploaded folder.",
      before: `${before} files`,
      after: `${newItems.length} files`,
    });
  }

  // 2) Remove duplicate images (by hash)
  const dupAct = actions.find((a) => a.type === "image_duplicate");
  if (dupAct && dupAct.method === "remove") {
    const before = newItems.length;

    const seen = new Set<string>();
    const filtered: ImageItem[] = [];

    for (const img of newItems) {
      if (!img.file.type.startsWith("image/")) continue;

      const hash = await hashFileSHA256(img.file);
      if (seen.has(hash)) continue;

      seen.add(hash);
      filtered.push(img);
    }

    newItems = filtered;

    logs.push({
      step: "Duplicates",
      action: "Removed duplicate images",
      explanation: "Duplicates were removed using SHA-256 file hashing.",
      before: `${before} images`,
      after: `${newItems.length} images`,
    });
  }

  return { newItems, logs };
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

    /* =========================
     CLEANING WORKFLOW (CHECKPOINTS)
  ========================= */

  const [validationIssues, setValidationIssues] = useState<ValidationIssue[]>([]);
  const [cleaningActions, setCleaningActions] = useState<CleaningAction[]>([]);
  const [cleaningLogs, setCleaningLogs] = useState<CleaningLogRow[]>([]);
  const [cleanedPreviewRows, setCleanedPreviewRows] = useState<any[] | null>(null);

  const [cleaningStage, setCleaningStage] = useState<
    "idle" | "validated" | "actions_selected" | "applied"
  >("idle");

  // outlier threshold (z-score cap default 3)
  const [zThreshold, setZThreshold] = useState<number>(3);
  
  const [cleaningStep, setCleaningStep] = useState<
  "validation" | "missing" | "outliers" | "duplicates" | "confirm" | "report"
>("validation");

const [currentMissingIndex, setCurrentMissingIndex] = useState(0);


  /* ---------- images ---------- */
  const [images, setImages] = useState<ImageItem[]>([]);
  const [folderLoading, setFolderLoading] = useState(false);
  const [labelDepth, setLabelDepth] = useState<number>(2);

  const [imageCleaningStep, setImageCleaningStep] = useState<
  "validation" | "duplicates" | "confirm" | "report"
>("validation");

const [removeDuplicateImages, setRemoveDuplicateImages] = useState(true);


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
      complete: (results: Papa.ParseResult<any>) => {
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

  const missingColumns = useMemo(() => {
  return columnAnalysis.filter((c) => c.missing > 0);
}, [columnAnalysis]);


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
   
  
  useEffect(() => {
    if (datasetKind === "structured" && structured) {
      const issues = validateStructured(
        structured.rows,
        structured.columns,
        columnAnalysis
      );
      setValidationIssues(issues);
      setCleaningStage("validated");
      setCleaningActions([]);
      setCleaningLogs([]);
      setCleanedPreviewRows(null);
      setCleaningStep("validation");
      setCurrentMissingIndex(0);

    }

    if (datasetKind === "images" && relabeledImages.length > 0) {
      const issues = validateImages(relabeledImages);
      setValidationIssues(issues);
      setCleaningStage("validated");
      setCleaningActions([]);
      setCleaningLogs([]);
      setCleanedPreviewRows(null);
    }
  }, [datasetKind, structured, columnAnalysis, relabeledImages]);

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

             {/* =========================
                ✅ CHECKPOINT 1 + 2 (STRUCTURED)
            ========================= */}
            <div className="mt-8 rounded-2xl border border-white/10 bg-white/5 p-6">
              <div className="flex items-start justify-between gap-4 flex-wrap">
                <div>
                  <p className="font-semibold text-lg">
                    ✅ Data Validation & Cleaning 
                  </p>
                  <p className="text-sm text-white/60 mt-1">
                    You decide each step before applying it.
                  </p>
                </div>

                <button
                  onClick={() => {
                    setCleaningActions([]);
                    setCleaningLogs([]);
                    setCleanedPreviewRows(null);
                    setCleaningStage("validated");
                  }}
                  className="rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-sm hover:bg-white/10 transition"
                >
                  Reset Cleaning
                </button>
              </div>

              {/* ✅ Checkpoint 1: Validation issues */}
              <div className="mt-5 rounded-2xl border border-white/10 bg-white/5 p-4">
                <p className="font-semibold">Checkpoint 1 — Validation Report</p>

                {validationIssues.length === 0 ? (
                  <p className="text-sm text-green-200 mt-2">
                    ✅ No issues found. Your dataset looks clean.
                  </p>
                ) : (
                  <div className="mt-2 space-y-2 text-sm text-white/70">
                    {validationIssues.map((issue) => (
                      <div
                        key={issue.id}
                        className="rounded-xl border border-white/10 bg-white/5 p-3"
                      >
                        <p className="font-semibold text-white/90">
                          {issue.description}
                        </p>
                        {issue.column && (
                          <p className="text-xs text-white/50 mt-1">
                            Column: {issue.column}
                          </p>
                        )}
                        {typeof issue.count === "number" && (
                          <p className="text-xs text-white/50 mt-1">
                            Count: {issue.count}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {/* ✅ USER FRIENDLY STEP-BY-STEP CLEANING */}
<div className="mt-5 rounded-2xl border border-white/10 bg-white/5 p-4">
  <div className="flex items-start justify-between gap-4 flex-wrap">
    <div>
      <p className="font-semibold text-lg">Cleaning Wizard</p>
      <p className="text-sm text-white/60 mt-1">
        We will ask you questions step-by-step before applying changes.
      </p>
    </div>

    {cleaningStep !== "validation" && (
      <button
        onClick={() => {
          if (cleaningStep === "missing") setCleaningStep("validation");
          else if (cleaningStep === "outliers") setCleaningStep("missing");
          else if (cleaningStep === "duplicates") setCleaningStep("outliers");
          else if (cleaningStep === "confirm") setCleaningStep("duplicates");
          else if (cleaningStep === "report") setCleaningStep("confirm");
        }}
        className="rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-sm hover:bg-white/10 transition"
      >
        ← Back
      </button>
    )}
  </div>

  {/* STEP 1: VALIDATION */}
  {cleaningStep === "validation" && (
    <div className="mt-4">
      <p className="font-semibold">Step 1 — Validation</p>
      <p className="text-sm text-white/60 mt-1">
        First, we check duplicates, missing values, and outliers.
      </p>

      <button
        onClick={() => setCleaningStep("missing")}
        className="mt-4 rounded-xl bg-white text-black px-5 py-2.5 text-sm font-semibold hover:bg-white/90 transition"
      >
        Next → Missing Values
      </button>
    </div>
  )}

  {/* STEP 2: MISSING VALUES (QUESTIONS PER COLUMN) */}
  {cleaningStep === "missing" && (
    <div className="mt-4">
      <p className="font-semibold">Step 2 — Missing Values</p>
      <p className="text-sm text-white/60 mt-1">
        We will handle missing values column-by-column.
      </p>

      {missingColumns.length === 0 ? (
        <>
          <p className="text-sm text-green-200 mt-3">
            ✅ No missing values detected.
          </p>
          <button
            onClick={() => setCleaningStep("outliers")}
            className="mt-4 rounded-xl bg-white text-black px-5 py-2.5 text-sm font-semibold hover:bg-white/90 transition"
          >
            Next → Outliers
          </button>
        </>
      ) : (
        <>
          <div className="mt-4 rounded-xl border border-white/10 bg-white/5 p-4">
            <p className="font-semibold text-lg">
              Column: {missingColumns[currentMissingIndex].name}
            </p>

            <p className="text-sm text-white/60 mt-1">
              Missing:{" "}
              <span className="text-white font-semibold">
                {missingColumns[currentMissingIndex].missing}
              </span>{" "}
              • Type:{" "}
              <span className="text-white font-semibold">
                {missingColumns[currentMissingIndex].type}
              </span>
            </p>

            <p className="text-sm text-white/70 mt-4 font-semibold">
              Question: How should we handle missing values here?
            </p>

            <select
              defaultValue={
                missingColumns[currentMissingIndex].type === "numeric"
                  ? "median"
                  : "mode"
              }
              onChange={(e) => {
                const col = missingColumns[currentMissingIndex].name;
                const method = e.target.value as
                  | "mean"
                  | "median"
                  | "mode"
                  | "drop";

                setCleaningActions((prev) => {
                  const removed = prev.filter(
                    (x) =>
                      !(
                        x.type === "missing" &&
                        (x as any).column === col
                      )
                  );

                  return [
                    ...removed,
                    { id: `missing_${col}`, type: "missing", column: col, method },
                  ];
                });
              }}
              className="mt-3 w-full max-w-sm rounded-xl bg-white/5 border border-white/10 px-4 py-3 text-sm text-white outline-none"
            >
              {missingColumns[currentMissingIndex].type === "numeric" && (
                <>
                  <option value="mean" className="text-black">Fill with Mean</option>
                  <option value="median" className="text-black">
                    Fill with Median (recommended)
                  </option>
                </>
              )}
              <option value="mode" className="text-black">Fill with Mode</option>
              <option value="drop" className="text-black">Drop rows</option>
            </select>

            <div className="mt-3 rounded-xl border border-white/10 bg-white/5 p-3 text-xs text-white/60">
              {missingColumns[currentMissingIndex].type === "numeric" ? (
                <>
                  ✅ <b>Median</b> is safer if the data has outliers.
                  <br />
                  ✅ <b>Mean</b> works when values are consistent.
                </>
              ) : (
                <>
                  ✅ <b>Mode</b> is best for categorical columns.
                  <br />
                  ✅ Dropping rows reduces dataset size.
                </>
              )}
            </div>
          </div>

          <div className="mt-4 flex gap-2 flex-wrap">
            <button
              onClick={() => setCurrentMissingIndex((p) => Math.max(0, p - 1))}
              className="rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-sm hover:bg-white/10 transition"
              disabled={currentMissingIndex === 0}
            >
              Previous
            </button>

            <button
              onClick={() => {
                if (currentMissingIndex < missingColumns.length - 1) {
                  setCurrentMissingIndex((p) => p + 1);
                } else {
                  setCleaningStep("outliers");
                }
              }}
              className="rounded-xl bg-white text-black px-5 py-2.5 text-sm font-semibold hover:bg-white/90 transition"
            >
              {currentMissingIndex < missingColumns.length - 1
                ? "Next Column →"
                : "Next → Outliers"}
            </button>
          </div>
        </>
      )}
    </div>
  )}
          


  {/* STEP 3: OUTLIERS */}
  {cleaningStep === "outliers" && (
    <div className="mt-4">
      <p className="font-semibold">Step 3 — Outliers</p>
      <p className="text-sm text-white/60 mt-1">
        Select numeric columns where we should cap outliers (Z-score).
      </p>

      <div className="mt-4 flex items-center gap-3 flex-wrap">
        <p className="text-sm text-white/70">Z Threshold:</p>
        <input
          type="number"
          value={zThreshold}
          step={0.1}
          min={2}
          max={5}
          onChange={(e) => setZThreshold(Number(e.target.value))}
          className="w-24 rounded-xl bg-white/5 border border-white/10 px-3 py-2 text-sm text-white outline-none"
        />
        <span className="text-xs text-white/50">Recommended: 3.0</span>
      </div>

      <div className="mt-3 space-y-2">
        {columnAnalysis.filter((c) => c.type === "numeric").map((c) => {
          const checked = cleaningActions.some(
            (a) => a.type === "outlier" && (a as any).column === c.name
          );

          return (
            <label
              key={c.name}
              className="flex items-center gap-2 text-sm text-white/70"
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={(e) => {
                  if (e.target.checked) {
                    setCleaningActions((prev) => [
                      ...prev,
                      {
                        id: `outlier_${c.name}`,
                        type: "outlier",
                        column: c.name,
                        method: "zscore_cap",
                        threshold: zThreshold,
                      },
                    ]);
                  } else {
                    setCleaningActions((prev) =>
                      prev.filter(
                        (x) =>
                          !(
                            x.type === "outlier" &&
                            (x as any).column === c.name
                          )
                      )
                    );
                  }
                }}
              />
              Cap outliers in "{c.name}"
            </label>
          );
        })}
      </div>

      <button
        onClick={() => setCleaningStep("duplicates")}
        className="mt-4 rounded-xl bg-white text-black px-5 py-2.5 text-sm font-semibold hover:bg-white/90 transition"
      >
        Next → Duplicates
      </button>
    </div>
  )}

  {/* STEP 4: DUPLICATES */}
  {cleaningStep === "duplicates" && (
    <div className="mt-4">
      <p className="font-semibold">Step 4 — Duplicates</p>

      <label className="flex items-center gap-2 text-sm text-white/70 mt-3">
        <input
          type="checkbox"
          checked={cleaningActions.some((a) => a.type === "duplicate")}
          onChange={(e) => {
            if (e.target.checked) {
              setCleaningActions((prev) => [
                ...prev,
                { id: "dup_remove", type: "duplicate", method: "remove" },
              ]);
            } else {
              setCleaningActions((prev) =>
                prev.filter((a) => a.type !== "duplicate")
              );
            }
          }}
        />
        Remove duplicate rows (recommended)
      </label>

      <button
        onClick={() => setCleaningStep("confirm")}
        className="mt-4 rounded-xl bg-white text-black px-5 py-2.5 text-sm font-semibold hover:bg-white/90 transition"
      >
        Next → Confirm
      </button>
    </div>
  )}

  {/* STEP 5: CONFIRM + APPLY */}
  {cleaningStep === "confirm" && (
    <div className="mt-4">
      <p className="font-semibold">Step 5 — Confirm & Apply</p>

      <button
        onClick={() => {
          if (!structured) return;

          const { newRows, logs } = applyStructuredCleaning(
            structured.rows,
            structured.columns,
            columnAnalysis,
            cleaningActions.map((a) =>
              a.type === "outlier" ? { ...a, threshold: zThreshold } : a
            )
          );

          setCleaningLogs(logs);
          setCleanedPreviewRows(newRows.slice(0, 10));
          setCleaningStep("report");
        }}
        className="mt-4 rounded-xl bg-white text-black px-5 py-2.5 text-sm font-semibold hover:bg-white/90 transition"
      >
        ✅ Apply Cleaning
      </button>
    </div>
  )}

  {/* STEP 6: REPORT */}
  {cleaningStep === "report" && (
    <div className="mt-4">
      <p className="font-semibold">Step 6 — Report</p>

      {cleaningLogs.length === 0 ? (
        <p className="text-sm text-green-200 mt-3">
          ✅ No cleaning actions applied.
        </p>
      ) : (
        <div className="mt-4 overflow-auto rounded-2xl border border-white/10">
          <table className="min-w-full text-sm">
            <thead className="bg-white/5 sticky top-0">
              <tr>
                <th className="px-4 py-3 text-left font-semibold text-white/80">
                  Step
                </th>
                <th className="px-4 py-3 text-left font-semibold text-white/80">
                  Action
                </th>
                <th className="px-4 py-3 text-left font-semibold text-white/80">
                  Before
                </th>
                <th className="px-4 py-3 text-left font-semibold text-white/80">
                  After
                </th>
              </tr>
            </thead>
            <tbody>
              {cleaningLogs.map((log, i) => (
                <tr key={i} className="border-t border-white/10">
                  <td className="px-4 py-3 text-white/80">{log.step}</td>
                  <td className="px-4 py-3 text-white/80">{log.action}</td>
                  <td className="px-4 py-3 text-white/80">{log.before}</td>
                  <td className="px-4 py-3 text-white/80">{log.after}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )}
</div>

                {/* APPLY BUTTON */}
                <div className="mt-5 flex items-center gap-3 flex-wrap">
                  <button
                    onClick={() => {
                      if (!structured) return;

                      setCleaningStage("actions_selected");

                      const { newRows, logs } = applyStructuredCleaning(
                        structured.rows,
                        structured.columns,
                        columnAnalysis,
                        cleaningActions.map((a) =>
                          a.type === "outlier"
                            ? { ...a, threshold: zThreshold }
                            : a
                        )
                      );

                      setCleaningLogs(logs);
                      setCleanedPreviewRows(newRows.slice(0, 10));
                      setCleaningStage("applied");
                      setCleaningStep("report");
                    }}
                    className="rounded-xl bg-white text-black px-5 py-2.5 text-sm font-semibold hover:bg-white/90 transition"
                  >
                    FINAL REPORTS
                  </button>

                  <p className="text-xs text-white/50">
                    Nothing will happen until you click Apply.
                  </p>
                </div>
               </div>

                 </div>
        )}

        {/* =========================
    IMAGE PREVIEW + CLEANING WIZARD
========================= */}
{datasetKind === "images" && relabeledImages.length > 0 && (
  <div className="rounded-3xl border border-white/10 bg-white/5 p-6 sm:p-8">
    <h2 className="text-xl sm:text-2xl font-semibold">Dataset Preview</h2>
    <p className="text-white/60 mt-1 text-sm">
      Labels are inferred from folder names.
    </p>

    {/* Distribution */}
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

    {/* Preview */}
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

    {/* =========================
        ✅ IMAGE CLEANING WIZARD
    ========================= */}
    <div className="mt-8 rounded-2xl border border-white/10 bg-white/5 p-6">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <p className="font-semibold text-lg">✅ Image Cleaning Wizard</p>
          <p className="text-sm text-white/60 mt-1">
            Step-by-step cleaning (like structured datasets).
          </p>
        </div>

        {imageCleaningStep !== "validation" && (
          <button
            onClick={() => {
              if (imageCleaningStep === "duplicates") setImageCleaningStep("validation");
              else if (imageCleaningStep === "confirm") setImageCleaningStep("duplicates");
              else if (imageCleaningStep === "report") setImageCleaningStep("confirm");
            }}
            className="rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-sm hover:bg-white/10 transition"
          >
            ← Back
          </button>
        )}
      </div>

      {/* STEP 1: VALIDATION */}
      {imageCleaningStep === "validation" && (
        <div className="mt-4">
          <p className="font-semibold">Step 1 — Validation</p>
          <p className="text-sm text-white/60 mt-1">
            We check invalid image files and show issues.
          </p>

          <div className="mt-4 rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="font-semibold">Validation Report</p>

            {validationIssues.length === 0 ? (
              <p className="text-sm text-green-200 mt-2">
                ✅ No image issues found.
              </p>
            ) : (
              <div className="mt-2 space-y-2 text-sm text-white/70">
                {validationIssues.map((issue) => (
                  <div
                    key={issue.id}
                    className="rounded-xl border border-white/10 bg-white/5 p-3"
                  >
                    <p className="font-semibold text-white/90">
                      {issue.description}
                    </p>
                    {typeof issue.count === "number" && (
                      <p className="text-xs text-white/50 mt-1">
                        Count: {issue.count}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          <button
            onClick={() => setImageCleaningStep("duplicates")}
            className="mt-4 rounded-xl bg-white text-black px-5 py-2.5 text-sm font-semibold hover:bg-white/90 transition"
          >
            Next → Duplicates
          </button>
        </div>
      )}

      {/* STEP 2: DUPLICATES */}
      {imageCleaningStep === "duplicates" && (
        <div className="mt-4">
          <p className="font-semibold">Step 2 — Duplicates</p>
          <p className="text-sm text-white/60 mt-1">
            Do you want us to remove duplicate images (same content)?
          </p>

          <label className="flex items-center gap-2 text-sm text-white/70 mt-3">
            <input
              type="checkbox"
              checked={removeDuplicateImages}
              onChange={(e) => setRemoveDuplicateImages(e.target.checked)}
            />
            Remove duplicate images (recommended)
          </label>

          <div className="mt-4 rounded-xl border border-white/10 bg-white/5 p-3 text-xs text-white/60">
            ✅ Duplicate detection is done using SHA-256 hashing (exact duplicates).
          </div>

          <button
            onClick={() => setImageCleaningStep("confirm")}
            className="mt-4 rounded-xl bg-white text-black px-5 py-2.5 text-sm font-semibold hover:bg-white/90 transition"
          >
            Next → Confirm
          </button>
        </div>
      )}

      {/* STEP 3: CONFIRM */}
      {imageCleaningStep === "confirm" && (
        <div className="mt-4">
          <p className="font-semibold">Step 3 — Confirm & Apply</p>
          <p className="text-sm text-white/60 mt-1">
            Review and apply your selected actions.
          </p>

          <div className="mt-4 rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-white/70 space-y-2">
            <p className="font-semibold text-white/90">Selected Actions:</p>
            <p>✅ Remove invalid image files</p>
            <p>{removeDuplicateImages ? "✅ Remove duplicate images" : "❌ Do not remove duplicates"}</p>
          </div>

          <button
            onClick={async () => {
              const actionsToApply: CleaningAction[] = [
                { id: "img_invalid_remove", type: "image_invalid", method: "remove" },
              ];

              if (removeDuplicateImages) {
                actionsToApply.push({
                  id: "img_dup_remove",
                  type: "image_duplicate",
                  method: "remove",
                });
              }

              const { newItems, logs } = await applyImageCleaning(
                relabeledImages,
                actionsToApply
              );

              setImages(newItems);
              setCleaningLogs(logs);
              setCleaningStage("applied");
              setImageCleaningStep("report");
            }}
            className="mt-4 rounded-xl bg-white text-black px-5 py-2.5 text-sm font-semibold hover:bg-white/90 transition"
          >
            ✅ Apply Image Cleaning
          </button>

          <p className="text-xs text-white/50 mt-3">
            Nothing will happen until you click Apply.
          </p>
        </div>
      )}

      {/* STEP 4: REPORT */}
      {imageCleaningStep === "report" && (
        <div className="mt-4">
          <p className="font-semibold">Step 4 — Report</p>

          {cleaningLogs.length === 0 ? (
            <p className="text-sm text-green-200 mt-3">
              ✅ No cleaning actions were applied.
            </p>
          ) : (
            <div className="mt-4 overflow-auto rounded-2xl border border-white/10">
              <table className="min-w-full text-sm">
                <thead className="bg-white/5 sticky top-0">
                  <tr>
                    <th className="px-4 py-3 text-left font-semibold text-white/80">
                      Step
                    </th>
                    <th className="px-4 py-3 text-left font-semibold text-white/80">
                      Action
                    </th>
                    <th className="px-4 py-3 text-left font-semibold text-white/80">
                      Before
                    </th>
                    <th className="px-4 py-3 text-left font-semibold text-white/80">
                      After
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {cleaningLogs.map((log, i) => (
                    <tr key={i} className="border-t border-white/10">
                      <td className="px-4 py-3 text-white/80">{log.step}</td>
                      <td className="px-4 py-3 text-white/80">{log.action}</td>
                      <td className="px-4 py-3 text-white/80">{log.before}</td>
                      <td className="px-4 py-3 text-white/80">{log.after}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
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
