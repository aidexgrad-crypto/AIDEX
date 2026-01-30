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
  quality: number = 0.85  // Reduced from 0.92 for faster processing
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

  // Use faster image smoothing for preprocessing
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'medium';  // 'medium' is faster than 'high'
  
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
  
  // AutoML states
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResults, setTrainingResults] = useState<any>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [detectedTaskType, setDetectedTaskType] = useState<string>("classification");

  // Prediction states
  const [predictFile, setPredictFile] = useState<File | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResults, setPredictionResults] = useState<any>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  // Image prediction states
  const [predictImages, setPredictImages] = useState<File[]>([]);
  const [isPredictingImages, setIsPredictingImages] = useState(false);
  const [imagePredictionResults, setImagePredictionResults] = useState<any>(null);
  const [imagePredictionError, setImagePredictionError] = useState<string | null>(null);

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

    // Process images in parallel batches for speed
    const batchSize = 10; // Process 10 images at a time
    const totalBatches = Math.ceil(state.images.length / batchSize);
    
    console.log(`🖼️ Processing ${state.images.length} images in ${totalBatches} batches...`);

    for (let batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
      const start = batchIndex * batchSize;
      const end = Math.min(start + batchSize, state.images.length);
      const batch = state.images.slice(start, end);

      // Process batch in parallel
      const batchPromises = batch.map(async (item) => {
        try {
          if (!item.file || !item.file.type?.startsWith("image/")) {
            return { status: 'skipped' };
          }

          const newFile = await resizeAndNormalizeImage(item.file, targetSize, 0.85);
          if (!newFile) {
            return { status: 'corrupted' };
          }

          return {
            status: 'success',
            data: {
              ...item,
              file: newFile,
              label: item.label ?? "unknown",
              relativePath: item.relativePath ?? newFile.name,
            }
          };
        } catch {
          return { status: 'corrupted' };
        }
      });

      const batchResults = await Promise.all(batchPromises);

      // Collect results
      batchResults.forEach((result) => {
        if (result.status === 'success') {
          resizedCount++;
          cleaned.push(result.data);
        } else if (result.status === 'skipped') {
          skippedNonImages++;
        } else if (result.status === 'corrupted') {
          removedCorrupted++;
        }
      });

      // Log progress
      const progress = Math.round(((batchIndex + 1) / totalBatches) * 100);
      console.log(`⏳ Progress: ${progress}% (${end}/${state.images.length} images)`);
    }

    const afterCount = cleaned.length;

    console.log(`✅ Image preprocessing complete: ${afterCount} images ready`);

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

    // ✅ STRUCTURED - Use backend cleaning
    if (state.datasetKind === "structured") {
      if (!canRunStructured || !structured) {
        setIsRunning(false);
        return;
      }

      try {
        // Call backend cleaning API
        const response = await fetch("/api/data/clean", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            data: structured.rows,
            target_column: state.targetColumn,
            protected_columns: protectedColumns
          }),
        });

        const result = await response.json();

        if (result.status === "success") {
          // Log performance optimization info
          if (result.total_rows && result.preview_rows) {
            console.log(`✅ Cleaning optimized: Showing ${result.preview_rows} of ${result.total_rows} rows (full data saved to ${result.cleaned_file_path})`);
          }
          
          // Update structured data with cleaned data
          setStructured({
            fileName: structured.fileName,
            rows: result.cleaned_data,
            columns: Object.keys(result.cleaned_data[0] || {}),
          });

          setSummary({
            beforeRows: result.summary.before_rows,
            beforeCols: result.summary.before_cols,
            afterRows: result.summary.after_rows,
            afterCols: result.summary.after_cols,
            removedDuplicates: result.summary.removed_duplicates,
            filledMissing: result.summary.filled_missing,
            handledOutliers: 0,
            scaledNumeric: false,
            encodedCategorical: false,
            droppedColumns: result.summary.dropped_columns,
            notes: result.summary.notes
          });

          setDone(true);
        } else {
          console.error("Cleaning failed:", result.error);
        }
      } catch (error) {
        console.error("Backend cleaning error:", error);
      }

      setIsRunning(false);
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
  };

  const markImportant = (col: string) => {
    setProtectedColumns((prev) => {
      if (prev.includes(col)) return prev;
      return [...prev, col];
    });
  };

  /* =========================
     AUTOML TRAINING
  ========================= */
  const detectTaskType = (targetCol: string, rows: any[]): string => {
    if (!rows || rows.length === 0) return "classification";
    
    // Get all target values
    const targetValues = rows.map(row => row[targetCol]).filter(v => v !== null && v !== undefined);
    if (targetValues.length === 0) return "classification";
    
    // Count unique values
    const uniqueValues = new Set(targetValues);
    const uniqueCount = uniqueValues.size;
    const totalCount = targetValues.length;
    
    // Check if values are numeric
    const allNumeric = targetValues.every(v => typeof v === 'number' || !isNaN(Number(v)));
    
    // Decision logic:
    // 1. If too many unique values (> 20% of total), likely regression
    // 2. If all numeric and high cardinality, likely regression
    // 3. If numeric with large ranges (> 1000), likely regression
    if (allNumeric) {
      const numericValues = targetValues.map(v => Number(v));
      const min = Math.min(...numericValues);
      const max = Math.max(...numericValues);
      const range = max - min;
      
      // If large range (e.g., prices, salaries), it's regression
      if (range > 1000 || uniqueCount > totalCount * 0.2) {
        return "regression";
      }
    }
    
    // Default to classification if few unique values
    if (uniqueCount <= 20) {
      return "classification";
    }
    
    // High cardinality suggests regression
    return "regression";
  };

  const handleTrainAutoML = async () => {
    if (!state.targetColumn) {
      setTrainingError("Please select a target column first");
      return;
    }

    if (!done) {
      setTrainingError("Please run data preprocessing first.");
      return;
    }

    setIsTraining(true);
    setTrainingError(null);
    setTrainingResults(null);

    try {
      // Auto-detect task type based on target column
      const detectedTaskType = detectTaskType(state.targetColumn, structured?.rows || []);
      console.log(`Auto-detected task type: ${detectedTaskType}`);
      setDetectedTaskType(detectedTaskType);
      
      const response = await fetch("/api/automl/train", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          target_column: state.targetColumn,
          task_type: detectedTaskType,
          test_size: 0.2,
          scaling_method: "standard",
          selection_priority: "balanced",
          project_name: `automl_${Date.now()}`
        }),
      });

      const data = await response.json();

      if (data.error) {
        setTrainingError(data.error);
      } else if (data.status === "success") {
        setTrainingResults(data);
      } else {
        setTrainingError("Training failed with unknown error");
      }
    } catch (error) {
      console.error("AutoML training error:", error);
      setTrainingError("Failed to communicate with backend");
    } finally {
      setIsTraining(false);
    }
  };

  /* =========================
     PREDICT ON NEW DATA
  ========================= */
  const handlePredictFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setPredictFile(file);
      setPredictionError(null);
    }
  };

  const handlePredict = async () => {
    if (!predictFile) {
      setPredictionError("Please select a CSV file first");
      return;
    }

    if (!trainingResults || !trainingResults.project_id) {
      setPredictionError("No trained model found. Please train a model first.");
      return;
    }

    setIsPredicting(true);
    setPredictionError(null);
    setPredictionResults(null);

    try {
      const formData = new FormData();
      formData.append("file", predictFile);
      formData.append("project_name", trainingResults.project_id);

      const response = await fetch("/api/automl/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.error) {
        setPredictionError(data.error);
      } else if (data.status === "success") {
        setPredictionResults(data);
      } else {
        setPredictionError("Prediction failed with unknown error");
      }
    } catch (error) {
      console.error("Prediction error:", error);
      setPredictionError("Failed to communicate with backend");
    } finally {
      setIsPredicting(false);
    }
  };

  /* =========================
     PREDICT ON NEW IMAGES
  ========================= */
  const handlePredictImagesUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      const imageFiles = Array.from(files).filter(f => f.type.startsWith("image/"));
      setPredictImages(imageFiles);
      setImagePredictionError(null);
    }
  };

  const handlePredictImages = async () => {
    if (predictImages.length === 0) {
      setImagePredictionError("Please select images first");
      return;
    }

    if (!trainingResults || !(trainingResults.project_id || trainingResults.project_name)) {
      setImagePredictionError("No trained model found. Please train a model first.");
      return;
    }

    setIsPredictingImages(true);
    setImagePredictionError(null);
    setImagePredictionResults(null);

    try {
      // Upload images first
      const formData = new FormData();
      predictImages.forEach((img) => {
        formData.append("files", img);
      });

      const uploadResponse = await fetch("/api/automl/upload-images", {
        method: "POST",
        body: formData,
      });

      const uploadData = await uploadResponse.json();

      if (uploadData.status !== "success") {
        const errorMsg = uploadData.error || "Failed to upload images";
        throw new Error(errorMsg);
      }

      // Predict using uploaded paths - use project_name for images, project_id for tabular
      const projectIdentifier = trainingResults.project_name || trainingResults.project_id;
      
      const response = await fetch("/api/automl/predict-images", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image_paths: uploadData.paths,
          project_name: projectIdentifier
        }),
      });

      const data = await response.json();

      if (data.error) {
        setImagePredictionError(data.error);
      } else if (data.status === "success") {
        setImagePredictionResults(data);
      } else {
        setImagePredictionError("Prediction failed with unknown error");
      }
    } catch (error) {
      console.error("Image prediction error:", error);
      setImagePredictionError(`Failed to predict: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsPredictingImages(false);
    }
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

            {/* AutoML Training Section */}
            <div style={{ marginTop: 24, paddingTop: 24, borderTop: "1px solid var(--border)" }}>
              <p style={{ fontWeight: 800, fontSize: 16 }}>Next Step: Train Models with AutoML</p>
              <p className="text-sm" style={{ color: "var(--muted)", marginTop: 6 }}>
                Your data is cleaned and ready. Start AutoML to automatically train and select the best model.
              </p>

              <div style={{ marginTop: 16 }}>
                <button
                  className="aidex-btn-primary"
                  onClick={handleTrainAutoML}
                  disabled={isTraining || !state.targetColumn}
                  style={{
                    opacity: (isTraining || !state.targetColumn) ? 0.6 : 1,
                    cursor: (isTraining || !state.targetColumn) ? "not-allowed" : "pointer",
                  }}
                >
                  {isTraining ? "Training Models..." : "Start AutoML Training"}
                </button>
              </div>

              {isTraining && (
                <div style={{ marginTop: 16 }}>
                  <p className="text-sm" style={{ color: "var(--muted)" }}>
                    Training multiple models on your cleaned dataset. This may take a few minutes...
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
                </div>
              )}

              {trainingError && (
                <div
                  style={{
                    marginTop: 16,
                    padding: 14,
                    borderRadius: 14,
                    border: "1px solid #ef4444",
                    background: "rgba(239, 68, 68, 0.1)",
                  }}
                >
                  <p style={{ fontSize: 14, fontWeight: 600, color: "#ef4444" }}>
                    Training Error
                  </p>
                  <p className="text-sm" style={{ color: "#dc2626", marginTop: 6 }}>
                    {trainingError}
                  </p>
                </div>
              )}

              {trainingResults && (
                <div
                  style={{
                    marginTop: 16,
                    padding: 16,
                    borderRadius: 14,
                    border: "1px solid #10b981",
                    background: "rgba(16, 185, 129, 0.05)",
                  }}
                >
                  <p style={{ fontSize: 16, fontWeight: 800, color: "#10b981" }}>
                    ✓ Training Completed Successfully!
                  </p>

                  <div style={{ marginTop: 16 }}>
                    <p style={{ fontWeight: 700, fontSize: 14 }}>
                      Best Model: {trainingResults.best_model}
                    </p>

                    <div
                      style={{
                        marginTop: 12,
                        display: "grid",
                        gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
                        gap: 12,
                      }}
                    >
                      {detectedTaskType === "classification" ? (
                        <>
                          <div
                            style={{
                              borderRadius: 10,
                              border: "1px solid var(--border)",
                              background: "rgba(15, 23, 42, 0.03)",
                              padding: 12,
                            }}
                          >
                            <p style={{ fontSize: 11, color: "var(--muted)" }}>Test Accuracy</p>
                            <p style={{ fontSize: 18, fontWeight: 800 }}>
                              {(trainingResults.test_scores.accuracy * 100).toFixed(2)}%
                            </p>
                          </div>

                          <div
                            style={{
                              borderRadius: 10,
                              border: "1px solid var(--border)",
                              background: "rgba(15, 23, 42, 0.03)",
                              padding: 12,
                            }}
                          >
                            <p style={{ fontSize: 11, color: "var(--muted)" }}>F1 Score</p>
                            <p style={{ fontSize: 18, fontWeight: 800 }}>
                              {(trainingResults.test_scores.f1 * 100).toFixed(2)}%
                            </p>
                          </div>

                          <div
                            style={{
                              borderRadius: 10,
                              border: "1px solid var(--border)",
                              background: "rgba(15, 23, 42, 0.03)",
                              padding: 12,
                            }}
                          >
                            <p style={{ fontSize: 11, color: "var(--muted)" }}>Precision</p>
                            <p style={{ fontSize: 18, fontWeight: 800 }}>
                              {(trainingResults.test_scores.precision * 100).toFixed(2)}%
                            </p>
                          </div>

                          <div
                            style={{
                              borderRadius: 10,
                              border: "1px solid var(--border)",
                              background: "rgba(15, 23, 42, 0.03)",
                              padding: 12,
                            }}
                          >
                            <p style={{ fontSize: 11, color: "var(--muted)" }}>Recall</p>
                            <p style={{ fontSize: 18, fontWeight: 800 }}>
                              {(trainingResults.test_scores.recall * 100).toFixed(2)}%
                            </p>
                          </div>
                        </>
                      ) : (
                        <>
                          <div
                            style={{
                              borderRadius: 10,
                              border: "1px solid var(--border)",
                              background: "rgba(15, 23, 42, 0.03)",
                              padding: 12,
                            }}
                          >
                            <p style={{ fontSize: 11, color: "var(--muted)" }}>R² Score</p>
                            <p style={{ fontSize: 18, fontWeight: 800 }}>
                              {trainingResults.test_scores.r2?.toFixed(4) || "N/A"}
                            </p>
                          </div>

                          <div
                            style={{
                              borderRadius: 10,
                              border: "1px solid var(--border)",
                              background: "rgba(15, 23, 42, 0.03)",
                              padding: 12,
                            }}
                          >
                            <p style={{ fontSize: 11, color: "var(--muted)" }}>RMSE</p>
                            <p style={{ fontSize: 18, fontWeight: 800 }}>
                              {trainingResults.test_scores.rmse?.toFixed(2) || "N/A"}
                            </p>
                          </div>

                          <div
                            style={{
                              borderRadius: 10,
                              border: "1px solid var(--border)",
                              background: "rgba(15, 23, 42, 0.03)",
                              padding: 12,
                            }}
                          >
                            <p style={{ fontSize: 11, color: "var(--muted)" }}>MAE</p>
                            <p style={{ fontSize: 18, fontWeight: 800 }}>
                              {trainingResults.test_scores.mae?.toFixed(2) || "N/A"}
                            </p>
                          </div>

                          <div
                            style={{
                              borderRadius: 10,
                              border: "1px solid var(--border)",
                              background: "rgba(15, 23, 42, 0.03)",
                              padding: 12,
                            }}
                          >
                            <p style={{ fontSize: 11, color: "var(--muted)" }}>MSE</p>
                            <p style={{ fontSize: 18, fontWeight: 800 }}>
                              {trainingResults.test_scores.mse?.toFixed(2) || "N/A"}
                            </p>
                          </div>
                        </>
                      )}
                    </div>

                    <p className="text-xs" style={{ color: "var(--muted)", marginTop: 12 }}>
                      Target: {trainingResults.target_column} | Dataset: {trainingResults.dataset_shape[0]} rows × {trainingResults.dataset_shape[1]} columns
                      {trainingResults.training_samples && (
                        <> | Training: {trainingResults.training_samples} samples | Test: {trainingResults.test_samples} samples | Features: {trainingResults.num_features}</>
                      )}
                    </p>

                    {trainingResults.warnings && trainingResults.warnings.length > 0 && (
                      <div
                        style={{
                          marginTop: 12,
                          padding: 12,
                          borderRadius: 10,
                          border: "1px solid #f59e0b",
                          background: "rgba(245, 158, 11, 0.1)",
                        }}
                      >
                        <p style={{ fontSize: 12, fontWeight: 600, color: "#f59e0b" }}>
                          ⚠️ Data Leakage Warning
                        </p>
                        <p className="text-xs" style={{ color: "#d97706", marginTop: 4 }}>
                          The following columns may cause data leakage: {trainingResults.warnings.join(", ")}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* ALL MODELS PERFORMANCE TABLE */}
              {trainingResults && trainingResults.all_models_test && (
                <div
                  style={{
                    marginTop: 16,
                    padding: 16,
                    borderRadius: 14,
                    border: "1px solid var(--border)",
                    background: "rgba(15, 23, 42, 0.03)",
                  }}
                >
                  <p style={{ fontSize: 16, fontWeight: 800, marginBottom: 12 }}>
                    All Models Performance
                  </p>

                  <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                      <thead>
                        <tr style={{ borderBottom: "1px solid var(--border)" }}>
                          <th style={{ padding: "10px 12px", textAlign: "left", fontWeight: 700 }}>Model</th>
                          {detectedTaskType === "classification" ? (
                            <>
                              <th style={{ padding: "10px 12px", textAlign: "right", fontWeight: 700 }}>Accuracy</th>
                              <th style={{ padding: "10px 12px", textAlign: "right", fontWeight: 700 }}>F1 Score</th>
                              <th style={{ padding: "10px 12px", textAlign: "right", fontWeight: 700 }}>Precision</th>
                              <th style={{ padding: "10px 12px", textAlign: "right", fontWeight: 700 }}>Recall</th>
                            </>
                          ) : (
                            <>
                              <th style={{ padding: "10px 12px", textAlign: "right", fontWeight: 700 }}>R² Score</th>
                              <th style={{ padding: "10px 12px", textAlign: "right", fontWeight: 700 }}>RMSE</th>
                              <th style={{ padding: "10px 12px", textAlign: "right", fontWeight: 700 }}>MAE</th>
                              <th style={{ padding: "10px 12px", textAlign: "right", fontWeight: 700 }}>MSE</th>
                            </>
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        {trainingResults.all_models_test
                          .sort((a: any, b: any) => detectedTaskType === "classification" ? b.f1 - a.f1 : b.r2 - a.r2)
                          .map((model: any, idx: number) => (
                            <tr
                              key={model.model_name}
                              style={{
                                borderBottom: idx < trainingResults.all_models_test.length - 1 ? "1px solid rgba(255,255,255,0.05)" : "none",
                                background: model.model_name === trainingResults.best_model ? "rgba(16, 185, 129, 0.08)" : "transparent"
                              }}
                            >
                              <td style={{ padding: "10px 12px", fontWeight: model.model_name === trainingResults.best_model ? 700 : 400 }}>
                                {model.model_name === trainingResults.best_model ? "★ " : ""}{model.model_name}
                              </td>
                              {detectedTaskType === "classification" ? (
                                <>
                                  <td style={{ padding: "10px 12px", textAlign: "right" }}>
                                    {(model.accuracy * 100).toFixed(2)}%
                                  </td>
                                  <td style={{ padding: "10px 12px", textAlign: "right" }}>
                                    {(model.f1 * 100).toFixed(2)}%
                                  </td>
                                  <td style={{ padding: "10px 12px", textAlign: "right" }}>
                                    {(model.precision * 100).toFixed(2)}%
                                  </td>
                                  <td style={{ padding: "10px 12px", textAlign: "right" }}>
                                    {(model.recall * 100).toFixed(2)}%
                                  </td>
                                </>
                              ) : (
                                <>
                                  <td style={{ padding: "10px 12px", textAlign: "right" }}>
                                    {model.r2?.toFixed(4) || "N/A"}
                                  </td>
                                  <td style={{ padding: "10px 12px", textAlign: "right" }}>
                                    {model.rmse?.toFixed(2) || "N/A"}
                                  </td>
                                  <td style={{ padding: "10px 12px", textAlign: "right" }}>
                                    {model.mae?.toFixed(2) || "N/A"}
                                  </td>
                                  <td style={{ padding: "10px 12px", textAlign: "right" }}>
                                    {model.mse?.toFixed(2) || "N/A"}
                                  </td>
                                </>
                              )}
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* PREDICT ON NEW DATA SECTION */}
              {trainingResults && trainingResults.status === "success" && (
                <div
                  style={{
                    marginTop: 16,
                    padding: 16,
                    borderRadius: 14,
                    border: "1px solid #3b82f6",
                    background: "rgba(59, 130, 246, 0.05)",
                  }}
                >
                  <p style={{ fontSize: 16, fontWeight: 800, marginBottom: 8, color: "#3b82f6" }}>
                    🔮 Predict on New Data
                  </p>

                  <p className="text-sm" style={{ color: "var(--muted)", marginBottom: 12 }}>
                    Upload a CSV file with the same features (without the target column) to get predictions using the trained {trainingResults.best_model} model.
                  </p>

                  <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
                    <input
                      type="file"
                      accept=".csv"
                      onChange={handlePredictFile}
                      disabled={isPredicting}
                      style={{
                        padding: "8px 12px",
                        borderRadius: 8,
                        border: "1px solid var(--border)",
                        background: "rgba(15, 23, 42, 0.5)",
                        fontSize: 13,
                        cursor: isPredicting ? "not-allowed" : "pointer",
                        opacity: isPredicting ? 0.6 : 1,
                      }}
                    />

                    <button
                      onClick={handlePredict}
                      disabled={!predictFile || isPredicting}
                      className="aidex-btn-primary"
                      style={{
                        opacity: (!predictFile || isPredicting) ? 0.6 : 1,
                        cursor: (!predictFile || isPredicting) ? "not-allowed" : "pointer",
                      }}
                    >
                      {isPredicting ? "Predicting..." : "Get Predictions"}
                    </button>

                    {predictFile && (
                      <span style={{ fontSize: 12, color: "var(--muted)" }}>
                        Selected: {predictFile.name}
                      </span>
                    )}
                  </div>

                  {predictionError && (
                    <div
                      style={{
                        marginTop: 12,
                        padding: 12,
                        borderRadius: 10,
                        border: "1px solid #ef4444",
                        background: "rgba(239, 68, 68, 0.1)",
                      }}
                    >
                      <p style={{ fontSize: 13, fontWeight: 600, color: "#ef4444" }}>
                        Error
                      </p>
                      <p className="text-sm" style={{ color: "#dc2626", marginTop: 4 }}>
                        {predictionError}
                      </p>
                    </div>
                  )}

                  {predictionResults && (
                    <div
                      style={{
                        marginTop: 16,
                        padding: 14,
                        borderRadius: 12,
                        border: "1px solid #10b981",
                        background: "rgba(16, 185, 129, 0.08)",
                      }}
                    >
                      <p style={{ fontSize: 15, fontWeight: 700, color: "#10b981", marginBottom: 12 }}>
                        ✓ Predictions Complete!
                      </p>

                      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12, marginBottom: 16 }}>
                        <div
                          style={{
                            padding: 12,
                            borderRadius: 10,
                            border: "1px solid var(--border)",
                            background: "rgba(15, 23, 42, 0.3)",
                          }}
                        >
                          <p style={{ fontSize: 11, color: "var(--muted)" }}>Model Used</p>
                          <p style={{ fontSize: 14, fontWeight: 700 }}>
                            {predictionResults.model_name}
                          </p>
                        </div>

                        <div
                          style={{
                            padding: 12,
                            borderRadius: 10,
                            border: "1px solid var(--border)",
                            background: "rgba(15, 23, 42, 0.3)",
                          }}
                        >
                          <p style={{ fontSize: 11, color: "var(--muted)" }}>Samples Predicted</p>
                          <p style={{ fontSize: 14, fontWeight: 700 }}>
                            {predictionResults.num_samples}
                          </p>
                        </div>
                      </div>

                      <p style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>
                        Prediction Distribution:
                      </p>
                      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 12 }}>
                        {Object.entries(predictionResults.prediction_distribution || {}).map(([label, count]: [string, any]) => (
                          <div
                            key={label}
                            style={{
                              padding: "6px 12px",
                              borderRadius: 8,
                              border: "1px solid var(--border)",
                              background: "rgba(59, 130, 246, 0.1)",
                            }}
                          >
                            <span style={{ fontSize: 13, fontWeight: 600 }}>
                              {label}:
                            </span>
                            <span style={{ fontSize: 13, marginLeft: 6 }}>
                              {count}
                            </span>
                          </div>
                        ))}
                      </div>

                      <p style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>
                        All Predictions ({predictionResults.predictions.length} samples):
                      </p>

                      <div
                        style={{
                          maxHeight: 300,
                          overflowY: "auto",
                          borderRadius: 10,
                          border: "1px solid var(--border)",
                          background: "rgba(15, 23, 42, 0.3)",
                        }}
                      >
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                          <thead style={{ position: "sticky", top: 0, background: "rgba(15, 23, 42, 0.95)", zIndex: 1 }}>
                            <tr style={{ borderBottom: "1px solid var(--border)" }}>
                              <th style={{ padding: "10px 12px", textAlign: "left", fontWeight: 700 }}>Sample #</th>
                              <th style={{ padding: "10px 12px", textAlign: "left", fontWeight: 700 }}>Prediction</th>
                            </tr>
                          </thead>
                          <tbody>
                            {predictionResults.predictions.map((pred: any, idx: number) => (
                              <tr
                                key={idx}
                                style={{
                                  borderBottom: idx < predictionResults.predictions.length - 1 ? "1px solid rgba(255,255,255,0.05)" : "none",
                                }}
                              >
                                <td style={{ padding: "8px 12px" }}>{idx + 1}</td>
                                <td style={{ padding: "8px 12px", fontWeight: 600 }}>
                                  <span
                                    style={{
                                      padding: "4px 10px",
                                      borderRadius: 6,
                                      background: pred === 1 ? "rgba(16, 185, 129, 0.2)" : "rgba(239, 68, 68, 0.2)",
                                      color: pred === 1 ? "#10b981" : "#ef4444",
                                      fontSize: 12,
                                      fontWeight: 700,
                                    }}
                                  >
                                    {pred}
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
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

            {/* IMAGE AUTOML TRAINING SECTION */}
            <div style={{ marginTop: 24, paddingTop: 24, borderTop: "1px solid var(--border)" }}>
              <p style={{ fontWeight: 800, fontSize: 16 }}>Next Step: Train Image Classification Models</p>
              <p className="text-sm" style={{ color: "var(--muted)", marginTop: 6 }}>
                Your images are cleaned and ready. Start AutoML to automatically train CNN models (ResNet, MobileNet) and select the best one.
              </p>

              <div style={{ marginTop: 16 }}>
                <button
                  className="aidex-btn-primary"
                  onClick={async () => {
                    setIsTraining(true);
                    setTrainingError(null);
                    setTrainingResults(null);

                    try {
                      // First, upload all image files to the backend
                      const formData = new FormData();
                      state.images.forEach((img) => {
                        formData.append("files", img.file);
                      });

                      const uploadResponse = await fetch("/api/automl/upload-images", {
                        method: "POST",
                        body: formData,
                      });

                      const uploadData = await uploadResponse.json();

                      if (uploadData.status !== "success") {
                        const errorMsg = uploadData.error || "Failed to upload images";
                        throw new Error(errorMsg);
                      }

                      // Prepare image data with server paths and labels
                      const imageData = state.images.map((img, idx) => ({
                        path: uploadData.paths[idx],
                        label: img.label,
                        file_path: uploadData.paths[idx]
                      }));

                      // Now train the models
                      const response = await fetch("/api/automl/train-images", {
                        method: "POST",
                        headers: {
                          "Content-Type": "application/json",
                        },
                        body: JSON.stringify({
                          image_data: imageData,
                          epochs: 5,  // Reduced from 10 to 5 for faster training
                          batch_size: 32,
                          learning_rate: 0.001,
                          test_size: 0.2,
                          early_stopping_patience: 2,  // Stop after 2 epochs without improvement
                          model_names: ["mobilenet_v2"],
                          project_name: `image_${Date.now()}`
                        }),
                      });

                      const data = await response.json();

                      if (data.error) {
                        setTrainingError(data.error);
                      } else if (data.status === "success") {
                        setTrainingResults(data);
                      } else {
                        setTrainingError("Training failed with unknown error");
                      }
                    } catch (error) {
                      console.error("Image training error:", error);
                      // Check if it's a timeout error (training might still be running)
                      const errorMsg = error instanceof Error ? error.message : String(error);
                      if (errorMsg.includes("fetch failed") || errorMsg.includes("timeout")) {
                        setTrainingError(`Training is taking longer than expected. The model is still training in the background. Please check the backend terminal for progress, or try refreshing the page in a few minutes to see if training completed.`);
                      } else {
                        setTrainingError(`Failed to train models: ${errorMsg}`);
                      }
                    } finally {
                      setIsTraining(false);
                    }
                  }}
                  disabled={isTraining}
                  style={{
                    opacity: isTraining ? 0.6 : 1,
                    cursor: isTraining ? "not-allowed" : "pointer",
                  }}
                >
                  {isTraining ? "Training Image Models..." : "Start Image AutoML Training"}
                </button>
              </div>

              {trainingError && (
                <div
                  style={{
                    marginTop: 12,
                    padding: 12,
                    borderRadius: 10,
                    border: "1px solid #ef4444",
                    background: "rgba(239, 68, 68, 0.1)",
                  }}
                >
                  <p style={{ fontSize: 12, fontWeight: 600, color: "#ef4444" }}>
                    ❌ Training Error
                  </p>
                  <p className="text-xs" style={{ color: "#dc2626", marginTop: 4 }}>
                    {trainingError}
                  </p>
                </div>
              )}

              {trainingResults && trainingResults.status === "success" && (
                <div
                  style={{
                    marginTop: 16,
                    padding: 16,
                    borderRadius: 14,
                    border: "1px solid #10b981",
                    background: "rgba(16, 185, 129, 0.1)",
                  }}
                >
                  <p style={{ fontSize: 16, fontWeight: 800, color: "#10b981" }}>
                    ✓ Image Models Trained Successfully!
                  </p>
                  <p className="text-sm" style={{ color: "var(--muted)", marginTop: 8 }}>
                    Best Model: {trainingResults.best_model}
                  </p>

                  <div style={{ marginTop: 16 }}>
                    <p style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>
                      Class Distribution:
                    </p>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                      {trainingResults.class_names?.map((className: string) => (
                        <div
                          key={className}
                          style={{
                            padding: "6px 12px",
                            borderRadius: 8,
                            border: "1px solid var(--border)",
                            background: "rgba(59, 130, 246, 0.1)",
                          }}
                        >
                          <span style={{ fontSize: 13, fontWeight: 600 }}>
                            {className}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {trainingResults.test_results && trainingResults.test_results.length > 0 && (
                    <div style={{ marginTop: 16 }}>
                      <p style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>
                        Model Performance:
                      </p>
                      <div style={{ overflowX: "auto" }}>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                          <thead>
                            <tr style={{ borderBottom: "1px solid var(--border)" }}>
                              <th style={{ padding: "10px 12px", textAlign: "left", fontWeight: 700 }}>Model</th>
                              <th style={{ padding: "10px 12px", textAlign: "right", fontWeight: 700 }}>Accuracy</th>
                              <th style={{ padding: "10px 12px", textAlign: "right", fontWeight: 700 }}>F1 Score</th>
                              <th style={{ padding: "10px 12px", textAlign: "right", fontWeight: 700 }}>Precision</th>
                              <th style={{ padding: "10px 12px", textAlign: "right", fontWeight: 700 }}>Recall</th>
                            </tr>
                          </thead>
                          <tbody>
                            {trainingResults.test_results.map((model: any, idx: number) => (
                              <tr
                                key={model.model_name}
                                style={{
                                  borderBottom: idx < trainingResults.test_results.length - 1 ? "1px solid rgba(255,255,255,0.05)" : "none",
                                  background: model.model_name === trainingResults.best_model ? "rgba(16, 185, 129, 0.08)" : "transparent"
                                }}
                              >
                                <td style={{ padding: "10px 12px", fontWeight: model.model_name === trainingResults.best_model ? 700 : 400 }}>
                                  {model.model_name === trainingResults.best_model ? "★ " : ""}{model.model_name}
                                </td>
                                <td style={{ padding: "10px 12px", textAlign: "right" }}>
                                  {(model.accuracy * 100).toFixed(2)}%
                                </td>
                                <td style={{ padding: "10px 12px", textAlign: "right" }}>
                                  {(model.f1 * 100).toFixed(2)}%
                                </td>
                                <td style={{ padding: "10px 12px", textAlign: "right" }}>
                                  {(model.precision * 100).toFixed(2)}%
                                </td>
                                <td style={{ padding: "10px 12px", textAlign: "right" }}>
                                  {(model.recall * 100).toFixed(2)}%
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  <p className="text-xs" style={{ color: "var(--muted)", marginTop: 12 }}>
                    Training: {trainingResults.train_size} images | Test: {trainingResults.test_size} images | Classes: {trainingResults.num_classes}
                  </p>
                </div>
              )}

              {/* PREDICT ON NEW IMAGES SECTION */}
              {trainingResults && trainingResults.status === "success" && (
                <div
                  style={{
                    marginTop: 16,
                    padding: 16,
                    borderRadius: 14,
                    border: "1px solid #3b82f6",
                    background: "rgba(59, 130, 246, 0.05)",
                  }}
                >
                  <p style={{ fontSize: 16, fontWeight: 800, marginBottom: 8, color: "#3b82f6" }}>
                    🔮 Predict on New Images
                  </p>

                  <p className="text-sm" style={{ color: "var(--muted)", marginBottom: 12 }}>
                    Upload images (without labels) to get predictions using the trained {trainingResults.best_model} model.
                  </p>

                  <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
                    <input
                      type="file"
                      accept="image/*"
                      multiple
                      onChange={handlePredictImagesUpload}
                      disabled={isPredictingImages}
                      style={{
                        padding: "8px 12px",
                        borderRadius: 8,
                        border: "1px solid var(--border)",
                        background: "rgba(15, 23, 42, 0.5)",
                        fontSize: 13,
                        cursor: isPredictingImages ? "not-allowed" : "pointer",
                        opacity: isPredictingImages ? 0.6 : 1,
                      }}
                    />

                    <button
                      onClick={handlePredictImages}
                      disabled={predictImages.length === 0 || isPredictingImages}
                      className="aidex-btn-primary"
                      style={{
                        opacity: (predictImages.length === 0 || isPredictingImages) ? 0.6 : 1,
                        cursor: (predictImages.length === 0 || isPredictingImages) ? "not-allowed" : "pointer",
                      }}
                    >
                      {isPredictingImages ? "Predicting..." : "Get Predictions"}
                    </button>

                    {predictImages.length > 0 && (
                      <span style={{ fontSize: 12, color: "var(--muted)" }}>
                        Selected: {predictImages.length} image(s)
                      </span>
                    )}
                  </div>

                  {imagePredictionError && (
                    <div
                      style={{
                        marginTop: 12,
                        padding: 12,
                        borderRadius: 10,
                        border: "1px solid #ef4444",
                        background: "rgba(239, 68, 68, 0.1)",
                      }}
                    >
                      <p style={{ fontSize: 13, fontWeight: 600, color: "#ef4444" }}>
                        Error
                      </p>
                      <p className="text-sm" style={{ color: "#dc2626", marginTop: 4 }}>
                        {imagePredictionError}
                      </p>
                    </div>
                  )}

                  {imagePredictionResults && (
                    <div
                      style={{
                        marginTop: 16,
                        padding: 14,
                        borderRadius: 12,
                        border: "1px solid #10b981",
                        background: "rgba(16, 185, 129, 0.08)",
                      }}
                    >
                      <p style={{ fontSize: 15, fontWeight: 700, color: "#10b981", marginBottom: 12 }}>
                        ✓ Predictions Complete!
                      </p>

                      <p className="text-sm" style={{ color: "var(--muted)", marginBottom: 8 }}>
                        Model: {imagePredictionResults.model_used} | Classes: {imagePredictionResults.class_names?.join(", ")}
                      </p>

                      <div style={{ 
                        maxHeight: 400, 
                        overflowY: "auto",
                        border: "1px solid var(--border)",
                        borderRadius: 8,
                        marginTop: 12
                      }}>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                          <thead style={{ position: "sticky", top: 0, background: "var(--background)", zIndex: 1 }}>
                            <tr style={{ borderBottom: "2px solid var(--border)" }}>
                              <th style={{ padding: "10px 12px", textAlign: "left", fontWeight: 700 }}>Image #</th>
                              <th style={{ padding: "10px 12px", textAlign: "left", fontWeight: 700 }}>Predicted Class</th>
                              <th style={{ padding: "10px 12px", textAlign: "right", fontWeight: 700 }}>Confidence</th>
                            </tr>
                          </thead>
                          <tbody>
                            {imagePredictionResults.results?.map((pred: any, idx: number) => (
                              <tr
                                key={idx}
                                style={{
                                  borderBottom: idx < imagePredictionResults.results.length - 1 ? "1px solid rgba(255,255,255,0.05)" : "none",
                                }}
                              >
                                <td style={{ padding: "10px 12px" }}>Image {idx + 1}</td>
                                <td style={{ padding: "10px 12px", fontWeight: 600, color: "#3b82f6" }}>
                                  {pred.predicted_label || pred.predicted_class || "Unknown"}
                                </td>
                                <td style={{ padding: "10px 12px", textAlign: "right" }}>
                                  {(pred.confidence * 100).toFixed(2)}%
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>

                      <p className="text-xs" style={{ color: "var(--muted)", marginTop: 8 }}>
                        Total predictions: {imagePredictionResults.results?.length || 0}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
