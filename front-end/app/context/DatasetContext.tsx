"use client";

import React, { createContext, useContext, useMemo, useState } from "react";

/* =========================
   TYPES
========================= */

export type DatasetKind = "none" | "structured" | "images" | "unsupported";

export type StructuredDataset = {
  fileName: string;
  rows: any[];
  columns: string[];
};

export type ImageItem = {
  file: File;
  label: string;
  relativePath: string;
};

export type TaskChoice = "predictCategory" | "predictNumber";

type DatasetState = {
  datasetKind: DatasetKind;
  errorMsg: string;

  structured: StructuredDataset | null;
  images: ImageItem[];

  targetColumn: string;
  taskChoice: TaskChoice | null;
};

type DatasetContextValue = {
  state: DatasetState;

  setDatasetKind: (v: DatasetKind) => void;
  setErrorMsg: (v: string) => void;

  setStructured: (v: StructuredDataset | null) => void;
  setImages: (v: ImageItem[]) => void;

  setTargetColumn: (v: string) => void;
  setTaskChoice: (v: TaskChoice | null) => void;

  resetAll: () => void;
};

const DatasetContext = createContext<DatasetContextValue | null>(null);

export function DatasetProvider({ children }: { children: React.ReactNode }) {
  const [datasetKind, setDatasetKind] = useState<DatasetKind>("none");
  const [errorMsg, setErrorMsg] = useState("");

  const [structured, setStructured] = useState<StructuredDataset | null>(null);
  const [images, setImages] = useState<ImageItem[]>([]);

  const [targetColumn, setTargetColumn] = useState("");
  const [taskChoice, setTaskChoice] = useState<TaskChoice | null>(null);

  const resetAll = () => {
    setDatasetKind("none");
    setErrorMsg("");

    setStructured(null);
    setImages([]);

    setTargetColumn("");
    setTaskChoice(null);

    // Optional: clear localStorage saves
    localStorage.removeItem("aidex_selected_target");
    localStorage.removeItem("aidex_task_choice");
  };

  const value = useMemo(
    () => ({
      state: {
        datasetKind,
        errorMsg,
        structured,
        images,
        targetColumn,
        taskChoice,
      },
      setDatasetKind,
      setErrorMsg,
      setStructured,
      setImages,
      setTargetColumn,
      setTaskChoice,
      resetAll,
    }),
    [datasetKind, errorMsg, structured, images, targetColumn, taskChoice]
  );

  return (
    <DatasetContext.Provider value={value}>{children}</DatasetContext.Provider>
  );
}

export function useDataset() {
  const ctx = useContext(DatasetContext);
  if (!ctx) {
    throw new Error("useDataset must be used inside <DatasetProvider />");
  }
  return ctx;
}
