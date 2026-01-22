import os
import pandas as pd
import numpy as np
from scipy import stats
from PIL import Image
import hashlib

def run_engine(input_data, target_column=None, decisions=None):
    engine = DataQualityEngine(input_data, target_column)
    report = engine.analyze()

    if decisions:
        return engine.apply_decisions(decisions)

    return report

class DataQualityEngine:

    # ================= INIT =================
    def __init__(self, input_data, target_column=None):
        self.input_data = input_data
        self.target_column = target_column

        self.df_original = None
        self.df = None
        self.unstructured_subtype = None

        self.report = {
            "data_type": None,
            "validation": {},
            "issues_detected": [],
            "cleaning_questions": [],
            "cleaning_steps": [],
            "before_after": {},
            "final_schema": {},
            "dataset_ready": None
        }

    # ================= DATA TYPE =================
    def detect_data_type(self):
        if isinstance(self.input_data, pd.DataFrame):
            return "structured"

        if isinstance(self.input_data, str):
            path = os.path.abspath(self.input_data)

            if path.endswith((".csv", ".xlsx")):
                return "structured"

            if os.path.isdir(path):
                self.unstructured_subtype = "image"
                return "unstructured"

            if path.endswith((".txt", ".log", ".json")):
                self.unstructured_subtype = "text"
                return "unstructured"

        raise ValueError("Unsupported input data")

    # ================= ANALYZE =================
    def analyze(self):
        data_type = self.detect_data_type()
        self.report["data_type"] = data_type

        if data_type == "structured":
            return self.analyze_structured()

        if data_type == "unstructured":
            return self.analyze_unstructured()

    # ================= STRUCTURED ANALYSIS =================
    def analyze_structured(self):
        if isinstance(self.input_data, pd.DataFrame):
            self.df = self.input_data.copy()
        else:
            self.df = pd.read_csv(self.input_data)

        self.df_original = self.df.copy()

        # Missing values
        missing = self.df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                self.report["issues_detected"].append({
                    "type": "missing_values",
                    "column": col,
                    "count": int(count)
                })

                self.report["cleaning_questions"].append({
                    "id": f"fill_missing_{col}",
                    "message": f"Column '{col}' has {count} missing values. Fill using median?",
                    "action": "fill_missing",
                    "column": col
                })

        # Duplicates
        dup_count = int(self.df.duplicated().sum())
        if dup_count > 0:
            self.report["issues_detected"].append({
                "type": "duplicates",
                "count": dup_count
            })

            self.report["cleaning_questions"].append({
                "id": "remove_duplicates",
                "message": f"{dup_count} duplicate rows detected. Remove duplicates?",
                "action": "remove_duplicates"
            })

        # Outliers
        for col in self.df.select_dtypes(include=[np.number]).columns:
            self.report["issues_detected"].append({
                "type": "outliers",
                "column": col,
                "methods": ["IQR", "Z-score"]
            })

            self.report["cleaning_questions"].append({
                "id": f"remove_outliers_{col}",
                "message": f"Outliers detected in '{col}'. Remove using IQR + Z-score?",
                "action": "remove_outliers",
                "column": col
            })

        return self.report

    # ================= UNSTRUCTURED ANALYSIS =================
    def analyze_unstructured(self):
        if self.unstructured_subtype == "image":
            self.report["issues_detected"].append({
                "type": "image_quality",
                "issue": "corrupted or duplicate images"
            })

            self.report["cleaning_questions"].append({
                "id": "clean_images",
                "message": "Remove corrupted or duplicate images?",
                "action": "clean_images"
            })

        if self.unstructured_subtype == "text":
            self.report["issues_detected"].append({
                "type": "text_noise",
                "issue": "empty or noisy text"
            })

            self.report["cleaning_questions"].append({
                "id": "clean_text",
                "message": "Remove empty or noisy text?",
                "action": "clean_text"
            })

        return self.report

    # ================= APPLY DECISIONS =================
    def apply_decisions(self, decisions: dict):

        for q in self.report["cleaning_questions"]:
            if not decisions.get(q["id"]):
                continue

            action = q["action"]

            if action == "fill_missing":
                col = q["column"]
                self.df[col] = self.df[col].fillna(self.df[col].median())
                self.report["cleaning_steps"].append(
                    f"Filled missing values in '{col}'"
                )

            if action == "remove_duplicates":
                self.df = self.df.drop_duplicates()
                self.report["cleaning_steps"].append(
                    "Removed duplicate rows"
                )

            if action == "remove_outliers":
                col = q["column"]

                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1

                self.df = self.df[
                    (self.df[col] >= Q1 - 1.5 * IQR) &
                    (self.df[col] <= Q3 + 1.5 * IQR)
                ]

                z_scores = np.abs(stats.zscore(self.df[col]))
                self.df = self.df[z_scores < 3]

                self.report["cleaning_steps"].append(
                    f"Removed outliers from '{col}' using IQR + Z-score"
                )

            if action == "clean_images":
                self.report["cleaning_steps"].append(
                    "User approved image cleaning"
                )

            if action == "clean_text":
                self.report["cleaning_steps"].append(
                    "User approved text cleaning"
                )

        self.report["before_after"] = {
            "before_shape": self.df_original.shape if self.df_original is not None else None,
            "after_shape": self.df.shape if self.df is not None else None
        }

        if self.df is not None:
            self.report["final_schema"] = {
                "columns": list(self.df.columns),
                "dtypes": self.df.dtypes.astype(str).to_dict()
            }

        self.report["dataset_ready"] = self.df is not None and not self.df.empty

        return self.df, self.report
