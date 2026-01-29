"""
Data Quality Engine for AIDEX
Analyzes and cleans datasets
"""

import pandas as pd
import numpy as np


class DataQualityEngine:
    """
    Engine for analyzing and cleaning data quality issues
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the Data Quality Engine
        
        Args:
            df: Input DataFrame
        """
        self.df = df.copy()
        
    def analyze(self):
        """
        Analyze the dataset for quality issues
        
        Returns:
            Dictionary with quality report
        """
        report = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing_values": {},
            "duplicates": 0,
            "column_stats": {}
        }
        
        # Missing values
        for col in self.df.columns:
            missing = self.df[col].isnull().sum()
            if missing > 0:
                report["missing_values"][col] = {
                    "count": int(missing),
                    "percentage": float(missing / len(self.df) * 100)
                }
        
        # Duplicates
        report["duplicates"] = int(self.df.duplicated().sum())
        
        # Column statistics
        for col in self.df.columns:
            stats = {
                "unique_values": int(self.df[col].nunique()),
                "most_common": None
            }
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                stats["mean"] = float(self.df[col].mean()) if not self.df[col].isna().all() else None
                stats["std"] = float(self.df[col].std()) if not self.df[col].isna().all() else None
                stats["min"] = float(self.df[col].min()) if not self.df[col].isna().all() else None
                stats["max"] = float(self.df[col].max()) if not self.df[col].isna().all() else None
            else:
                if not self.df[col].empty:
                    most_common = self.df[col].mode()
                    if len(most_common) > 0:
                        stats["most_common"] = str(most_common[0])
            
            report["column_stats"][col] = stats
        
        return report
    
    def apply_decisions(self, decisions: dict):
        """
        Apply cleaning decisions to the dataset
        
        Args:
            decisions: Dictionary with cleaning decisions
            
        Returns:
            Tuple of (cleaned_df, report)
        """
        df_cleaned = self.df.copy()
        report = {
            "original_shape": self.df.shape,
            "actions": []
        }
        
        # Handle missing values
        if "fill_missing" in decisions:
            for col, method in decisions["fill_missing"].items():
                if col in df_cleaned.columns:
                    if method == "mean":
                        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                        report["actions"].append(f"Filled {col} with mean")
                    elif method == "median":
                        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                        report["actions"].append(f"Filled {col} with median")
                    elif method == "mode":
                        df_cleaned[col].fillna(df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else None, inplace=True)
                        report["actions"].append(f"Filled {col} with mode")
                    elif method == "drop":
                        df_cleaned = df_cleaned.dropna(subset=[col])
                        report["actions"].append(f"Dropped rows with missing {col}")
        
        # Remove duplicates
        if decisions.get("remove_duplicates", False):
            before = len(df_cleaned)
            df_cleaned = df_cleaned.drop_duplicates()
            after = len(df_cleaned)
            if before > after:
                report["actions"].append(f"Removed {before - after} duplicate rows")
        
        # Drop columns
        if "drop_columns" in decisions:
            cols_to_drop = [c for c in decisions["drop_columns"] if c in df_cleaned.columns]
            if cols_to_drop:
                df_cleaned = df_cleaned.drop(columns=cols_to_drop)
                report["actions"].append(f"Dropped columns: {', '.join(cols_to_drop)}")
        
        report["final_shape"] = df_cleaned.shape
        report["cleaned_df"] = df_cleaned
        
        return df_cleaned, report
