from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import shutil
import sys
import joblib
import torch
from typing import List

# Add AutoML to path
automl_path = os.path.join(os.path.dirname(__file__), '..', 'AutoML')
sys.path.insert(0, automl_path)

from Data_Pre_Processing.Data_quality_engine import DataQualityEngine
from aidex_pipeline import run_aidex
from image_pipeline import run_image_pipeline, ImagePipeline

app = FastAPI()

# ===================== CORS =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== STORAGE =====================
UPLOAD_DIR = "temp_uploads"
CLEANED_DIR = "cleaned_datasets"
MODEL_DIR = "saved_models"
IMAGE_UPLOAD_DIR = "temp_uploads/images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_UPLOAD_DIR, exist_ok=True)

LAST_FILE_PATH = None
CLEANED_FILE_PATH = None
LAST_TRAINED_PIPELINE = None
LAST_PROJECT_NAME = None
LAST_IMAGE_PIPELINE = None

# ===================== ANALYZE =====================
@app.post("/data-quality/analyze")
async def analyze_data(file: UploadFile = File(...)):
    global LAST_FILE_PATH

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    LAST_FILE_PATH = file_path

    df = pd.read_csv(file_path)

    engine = DataQualityEngine(df)
    report = engine.analyze()

    return report

# ===================== APPLY CLEANING =====================
@app.post("/data-quality/apply")
async def apply_cleaning(payload: dict):
    global LAST_FILE_PATH, CLEANED_FILE_PATH

    if LAST_FILE_PATH is None:
        return {"error": "No dataset analyzed yet"}

    decisions = payload.get("decisions", {})

    df = pd.read_csv(LAST_FILE_PATH)

    engine = DataQualityEngine(df)
    clean_df, report = engine.apply_decisions(decisions)

    # Save cleaned data for AutoML
    cleaned_filename = f"cleaned_{os.path.basename(LAST_FILE_PATH)}"
    CLEANED_FILE_PATH = os.path.join(CLEANED_DIR, cleaned_filename)
    clean_df.to_csv(CLEANED_FILE_PATH, index=False)
    
    report["cleaned_file_path"] = CLEANED_FILE_PATH
    report["cleaned_shape"] = clean_df.shape

    return report


# ===================== CLEAN DATA ENDPOINT =====================
class CleanDataRequest(BaseModel):
    data: list
    target_column: str = None
    protected_columns: list = []


@app.post("/data/clean")
async def clean_data(request: CleanDataRequest):
    """
    Clean data with automatic preprocessing
    """
    global CLEANED_FILE_PATH
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        if df.empty:
            return {"error": "Dataset is empty"}
        
        # IMPORTANT: Store original target column before any processing
        target_original = None
        if request.target_column and request.target_column in df.columns:
            target_original = df[request.target_column].copy()
            print(f"\n[DEBUG] Original target unique values: {target_original.unique()}")
            print(f"[DEBUG] Original target distribution:\n{target_original.value_counts()}")
        
        before_shape = df.shape
        before_rows = len(df)
        before_cols = len(df.columns)
        
        summary = {
            "before_rows": before_rows,
            "before_cols": before_cols,
            "removed_duplicates": 0,
            "filled_missing": 0,
            "dropped_columns": [],
            "notes": []
        }
        
        protected = request.protected_columns
        if request.target_column:
            protected.append(request.target_column)
        
        # 1. Remove constant columns
        cols_to_drop = []
        for col in df.columns:
            if col in protected:
                continue
            unique_values = df[col].dropna().nunique()
            if unique_values <= 1:
                cols_to_drop.append(col)
                summary["dropped_columns"].append({"name": col, "reason": "Constant column"})
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            summary["notes"].append(f"Removed {len(cols_to_drop)} constant columns")
        
        # 2. Remove duplicates (optimized - only check if dataset is reasonable size)
        dup_count = 0
        if len(df) < 100000:  # Only check for duplicates on datasets < 100k rows
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                df = df.drop_duplicates()
                summary["removed_duplicates"] = int(dup_count)
                summary["notes"].append(f"Removed {dup_count} duplicate rows")
        else:
            summary["notes"].append("Skipped duplicate check (large dataset - use manual check if needed)")
        
        # 3. Drop columns with >40% missing
        missing_threshold = 0.4
        cols_to_drop = []
        for col in df.columns:
            if col in protected:
                continue
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > missing_threshold:
                cols_to_drop.append(col)
                summary["dropped_columns"].append({
                    "name": col,
                    "reason": f"Too many missing values ({missing_pct*100:.1f}%)"
                })
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            summary["notes"].append(f"Dropped {len(cols_to_drop)} columns with >40% missing values")
        
        # 4. Fill missing values (optimized - vectorized operations)
        filled_count = 0
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns
        
        # Fill numeric columns with median
        for col in numeric_cols:
            missing = df[col].isnull().sum()
            if missing > 0:
                df[col] = df[col].fillna(df[col].median())
                filled_count += missing
        
        # Fill categorical columns with mode (first occurrence)
        for col in categorical_cols:
            missing = df[col].isnull().sum()
            if missing > 0:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                filled_count += missing
        
        summary["filled_missing"] = int(filled_count)
        if filled_count > 0:
            summary["notes"].append(f"Filled {filled_count} missing values")
        
        # 5. Handle outliers - DISABLED for speed (can be slow on large datasets)
        # Enable by uncommenting if needed
        # outlier_count = 0
        # for col in numeric_cols:
        #     Q1 = df[col].quantile(0.25)
        #     Q3 = df[col].quantile(0.75)
        #     IQR = Q3 - Q1
        #     lower = Q1 - 1.5 * IQR
        #     upper = Q3 + 1.5 * IQR
        #     outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        #     if outliers > 0:
        #         df[col] = df[col].clip(lower, upper)
        #         outlier_count += outliers
        # if outlier_count > 0:
        #     summary["notes"].append(f"Capped {outlier_count} outlier values")
        
        summary["after_rows"] = len(df)
        summary["after_cols"] = len(df.columns)
        
        # CRITICAL: Restore original target column if it was modified
        if target_original is not None and request.target_column in df.columns:
            df[request.target_column] = target_original
            print(f"\n[DEBUG] Restored target column")
            print(f"[DEBUG] Final target unique values: {df[request.target_column].unique()}")
            print(f"[DEBUG] Final target distribution:\n{df[request.target_column].value_counts()}")
        
        # Save cleaned data
        import time
        cleaned_filename = f"cleaned_{int(time.time())}.csv"
        CLEANED_FILE_PATH = os.path.join(CLEANED_DIR, cleaned_filename)
        df.to_csv(CLEANED_FILE_PATH, index=False)
        
        print(f"\n[INFO] Cleaned data saved to: {CLEANED_FILE_PATH}")
        
        # Return cleaned data and summary (optimized for large datasets)
        # Only return first 1000 rows for preview, full data is saved in CSV
        preview_size = min(1000, len(df))
        return {
            "status": "success",
            "cleaned_data": df.head(preview_size).to_dict('records'),
            "summary": summary,
            "cleaned_file_path": CLEANED_FILE_PATH,
            "total_rows": len(df),
            "preview_rows": preview_size
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


# ===================== AUTOML TRAINING =====================
class AutoMLRequest(BaseModel):
    target_column: str
    task_type: str = "classification"
    test_size: float = 0.2
    scaling_method: str = "standard"
    selection_priority: str = "balanced"
    project_name: str = "automl_project"


@app.post("/automl/train")
async def train_automl(request: AutoMLRequest):
    global CLEANED_FILE_PATH

    try:
        # Load cleaned data from file
        if not CLEANED_FILE_PATH or not os.path.exists(CLEANED_FILE_PATH):
            return {"error": "No cleaned dataset available. Please run data cleaning first."}
        
        df = pd.read_csv(CLEANED_FILE_PATH)
        
        # Ensure we have data
        if df.empty:
            return {"error": "Dataset is empty."}
        
        # Verify target column exists
        if request.target_column not in df.columns:
            return {
                "error": f"Target column '{request.target_column}' not found in cleaned dataset.",
                "available_columns": list(df.columns)
            }

        print(f"\n{'='*80}")
        print(f"STARTING AUTOML TRAINING")
        print(f"{'='*80}")
        print(f"Dataset shape: {df.shape}")
        print(f"Target column: {request.target_column}")
        print(f"Task type: {request.task_type}")
        print(f"\nColumn names: {df.columns.tolist()}")
        print(f"\nFirst few rows of data:")
        print(df.head())
        print(f"\nTarget column info:")
        print(f"  Data type: {df[request.target_column].dtype}")
        print(f"  Unique values: {df[request.target_column].unique()}")
        print(f"\nTarget distribution:")
        print(df[request.target_column].value_counts())
        print(f"  Class balance: {df[request.target_column].value_counts(normalize=True) * 100}")
        
        # Check for potential data leakage - look for columns similar to target
        leakage_candidates = []
        target_lower = request.target_column.lower()
        for col in df.columns:
            if col != request.target_column:
                col_lower = col.lower()
                # Check if column name is too similar to target
                if target_lower in col_lower or col_lower in target_lower:
                    # Check correlation/similarity
                    if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[request.target_column]):
                        corr = df[col].corr(df[request.target_column].astype(float) if not pd.api.types.is_numeric_dtype(df[request.target_column]) else df[request.target_column])
                        if abs(corr) > 0.95:
                            leakage_candidates.append(f"{col} (correlation: {corr:.3f})")
        
        if leakage_candidates:
            print(f"\n⚠️  WARNING: Potential data leakage detected!")
            print(f"Suspicious columns: {', '.join(leakage_candidates)}")
            print(f"Consider removing these columns before training.\n")
        
        # Run AIDEX pipeline
        pipeline = run_aidex(
            data_path=df,
            target_column=request.target_column,
            task_type=request.task_type,
            test_size=request.test_size,
            scaling_method=request.scaling_method,
            selection_priority=request.selection_priority,
            project_id=request.project_name
        )

        # Extract results
        best_model = pipeline.best_model_name
        cv_results = pipeline.cv_results[pipeline.cv_results['model_name'] == best_model].iloc[0]
        test_results = pipeline.test_results[pipeline.test_results['model_name'] == best_model].iloc[0]

        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED")
        print(f"Best Model: {best_model}")
        print(f"CV F1 Score: {cv_results['f1_mean']:.4f}")
        print(f"Test F1 Score: {test_results['f1']:.4f}")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"{'='*80}\n")

        # Save model with metadata for inference
        import time
        project_timestamp = int(time.time() * 1000)
        project_name = f"automl_{project_timestamp}"
        model_path = os.path.join(MODEL_DIR, f"{project_name}_pipeline.pkl")
        
        # Get target label mapping (original values before encoding)
        target_labels = None
        if request.task_type == 'classification':
            # Get unique target values from original data
            target_labels = {i: label for i, label in enumerate(sorted(df[request.target_column].unique()))}
            print(f"Target label mapping: {target_labels}")
        
        # Create metadata for clean inference
        model_metadata = {
            'project_name': project_name,
            'target_column': request.target_column,
            'training_columns': df.drop(columns=[request.target_column]).columns.tolist(),
            'feature_types': pipeline.preparator.feature_stats.get('types', {}),
            'task_type': request.task_type,
            'best_model': best_model,
            'training_shape': df.shape,
            'timestamp': project_timestamp,
            'target_labels': target_labels  # Store the label mapping
        }
        
        # Save pipeline and metadata together
        joblib.dump({
            'pipeline': pipeline,
            'metadata': model_metadata
        }, model_path)
        
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Saved metadata: {list(model_metadata.keys())}")

        # Convert all models results to list of dicts
        all_cv_results = []
        all_test_results = []
        
        for _, row in pipeline.cv_results.iterrows():
            all_cv_results.append({
                "model_name": row['model_name'],
                "accuracy": float(row['accuracy_mean']),
                "precision": float(row['precision_mean']),
                "recall": float(row['recall_mean']),
                "f1": float(row['f1_mean']),
                "training_time": float(row['training_time'])
            })
        
        for _, row in pipeline.test_results.iterrows():
            all_test_results.append({
                "model_name": row['model_name'],
                "accuracy": float(row['accuracy']),
                "precision": float(row['precision']),
                "recall": float(row['recall']),
                "f1": float(row['f1'])
            })
        
        # Get predictions from best model
        best_model_obj = pipeline.trainer.models[best_model]
        predictions = best_model_obj.predict(pipeline.X_test)
        
        # Store globally for session
        global LAST_TRAINED_PIPELINE, LAST_PROJECT_NAME
        LAST_TRAINED_PIPELINE = pipeline
        LAST_PROJECT_NAME = project_name  # Use the generated project name

        # Convert predictions to list
        predictions_list = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)

        return {
            "status": "success",
            "best_model": best_model,
            "cv_scores": {
                "accuracy": float(cv_results['accuracy_mean']),
                "precision": float(cv_results['precision_mean']),
                "recall": float(cv_results['recall_mean']),
                "f1": float(cv_results['f1_mean'])
            },
            "test_scores": {
                "accuracy": float(test_results['accuracy']),
                "precision": float(test_results['precision']),
                "recall": float(test_results['recall']),
                "f1": float(test_results['f1'])
            },
            "all_models_cv": all_cv_results,
            "all_models_test": all_test_results,
            "predictions": predictions_list,
            "actual_labels": pipeline.y_test.tolist() if hasattr(pipeline.y_test, 'tolist') else list(pipeline.y_test),
            "project_id": project_name,  # Return generated project name
            "dataset_shape": list(df.shape),
            "target_column": request.target_column,
            "training_samples": int(len(pipeline.X_train)),
            "test_samples": int(len(pipeline.X_test)),
            "num_features": int(pipeline.X_train.shape[1]),
            "warnings": leakage_candidates if leakage_candidates else []
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n{'='*80}")
        print(f"ERROR IN AUTOML TRAINING")
        print(f"{'='*80}")
        print(error_trace)
        print(f"{'='*80}\n")
        
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


# ===================== PREDICT ON NEW DATA =====================
@app.post("/automl/predict")
async def predict_new_data(file: UploadFile = File(...), project_name: str = Form(...)):
    """
    Make predictions on new unseen data using trained model.
    Accepts CSV file with same features as training data (without target column).
    """
    try:
        print(f"\n{'='*80}")
        print(f"STARTING PREDICTION")
        print(f"{'='*80}")
        print(f"Project: {project_name}")
        
        # Load model
        model_path = os.path.join(MODEL_DIR, f"{project_name}_pipeline.pkl")
        if not os.path.exists(model_path):
            return {
                "status": "error",
                "error": f"Model not found: {project_name}. Please train a model first."
            }
        
        print(f"Loading model from: {model_path}")
        saved_data = joblib.load(model_path)
        pipeline = saved_data['pipeline']
        metadata = saved_data['metadata']
        
        print(f"Model loaded successfully!")
        print(f"Target column: {metadata['target_column']}")
        print(f"Expected features: {metadata['training_columns']}")
        print(f"Task type: {metadata['task_type']}")
        
        # Save and load new data
        file_path = os.path.join(UPLOAD_DIR, f"predict_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"\nLoading new data from: {file.filename}")
        new_data = pd.read_csv(file_path)
        original_data = new_data.copy()  # Keep original for output
        
        print(f"New data shape: {new_data.shape}")
        print(f"New data columns: {new_data.columns.tolist()}")
        
        # Validate: Check if target column is present (it shouldn't be)
        target_col = metadata['target_column']
        if target_col in new_data.columns:
            print(f"\n⚠️  Warning: Target column '{target_col}' found in new data. It will be removed for prediction.")
            new_data = new_data.drop(columns=[target_col])
        
        # Validate: Check if all required features are present
        missing_features = set(metadata['training_columns']) - set(new_data.columns)
        extra_features = set(new_data.columns) - set(metadata['training_columns'])
        
        if missing_features:
            return {
                "status": "error",
                "error": f"Missing required features: {list(missing_features)}",
                "expected_features": metadata['training_columns'],
                "provided_features": new_data.columns.tolist()
            }
        
        if extra_features:
            print(f"\n⚠️  Warning: Extra features found: {list(extra_features)}. They will be ignored.")
            new_data = new_data[metadata['training_columns']]
        
        # Ensure columns are in the same order as training
        new_data = new_data[metadata['training_columns']]
        
        print(f"\n✓ Feature validation passed!")
        print(f"Features aligned: {new_data.shape[1]} features")
        
        # Apply the same preprocessing transformations
        print(f"\nApplying preprocessing transformations...")
        
        # Handle missing values (using fitted imputers)
        new_data = pipeline.preparator.handle_missing_values(new_data, fit=False)
        print(f"✓ Missing values handled")
        
        # Encode categorical features (using fitted encoders)
        print(f"Before encoding - data types:")
        for col in new_data.columns:
            if new_data[col].dtype == 'object':
                print(f"  {col}: {new_data[col].dtype} - Sample: {new_data[col].iloc[0]}")
        
        new_data = pipeline.preparator.encode_categorical_features(new_data, fit=False)
        print(f"✓ Categorical features encoded")
        
        # Check if any object columns remain after encoding
        object_cols = new_data.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            print(f"\n⚠️  WARNING: Still have object columns after encoding: {object_cols}")
            print(f"Forcing encoding for these columns...")
            for col in object_cols:
                print(f"  Encoding {col}: {new_data[col].unique()[:5]}")
                # Force encode using categorical codes
                new_data[col] = pd.Categorical(new_data[col]).codes
            print(f"✓ Forced encoding complete")
        
        # Scale features (using fitted scaler)
        if pipeline.preparator.scaler is not None:
            # Get the columns that were scaled during training
            scaler_features = pipeline.preparator.scaler_features
            if scaler_features is not None:
                # Only scale the columns that were scaled during training
                new_data[scaler_features] = pipeline.preparator.scaler.transform(new_data[scaler_features])
            else:
                # Fallback: scale all numerical columns
                new_data_scaled = pipeline.preparator.scaler.transform(new_data)
                new_data = pd.DataFrame(new_data_scaled, columns=new_data.columns, index=new_data.index)
        
        print(f"✓ Preprocessing complete!")
        print(f"Transformed data shape: {new_data.shape}")
        
        # Make predictions using the best model
        best_model_name = metadata['best_model']
        best_model = pipeline.trainer.models[best_model_name]
        
        print(f"\nMaking predictions with {best_model_name}...")
        predictions = best_model.predict(new_data)
        
        # Replace any NaN in predictions
        predictions = np.nan_to_num(predictions, nan=0.0)
        
        # Get prediction probabilities (if available for classification)
        probabilities = None
        if metadata['task_type'] == 'classification' and hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(new_data)
            # Replace NaN in probabilities
            probabilities = np.nan_to_num(probabilities, nan=0.0)
            print(f"✓ Prediction probabilities calculated")
        
        print(f"✓ Predictions complete!")
        print(f"Number of predictions: {len(predictions)}")
        
        # Map numeric predictions back to original labels
        predictions_mapped = predictions.copy()
        if metadata['task_type'] == 'classification' and metadata.get('target_labels'):
            target_labels = metadata['target_labels']
            # Convert numeric predictions to original labels
            predictions_mapped = np.array([target_labels.get(int(p), p) for p in predictions])
            print(f"✓ Predictions mapped to original labels: {target_labels}")
        
        # Prepare results with original data
        results_df = original_data.copy()
        results_df[f'{metadata["target_column"]}_prediction'] = predictions_mapped  # Use target column name
        
        # Add probabilities if available
        if probabilities is not None:
            # Get class labels
            classes = best_model.classes_ if hasattr(best_model, 'classes_') else None
            if classes is not None:
                for i, class_label in enumerate(classes):
                    results_df[f'probability_class_{class_label}'] = probabilities[:, i]
        
        # Replace NaN values with None for JSON serialization
        results_df = results_df.fillna(value=None)
        
        # Convert to dict with additional NaN handling
        results_dict = results_df.to_dict('records')
        # Clean any remaining NaN values (in case fillna didn't catch them all)
        import math
        def clean_value(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        results_dict = [{k: clean_value(v) for k, v in record.items()} for record in results_dict]
        
        # Save results to file
        results_filename = f"predictions_results_{project_name}.csv"
        results_path = os.path.join(os.path.dirname(MODEL_DIR), results_filename)
        results_df.to_csv(results_path, index=False)
        
        print(f"\n✓ Results saved to: {results_path}")
        print(f"{'='*80}")
        print(f"PREDICTION COMPLETED")
        print(f"{'='*80}\n")
        
        # Prepare response
        response = {
            "status": "success",
            "project_name": project_name,
            "model_used": best_model_name,
            "task_type": metadata['task_type'],
            "num_predictions": int(len(predictions)),
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            "results_with_data": results_dict,  # Use cleaned dict instead of raw to_dict
            "results_file": results_filename,
            "summary": {
                "total_rows": int(len(predictions)),
                "unique_predictions": int(pd.Series(predictions).nunique()),
            }
        }
        
        # Add prediction distribution summary
        if metadata['task_type'] == 'classification':
            pred_counts = pd.Series(predictions).value_counts().to_dict()
            response["summary"]["prediction_distribution"] = {str(k): int(v) for k, v in pred_counts.items()}
        else:
            pred_stats = {
                "min": float(predictions.min()) if not np.isnan(predictions.min()) else None,
                "max": float(predictions.max()) if not np.isnan(predictions.max()) else None,
                "mean": float(predictions.mean()) if not np.isnan(predictions.mean()) else None,
                "median": float(np.median(predictions)) if not np.isnan(np.median(predictions)) else None
            }
            response["summary"]["prediction_stats"] = pred_stats
        
        # Add probabilities if available
        if probabilities is not None:
            response["probabilities"] = probabilities.tolist()
            response["class_labels"] = classes.tolist() if classes is not None else None
        
        return response
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n{'='*80}")
        print(f"ERROR IN PREDICTION")
        print(f"{'='*80}")
        print(error_trace)
        print(f"{'='*80}\n")
        
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_trace
        }


# ===================== IMAGE TRAINING =====================
class ImageTrainingRequest(BaseModel):
    image_data: List[dict]  # List of {path: str, label: str}
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    test_size: float = 0.2
    model_names: List[str] = None
    project_name: str = "image_project"


@app.post("/automl/train-images")
async def train_images(request: ImageTrainingRequest):
    """Train image classification models"""
    global LAST_IMAGE_PIPELINE, LAST_PROJECT_NAME
    
    try:
        print(f"\n{'='*80}")
        print(f"STARTING IMAGE TRAINING")
        print(f"{'='*80}")
        print(f"Number of images: {len(request.image_data)}")
        print(f"Epochs: {request.epochs}")
        print(f"Batch size: {request.batch_size}")
        print(f"Models: {request.model_names}")
        
        # Extract paths and labels
        image_paths = []
        labels = []
        
        for item in request.image_data:
            # Handle both 'path' and 'file_path' keys
            path = item.get('path') or item.get('file_path')
            if path:
                image_paths.append(path)
                labels.append(item.get('label', 'unknown'))
        
        if len(image_paths) == 0:
            return {"error": "No valid image paths provided"}
        
        print(f"\nProcessing {len(image_paths)} images")
        print(f"Unique labels: {sorted(list(set(labels)))}")
        
        # Use default models if not specified
        if request.model_names is None or len(request.model_names) == 0:
            request.model_names = ['resnet18', 'mobilenet_v2']
        
        # Run image pipeline
        pipeline = run_image_pipeline(
            image_paths=image_paths,
            labels=labels,
            image_size=224,
            batch_size=request.batch_size,
            test_size=request.test_size,
            epochs=request.epochs,
            model_names=request.model_names,
            learning_rate=request.learning_rate,
            pretrained=True,
            project_id=request.project_name
        )
        
        # Save pipeline
        import time
        project_timestamp = int(time.time() * 1000)
        project_name = f"image_{project_timestamp}"
        model_path = os.path.join(MODEL_DIR, f"{project_name}_pipeline.pth")
        
        # Save with torch
        torch.save({
            'pipeline': pipeline,
            'class_names': pipeline.data_info['class_names'],
            'idx_to_label': pipeline.data_info['idx_to_label'],
            'label_to_idx': pipeline.data_info['label_to_idx'],
            'best_model_name': pipeline.best_model_name,
            'num_classes': pipeline.data_info['num_classes'],
            'image_size': pipeline.image_size,
            'project_name': project_name
        }, model_path)
        
        # Store globally
        LAST_IMAGE_PIPELINE = pipeline
        LAST_PROJECT_NAME = project_name
        
        print(f"✓ Model saved to: {model_path}")
        
        # Prepare response
        summary = pipeline.get_summary()
        
        # Format training results
        training_results = []
        for model_name, history in pipeline.training_results.items():
            training_results.append({
                'model_name': model_name,
                'best_val_acc': float(history['best_val_acc']),
                'final_train_acc': float(history['train_acc'][-1]) if history['train_acc'] else 0.0,
                'final_val_acc': float(history['val_acc'][-1]) if history['val_acc'] else 0.0,
                'training_time': float(history['training_time'])
            })
        
        # Format test results
        test_results = []
        for model_name, metrics in pipeline.test_results.items():
            test_results.append({
                'model_name': model_name,
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1']),
                'per_class_metrics': metrics['per_class_metrics']
            })
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED")
        print(f"Best Model: {pipeline.best_model_name}")
        print(f"{'='*80}\n")
        
        return {
            'status': 'success',
            'project_name': project_name,
            'best_model': pipeline.best_model_name,
            'num_classes': pipeline.data_info['num_classes'],
            'class_names': pipeline.data_info['class_names'],
            'train_size': pipeline.data_info['train_size'],
            'test_size': pipeline.data_info['test_size'],
            'training_results': training_results,
            'test_results': test_results,
            'model_path': model_path
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n{'='*80}")
        print(f"ERROR IN IMAGE TRAINING")
        print(f"{'='*80}")
        print(error_trace)
        print(f"{'='*80}\n")
        
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_trace
        }


# ===================== IMAGE PREDICTION =====================
class ImagePredictionRequest(BaseModel):
    image_paths: List[str]
    project_name: str = None


@app.post("/automl/predict-images")
async def predict_images(request: ImagePredictionRequest):
    """Make predictions on images"""
    global LAST_IMAGE_PIPELINE, LAST_PROJECT_NAME
    
    try:
        print(f"\n{'='*80}")
        print(f"STARTING IMAGE PREDICTION")
        print(f"{'='*80}")
        
        # Determine which pipeline to use
        project_name = request.project_name or LAST_PROJECT_NAME
        
        if project_name and project_name != LAST_PROJECT_NAME:
            # Load pipeline from file
            model_path = os.path.join(MODEL_DIR, f"{project_name}_pipeline.pth")
            if not os.path.exists(model_path):
                return {"error": f"Model not found: {project_name}"}
            
            print(f"Loading pipeline from: {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            pipeline = checkpoint['pipeline']
        elif LAST_IMAGE_PIPELINE:
            # Use last trained pipeline
            pipeline = LAST_IMAGE_PIPELINE
            project_name = LAST_PROJECT_NAME
        else:
            return {"error": "No trained model available. Please train a model first."}
        
        print(f"Using project: {project_name}")
        print(f"Number of images to predict: {len(request.image_paths)}")
        
        # Make predictions
        results = pipeline.predict(request.image_paths)
        
        print(f"✓ Predictions complete")
        print(f"{'='*80}\n")
        
        return {
            'status': 'success',
            'project_name': project_name,
            'results': results['predictions'],
            'model_used': results['model_used'],
            'num_classes': results['num_classes'],
            'class_names': results['class_names']
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n{'='*80}")
        print(f"ERROR IN IMAGE PREDICTION")
        print(f"{'='*80}")
        print(error_trace)
        print(f"{'='*80}\n")
        
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_trace
        }


# ===================== IMAGE UPLOAD =====================
@app.post("/automl/upload-images")
async def upload_images(request: Request):
    """Upload multiple images and return their paths"""
    try:
        print(f"\n{'='*60}")
        print(f"IMAGE UPLOAD REQUEST RECEIVED")
        print(f"Content-Type: {request.headers.get('content-type')}")
        print(f"{'='*60}\n")
        
        # Ensure the image upload directory exists
        os.makedirs(IMAGE_UPLOAD_DIR, exist_ok=True)
        print(f"Upload directory: {os.path.abspath(IMAGE_UPLOAD_DIR)}")
        
        # Get form data
        form = await request.form()
        files = form.getlist('files')
        
        if not files:
            print("No files found in form data")
            print(f"Form keys: {list(form.keys())}")
            return {
                'status': 'error',
                'error': 'No files provided'
            }
        
        print(f"Number of files: {len(files)}")
        
        uploaded_paths = []
        
        for idx, file in enumerate(files):
            print(f"Processing file {idx + 1}/{len(files)}: {file.filename}")
            # Create unique filename
            import time
            timestamp = int(time.time() * 1000)
            filename = f"{timestamp}_{file.filename}"
            file_path = os.path.join(IMAGE_UPLOAD_DIR, filename)
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Get absolute path
            abs_path = os.path.abspath(file_path)
            uploaded_paths.append(abs_path)
            print(f"  ✓ Saved to: {abs_path}")
        
        print(f"\n✓ Successfully uploaded {len(uploaded_paths)} images\n")
        
        return {
            'status': 'success',
            'uploaded_count': len(uploaded_paths),
            'paths': uploaded_paths
        }
        
    except Exception as e:
        print(f"\n❌ ERROR during image upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'error': str(e)
        }



