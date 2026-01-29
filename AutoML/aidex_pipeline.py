"""
AIDEX - Complete Pipeline Integration
End-to-end automated machine learning workflow
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

warnings.filterwarnings('ignore')

# Import AIDEX modules
from Data_Pre_Processing.feature_preparation import FeaturePreparator, auto_prepare_features
from Model_Training.model_trainer import ModelTrainer, auto_train_and_select
from Model_Training.model_selector import ModelSelector, auto_select_model


class AIDEXPipeline:
    """
    Complete AIDEX automated machine learning pipeline
    """
    
    def __init__(self, task_type='classification', test_size=0.2, random_state=42,
                 project_id: Optional[str] = None):
        """
        Initialize AIDEX Pipeline
        
        Args:
            task_type: 'classification' or 'regression'
            test_size: Proportion of dataset for testing (0.0 - 1.0)
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type
        self.test_size = test_size
        self.random_state = random_state
        # Project-centric context
        # In a full AIDEX deployment this ID would come from the project service / database.
        # Here we keep it simple but still scope everything to a project identifier.
        self.project_id = project_id or f"aidex_project_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        self.preparator = None
        self.trainer = None
        self.selector = None
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.cv_results = None
        self.test_results = None
        self.best_model_name = None
        
    def load_data(self, filepath_or_dataframe, target_column):
        """
        Load dataset from file or DataFrame
        
        Args:
            filepath_or_dataframe: Path to CSV file or pandas DataFrame
            target_column: Name of the target column
            
        Returns:
            Loaded DataFrame
        """
        if isinstance(filepath_or_dataframe, str):
            print(f"Loading data from: {filepath_or_dataframe}")
            df = pd.read_csv(filepath_or_dataframe)
        else:
            df = filepath_or_dataframe.copy()
        
        print(f"Dataset shape: {df.shape}")
        print(f"Target column: {target_column}")
        print(f"\nDataset info:")
        print(df.info())
        
        return df
    
    def prepare_features(self, df, target_column, 
                        scaling_method='standard',
                        create_interactions=False,
                        select_features=False,
                        k_features='all',
                        handle_missing='auto'):
        """
        Prepare features for modeling
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            scaling_method: 'standard', 'minmax', 'robust', or None
            create_interactions: Whether to create interaction features
            select_features: Whether to perform feature selection
            k_features: Number of features to select
            handle_missing: Strategy for missing values ('auto', 'mean', 'median', 'mode', 'knn', 'drop', or None)
            
        Returns:
            Prepared train and test sets
        """
        print("\n" + "="*80)
        print("STEP 1: FEATURE PREPARATION")
        print("="*80)
        
        # Prepare features
        X, y, self.preparator = auto_prepare_features(
            df,
            target_col=target_column,
            task_type=self.task_type,
            scaling_method=scaling_method,
            create_interactions=create_interactions,
            select_features=select_features,
            k_features=k_features,
            handle_missing=handle_missing
        )
        
        print(f"\nFeature preparation complete!")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of samples: {X.shape[0]}")
        
        feature_info = self.preparator.get_feature_info()
        print(f"\nFeature types identified:")
        for ftype, cols in feature_info['feature_types'].items():
            if cols:
                print(f"  {ftype}: {len(cols)} features")
        
        # Split into train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y if self.task_type == 'classification' else None
        )
        
        print(f"\nData split:")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self, cv_folds=5, verbose=True):
        """
        Train and evaluate multiple models
        
        Args:
            cv_folds: Number of cross-validation folds
            verbose: Whether to print progress
            
        Returns:
            Cross-validation results DataFrame
        """
        print("\n" + "="*80)
        print("STEP 2: MODEL TRAINING & EVALUATION")
        print("="*80)
        
        self.trainer = ModelTrainer(task_type=self.task_type, random_state=self.random_state)
        
        self.cv_results = self.trainer.train_and_evaluate_all(
            self.X_train,
            self.y_train,
            cv_folds=cv_folds,
            verbose=verbose
        )
        
        print(f"\n[OK] Trained {len(self.cv_results)} models successfully!")
        
        return self.cv_results
    
    def test_models(self, verbose=True):
        """
        Test all trained models on holdout test set
        
        Returns:
            Test results DataFrame
        """
        print("\n" + "="*80)
        print("STEP 3: TESTING ON HOLDOUT SET")
        print("="*80)
        
        self.test_results = self.trainer.test_all_models(self.X_test, self.y_test)
        
        if verbose:
            print("\nTest Results (Top 5):")
            print(self.test_results.head())
        
        return self.test_results
    
    def select_best_model(self, priority='balanced', min_performance=None, verbose=True):
        """
        Select the best model based on criteria
        
        Args:
            priority: 'performance', 'speed', 'interpretability', 'stable', or 'balanced'
            min_performance: Minimum acceptable performance
            verbose: Whether to print selection summary
            
        Returns:
            Selected model name
        """
        print("\n" + "="*80)
        print("STEP 4: MODEL SELECTION")
        print("="*80)
        
        self.best_model_name, self.selector = auto_select_model(
            self.cv_results,
            task_type=self.task_type,
            priority=priority,
            min_performance=min_performance,
            verbose=verbose
        )
        
        return self.best_model_name
    
    def run_complete_pipeline(self, df, target_column,
                             scaling_method='standard',
                             create_interactions=False,
                             select_features=False,
                             k_features='all',
                             cv_folds=5,
                             selection_priority='balanced',
                             min_performance=None,
                             handle_missing='auto'):
        """
        Run the complete AIDEX pipeline
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            scaling_method: Feature scaling method
            create_interactions: Whether to create interaction features
            select_features: Whether to perform feature selection
            k_features: Number of features to select
            cv_folds: Number of cross-validation folds
            selection_priority: Model selection priority
            min_performance: Minimum acceptable performance
            handle_missing: Strategy for missing values ('auto', 'mean', 'median', 'mode', 'knn', 'drop', or None)
            
        Returns:
            Dictionary with all results
        """
        print("\n" + "="*80)
        print("AIDEX - AUTOMATED MACHINE LEARNING PIPELINE")
        print("="*80)
        
        # Step 1: Feature Preparation
        self.prepare_features(
            df, target_column,
            scaling_method=scaling_method,
            create_interactions=create_interactions,
            select_features=select_features,
            k_features=k_features,
            handle_missing=handle_missing
        )
        
        # Step 2: Model Training
        self.train_models(cv_folds=cv_folds, verbose=True)
        
        # Step 3: Testing
        self.test_models(verbose=True)
        
        # Step 4: Model Selection
        self.select_best_model(priority=selection_priority, 
                              min_performance=min_performance,
                              verbose=True)
        
        # Generate summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        
        return self.get_results_summary()
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a **project-scoped**, serialization‑friendly summary of pipeline results.
        
        This intentionally avoids returning raw models so that the summary can be
        safely stored (e.g. as JSON) and used later to reconstruct reports without
        re‑running the full pipeline.
        """
        # Safely convert DataFrames/Series to plain Python structures
        cv_results_serializable = None
        if self.cv_results is not None:
            cv_results_serializable = self.cv_results.to_dict(orient='records')

        test_results_serializable = None
        if self.test_results is not None:
            test_results_serializable = self.test_results.to_dict(orient='records')

        feature_info = None
        if self.preparator is not None:
            try:
                feature_info = self.preparator.get_feature_info()
            except Exception:
                feature_info = None

        summary: Dict[str, Any] = {
            'project_id': self.project_id,
            'task_type': self.task_type,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'best_model_name': self.best_model_name,
            'cv_results': cv_results_serializable,
            'test_results': test_results_serializable,
            'feature_info': feature_info,
            'dataset': {
                'n_samples_total': int(self.X_train.shape[0] + self.X_test.shape[0]) if self.X_train is not None and self.X_test is not None else None,
                'n_features': int(self.X_train.shape[1]) if self.X_train is not None else None,
                'n_train': int(self.X_train.shape[0]) if self.X_train is not None else None,
                'n_test': int(self.X_test.shape[0]) if self.X_test is not None else None,
            }
        }

        return summary

    # ------------------------------------------------------------------
    # Project‑centric persistence & reporting
    # ------------------------------------------------------------------
    def _ensure_project_dir(self, output_dir: str) -> str:
        """
        Ensure that a directory exists for storing project‑scoped artifacts.
        """
        project_root = os.path.join(output_dir, self.project_id)
        os.makedirs(project_root, exist_ok=True)
        return project_root

    def save_project_state(self, output_dir: str = "projects") -> str:
        """
        Persist a structured, project‑scoped snapshot of the pipeline state.

        The saved JSON can be used later to regenerate reports and dashboards
        **without** re‑running heavy training logic, honoring AIDEX's rule that
        results come from stored state rather than recomputation.
        """
        project_root = self._ensure_project_dir(output_dir)
        state_path = os.path.join(project_root, "project_state.json")

        state = self.get_results_summary()
        state['saved_at_utc'] = datetime.utcnow().isoformat() + "Z"

        # Use a basic JSON dump; all nested objects are converted in get_results_summary
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        return state_path

    def save_project_report(self, output_dir: str = "projects") -> str:
        """
        Generate a human‑readable, project‑scoped report.

        This report is meant for UI consumption and for non‑technical users. It
        explains, in natural language, what AIDEX has done: data split, model
        performance, and high‑level recommendations.
        """
        project_root = self._ensure_project_dir(output_dir)
        report_path = os.path.join(project_root, "report.txt")

        summary = self.get_results_summary()
        best_model_name = summary.get("best_model_name") or "N/A"
        dataset_info = summary.get("dataset", {})

        # Extract headline metrics from test results if available
        headline_accuracy = None
        headline_f1 = None
        if summary.get("test_results"):
            # Find the row that matches the best model, fall back to first row
            best_row = None
            for row in summary["test_results"]:
                if row.get("model_name") == best_model_name:
                    best_row = row
                    break
            if best_row is None:
                best_row = summary["test_results"][0]

            headline_accuracy = best_row.get("accuracy")
            headline_f1 = best_row.get("f1")

        lines = []
        lines.append("=" * 80)
        lines.append("AIDEX PROJECT REPORT")
        lines.append("=" * 80)
        lines.append(f"Project ID        : {self.project_id}")
        lines.append(f"Task Type         : {self.task_type}")
        lines.append(f"Created At (UTC)  : {datetime.utcnow().isoformat()}Z")
        lines.append("")
        lines.append("1) DATASET OVERVIEW")
        lines.append("-" * 80)
        lines.append(f"Total samples     : {dataset_info.get('n_samples_total')}")
        lines.append(f"Training samples  : {dataset_info.get('n_train')}")
        lines.append(f"Test samples      : {dataset_info.get('n_test')}")
        lines.append(f"Input features    : {dataset_info.get('n_features')}")

        if summary.get("feature_info"):
            ft = summary["feature_info"].get("feature_types", {})
            lines.append("")
            lines.append("Detected feature types:")
            for ftype, cols in ft.items():
                if cols:
                    lines.append(f"  - {ftype}: {len(cols)} features")

        lines.append("")
        lines.append("2) SELECTED MODEL")
        lines.append("-" * 80)
        lines.append(f"Chosen model      : {best_model_name}")
        if headline_accuracy is not None:
            lines.append(f"Test accuracy     : {headline_accuracy:.4f} ({headline_accuracy*100:.2f}%)")
        if headline_f1 is not None:
            lines.append(f"Test F1 (weighted): {headline_f1:.4f} ({headline_f1*100:.2f}%)")

        lines.append("")
        lines.append("3) INTERPRETATION FOR NON‑TECHNICAL USERS")
        lines.append("-" * 80)
        lines.append(
            "AIDEX automatically cleaned your data, encoded categorical features, "
            "scaled numeric values, trained multiple candidate models, and "
            "selected the one that offers the best balance of accuracy, stability, "
            "and robustness. All of these decisions were made internally based on "
            "the project state – you did not need to configure algorithms or "
            "hyperparameters manually."
        )
        lines.append("")
        lines.append("4) NEXT STEPS")
        lines.append("-" * 80)
        lines.append(
            "• You can safely reuse this project ID to re‑open results in the UI "
            "without retraining models.\n"
            "• Use the stored project_state.json file for dashboards, audits, or "
            "regulatory documentation.\n"
            "• When new data arrives, create a new project or a new project "
            "version instead of overwriting this one."
        )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return report_path
    
    def predict(self, new_data):
        """
        Make predictions on new data using the best model
        
        Args:
            new_data: DataFrame with same structure as training data (without target)
            
        Returns:
            Array of predictions
        """
        if self.preparator is None or self.trainer is None:
            raise ValueError("Pipeline must be run before making predictions.")
        
        # Prepare features (transform only, no fit)
        X_prepared, _ = self.preparator.prepare_features(new_data, fit=False)
        
        # Make predictions
        predictions = self.trainer.predict(X_prepared, model_name=self.best_model_name)
        
        return predictions
    
    def get_best_model(self):
        """
        Get the trained best model
        
        Returns:
            Trained model instance
        """
        if self.trainer is None or self.best_model_name is None:
            raise ValueError("Pipeline must be run before accessing the best model.")
        
        return self.trainer.get_model(self.best_model_name)


def run_aidex(data_path,
              target_column,
              task_type: str = 'classification',
              test_size: float = 0.2,
              scaling_method: str = 'standard',
              create_interactions: bool = False,
              select_features: bool = False,
              cv_folds: int = 5,
              selection_priority: str = 'balanced',
              project_id: Optional[str] = None):
    """
    Convenience function to run complete AIDEX pipeline
    
    Args:
        data_path: Path to CSV file or pandas DataFrame
        target_column: Name of target column
        task_type: 'classification' or 'regression'
        test_size: Proportion for test set
        scaling_method: Feature scaling method
        create_interactions: Whether to create interaction features
        select_features: Whether to perform feature selection
        cv_folds: Number of cross-validation folds
        selection_priority: Model selection priority
        
    Returns:
        AIDEXPipeline instance with results
    """
    # Initialize pipeline in the context of a specific project
    pipeline = AIDEXPipeline(task_type=task_type, test_size=test_size, project_id=project_id)
    
    # Load data
    df = pipeline.load_data(data_path, target_column)
    
    # Run complete pipeline (fully automatic – no user decisions)
    pipeline.run_complete_pipeline(
        df, target_column,
        scaling_method=scaling_method,
        create_interactions=create_interactions,
        select_features=select_features,
        cv_folds=cv_folds,
        selection_priority=selection_priority
    )
    
    return pipeline


if __name__ == "__main__":
    """
    Example usage of AIDEX Pipeline
    """
    print("\n" + "="*80)
    print("AIDEX - EXAMPLE USAGE")
    print("="*80)
    
    # Example with synthetic data
    from sklearn.datasets import make_classification
    
    # Create sample dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, random_state=42)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    # Run AIDEX pipeline
    pipeline = run_aidex(
        data_path=df,
        target_column='target',
        task_type='classification',
        test_size=0.2,
        scaling_method='standard',
        create_interactions=False,
        select_features=False,
        cv_folds=5,
        selection_priority='balanced'
    )
    
    print("\n" + "="*80)
    print("Example complete! You can now use this pipeline with your own data.")
    print("="*80)
