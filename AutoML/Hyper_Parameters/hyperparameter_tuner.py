"""
Hyperparameter Tuning Module for AIDEX
Automatically optimizes model hyperparameters to improve performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, r2_score
from typing import Dict, List, Tuple, Optional, Any
import warnings
import time
warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """
    Automated hyperparameter tuning for machine learning models.
    Supports Grid Search, Random Search, and provides optimized parameters.
    """
    
    def __init__(self, task_type: str = 'classification', random_state: int = 42):
        """
        Initialize the Hyperparameter Tuner
        
        Args:
            task_type: 'classification' or 'regression'
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type.lower()
        self.random_state = random_state
        self.best_params = {}
        self.tuning_results = {}
        self.best_estimators = {}
        
    def get_param_grid(self, model_name: str) -> Dict[str, List]:
        """
        Get hyperparameter search space for different models
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of hyperparameter ranges
        """
        param_grids = {
            'logistic regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [100, 200, 500, 1000]
            },
            'decision tree': {
                'max_depth': [3, 5, 7, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'criterion': ['gini', 'entropy'] if self.task_type == 'classification' else ['squared_error', 'friedman_mse']
            },
            'random forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'gradient boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10, -1],
                'num_leaves': [15, 31, 63, 127],
                'min_child_samples': [10, 20, 30],
                'subsample': [0.8, 0.9, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            },
            'k-nearest neighbors': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'naive bayes': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            },
            'ridge regression': {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            },
            'lasso regression': {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
            },
            'elasticnet': {
                'alpha': [0.001, 0.01, 0.1, 1, 10],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        }
        
        model_name_lower = model_name.lower()
        for key, grid in param_grids.items():
            if key in model_name_lower:
                return grid
        
        # Default minimal grid for unknown models
        return {}
    
    def get_small_param_grid(self, model_name: str) -> Dict[str, List]:
        """
        Get smaller hyperparameter search space for faster tuning
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of reduced hyperparameter ranges
        """
        small_grids = {
            'decision tree': {
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 10],
                'min_samples_leaf': [1, 4]
            },
            'random forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'max_features': ['sqrt', None]
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [5, 10, -1],
                'num_leaves': [31, 63]
            }
        }
        
        model_name_lower = model_name.lower()
        for key, grid in small_grids.items():
            if key in model_name_lower:
                return grid
        
        # Fall back to full grid
        return self.get_param_grid(model_name)
    
    def tune_model(self, model, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                  method: str = 'random', cv_folds: int = 5, n_iter: int = 20,
                  param_grid: Dict = None, verbose: bool = True) -> Tuple[Any, Dict, float]:
        """
        Tune hyperparameters for a single model
        
        Args:
            model: Sklearn-compatible model instance
            model_name: Name of the model
            X_train: Training features
            y_train: Training target
            method: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
            cv_folds: Number of cross-validation folds
            n_iter: Number of iterations for random search
            param_grid: Custom parameter grid (uses default if None)
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_estimator, best_params, best_score)
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Tuning {model_name}...")
            print(f"Method: {method.upper()}")
            print(f"{'='*80}")
        
        # Get parameter grid
        if param_grid is None:
            if method == 'random':
                param_grid = self.get_param_grid(model_name)
            else:
                param_grid = self.get_small_param_grid(model_name)
        
        if not param_grid:
            if verbose:
                print(f"[WARN] No parameter grid available for {model_name}")
                print("[INFO] Using default parameters and fitting model...")
            
            # Fit the model with default parameters
            model.fit(X_train, y_train)
            
            # Calculate score
            if self.task_type == 'classification':
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, X_train, y_train, cv=cv_folds, 
                                        scoring=make_scorer(f1_score, average='weighted', zero_division=0))
                best_score = scores.mean()
            else:
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                best_score = scores.mean()
            
            self.best_estimators[model_name] = model
            self.best_params[model_name] = {}
            self.tuning_results[model_name] = {
                'best_score': best_score,
                'best_params': {},
                'tuning_time': 0.0,
                'n_iterations': 1
            }
            
            if verbose:
                print("[OK] Model fitted with default parameters")
                print(f"[METRIC] CV Score: {best_score:.4f}")
            
            return model, {}, best_score
        
        # Define scoring metric
        if self.task_type == 'classification':
            scoring = make_scorer(f1_score, average='weighted', zero_division=0)
        else:
            scoring = make_scorer(r2_score)
        
        start_time = time.time()
        
        try:
            # Perform hyperparameter search
            if method == 'grid':
                search = GridSearchCV(
                    model,
                    param_grid,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=1,
                    verbose=0
                )
            else:  # random search
                search = RandomizedSearchCV(
                    model,
                    param_grid,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=1,
                    random_state=self.random_state,
                    verbose=0
                )
            
            search.fit(X_train, y_train)
            
            tuning_time = time.time() - start_time
            
            # Store results
            self.best_params[model_name] = search.best_params_
            self.best_estimators[model_name] = search.best_estimator_
            self.tuning_results[model_name] = {
                'best_score': search.best_score_,
                'best_params': search.best_params_,
                'tuning_time': tuning_time,
                'n_iterations': len(search.cv_results_['mean_test_score'])
            }
            
            if verbose:
                print(f"\n[OK] Tuning completed in {tuning_time:.2f} seconds")
                print(f"[METRIC] Best Score: {search.best_score_:.4f}")
                print("[PARAMS] Best Parameters:")
                for param, value in search.best_params_.items():
                    print(f"   {param}: {value}")
                print(f"[INFO] Iterations tested: {self.tuning_results[model_name]['n_iterations']}")
            
            return search.best_estimator_, search.best_params_, search.best_score_
            
        except Exception as e:
            if verbose:
                print(f"[ERROR] Tuning failed: {str(e)}")
            return model, {}, 0.0
    
    def tune_multiple_models(self, models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                           method: str = 'random', cv_folds: int = 5, n_iter: int = 20,
                           verbose: bool = True) -> pd.DataFrame:
        """
        Tune hyperparameters for multiple models
        
        Args:
            models: Dictionary of model_name: model_instance
            X_train: Training features
            y_train: Training target
            method: 'grid' or 'random'
            cv_folds: Number of cross-validation folds
            n_iter: Number of iterations for random search
            verbose: Whether to print progress
            
        Returns:
            DataFrame with tuning results comparison
        """
        results = []
        
        for model_name, model in models.items():
            best_estimator, best_params, best_score = self.tune_model(
                model, model_name, X_train, y_train,
                method=method, cv_folds=cv_folds, n_iter=n_iter, verbose=verbose
            )
            
            if best_score > 0:
                results.append({
                    'model_name': model_name,
                    'best_score': best_score,
                    'tuning_time': self.tuning_results[model_name]['tuning_time'],
                    'n_iterations': self.tuning_results[model_name]['n_iterations']
                })
        
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results).sort_values('best_score', ascending=False)
        
        if verbose:
            print("\n" + "="*80)
            print("HYPERPARAMETER TUNING SUMMARY")
            print("="*80)
            print(results_df.to_string(index=False))
            print("="*80)
        
        return results_df
    
    def get_tuned_model(self, model_name: str) -> Optional[Any]:
        """
        Get the best tuned model for a specific model name
        
        Args:
            model_name: Name of the model
            
        Returns:
            Best estimator or None if not found
        """
        return self.best_estimators.get(model_name)
    
    def get_best_params(self, model_name: str) -> Optional[Dict]:
        """
        Get the best parameters for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of best parameters or None
        """
        return self.best_params.get(model_name)
    
    def compare_before_after(self, model_name: str, default_score: float) -> Dict[str, Any]:
        """
        Compare performance before and after tuning
        
        Args:
            model_name: Name of the model
            default_score: Score before tuning
            
        Returns:
            Dictionary with comparison metrics
        """
        if model_name not in self.tuning_results:
            return {}
        
        tuned_score = self.tuning_results[model_name]['best_score']
        improvement = tuned_score - default_score
        improvement_pct = (improvement / default_score) * 100 if default_score > 0 else 0
        
        return {
            'model_name': model_name,
            'default_score': default_score,
            'tuned_score': tuned_score,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }


def quick_tune_best_model(model, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                         task_type: str = 'classification', method: str = 'random',
                         cv_folds: int = 5, n_iter: int = 20) -> Tuple[Any, Dict, Dict]:
    """
    Convenience function to quickly tune the best model
    
    Args:
        model: Model instance to tune
        model_name: Name of the model
        X_train: Training features
        y_train: Training target
        task_type: 'classification' or 'regression'
        method: 'grid' or 'random'
        cv_folds: Number of cross-validation folds
        n_iter: Number of iterations for random search
        
    Returns:
        Tuple of (tuned_model, best_params, tuning_info)
    """
    tuner = HyperparameterTuner(task_type=task_type)
    
    X_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_array = y_train.values if isinstance(y_train, pd.Series) else y_train
    
    best_estimator, best_params, best_score = tuner.tune_model(
        model, model_name, X_array, y_array,
        method=method, cv_folds=cv_folds, n_iter=n_iter, verbose=True
    )
    
    tuning_info = tuner.tuning_results.get(model_name, {})
    
    return best_estimator, best_params, tuning_info


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    print("Hyperparameter Tuning Module for AIDEX")
    print("=" * 60)
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test tuning
    model = RandomForestClassifier(random_state=42)
    tuned_model, best_params, tuning_info = quick_tune_best_model(
        model, 'Random Forest', X_train, y_train,
        task_type='classification', method='random', n_iter=10
    )
    
    print(f"\n[OK] Tuning completed!")
    print(f"Best score: {tuning_info.get('best_score', 0):.4f}")
    print(f"Best parameters: {best_params}")
