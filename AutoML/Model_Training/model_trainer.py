"""
Model Training & Evaluation Module for AIDEX
Automatically trains, evaluates, and selects the best machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_curve
)

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from typing import Dict, List, Tuple, Optional, Any
import warnings
import time
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Automated machine learning model trainer and evaluator.
    Trains multiple models, evaluates performance, and selects the best model.
    """
    
    def __init__(self, task_type: str = 'classification', random_state: int = 42):
        """
        Initialize the Model Trainer
        
        Args:
            task_type: 'classification' or 'regression'
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type.lower()
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        self.trained_models = {}
        
    def get_default_models(self) -> Dict[str, Any]:
        """
        Get default models based on task type
        
        Returns:
            Dictionary of model name to model instance
        """
        if self.task_type == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
                'Random Forest': RandomForestClassifier(random_state=self.random_state, n_estimators=100),
                'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state, n_estimators=100),
                'XGBoost': XGBClassifier(random_state=self.random_state, n_estimators=100, eval_metric='logloss'),
                'LightGBM': LGBMClassifier(random_state=self.random_state, n_estimators=100, verbose=-1),
                'Extra Trees': ExtraTreesClassifier(random_state=self.random_state, n_estimators=100),
                'AdaBoost': AdaBoostClassifier(random_state=self.random_state, n_estimators=100),
                'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
                'Naive Bayes': GaussianNB(),
                # 'Support Vector Machine': SVC(random_state=self.random_state, probability=True)  # Temporarily disabled - too slow for large datasets
            }
        else:  # regression
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(random_state=self.random_state),
                'Lasso Regression': Lasso(random_state=self.random_state),
                'ElasticNet': ElasticNet(random_state=self.random_state),
                'Decision Tree': DecisionTreeRegressor(random_state=self.random_state),
                'Random Forest': RandomForestRegressor(random_state=self.random_state, n_estimators=100),
                'Gradient Boosting': GradientBoostingRegressor(random_state=self.random_state, n_estimators=100),
                'XGBoost': XGBRegressor(random_state=self.random_state, n_estimators=100),
                'LightGBM': LGBMRegressor(random_state=self.random_state, n_estimators=100, verbose=-1),
                'Extra Trees': ExtraTreesRegressor(random_state=self.random_state, n_estimators=100),
                'AdaBoost': AdaBoostRegressor(random_state=self.random_state, n_estimators=100),
                'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
                'Support Vector Machine': SVR()
            }
        
        return models
    
    def evaluate_classification_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                                     cv_folds: int = 5) -> Dict[str, float]:
        """
        Evaluate a classification model using cross-validation
        
        Args:
            model: Sklearn-compatible model
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of evaluation metrics
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted'
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(y)) == 2:
            scoring['roc_auc'] = 'roc_auc'
        
        # NOTE (Windows compatibility): parallel CV can trigger joblib/loky CPU
        # introspection that relies on deprecated tools like WMIC. To keep AIDEX
        # stable across Windows installs, default to single-process CV.
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=1,
        )
        
        metrics = {
            'accuracy_mean': cv_results['test_accuracy'].mean(),
            'accuracy_std': cv_results['test_accuracy'].std(),
            'precision_mean': cv_results['test_precision'].mean(),
            'precision_std': cv_results['test_precision'].std(),
            'recall_mean': cv_results['test_recall'].mean(),
            'recall_std': cv_results['test_recall'].std(),
            'f1_mean': cv_results['test_f1'].mean(),
            'f1_std': cv_results['test_f1'].std(),
            'train_accuracy_mean': cv_results['train_accuracy'].mean(),
        }
        
        if 'test_roc_auc' in cv_results:
            metrics['roc_auc_mean'] = cv_results['test_roc_auc'].mean()
            metrics['roc_auc_std'] = cv_results['test_roc_auc'].std()
        
        # Calculate overfitting indicator
        metrics['overfitting'] = metrics['train_accuracy_mean'] - metrics['accuracy_mean']
        
        return metrics
    
    def evaluate_regression_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                                  cv_folds: int = 5) -> Dict[str, float]:
        """
        Evaluate a regression model using cross-validation
        
        Args:
            model: Sklearn-compatible model
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of evaluation metrics
        """
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        scoring = {
            'r2': 'r2',
            'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error',
            'neg_rmse': 'neg_root_mean_squared_error'
        }
        
        # See note above: keep regression CV single-process for Windows stability.
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=1,
        )
        
        metrics = {
            'r2_mean': cv_results['test_r2'].mean(),
            'r2_std': cv_results['test_r2'].std(),
            'mse_mean': -cv_results['test_neg_mse'].mean(),
            'mse_std': cv_results['test_neg_mse'].std(),
            'rmse_mean': -cv_results['test_neg_rmse'].mean(),
            'rmse_std': cv_results['test_neg_rmse'].std(),
            'mae_mean': -cv_results['test_neg_mae'].mean(),
            'mae_std': cv_results['test_neg_mae'].std(),
            'train_r2_mean': cv_results['train_r2'].mean(),
        }
        
        # Calculate overfitting indicator
        metrics['overfitting'] = metrics['train_r2_mean'] - metrics['r2_mean']
        
        return metrics
    
    def train_and_evaluate_all(self, X: pd.DataFrame, y: pd.Series, 
                              cv_folds: int = 5, custom_models: Dict = None,
                              verbose: bool = True) -> pd.DataFrame:
        """
        Train and evaluate all models
        
        Args:
            X: Feature dataframe
            y: Target series
            cv_folds: Number of cross-validation folds
            custom_models: Optional dictionary of custom models to include
            verbose: Whether to print progress
            
        Returns:
            DataFrame with model performance comparison
        """
        # Get default models
        self.models = self.get_default_models()
        
        # Add custom models if provided
        if custom_models:
            self.models.update(custom_models)
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        results_list = []
        
        for model_name, model in self.models.items():
            if verbose:
                print(f"Training {model_name}...", end=' ')
            
            start_time = time.time()
            
            try:
                # Evaluate model
                if self.task_type == 'classification':
                    metrics = self.evaluate_classification_model(model, X_array, y_array, cv_folds)
                    primary_metric = metrics['f1_mean']
                else:
                    metrics = self.evaluate_regression_model(model, X_array, y_array, cv_folds)
                    primary_metric = metrics['r2_mean']
                
                training_time = time.time() - start_time
                metrics['training_time'] = training_time
                metrics['model_name'] = model_name
                
                # Train on full dataset for final model
                model.fit(X_array, y_array)
                self.trained_models[model_name] = model
                
                self.results[model_name] = metrics
                results_list.append(metrics)
                
                # Track best model
                if primary_metric > self.best_score:
                    self.best_score = primary_metric
                    self.best_model = model
                    self.best_model_name = model_name
                
                if verbose:
                    print(f"[OK] (Score: {primary_metric:.4f}, Time: {training_time:.2f}s)")
                
            except Exception as e:
                if verbose:
                    print(f"[ERROR] (Error: {str(e)})")
                continue
        
        # Create results dataframe
        results_df = pd.DataFrame(results_list)
        
        # Sort by primary metric
        if self.task_type == 'classification':
            results_df = results_df.sort_values('f1_mean', ascending=False)
        else:
            results_df = results_df.sort_values('r2_mean', ascending=False)
        
        return results_df
    
    def test_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Test a specific trained model on test data
        
        Args:
            model_name: Name of the model to test
            X_test: Test feature dataframe
            y_test: Test target series
            
        Returns:
            Dictionary with test metrics and predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' has not been trained yet.")
        
        model = self.trained_models[model_name]
        X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        y_test_array = y_test.values if isinstance(y_test, pd.Series) else y_test
        
        # Make predictions
        y_pred = model.predict(X_test_array)
        
        test_results = {
            'model_name': model_name,
            'predictions': y_pred
        }
        
        if self.task_type == 'classification':
            # Classification metrics
            test_results['accuracy'] = accuracy_score(y_test_array, y_pred)
            test_results['precision'] = precision_score(y_test_array, y_pred, average='weighted', zero_division=0)
            test_results['recall'] = recall_score(y_test_array, y_pred, average='weighted', zero_division=0)
            test_results['f1'] = f1_score(y_test_array, y_pred, average='weighted', zero_division=0)
            test_results['confusion_matrix'] = confusion_matrix(y_test_array, y_pred)
            
            # ROC AUC for binary classification
            if len(np.unique(y_test_array)) == 2 and hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_array)[:, 1]
                test_results['roc_auc'] = roc_auc_score(y_test_array, y_proba)
                test_results['probabilities'] = y_proba
            
            # Classification report
            test_results['classification_report'] = classification_report(y_test_array, y_pred)
            
        else:  # regression
            # Regression metrics
            test_results['r2'] = r2_score(y_test_array, y_pred)
            test_results['mse'] = mean_squared_error(y_test_array, y_pred)
            test_results['rmse'] = np.sqrt(test_results['mse'])
            test_results['mae'] = mean_absolute_error(y_test_array, y_pred)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mask = y_test_array != 0
            if mask.sum() > 0:
                test_results['mape'] = np.mean(np.abs((y_test_array[mask] - y_pred[mask]) / y_test_array[mask])) * 100
        
        return test_results
    
    def test_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Test all trained models on test data
        
        Args:
            X_test: Test feature dataframe
            y_test: Test target series
            
        Returns:
            DataFrame with test performance for all models
        """
        test_results_list = []
        
        for model_name in self.trained_models.keys():
            results = self.test_model(model_name, X_test, y_test)
            
            # Extract only numeric metrics for dataframe
            if self.task_type == 'classification':
                test_results_list.append({
                    'model_name': model_name,
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1': results['f1'],
                    'roc_auc': results.get('roc_auc', None)
                })
            else:
                test_results_list.append({
                    'model_name': model_name,
                    'r2': results['r2'],
                    'mse': results['mse'],
                    'rmse': results['rmse'],
                    'mae': results['mae'],
                    'mape': results.get('mape', None)
                })
        
        test_results_df = pd.DataFrame(test_results_list)
        
        # Sort by primary metric
        if self.task_type == 'classification':
            test_results_df = test_results_df.sort_values('f1', ascending=False)
        else:
            test_results_df = test_results_df.sort_values('r2', ascending=False)
        
        return test_results_df
    
    def get_best_model(self) -> Tuple[str, Any, float]:
        """
        Get the best performing model
        
        Returns:
            Tuple of (model_name, model, score)
        """
        return self.best_model_name, self.best_model, self.best_score
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a specific trained model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Trained model instance
        """
        return self.trained_models.get(model_name)
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        Make predictions using a trained model
        
        Args:
            X: Feature dataframe
            model_name: Name of model to use (uses best model if None)
            
        Returns:
            Array of predictions
        """
        if model_name is None:
            model = self.best_model
        else:
            model = self.trained_models.get(model_name)
        
        if model is None:
            raise ValueError("No trained model available for prediction.")
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return model.predict(X_array)
    
    def predict_proba(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        Get prediction probabilities (classification only)
        
        Args:
            X: Feature dataframe
            model_name: Name of model to use (uses best model if None)
            
        Returns:
            Array of prediction probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks.")
        
        if model_name is None:
            model = self.best_model
        else:
            model = self.trained_models.get(model_name)
        
        if model is None:
            raise ValueError("No trained model available for prediction.")
        
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model does not support probability predictions.")
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return model.predict_proba(X_array)


def auto_train_and_select(X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame = None, y_test: pd.Series = None,
                         task_type: str = 'classification', cv_folds: int = 5,
                         verbose: bool = True) -> Tuple[ModelTrainer, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Convenience function for automatic model training and selection
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Optional test features
        y_test: Optional test target
        task_type: 'classification' or 'regression'
        cv_folds: Number of cross-validation folds
        verbose: Whether to print progress
        
    Returns:
        Tuple of (trainer, cv_results, test_results)
    """
    trainer = ModelTrainer(task_type=task_type)
    
    # Train and evaluate with cross-validation
    cv_results = trainer.train_and_evaluate_all(X_train, y_train, cv_folds=cv_folds, verbose=verbose)
    
    # Test on holdout set if provided
    test_results = None
    if X_test is not None and y_test is not None:
        if verbose:
            print("\nTesting on holdout set...")
        test_results = trainer.test_all_models(X_test, y_test)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Best Model: {trainer.best_model_name}")
        print(f"Best Score: {trainer.best_score:.4f}")
        print(f"{'='*60}")
    
    return trainer, cv_results, test_results


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification, make_regression
    
    print("Model Training & Evaluation Module for AIDEX")
    print("=" * 60)
    
    # Classification example
    print("\n--- CLASSIFICATION EXAMPLE ---")
    X_class, y_class = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                          n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
    
    X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    X_test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    y_train_series = pd.Series(y_train, name='target')
    y_test_series = pd.Series(y_test, name='target')
    
    trainer, cv_results, test_results = auto_train_and_select(
        X_train_df, y_train_series,
        X_test_df, y_test_series,
        task_type='classification',
        cv_folds=5
    )
    
    print("\nCross-Validation Results:")
    print(cv_results[['model_name', 'accuracy_mean', 'f1_mean', 'training_time']].head())
    
    print("\nTest Results:")
    print(test_results[['model_name', 'accuracy', 'f1', 'roc_auc']].head())
    
    # Make predictions with best model
    predictions = trainer.predict(X_test_df)
    print(f"\nSample predictions: {predictions[:10]}")
