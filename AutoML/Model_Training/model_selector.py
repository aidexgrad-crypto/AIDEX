"""
Model Selection Module for AIDEX
Intelligently selects the best model based on multiple criteria and business requirements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class ModelSelector:
    """
    Intelligent model selection based on multiple criteria including:
    - Performance metrics
    - Model complexity
    - Training/inference time
    - Overfitting tendency
    - Cross-validation stability
    - Business requirements
    """
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize the Model Selector
        
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type.lower()
        self.selection_criteria = {}
        self.selected_model = None
        self.selection_report = {}
        
    def calculate_complexity_score(self, model_name: str) -> float:
        """
        Assign complexity scores to different model types
        Lower score = less complex (more interpretable)
        
        Args:
            model_name: Name of the model
            
        Returns:
            Complexity score (0-1)
        """
        complexity_map = {
            'logistic regression': 0.1,
            'linear regression': 0.1,
            'ridge regression': 0.15,
            'lasso regression': 0.15,
            'elasticnet': 0.15,
            'naive bayes': 0.2,
            'decision tree': 0.3,
            'k-nearest neighbors': 0.4,
            # 'support vector machine': 0.5,  # Removed - too slow for large datasets
            'random forest': 0.6,
            'extra trees': 0.6,
            'adaboost': 0.7,
            'gradient boosting': 0.75,
            'xgboost': 0.8,
            'lightgbm': 0.8
        }
        
        model_name_lower = model_name.lower()
        for key, score in complexity_map.items():
            if key in model_name_lower:
                return score
        
        return 0.5  # Default complexity
    
    def calculate_stability_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate stability based on cross-validation standard deviation
        Lower std = more stable
        
        Args:
            metrics: Dictionary of metrics including std values
            
        Returns:
            Stability score (0-1, higher is better)
        """
        if self.task_type == 'classification':
            std_key = 'f1_std'
        else:
            std_key = 'r2_std'
        
        if std_key not in metrics:
            return 0.5
        
        std_value = metrics[std_key]
        # Convert std to stability score (lower std = higher stability)
        # Assume std > 0.2 is very unstable, std < 0.01 is very stable
        stability = max(0, min(1, 1 - (std_value / 0.2)))
        
        return stability
    
    def calculate_speed_score(self, training_time: float, max_time: float = None) -> float:
        """
        Calculate speed score based on training time
        
        Args:
            training_time: Training time in seconds
            max_time: Maximum training time observed (for normalization)
            
        Returns:
            Speed score (0-1, higher is faster)
        """
        if max_time is None or max_time == 0:
            return 0.5
        
        # Normalize by max time and invert (faster = higher score)
        speed_score = 1 - (training_time / max_time)
        return max(0, min(1, speed_score))
    
    def calculate_overfitting_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overfitting penalty
        
        Args:
            metrics: Dictionary of metrics including overfitting indicator
            
        Returns:
            Overfitting score (0-1, higher is less overfitting)
        """
        overfitting = metrics.get('overfitting', 0)
        
        # Overfitting > 0.15 is concerning
        if overfitting < 0.05:
            return 1.0
        elif overfitting < 0.1:
            return 0.8
        elif overfitting < 0.15:
            return 0.6
        elif overfitting < 0.25:
            return 0.4
        else:
            return 0.2
    
    def select_by_performance(self, results_df: pd.DataFrame, 
                             top_n: int = 3) -> pd.DataFrame:
        """
        Select top N models by primary performance metric
        
        Args:
            results_df: DataFrame with model results
            top_n: Number of top models to return
            
        Returns:
            DataFrame with top N models
        """
        if self.task_type == 'classification':
            metric_col = 'f1_mean'
        else:
            metric_col = 'r2_mean'
        
        if metric_col not in results_df.columns:
            raise ValueError(f"Required metric '{metric_col}' not found in results.")
        
        top_models = results_df.nlargest(top_n, metric_col)
        return top_models
    
    def select_by_criteria(self, results_df: pd.DataFrame,
                          weights: Dict[str, float] = None,
                          min_performance: float = None) -> Tuple[str, Dict[str, float]]:
        """
        Select model based on weighted criteria
        
        Args:
            results_df: DataFrame with model results
            weights: Dictionary of weights for different criteria
                    {
                        'performance': 0.5,
                        'stability': 0.2,
                        'speed': 0.1,
                        'simplicity': 0.1,
                        'no_overfitting': 0.1
                    }
            min_performance: Minimum acceptable performance score
            
        Returns:
            Tuple of (selected_model_name, selection_scores)
        """
        # Default weights
        if weights is None:
            weights = {
                'performance': 0.5,
                'stability': 0.2,
                'speed': 0.1,
                'simplicity': 0.1,
                'no_overfitting': 0.1
            }
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Get primary metric
        if self.task_type == 'classification':
            perf_metric = 'f1_mean'
        else:
            perf_metric = 'r2_mean'
        
        # Calculate max time for normalization
        max_time = results_df['training_time'].max()
        
        # Calculate composite scores
        composite_scores = []
        
        for idx, row in results_df.iterrows():
            model_name = row['model_name']
            
            # Performance score (normalized)
            perf_score = row[perf_metric]
            if min_performance and perf_score < min_performance:
                continue  # Skip models below minimum threshold
            
            # Normalize performance to 0-1 range
            max_perf = results_df[perf_metric].max()
            min_perf = results_df[perf_metric].min()
            if max_perf > min_perf:
                perf_score_norm = (perf_score - min_perf) / (max_perf - min_perf)
            else:
                perf_score_norm = 1.0
            
            # Other scores
            stability_score = self.calculate_stability_score(row)
            speed_score = self.calculate_speed_score(row['training_time'], max_time)
            simplicity_score = 1 - self.calculate_complexity_score(model_name)
            no_overfit_score = self.calculate_overfitting_score(row)
            
            # Calculate weighted composite score
            composite = (
                weights.get('performance', 0) * perf_score_norm +
                weights.get('stability', 0) * stability_score +
                weights.get('speed', 0) * speed_score +
                weights.get('simplicity', 0) * simplicity_score +
                weights.get('no_overfitting', 0) * no_overfit_score
            )
            
            composite_scores.append({
                'model_name': model_name,
                'composite_score': composite,
                'performance_score': perf_score_norm,
                'stability_score': stability_score,
                'speed_score': speed_score,
                'simplicity_score': simplicity_score,
                'no_overfitting_score': no_overfit_score,
                'raw_performance': perf_score
            })
        
        if not composite_scores:
            raise ValueError("No models meet the minimum performance threshold.")
        
        # Select best composite score
        best = max(composite_scores, key=lambda x: x['composite_score'])
        
        self.selected_model = best['model_name']
        self.selection_report = best
        
        return best['model_name'], best
    
    def select_by_business_priority(self, results_df: pd.DataFrame,
                                    priority: str = 'balanced',
                                    min_performance: float = None) -> Tuple[str, Dict[str, float]]:
        """
        Select model based on business priorities
        
        Args:
            results_df: DataFrame with model results
            priority: Business priority
                     'performance' - maximize accuracy/score
                     'speed' - fast training and inference
                     'interpretability' - simple, explainable models
                     'stable' - consistent across different data splits
                     'balanced' - balance all factors
            min_performance: Minimum acceptable performance
            
        Returns:
            Tuple of (selected_model_name, selection_scores)
        """
        priority_weights = {
            'performance': {
                'performance': 0.8,
                'stability': 0.1,
                'speed': 0.05,
                'simplicity': 0.0,
                'no_overfitting': 0.05
            },
            'speed': {
                'performance': 0.3,
                'stability': 0.1,
                'speed': 0.5,
                'simplicity': 0.05,
                'no_overfitting': 0.05
            },
            'interpretability': {
                'performance': 0.3,
                'stability': 0.1,
                'speed': 0.1,
                'simplicity': 0.4,
                'no_overfitting': 0.1
            },
            'stable': {
                'performance': 0.4,
                'stability': 0.4,
                'speed': 0.05,
                'simplicity': 0.05,
                'no_overfitting': 0.1
            },
            'balanced': {
                'performance': 0.4,
                'stability': 0.2,
                'speed': 0.15,
                'simplicity': 0.15,
                'no_overfitting': 0.1
            }
        }
        
        weights = priority_weights.get(priority, priority_weights['balanced'])
        
        return self.select_by_criteria(results_df, weights=weights, 
                                      min_performance=min_performance)
    
    def compare_models(self, results_df: pd.DataFrame, 
                      model_names: List[str]) -> pd.DataFrame:
        """
        Detailed comparison of specific models
        
        Args:
            results_df: DataFrame with model results
            model_names: List of model names to compare
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for model_name in model_names:
            model_row = results_df[results_df['model_name'] == model_name]
            if model_row.empty:
                continue
            
            row = model_row.iloc[0]
            
            comparison_data.append({
                'model': model_name,
                'performance': row[f"{'f1' if self.task_type == 'classification' else 'r2'}_mean"],
                'stability': self.calculate_stability_score(row),
                'speed': self.calculate_speed_score(row['training_time'], results_df['training_time'].max()),
                'simplicity': 1 - self.calculate_complexity_score(model_name),
                'no_overfitting': self.calculate_overfitting_score(row),
                'training_time': row['training_time']
            })
        
        return pd.DataFrame(comparison_data)
    
    def recommend_ensemble(self, results_df: pd.DataFrame, 
                          top_n: int = 3,
                          diversity_threshold: float = 0.3) -> List[str]:
        """
        Recommend models for ensemble based on performance and diversity
        
        Args:
            results_df: DataFrame with model results
            top_n: Number of models to recommend
            diversity_threshold: Minimum complexity difference for diversity
            
        Returns:
            List of recommended model names for ensemble
        """
        # Get top performers
        top_models = self.select_by_performance(results_df, top_n=top_n * 2)
        
        # Select diverse models
        selected = []
        complexity_scores = []
        
        for _, row in top_models.iterrows():
            model_name = row['model_name']
            complexity = self.calculate_complexity_score(model_name)
            
            # Check diversity
            if not complexity_scores or all(abs(complexity - c) >= diversity_threshold for c in complexity_scores):
                selected.append(model_name)
                complexity_scores.append(complexity)
                
                if len(selected) >= top_n:
                    break
        
        # If not enough diverse models, add best remaining
        if len(selected) < top_n:
            for _, row in top_models.iterrows():
                model_name = row['model_name']
                if model_name not in selected:
                    selected.append(model_name)
                    if len(selected) >= top_n:
                        break
        
        return selected
    
    def get_selection_report(self) -> Dict[str, Any]:
        """
        Get detailed report of the selection process
        
        Returns:
            Dictionary with selection details
        """
        return {
            'selected_model': self.selected_model,
            'selection_scores': self.selection_report,
            'task_type': self.task_type
        }
    
    def print_selection_summary(self, results_df: pd.DataFrame, 
                               selected_model: str = None,
                               show_top_n: int = 5):
        """
        Print a formatted summary of model selection
        
        Args:
            results_df: DataFrame with model results
            selected_model: Name of selected model (uses self.selected_model if None)
            show_top_n: Number of top models to show
        """
        if selected_model is None:
            selected_model = self.selected_model
        
        print("\n" + "="*80)
        print("MODEL SELECTION SUMMARY")
        print("="*80)
        
        # Show top N models
        print(f"\nTop {show_top_n} Models by Performance:")
        print("-"*80)
        
        if self.task_type == 'classification':
            metric_cols = ['model_name', 'accuracy_mean', 'f1_mean', 'training_time']
            primary_metric = 'f1_mean'
        else:
            metric_cols = ['model_name', 'r2_mean', 'rmse_mean', 'training_time']
            primary_metric = 'r2_mean'
        
        display_cols = [col for col in metric_cols if col in results_df.columns]
        top_models = results_df.nlargest(show_top_n, primary_metric)[display_cols]
        print(top_models.to_string(index=False))
        
        # Show selected model details
        if selected_model:
            print(f"\n{'='*80}")
            print(f"SELECTED MODEL: {selected_model}")
            print("="*80)
            
            model_row = results_df[results_df['model_name'] == selected_model].iloc[0]
            
            print(f"\nPerformance Metrics:")
            if self.task_type == 'classification':
                print(f"  Accuracy: {model_row.get('accuracy_mean', 'N/A'):.4f} +/- {model_row.get('accuracy_std', 0):.4f}")
                print(f"  F1 Score: {model_row.get('f1_mean', 'N/A'):.4f} +/- {model_row.get('f1_std', 0):.4f}")
                print(f"  Precision: {model_row.get('precision_mean', 'N/A'):.4f} +/- {model_row.get('precision_std', 0):.4f}")
                print(f"  Recall: {model_row.get('recall_mean', 'N/A'):.4f} +/- {model_row.get('recall_std', 0):.4f}")
                if 'roc_auc_mean' in model_row:
                    print(f"  ROC AUC: {model_row['roc_auc_mean']:.4f} +/- {model_row.get('roc_auc_std', 0):.4f}")
            else:
                print(f"  R² Score: {model_row.get('r2_mean', 'N/A'):.4f} +/- {model_row.get('r2_std', 0):.4f}")
                print(f"  RMSE: {model_row.get('rmse_mean', 'N/A'):.4f} +/- {model_row.get('rmse_std', 0):.4f}")
                print(f"  MAE: {model_row.get('mae_mean', 'N/A'):.4f} +/- {model_row.get('mae_std', 0):.4f}")
            
            print(f"\nModel Characteristics:")
            print(f"  Training Time: {model_row.get('training_time', 'N/A'):.2f} seconds")
            print(f"  Overfitting: {model_row.get('overfitting', 'N/A'):.4f}")
            print(f"  Complexity Score: {self.calculate_complexity_score(selected_model):.2f}")
            print(f"  Stability Score: {self.calculate_stability_score(model_row):.2f}")
            
            if self.selection_report:
                print(f"\nSelection Scores:")
                print(f"  Composite Score: {self.selection_report.get('composite_score', 'N/A'):.4f}")
                print(f"  Performance: {self.selection_report.get('performance_score', 'N/A'):.4f}")
                print(f"  Stability: {self.selection_report.get('stability_score', 'N/A'):.4f}")
                print(f"  Speed: {self.selection_report.get('speed_score', 'N/A'):.4f}")
                print(f"  Simplicity: {self.selection_report.get('simplicity_score', 'N/A'):.4f}")
        
        print("\n" + "="*80)


def auto_select_model(results_df: pd.DataFrame, 
                     task_type: str = 'classification',
                     priority: str = 'balanced',
                     min_performance: float = None,
                     verbose: bool = True) -> Tuple[str, ModelSelector]:
    """
    Convenience function for automatic model selection
    
    Args:
        results_df: DataFrame with model results from ModelTrainer
        task_type: 'classification' or 'regression'
        priority: 'performance', 'speed', 'interpretability', 'stable', or 'balanced'
        min_performance: Minimum acceptable performance threshold
        verbose: Whether to print selection summary
        
    Returns:
        Tuple of (selected_model_name, selector)
    """
    selector = ModelSelector(task_type=task_type)
    
    selected_model, scores = selector.select_by_business_priority(
        results_df, 
        priority=priority,
        min_performance=min_performance
    )
    
    if verbose:
        selector.print_selection_summary(results_df, selected_model)
    
    return selected_model, selector


if __name__ == "__main__":
    # Example usage
    print("Model Selection Module for AIDEX")
    print("=" * 60)
    
    # Create sample results
    sample_results = pd.DataFrame({
        'model_name': ['Random Forest', 'XGBoost', 'Logistic Regression', 'Decision Tree', 'Naive Bayes'],
        'f1_mean': [0.89, 0.91, 0.85, 0.82, 0.88],
        'f1_std': [0.02, 0.04, 0.01, 0.05, 0.03],
        'accuracy_mean': [0.88, 0.90, 0.84, 0.80, 0.87],
        'accuracy_std': [0.02, 0.03, 0.01, 0.06, 0.03],
        'precision_mean': [0.87, 0.89, 0.83, 0.79, 0.86],
        'recall_mean': [0.90, 0.92, 0.86, 0.84, 0.89],
        'training_time': [5.2, 8.5, 1.2, 0.8, 15.3],
        'overfitting': [0.08, 0.12, 0.03, 0.18, 0.10]
    })
    
    # Test different selection strategies
    selector = ModelSelector(task_type='classification')
    
    print("\n--- Selecting by Performance Priority ---")
    model, scores = selector.select_by_business_priority(sample_results, priority='performance')
    print(f"Selected: {model} (Score: {scores['composite_score']:.4f})")
    
    print("\n--- Selecting by Interpretability Priority ---")
    model, scores = selector.select_by_business_priority(sample_results, priority='interpretability')
    print(f"Selected: {model} (Score: {scores['composite_score']:.4f})")
    
    print("\n--- Selecting by Speed Priority ---")
    model, scores = selector.select_by_business_priority(sample_results, priority='speed')
    print(f"Selected: {model} (Score: {scores['composite_score']:.4f})")
    
    print("\n--- Ensemble Recommendation ---")
    ensemble = selector.recommend_ensemble(sample_results, top_n=3)
    print(f"Recommended ensemble: {ensemble}")
