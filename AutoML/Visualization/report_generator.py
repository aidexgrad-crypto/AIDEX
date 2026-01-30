"""
AIDEX Report and Visualization Generator
Generates comprehensive reports and visualizations for ML models
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import os
from datetime import datetime
import base64
from io import BytesIO

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class ReportGenerator:
    """Generate comprehensive ML reports with visualizations"""
    
    def __init__(self, pipeline, task_type: str, project_name: str):
        """
        Initialize report generator
        
        Args:
            pipeline: Trained AIDEX pipeline
            task_type: 'classification' or 'regression'
            project_name: Name of the project
        """
        self.pipeline = pipeline
        self.task_type = task_type
        self.project_name = project_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return image_base64
    
    def generate_model_comparison_chart(self) -> str:
        """Generate model comparison bar chart"""
        if self.task_type == 'classification':
            metric = 'f1'
            metric_label = 'F1 Score'
        else:
            metric = 'r2'
            metric_label = 'R² Score'
        
        # Get test results
        test_results = self.pipeline.test_results.sort_values(metric, ascending=False).head(6)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#10b981' if i == 0 else '#3b82f6' for i in range(len(test_results))]
        
        bars = ax.barh(test_results['model_name'], test_results[metric], color=colors)
        ax.set_xlabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(f'Model Performance Comparison - {metric_label}', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, test_results[metric])):
            ax.text(value, bar.get_y() + bar.get_height()/2, 
                   f' {value:.4f}', va='center', fontweight='bold')
        
        # Add best model star
        ax.text(test_results[metric].iloc[0], 0, ' ★', 
               va='center', fontsize=16, color='#fbbf24')
        
        plt.tight_layout()
        return self.plot_to_base64(fig)
    
    def generate_predictions_chart(self) -> str:
        """Generate actual vs predicted chart"""
        best_model = self.pipeline.trainer.models[self.pipeline.best_model_name]
        
        if self.task_type == 'regression':
            # Get predictions on original scale
            if hasattr(self.pipeline, 'target_scaler') and self.pipeline.target_scaler is not None:
                predictions_scaled = best_model.predict(self.pipeline.X_test)
                predictions = self.pipeline.target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
                actuals = self.pipeline.y_test_original.values
            else:
                predictions = best_model.predict(self.pipeline.X_test)
                actuals = self.pipeline.y_test.values
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Scatter plot
            ax1.scatter(actuals, predictions, alpha=0.6, s=50, color='#3b82f6')
            
            # Perfect prediction line
            min_val = min(actuals.min(), predictions.min())
            max_val = max(actuals.max(), predictions.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 
                    'r--', lw=2, label='Perfect Prediction')
            
            ax1.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
            ax1.set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Residual plot
            residuals = actuals - predictions
            ax2.scatter(predictions, residuals, alpha=0.6, s=50, color='#10b981')
            ax2.axhline(y=0, color='r', linestyle='--', lw=2)
            ax2.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
            ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
        else:  # classification
            from sklearn.metrics import confusion_matrix
            predictions = best_model.predict(self.pipeline.X_test)
            actuals = self.pipeline.y_test.values
            
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(actuals, predictions)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                       cbar_kws={'label': 'Count'})
            ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return self.plot_to_base64(fig)
    
    def generate_feature_importance_chart(self) -> Optional[str]:
        """Generate feature importance chart (if model supports it)"""
        best_model = self.pipeline.trainer.models[self.pipeline.best_model_name]
        
        # Check if model has feature importance
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_names = self.pipeline.X_train.columns
            
            # Get top 15 features
            indices = np.argsort(importances)[-15:]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
            
            ax.barh(range(len(indices)), importances[indices], color=colors)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
            ax.set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            return self.plot_to_base64(fig)
        
        return None
    
    def generate_data_distribution_chart(self) -> str:
        """Generate target variable distribution chart"""
        if self.task_type == 'regression':
            # Use original scale for regression
            if hasattr(self.pipeline, 'y_train_original'):
                y_train = self.pipeline.y_train_original
                y_test = self.pipeline.y_test_original
            else:
                y_train = self.pipeline.y_train
                y_test = self.pipeline.y_test
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Distribution plot
            ax1.hist(y_train, bins=30, alpha=0.7, color='#3b82f6', label='Train', edgecolor='black')
            ax1.hist(y_test, bins=30, alpha=0.7, color='#10b981', label='Test', edgecolor='black')
            ax1.set_xlabel('Target Value', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax1.set_title('Target Distribution', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot([y_train, y_test], labels=['Train', 'Test'],
                       patch_artist=True,
                       boxprops=dict(facecolor='#3b82f6', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
            ax2.set_ylabel('Target Value', fontsize=12, fontweight='bold')
            ax2.set_title('Target Distribution (Box Plot)', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
        else:  # classification
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Train distribution
            train_counts = self.pipeline.y_train.value_counts()
            ax1.bar(range(len(train_counts)), train_counts.values, color='#3b82f6', alpha=0.7, edgecolor='black')
            ax1.set_xticks(range(len(train_counts)))
            ax1.set_xticklabels(train_counts.index)
            ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax1.set_title('Training Set Class Distribution', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Test distribution
            test_counts = self.pipeline.y_test.value_counts()
            ax2.bar(range(len(test_counts)), test_counts.values, color='#10b981', alpha=0.7, edgecolor='black')
            ax2.set_xticks(range(len(test_counts)))
            ax2.set_xticklabels(test_counts.index)
            ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax2.set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return self.plot_to_base64(fig)
    
    def generate_html_report(self) -> str:
        """Generate comprehensive HTML report"""
        # Generate all charts
        model_comparison = self.generate_model_comparison_chart()
        predictions_chart = self.generate_predictions_chart()
        feature_importance = self.generate_feature_importance_chart()
        distribution_chart = self.generate_data_distribution_chart()
        
        # Get metrics
        best_results = self.pipeline.test_results[
            self.pipeline.test_results['model_name'] == self.pipeline.best_model_name
        ].iloc[0]
        
        # Build HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AIDEX Report - {self.project_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1f2937; border-bottom: 3px solid #3b82f6; padding-bottom: 10px; }}
        h2 {{ color: #374151; margin-top: 30px; border-left: 4px solid #10b981; padding-left: 15px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 8px; color: white; text-align: center; }}
        .metric-card.green {{ background: linear-gradient(135deg, #10b981 0%, #059669 100%); }}
        .metric-card.blue {{ background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); }}
        .metric-card.purple {{ background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); }}
        .metric-value {{ font-size: 32px; font-weight: bold; margin: 10px 0; }}
        .metric-label {{ font-size: 14px; opacity: 0.9; }}
        .chart {{ margin: 20px 0; text-align: center; }}
        .chart img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f3f4f6; font-weight: bold; color: #374151; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 2px solid #e5e7eb; text-align: center; color: #6b7280; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AIDEX AutoML Report</h1>
        <p><strong>Project:</strong> {self.project_name}</p>
        <p><strong>Task Type:</strong> {self.task_type.capitalize()}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Best Model:</strong> {self.pipeline.best_model_name}</p>
        
        <h2>📊 Performance Metrics</h2>
        <div class="metric-grid">
"""
        
        if self.task_type == 'classification':
            html += f"""
            <div class="metric-card green">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{best_results['accuracy']*100:.2f}%</div>
            </div>
            <div class="metric-card blue">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value">{best_results['f1']*100:.2f}%</div>
            </div>
            <div class="metric-card purple">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{best_results['precision']*100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{best_results['recall']*100:.2f}%</div>
            </div>
"""
        else:
            html += f"""
            <div class="metric-card green">
                <div class="metric-label">R² Score</div>
                <div class="metric-value">{best_results['r2']:.4f}</div>
            </div>
            <div class="metric-card blue">
                <div class="metric-label">RMSE</div>
                <div class="metric-value">{best_results['rmse']:.2f}</div>
            </div>
            <div class="metric-card purple">
                <div class="metric-label">MAE</div>
                <div class="metric-value">{best_results['mae']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">MSE</div>
                <div class="metric-value">{best_results['mse']:.2f}</div>
            </div>
"""
        
        html += f"""
        </div>
        
        <h2>📈 Model Comparison</h2>
        <div class="chart">
            <img src="data:image/png;base64,{model_comparison}" alt="Model Comparison">
        </div>
        
        <h2>🎯 Predictions Analysis</h2>
        <div class="chart">
            <img src="data:image/png;base64,{predictions_chart}" alt="Predictions">
        </div>
"""
        
        if feature_importance:
            html += f"""
        <h2>🔍 Feature Importance</h2>
        <div class="chart">
            <img src="data:image/png;base64,{feature_importance}" alt="Feature Importance">
        </div>
"""
        
        html += f"""
        <h2>📊 Data Distribution</h2>
        <div class="chart">
            <img src="data:image/png;base64,{distribution_chart}" alt="Data Distribution">
        </div>
        
        <h2>📋 All Models Performance</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
"""
        
        if self.task_type == 'classification':
            html += "<th>Accuracy</th><th>F1 Score</th><th>Precision</th><th>Recall</th>"
        else:
            html += "<th>R² Score</th><th>RMSE</th><th>MAE</th><th>MSE</th>"
        
        html += """
                </tr>
            </thead>
            <tbody>
"""
        
        for _, row in self.pipeline.test_results.iterrows():
            star = "★ " if row['model_name'] == self.pipeline.best_model_name else ""
            html += f"<tr><td><strong>{star}{row['model_name']}</strong></td>"
            
            if self.task_type == 'classification':
                html += f"<td>{row['accuracy']*100:.2f}%</td><td>{row['f1']*100:.2f}%</td><td>{row['precision']*100:.2f}%</td><td>{row['recall']*100:.2f}%</td>"
            else:
                html += f"<td>{row['r2']:.4f}</td><td>{row['rmse']:.2f}</td><td>{row['mae']:.2f}</td><td>{row['mse']:.2f}</td>"
            
            html += "</tr>"
        
        html += """
            </tbody>
        </table>
        
        <div class="footer">
            <p>Generated by AIDEX - Automated ML Platform</p>
            <p>© 2026 AIDEX. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def save_report(self, output_dir: str = "reports") -> str:
        """Save HTML report to file"""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"aidex_report_{self.project_name}_{self.timestamp}.html"
        filepath = os.path.join(output_dir, filename)
        
        html = self.generate_html_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return filepath


def generate_report(pipeline, task_type: str, project_name: str, output_dir: str = "reports") -> str:
    """
    Convenience function to generate report
    
    Args:
        pipeline: Trained AIDEX pipeline
        task_type: 'classification' or 'regression'
        project_name: Name of the project
        output_dir: Directory to save report
        
    Returns:
        Path to saved report
    """
    generator = ReportGenerator(pipeline, task_type, project_name)
    return generator.save_report(output_dir)
