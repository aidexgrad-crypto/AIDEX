"""
AIDEX Project Viewer
Simple UI-like viewer for displaying project results
"""

import os
from typing import Dict, Any, Optional
from project_loader import ProjectLoader
import pandas as pd


class ProjectViewer:
    """
    Displays AIDEX project results in a user-friendly format.
    
    This is a simple console-based viewer. In a full AIDEX deployment,
    this would be replaced by a web UI that consumes the same JSON state.
    """
    
    def __init__(self, projects_dir: str = "projects"):
        """
        Initialize the Project Viewer
        
        Args:
            projects_dir: Directory where projects are stored
        """
        self.loader = ProjectLoader(projects_dir=projects_dir)
    
    def display_project_list(self):
        """
        Display a list of all available projects
        """
        projects = self.loader.list_projects()
        
        print("\n" + "="*80)
        print("AIDEX PROJECTS")
        print("="*80)
        
        if not projects:
            print("\n⚠️  No projects found.")
            print("   Run a pipeline to create your first project.")
            return
        
        print(f"\n📁 Found {len(projects)} project(s):\n")
        
        for i, project_id in enumerate(projects, 1):
            summary = self.loader.get_project_summary(project_id)
            if summary:
                print(f"{i}. {project_id}")
                print(f"   Task: {summary.get('task_type', 'N/A')}")
                print(f"   Best Model: {summary.get('best_model_name', 'N/A')}")
                
                metrics = summary.get('best_model_metrics', {})
                if metrics:
                    acc = metrics.get('accuracy', 0)
                    print(f"   Accuracy: {acc*100:.2f}%" if acc else "   Accuracy: N/A")
                
                saved_at = summary.get('saved_at', 'N/A')
                print(f"   Created: {saved_at}")
                print()
    
    def display_project_overview(self, project_id: str):
        """
        Display a comprehensive overview of a project
        
        Args:
            project_id: The project ID to display
        """
        state = self.loader.load_project(project_id)
        if state is None:
            print(f"\n❌ Project '{project_id}' not found.")
            return
        
        print("\n" + "="*80)
        print(f"AIDEX PROJECT: {project_id}")
        print("="*80)
        
        # Basic Info
        print("\n📋 PROJECT INFORMATION")
        print("-"*80)
        print(f"Project ID      : {state.get('project_id', 'N/A')}")
        print(f"Task Type       : {state.get('task_type', 'N/A')}")
        print(f"Created At (UTC): {state.get('saved_at_utc', 'N/A')}")
        
        # Dataset Info
        dataset = state.get('dataset', {})
        print("\n📊 DATASET INFORMATION")
        print("-"*80)
        print(f"Total Samples   : {dataset.get('n_samples_total', 'N/A'):,}" if dataset.get('n_samples_total') else "Total Samples   : N/A")
        print(f"Training Samples: {dataset.get('n_train', 'N/A'):,}" if dataset.get('n_train') else "Training Samples: N/A")
        print(f"Test Samples    : {dataset.get('n_test', 'N/A'):,}" if dataset.get('n_test') else "Test Samples    : N/A")
        print(f"Features        : {dataset.get('n_features', 'N/A'):,}" if dataset.get('n_features') else "Features        : N/A")
        
        # Feature Types
        feature_info = state.get('feature_info', {})
        if feature_info:
            feature_types = feature_info.get('feature_types', {})
            if feature_types:
                print("\n🔍 FEATURE TYPES DETECTED")
                print("-"*80)
                for ftype, cols in feature_types.items():
                    if cols:
                        print(f"  {ftype.capitalize()}: {len(cols)} features")
        
        # Best Model
        best_model_name = state.get('best_model_name')
        print("\n🏆 SELECTED MODEL")
        print("-"*80)
        print(f"Model Name: {best_model_name or 'N/A'}")
        
        # Best Model Metrics
        test_results = state.get('test_results', [])
        if test_results and best_model_name:
            best_result = None
            for result in test_results:
                if result.get('model_name') == best_model_name:
                    best_result = result
                    break
            
            if best_result:
                print("\n📈 PERFORMANCE METRICS (Test Set)")
                print("-"*80)
                if 'accuracy' in best_result:
                    print(f"Accuracy  : {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
                if 'f1' in best_result:
                    print(f"F1 Score  : {best_result['f1']:.4f} ({best_result['f1']*100:.2f}%)")
                if 'precision' in best_result:
                    print(f"Precision : {best_result['precision']:.4f} ({best_result['precision']*100:.2f}%)")
                if 'recall' in best_result:
                    print(f"Recall    : {best_result['recall']:.4f} ({best_result['recall']*100:.2f}%)")
        
        # All Models Comparison
        if test_results:
            print("\n📊 ALL MODELS COMPARISON (Test Set)")
            print("-"*80)
            
            # Create DataFrame for nice display
            df = pd.DataFrame(test_results)
            display_cols = ['model_name']
            
            if 'accuracy' in df.columns:
                display_cols.append('accuracy')
            if 'f1' in df.columns:
                display_cols.append('f1')
            if 'precision' in df.columns:
                display_cols.append('precision')
            if 'recall' in df.columns:
                display_cols.append('recall')
            
            display_df = df[display_cols].copy()
            
            # Format percentages
            for col in ['accuracy', 'f1', 'precision', 'recall']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f} ({x*100:.2f}%)" if x is not None else "N/A")
            
            # Rename columns
            display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
            
            print(display_df.to_string(index=False))
        
        # Cross-Validation Results
        cv_results = state.get('cv_results', [])
        if cv_results:
            print("\n🔄 CROSS-VALIDATION RESULTS (Top 5)")
            print("-"*80)
            
            cv_df = pd.DataFrame(cv_results)
            
            # Select columns to display
            display_cols = ['model_name']
            if 'accuracy_mean' in cv_df.columns:
                display_cols.append('accuracy_mean')
            if 'f1_mean' in cv_df.columns:
                display_cols.append('f1_mean')
            if 'training_time' in cv_df.columns:
                display_cols.append('training_time')
            
            cv_display = cv_df[display_cols].copy()
            
            # Format
            if 'accuracy_mean' in cv_display.columns:
                cv_display['accuracy_mean'] = cv_display['accuracy_mean'].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
            if 'f1_mean' in cv_display.columns:
                cv_display['f1_mean'] = cv_display['f1_mean'].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
            if 'training_time' in cv_display.columns:
                cv_display['training_time'] = cv_display['training_time'].apply(lambda x: f"{x:.2f}s" if x is not None else "N/A")
            
            cv_display.columns = [col.replace('_', ' ').title() for col in cv_display.columns]
            
            print(cv_display.head(5).to_string(index=False))
        
        print("\n" + "="*80)
    
    def display_project_report(self, project_id: str):
        """
        Display the human-readable report
        
        Args:
            project_id: The project ID
        """
        report = self.loader.get_project_report(project_id)
        if report:
            print("\n" + "="*80)
            print(f"PROJECT REPORT: {project_id}")
            print("="*80)
            print(report)
            print("="*80)
        else:
            print(f"\n❌ Report not found for project '{project_id}'")


def view_project(project_id: str, projects_dir: str = "projects"):
    """
    Convenience function to view a project
    
    Args:
        project_id: The project ID to view
        projects_dir: Directory where projects are stored
    """
    viewer = ProjectViewer(projects_dir=projects_dir)
    viewer.display_project_overview(project_id)


if __name__ == "__main__":
    """
    Example usage of Project Viewer
    """
    viewer = ProjectViewer()
    
    # Display list of projects
    viewer.display_project_list()
    
    # If projects exist, display first one
    projects = viewer.loader.list_projects()
    if projects:
        print("\n" + "="*80)
        print("VIEWING FIRST PROJECT")
        print("="*80)
        viewer.display_project_overview(projects[0])
        
        print("\n" + "="*80)
        print("VIEWING PROJECT REPORT")
        print("="*80)
        viewer.display_project_report(projects[0])
