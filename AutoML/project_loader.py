"""
AIDEX Project Loader
Loads and reconstructs project state from saved JSON files
"""

import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd


class ProjectLoader:
    """
    Loads AIDEX project state from persisted JSON files.
    
    This allows you to re-open projects and view results without
    re-running the entire pipeline - honoring AIDEX's principle
    that results come from stored state, not recomputation.
    """
    
    def __init__(self, projects_dir: str = "projects"):
        """
        Initialize the Project Loader
        
        Args:
            projects_dir: Directory where projects are stored
        """
        self.projects_dir = projects_dir
    
    def list_projects(self) -> list:
        """
        List all available project IDs
        
        Returns:
            List of project IDs (directory names)
        """
        if not os.path.exists(self.projects_dir):
            return []
        
        projects = []
        for item in os.listdir(self.projects_dir):
            project_path = os.path.join(self.projects_dir, item)
            if os.path.isdir(project_path):
                state_file = os.path.join(project_path, "project_state.json")
                if os.path.exists(state_file):
                    projects.append(item)
        
        return sorted(projects)
    
    def load_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a project's state from JSON
        
        Args:
            project_id: The project ID to load
            
        Returns:
            Dictionary with project state, or None if not found
        """
        project_path = os.path.join(self.projects_dir, project_id)
        state_file = os.path.join(project_path, "project_state.json")
        
        if not os.path.exists(state_file):
            return None
        
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            return state
        except Exception as e:
            print(f"Error loading project {project_id}: {e}")
            return None
    
    def get_project_report(self, project_id: str) -> Optional[str]:
        """
        Get the human-readable report text
        
        Args:
            project_id: The project ID
            
        Returns:
            Report text as string, or None if not found
        """
        project_path = os.path.join(self.projects_dir, project_id)
        report_file = os.path.join(project_path, "report.txt")
        
        if not os.path.exists(report_file):
            return None
        
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error loading report for {project_id}: {e}")
            return None
    
    def project_exists(self, project_id: str) -> bool:
        """
        Check if a project exists
        
        Args:
            project_id: The project ID to check
            
        Returns:
            True if project exists, False otherwise
        """
        project_path = os.path.join(self.projects_dir, project_id)
        state_file = os.path.join(project_path, "project_state.json")
        return os.path.exists(state_file)
    
    def get_project_summary(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a quick summary of a project (without loading full state)
        
        Args:
            project_id: The project ID
            
        Returns:
            Dictionary with summary info, or None if not found
        """
        state = self.load_project(project_id)
        if state is None:
            return None
        
        # Extract key info
        summary = {
            'project_id': state.get('project_id'),
            'task_type': state.get('task_type'),
            'best_model_name': state.get('best_model_name'),
            'saved_at': state.get('saved_at_utc'),
            'dataset': state.get('dataset', {}),
        }
        
        # Get best model metrics from test results
        test_results = state.get('test_results', [])
        if test_results:
            best_model_name = summary['best_model_name']
            for result in test_results:
                if result.get('model_name') == best_model_name:
                    summary['best_model_metrics'] = {
                        'accuracy': result.get('accuracy'),
                        'f1': result.get('f1'),
                        'precision': result.get('precision'),
                        'recall': result.get('recall')
                    }
                    break
        
        return summary


def load_project(project_id: str, projects_dir: str = "projects") -> Optional[Dict[str, Any]]:
    """
    Convenience function to load a project
    
    Args:
        project_id: The project ID to load
        projects_dir: Directory where projects are stored
        
    Returns:
        Project state dictionary, or None if not found
    """
    loader = ProjectLoader(projects_dir=projects_dir)
    return loader.load_project(project_id)


def list_all_projects(projects_dir: str = "projects") -> list:
    """
    Convenience function to list all projects
    
    Args:
        projects_dir: Directory where projects are stored
        
    Returns:
        List of project IDs
    """
    loader = ProjectLoader(projects_dir=projects_dir)
    return loader.list_projects()


if __name__ == "__main__":
    """
    Example usage of Project Loader
    """
    print("="*80)
    print("AIDEX PROJECT LOADER - EXAMPLE USAGE")
    print("="*80)
    
    loader = ProjectLoader()
    
    # List all projects
    projects = loader.list_projects()
    print(f"\n📁 Found {len(projects)} project(s):")
    for pid in projects:
        print(f"   - {pid}")
    
    if projects:
        # Load first project
        project_id = projects[0]
        print(f"\n📂 Loading project: {project_id}")
        
        state = loader.load_project(project_id)
        if state:
            print(f"   [OK] Project loaded successfully")
            print(f"   Task Type: {state.get('task_type')}")
            print(f"   Best Model: {state.get('best_model_name')}")
            print(f"   Saved At: {state.get('saved_at_utc')}")
        
        # Get summary
        summary = loader.get_project_summary(project_id)
        if summary:
            print(f"\n📊 Project Summary:")
            print(f"   Dataset: {summary['dataset'].get('n_samples_total')} samples")
            if 'best_model_metrics' in summary:
                metrics = summary['best_model_metrics']
                print(f"   Best Model Accuracy: {metrics.get('accuracy', 0)*100:.2f}%")
        
        # Get report
        report = loader.get_project_report(project_id)
        if report:
            print(f"\n📄 Report Preview (first 500 chars):")
            print("-"*80)
            print(report[:500] + "..." if len(report) > 500 else report)
    else:
        print("\n⚠️  No projects found. Run a pipeline first to create projects.")
