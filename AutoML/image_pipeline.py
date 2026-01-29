"""
AIDEX Image Pipeline
End-to-end automated image classification workflow
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from typing import Dict, List, Optional, Union
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Import image processing modules
from Image_Processing.image_loader import ImagePreprocessor
from Image_Processing.image_trainer import ImageModelTrainer


class ImagePipeline:
    """Complete AIDEX image classification pipeline"""
    
    def __init__(self, image_size: int = 224, batch_size: int = 32,
                 test_size: float = 0.2, random_state: int = 42,
                 project_id: Optional[str] = None):
        """
        Initialize Image Pipeline
        
        Args:
            image_size: Target size for images (default: 224)
            batch_size: Batch size for training
            test_size: Proportion of dataset for testing
            random_state: Random seed for reproducibility
            project_id: Project identifier
        """
        self.image_size = image_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.project_id = project_id or f"image_project_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        self.preprocessor = ImagePreprocessor(image_size=image_size, batch_size=batch_size)
        self.trainer = None
        
        self.data_info = None
        self.training_results = {}
        self.test_results = {}
        self.best_model_name = None
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n{'='*60}")
        print(f"AIDEX IMAGE PIPELINE INITIALIZED")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Image size: {image_size}x{image_size}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")
    
    def load_and_prepare_data(self, image_paths: List[str], labels: List[str]) -> Dict:
        """
        Load and prepare image datasets
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            
        Returns:
            Dictionary with dataset information
        """
        print(f"\n{'='*60}")
        print(f"LOADING AND PREPARING IMAGE DATA")
        print(f"{'='*60}")
        
        # Validate images
        valid_paths, valid_indices = self.preprocessor.validate_images(image_paths)
        valid_labels = [labels[i] for i in valid_indices]
        
        if len(valid_paths) == 0:
            raise ValueError("No valid images found!")
        
        # Prepare datasets
        self.data_info = self.preprocessor.prepare_datasets(
            valid_paths, valid_labels,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        print(f"✓ Data preparation complete")
        print(f"  Train samples: {self.data_info['train_size']}")
        print(f"  Test samples: {self.data_info['test_size']}")
        print(f"  Number of classes: {self.data_info['num_classes']}")
        print(f"  Classes: {self.data_info['class_names']}")
        print(f"{'='*60}\n")
        
        return self.data_info
    
    def train_models(self, model_names: List[str] = None,
                    epochs: int = 10, learning_rate: float = 0.001,
                    pretrained: bool = True) -> Dict:
        """
        Train multiple models
        
        Args:
            model_names: List of model architectures to train
            epochs: Number of training epochs
            learning_rate: Learning rate
            pretrained: Use pretrained weights
            
        Returns:
            Dictionary with training results
        """
        if self.data_info is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        # Default models
        if model_names is None:
            model_names = ['resnet18', 'mobilenet_v2']  # Fast models by default
        
        # Initialize trainer
        self.trainer = ImageModelTrainer(
            num_classes=self.data_info['num_classes'],
            device=self.device
        )
        
        print(f"\n{'='*60}")
        print(f"TRAINING {len(model_names)} MODEL(S)")
        print(f"{'='*60}")
        print(f"Models: {', '.join(model_names)}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Pretrained: {pretrained}")
        print(f"{'='*60}\n")
        
        # Train each model
        for model_name in model_names:
            try:
                model = self.trainer.create_model(model_name, pretrained=pretrained)
                history = self.trainer.train_model(
                    model=model,
                    model_name=model_name,
                    train_loader=self.data_info['train_loader'],
                    val_loader=self.data_info['test_loader'],
                    epochs=epochs,
                    learning_rate=learning_rate
                )
                self.training_results[model_name] = history
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                continue
        
        # Determine best model
        if self.trainer.best_model_name:
            self.best_model_name = self.trainer.best_model_name
            print(f"\n{'='*60}")
            print(f"BEST MODEL: {self.best_model_name}")
            print(f"Best Validation Accuracy: {self.trainer.best_model_accuracy:.2f}%")
            print(f"{'='*60}\n")
        
        return self.training_results
    
    def evaluate_models(self) -> Dict:
        """
        Evaluate all trained models on test set
        
        Returns:
            Dictionary with test results
        """
        if self.trainer is None or len(self.trainer.models) == 0:
            raise ValueError("No models trained. Call train_models() first.")
        
        print(f"\n{'='*60}")
        print(f"EVALUATING MODELS ON TEST SET")
        print(f"{'='*60}")
        
        for model_name, model in self.trainer.models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Get detailed metrics
            metrics = self.trainer.compute_detailed_metrics(
                model=model,
                data_loader=self.data_info['test_loader'],
                class_names=self.data_info['class_names']
            )
            
            self.test_results[model_name] = metrics
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
        
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*60}\n")
        
        return self.test_results
    
    def predict(self, image_paths: List[str], model_name: Optional[str] = None) -> Dict:
        """
        Make predictions on new images
        
        Args:
            image_paths: List of image paths to predict
            model_name: Model to use (default: best model)
            
        Returns:
            Dictionary with predictions
        """
        if self.trainer is None:
            raise ValueError("No models trained. Call train_models() first.")
        
        # Use best model if not specified
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.trainer.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.trainer.models.keys())}")
        
        model = self.trainer.models[model_name]
        
        print(f"\n{'='*60}")
        print(f"MAKING PREDICTIONS")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Number of images: {len(image_paths)}")
        
        # Prepare data
        predict_loader = self.preprocessor.prepare_prediction_data(image_paths)
        
        # Get predictions
        predictions, probabilities = self.trainer.get_predictions(model, predict_loader)
        
        # Map to class names
        idx_to_label = self.data_info['idx_to_label']
        predicted_labels = [idx_to_label[int(pred)] for pred in predictions]
        
        print(f"✓ Predictions complete")
        print(f"{'='*60}\n")
        
        # Prepare results
        results = []
        for i, (path, label, probs) in enumerate(zip(image_paths, predicted_labels, probabilities)):
            result = {
                'image_path': path,
                'predicted_label': label,
                'confidence': float(np.max(probs)),
                'probabilities': {
                    self.data_info['class_names'][j]: float(probs[j])
                    for j in range(len(probs))
                }
            }
            results.append(result)
        
        return {
            'predictions': results,
            'model_used': model_name,
            'num_classes': self.data_info['num_classes'],
            'class_names': self.data_info['class_names']
        }
    
    def get_summary(self) -> Dict:
        """
        Get comprehensive pipeline summary
        
        Returns:
            Dictionary with complete summary
        """
        summary = {
            'project_id': self.project_id,
            'device': str(self.device),
            'image_size': self.image_size,
            'batch_size': self.batch_size
        }
        
        if self.data_info:
            summary['data'] = {
                'num_classes': self.data_info['num_classes'],
                'class_names': self.data_info['class_names'],
                'train_size': self.data_info['train_size'],
                'test_size': self.data_info['test_size']
            }
        
        if self.training_results:
            summary['training'] = {
                model_name: {
                    'best_val_acc': history['best_val_acc'],
                    'training_time': history['training_time'],
                    'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0,
                    'final_val_acc': history['val_acc'][-1] if history['val_acc'] else 0
                }
                for model_name, history in self.training_results.items()
            }
        
        if self.test_results:
            summary['test'] = self.test_results
        
        if self.best_model_name:
            summary['best_model'] = self.best_model_name
        
        return summary


def run_image_pipeline(image_paths: List[str], labels: List[str],
                      image_size: int = 224, batch_size: int = 32,
                      test_size: float = 0.2, epochs: int = 10,
                      model_names: List[str] = None,
                      learning_rate: float = 0.001,
                      pretrained: bool = True,
                      project_id: Optional[str] = None) -> ImagePipeline:
    """
    Run complete image classification pipeline
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        image_size: Target image size
        batch_size: Training batch size
        test_size: Test set proportion
        epochs: Training epochs
        model_names: List of models to train
        learning_rate: Learning rate
        pretrained: Use pretrained weights
        project_id: Project identifier
        
    Returns:
        Completed ImagePipeline object
    """
    print(f"\n{'='*80}")
    print(f"STARTING AIDEX IMAGE PIPELINE")
    print(f"{'='*80}\n")
    
    # Initialize pipeline
    pipeline = ImagePipeline(
        image_size=image_size,
        batch_size=batch_size,
        test_size=test_size,
        project_id=project_id
    )
    
    # Load and prepare data
    pipeline.load_and_prepare_data(image_paths, labels)
    
    # Train models
    pipeline.train_models(
        model_names=model_names,
        epochs=epochs,
        learning_rate=learning_rate,
        pretrained=pretrained
    )
    
    # Evaluate models
    pipeline.evaluate_models()
    
    # Print summary
    summary = pipeline.get_summary()
    print(f"\n{'='*80}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Best Model: {summary.get('best_model', 'N/A')}")
    if 'test' in summary and summary['best_model']:
        best_metrics = summary['test'].get(summary['best_model'], {})
        print(f"Test Accuracy: {best_metrics.get('accuracy', 0):.4f}")
        print(f"Test F1 Score: {best_metrics.get('f1', 0):.4f}")
    print(f"{'='*80}\n")
    
    return pipeline
