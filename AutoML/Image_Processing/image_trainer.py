"""
Image Model Trainer for AIDEX
Provides CNN architectures and training capabilities for image classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from typing import Dict, Optional, List, Tuple
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class ImageModelTrainer:
    """Trains and evaluates image classification models"""
    
    def __init__(self, num_classes: int, device: Optional[str] = None):
        """
        Args:
            num_classes: Number of output classes
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.num_classes = num_classes
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.models = {}
        self.training_history = {}
        self.best_model_name = None
        self.best_model_accuracy = 0.0
    
    def create_model(self, model_name: str, pretrained: bool = True) -> nn.Module:
        """
        Create a CNN model
        
        Args:
            model_name: Name of the model architecture
            pretrained: Whether to use pretrained weights
            
        Returns:
            PyTorch model
        """
        print(f"\nCreating {model_name} model (pretrained={pretrained})...")
        
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
            
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
            
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)
        
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)
        
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
        
        elif model_name == 'simple_cnn':
            # Simple custom CNN for quick training
            model = SimpleCNN(self.num_classes)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model.to(self.device)
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
    
    def train_model(self, model: nn.Module, model_name: str,
                   train_loader: DataLoader, val_loader: DataLoader,
                   epochs: int = 10, learning_rate: float = 0.001,
                   early_stopping_patience: int = 3) -> Dict:
        """
        Train a model
        
        Args:
            model: PyTorch model to train
            model_name: Name for tracking
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            early_stopping_patience: Epochs to wait before early stopping
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"TRAINING {model_name.upper()}")
        print(f"{'='*60}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Progress update
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(train_loader)} - "
                          f"Loss: {loss.item():.4f}")
            
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation phase
            val_loss, val_acc = self.evaluate_model(model, val_loader, criterion)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            print(f"\nEpoch {epoch + 1}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f"  ✓ New best validation accuracy!")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{early_stopping_patience})")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        training_time = time.time() - start_time
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        print(f"\n{'='*60}")
        print(f"Training completed in {training_time:.1f}s")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        history['training_time'] = training_time
        history['best_val_acc'] = best_val_acc
        
        # Store model and history
        self.models[model_name] = model
        self.training_history[model_name] = history
        
        # Update best model
        if best_val_acc > self.best_model_accuracy:
            self.best_model_name = model_name
            self.best_model_accuracy = best_val_acc
        
        return history
    
    def evaluate_model(self, model: nn.Module, data_loader: DataLoader,
                      criterion: nn.Module = None) -> Tuple[float, float]:
        """
        Evaluate model on a dataset
        
        Args:
            model: Model to evaluate
            data_loader: Data loader
            criterion: Loss function (optional)
            
        Returns:
            Tuple of (loss, accuracy)
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def get_predictions(self, model: nn.Module, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and probabilities
        
        Args:
            model: Model to use
            data_loader: Data loader
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(self.device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def compute_detailed_metrics(self, model: nn.Module, data_loader: DataLoader,
                                class_names: List[str]) -> Dict:
        """
        Compute detailed metrics including per-class performance
        
        Args:
            model: Model to evaluate
            data_loader: Data loader
            class_names: List of class names
            
        Returns:
            Dictionary of metrics
        """
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Per-class metrics
        per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
        per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        
        per_class_metrics = {}
        for idx, class_name in enumerate(class_names):
            per_class_metrics[class_name] = {
                'precision': float(per_class_precision[idx]),
                'recall': float(per_class_recall[idx]),
                'f1': float(per_class_f1[idx])
            }
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class_metrics
        }


class SimpleCNN(nn.Module):
    """Simple CNN architecture for quick training"""
    
    def __init__(self, num_classes: int):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
