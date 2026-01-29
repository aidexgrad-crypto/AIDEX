"""
Image Data Loader and Preprocessor for AIDEX
Handles loading, augmentation, and preprocessing of image datasets
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class ImageDataset(Dataset):
    """Custom Dataset for loading images"""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform=None, class_names: List[str] = None):
        """
        Args:
            image_paths: List of paths to images
            labels: List of corresponding labels (integers)
            transform: torchvision transforms to apply
            class_names: List of class names (optional)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ImagePreprocessor:
    """Handles image preprocessing and data loading"""
    
    def __init__(self, image_size: int = 224, batch_size: int = 32):
        """
        Args:
            image_size: Target size for images (default: 224 for most pre-trained models)
            batch_size: Batch size for data loaders
        """
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Define transforms for training (with augmentation)
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
        
        # Define transforms for validation/testing (no augmentation)
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # For prediction (similar to validation but return original size info)
        self.predict_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_image_from_path(self, image_path: str) -> Image.Image:
        """Load a single image from path"""
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def prepare_datasets(self, image_paths: List[str], labels: List[str],
                        test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Prepare train and test datasets
        
        Args:
            image_paths: List of image file paths
            labels: List of string labels
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Dictionary containing datasets, loaders, and metadata
        """
        from sklearn.model_selection import train_test_split
        
        # Convert string labels to integers
        unique_labels = sorted(list(set(labels)))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        numeric_labels = [label_to_idx[label] for label in labels]
        
        print(f"\n{'='*60}")
        print(f"IMAGE DATA PREPARATION")
        print(f"{'='*60}")
        print(f"Total images: {len(image_paths)}")
        print(f"Number of classes: {len(unique_labels)}")
        print(f"Classes: {unique_labels}")
        print(f"Label distribution:")
        for label in unique_labels:
            count = labels.count(label)
            print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, numeric_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=numeric_labels
        )
        
        print(f"\nTrain set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")
        print(f"{'='*60}\n")
        
        # Create datasets
        train_dataset = ImageDataset(X_train, y_train, 
                                    transform=self.train_transform,
                                    class_names=unique_labels)
        test_dataset = ImageDataset(X_test, y_test, 
                                   transform=self.val_transform,
                                   class_names=unique_labels)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, 
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=0,  # Windows compatibility
                                 pin_memory=True)
        
        test_loader = DataLoader(test_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True)
        
        return {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'num_classes': len(unique_labels),
            'class_names': unique_labels,
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    def prepare_prediction_data(self, image_paths: List[str]) -> DataLoader:
        """
        Prepare data loader for prediction
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            DataLoader for predictions
        """
        # Create dummy labels (not used in prediction)
        dummy_labels = [0] * len(image_paths)
        
        predict_dataset = ImageDataset(image_paths, dummy_labels,
                                      transform=self.predict_transform)
        
        predict_loader = DataLoader(predict_dataset,
                                   batch_size=self.batch_size,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=True)
        
        return predict_loader
    
    def validate_images(self, image_paths: List[str]) -> Tuple[List[str], List[int]]:
        """
        Validate images and return valid paths with their indices
        
        Args:
            image_paths: List of image paths to validate
            
        Returns:
            Tuple of (valid_paths, valid_indices)
        """
        valid_paths = []
        valid_indices = []
        
        for idx, path in enumerate(image_paths):
            try:
                img = Image.open(path)
                img.verify()  # Verify it's actually an image
                valid_paths.append(path)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Invalid image {path}: {e}")
        
        print(f"Validated {len(valid_paths)}/{len(image_paths)} images")
        return valid_paths, valid_indices


def get_image_augmentation_transforms(augmentation_level: str = 'medium',
                                     image_size: int = 224) -> transforms.Compose:
    """
    Get augmentation transforms based on level
    
    Args:
        augmentation_level: 'light', 'medium', or 'heavy'
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    base_transforms = [
        transforms.Resize((image_size, image_size)),
    ]
    
    if augmentation_level == 'light':
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=0.3),
        ]
    elif augmentation_level == 'medium':
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    elif augmentation_level == 'heavy':
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
    else:
        aug_transforms = []
    
    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]
    
    return transforms.Compose(base_transforms + aug_transforms + final_transforms)
