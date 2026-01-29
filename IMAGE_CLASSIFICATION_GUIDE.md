# Image Classification with AIDEX

This guide explains how to use AIDEX for training image classification models with automatic model selection and hyperparameter tuning.

## Overview

AIDEX now supports **image classification** in addition to tabular data. The system automatically:
- Preprocesses and augments images
- Trains multiple CNN architectures (ResNet, MobileNet, EfficientNet)
- Selects the best performing model
- Provides easy-to-use prediction endpoints

## Features

### Image Preprocessing
- **Automatic resizing**: All images resized to 224x224 (standard for CNNs)
- **Data augmentation**: Random flips, rotations, color jitter
- **Normalization**: ImageNet mean/std normalization
- **Validation**: Corrupt image detection and removal

### Model Architectures
The system supports multiple pre-trained CNN architectures:
- **ResNet18/34/50**: Deep residual networks (best for accuracy)
- **MobileNetV2**: Lightweight model (best for speed)
- **EfficientNet B0**: Balanced efficiency
- **VGG16**: Classic CNN architecture
- **SimpleCNN**: Custom lightweight model

### Training Features
- **Transfer Learning**: Uses ImageNet pre-trained weights
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **GPU Support**: Automatic CUDA detection
- **Progress Tracking**: Real-time training metrics

## Usage

### 1. Upload Images

Upload a folder containing images organized by class:
```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── class3/
    ├── image1.jpg
    └── ...
```

### 2. Frontend Integration

In the preprocessing page, after cleaning images:

```typescript
// Upload images to backend
const formData = new FormData();
state.images.forEach((img) => {
  formData.append("files", img.file);
});

const uploadResponse = await fetch("/api/automl/upload-images", {
  method: "POST",
  body: formData,
});

const uploadData = await uploadResponse.json();

// Train models
const imageData = state.images.map((img, idx) => ({
  path: uploadData.paths[idx],
  label: img.label,
}));

const response = await fetch("/api/automl/train-images", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    image_data: imageData,
    epochs: 10,
    batch_size: 32,
    learning_rate: 0.001,
    test_size: 0.2,
    model_names: ["resnet18", "mobilenet_v2"],
    project_name: `image_${Date.now()}`
  }),
});
```

### 3. Backend API

#### Train Images Endpoint
```python
POST /automl/train-images

Request Body:
{
  "image_data": [
    {"path": "/path/to/image1.jpg", "label": "cat"},
    {"path": "/path/to/image2.jpg", "label": "dog"}
  ],
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 0.001,
  "test_size": 0.2,
  "model_names": ["resnet18", "mobilenet_v2"],
  "project_name": "my_project"
}

Response:
{
  "status": "success",
  "project_name": "image_1234567890",
  "best_model": "resnet18",
  "num_classes": 3,
  "class_names": ["cat", "dog", "bird"],
  "train_size": 800,
  "test_size": 200,
  "training_results": [...],
  "test_results": [
    {
      "model_name": "resnet18",
      "accuracy": 0.95,
      "f1": 0.94,
      "precision": 0.96,
      "recall": 0.93
    }
  ]
}
```

#### Predict Images Endpoint
```python
POST /automl/predict-images

Request Body:
{
  "image_paths": ["/path/to/new_image1.jpg", "/path/to/new_image2.jpg"],
  "project_name": "image_1234567890"  // optional, uses last trained model if not provided
}

Response:
{
  "status": "success",
  "project_name": "image_1234567890",
  "model_used": "resnet18",
  "results": [
    {
      "image_path": "/path/to/new_image1.jpg",
      "predicted_label": "cat",
      "confidence": 0.98,
      "probabilities": {
        "cat": 0.98,
        "dog": 0.01,
        "bird": 0.01
      }
    }
  ]
}
```

### 4. Direct Python Usage

You can also use the pipeline directly in Python:

```python
from AutoML.image_pipeline import run_image_pipeline

# Prepare data
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
labels = ["cat", "dog", "cat"]

# Run pipeline
pipeline = run_image_pipeline(
    image_paths=image_paths,
    labels=labels,
    image_size=224,
    batch_size=32,
    test_size=0.2,
    epochs=10,
    model_names=["resnet18", "mobilenet_v2"],
    learning_rate=0.001,
    pretrained=True,
    project_id="my_project"
)

# Make predictions
new_images = ["new_img1.jpg", "new_img2.jpg"]
results = pipeline.predict(new_images)

print(f"Best model: {pipeline.best_model_name}")
for result in results['predictions']:
    print(f"{result['image_path']}: {result['predicted_label']} ({result['confidence']:.2%})")
```

## Architecture Details

### Image Preprocessing Pipeline
1. **Load Images**: Read from file paths
2. **Validate**: Check for corrupt/invalid images
3. **Resize**: Resize to target size (224x224)
4. **Augment**: Apply random transformations (training only)
5. **Normalize**: Apply ImageNet normalization
6. **Batch**: Create mini-batches for efficient training

### Training Pipeline
1. **Data Split**: Split into train/test sets (stratified)
2. **Model Creation**: Initialize models with pretrained weights
3. **Fine-tuning**: Replace final layers for custom classes
4. **Training Loop**:
   - Forward pass through network
   - Calculate loss (CrossEntropyLoss)
   - Backward pass and weight updates
   - Validation on test set
5. **Model Selection**: Choose best based on validation accuracy
6. **Save**: Persist trained models and metadata

### Model Architecture

Example: ResNet18 for 3-class classification
```
Input (3x224x224)
    ↓
ResNet18 Backbone (pretrained)
    ↓
Global Average Pooling
    ↓
Fully Connected (512 → 3)
    ↓
Softmax
    ↓
Output (3 classes)
```

## Performance Tips

### For Better Accuracy
- Use more training epochs (20-30)
- Try ResNet50 or EfficientNet
- Ensure balanced class distribution
- Use larger batch size if memory allows

### For Faster Training
- Use MobileNetV2 or SimpleCNN
- Reduce image size to 128x128
- Increase batch size
- Use fewer epochs with early stopping

### Memory Optimization
- Reduce batch size if OOM errors occur
- Use smaller models (MobileNetV2, SimpleCNN)
- Resize images to 128x128 instead of 224x224
- Enable mixed precision training (future feature)

## File Structure

```
AutoML/
├── image_pipeline.py           # Main pipeline orchestrator
├── Image_Processing/
│   ├── __init__.py
│   ├── image_loader.py         # Data loading and preprocessing
│   └── image_trainer.py        # Model training and evaluation
Back-End/
├── main.py                     # FastAPI endpoints
└── saved_models/               # Saved model checkpoints
front-end/
└── app/
    ├── api/automl/
    │   ├── train-images/       # Training API route
    │   ├── predict-images/     # Prediction API route
    │   └── upload-images/      # Image upload route
    └── preprocessing/
        └── page.tsx            # UI with training button
```

## Dependencies

```
torch>=2.0.0          # PyTorch for deep learning
torchvision>=0.15.0   # Pre-trained models and transforms
Pillow>=9.0.0         # Image loading and processing
numpy>=1.21.0         # Numerical operations
scikit-learn>=1.0.0   # Metrics and data splitting
```

Install with:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` to 16 or 8
- Use smaller model (MobileNetV2)
- Reduce image size

### Training Too Slow
- Enable GPU if available
- Use smaller model
- Reduce number of epochs
- Increase batch size

### Poor Accuracy
- Check class balance in dataset
- Increase training epochs
- Try different models
- Ensure images are labeled correctly
- Use data augmentation

### Import Errors
- Ensure PyTorch is installed: `pip install torch torchvision`
- Check Python version (3.8+)
- Verify CUDA installation for GPU support

## Examples

### Binary Classification (Cat vs Dog)
```python
image_paths = ["cat1.jpg", "dog1.jpg", "cat2.jpg", "dog2.jpg"]
labels = ["cat", "dog", "cat", "dog"]

pipeline = run_image_pipeline(
    image_paths=image_paths,
    labels=labels,
    epochs=15,
    model_names=["resnet18"],
)
```

### Multi-class Classification
```python
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
labels = ["cat", "dog", "bird", "cat"]

pipeline = run_image_pipeline(
    image_paths=image_paths,
    labels=labels,
    epochs=20,
    model_names=["resnet18", "mobilenet_v2"],
)
```

## Next Steps

- [ ] Add data augmentation presets (light, medium, heavy)
- [ ] Implement mixed precision training for faster training
- [ ] Add model interpretability (Grad-CAM)
- [ ] Support for object detection
- [ ] Add model ensemble predictions
- [ ] Implement active learning for label suggestions

## Support

For issues or questions:
- Check the logs in the backend terminal
- Verify image paths are correct
- Ensure sufficient disk space for model checkpoints
- Check GPU memory availability

## License

Same as AIDEX main project.
