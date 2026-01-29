# Complete Example: Training an Image Classifier with AIDEX

This example walks through the entire process of training an image classification model using AIDEX.

## Scenario: Cat vs Dog Classifier

We'll build a classifier that can distinguish between images of cats and dogs.

## Step 1: Prepare Your Dataset

### Dataset Structure
```
my_pet_dataset/
├── cat/
│   ├── cat_001.jpg
│   ├── cat_002.jpg
│   ├── cat_003.jpg
│   └── ... (more cat images)
└── dog/
    ├── dog_001.jpg
    ├── dog_002.jpg
    ├── dog_003.jpg
    └── ... (more dog images)
```

**Requirements**:
- At least 20 images per class (more is better!)
- Images can be any size (AIDEX will resize them)
- Supported formats: .jpg, .jpeg, .png
- Organized in folders by class name

## Step 2: Upload Dataset via UI

### 2.1 Start AIDEX

**Terminal 1 - Backend**:
```powershell
cd c:\Users\DELL\documents\aidex
.\venv\Scripts\Activate.ps1
$env:PYTHONPATH = "C:\Users\DELL\documents\aidex\AutoML"
python -m uvicorn Back-End.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend**:
```powershell
cd c:\Users\DELL\documents\aidex\front-end
npm run dev
```

### 2.2 Upload Images

1. Open browser: `http://localhost:3000`
2. Click **"Upload Image Folder"**
3. Select your `my_pet_dataset` folder
4. Wait for upload to complete
5. You should see: "Images loaded: cat (25), dog (30)"

## Step 3: Data Overview

Navigate to the **Overview** page:

- **Total Images**: 55
- **Classes**: cat, dog
- **Class Distribution**:
  - cat: 25 images (45%)
  - dog: 30 images (55%)

This shows your dataset is reasonably balanced!

## Step 4: Image Preprocessing

Click **"Go to Preprocessing"**

### What AIDEX Does Automatically:
1. ✅ Validates all images (removes corrupted ones)
2. ✅ Resizes to 224×224 pixels
3. ✅ Normalizes pixel values
4. ✅ Prepares for training

Click **"Start Preparation"** and wait ~10 seconds.

### Results:
```
✓ Image Preparation Completed

Images: 55 → 55
Resized: 55
Removed corrupted: 0
Image size: 224 × 224

What AIDEX did for you:
• Resized all images to 224x224
• Normalized pixel values
• Validated image integrity
• Prepared augmentation pipeline
```

You'll see a preview of 8 sample images.

## Step 5: Train Models

Scroll down to **"Next Step: Train Image Classification Models"**

Click **"Start Image AutoML Training"**

### Training Process (takes 3-5 minutes):

**Console Output**:
```
================================================================================
STARTING AIDEX IMAGE PIPELINE
================================================================================

Device: cuda (or cpu)
Image size: 224x224
Batch size: 32

============================================================
LOADING AND PREPARING IMAGE DATA
============================================================
Total images: 55
Number of classes: 2
Classes: ['cat', 'dog']

Train set: 44 images
Test set: 11 images

============================================================
TRAINING RESNET18
============================================================
Epoch 1/10 (12.3s)
  Train Loss: 0.6543, Train Acc: 68.18%
  Val Loss: 0.5234, Val Acc: 72.73%
  ✓ New best validation accuracy!

Epoch 2/10 (11.8s)
  Train Loss: 0.4321, Train Acc: 81.82%
  Val Loss: 0.3456, Val Acc: 90.91%
  ✓ New best validation accuracy!

...

Training completed in 127.4s
Best validation accuracy: 90.91%

============================================================
TRAINING MOBILENET_V2
============================================================
...
```

## Step 6: View Results

### Training Summary

The UI displays:

```
✓ Image Models Trained Successfully!

Best Model: resnet18

Class Distribution:
[cat] [dog]

Model Performance:
┌──────────────┬──────────┬──────────┬───────────┬────────┐
│ Model        │ Accuracy │ F1 Score │ Precision │ Recall │
├──────────────┼──────────┼──────────┼───────────┼────────┤
│ ★ resnet18   │  90.91%  │  90.48%  │   92.00%  │ 90.00% │
│ mobilenet_v2 │  86.36%  │  85.71%  │   88.00%  │ 84.00% │
└──────────────┴──────────┴──────────┴───────────┴────────┘

Training: 44 images | Test: 11 images | Classes: 2
```

**Interpretation**:
- **ResNet18** achieved 90.91% accuracy (best model)
- **MobileNetV2** achieved 86.36% accuracy
- Both models perform well, ResNet18 is slightly better
- High precision (92%) means few false positives
- High recall (90%) means few false negatives

## Step 7: Make Predictions (Python)

### Using the Trained Model

Save this as `predict_pet.py`:

```python
import sys
import os
sys.path.insert(0, 'c:/Users/DELL/documents/aidex/AutoML')

import torch
from image_pipeline import ImagePipeline

# Load the trained pipeline
model_path = "c:/Users/DELL/documents/aidex/Back-End/saved_models/image_1234567890_pipeline.pth"
checkpoint = torch.load(model_path, map_location='cpu')
pipeline = checkpoint['pipeline']

# Predict on new images
new_images = [
    "c:/path/to/new_cat.jpg",
    "c:/path/to/new_dog.jpg",
    "c:/path/to/mystery_pet.jpg"
]

results = pipeline.predict(new_images)

# Print results
for result in results['predictions']:
    print(f"\n{'='*60}")
    print(f"Image: {result['image_path']}")
    print(f"Prediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"\nAll probabilities:")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.1%}")
```

**Output**:
```
============================================================
Image: c:/path/to/new_cat.jpg
Prediction: cat
Confidence: 95.3%

All probabilities:
  cat: 95.3%
  dog: 4.7%

============================================================
Image: c:/path/to/new_dog.jpg
Prediction: dog
Confidence: 98.1%

All probabilities:
  cat: 1.9%
  dog: 98.1%

============================================================
Image: c:/path/to/mystery_pet.jpg
Prediction: cat
Confidence: 67.2%

All probabilities:
  cat: 67.2%
  dog: 32.8%
```

## Step 8: Make Predictions (API)

### Using the REST API

```python
import requests
import json

# Predict via API
response = requests.post(
    "http://localhost:8000/automl/predict-images",
    json={
        "image_paths": [
            "c:/path/to/new_cat.jpg",
            "c:/path/to/new_dog.jpg"
        ],
        "project_name": "image_1234567890"  # Use your project name
    }
)

results = response.json()

if results['status'] == 'success':
    for pred in results['results']:
        print(f"{pred['image_path']}: {pred['predicted_label']} ({pred['confidence']:.1%})")
else:
    print(f"Error: {results['error']}")
```

## Advanced Example: Multi-Class Classification

### Scenario: Classify 5 Types of Flowers

**Dataset Structure**:
```
flowers/
├── rose/
│   └── ... (20 images)
├── daisy/
│   └── ... (20 images)
├── tulip/
│   └── ... (20 images)
├── sunflower/
│   └── ... (20 images)
└── lily/
    └── ... (20 images)
```

### Training Configuration

For better accuracy with more classes:

```typescript
// In the frontend training call
await fetch("/api/automl/train-images", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    image_data: imageData,
    epochs: 20,  // ← More epochs for complex task
    batch_size: 32,
    learning_rate: 0.0005,  // ← Lower learning rate
    test_size: 0.2,
    model_names: ["resnet18", "resnet34", "mobilenet_v2"],  // ← Try multiple
    project_name: `flowers_${Date.now()}`
  }),
});
```

### Expected Results

With 5 classes and 20 images each:

```
Model Performance:
┌───────────┬──────────┬──────────┬───────────┬────────┐
│ Model     │ Accuracy │ F1 Score │ Precision │ Recall │
├───────────┼──────────┼──────────┼───────────┼────────┤
│ ★ resnet34│  82.00%  │  81.23%  │   83.50%  │ 80.00% │
│ resnet18  │  78.00%  │  77.45%  │   79.00%  │ 76.50% │
│ mobilenet │  75.00%  │  74.12%  │   76.00%  │ 73.00% │
└───────────┴──────────┴──────────┴───────────┴────────┘

Per-class metrics:
  rose: F1=0.85, Precision=0.87, Recall=0.83
  daisy: F1=0.82, Precision=0.80, Recall=0.84
  tulip: F1=0.78, Precision=0.81, Recall=0.75
  sunflower: F1=0.83, Precision=0.85, Recall=0.81
  lily: F1=0.78, Precision=0.77, Recall=0.79
```

## Performance Optimization

### Scenario: Large Dataset (1000+ images)

**Recommendations**:

1. **Increase Batch Size**:
   ```json
   { "batch_size": 64 }  // Faster training
   ```

2. **Use GPU**:
   ```powershell
   # Install CUDA version of PyTorch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **More Epochs**:
   ```json
   { "epochs": 30 }  // Better convergence
   ```

4. **Try Larger Models**:
   ```json
   { "model_names": ["resnet50", "efficientnet_b0"] }
   ```

### Expected Training Times

| Dataset Size | Model      | Device | Time    |
|-------------|------------|--------|---------|
| 100 images  | ResNet18   | CPU    | 2 min   |
| 100 images  | ResNet18   | GPU    | 30 sec  |
| 500 images  | ResNet18   | CPU    | 8 min   |
| 500 images  | ResNet18   | GPU    | 2 min   |
| 1000 images | ResNet50   | GPU    | 5 min   |
| 1000 images | EfficientNet| GPU   | 7 min   |

## Common Mistakes to Avoid

### ❌ Unbalanced Dataset
```
cat: 100 images
dog: 10 images  ← Too few!
```
**Solution**: Aim for similar numbers in each class.

### ❌ Too Few Images
```
Each class: 5 images  ← Not enough!
```
**Solution**: Use at least 20 images per class.

### ❌ Wrong Folder Structure
```
images/
├── IMG_001.jpg  ← No class labels!
├── IMG_002.jpg
```
**Solution**: Organize in class folders.

### ❌ Mixed Image Types
```
cat/
├── cat.jpg
├── dog.jpg  ← Wrong folder!
```
**Solution**: Ensure images are in correct class folders.

## Troubleshooting

### Issue: Low Accuracy (< 60%)

**Possible Causes**:
1. Not enough training data
2. Images mislabeled
3. Classes are too similar
4. Need more epochs

**Solutions**:
- Add more images (50+ per class)
- Verify labels are correct
- Train for 20-30 epochs
- Try ResNet50 instead of ResNet18

### Issue: Overfitting

**Symptoms**:
- Train accuracy: 95%
- Test accuracy: 65%

**Solutions**:
- More diverse training data
- Data augmentation (already enabled)
- Fewer epochs
- Use dropout (enabled in models)

### Issue: Training Stopped Early

**Cause**: Early stopping triggered (no improvement for 3 epochs)

**Solution**: This is normal! The model has converged.

## Summary

This complete example showed:
1. ✅ Dataset preparation
2. ✅ Uploading via UI
3. ✅ Image preprocessing
4. ✅ Training multiple models
5. ✅ Viewing results
6. ✅ Making predictions
7. ✅ Performance optimization
8. ✅ Troubleshooting

**Key Takeaways**:
- Start with small datasets to test
- Ensure balanced classes
- Use GPU for faster training
- Try multiple models to find the best
- Monitor per-class metrics
- Validate on separate test images

---

**Next Steps**: Try with your own image dataset! 🚀
