# Setup Guide for Image Training Feature

## Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- Existing AIDEX installation

## Installation Steps

### 1. Install Python Dependencies

Navigate to your AIDEX directory and activate your virtual environment:

```powershell
cd c:\Users\DELL\documents\aidex
.\venv\Scripts\Activate.ps1
```

Install the new image processing dependencies:

```powershell
# Install PyTorch (CPU version - faster to install)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# OR install PyTorch with CUDA support for GPU acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Pillow for image processing
pip install Pillow
```

**Note**: If you have an NVIDIA GPU, install the CUDA version for much faster training!

### 2. Verify Installation

Test that PyTorch is installed correctly:

```powershell
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch version: 2.x.x
TorchVision version: 0.x.x
CUDA available: True  (or False if CPU only)
```

### 3. Start the Backend Server

```powershell
cd c:\Users\DELL\documents\aidex
.\venv\Scripts\Activate.ps1

# Set the Python path
$env:PYTHONPATH = "C:\Users\DELL\documents\aidex\AutoML"

# Start the server
python -m uvicorn Back-End.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Start the Frontend

In a new terminal:

```powershell
cd c:\Users\DELL\documents\aidex\front-end
npm run dev
```

You should see:
```
ready - started server on 0.0.0.0:3000
```

### 5. Test the Feature

1. Open your browser to `http://localhost:3000`
2. Log in (if authentication is enabled)
3. Upload an image folder with labeled subdirectories:
   ```
   my_images/
   ├── cats/
   │   ├── cat1.jpg
   │   └── cat2.jpg
   └── dogs/
       ├── dog1.jpg
       └── dog2.jpg
   ```
4. Go to Overview page
5. Go to Preprocessing page
6. Wait for image preprocessing to complete
7. Click "Start Image AutoML Training"
8. Wait for training to complete (may take a few minutes)
9. View results!

## Troubleshooting

### Issue: "Import torch could not be resolved"

**Solution**: PyTorch is not installed. Run:
```powershell
pip install torch torchvision Pillow
```

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size or use CPU:
- Edit the training request in the frontend to use `batch_size: 16` or `8`
- Or switch to CPU-only PyTorch

### Issue: Training is very slow

**Solutions**:
1. Install CUDA version of PyTorch for GPU acceleration
2. Use smaller models like MobileNetV2
3. Reduce number of epochs
4. Increase batch size (if memory allows)

### Issue: Backend won't start

**Check**:
1. Virtual environment is activated: `.\venv\Scripts\Activate.ps1`
2. PYTHONPATH is set: `$env:PYTHONPATH = "C:\Users\DELL\documents\aidex\AutoML"`
3. No other process is using port 8000
4. All dependencies are installed: `pip list | Select-String "torch|vision|Pillow"`

### Issue: "No module named 'Image_Processing'"

**Solution**: Make sure the `__init__.py` file exists:
```powershell
Test-Path c:\Users\DELL\documents\aidex\AutoML\Image_Processing\__init__.py
```

If it returns `False`, the file is missing.

### Issue: Images not uploading

**Check**:
1. Backend is running on port 8000
2. Frontend is configured to connect to backend (check `.env` file)
3. Image upload directory exists: `c:\Users\DELL\documents\aidex\Back-End\temp_uploads\images\`

## Performance Tips

### For Best Accuracy
- Use at least 50 images per class
- Train for 20-30 epochs
- Use ResNet18 or ResNet50
- Ensure balanced class distribution

### For Fastest Training
- Use MobileNetV2 model
- Reduce epochs to 5-10
- Use smaller image size (128x128)
- Increase batch size if memory allows

### For GPU Training
1. Install CUDA Toolkit from NVIDIA
2. Install PyTorch with CUDA:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. Verify GPU is detected:
   ```powershell
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Quick Test Script

Save this as `test_image_training.py`:

```python
from AutoML.image_pipeline import run_image_pipeline

# Test with dummy data
print("Testing image pipeline...")

image_paths = ["test1.jpg", "test2.jpg"]
labels = ["cat", "dog"]

try:
    pipeline = run_image_pipeline(
        image_paths=image_paths,
        labels=labels,
        epochs=1,  # Just 1 epoch for testing
        model_names=["resnet18"],
    )
    print("✅ Pipeline works!")
except Exception as e:
    print(f"❌ Error: {e}")
```

Run it:
```powershell
python test_image_training.py
```

## File Checklist

Verify these files exist:

- [x] `AutoML/image_pipeline.py`
- [x] `AutoML/Image_Processing/__init__.py`
- [x] `AutoML/Image_Processing/image_loader.py`
- [x] `AutoML/Image_Processing/image_trainer.py`
- [x] `Back-End/main.py` (updated)
- [x] `front-end/app/api/automl/train-images/route.ts`
- [x] `front-end/app/api/automl/predict-images/route.ts`
- [x] `front-end/app/api/automl/upload-images/route.ts`
- [x] `front-end/app/preprocessing/page.tsx` (updated)
- [x] `IMAGE_CLASSIFICATION_GUIDE.md`
- [x] `IMAGE_TRAINING_SUMMARY.md`
- [x] `SETUP_GUIDE.md` (this file)

## Next Steps

After successful setup:

1. Try with a small dataset (10-20 images per class)
2. Test with 2-3 classes first
3. Use 5-10 epochs for initial testing
4. Once working, scale up to your full dataset
5. Experiment with different models
6. Fine-tune hyperparameters for best results

## Support

If you encounter issues:
1. Check the backend logs in the terminal
2. Check the browser console for frontend errors
3. Verify all files are in place
4. Ensure dependencies are installed
5. Try restarting both backend and frontend

## Useful Commands

```powershell
# Check if PyTorch is installed
pip show torch

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"

# List all installed packages
pip list

# Restart backend
# Ctrl+C to stop, then run:
python -m uvicorn Back-End.main:app --reload --host 0.0.0.0 --port 8000

# Restart frontend
# Ctrl+C to stop, then run:
npm run dev
```

---

**Status**: Ready to install and test! 🚀
