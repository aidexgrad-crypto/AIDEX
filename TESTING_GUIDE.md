# Quick Start Guide - AutoML Integration Testing

## Prerequisites

1. Git is installed (you've already resolved this)
2. Python virtual environment is set up
3. Node.js and npm are installed

## Step-by-Step Testing

### 1. Start the Backend Server

Open a terminal in the project root:

```bash
cd Back-End
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### 2. Start the Frontend Development Server

Open a NEW terminal in the project root:

```bash
cd front-end
npm run dev
```

You should see:

```
ready - started server on 0.0.0.0:3000, url: http://localhost:3000
```

### 3. Test the Integration

1. **Open Browser**: Navigate to `http://localhost:3000`

2. **Login/Auth**: Complete authentication if prompted

3. **Upload Dataset**:
   - Click "Upload Dataset"
   - Select `heart-attack-risk-prediction-dataset.csv` (or your dataset)
   - Wait for upload to complete

4. **Overview Page**:
   - Review dataset statistics
   - Select target column (e.g., "Heart Attack Risk (Text)")
   - Click "Continue to Preprocessing"

5. **Preprocessing Page**:
   - Click "Start Data Preparation"
   - Wait for preprocessing to complete
   - Review cleaning summary

6. **AutoML Training**:
   - Scroll down to see "Next Step: Train Models with AutoML" section
   - Click "Start AutoML Training"
   - Wait for training (this may take 2-5 minutes)
   - View results showing best model and performance metrics

### 4. Expected Results

After AutoML training completes, you should see:

- ✓ Training Completed Successfully!
- Best Model name (e.g., "RandomForest", "GradientBoosting")
- Performance metrics:
  - Test Accuracy: ~90-95%
  - F1 Score: ~90-95%
  - Precision: ~90-95%
  - Recall: ~90-95%

### 5. Verify Backend Storage

Check that files were created:

```bash
# Check uploaded file
dir Back-End\temp_uploads

# Check cleaned dataset
dir Back-End\cleaned_datasets

# Check AutoML project results
dir projects
```

## Troubleshooting

### Backend Won't Start

```bash
# Check if port 8000 is already in use
netstat -ano | findstr :8000

# If it's in use, kill the process or use a different port
python -m uvicorn main:app --reload --port 8001
# Then update frontend API routes to use port 8001
```

### Frontend Won't Start

```bash
# Clear cache and reinstall dependencies
cd front-end
rm -rf .next node_modules
npm install
npm run dev
```

### AutoML Training Fails

**Error: "No cleaned dataset available"**

- Solution: Make sure you completed the preprocessing step first
- The backend saves cleaned data in `Back-End/cleaned_datasets/`

**Error: "Target column not found"**

- Solution: Verify you selected a target column in the Overview page
- Check that the column wasn't removed during preprocessing

**Error: "Failed to communicate with backend"**

- Solution: Ensure backend is running on port 8000
- Check browser console for detailed error messages
- Verify CORS settings in backend

### Import Errors in Backend

```bash
# Activate virtual environment
cd Back-End
..\venv\Scripts\activate

# Install missing packages
pip install fastapi uvicorn pandas scikit-learn xgboost lightgbm
```

## Testing with Sample Data

Use the included `heart-attack-risk-prediction-dataset.csv`:

- **Target Column**: "Heart Attack Risk (Text)"
- **Task Type**: Classification
- **Expected Accuracy**: 90-95%
- **Expected Best Models**: RandomForest, GradientBoosting, or XGBoost

## API Testing (Optional)

Test backend endpoints directly:

```bash
# Test health check (if you add one)
curl http://127.0.0.1:8000/

# Test AutoML endpoint (after preprocessing)
curl -X POST http://127.0.0.1:8000/automl/train \
  -H "Content-Type: application/json" \
  -d '{"target_column": "Heart Attack Risk (Text)", "task_type": "classification"}'
```

## Next Steps After Testing

Once the integration works:

1. **Customize AutoML Parameters**: Modify the frontend to allow users to choose:
   - Task type (classification/regression)
   - Test size
   - Scaling method
   - Selection priority

2. **Add Model Persistence**: Save trained models to disk for later use

3. **Add Prediction Interface**: Create a page to make predictions with the trained model

4. **Add Model Comparison**: Show performance of all models, not just the best

5. **Add Visualization**: Display confusion matrix, ROC curves, feature importance

## Success Criteria

✅ Backend starts without errors
✅ Frontend starts without errors  
✅ Dataset uploads successfully
✅ Preprocessing completes and shows summary
✅ AutoML training button appears after preprocessing
✅ AutoML training completes successfully
✅ Results display with model name and metrics
✅ Cleaned dataset saved in `Back-End/cleaned_datasets/`
✅ No errors in browser console
✅ No errors in backend logs

## Support

If you encounter issues:

1. Check the browser console (F12) for frontend errors
2. Check the backend terminal for Python errors
3. Review `INTEGRATION_GUIDE.md` for detailed documentation
4. Verify all dependencies are installed
5. Ensure file paths are correct (Windows vs Linux)

Happy testing! 🚀
