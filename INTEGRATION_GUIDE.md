# AutoML Integration Guide

## Overview

The AutoML pipeline has been successfully integrated with the preprocessing module. The cleaned data from preprocessing now flows directly into the AutoML training pipeline.

## How It Works

### 1. Data Flow Architecture

```
Upload Dataset → Data Quality Analysis → Preprocessing → AutoML Training → Best Model Selection
```

### 2. Backend Integration

#### Files Modified:

- **`Back-End/main.py`**: Added AutoML endpoints and data handling

#### New Endpoints:

**POST `/data-quality/apply`** (Updated)

- Applies cleaning decisions to the dataset
- **NEW**: Saves cleaned data to `cleaned_datasets/` directory
- Returns cleaning report with `cleaned_file_path`

**POST `/automl/train`** (New)

- Receives parameters for AutoML training
- Loads cleaned dataset from saved location
- Runs the AIDEX pipeline
- Returns model performance metrics

Request body:

```json
{
  "target_column": "column_name",
  "task_type": "classification",
  "test_size": 0.2,
  "scaling_method": "standard",
  "selection_priority": "balanced",
  "project_name": "my_project"
}
```

Response:

```json
{
  "status": "success",
  "best_model": "RandomForest",
  "cv_scores": {
    "accuracy": 0.95,
    "precision": 0.94,
    "recall": 0.96,
    "f1": 0.95
  },
  "test_scores": {
    "accuracy": 0.94,
    "precision": 0.93,
    "recall": 0.95,
    "f1": 0.94
  },
  "project_id": "my_project",
  "dataset_shape": [1000, 20],
  "target_column": "column_name"
}
```

### 3. Frontend Integration

#### Files Created:

- **`front-end/app/api/automl/train/route.ts`**: API route to communicate with backend

#### Files Modified:

- **`front-end/app/preprocessing/page.tsx`**: Added AutoML training UI

#### New Features:

1. **AutoML Training Button**: Appears after preprocessing is complete
2. **Training Progress Indicator**: Shows while models are being trained
3. **Results Display**: Shows the best model and performance metrics
4. **Error Handling**: Displays clear error messages if training fails

## Usage Instructions

### Step 1: Start the Backend

```bash
cd Back-End
uvicorn main:app --reload
```

The backend will run on `http://127.0.0.1:8000`

### Step 2: Start the Frontend

```bash
cd front-end
npm run dev
```

The frontend will run on `http://localhost:3000`

### Step 3: Use the Application

1. **Upload Dataset**: Go to the upload page and upload your CSV dataset
2. **Overview**: Review dataset statistics and select target column
3. **Preprocessing**: Click "Start Data Preparation" to clean the data
4. **AutoML Training**: After preprocessing completes, click "Start AutoML Training"
5. **View Results**: See the best model and its performance metrics

## Technical Details

### Data Storage

- **Original uploads**: `Back-End/temp_uploads/`
- **Cleaned datasets**: `Back-End/cleaned_datasets/`
- **Project results**: `projects/<project_name>/`

### AutoML Pipeline Features

The integration uses the complete AIDEX pipeline:

- Automatic feature type detection
- Missing value handling
- Feature scaling (Standard, MinMax, Robust)
- Categorical encoding
- Model training (Multiple algorithms)
- Cross-validation
- Model selection based on metrics
- Hyperparameter tuning support

### Supported Models

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Support Vector Machine
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Neural Network (MLP)

### Model Selection Criteria

- **Balanced**: Optimizes F1 score
- **Accuracy**: Maximizes accuracy
- **Precision**: Optimizes precision
- **Recall**: Optimizes recall

## Error Handling

The system handles various error scenarios:

- No cleaned dataset available
- Target column not found
- Invalid parameters
- Training failures
- Backend connection issues

All errors are displayed in the UI with descriptive messages.

## Future Enhancements

Potential improvements:

1. Support for regression tasks
2. Custom hyperparameter configuration from UI
3. Model comparison visualization
4. Feature importance display
5. Model deployment options
6. Batch prediction interface
7. Model versioning and tracking

## Troubleshooting

### Backend Not Responding

- Ensure backend is running on port 8000
- Check for Python dependency issues
- Verify AutoML modules are importable

### Training Fails

- Verify target column exists in cleaned data
- Check data has enough samples
- Ensure no data quality issues remain

### Frontend API Errors

- Check browser console for detailed errors
- Verify backend URL in API routes
- Ensure CORS is properly configured

## Dependencies

### Backend Requirements

- FastAPI
- pandas
- scikit-learn
- xgboost
- lightgbm
- All AIDEX AutoML modules

### Frontend Requirements

- Next.js
- React
- TypeScript

## Conclusion

The AutoML integration is complete and functional. The preprocessing output now seamlessly feeds into the AutoML pipeline, providing an end-to-end automated machine learning workflow.
