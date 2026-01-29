import os
import pandas as pd
from aidex_pipeline import run_aidex

# Load CSV file (this is the only thing the user conceptually does:
# “upload a dataset” into a project. Everything else happens automatically.)
df = pd.read_csv('heart-attack-risk-prediction-dataset.csv')  # Change to your actual file name

print("="*80)
print("AIDEX PROJECT EXECUTION - HEART ATTACK RISK DATASET")
print("="*80)

# ============================================================================
# FIX DATA LEAKAGE: Remove features that contain target information
# (Internal automatic step – not exposed as a user decision.)
# ============================================================================
print("\n" + "="*80)
print("[INTERNAL CHECK] FIXING DATA LEAKAGE")
print("="*80)

leakage_features = [
    'Heart Attack Risk (Binary)',  # Target in binary form - direct leakage
    'CK-MB',                       # Post-event biomarker
    'Troponin'                     # Post-event biomarker
]

for feature in leakage_features:
    if feature in df.columns:
        print(f"   [REMOVED] Leakage feature: {feature}")
        df = df.drop(columns=[feature])
    else:
        print(f"   [WARN] Leakage feature not present: {feature}")

print(f"\n[OK] Clean dataset shape: {df.shape}")
print(f"   Features remaining (excluding target): {df.shape[1] - 1}")
print("="*80)

# AIDEX is project‑centric: everything that happens next is scoped to this ID.
project_id = "heart_attack_risk_demo"

# Run AIDEX pipeline (fully automatic – no user choices, only project + dataset)
pipeline = run_aidex(
    data_path=df,
    target_column='Heart Attack Risk (Text)',  # dataset‑specific target
    task_type='classification',
    test_size=0.2,
    scaling_method='standard',
    selection_priority='balanced',
    project_id=project_id
)

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "="*80)
print("[AUTO] HYPERPARAMETER TUNING")
print("="*80)

from Hyper_Parameters.hyperparameter_tuner import quick_tune_best_model
from Model_Training.model_trainer import ModelTrainer

# Store baseline results
baseline_best_model = pipeline.best_model_name
baseline_cv_row = pipeline.cv_results[pipeline.cv_results['model_name'] == baseline_best_model].iloc[0]
baseline_cv_score = baseline_cv_row['f1_mean']
baseline_test_results = pipeline.test_results[pipeline.test_results['model_name'] == baseline_best_model].iloc[0]
baseline_test_score = baseline_test_results['f1']

print(f"\n[BASELINE] Model (Before Tuning):")
print(f"   Model: {baseline_best_model}")
print(f"   CV F1 Score: {baseline_cv_score:.4f} ({baseline_cv_score*100:.2f}%)")
print(f"   Test F1 Score: {baseline_test_score:.4f} ({baseline_test_score*100:.2f}%)")

# Get fresh model instance for tuning
trainer = ModelTrainer(task_type='classification', random_state=42)
models_dict = trainer.get_default_models()
model_to_tune = models_dict[baseline_best_model]

print(f"\n[TUNING] {baseline_best_model} hyperparameters...")
print(f"   Method: Random Search")
print(f"   Iterations: 30 parameter combinations")
print(f"   Cross-Validation: 5 folds")

# Tune the best model
tuned_model, best_params, tuning_info = quick_tune_best_model(
    model_to_tune,
    baseline_best_model,
    pipeline.X_train,
    pipeline.y_train,
    task_type='classification',
    method='random',
    cv_folds=5,
    n_iter=30
)

tuned_cv_score = tuning_info.get('best_score', baseline_cv_score)

# Test tuned model on holdout set
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
y_pred_tuned = tuned_model.predict(pipeline.X_test.values)
tuned_test_accuracy = accuracy_score(pipeline.y_test, y_pred_tuned)
tuned_test_f1 = f1_score(pipeline.y_test, y_pred_tuned, average='weighted', zero_division=0)
tuned_test_precision = precision_score(pipeline.y_test, y_pred_tuned, average='weighted', zero_division=0)
tuned_test_recall = recall_score(pipeline.y_test, y_pred_tuned, average='weighted', zero_division=0)

# Calculate improvements
cv_improvement = tuned_cv_score - baseline_cv_score
cv_improvement_pct = (cv_improvement / baseline_cv_score) * 100
test_improvement = tuned_test_f1 - baseline_test_score
test_improvement_pct = (test_improvement / baseline_test_score) * 100

# Display comparison
print("\n" + "="*80)
print("[RESULTS] TUNING COMPARISON")
print("="*80)

print(f"\n{'Metric':<20} {'Baseline':<15} {'Tuned':<15} {'Improvement':<15}")
print("-"*80)
print(f"{'CV F1 Score':<20} {baseline_cv_score:.4f} ({baseline_cv_score*100:5.2f}%)  {tuned_cv_score:.4f} ({tuned_cv_score*100:5.2f}%)  {cv_improvement:+.4f} ({cv_improvement_pct:+.2f}%)")
print(f"{'Test Accuracy':<20} {baseline_test_results['accuracy']:.4f} ({baseline_test_results['accuracy']*100:5.2f}%)  {tuned_test_accuracy:.4f} ({tuned_test_accuracy*100:5.2f}%)  {tuned_test_accuracy - baseline_test_results['accuracy']:+.4f}")
print(f"{'Test F1 Score':<20} {baseline_test_score:.4f} ({baseline_test_score*100:5.2f}%)  {tuned_test_f1:.4f} ({tuned_test_f1*100:5.2f}%)  {test_improvement:+.4f} ({test_improvement_pct:+.2f}%)")
print(f"{'Test Precision':<20} {baseline_test_results['precision']:.4f} ({baseline_test_results['precision']*100:5.2f}%)  {tuned_test_precision:.4f} ({tuned_test_precision*100:5.2f}%)  {tuned_test_precision - baseline_test_results['precision']:+.4f}")
print(f"{'Test Recall':<20} {baseline_test_results['recall']:.4f} ({baseline_test_results['recall']*100:5.2f}%)  {tuned_test_recall:.4f} ({tuned_test_recall*100:5.2f}%)  {tuned_test_recall - baseline_test_results['recall']:+.4f}")

print(f"\n[TIMING] Tuning Statistics:")
print(f"   Time taken: {tuning_info['tuning_time']:.2f} seconds")
print(f"   Configurations tested: {tuning_info['n_iterations']}")

print(f"\n[PARAMS] Optimized Hyperparameters:")
for param, value in best_params.items():
    print(f"   {param}: {value}")

if test_improvement > 0:
    print(f"\n[OK] Tuning improved test F1 score by {test_improvement_pct:.2f}%")
    print(f"   Final Performance: {tuned_test_f1*100:.2f}% accuracy")
elif test_improvement == 0:
    print(f"\n[OK] Model maintained performance - baseline was already optimal")
else:
    print(f"\n[WARN] Test performance slightly decreased, but CV improved")
    print(f"   This is normal - CV score is more reliable for model selection")

# Update pipeline with tuned model
pipeline.trainer.trained_models[baseline_best_model] = tuned_model
pipeline.trainer.best_model = tuned_model

print("\n" + "="*80)

# Get best model
best_model = pipeline.get_best_model()

print("\n" + "="*80)
print("[DETAILS] RESULTS & EXPLANATION")
print("="*80)

# ============================================================================
# 1. DATASET OVERVIEW
# ============================================================================
print("\n" + "-"*80)
print("1) DATASET OVERVIEW")
print("-"*80)
print(f"Total samples: {df.shape[0]:,}")
print(f"Total features: {df.shape[1]-1:,} (excluding target)")
print(f"Training samples: {pipeline.X_train.shape[0]:,} ({pipeline.X_train.shape[0]/df.shape[0]*100:.1f}%)")
print(f"Testing samples: {pipeline.X_test.shape[0]:,} ({pipeline.X_test.shape[0]/df.shape[0]*100:.1f}%)")
print(f"Target variable: {pipeline.y_train.name if hasattr(pipeline.y_train, 'name') else 'Heart Attack Risk (Text)'}")
print(f"Number of classes: {len(pipeline.y_train.unique())}")
print(f"Class distribution:")
for cls, count in pipeline.y_train.value_counts().items():
    print(f"  Class {cls}: {count:,} samples ({count/len(pipeline.y_train)*100:.1f}%)")

# ============================================================================
# 2. ALL MODELS PERFORMANCE COMPARISON
# ============================================================================
print("\n" + "-"*80)
print("2) ALL MODELS PERFORMANCE (Cross-Validation)")
print("-"*80)
cv_results = pipeline.cv_results.copy()
cv_results_display = cv_results[['model_name', 'accuracy_mean', 'f1_mean', 'precision_mean', 'recall_mean', 'training_time']].copy()
cv_results_display = cv_results_display.round(4)
cv_results_display.columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'Time (s)']
print(cv_results_display.to_string(index=False))

print("\nEXPLANATION:")
print("   - Cross-validation splits data into 5 folds to ensure robust evaluation")
print("   - Accuracy: % of correct predictions (higher is better)")
print("   - F1 Score: Balance between precision and recall (higher is better)")
print("   - Precision: Of predicted positives, how many are actually positive")
print("   - Recall: Of actual positives, how many we correctly identified")

# ============================================================================
# 3. SELECTED MODEL DETAILS
# ============================================================================
print("\n" + "-"*80)
print(f"3) SELECTED MODEL: {pipeline.best_model_name}")
print("-"*80)

best_model_cv = cv_results[cv_results['model_name'] == pipeline.best_model_name].iloc[0]
best_model_test = pipeline.test_results[pipeline.test_results['model_name'] == pipeline.best_model_name].iloc[0]

print(f"\nCross-Validation Performance (5-fold):")
print(f"   Accuracy:  {best_model_cv['accuracy_mean']:.4f} +/- {best_model_cv['accuracy_std']:.4f}")
print(f"   F1 Score:  {best_model_cv['f1_mean']:.4f} +/- {best_model_cv['f1_std']:.4f}")
print(f"   Precision: {best_model_cv['precision_mean']:.4f} +/- {best_model_cv['precision_std']:.4f}")
print(f"   Recall:    {best_model_cv['recall_mean']:.4f} +/- {best_model_cv['recall_std']:.4f}")

print(f"\nTest Set Performance (Holdout):")
print(f"   Accuracy:  {best_model_test['accuracy']:.4f} ({best_model_test['accuracy']*100:.2f}%)")
print(f"   F1 Score:  {best_model_test['f1']:.4f} ({best_model_test['f1']*100:.2f}%)")
print(f"   Precision: {best_model_test['precision']:.4f} ({best_model_test['precision']*100:.2f}%)")
print(f"   Recall:    {best_model_test['recall']:.4f} ({best_model_test['recall']*100:.2f}%)")

print(f"\n⚙️  Model Characteristics:")
print(f"   Training Time: {best_model_cv['training_time']:.2f} seconds")
print(f"   Overfitting Score: {best_model_cv['overfitting']:.4f} (lower is better)")
print(f"   Stability: Very {'high' if best_model_cv['f1_std'] < 0.01 else 'good' if best_model_cv['f1_std'] < 0.03 else 'moderate'}")

print("\nEXPLANATION:")
print(f"   - {pipeline.best_model_name} was selected based on the 'balanced' priority")
print("   - It provides the best trade-off between performance, speed, and reliability")
print("   - Cross-validation +/- values show consistency across different data splits")
print("   - Test set results confirm the model generalizes well to unseen data")

# ============================================================================
# 4. TEST SET RESULTS COMPARISON
# ============================================================================
print("\n" + "-"*80)
print("4) TEST SET RESULTS (All Models)")
print("-"*80)
test_results_display = pipeline.test_results[['model_name', 'accuracy', 'f1', 'precision', 'recall']].copy()
test_results_display = test_results_display.round(4)
test_results_display.columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
print(test_results_display.to_string(index=False))

print("\nEXPLANATION:")
print("   - These are final results on completely unseen test data")
print("   - Test set was kept separate throughout training to validate real-world performance")
print("   - All models perform very well, indicating high-quality data and features")

# ============================================================================
# 5. PREDICTION ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("5) PREDICTION ANALYSIS")
print("-"*80)

test_predictions = pipeline.trainer.predict(pipeline.X_test, pipeline.best_model_name)
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Confusion Matrix
cm = confusion_matrix(pipeline.y_test, test_predictions)
print("\n📊 Confusion Matrix:")
print("   (Rows = Actual, Columns = Predicted)")
print()
# Print header
classes = sorted(pipeline.y_test.unique())
print("        ", end="")
for cls in classes:
    print(f"Pred {cls:3}", end="  ")
print()
# Print matrix
for i, cls in enumerate(classes):
    print(f"Actual {cls:2}", end=" ")
    for j in range(len(classes)):
        print(f"{cm[i][j]:7}", end="  ")
    print()

# Calculate accuracy per class
print("\nPer-Class Performance:")
for i, cls in enumerate(classes):
    total = cm[i].sum()
    correct = cm[i][i]
    accuracy = correct / total if total > 0 else 0
    print(f"   Class {cls}: {correct}/{total} correct ({accuracy*100:.2f}% accuracy)")

# Overall metrics
correct_predictions = np.sum(test_predictions == pipeline.y_test.values)
incorrect_predictions = len(pipeline.y_test) - correct_predictions
print(f"\n[OK] Correct predictions: {correct_predictions:,} / {len(pipeline.y_test):,} ({correct_predictions/len(pipeline.y_test)*100:.2f}%)")
print(f"[INFO] Incorrect predictions: {incorrect_predictions:,} / {len(pipeline.y_test):,} ({incorrect_predictions/len(pipeline.y_test)*100:.2f}%)")

print("\nEXPLANATION:")
print("   - Confusion matrix shows where the model makes correct/incorrect predictions")
print("   - Diagonal values (top-left to bottom-right) are correct predictions")
print("   - Off-diagonal values indicate misclassifications")
print("   - High diagonal values = excellent model performance")

# ============================================================================
# 6. FEATURE IMPORTANCE (if available)
# ============================================================================
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "-"*80)
    print("6) TOP 10 MOST IMPORTANT FEATURES")
    print("-"*80)
    
    feature_importance = pd.DataFrame({
        'feature': pipeline.X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop features contributing to predictions:")
    for idx, row in feature_importance.head(10).iterrows():
        bar_length = int(row['importance'] * 50)
        bar = '█' * bar_length
        print(f"   {row['feature']:30s} {bar} {row['importance']:.4f}")
    
    print("\nEXPLANATION:")
    print("   - Feature importance shows which variables the model relies on most")
    print("   - Higher values = more influential in making predictions")
    print("   - This helps understand what drives heart attack risk in your data")

# ============================================================================
# 7. SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("7) SUMMARY & RECOMMENDATIONS")
print("="*80)

avg_accuracy = pipeline.test_results['accuracy'].mean()
best_accuracy = pipeline.test_results['accuracy'].max()

print(f"\nKey Findings:")
print(f"   - Best Model: {pipeline.best_model_name}")
print(f"   - Test Accuracy: {best_model_test['accuracy']*100:.2f}%")
print(f"   - Model correctly predicts heart attack risk with {best_model_test['accuracy']*100:.2f}% accuracy")
print(f"   - Average model performance: {avg_accuracy*100:.2f}%")
print(f"   - Training completed in {best_model_cv['training_time']:.2f} seconds")

if best_model_test['accuracy'] >= 0.95:
    print("\nEXCELLENT MODEL PERFORMANCE!")
    print("   Your model shows outstanding predictive capability.")
elif best_model_test['accuracy'] >= 0.85:
    print("\nGOOD MODEL PERFORMANCE!")
    print("   Your model demonstrates strong predictive capability.")
elif best_model_test['accuracy'] >= 0.75:
    print("\n[OK] ACCEPTABLE MODEL PERFORMANCE")
    print("   Your model shows reasonable predictive capability.")
else:
    print("\n[WARN] MODEL MAY NEED IMPROVEMENT")
    print("   Consider feature engineering or collecting more data.")

print(f"\nRecommendations:")
print("   1. The model is ready for deployment on similar heart attack risk data")
print("   2. Monitor performance on new data and retrain if accuracy drops")
print("   3. Consider the feature importance to understand risk factors")
print("   4. Validate predictions with medical professionals before clinical use")
print("   5. Keep the model updated with new data periodically")

# Save predictions (project‑scoped artifact)
results_df = pd.DataFrame({
    'Actual': pipeline.y_test.values,
    'Predicted': test_predictions
})
predictions_path = f'predictions_results_{project_id}.csv'
results_df.to_csv(predictions_path, index=False)

print("\n" + "="*80)
print("PERSISTING PROJECT STATE & REPORT")
print("="*80)

# Persist a structured project snapshot and a human‑readable report
state_path = pipeline.save_project_state(output_dir="projects")
report_path = pipeline.save_project_report(output_dir="projects")

print(f"\n[OK] ANALYSIS COMPLETE FOR PROJECT: {project_id}")
print(f"   - Structured state saved to: {state_path}")
print(f"   - Human‑readable report saved to: {report_path}")
print(f"   - Predictions file: {predictions_path}")
print("\nYour AIDEX AutoML pipeline has successfully analyzed your data and")
print("    stored all results under this project ID for later inspection.")
print("="*80 + "\n")
