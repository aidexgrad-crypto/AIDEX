import sys
import os
import warnings

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

try:
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
except ImportError:
    print("Missing dependency: pandas not installed. Run: pip install pandas scikit-learn shap lime")
    sys.exit(1)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from AutoML.Explainability.explainability_api import explain_prediction


# -------------------------------------------------
# 1. Load dataset
# -------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target


# -------------------------------------------------
# 2. Train-test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------------------------
# 3. Train a temporary model (AutoML will replace this)
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)


# -------------------------------------------------
# 4. Explain ONE prediction using Explainability API
# -------------------------------------------------
explanation = explain_prediction(
    model=model,
    X_train=X_train,
    X_test=X_test,
    index=0
)


# -------------------------------------------------
# 5. Output explanation (for non-technical users)
# -------------------------------------------------
print("\n==============================")
print("FINAL EXPLANATION FOR USER")
print("==============================\n")

print(explanation)
