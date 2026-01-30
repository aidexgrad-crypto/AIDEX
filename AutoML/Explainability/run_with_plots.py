"""
Generate SHAP and LIME graphs (human-readable visual explanations) and save them to files.
Also prints the same text explanation as the other scripts.

Run from Explainability folder: python run_with_plots.py
Graphs are saved in ./explainability_plots/ (SHAP global, SHAP local, LIME local).
"""
import matplotlib
matplotlib.use("Agg")  # no GUI needed when saving to file

import sys
import os
import warnings

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "../.."))
if _root not in sys.path:
    sys.path.insert(0, _root)

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

try:
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score
except ImportError:
    print("Missing dependency. Run: pip install pandas scikit-learn shap lime")
    sys.exit(1)

from AutoML.Explainability.explainability_api import (
    explain_prediction,
    get_explanation_context,
    generate_and_save_plots,
)

OUTPUT_DIR = os.path.join(_here, "explainability_plots")


def main():
    print("Loading data and training model...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }

    print("\n" + "=" * 60)
    print("TEXT EXPLANATION (human-readable)")
    print("=" * 60)
    text = explain_prediction(
        model=model,
        X_train=X_train,
        X_test=X_test,
        index=0,
        model_metrics=metrics,
    )
    print(text)

    print("\n" + "=" * 60)
    print("GENERATING GRAPHS (SHAP + LIME)")
    print("=" * 60)
    paths = generate_and_save_plots(
        model=model,
        X_train=X_train,
        X_test=X_test,
        index=0,
        output_dir=OUTPUT_DIR,
    )
    print("Graphs saved to:")
    for p in paths:
        print("  ", os.path.abspath(p))
    print("\nOpen the PNG files above to see SHAP (global + local) and LIME explanations.")
    print("Done.")


if __name__ == "__main__":
    main()
