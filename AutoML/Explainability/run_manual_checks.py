"""
Run manual checks non-interactively:
1. SHAP + API (full explanation)
2. Chatbot answers for each standard question (real SHAP/LIME context)
"""

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
except ImportError:
    print("Missing dependency: pandas not installed. Run: pip install pandas scikit-learn shap lime")
    sys.exit(1)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from AutoML.Explainability.explainability_api import get_explanation_context, explain_prediction
from AutoML.Explainability.chatbot import answer_from_context


def main():
    print("=" * 60)
    print("MANUAL CHECK 1: SHAP + API (full explanation)")
    print("=" * 60)

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

    full_text = explain_prediction(
        model=model, X_train=X_train, X_test=X_test, index=0, model_metrics=metrics
    )
    print(full_text)
    print()

    print("=" * 60)
    print("MANUAL CHECK 2: Chatbot (real SHAP/LIME answers)")
    print("=" * 60)

    context = get_explanation_context(
        model, X_train, X_test, index=0, model_metrics=metrics
    )

    questions = [
        "Why did the model predict this?",
        "What features matter?",
        "Which features drive the model?",
        "What is the feature impact for this prediction?",
        "How good is the model?",
        "What are the limitations?",
        "Explain with LIME",
    ]

    for q in questions:
        print("You:", q)
        reply = answer_from_context(q, context=context)
        print("Chatbot:", reply)
        print()

    print("=" * 60)
    print("MANUAL CHECK 3: No context (chatbot should ask to run API)")
    print("=" * 60)
    reply = answer_from_context("Why did the model predict this?", context=None)
    print("You: Why did the model predict this?")
    print("Chatbot:", reply)
    print()
    print("Done. All manual checks run.")


if __name__ == "__main__":
    main()
