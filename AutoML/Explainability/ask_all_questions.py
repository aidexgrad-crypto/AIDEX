"""
Ask every question type the chatbot is trained on (with real SHAP/LIME context).
One question per pattern, plus default and no-context cases.
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
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score
except ImportError:
    print("Missing dependency: pandas and/or scikit-learn not installed.")
    print("\nInstall with (run in terminal with your venv activated):")
    print("  pip install pandas scikit-learn shap lime")
    print("Or from AIDEX root:")
    print("  pip install -r AutoML/Explainability/requirements.txt")
    sys.exit(1)

from AutoML.Explainability.explainability_api import get_explanation_context
from AutoML.Explainability.chatbot import answer_from_context


# Every question type the bot handles (by keyword pattern)
ALL_QUESTIONS = [
    # --- LIME / alternative (checked first) ---
    "Explain with LIME",
    "Give me an alternative explanation",
    "Another way to explain this prediction?",
    # --- Why / predict / explain / how did/does ---
    "Why did the model predict this?",
    "Explain the prediction",
    "What was the result?",
    "How does the model predict?",
    # --- Features matter / drive / overall ---
    "What features matter?",
    "Which features drive the model?",
    "Top features?",
    "What's overall importance?",
    # --- Impact / contribution / SHAP / value / weight ---
    "What is the feature impact for this prediction?",
    "SHAP values?",
    "What are the contributions?",
    "Feature weights for this prediction?",
    # --- Performance / accuracy / metric ---
    "How good is the model?",
    "What's the performance?",
    "Accuracy?",
    "Precision and recall?",
    # --- Limitations / trust / caveats ---
    "What are the limitations?",
    "Can I trust this model?",
    "What are the caveats?",
    "Does it generalize?",
    # --- Default (no keyword match → summary) ---
    "Hello",
    "Tell me something",
]

NO_CONTEXT_QUESTION = "Why did the model predict this?"


def main():
    print("Building context (SHAP + LIME)...")
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
    context = get_explanation_context(model, X_train, X_test, index=0, model_metrics=metrics)
    print("Context ready.\n")

    print("=" * 70)
    print("ALL QUESTIONS WITH CONTEXT (every type the bot is trained on)")
    print("=" * 70)

    for i, q in enumerate(ALL_QUESTIONS, 1):
        print(f"\n[{i}] You: {q}")
        reply = answer_from_context(q, context=context)
        # Truncate long replies for readability; full reply is still computed
        if len(reply) > 400:
            print("Chatbot:", reply[:400].rstrip(), "...")
        else:
            print("Chatbot:", reply)

    print("\n" + "=" * 70)
    print("NO CONTEXT (bot should ask to run API, not invent an answer)")
    print("=" * 70)
    print(f"\nYou: {NO_CONTEXT_QUESTION}")
    reply = answer_from_context(NO_CONTEXT_QUESTION, context=None)
    print("Chatbot:", reply)
    print("\nDone.")


if __name__ == "__main__":
    main()
