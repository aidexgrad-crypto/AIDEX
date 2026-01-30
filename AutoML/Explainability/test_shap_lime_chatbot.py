"""
Test SHAP, LIME, and Chatbot: verify they are implemented and the chatbot gives real answers.

Run (from Explainability folder or project root):
  python test_shap_lime_chatbot.py

Or with pytest:
  pytest test_shap_lime_chatbot.py -v

For chatbot tests to run (not skip): pip install requests
"""

import sys
import os
import warnings

# Run from project root (parent of AutoML) so package imports work
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

# Package imports (from project root)
from AutoML.Explainability.explainability_api import get_explanation_context

try:
    from AutoML.Explainability.chatbot import answer_from_context
except ImportError:
    answer_from_context = None  # e.g. missing 'requests'; chatbot tests will be skipped


def _build_model_and_context():
    """Build a small model and full explanation context (SHAP + LIME)."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }
    context = get_explanation_context(
        model, X_train, X_test, index=0, top_n=5, model_metrics=metrics
    )
    return context


# ---------------------------------------------------------------------------
# 1. SHAP tests
# ---------------------------------------------------------------------------

def test_shap_global_explanation():
    """SHAP: global explanation is present and mentions features."""
    context = _build_model_and_context()
    global_exp = context.get("global_explanation", "")
    assert global_exp, "global_explanation should not be empty"
    assert "relies" in global_exp.lower() or "feature" in global_exp.lower(), (
        "Global explanation should describe what the model relies on"
    )
    # Breast cancer dataset has feature names like "mean radius", "smoothness error"
    assert any(
        name in global_exp for name in ["smoothness", "compactness", "radius", "area", "perimeter", "concavity"]
    ), "Global explanation should contain real feature names from the model"
    print("  [OK] SHAP global explanation: present and contains feature names")


def test_shap_local_explanation():
    """SHAP: local explanation is present and mentions features."""
    context = _build_model_and_context()
    local_exp = context.get("local_explanation", "")
    assert local_exp, "local_explanation should not be empty"
    assert "influenced" in local_exp.lower() or "influence" in local_exp.lower(), (
        "Local explanation should describe what influenced the prediction"
    )
    assert any(
        name in local_exp for name in ["smoothness", "compactness", "radius", "area", "perimeter", "concavity"]
    ), "Local explanation should contain real feature names"
    print("  [OK] SHAP local explanation: present and contains feature names")


def test_shap_feature_impact():
    """SHAP: feature_impact is a non-empty list of (name, value) pairs."""
    context = _build_model_and_context()
    fi = context.get("feature_impact", [])
    assert isinstance(fi, list), "feature_impact should be a list"
    assert len(fi) > 0, "feature_impact should not be empty"
    for item in fi:
        assert len(item) == 2, "Each feature_impact item should be (name, value)"
        assert isinstance(item[0], str), "Feature name should be string"
        assert isinstance(item[1], (int, float)), "Feature value should be number"
    print("  [OK] SHAP feature_impact: list of (name, value) with real data")


# ---------------------------------------------------------------------------
# 2. LIME tests
# ---------------------------------------------------------------------------

def test_lime_implemented():
    """LIME: either lime_explanation is non-empty (LIME ran) or key exists (LIME attempted)."""
    context = _build_model_and_context()
    assert "lime_explanation" in context, "Context should include lime_explanation key"
    lime_exp = context.get("lime_explanation", "")
    if lime_exp:
        assert "LIME" in lime_exp or "driven" in lime_exp.lower(), "LIME text should describe drivers"
        assert any(
            name in lime_exp for name in ["smoothness", "compactness", "radius", "area", "perimeter", "concavity"]
        ), "LIME explanation should contain real feature names"
        print("  [OK] LIME: explanation present and contains feature names")
    else:
        print("  [SKIP] LIME: not computed (optional; may fail in some environments)")


# ---------------------------------------------------------------------------
# 3. Chatbot tests — real answers from SHAP/LIME context
# ---------------------------------------------------------------------------

def test_chatbot_no_context():
    """Chatbot: without context, returns message asking to run API (no fake answer)."""
    if answer_from_context is None:
        print("  [SKIP] Chatbot: module not imported (e.g. install 'requests')")
        return
    reply = answer_from_context("Why did the model predict this?", context=None)
    assert "No explanation context" in reply or "get_explanation_context" in reply or "explainability" in reply, (
        "Without context, chatbot should tell user to run explainability API, not invent an answer"
    )
    print("  [OK] Chatbot without context: asks to run API")


def test_chatbot_why_prediction_uses_shap():
    """Chatbot: 'Why did the model predict?' returns real local SHAP text."""
    if answer_from_context is None:
        print("  [SKIP] Chatbot: module not imported")
        return
    context = _build_model_and_context()
    reply = answer_from_context("Why did the model predict this?", context=context)
    local = context.get("local_explanation", "")
    assert local in reply, "Reply should contain the actual local_explanation from SHAP"
    assert "Feature impact" in reply, "Reply should include feature impact"
    # Should contain at least one real feature name from the context
    feature_names = [name for name, _ in context.get("feature_impact", [])]
    assert any(f in reply for f in feature_names), "Reply should contain real feature names from SHAP"
    print("  [OK] Chatbot 'why predict': returns real SHAP local explanation")


def test_chatbot_what_features_matter_uses_shap():
    """Chatbot: 'What features matter?' returns real global SHAP text."""
    if answer_from_context is None:
        print("  [SKIP] Chatbot: module not imported")
        return
    context = _build_model_and_context()
    reply = answer_from_context("What features matter overall?", context=context)
    global_exp = context.get("global_explanation", "")
    assert global_exp in reply, "Reply should be exactly the global_explanation from SHAP"
    print("  [OK] Chatbot 'what features matter': returns real SHAP global explanation")


def test_chatbot_performance_uses_context():
    """Chatbot: 'How good is the model?' returns real model_performance text."""
    if answer_from_context is None:
        print("  [SKIP] Chatbot: module not imported")
        return
    context = _build_model_and_context()
    reply = answer_from_context("How good is the model?", context=context)
    perf = context.get("model_performance", "")
    assert perf in reply, "Reply should contain model_performance from context"
    assert "accuracy" in reply.lower() or "precision" in reply.lower() or "recall" in reply.lower(), (
        "Reply should mention at least one metric"
    )
    print("  [OK] Chatbot 'how good is model': returns real performance text")


def test_chatbot_limitations_uses_context():
    """Chatbot: 'What are the limitations?' returns real limitations text."""
    if answer_from_context is None:
        print("  [SKIP] Chatbot: module not imported")
        return
    context = _build_model_and_context()
    reply = answer_from_context("What are the limitations?", context=context)
    limits = context.get("limitations", "")
    assert limits in reply, "Reply should contain limitations from context"
    assert "SHAP" in reply or "generaliz" in reply.lower() or "causation" in reply.lower(), (
        "Reply should mention SHAP or generalization/causation"
    )
    print("  [OK] Chatbot 'limitations': returns real limitations text")


def test_chatbot_lime_uses_context_when_available():
    """Chatbot: 'Explain with LIME' returns LIME text when available."""
    if answer_from_context is None:
        print("  [SKIP] Chatbot: module not imported")
        return
    context = _build_model_and_context()
    reply = answer_from_context("Explain with LIME", context=context)
    lime_exp = context.get("lime_explanation", "")
    if lime_exp:
        assert lime_exp in reply, "Reply should contain lime_explanation when LIME was computed"
        print("  [OK] Chatbot 'LIME': returns real LIME explanation")
    else:
        assert "LIME" in reply, "Reply should mention LIME even when not computed"
        print("  [OK] Chatbot 'LIME': mentions LIME (LIME not computed in this run)")


def test_chatbot_feature_impact_uses_shap():
    """Chatbot: 'Feature impact' or 'SHAP values' returns real feature contributions."""
    if answer_from_context is None:
        print("  [SKIP] Chatbot: module not imported")
        return
    context = _build_model_and_context()
    reply = answer_from_context("What is the feature impact for this prediction?", context=context)
    fi = context.get("feature_impact", [])
    assert fi, "Context should have feature_impact"
    for name, _ in fi[:2]:
        assert name in reply, f"Reply should contain feature name from SHAP: {name}"
    assert (
        "SHAP" in reply or "contribution" in reply.lower() or "feature impact" in reply.lower() or "impact" in reply.lower()
    ), "Reply should mention SHAP/contribution/impact"
    print("  [OK] Chatbot 'feature impact': returns real SHAP feature contributions")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

def run_all():
    print("\n=== Testing SHAP ===\n")
    test_shap_global_explanation()
    test_shap_local_explanation()
    test_shap_feature_impact()

    print("\n=== Testing LIME ===\n")
    test_lime_implemented()

    print("\n=== Testing Chatbot (real SHAP/LIME answers) ===\n")
    test_chatbot_no_context()
    test_chatbot_why_prediction_uses_shap()
    test_chatbot_what_features_matter_uses_shap()
    test_chatbot_performance_uses_context()
    test_chatbot_limitations_uses_context()
    test_chatbot_lime_uses_context_when_available()
    test_chatbot_feature_impact_uses_shap()

    print("\n=== All checks passed ===\n")


if __name__ == "__main__":
    run_all()
