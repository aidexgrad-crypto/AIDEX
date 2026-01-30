"""
Explainability API — glue between trained model, SHAP (XAIEngine), and human-readable output.

- Takes the final trained model + data, calls SHAP, returns one clean explanation text.
- Also exposes a context dict for the chatbot (global, local, feature impact, performance, limitations).
"""

from .xai_engine import XAIEngine, simple_text_explanation
from .explanation_text import ExplanationTextGenerator


def get_explanation_context(
    model,
    X_train,
    X_test,
    index=0,
    top_n=5,
    model_metrics=None,
):
    """
    Build full explanation context from the trained model using SHAP.

    Args:
        model: Trained sklearn-style model (e.g. RandomForestClassifier).
        X_train: Training features (DataFrame).
        X_test: Test features (DataFrame).
        index: Row index in X_test for local explanation.
        top_n: Number of top features for global/local.
        model_metrics: Optional dict, e.g. {"accuracy": 0.95, "precision": 0.94, "recall": 0.93}.

    Returns:
        dict with:
            - global_explanation: str
            - local_explanation: str
            - feature_impact: list of (feature_name, impact) for local
            - model_performance: str
            - limitations: str
    """
    engine = XAIEngine(model, X_train, X_test)
    engine.compute_shap_values()

    text_gen = ExplanationTextGenerator()

    global_top = engine.get_global_top_features(top_n=top_n)
    local_top = engine.get_top_features(index=index, top_n=top_n)

    global_explanation = text_gen.generate_global_explanation(global_top)
    local_explanation = text_gen.generate_local_explanation(local_top)

    # Feature impact as list of (name, value) for local prediction
    feature_impact = local_top

    # LIME explanation (alternative local explanation)
    lime_explanation = ""
    try:
        lime_top = engine.get_lime_top_features(index=index, top_n=top_n)
        lime_explanation = text_gen.generate_lime_explanation(lime_top)
    except Exception:
        lime_explanation = ""

    # Model performance text
    if model_metrics:
        parts = [f"{k}: {v:.2%}" if isinstance(v, float) else f"{k}: {v}" for k, v in model_metrics.items()]
        model_performance = "Model performance: " + ", ".join(parts) + "."
    else:
        model_performance = "Model performance metrics were not provided."

    # Limitations (standard XAI/ML caveats)
    limitations = (
        "Explanations are based on SHAP and reflect feature contributions on this dataset. "
        "They do not guarantee causation. The model may not generalize to very different data; "
        "use within the intended domain and monitor performance over time."
    )

    return {
        "global_explanation": global_explanation,
        "local_explanation": local_explanation,
        "feature_impact": feature_impact,
        "model_performance": model_performance,
        "limitations": limitations,
        "lime_explanation": lime_explanation,
    }


def explain_prediction(model, X_train, X_test, index=0, top_n=5, model_metrics=None):
    """
    Return one clean explanation text for a single prediction (and model overview).

    This is the main API used by downstream code (e.g. test_explainability.py).
    """
    ctx = get_explanation_context(
        model=model,
        X_train=X_train,
        X_test=X_test,
        index=index,
        top_n=top_n,
        model_metrics=model_metrics,
    )

    # One clean paragraph for the user
    lines = [
        "For this prediction:",
        ctx["local_explanation"],
        "",
        "Overall, the model relies on:",
        ctx["global_explanation"],
        "",
        ctx["model_performance"],
        "",
        "Limitations:",
        ctx["limitations"],
    ]
    return "\n".join(lines)


def generate_and_save_plots(
    model,
    X_train,
    X_test,
    index=0,
    output_dir="explainability_plots",
):
    """
    Generate SHAP and LIME graphs and save them to files (human-readable visual explanations).
    Returns the list of saved file paths and prints the same text explanation as explain_prediction.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    engine = XAIEngine(model, X_train, X_test)
    engine.compute_shap_values()

    paths = []
    # SHAP global (summary plot)
    p_global = os.path.join(output_dir, "shap_global_summary.png")
    engine.save_global_explanation(path=p_global)
    paths.append(p_global)

    # SHAP local (bar plot for one prediction)
    p_local = os.path.join(output_dir, "shap_local_prediction.png")
    engine.save_local_explanation(index=index, path=p_local)
    paths.append(p_local)

    # LIME local
    p_lime = os.path.join(output_dir, "lime_local_prediction.png")
    try:
        engine.save_lime_explanation(index=index, path=p_lime)
        paths.append(p_lime)
    except Exception:
        pass

    return paths
