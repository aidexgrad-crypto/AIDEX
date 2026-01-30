"""
Chatbot for Explainability: answers user questions about
- Why the prediction happened (local explanation)
- What features matter (global / feature impact)
- Model performance and limitations

Uses explanation context from explainability_api when available;
with Grok API for natural language, or a keyword-based fallback when no API key.
"""

import os
import re

# Grok (xAI) configuration
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_ENDPOINT = "https://api.x.ai/v1/chat/completions"


def _context_to_system_text(context):
    """Turn explanation context dict into a short paragraph for the LLM."""
    if not context:
        return ""
    fi = context.get("feature_impact") or []
    fi_str = str([f"{n}: {v:.3f}" for n, v in fi]) if fi else "[]"
    parts = [
        "Use ONLY the following facts when answering; do not invent numbers.",
        "",
        "Local (this prediction): " + (context.get("local_explanation") or ""),
        "Global (what matters overall): " + (context.get("global_explanation") or ""),
        "Feature impact (top drivers for this prediction): " + fi_str,
        "Model performance: " + (context.get("model_performance") or ""),
        "Limitations: " + (context.get("limitations") or ""),
    ]
    if context.get("lime_explanation"):
        parts.append("LIME (alternative local explanation): " + context["lime_explanation"])
    return "\n".join(parts)


def answer_from_context(user_input: str, context: dict) -> str:
    """
    Answer from explanation context when no external API is used.
    Uses keyword matching so the chatbot always gives real SHAP-based (and LIME) answers.
    """
    if not context:
        return (
            "No explanation context is available. Run the explainability API first "
            "(e.g. get_explanation_context or explain_prediction) with your trained model and data, "
            "then ask again. The chatbot answers from that SHAP/LIME context."
        )

    user_lower = user_input.lower().strip()
    local = context.get("local_explanation") or ""
    global_ = context.get("global_explanation") or ""
    feature_impact = context.get("feature_impact") or []
    performance = context.get("model_performance") or ""
    limitations = context.get("limitations") or ""
    lime_text = context.get("lime_explanation") or ""

    # LIME / alternative explanation (check before generic "explain")
    if re.search(r"\blime\b|\balternative\b|\banother (way|explanation)\b", user_lower):
        if lime_text:
            return lime_text
        return "LIME explanation was not computed for this context. Use explainability_api with LIME enabled."

    # Why did the model predict / why this prediction / explain prediction
    if re.search(r"\bwhy\b|\bpredict|prediction|result\b|\bexplain\b|\bhow (did|does)\b", user_lower):
        fi = _format_feature_impact(feature_impact)
        return local + "\n\n(Feature impact: " + fi + ")"

    # What features matter / which features / global importance / top / drivers
    if re.search(r"\b(feature|important|matter|drive|influence|top|driver|overall)\b", user_lower):
        return global_

    # Feature impact / contribution / SHAP values for this prediction
    if re.search(r"\b(impact|contribution|contributions|shap|value|weight)\b", user_lower):
        fi = _format_feature_impact(feature_impact)
        return "For this prediction, feature contributions (SHAP): " + fi + "."

    # Performance / accuracy / how good
    if re.search(r"\b(performance|accuracy|precision|recall|how good|metric)\b", user_lower):
        return performance

    # Limitations / caveats / trust
    if re.search(r"\b(limit|limitation|limitations|trust|caveat|caveats|rely|reliable|generaliz|generalize|error)\b", user_lower):
        return limitations

    # Default: give a short summary using real SHAP/LIME data
    lines = [
        "Here’s what we have from SHAP for this model:",
        "- For this prediction: " + local,
        "- Overall important features: " + global_,
        "- Model performance: " + performance,
        "- Limitations: " + limitations,
    ]
    if lime_text:
        lines.append("- LIME (alternative): " + lime_text[:200] + ("..." if len(lime_text) > 200 else ""))
    lines.append("\nAsk: “Why did the model predict this?”, “What features matter?”, “How good is the model?”, “Limitations?”, or “Explain with LIME”.")
    return "\n".join(lines)


def _format_feature_impact(feature_impact, max_items=5):
    if not feature_impact:
        return "none"
    parts = [f"{name} ({val:+.3f})" for name, val in feature_impact[:max_items]]
    return "; ".join(parts)


def chat_with_grok(prompt: str, context: dict = None) -> str:
    """
    Send a prompt to Grok (xAI) with optional explanation context.
    If GROK_API_KEY is not set, falls back to answer_from_context when context is provided.
    """
    if not GROK_API_KEY:
        if context:
            return answer_from_context(prompt, context)
        return f"(Fallback: no API key and no explanation context.) Ask after loading model explanations. You asked: {prompt}"

    system_content = (
        "You are an explainability assistant for an AutoML system. "
        "You explain model predictions, feature importance, performance, and limitations "
        "in simple language for non-technical users. "
        "Only use the facts provided; do not invent numbers or features."
    )

    if context:
        system_content += "\n\n" + _context_to_system_text(context)

    payload = {
        "model": "grok-1",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }

    try:
        import requests
        response = requests.post(
            GROK_ENDPOINT,
            headers={
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        if context:
            return answer_from_context(prompt, context)
        return f"(Fallback explanation) {prompt}"


if __name__ == "__main__":
    try:
        from explainability_api import get_explanation_context
    except ImportError:
        from AutoML.Explainability.explainability_api import get_explanation_context

    print("Explainability Chatbot (type 'exit' to quit)\n")

    if not GROK_API_KEY:
        print("GROK_API_KEY not set. Using local SHAP-based answers when context is loaded.\n")

    # Example: load context from a toy model (so the chatbot has real data to answer from)
    context = None
    try:
        import pandas as pd
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
        }
        context = get_explanation_context(model, X_train, X_test, index=0, model_metrics=metrics)
        print("Explanation context loaded (one sample). You can ask about the prediction, features, performance, limitations.\n")
    except Exception as e:
        print(f"Could not load example context: {e}. Chatbot will use fallback until you provide context.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            print("Chatbot: Goodbye.")
            break
        response = chat_with_grok(user_input, context=context)
        print("Chatbot:", response)
