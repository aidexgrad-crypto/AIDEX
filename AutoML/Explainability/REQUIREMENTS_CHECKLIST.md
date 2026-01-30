# Explainability Layer — Requirements Checklist

Requirements from supervisor: **Global explanation**, **Local explanation**, **Feature impact**, **Model performance**, **Error & limitations**, **Chatbot / Q&A**.

| # | Requirement | Status | Where it's done |
|---|-------------|--------|------------------|
| 1 | **Global explanation** | ✅ | `xai_engine.py`: `get_global_top_features()`, `global_explanation()`; `explainability_api.py`: `get_explanation_context()` → `global_explanation`; `explanation_text.py`: `generate_global_explanation()` |
| 2 | **Local explanation** | ✅ | `xai_engine.py`: `get_top_features()`, `local_explanation()`; `explainability_api.py`: `get_explanation_context()` → `local_explanation`; `explanation_text.py`: `generate_local_explanation()` |
| 3 | **Feature impact explanation** | ✅ | `xai_engine.py`: `get_top_features()` (SHAP values per feature); `explainability_api.py`: `feature_impact` in context; chatbot uses it for "what features matter" |
| 4 | **Model performance explanation** | ✅ | `explainability_api.py`: `get_explanation_context(model_metrics=...)` → `model_performance`; chatbot answers performance/accuracy questions |
| 5 | **Error & limitations explanation** | ✅ | `explainability_api.py`: `limitations` in context (SHAP caveats, generalization); chatbot answers limit/trust/caveat questions |
| 6 | **Chatbot / Q&A for users** | ✅ | `chatbot.py`: `chat_with_grok()`, `answer_from_context()` — answers **why prediction**, **what features matter**, **model limitations** using **real SHAP context** from API |

---

## SHAP (CORE) — Done

- **Global**: mean |SHAP| → top features text.
- **Local**: per-instance SHAP → "mainly influenced by X, Y, Z".
- **Human-readable**: `explanation_text.py` + `explainability_api.explain_prediction()`.

---

## LIME — Integrated

- **Engine**: `xai_engine.py`: `lime_explanation()`, `get_lime_top_features()`.
- **API**: `explainability_api.py`: LIME text in `get_explanation_context()` → `lime_explanation`.
- **Chatbot**: Answers "LIME" / "alternative explanation" with LIME-based text.

---

## Graphs (SHAP + LIME visual explanations)

- **Text** is always returned (human-readable sentences); **graphs** are generated when you run with plots.
- Run **`python run_with_plots.py`** (from `AutoML/Explainability`) to generate and save:
  - **shap_global_summary.png** — global feature importance (model-level)
  - **shap_local_prediction.png** — local SHAP bar plot for one prediction
  - **lime_local_prediction.png** — LIME explanation for the same prediction
- Graphs are saved in **`AutoML/Explainability/explainability_plots/`**. Open the PNG files to view them.
- **`run_everything.py`** now includes this step, so running all checks also generates the graphs.

---

## How to verify

1. **SHAP + API**: Run `test_explainability.py` — prints one full explanation (SHAP).
2. **Chatbot (real SHAP/LIME answers)**: Run `chatbot.py`. It loads context from `get_explanation_context()` (SHAP + LIME). Then ask:
   - "Why did the model predict this?" → local SHAP explanation + feature impact
   - "What features matter?" / "Which features drive the model?" → global SHAP
   - "Feature impact" / "SHAP values" → contribution list for this prediction
   - "How good is the model?" → model performance text
   - "What are the limitations?" → limitations text
   - "Explain with LIME" / "Alternative explanation" → LIME text
3. **No API key**: Without `GROK_API_KEY`, the chatbot uses `answer_from_context()` only — all answers come from the same SHAP/LIME context (no generic fallback).

---

## How to test (automated)

Run the dedicated test script (from `Explainability` folder or project root):

```bash
python test_shap_lime_chatbot.py
```

**What it checks:**

| Test | What is verified |
|------|------------------|
| **SHAP** | Global explanation exists and contains real feature names; local explanation exists and contains feature names; `feature_impact` is a non-empty list of (name, value). |
| **LIME** | `lime_explanation` is in context; when LIME runs, the text contains real feature names. |
| **Chatbot** | With no context → message to run API (no fake answer). With context → "Why predict?" returns actual `local_explanation`; "What features matter?" returns `global_explanation`; "How good is the model?" returns `model_performance`; "Limitations?" returns `limitations`; "LIME" returns LIME text; "Feature impact" returns real SHAP contributions. |

**Chatbot tests** require `requests` (they are skipped if the chatbot module cannot be imported). Install with: `pip install requests`.
