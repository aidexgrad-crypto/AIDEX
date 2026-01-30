"""
Run all Explainability checks in one go:
1. test_explainability.py (SHAP + API)
2. test_shap_lime_chatbot.py (SHAP, LIME, chatbot tests)
3. run_manual_checks.py (manual checks 1-3)
4. ask_all_questions.py (every question type)

Run from Explainability folder: python run_everything.py
Or from AIDEX root: python AutoML/Explainability/run_everything.py
"""

import subprocess
import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
os.chdir(_here)

scripts = [
    ("test_explainability.py", "SHAP + API"),
    ("test_shap_lime_chatbot.py", "SHAP / LIME / Chatbot tests"),
    ("run_manual_checks.py", "Manual checks"),
    ("ask_all_questions.py", "All chatbot questions"),
    ("run_with_plots.py", "SHAP + LIME graphs (saved to explainability_plots/)"),
]

failed = []
for script, name in scripts:
    print("\n" + "=" * 60)
    print(f"Running: {name} ({script})")
    print("=" * 60)
    code = subprocess.call([sys.executable, script])
    if code != 0:
        failed.append(script)

if failed:
    print("\n*** FAILED:", ", ".join(failed))
    sys.exit(1)
print("\n*** All runs completed successfully.")
