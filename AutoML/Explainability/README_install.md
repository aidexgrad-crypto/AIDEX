# Explainability – install dependencies

If you get `ModuleNotFoundError: No module named 'pandas'` (or `sklearn`, `shap`, `lime`), install the dependencies.

**With venv activated** (e.g. `(.venv) PS C:\Users\janah\Documents\AIDEX>`):

```powershell
pip install pandas scikit-learn shap lime
```

Or install everything from the project root:

```powershell
pip install -r requirements.txt
```

Then run:

```powershell
python ask_all_questions.py
```

(or `python AutoML\Explainability\ask_all_questions.py` from the AIDEX root)
