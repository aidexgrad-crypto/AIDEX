# Install and run (do this once)

The code **does** import pandas; the error means the **package** is not installed in your Python environment.

**With your venv activated** (you should see `(.venv)` in the prompt), run:

```powershell
pip install pandas scikit-learn shap lime
```

Or from the AIDEX folder:

```powershell
pip install -r AutoML/Explainability/requirements.txt
```

Then run everything:

**From AIDEX root** (recommended):

```powershell
python run_everything.py
```

**Or** from AIDEX root with full path:

```powershell
python AutoML\Explainability\run_everything.py
```

**From inside** `AutoML\Explainability` — use only the script name (no path):

```powershell
cd AutoML\Explainability
python run_everything.py
```

Do **not** run `python AutoML\Explainability\run_everything.py` when you are already inside `AutoML\Explainability`; that doubles the path and fails.
