"""Run all Explainability checks. Script lives in AutoML/Explainability/."""
import subprocess
import sys
import os

_script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AutoML", "Explainability")
_script_path = os.path.join(_script_dir, "run_everything.py")
if not os.path.isfile(_script_path):
    print("Not found:", _script_path)
    sys.exit(1)
sys.exit(subprocess.call([sys.executable, _script_path], cwd=_script_dir))
