"""Run ask_all_questions from project root. Script lives in AutoML/Explainability/."""
import subprocess
import sys
import os

script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AutoML", "Explainability")
script_path = os.path.join(script_dir, "ask_all_questions.py")
sys.exit(subprocess.call([sys.executable, script_path], cwd=script_dir))
