"""Root entry point for cloud deployments (Streamlit Cloud, Hugging Face Spaces).

Runs the real app located at ``app/app.py``.
"""

import runpy
from pathlib import Path

APP = Path(__file__).resolve().parent / "app" / "app.py"

runpy.run_path(str(APP), run_name="__main__")
