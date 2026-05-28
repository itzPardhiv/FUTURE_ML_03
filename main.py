"""Launch SmartHire AI from the project virtual environment.

This entrypoint avoids accidentally using a global Streamlit install on
Windows by re-running the app with the workspace's local .venv interpreter
when it is available.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
	return Path(__file__).resolve().parent


def _venv_python() -> Path:
	if os.name == "nt":
		return _project_root() / ".venv" / "Scripts" / "python.exe"
	return _project_root() / ".venv" / "bin" / "python"


def main() -> int:
	project_root = _project_root()
	venv_python = _venv_python()
	app_path = project_root / "app" / "app.py"

	if not app_path.exists():
		print(f"Could not find app entrypoint: {app_path}")
		return 1

	if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
		command = [str(venv_python), "-m", "streamlit", "run", str(app_path)]
	else:
		command = [sys.executable, "-m", "streamlit", "run", str(app_path)]

	return subprocess.call(command, cwd=str(project_root))


if __name__ == "__main__":
	raise SystemExit(main())
