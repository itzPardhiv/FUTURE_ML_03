"""Data loader utilities for SmartHire AI.

Provides safe CSV reading, dataset loading helpers and schema validation.

Functions:
 - safe_read_csv(path_or_buffer)
 - load_resumes(root)
 - load_job_descriptions(root)
 - load_monster_jobs(root)
 - load_ranked_results(root)
 - validate_schema(df, required_columns)

This module centralizes file IO and encoding fallbacks so the Streamlit
app and tests can call a single API for robust dataset access.
"""

from typing import Optional, List
import os
import pandas as pd

DEFAULT_ENCODINGS = ["utf-8", "latin1", "iso-8859-1", "cp1252"]


def safe_read_csv(path_or_buffer, encodings: Optional[List[str]] = None, **kwargs) -> Optional[pd.DataFrame]:
	"""Read a CSV file safely trying multiple encodings.

	Args:
		path_or_buffer: file path or file-like object accepted by pandas.
		encodings: optional list of encodings to attempt (defaults provided).
		**kwargs: passed to `pd.read_csv`.

	Returns:
		pd.DataFrame on success, or None on failure.
	"""
	encs = encodings or DEFAULT_ENCODINGS

	for enc in encs:
		try:
			df = pd.read_csv(path_or_buffer, encoding=enc, on_bad_lines='skip', **kwargs)
			return df
		except Exception:
			continue

	return None


def _abs_path(root: str, *parts) -> str:
	return os.path.join(root, *parts)


def load_resumes(root: str) -> Optional[pd.DataFrame]:
	"""Load default resumes dataset from `data/Resume/Resume.csv`.

	Returns None if the file cannot be read.
	"""
	path = _abs_path(root, "data", "Resume", "Resume.csv")
	if not os.path.exists(path):
		return None
	return safe_read_csv(path)


def load_job_descriptions(root: str) -> Optional[pd.DataFrame]:
	"""Load `data/job_descriptions.csv` if present."""
	path = _abs_path(root, "data", "job_descriptions.csv")
	if not os.path.exists(path):
		return None
	return safe_read_csv(path)


def load_monster_jobs(root: str) -> Optional[pd.DataFrame]:
	"""Load `data/monster_com-job_sample.csv` if present."""
	path = _abs_path(root, "data", "monster_com-job_sample.csv")
	if not os.path.exists(path):
		return None
	return safe_read_csv(path)


def load_ranked_results(root: str) -> Optional[pd.DataFrame]:
	"""Load ranking output (outputs/ranked_candidates.csv) if available."""
	path = _abs_path(root, "outputs", "ranked_candidates.csv")
	if not os.path.exists(path):
		return None
	return safe_read_csv(path)


def validate_schema(df: pd.DataFrame, required_columns: List[str]) -> (bool, str):
	"""Validate that DataFrame contains required columns.

	Returns a tuple (is_valid, message).
	"""
	if df is None:
		return False, "DataFrame is None"

	missing = [c for c in required_columns if c not in df.columns]
	if missing:
		return False, f"Missing columns: {missing}"

	if df.empty:
		return False, "DataFrame is empty"

	return True, "OK"
