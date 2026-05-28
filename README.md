# SmartHire AI — Community Edition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)](https://streamlit.io/)

SmartHire AI is an open-source Applicant Tracking System (ATS) that helps recruiters and hiring teams automatically screen resumes, extract skills, match candidates to job descriptions, compute ATS scores, and rank candidates — all from a polished Streamlit dashboard.

One-line pitch: Turn resume screening into an efficient, repeatable, and explainable process for hiring teams.

---

## Business Problem

Recruiters spend too much time manually screening resumes. SmartHire AI reduces time-to-hire by automating the first-pass screening with clear explainability and recruiter-friendly insights.

## Project Objective

Provide a lightweight, local-first ATS prototype suitable for interview portfolios and lightweight deployments. The app is intentionally CPU-friendly and uses scikit-learn TF‑IDF + LinearSVC models.

## Key Features

- Resume screening and ranking
- Regex-based skill extraction and skill-gap analysis
- Category classification (LinearSVC) with TF‑IDF features
- Exportable ranked results (CSV)
- Streamlit dashboard with visual analytics (Plotly, WordCloud)
- Safe CSV loading with multi-encoding support

## Architecture & Workflow

1. Data ingestion (CSV resumes & JDs)
2. Text cleaning and preprocessing
3. Skill extraction (regex matching)
4. TF‑IDF vectorization and cosine similarity
5. ATS score calculation and ranking
6. Dashboard visualization and export

## Scoring Formula

The default ATS formula is configurable but uses business-prioritized weights:

ATS Score = 50% * Similarity + 30% * Skill Match + 20% * Role Match

- Similarity: TF‑IDF cosine similarity between resume and job description (0-100)
- Skill Match: % of required skills found in the resume (0-100)
- Role Match: category relevance heuristic (0-100)

Weights are configurable in `src/scoring.py` and normalized automatically.

## Folder Structure

```
FUTURE_ML_03/
├─ app/                # Streamlit app pages and UI
├─ src/                # Core processing & ML helpers
├─ data/               # Datasets (local, not for git history)
├─ outputs/            # Generated exports (ignored by git)
├─ notebooks/          # Optional analysis notebooks
├─ requirements.txt
├─ main.py             # Launcher that uses local venv on Windows
├─ README.md
└─ .gitignore          # Created to avoid committing local artifacts
```

## Disk size & dataset note

- This repository may be large locally because of a committed virtual environment and datasets. Do NOT commit `.venv/` to GitHub. Use the following commands to remove the venv from git tracking and commit the `.gitignore`:

```bash
git rm -r --cached .venv
git add .gitignore
git commit -m "Remove virtual environment and clean repo tracking"
```

- After removing `.venv`, verify repo size locally:

```bash
# Show total size
du -sh .

# Show size excluding the virtual environment (Linux/macOS)
du -sh --exclude=.venv .
```

- Large datasets (CSV) should be stored separately (release assets or cloud) to keep the repository lightweight.

## Installation (recommended)

1. Clone the repo

```bash
git clone <repo-url>
cd FUTURE_ML_03
```

2. Create & activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the app

```bash
python main.py
```

If you see errors about missing datasets, place the sample CSVs under `data/` as described above.

## Development notes & assumptions

- `src/data_loader.py` centralizes safe CSV loading with encoding fallbacks.
- `src/scoring.py` now supports configurable weights (normalized).
- The app is defensive about missing datasets: it will show friendly Streamlit warnings rather than crash.

## Why `.venv` is excluded

- Virtual environments contain platform-specific binary wheels and large package caches that bloat git history. Producing a clean, reproducible environment should be done locally by each contributor.

## Future improvements

- PDF resume parsing
- BERT or transformer-based matching for higher quality matches
- Persistent storage (SQLite or simple REST API)
- Unit tests and CI
- Automated dataset download / release assets

---

## Author

Made by A.J. Pardhiv — contact via GitHub or LinkedIn (placeholders in author section).

