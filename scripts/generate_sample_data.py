"""Generate small sample datasets for onboarding and tests.

Creates `data/sample_resumes.csv` and `data/sample_job_descriptions.csv` with a few rows.
"""
import os
import csv

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
SAMPLE_DIR = DATA_DIR

os.makedirs(SAMPLE_DIR, exist_ok=True)

resumes = [
    {"Resume_str": "Experienced Python developer with pandas and scikit-learn. Worked on data analysis and machine learning." , "Category": "INFORMATION-TECHNOLOGY"},
    {"Resume_str": "Digital marketer skilled in SEO, social media and Google Analytics.", "Category": "DIGITAL-MEDIA"},
    {"Resume_str": "Registered nurse with experience in patient care and healthcare administration.", "Category": "HEALTHCARE"},
    {"Resume_str": "Finance professional experienced in financial analysis, budgeting, and QuickBooks.", "Category": "FINANCE"},
    {"Resume_str": "Project manager with agile, scrum and team leadership experience.", "Category": "BUSINESS-DEVELOPMENT"}
]

jobs = [
    {"JobID": 1, "JobDescription": "Looking for a Python data scientist with experience in pandas, numpy, scikit-learn and machine learning."},
    {"JobID": 2, "JobDescription": "Marketing specialist skilled in SEO, content marketing and Google Analytics."},
    {"JobID": 3, "JobDescription": "Registered Nurse needed for patient care and healthcare coordination."}
]

with open(os.path.join(SAMPLE_DIR, "sample_resumes.csv"), "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["Resume_str", "Category"])
    writer.writeheader()
    for r in resumes:
        writer.writerow(r)

with open(os.path.join(SAMPLE_DIR, "sample_job_descriptions.csv"), "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["JobID", "JobDescription"])
    writer.writeheader()
    for j in jobs:
        writer.writerow(j)

print("Sample data created:")
print(" - data/sample_resumes.csv")
print(" - data/sample_job_descriptions.csv")
