import os
from src.scoring import calculate_ats_score, calculate_skill_match, calculate_role_score


def test_calculate_ats_default_weights():
    sim = 80.0
    skill = 60.0
    role = 50.0
    score = calculate_ats_score(sim, skill, role)
    # Default: 0.5*80 + 0.3*60 + 0.2*50 = 40 + 18 + 10 = 68
    assert abs(score - 68.0) < 1e-6


def test_calculate_ats_custom_weights():
    sim = 100.0
    skill = 0.0
    role = 0.0
    score = calculate_ats_score(sim, skill, role, weights={"similarity": 1.0, "skill": 0.0, "role": 0.0})
    assert abs(score - 100.0) < 1e-6


def test_skill_match():
    candidate = ["python", "pandas", "numpy"]
    required = ["python", "scikit-learn"]

    assert calculate_skill_match(candidate, required) == (1 / 2) * 100


def test_role_score():
    category = "data science"
    jd = "We are hiring a Data Science engineer with machine learning experience."
    rs = calculate_role_score(category, jd)
    assert rs >= 50
