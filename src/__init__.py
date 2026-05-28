"""SmartHire AI source package."""

from .category_model import predict_resume_category, train_category_classifier
from .preprocessing import clean_text
from .scoring import (
    calculate_ats_score,
    calculate_role_score,
    calculate_similarity,
    calculate_skill_match,
    candidate_recommendation,
    skill_gap_severity,
)
from .skill_extractor import extract_skills
