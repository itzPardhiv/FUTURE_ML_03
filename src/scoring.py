from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Default weights reflect business priorities: similarity=50%, skill=30%, role=20%
DEFAULT_WEIGHTS = {
    "similarity": 0.50,
    "skill": 0.30,
    "role": 0.20,
}


def calculate_similarity(resumes, job_description):
    """
    Calculate cosine similarity between resumes and job description using TF-IDF.
    
    Args:
        resumes (list): List of resume texts
        job_description (str): Job description text
        
    Returns:
        array: Similarity scores (0-1) for each resume
    """
    documents = resumes + [job_description]
    
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    vectors = vectorizer.fit_transform(documents)
    
    similarity = cosine_similarity(vectors[:-1], vectors[-1])
    return similarity.flatten()


def calculate_skill_match(candidate_skills, required_skills):
    """
    Calculate percentage of required skills found in candidate.
    
    Args:
        candidate_skills (list): Skills found in candidate resume
        required_skills (list): Skills required for the job
        
    Returns:
        float: Skill match percentage (0-100)
    """
    if not required_skills:
        return 0
    
    matched = set(candidate_skills).intersection(set(required_skills))
    return (len(matched) / len(required_skills)) * 100


def calculate_role_score(category, job_description):
    """
    Calculate role relevance score based on category and job description.
    
    Args:
        category (str): Resume category/job title
        job_description (str): Job description text
        
    Returns:
        float: Role match score (0-100)
    """
    if not category or not job_description:
        return 50
    
    category = str(category).lower().strip()
    job_description = str(job_description).lower()
    
    # Exact match gets full score
    if category in job_description:
        return 100
    
    # Split category into keywords and check
    category_parts = category.split()
    matches = sum(1 for part in category_parts if part in job_description)
    
    if matches > 0:
        return 50 + (matches / len(category_parts)) * 50
    
    return 50


def calculate_ats_score(similarity_score, skill_score, role_score, weights: dict = None):
    """
    Calculate final ATS score using weighted formula.

    The formula is configurable via `weights` to make business rules easy to adjust.

    Args:
        similarity_score (float): Resume-to-JD similarity (0-100)
        skill_score (float): Skill match percentage (0-100)
        role_score (float): Role relevance score (0-100)
        weights (dict): Optional weights with keys 'similarity','skill','role'.

    Returns:
        float: Final ATS score (0-100)
    """
    w = weights or DEFAULT_WEIGHTS

    sim_w = w.get("similarity", DEFAULT_WEIGHTS["similarity"])
    skill_w = w.get("skill", DEFAULT_WEIGHTS["skill"])
    role_w = w.get("role", DEFAULT_WEIGHTS["role"])

    total = sim_w + skill_w + role_w
    if total <= 0:
        # Avoid division by zero; fall back to defaults
        sim_w = DEFAULT_WEIGHTS["similarity"]
        skill_w = DEFAULT_WEIGHTS["skill"]
        role_w = DEFAULT_WEIGHTS["role"]
        total = sim_w + skill_w + role_w

    # Normalize weights to 1.0 in case custom weights were provided
    sim_w /= total
    skill_w /= total
    role_w /= total

    ats_score = (
        similarity_score * sim_w +
        skill_score * skill_w +
        role_score * role_w
    )

    return round(ats_score, 2)


def candidate_recommendation(score):
    """
    Generate hiring recommendation based on ATS score.
    
    Args:
        score (float): ATS score (0-100)
        
    Returns:
        str: Recommendation category
    """
    if score >= 85:
        return "Strong Hire"
    elif score >= 75:
        return "Good Match"
    elif score >= 60:
        return "Moderate Match"
    elif score >= 40:
        return "Weak Match"
    else:
        return "Not Recommended"


def skill_gap_severity(missing_skills):
    """
    Classify the severity of skill gaps.
    
    Args:
        missing_skills (list): Required skills not found in resume
        
    Returns:
        str: Severity classification
    """
    count = len(missing_skills)
    
    if count == 0:
        return "No Gap"
    elif count <= 2:
        return "Minor Gap"
    elif count <= 5:
        return "Moderate Gap"
    else:
        return "Major Gap"