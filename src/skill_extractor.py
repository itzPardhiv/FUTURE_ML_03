import re

# Comprehensive skills database organized by category
SKILLS = [
    # Programming Languages
    "python", "java", "c++", "c#", "javascript", "typescript", "php", "ruby",
    "golang", "rust", "scala", "kotlin", "swift", "objective-c", "perl", "r",
    
    # Web Development
    "html", "css", "react", "angular", "vue.js", "node.js", "express", "fastapi",
    "flask", "django", "next.js", "nuxt.js", "bootstrap", "tailwind", "webpack",
    
    # Data Science & ML
    "machine learning", "deep learning", "nlp", "computer vision", "data analysis",
    "data analytics", "data visualization", "statistics", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch", "keras", "matplotlib", "seaborn",
    "plotly", "jupyter", "power bi", "tableau", "looker",
    
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "cassandra", "elasticsearch",
    "dynamodb", "firestore", "oracle", "sqlserver",
    
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "cloud computing", "docker",
    "kubernetes", "jenkins", "gitlab", "circleci", "terraform", "ansible",
    "cloud infrastructure",
    
    # Tools & Version Control
    "git", "github", "gitlab", "bitbucket", "jira", "confluence", "slack",
    "linux", "unix", "windows", "macos",
    
    # Soft Skills
    "communication", "leadership", "problem solving", "teamwork", "critical thinking",
    "project management", "agile", "scrum", "kanban", "time management",
    "collaboration", "attention to detail", "analytical", "strategic thinking",
    
    # Data Engineering
    "apache spark", "hadoop", "airflow", "etl", "data pipeline", "big data",
    "stream processing", "kafka", "rabbitmq",
    
    # Marketing & Business
    "seo", "sem", "social media", "content marketing", "digital marketing",
    "marketing automation", "analytics", "google analytics", "business analysis",
    
    # HR & Recruitment
    "recruitment", "talent acquisition", "employee relations", "hr management",
    "performance management", "onboarding",
    
    # Finance & Accounting
    "financial analysis", "accounting", "budgeting", "financial modeling",
    "accounting software", "quickbooks", "sap", "erp",
    
    # Design & UX
    "figma", "ui design", "ux design", "graphic design", "adobe xd", "sketch",
    "wireframing", "prototyping", "user research",
    
    # Cybersecurity
    "network security", "penetration testing", "infosec", "information security",
    "security auditing", "firewall", "vpn", "ssl",
    
    # Mobile Development
    "android", "ios", "flutter", "react native", "swift", "kotlin",
    "mobile development", "xamarin",
    
    # QA & Testing
    "qa", "testing", "selenium", "junit", "pytest", "test automation",
    "manual testing", "performance testing",
    
    # Soft Tools
    "excel", "word", "powerpoint", "outlook", "salesforce", "servicenow",
]


def extract_skills(text):
    """
    Extract skills from text using word boundary matching.
    
    Args:
        text (str): Input text to extract skills from
        
    Returns:
        list: Unique list of found skills (sorted)
    """
    if not text:
        return []
    
    text = str(text).lower()
    found_skills = []
    
    for skill in SKILLS:
        # Use word boundaries to avoid false matches (e.g., "c" in "can")
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found_skills.append(skill)
    
    # Remove duplicates and sort for consistency
    unique_skills = sorted(list(set(found_skills)))
    
    return unique_skills