# Welcome! I have added comments for better undersanding and readability.

#Importing Libraries
import re
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load model once
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# AI / ML / Data Science
AI_ML_SKILLS = [
    "python",
    "r",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "nlp",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
    "time series forecasting",
    "recommendation systems",
    "statistics",
    "probability",
]

ML_LIB_SKILLS = [
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "keras",
    "xgboost",
    "lightgbm",
    "matplotlib",
    "seaborn",
]

DATA_ENGINEERING_SKILLS = [
    "sql",
    "nosql",
    "postgresql",
    "mysql",
    "mongodb",
    "cassandra",
    "data warehousing",
    "data pipelines",
    "etl",
    "apache spark",
    "hadoop",
    "kafka",
]

# Web Dev – Frontend
FRONTEND_SKILLS = [
    "html",
    "css",
    "javascript",
    "typescript",
    "react",
    "react.js",
    "next.js",
    "angular",
    "vue",
    "tailwind css",
    "redux",
]

# Web Dev – Backend
BACKEND_SKILLS = [
    "node.js",
    "express",
    "django",
    "flask",
    "fastapi",
    "spring boot",
    "java",
    "c#",
    ".net",
    "golang",
    "php",
    "laravel",
    "rest api",
    "graphql",
]

# DevOps / Cloud / MLOps
DEVOPS_CLOUD_SKILLS = [
    "devops",
    "ci cd",
    "ci/cd",
    "jenkins",
    "github actions",
    "gitlab ci",
    "docker",
    "kubernetes",
    "helm",
    "terraform",
    "ansible",
    "linux",
    "shell scripting",
    "bash",
    "aws",
    "azure",
    "gcp",
    "s3",
    "ec2",
    "lambda",
    "cloudwatch",
    "cloudformation",
    "mlops",
    "model deployment",
    "mlflow",
    "kubeflow",
]

# Cyber Security
CYBERSEC_SKILLS = [
    "cyber security",
    "penetration testing",
    "vulnerability assessment",
    "network security",
    "web application security",
    "owasp",
    "burp suite",
    "nmap",
    "wireshark",
    "metasploit",
    "firewalls",
    "ids ips",
    "security monitoring",
]

# QA / Testing
QA_TESTING_SKILLS = [
    "software testing",
    "manual testing",
    "automation testing",
    "unit testing",
    "integration testing",
    "system testing",
    "regression testing",
    "selenium",
    "pytest",
    "junit",
    "testng",
    "postman",
    "jmeter",
    "api testing",
]

# General Software Engineering / Tools
GENERAL_SWE_SKILLS = [
    "data structures",
    "algorithms",
    "object oriented programming",
    "oop",
    "design patterns",
    "git",
    "github",
    "gitlab",
    "bitbucket",
    "jira",
    "agile",
    "scrum",
    "microservices",
    "distributed systems",
]
skills_list = (AI_ML_SKILLS + ML_LIB_SKILLS + DATA_ENGINEERING_SKILLS +
               FRONTEND_SKILLS + BACKEND_SKILLS + DEVOPS_CLOUD_SKILLS +
               CYBERSEC_SKILLS + QA_TESTING_SKILLS + GENERAL_SWE_SKILLS)

# Role / Domain keywords (optional – can help you classify JD)
ROLE_KEYWORDS = [
    "machine learning engineer",
    "data scientist",
    "data engineer",
    "ai engineer",
    "mlops engineer",
    "software engineer",
    "backend developer",
    "frontend developer",
    "fullstack developer",
    "full stack developer",
    "devops engineer",
    "cloud engineer",
    "cyber security engineer",
    "penetration tester",
    "quality assurance engineer",
    "qa engineer",
    "test engineer",
]

SKILL_SYNONYMS = {
    # AI / ML
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "artificial intelligence": ["artificial intelligence", "ai"],
    "nlp": ["nlp", "natural language processing"],
    "computer vision": ["computer vision", "cv"],
    "time series forecasting":
    ["time series forecasting", "time-series forecasting", "time series"],
    "recommendation systems":
    ["recommendation systems", "recommendation engine", "recommender system"],

    # libs
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "matplotlib": ["matplotlib", "plt"],
    "apache spark": ["apache spark", "spark"],
    "hadoop": ["hadoop"],
    "kafka": ["kafka", "apache kafka"],

    # web frontend
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "react": ["react", "reactjs", "react.js"],
    "next.js": ["next.js", "nextjs"],
    "angular": ["angular", "angularjs", "angular.js"],
    "vue": ["vue", "vue.js", "vuejs"],
    "tailwind css": ["tailwind", "tailwindcss", "tailwind css"],

    # web backend
    "node.js": ["node.js", "nodejs", "node"],
    "rest api": ["rest", "rest api", "restful apis", "restful services"],
    "graphql": ["graphql"],
    ".net": [".net", "dotnet"],
    "c#": ["c#", "c sharp"],

    # devops / cloud / mlops
    "ci cd": ["ci cd", "ci/cd"],
    "jenkins": ["jenkins"],
    "github actions": ["github actions"],
    "gitlab ci": ["gitlab ci", "gitlab-ci"],
    "docker": ["docker", "containers", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud platform"],
    "cloudformation": ["cloudformation", "aws cloudformation"],
    "mlops": ["mlops", "ml ops"],

    # cybersec
    "cyber security": ["cyber security", "cybersecurity"],
    "penetration testing":
    ["penetration testing", "pentesting", "pen testing"],
    "web application security": ["web application security", "web security"],
    "owasp": ["owasp", "owasp top 10"],
    "nmap": ["nmap"],
    "wireshark": ["wireshark"],

    # QA / testing
    "software testing": ["software testing", "testing"],
    "automation testing": ["automation testing", "test automation"],
    "unit testing": ["unit testing", "unit tests"],
    "selenium": ["selenium"],
    "pytest": ["pytest"],
    "junit": ["junit"],
    "postman": ["postman"],
    "api testing": ["api testing", "api automation"],

    # general SWE
    "object oriented programming": ["object oriented programming", "oop"],
    "git": ["git"],
    "github": ["github"],
    "gitlab": ["gitlab"],
    "jira": ["jira"],
    "microservices": ["microservices", "microservice architecture"],
}


#Here text is convereted into lowercase and special characters are removed before TF-IDF.
#re.sub(): Replaces all occurrences of a pattern in a string with a specified replacement string.
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


COMPILED_SKILL_PATTERNS = {}
for skill, variants in SKILL_SYNONYMS.items():
    # Create a single, optimized pattern for all variants of a skill
    # Example: r'\b(scikit-learn|sklearn|scikit learn)\b'
    pattern = r'\b(' + '|'.join(map(re.escape, variants)) + r')\b'
    COMPILED_SKILL_PATTERNS[skill] = re.compile(
        pattern, re.IGNORECASE)  # Added re.IGNORECASE for robustness


def extract_skills(text: str) -> list[str]:
    """Extract canonical skills from text using pre-compiled regex patterns."""
    text_lower = text.lower()
    found = set()

    # Iterate through the pre-compiled patterns
    for skill, pattern in COMPILED_SKILL_PATTERNS.items():
        if pattern.search(text_lower):
            found.add(skill)

    return sorted(found)


def compute_final_score(base_score: float, 
                        matched_skills: list[str], 
                        jd_skills: list[str]) -> tuple[float, float]:
    
    # --- 1. Define Local Core Skills and Weights ---
    # Define a set of high-priority skills and assign a custom weight (e.g., 3x normal)
    HIGH_PRIORITY_SKILLS = {
        "python", "machine learning", "deep learning", "nlp", "statistics", "sql", 
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch"
    }
    HIGH_PRIORITY_WEIGHT = 3.0
    DEFAULT_WEIGHT = 1.0

    # Helper function to get the weight for a single skill
    def get_skill_weight(skill: str) -> float:
        return HIGH_PRIORITY_WEIGHT if skill in HIGH_PRIORITY_SKILLS else DEFAULT_WEIGHT

    
    # 2. Calculate the total required weight (Denominator) 
    total_jd_weight = sum(get_skill_weight(skill) for skill in jd_skills)
    
    # --- 3. Calculate the total matched weight (Numerator) ---
    total_matched_weight = sum(get_skill_weight(skill) for skill in matched_skills)
    
    
    if total_jd_weight == 0:
        weighted_skill_coverage = 0.0
    else:
        # 4. Calculate Weighted Skill Coverage (Capped at 100%)
        weighted_coverage_ratio = min(total_matched_weight / total_jd_weight, 1.0)
        weighted_skill_coverage = weighted_coverage_ratio * 100

    # 5. Calculate Final Score (Using your 60% Base / 40% Weighted Skill split)
    final = 0.6 * base_score + 0.4 * weighted_skill_coverage
    
    # Return the final score and the new weighted coverage score
    return final, weighted_skill_coverage


def build_summary(final_score: float, matched_skills: list[str],
                  missing_skills: list[str]) -> str:
    if final_score >= 75:
        fit_label = "Strong fit"
    elif final_score >= 50:
        fit_label = "Moderate fit"
    else:
        fit_label = "Weak fit"

    matched_text = ", ".join(matched_skills) if matched_skills else "None"
    missing_text = ", ".join(missing_skills) if missing_skills else "None"

    return (f"{fit_label} for this role.\n"
            f"Matched skills: {matched_text}.\n"
            f"Missing skills: {missing_text}.")


def embedding_base_score(resume_text: str, jd_text: str) -> float:
    texts = [resume_text, jd_text]
    embeddings = embed_model.encode(texts,
                                    convert_to_numpy=True,
                                    normalize_embeddings=True)
    cos_sim = cosine_similarity(embeddings[0].reshape(1, -1),
                                embeddings[1].reshape(1, -1))[0][0]
    return cos_sim * 100

def filter_text_by_sections(text: str) -> str:
    """
    Filters the resume text to include only relevant sections for Base Score calculation.
    Excludes personal/interest sections which drag down semantic similarity.
    """
    text_lower = text.lower()
    
    # Define sections we want to INCLUDE (Technical Focus)
    include_keywords = [
        "summary", "profile", "technical skills", "skills", 
        "projects", "experience", "work history", "education"
    ]
    
    # Define sections we want to EXCLUDE (Personal/Non-Technical Focus)
    exclude_keywords = [
        "focus & commitment", "interests", "hobbies", 
        "personal details", "references", "awards"
    ]

    # Find the indices of the excluded sections
    exclusion_indices = {}
    for keyword in exclude_keywords:
        match = re.search(r'\n\s*' + re.escape(keyword) + r'\s*\n', text_lower, re.IGNORECASE)
        if match:
            exclusion_indices[match.start()] = match.end()

    if not exclusion_indices:
        return text  # No exclusions found, return original text

    # Simple implementation: keep text before the first excluded section header
    # If the user always places "Focus & Commitment" last, this is simple and effective.
    first_exclusion_index = min(exclusion_indices.keys())
    
    return text[:first_exclusion_index]
    
def evaluate_resume(resume_text: str, jd_text: str) -> dict:
    
    resume_filtered = filter_text_by_sections(resume_text)
    # 1. Clean
    resume_clean = clean_text(resume_filtered) 
    jd_clean = clean_text(jd_text)

    # 2. TF-IDF similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    tfidf_base = cos_sim * 100

    # 3. Embedding similarity
    emb_base = embedding_base_score(resume_filtered, jd_text)

    # 4. Combined base
    combined_base = 0.3 * tfidf_base + 0.7 * emb_base

    # 5. Skills
    # Note: We now pass the clean text, but the skill patterns are more robust.
    resume_skills = extract_skills(resume_clean)
    jd_skills = extract_skills(jd_clean)
    matched_skills = sorted(list(set(resume_skills) & set(jd_skills)))
    missing_skills = sorted(list(set(jd_skills) - set(resume_skills)))

    # 6. Final score + summary
    final_score, skill_coverage = compute_final_score(combined_base,
                                                      matched_skills,
                                                      jd_skills)
    summary = build_summary(final_score, matched_skills, missing_skills)

    return {
        "tfidf_base": tfidf_base,
        "emb_base": emb_base,
        "base_score": combined_base,
        "skill_coverage": skill_coverage,
        "final_score": final_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "summary": summary,
    }
