"""Skill extraction against a fixed vocabulary.

Gives the recruiter something explainable next to the semantic score:
which skills the job asks for that the resume has, and which are missing.
"""

import re

# canonical name -> extra spellings that mean the same thing
SKILLS: dict[str, list[str]] = {
    # programming
    "python": [], "java": [], "javascript": ["js"], "typescript": [], "c++": ["cpp"], "c#": ["c sharp"],
    "sql": [], "r": [], "scala": [], "go": ["golang"], "php": [], "bash": ["shell scripting"],
    # data & ml
    "machine learning": ["ml"], "deep learning": [], "nlp": ["natural language processing"],
    "computer vision": ["opencv"], "pandas": [], "numpy": [], "scikit-learn": ["sklearn", "scikit learn"],
    "tensorflow": [], "pytorch": [], "keras": [], "transformers": ["hugging face", "huggingface"],
    "llm": ["large language models", "generative ai", "genai"], "rag": ["retrieval augmented generation"],
    "statistics": [], "data analysis": ["data analytics"], "data visualization": [],
    "power bi": ["powerbi"], "tableau": [], "excel": [], "hadoop": [], "spark": ["pyspark"],
    "hive": [], "etl": [], "data warehousing": ["data warehouse"], "airflow": [],
    # web & backend
    "html": [], "css": [], "react": ["react.js", "reactjs"], "angular": [], "node.js": ["nodejs"],
    "express": ["express.js"], "next.js": ["nextjs"], "django": [], "flask": [], "fastapi": [],
    "spring": ["spring boot"], "rest api": ["rest apis", "restful"], ".net": ["dotnet", "asp.net"],
    "mysql": [], "postgresql": ["postgres"], "mongodb": [], "oracle": [], "redis": [],
    # cloud & devops
    "aws": ["amazon web services"], "azure": [], "gcp": ["google cloud"], "docker": [], "kubernetes": ["k8s"],
    "jenkins": [], "git": ["github", "gitlab"], "ci/cd": ["cicd"], "linux": [], "terraform": [], "ansible": [],
    # testing & security
    "selenium": [], "manual testing": [], "automation testing": ["test automation"], "jira": [],
    "network security": [], "firewalls": ["firewall"], "penetration testing": [], "blockchain": [],
    "solidity": [], "sap": [], "abap": [],
    # business & other
    "project management": [], "agile": ["scrum"], "business analysis": ["requirement gathering"],
    "stakeholder management": [], "sales": [], "marketing": [], "recruitment": ["recruiting", "talent acquisition"],
    "communication": [], "leadership": [], "autocad": [], "solidworks": [], "matlab": [],
}

_PATTERNS = {
    skill: re.compile(
        r"(?<![a-z0-9+#.])(?:" + "|".join(re.escape(a) for a in [skill, *aliases]) + r")(?![a-z0-9+#])"
    )
    for skill, aliases in SKILLS.items()
}

# "r" and "go" are ordinary words too; only count them when listed next to another language
_LANGUAGE_NEIGHBOURS = r"(?:python|sql|java|scala|c\+\+)"


def extract_skills(clean: str) -> set[str]:
    """Skills mentioned in already-cleaned text (see clean_text)."""
    found = set()
    for skill, pattern in _PATTERNS.items():
        if skill in ("r", "go"):
            near = rf"{_LANGUAGE_NEIGHBOURS}[\s,/]+{skill}(?![a-z0-9])|(?<![a-z0-9]){skill}[\s,/]+{_LANGUAGE_NEIGHBOURS}"
            if re.search(near, clean):
                found.add(skill)
            continue
        if pattern.search(clean):
            found.add(skill)
    return found
