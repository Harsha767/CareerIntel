"""CareerIntel: explainable, provisional job relevance suggestions.

These are heuristic predictions, not verified ground-truth labels.
Original candidate data and manual reviews are never overwritten.
"""

from ast import literal_eval
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "evaluation"
    / "evaluation_candidates.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "evaluation"
    / "evaluation_candidates_ai_suggested.csv"
)


ALIASES = {
    "py": "python",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "powerbi": "power bi",
    "ms excel": "excel",
    "microsoft excel": "excel",
    "postgres": "postgresql",
    "structured query language": "sql",
    "ml": "machine learning",
    "natural language processing": "nlp",
    "artificial intelligence": "ai",
    "amazon web services": "aws",
    "google cloud platform": "gcp",
    "microsoft azure": "azure",
}

GENERIC_SKILLS = {
    "communication",
    "teamwork",
    "team collaboration",
    "customer service",
    "leadership",
    "time management",
    "problem solving",
    "documentation",
    "organization",
    "attention to detail",
    "project management",
}


# Explicit role-family patterns provide evidence beyond exact skills.
ROLE_FAMILIES = {
    "data_analytics": [
        "data analyst",
        "data scientist",
        "data engineer",
        "business intelligence",
        "business intelligence analyst",
        "analytics engineer",
        "data analytics",
        "machine learning engineer",
        "machine learning scientist",
        "statistical analyst",
        "quantitative analyst",
        "bi analyst",
    ],
    "software": [
        "software engineer",
        "software developer",
        "backend developer",
        "frontend developer",
        "full stack developer",
        "web developer",
        "application developer",
    ],
    "marketing": [
        "marketing analyst",
        "digital marketing",
        "marketing specialist",
        "marketing manager",
    ],
    "finance": [
        "financial analyst",
        "accountant",
        "accounting specialist",
        "finance analyst",
    ],
    "healthcare": [
        "nurse",
        "clinical nurse",
        "medical assistant",
        "healthcare specialist",
    ],
}


def clean_text(value):
    """Safely convert values to text."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text.lower() in {"nan", "none", "null"}:
        return ""

    return text


def normalize_skill(value):
    """Normalize skill spelling and aliases."""
    text = clean_text(value).lower()
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[.,;:]+$", "", text)

    return ALIASES.get(text, text)


def parse_list(value):
    """Parse lists saved as Python representations or plain text."""
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        text = clean_text(value)

        if not text:
            return []

        try:
            parsed = literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                items = list(parsed)
            else:
                items = [text]
        except (ValueError, SyntaxError):
            items = re.split(r"[,;\n|]+", text)

    normalized = []

    for item in items:
        skill = normalize_skill(item)

        if skill and skill not in {"nan", "none", "null"}:
            normalized.append(skill)

    return list(dict.fromkeys(normalized))


def contains_phrase(text, phrase):
    """Match a phrase with word boundaries where possible."""
    text = normalize_skill(text)
    phrase = normalize_skill(phrase)

    if not text or not phrase:
        return False

    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, text) is not None


def get_profile_skills(row):
    """Use structured profile skills and recognized profile-text skills."""
    structured = parse_list(row.get("profile_skills", ""))

    profile_text = normalize_skill(row.get("profile_text", ""))

    # Recognized technical skills; do not treat arbitrary words as skills.
    known_skills = {
        "python",
        "sql",
        "r",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "excel",
        "power bi",
        "tableau",
        "machine learning",
        "deep learning",
        "data analysis",
        "data visualization",
        "statistics",
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "pytorch",
        "nlp",
        "natural language processing",
        "aws",
        "azure",
        "gcp",
        "spark",
        "hadoop",
        "etl",
        "data modeling",
        "data engineering",
        "dashboard development",
        "business intelligence",
        "database management",
        "data cleaning",
        "predictive modeling",
        "statistics",
    }

    detected = [
        skill
        for skill in known_skills
        if contains_phrase(profile_text, skill)
    ]

    return sorted(set(structured + detected))


def get_role_family(text):
    """Return the first recognized role family."""
    normalized = normalize_skill(text)

    for family, phrases in ROLE_FAMILIES.items():
        if any(contains_phrase(normalized, phrase) for phrase in phrases):
            return family

    return None


def determine_suggestion(row):
    """Generate a conservative, explainable relevance suggestion."""
    profile_text = clean_text(row.get("profile_text", ""))
    job_title = clean_text(row.get("job_title", ""))
    job_text = clean_text(row.get("job_text", ""))
    job_skills = parse_list(row.get("job_skills", ""))

    profile_skills = get_profile_skills(row)

    # If the job has no structured skill list, use its description as
    # evidence only for skills already identified in the profile.
    if not job_skills:
        job_skills = [
            skill
            for skill in profile_skills
            if contains_phrase(job_text, skill)
        ]

    profile_set = set(profile_skills)
    job_set = set(job_skills)

    matched = sorted(profile_set & job_set)

    specific_matches = [
        skill for skill in matched
        if skill not in GENERIC_SKILLS
    ]

    specific_requirements = [
        skill for skill in job_skills
        if skill not in GENERIC_SKILLS
    ]

    coverage = (
        len(specific_matches) / len(specific_requirements)
        if specific_requirements
        else 0.0
    )

    profile_family = get_role_family(profile_text)
    job_family = get_role_family(job_title)

    same_family = (
        profile_family is not None
        and profile_family == job_family
    )

    # Technical skill evidence is stronger than generic wording.
    if len(specific_matches) >= 2 and coverage >= 0.15:
        label = 1
        confidence = "high"
        reason = (
            "Multiple specific skills match: "
            + ", ".join(specific_matches)
        )

    elif same_family and len(specific_matches) >= 1:
        label = 1
        confidence = "medium"
        reason = (
            f"Matching role family ({profile_family}) and skill overlap: "
            + ", ".join(specific_matches)
        )

    elif same_family:
        label = 1
        confidence = "medium"
        reason = (
            f"The candidate profile and job title share the "
            f"{profile_family} role family. Verify the actual requirements."
        )

    elif len(specific_matches) >= 1 and coverage >= 0.40:
        label = 1
        confidence = "medium"
        reason = (
            f"Specific skill coverage is {coverage:.0%}; matches: "
            + ", ".join(specific_matches)
        )

    elif not profile_skills:
        label = -1
        confidence = "low"
        reason = (
            "The profile has insufficient structured or recognizable "
            "skill evidence to assess relevance."
        )

    elif not job_skills and not job_family:
        label = -1
        confidence = "low"
        reason = (
            "Job requirements and role family are insufficiently clear. "
            "Manual review is recommended."
        )

    elif matched:
        label = -1
        confidence = "low"
        reason = (
            "Some skills overlap, but evidence is insufficient to make "
            "a reliable relevance decision."
        )

    else:
        label = 0
        confidence = "low"
        reason = (
            "No matching specific skills or recognized role family "
            "was detected. This is a provisional negative, not proof "
            "that the job is irrelevant."
        )

    return {
        "suggested_relevant": label,
        "suggestion_confidence": confidence,
        "matched_skills": ", ".join(matched),
        "skill_coverage": round(coverage, 4),
        "suggestion_reason": reason,
        "label_source": "heuristic_provisional",
    }


def main():
    print("=" * 64)
    print("CAREERINTEL | AUTOMATIC RELEVANCE SUGGESTIONS")
    print("=" * 64)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE, dtype={"job_id": str})

    required = {
        "user_id",
        "profile_text",
        "job_id",
        "job_title",
        "job_text",
        "job_skills",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    suggestions = pd.DataFrame(
        df.apply(determine_suggestion, axis=1).tolist()
    )

    # Keep source columns intact. Store suggestions in separate columns.
    result = df.reset_index(drop=True).copy()

    for column in suggestions.columns:
        # Avoid duplicate suggestion columns if this script is rerun
        # using its own output as input.
        result[column] = suggestions[column].values

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print(f"\nCandidate-job pairs: {len(result)}")

    print("\nSuggestion distribution:")
    print(
        result["suggested_relevant"]
        .map({
            1: "Likely relevant (1)",
            0: "Likely irrelevant (0)",
            -1: "Needs review (-1)",
        })
        .value_counts()
        .to_string()
    )

    print("\nConfidence distribution:")
    print(result["suggestion_confidence"].value_counts().to_string())

    print("\nExample suggestions:")
    columns = [
        "user_id",
        "job_title",
        "suggested_relevant",
        "suggestion_confidence",
        "matched_skills",
        "suggestion_reason",
    ]

    print(result[columns].head(12).to_string(index=False))

    print(f"\nSaved to:\n{OUTPUT_FILE}")

    print(
        "\nReminder: these are heuristic suggestions, not verified "
        "ground-truth labels. -1 means uncertain."
    )


if __name__ == "__main__":
    main()