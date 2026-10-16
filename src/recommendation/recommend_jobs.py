"""CareerIntel: Hybrid Job Recommendation Engine.

Baseline:
- TF-IDF cosine similarity for textual relevance.
- Candidate skill coverage.
- Job requirement coverage.
- Hybrid recommendation ranking.

Note:
The scoring weights are initial baseline choices.
They must be evaluated experimentally before claiming improvement.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "jobs"
    / "jobs_with_normalized_skills.csv"
)

OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. TEXT CLEANING
# =========================================================

def clean(value: Any) -> str:
    """Convert a value into clean text without raising on missing values."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    return re.sub(r"\s+", " ", text).strip()


# =========================================================
# 2. SKILL NORMALIZATION
# =========================================================

def normalize_skill(skill: Any) -> str:
    """Normalize skill names for consistent comparison."""

    value = clean(skill).casefold()

    aliases = {
        "scikit learn": "scikit-learn",
        "scikit-learn": "scikit-learn",
        "py": "python",
        "structured query language": "sql",
        "ml": "machine learning",
        "data visualisation": "data visualization",
    }

    return aliases.get(value, value)


# =========================================================
# 3. SKILL PARSING
# =========================================================

def parse_skills(value: Any) -> set[str]:
    """
    Parse Python-style lists or explicitly delimited skill text.

    Multiword skills such as 'machine learning' remain intact.
    """

    if value is None:
        return set()

    try:
        if pd.isna(value):
            return set()
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text:
        return set()

    # Try parsing Python-style lists first.
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        items = parsed

    elif isinstance(parsed, str):
        items = [parsed]

    else:
        # Split only on explicit delimiters.
        # Never split on spaces because skills can be multiword.
        items = re.split(r"[,;|]", text)

    skills = set()

    for item in items:
        skill = normalize_skill(item)

        if skill:
            skills.add(skill)

    return skills


def format_skills(value: Any) -> str:
    """Convert a skill field into readable text."""

    return " ".join(sorted(parse_skills(value)))


# =========================================================
# 4. LOAD AND PREPARE JOB DATA
# =========================================================

def load_jobs() -> pd.DataFrame:
    """Load processed jobs and prepare searchable text and skill sets."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Job dataset not found:\n{DATA_FILE}\n\n"
            "Check that the processed job dataset exists."
        )

    jobs = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    ).copy()

    # -----------------------------------------------------
    # A. SUPPORT AVAILABLE TITLE COLUMNS
    # -----------------------------------------------------

    if "title" not in jobs.columns:

        if "normalized_title" in jobs.columns:
            jobs["title"] = jobs["normalized_title"]

        else:
            raise ValueError(
                "Dataset must contain 'title' or 'normalized_title'."
            )

    # -----------------------------------------------------
    # B. SUPPORT AVAILABLE DESCRIPTION COLUMNS
    # -----------------------------------------------------

    if "job_text" not in jobs.columns:

        if "job_description" in jobs.columns:
            jobs["job_text"] = jobs["job_description"]

        else:
            jobs["job_text"] = ""

    # -----------------------------------------------------
    # C. CREATE OPTIONAL COLUMNS WHEN MISSING
    # -----------------------------------------------------

    optional_columns = [
        "company_name",
        "industry",
        "function",
        "experience_level",
        "employment_type",
        "work_model",
        "education_level",
    ]

    for column in optional_columns:

        if column not in jobs.columns:
            jobs[column] = ""

    # -----------------------------------------------------
    # D. PRESERVE ALL AVAILABLE SKILL SOURCES
    # -----------------------------------------------------

    skill_columns = [
        "normalized_skills",
        "parsed_skills",
        "skills_required",
    ]

    for column in skill_columns:

        if column not in jobs.columns:
            jobs[column] = ""

    # Clean the main text fields.
    jobs["title"] = jobs["title"].map(clean)
    jobs["job_text"] = jobs["job_text"].map(clean)

    # Combine skills from every available source.
    combined_skill_sets = []

    for _, row in jobs[skill_columns].iterrows():

        job_skills = set()

        for column in skill_columns:
            job_skills.update(
                parse_skills(row[column])
            )

        combined_skill_sets.append(job_skills)

    jobs["skill_set"] = combined_skill_sets

    # Store a consistent, readable skill representation.
    jobs["normalized_skills"] = jobs["skill_set"].map(
        lambda values: ", ".join(sorted(values))
    )

    # -----------------------------------------------------
    # E. PREPARE SEARCHABLE TEXT
    # -----------------------------------------------------

    text_columns = [
        "title",
        "job_text",
        "normalized_skills",
        "industry",
        "function",
        "experience_level",
        "employment_type",
        "work_model",
        "education_level",
    ]

    for column in text_columns:
        jobs[column] = jobs[column].map(clean)

    jobs["search_text"] = (
        jobs["title"] + " "
        + jobs["job_text"] + " "
        + jobs["normalized_skills"] + " "
        + jobs["industry"] + " "
        + jobs["function"] + " "
        + jobs["experience_level"] + " "
        + jobs["employment_type"] + " "
        + jobs["work_model"] + " "
        + jobs["education_level"]
    ).str.strip()

    # Remove records without searchable information.
    jobs = jobs[
        jobs["search_text"].ne("")
    ].reset_index(drop=True)

    if jobs.empty:
        raise ValueError(
            "No searchable job records remain after preprocessing."
        )

    return jobs


# =========================================================
# 5. JOB RECOMMENDATION ENGINE
# =========================================================

def recommend_jobs(
    profile: str,
    preferred_skills: list[str] | None = None,
    top_k: int = 10,
    min_similarity: float = 0.0,
) -> pd.DataFrame:
    """
    Recommend jobs using TF-IDF and skill matching.

    Candidate skill coverage:
        Matching candidate skills / all candidate skills

    Requirement coverage:
        Matching candidate skills / identified job requirements

    Hybrid score when requirements are available:
        65% content similarity
        15% candidate skill coverage
        20% requirement coverage

    When requirement information is unavailable:
        81.25% content similarity
        18.75% candidate skill coverage

    These are baseline weights, not experimentally validated weights.
    """

    # -----------------------------------------------------
    # A. VALIDATE INPUTS
    # -----------------------------------------------------

    if not isinstance(profile, str) or not profile.strip():
        raise ValueError(
            "Provide a non-empty career profile."
        )

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
    ):
        raise ValueError(
            "top_k must be a positive integer."
        )

    if (
        not isinstance(min_similarity, (int, float))
        or not 0 <= min_similarity <= 1
    ):
        raise ValueError(
            "min_similarity must be a number between 0 and 1."
        )

    # -----------------------------------------------------
    # B. LOAD JOB DATA
    # -----------------------------------------------------

    jobs = load_jobs()

    # -----------------------------------------------------
    # C. CALCULATE TF-IDF CONTENT SIMILARITY
    # -----------------------------------------------------

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100_000,
        sublinear_tf=True,
        min_df=1,
    )

    try:

        job_matrix = vectorizer.fit_transform(
            jobs["search_text"]
        )

        profile_vector = vectorizer.transform(
            [profile.strip()]
        )

    except ValueError as exc:

        raise ValueError(
            "Unable to build the TF-IDF vocabulary. "
            "Check that job descriptions contain searchable text."
        ) from exc

    content_scores = cosine_similarity(
        profile_vector,
        job_matrix,
    ).ravel()

    # -----------------------------------------------------
    # D. NORMALIZE CANDIDATE SKILLS
    # -----------------------------------------------------

    user_skills = set()

    for skill in preferred_skills or []:
        user_skills.update(
            parse_skills(skill)
        )

    # -----------------------------------------------------
    # E. CALCULATE SKILL COVERAGE
    # -----------------------------------------------------

    skill_scores = []
    requirement_scores = []
    matched_skill_details = []

    for _, row in jobs.iterrows():

        job_skills = row["skill_set"]

        matching_skills = user_skills.intersection(
            job_skills
        )

        matched_skill_details.append(
            ", ".join(sorted(matching_skills))
        )

        # Proportion of the candidate's skills present in this job.
        if user_skills:

            candidate_coverage = (
                len(matching_skills) / len(user_skills)
            )

        else:
            candidate_coverage = 0.0

        skill_scores.append(candidate_coverage)

        # -------------------------------------------------
        # F. CALCULATE JOB REQUIREMENT COVERAGE
        # -------------------------------------------------

        required_skills = parse_skills(
            row["skills_required"]
        )

        # Use parsed_skills if explicit requirements are absent.
        if not required_skills:

            required_skills = parse_skills(
                row["parsed_skills"]
            )

        if required_skills:

            if user_skills:

                requirement_coverage = (
                    len(
                        user_skills.intersection(
                            required_skills
                        )
                    )
                    / len(required_skills)
                )

            else:
                requirement_coverage = 0.0

        else:
            # Unknown requirements should not be treated as zero.
            requirement_coverage = float("nan")

        requirement_scores.append(
            requirement_coverage
        )

    skill_scores = pd.Series(
        skill_scores,
        dtype=float,
    ).to_numpy()

    requirement_scores = pd.Series(
        requirement_scores,
        dtype=float,
    ).to_numpy()

    # -----------------------------------------------------
    # G. CALCULATE HYBRID RECOMMENDATION SCORE
    # -----------------------------------------------------

    has_requirements = ~pd.isna(
        requirement_scores
    )

    # Base formula for jobs with known requirement information.
    final_scores = (
        0.65 * content_scores
        + 0.15 * skill_scores
        + 0.20 * pd.Series(
            requirement_scores
        ).fillna(0).to_numpy()
    )

    # Redistribute the missing requirement weight when requirement
    # information is unavailable. This avoids penalizing missing data.
    final_scores[~has_requirements] = (
        0.8125 * content_scores[~has_requirements]
        + 0.1875 * skill_scores[~has_requirements]
    )

    # -----------------------------------------------------
    # H. BUILD RESULTS
    # -----------------------------------------------------

    results = jobs.copy()

    results["content_score"] = content_scores

    # FIX: Explicitly create skill_coverage.
    # This prevents KeyError during output and validation.
    results["skill_coverage"] = skill_scores

    results["requirement_coverage"] = requirement_scores

    results["recommendation_score"] = final_scores

    results["matching_skills"] = matched_skill_details

    # Filter using the content-similarity threshold.
    results = results[
        results["content_score"] >= min_similarity
    ].copy()

    # Sort by the hybrid score, then textual relevance.
    results = results.sort_values(
        [
            "recommendation_score",
            "content_score",
        ],
        ascending=False,
    )

    # -----------------------------------------------------
    # I. REMOVE DUPLICATE JOBS
    # -----------------------------------------------------

    results["_title_key"] = (
        results["title"]
        .str.casefold()
        .str.strip()
    )

    results["_company_key"] = (
        results["company_name"]
        .map(clean)
        .str.casefold()
    )

    results = results.drop_duplicates(
        subset=[
            "_title_key",
            "_company_key",
        ],
        keep="first",
    )

    results = results.head(top_k).copy()

    # -----------------------------------------------------
    # J. SELECT OUTPUT COLUMNS
    # -----------------------------------------------------

    output_columns = [
        "title",
        "company_name",
        "industry",
        "function",
        "experience_level",
        "work_model",
        "employment_type",
        "normalized_skills",
        "matching_skills",
        "content_score",
        "skill_coverage",
        "requirement_coverage",
        "recommendation_score",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in results.columns
    ]

    return results[
        output_columns
    ].reset_index(drop=True)


# =========================================================
# 6. RUN THE RECOMMENDATION ENGINE
# =========================================================

if __name__ == "__main__":

    print("=" * 65)
    print("CAREERINTEL | JOB RECOMMENDATION ENGINE")
    print("=" * 65)

    # Example candidate profile.
    profile = (
        "Aspiring Data Scientist with Python, SQL, "
        "machine learning, statistics, data analysis, "
        "pandas, and data visualization skills."
    )

    skills = [
        "Python",
        "SQL",
        "Machine Learning",
        "Statistics",
        "Data Analysis",
        "Pandas",
        "Data Visualization",
    ]

    # Normalize profile skills.
    normalized_profile_skills = set()

    for skill in skills:

        normalized_profile_skills.update(
            parse_skills(skill)
        )

    print("\nProfile skills:")
    print(sorted(normalized_profile_skills))

    # -----------------------------------------------------
    # DATASET VALIDATION
    # -----------------------------------------------------

    jobs = load_jobs()

    print("\n=== DATASET VALIDATION ===")

    print(
        "Total searchable jobs:",
        len(jobs),
    )

    print(
        "Jobs containing at least one parsed skill:",
        int(jobs["skill_set"].map(bool).sum()),
    )

    matching_job_count = sum(
        bool(
            normalized_profile_skills.intersection(
                job_skills
            )
        )
        for job_skills in jobs["skill_set"]
    )

    print(
        "Jobs matching at least one profile skill:",
        matching_job_count,
    )

    # -----------------------------------------------------
    # GENERATE RECOMMENDATIONS
    # -----------------------------------------------------

    recommendations = recommend_jobs(
        profile=profile,
        preferred_skills=skills,
        top_k=10,
    )

    # Save results.
    output_file = (
        OUTPUT_DIR / "sample_recommendations.csv"
    )

    recommendations.to_csv(
        output_file,
        index=False,
    )

    print("\n=== RECOMMENDATION RESULTS ===")

    print(
        "Recommendations generated:",
        len(recommendations),
    )

    print(
        "Saved to:",
        output_file.relative_to(ROOT),
    )

    # -----------------------------------------------------
    # DISPLAY TOP RECOMMENDATIONS
    # -----------------------------------------------------

    if not recommendations.empty:

        display_columns = [
            "title",
            "company_name",
            "matching_skills",
            "content_score",
            "skill_coverage",
            "requirement_coverage",
            "recommendation_score",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in recommendations.columns
        ]

        print("\n=== TOP 10 RECOMMENDATIONS ===")

        print(
            recommendations[
                display_columns
            ].to_string(index=False)
        )

        # -------------------------------------------------
        # RECOMMENDATION VALIDATION
        # -------------------------------------------------

        print("\n=== RECOMMENDATION VALIDATION ===")

        print(
            "Recommendations with positive skill coverage:",
            int(
                (
                    recommendations["skill_coverage"] > 0
                ).sum()
            ),
        )

        print(
            "Mean skill coverage:",
            round(
                recommendations["skill_coverage"].mean(),
                4,
            ),
        )

        print(
            "Recommendations with known requirements:",
            int(
                recommendations[
                    "requirement_coverage"
                ].notna().sum()
            ),
        )

        known_requirement_scores = (
            recommendations[
                "requirement_coverage"
            ].dropna()
        )

        if not known_requirement_scores.empty:

            print(
                "Mean requirement coverage:",
                round(
                    known_requirement_scores.mean(),
                    4,
                ),
            )

        print(
            "Mean content similarity:",
            round(
                recommendations["content_score"].mean(),
                4,
            ),
        )

        print(
            "Mean recommendation score:",
            round(
                recommendations[
                    "recommendation_score"
                ].mean(),
                4,
            ),
        )

    print("\nExecution completed.")