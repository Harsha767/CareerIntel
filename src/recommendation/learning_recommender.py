"""
CAREERINTEL — Career-Focused Learning Recommender

Identifies skill-development priorities for a target career
using the processed job dataset.

Priority is based on the frequency of each skill among
matching job records. This is descriptive analysis, not a
claim that learning a skill guarantees employment.
"""

import ast
import re
from pathlib import Path

import pandas as pd

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
# 1. SKILL NORMALIZATION
# =========================================================

def normalize_skill(value):
    """Normalize skill names for consistent comparison."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    skill = re.sub(
        r"\s+",
        " ",
        str(value).strip().casefold(),
    )

    aliases = {
        "py": "python",
        "ml": "machine learning",
        "structured query language": "sql",
        "data visualisation": "data visualization",
        "scikit learn": "scikit-learn",
    }

    return aliases.get(skill, skill)


# =========================================================
# 2. PARSE SKILLS
# =========================================================

def parse_skills(value):
    """Parse list-like or delimiter-separated skill values."""

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

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        items = parsed
    elif isinstance(parsed, str):
        items = [parsed]
    else:
        items = re.split(r"[,;|]", text)

    return {
        skill
        for item in items
        if (skill := normalize_skill(item))
    }


# =========================================================
# 3. LOAD PROCESSED JOB DATA
# =========================================================

def load_jobs():
    """Load and validate the processed job dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n{DATA_FILE}"
        )

    jobs = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )

    required_columns = {
        "title",
        "normalized_skills",
    }

    missing_columns = required_columns - set(jobs.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Ensure optional skill columns exist.
    for column in [
        "skills_required",
        "parsed_skills",
    ]:
        if column not in jobs.columns:
            jobs[column] = ""

    jobs["title"] = (
        jobs["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return jobs


# =========================================================
# 4. BUILD SKILL SETS
# =========================================================

def build_job_skills(jobs):
    """
    Combine available skill fields without discarding
    populated information from other columns.
    """

    job_skill_sets = []

    for _, row in jobs.iterrows():

        skills = set()

        for column in [
            "skills_required",
            "parsed_skills",
            "normalized_skills",
        ]:
            skills.update(
                parse_skills(row.get(column, ""))
            )

        job_skill_sets.append(skills)

    return job_skill_sets


# =========================================================
# 5. FILTER JOBS BY TARGET CAREER
# =========================================================

CAREER_KEYWORDS = {
    "data science": [
        "data scientist",
        "data science",
        "data science and business intelligence",
    ],
    "data analyst": [
        "data analyst",
        "data analytics",
        "business data analyst",
        "business intelligence analyst",
    ],
    "data engineering": [
        "data engineer",
        "data engineering",
    ],
    "machine learning": [
        "machine learning",
        "ml engineer",
        "mlops",
        "ai scientist",
    ],
    "business intelligence": [
        "business intelligence",
        "bi analyst",
        "bi developer",
    ],
}


def filter_jobs_by_career(jobs, career):
    """Select job records relevant to the chosen career."""

    career_key = career.casefold().strip()

    if career_key not in CAREER_KEYWORDS:
        supported = ", ".join(CAREER_KEYWORDS)

        raise ValueError(
            f"Unsupported career: {career!r}\n"
            f"Choose one of: {supported}"
        )

    keywords = CAREER_KEYWORDS[career_key]

    pattern = "|".join(
        re.escape(keyword)
        for keyword in keywords
    )

    matching_mask = jobs["title"].str.contains(
        pattern,
        case=False,
        regex=True,
        na=False,
    )

    career_jobs = jobs.loc[matching_mask].copy()

    if career_jobs.empty:
        raise ValueError(
            f"No jobs found for career {career!r}. "
            "Inspect the available job titles first."
        )

    return career_jobs


# =========================================================
# 6. CALCULATE LEARNING PRIORITIES
# =========================================================

def recommend_learning(
    candidate_skills,
    career="data science",
    top_n=15,
):
    """
    Recommend missing skills based on their prevalence
    in job records matching the target career.
    """

    jobs = load_jobs()

    candidate_skills = {
        skill
        for item in candidate_skills
        if (skill := normalize_skill(item))
    }

    career_jobs = filter_jobs_by_career(
        jobs,
        career,
    )

    career_jobs["skill_set"] = build_job_skills(
        career_jobs
    )

    # Remove records without usable skill information.
    career_jobs = career_jobs[
        career_jobs["skill_set"].map(bool)
    ].copy()

    if career_jobs.empty:
        raise ValueError(
            f"Matching jobs for {career!r} contain "
            "no usable skill information."
        )

    # Count jobs containing each skill.
    skill_frequency = {}

    for skill_set in career_jobs["skill_set"]:

        # Count at most once per job.
        for skill in skill_set:
            skill_frequency[skill] = (
                skill_frequency.get(skill, 0) + 1
            )

    total_jobs = len(career_jobs)

    recommendations = []

    for skill, frequency in skill_frequency.items():

        # Exclude skills already held by the candidate.
        if skill in candidate_skills:
            continue

        recommendations.append({
            "skill": skill,
            "jobs_requiring_skill": frequency,
            "job_frequency_percentage": round(
                frequency / total_jobs * 100,
                2,
            ),
        })

    result = pd.DataFrame(recommendations)

    if result.empty:
        return result

    result = result.sort_values(
        [
            "jobs_requiring_skill",
            "skill",
        ],
        ascending=[False, True],
    ).head(top_n).reset_index(drop=True)

    result.insert(
        0,
        "priority",
        range(1, len(result) + 1),
    )

    return result


# =========================================================
# 7. SAVE RESULTS
# =========================================================

def save_recommendations(recommendations):
    """Save learning priorities to a CSV report."""

    output_file = (
        OUTPUT_DIR
        / "learning_recommendations.csv"
    )

    recommendations.to_csv(
        output_file,
        index=False,
    )

    return output_file


# =========================================================
# 8. MAIN EXECUTION
# =========================================================

if __name__ == "__main__":

    print("=" * 65)
    print("CAREERINTEL | CAREER-FOCUSED LEARNING RECOMMENDER")
    print("=" * 65)

    candidate_skills = [
        "Python",
        "SQL",
        "Machine Learning",
        "Statistics",
        "Data Analysis",
        "Pandas",
        "Data Visualization",
    ]

    target_career = "data science"

    print(f"\nTarget career: {target_career.title()}")
    print("Candidate skills:")
    print(", ".join(sorted(candidate_skills)))

    recommendations = recommend_learning(
        candidate_skills=candidate_skills,
        career=target_career,
        top_n=15,
    )

    if recommendations.empty:

        print(
            "\nNo additional missing skills were identified "
            "for the selected career."
        )

    else:

        print("\nTOP SKILLS TO CONSIDER LEARNING\n")

        print(
            recommendations.to_string(index=False)
        )

        output_file = save_recommendations(
            recommendations
        )

        print("\nReport saved to:")
        print(output_file.relative_to(ROOT))

    print("\nLearning analysis completed successfully.")