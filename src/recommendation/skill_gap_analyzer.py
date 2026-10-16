"""CareerIntel: Skill Gap Analysis Module.

Compares candidate skills against the skills required for a target job.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

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

def normalize_skill(value: Any) -> str:
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
        "scikit learn": "scikit-learn",
        "data visualisation": "data visualization",
    }

    return aliases.get(skill, skill)


# =========================================================
# 2. PARSE SKILLS
# =========================================================

def parse_skills(value: Any) -> set[str]:
    """Parse lists or delimited skill strings."""

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
# 3. LOAD JOB DATA
# =========================================================

def load_jobs() -> pd.DataFrame:
    """Load the processed job dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Job dataset not found: {DATA_FILE}"
        )

    jobs = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )

    if "title" not in jobs.columns:

        if "normalized_title" in jobs.columns:
            jobs["title"] = jobs["normalized_title"]

        else:
            raise ValueError(
                "The dataset needs a title or normalized_title column."
            )

    # Preserve compatibility with the current dataset.
    for column in [
        "company_name",
        "normalized_skills",
        "parsed_skills",
        "skills_required",
    ]:

        if column not in jobs.columns:
            jobs[column] = ""

    return jobs


# =========================================================
# 4. EXTRACT JOB REQUIREMENTS
# =========================================================

def get_required_skills(job: pd.Series) -> set[str]:
    """Collect skills from all available job skill columns."""

    required = set()

    for column in [
        "skills_required",
        "parsed_skills",
        "normalized_skills",
    ]:

        required.update(
            parse_skills(job.get(column, ""))
        )

    return required


# =========================================================
# 5. ANALYZE A CANDIDATE'S SKILL GAPS
# =========================================================

def analyze_skill_gap(
    candidate_skills: list[str],
    job_title: str,
    company_name: str | None = None,
) -> dict[str, Any]:
    """
    Compare candidate skills against a target job.

    If company_name is provided, both job title and company
    must match the selected job.
    """

    jobs = load_jobs()

    # Normalize candidate skills.
    candidate = {
        skill
        for item in candidate_skills
        if (skill := normalize_skill(item))
    }

    # Match the target job title.
    title_mask = (
        jobs["title"]
        .astype(str)
        .str.strip()
        .str.casefold()
        == job_title.strip().casefold()
    )

    matches = jobs[title_mask].copy()

    # Optionally identify a specific company.
    if company_name:

        if "company_name" in matches.columns:

            company_mask = (
                matches["company_name"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.casefold()
                == company_name.strip().casefold()
            )

            matches = matches[company_mask]

    if matches.empty:
        raise ValueError(
            f"No matching job found for title={job_title!r}, "
            f"company={company_name!r}."
        )

    # Use the first matching job record.
    job = matches.iloc[0]

    required = get_required_skills(job)

    if not required:
        raise ValueError(
            "No structured skills were found for this job. "
            "We should not calculate a skill-gap percentage "
            "without identified requirements."
        )

    possessed = candidate.intersection(required)
    missing = required.difference(candidate)

    coverage = len(possessed) / len(required)

    return {
        "job_title": str(job["title"]),
        "company_name": str(job.get("company_name", "")),
        "candidate_skills": sorted(candidate),
        "required_skills": sorted(required),
        "possessed_skills": sorted(possessed),
        "missing_skills": sorted(missing),
        "matched_skill_count": len(possessed),
        "required_skill_count": len(required),
        "missing_skill_count": len(missing),
        "skill_coverage": round(coverage, 4),
        "skill_gap_percentage": round(
            (1 - coverage) * 100,
            2,
        ),
    }


# =========================================================
# 6. SAVE SKILL-GAP RESULTS
# =========================================================

def save_skill_gap_report(
    report: dict[str, Any],
) -> Path:
    """Save a skill-gap report as JSON."""

    import json

    output_file = OUTPUT_DIR / "skill_gap_report.json"

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_file


# =========================================================
# 7. RUN THE MODULE
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CAREERINTEL | SKILL GAP ANALYSIS")
    print("=" * 60)

    candidate_skills = [
        "Python",
        "SQL",
        "Machine Learning",
        "Statistics",
        "Data Analysis",
        "Pandas",
        "Data Visualization",
    ]

    jobs = load_jobs()

    # Select a job with structured skill information.
    eligible_jobs = jobs[
        jobs["normalized_skills"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    if eligible_jobs.empty:
        raise ValueError(
            "No jobs with normalized skills were found."
        )

    target_job = eligible_jobs.iloc[0]

    report = analyze_skill_gap(
        candidate_skills=candidate_skills,
        job_title=str(target_job["title"]),
        company_name=(
            str(target_job["company_name"])
            if pd.notna(target_job["company_name"])
            else None
        ),
    )

    print("\nTarget job:")
    print(report["job_title"])

    print("\nCompany:")
    print(report["company_name"])

    print("\nRequired skills:")
    print(report["required_skills"])

    print("\nSkills the candidate already possesses:")
    print(report["possessed_skills"])

    print("\nMissing skills:")
    print(report["missing_skills"])

    print("\nSkill coverage:")
    print(f'{report["skill_coverage"] * 100:.2f}%')

    print("\nSkill gap:")
    print(f'{report["skill_gap_percentage"]:.2f}%')

    output_file = save_skill_gap_report(report)

    print("\nReport saved to:")
    print(output_file.relative_to(ROOT))

    print("\nSkill-gap analysis completed.")