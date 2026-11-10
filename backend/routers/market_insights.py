import ast
from pathlib import Path
from collections import Counter

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(
    prefix="/api",
    tags=["Market Insights"]
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jobs"
    / "jobs_with_normalized_skills.csv"
)


def clean_value(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def parse_skills(value):
    """Convert stored skill values into a clean list."""

    if isinstance(value, list):
        return [
            str(skill).strip()
            for skill in value
            if str(skill).strip()
        ]

    if not isinstance(value, str) or not value.strip():
        return []

    value = value.strip()

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, (list, tuple, set)):
            return [
                str(skill).strip()
                for skill in parsed
                if str(skill).strip()
            ]
    except (ValueError, SyntaxError):
        pass

    return [
        skill.strip().strip("'\"")
        for skill in value.replace(";", ",").split(",")
        if skill.strip()
    ]


def get_top_values(data, column, limit=10):
    """Return the most common values in a dataset column."""

    if column not in data.columns:
        return []

    values = (
        data[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    values = values[
        (values != "")
        & (values.str.lower() != "nan")
        & (values.str.lower() != "none")
    ]

    counts = values.value_counts().head(limit)

    return [
        {
            "name": str(name),
            "count": int(count)
        }
        for name, count in counts.items()
    ]


@router.get("/market-insights")
def get_market_insights():
    try:
        if not DATA_FILE.exists():
            raise HTTPException(
                status_code=404,
                detail="Processed jobs dataset was not found."
            )

        data = pd.read_csv(DATA_FILE)

        if data.empty:
            raise HTTPException(
                status_code=404,
                detail="The processed jobs dataset is empty."
            )

        # Find available structured skill information.
        skill_column = next(
            (
                column
                for column in [
                    "normalized_skills",
                    "parsed_skills",
                    "skills_required"
                ]
                if column in data.columns
            ),
            None
        )

        skill_counter = Counter()

        if skill_column:
            for value in data[skill_column]:
                skills = {
                    skill.lower()
                    for skill in parse_skills(value)
                    if skill
                }

                # Count each skill at most once per job.
                skill_counter.update(skills)

        top_skills = [
            {
                "name": skill,
                "count": int(count)
            }
            for skill, count in skill_counter.most_common(10)
        ]

        # Build the complete market overview.
        return {
            "status": "success",
            "total_jobs": int(len(data)),

            "top_skills": top_skills,

            "industries": get_top_values(
                data, "industry", 8
            ),

            "job_titles": get_top_values(
                data, "title", 10
            ),

            "experience_levels": get_top_values(
                data, "experience_level", 8
            ),

            "work_models": get_top_values(
                data, "work_model", 8
            ),

            "employment_types": get_top_values(
                data, "employment_type", 8
            )
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Market insights calculation failed: {str(exc)}"
        )