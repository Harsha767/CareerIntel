"""Build reproducible candidate pools for CareerIntel evaluation.

Candidate selection uses profile-related titles and skill overlap.
It does NOT generate relevance labels or claim ground truth.
Existing evaluation files are never overwritten.
"""

from pathlib import Path
import ast
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

JOB_FILE = (
    ROOT / "data" / "processed" / "jobs" / "jobs_with_job_ids.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "evaluation_candidates_balanced.csv"
)

RANDOM_SEED = 42
TARGET_COUNT = 30
DISTRACTOR_COUNT = 30

PROFILES = {
    "DS001": {
        "profile": (
            "Aspiring Data Scientist with Python, SQL, machine learning, "
            "statistics, data analysis, pandas, and data visualization."
        ),
        "skills": {
            "python", "sql", "machine learning", "statistics",
            "data analysis", "pandas", "data visualization",
        },
        "title_terms": [
            "data scientist", "data science", "statistician",
            "research scientist", "data analyst", "analytics",
        ],
    },
    "DA001": {
        "profile": (
            "Data Analyst skilled in SQL, Python, data analysis, "
            "reporting, dashboards, Excel, and data visualization."
        ),
        "skills": {
            "sql", "python", "data analysis", "reporting",
            "dashboards", "excel", "data visualization",
        },
        "title_terms": [
            "data analyst", "business intelligence", "business analyst",
            "reporting analyst", "analytics", "bi analyst",
        ],
    },
    "ML001": {
        "profile": (
            "Machine Learning Engineer interested in Python, machine "
            "learning, deep learning, model evaluation, and deployment."
        ),
        "skills": {
            "python", "machine learning", "deep learning",
            "model evaluation", "model deployment",
        },
        "title_terms": [
            "machine learning", "ml engineer", "ai engineer",
            "artificial intelligence", "deep learning",
            "computer vision", "nlp engineer",
        ],
    },
}


def clean(value):
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_skills(value):
    """Parse list-like or comma/semicolon/pipe-separated skill values."""
    text = clean(value)
    if not text:
        return set()

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            items = parsed
        elif isinstance(parsed, str):
            items = [parsed]
        else:
            items = re.split(r"[,;|]", text)
    except (ValueError, SyntaxError):
        items = re.split(r"[,;|]", text)

    aliases = {
        "py": "python",
        "ml": "machine learning",
        "data visualisation": "data visualization",
        "scikit learn": "scikit-learn",
    }

    result = set()
    for item in items:
        skill = clean(item).casefold()
        skill = aliases.get(skill, skill)
        if skill:
            result.add(skill)

    return result


def main():
    if not JOB_FILE.exists():
        raise FileNotFoundError(f"Job dataset not found: {JOB_FILE}")

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}\n"
            "Nothing was overwritten. Rename or move that file "
            "if you intentionally want to regenerate it."
        )

    jobs = pd.read_csv(JOB_FILE, low_memory=False)

    if "job_id" not in jobs.columns or "title" not in jobs.columns:
        raise ValueError("Source data must contain job_id and title.")

    for column in [
        "company_name", "job_text", "normalized_skills",
        "parsed_skills", "skills_required",
    ]:
        if column not in jobs.columns:
            jobs[column] = ""

    jobs["title"] = jobs["title"].map(clean)
    jobs["job_id"] = jobs["job_id"].map(clean)

    jobs = jobs[
        jobs["job_id"].ne("") & jobs["title"].ne("")
    ].drop_duplicates(subset="job_id").copy()

    jobs["all_skills"] = jobs.apply(
        lambda row: (
            parse_skills(row["normalized_skills"])
            | parse_skills(row["parsed_skills"])
            | parse_skills(row["skills_required"])
        ),
        axis=1,
    )

    jobs["title_lower"] = jobs["title"].str.casefold()

    records = []

    for profile_id, profile in PROFILES.items():
        skill_set = profile["skills"]
        title_terms = profile["title_terms"]

        scored = jobs.copy()

        scored["title_match_count"] = scored["title_lower"].map(
            lambda title: sum(term in title for term in title_terms)
        )

        scored["skill_overlap_count"] = scored["all_skills"].map(
            lambda skills: len(skill_set.intersection(skills))
        )

        # Selection score only; this is NOT a relevance label.
        scored["selection_score"] = (
            3 * scored["title_match_count"]
            + scored["skill_overlap_count"]
        )

        # Target-role candidates are chosen from the strongest title/skill
        # matches, with a deterministic tie-break.
        target_pool = scored[scored["selection_score"] > 0].sort_values(
            ["selection_score", "title_match_count", "skill_overlap_count"],
            ascending=False,
        )

        targets = target_pool.head(TARGET_COUNT).copy()

        # Distractors come from jobs with no title-term match and minimal
        # skill overlap where possible. Selection does not imply irrelevance.
        used_ids = set(targets["job_id"])
        remaining = scored[~scored["job_id"].isin(used_ids)].copy()

        low_match = remaining[
            (remaining["title_match_count"] == 0)
            & (remaining["skill_overlap_count"] <= 1)
        ]

        if len(low_match) < DISTRACTOR_COUNT:
            low_match = remaining.sort_values(
                ["selection_score", "title_match_count"],
                ascending=True,
            )

        distractors = low_match.sample(
            n=min(DISTRACTOR_COUNT, len(low_match)),
            random_state=RANDOM_SEED,
        )

        selected = [
            (row, "target_role_candidate")
            for _, row in targets.iterrows()
        ] + [
            (row, "distractor_candidate")
            for _, row in distractors.iterrows()
        ]

        for row, stratum in selected:
            records.append({
                "user_id": profile_id,
                "profile_text": profile["profile"],
                "profile_skills": ", ".join(sorted(skill_set)),
                "job_id": row["job_id"],
                "job_title": row["title"],
                "company_name": clean(row["company_name"]),
                "job_text": clean(row["job_text"]),
                "job_skills": clean(row["normalized_skills"]),
                "candidate_stratum": stratum,
                "selection_score": int(row["selection_score"]),
                "relevant": "",
                "review_notes": "",
            })

        print(
            f"{profile_id}: {len(targets)} target-role candidates; "
            f"{len(distractors)} distractor candidates"
        )

    evaluation = pd.DataFrame(records)

    if evaluation.empty:
        raise ValueError("No evaluation candidates were generated.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 65)
    print("CAREERINTEL — BALANCED CANDIDATE POOL")
    print("=" * 65)
    print(f"Source jobs available: {len(jobs)}")
    print(f"Candidate rows: {len(evaluation)}")
    print(f"Profiles: {evaluation['user_id'].nunique()}")
    print(f"Relevance labels populated: {evaluation['relevant'].ne('').sum()}")
    print(f"Saved to: {OUTPUT_FILE.relative_to(ROOT)}")
    print("\nCandidate stratum counts:")
    print(
        evaluation.groupby(
            ["user_id", "candidate_stratum"]
        ).size().to_string()
    )
    print("\nLabels remain blank for independent review.")
    print("Selection scores must not be interpreted as relevance labels.")


if __name__ == "__main__":
    main()
