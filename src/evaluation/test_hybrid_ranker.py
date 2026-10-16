"""Functional test of CareerIntel's hybrid recommendation model."""

import ast
from pathlib import Path

import pandas as pd

from careerintel.models.hybrid_ranker import HybridRanker

ROOT = Path(__file__).resolve().parents[2]

JOB_FILE = (
    ROOT
    / "data"
    / "processed"
    / "jobs"
    / "jobs_with_job_ids.csv"
)


def parse_skills(value):
    """Convert profile skills into a Python list."""

    if isinstance(value, list):
        return value

    if value is None or pd.isna(value):
        return []

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            return list(parsed)

        if isinstance(parsed, str):
            return [parsed]

    except (ValueError, SyntaxError):
        pass

    return [skill.strip() for skill in text.split(",") if skill.strip()]


def main():
    if not JOB_FILE.exists():
        raise FileNotFoundError(f"Job dataset not found: {JOB_FILE}")

    # Load a manageable sample for the first functional test.
    jobs_df = pd.read_csv(JOB_FILE, nrows=1000, low_memory=False)

    required_columns = {"job_id", "title"}
    missing = required_columns - set(jobs_df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    jobs_df = jobs_df.dropna(subset=["job_id", "title"]).copy()

    jobs_df["job_id"] = jobs_df["job_id"].astype(str).str.strip()
    jobs_df["title"] = jobs_df["title"].astype(str).str.strip()

    jobs_df = jobs_df[
        jobs_df["job_id"].ne("") & jobs_df["title"].ne("")
    ].drop_duplicates(subset="job_id")

    jobs = jobs_df.to_dict(orient="records")

    if not jobs:
        raise ValueError("No valid jobs found in the sample.")

    profile_text = (
        "Aspiring Data Scientist with Python, SQL, statistics, "
        "machine learning, data analysis, pandas, and data visualization."
    )

    profile_skills = [
        "Python",
        "SQL",
        "Statistics",
        "Machine Learning",
        "Data Analysis",
        "Pandas",
        "Data Visualization",
    ]

    print("=" * 65)
    print("CAREERINTEL — HYBRID RANKER FUNCTIONAL TEST")
    print("=" * 65)
    print(f"Jobs loaded: {len(jobs)}")
    print("Profile: Aspiring Data Scientist")
    print("\nPreparing semantic embeddings...")

    ranker = HybridRanker()
    ranker.fit(jobs)

    results = ranker.rank(
        profile_text=profile_text,
        profile_skills=profile_skills,
        top_k=10,
    )

    print("\nTop 10 recommendations:")
    print("-" * 65)

    for position, result in enumerate(results, start=1):
        print(
            f"{position:2}. {result['job_title']}\n"
            f"    Job ID: {result['job_id']}\n"
            f"    Hybrid: {result['hybrid_score']:.4f} | "
            f"Semantic: {result['semantic_score']:.4f} | "
            f"Skills: {result['skill_score']:.4f} | "
            f"Title: {result['title_score']:.4f}\n"
            f"    Matched skills: "
            f"{', '.join(result['matched_skills']) or 'None detected'}\n"
        )

    print("Functional test completed.")
    print("No dataset or evaluation files were modified.")


if __name__ == "__main__":
    main()