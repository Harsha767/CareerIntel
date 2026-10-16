"""CareerIntel: prepare a human-reviewed recommendation evaluation dataset."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

JOB_FILE = (
    ROOT
    / "data"
    / "processed"
    / "jobs"
    / "jobs_with_job_ids.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "evaluation_candidates.csv"


PROFILES = {
    "DS001": {
        "profile": (
            "Aspiring Data Scientist with Python, SQL, machine learning, "
            "statistics, data analysis, pandas, and data visualization."
        ),
        "skills": [
            "python",
            "sql",
            "machine learning",
            "statistics",
            "data analysis",
            "pandas",
            "data visualization",
        ],
    },
    "DA001": {
        "profile": (
            "Data Analyst skilled in SQL, Python, data analysis, "
            "reporting, dashboards, Excel, and data visualization."
        ),
        "skills": [
            "sql",
            "python",
            "data analysis",
            "reporting",
            "dashboards",
            "excel",
            "data visualization",
        ],
    },
    "ML001": {
        "profile": (
            "Machine Learning Engineer interested in Python, machine "
            "learning, deep learning, model evaluation, and deployment."
        ),
        "skills": [
            "python",
            "machine learning",
            "deep learning",
            "model evaluation",
            "model deployment",
        ],
    },
}


def main():
    if not JOB_FILE.exists():
        raise FileNotFoundError(
            f"Job dataset not found: {JOB_FILE}"
        )

    jobs = pd.read_csv(JOB_FILE, low_memory=False)

    required_columns = {"job_id", "title"}
    missing = required_columns - set(jobs.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Keep a manageable and diverse candidate pool.
    jobs = jobs.copy()

    jobs["title"] = jobs["title"].fillna("").astype(str)

    if "job_text" in jobs.columns:
        jobs["job_text"] = jobs["job_text"].fillna("").astype(str)
    else:
        jobs["job_text"] = ""

    if "normalized_skills" in jobs.columns:
        jobs["normalized_skills"] = (
            jobs["normalized_skills"].fillna("").astype(str)
        )
    else:
        jobs["normalized_skills"] = ""

    jobs = jobs[
        jobs["title"].str.strip().ne("")
    ].drop_duplicates(subset="job_id")

    # Use a reproducible sample so the candidate pool can be recreated.
    sample_size = min(100, len(jobs))

    candidates = jobs.sample(
        n=sample_size,
        random_state=42,
    )

    records = []

    for profile_id, profile_data in PROFILES.items():
        for _, job in candidates.iterrows():
            records.append(
                {
                    "user_id": profile_id,
                    "profile_text": profile_data["profile"],
                    "profile_skills": ", ".join(
                        profile_data["skills"]
                    ),
                    "job_id": str(job["job_id"]),
                    "job_title": job["title"],
                    "company_name": job.get(
                        "company_name", ""
                    ),
                    "job_text": job["job_text"],
                    "job_skills": job["normalized_skills"],
                    "relevant": "",
                    "review_notes": "",
                }
            )

    evaluation = pd.DataFrame(records)

    evaluation.to_csv(OUTPUT_FILE, index=False)

    print("=" * 65)
    print("CAREERINTEL | EVALUATION DATASET PREPARATION")
    print("=" * 65)

    print(f"Profiles: {len(PROFILES)}")
    print(f"Sampled jobs: {sample_size}")
    print(f"Profile-job pairs: {len(evaluation)}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nReview instructions:")
    print("1. Open the generated CSV in a spreadsheet.")
    print("2. Review each profile against its job.")
    print("3. Set relevant to 1 if the job is relevant.")
    print("4. Set relevant to 0 if it is not relevant.")
    print("5. Leave uncertain cases blank until reviewed.")
    print("6. Record any useful explanation in review_notes.")

    print("\nNo relevance labels were generated automatically.")


if __name__ == "__main__":
    main()