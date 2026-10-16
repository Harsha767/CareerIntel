from pathlib import Path

from datasets import load_dataset
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "jobs"

DATASET_NAME = "NextGig-Rocks/global-job-postings-multi-ats"

SELECTED_COLUMNS = [
    "title",
    "normalized_title",
    "company_name",
    "ats_name",
    "industry",
    "function",
    "occupational_category",
    "employment_type",
    "work_model",
    "experience_level",
    "years_experience_numeric",
    "education_level",
    "skills_required",
    "minimum_qualifications",
    "preferred_qualifications",
    "responsibilities",
    "certifications",
    "languages_required",
    "salary_min",
    "salary_max",
    "salary_currency",
    "city",
    "country",
    "location_resolved",
    "locations",
    "remote_eligibility_regions",
    "date_posted",
    "job_description",
]


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CareerIntel job dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    rows = []

    for index, row in enumerate(dataset):
        cleaned_row = {
            column: row.get(column)
            for column in SELECTED_COLUMNS
        }

        rows.append(cleaned_row)

        if (index + 1) % 1000 == 0:
            print(f"Processed {index + 1:,} jobs")

        # Development sample: first 10,000 jobs.
        if index + 1 >= 10000:
            break

    df = pd.DataFrame(rows)

    output_path = RAW_DATA_DIR / "jobs_sample_10000.csv"

    df.to_csv(output_path, index=False)

    print()
    print("=" * 60)
    print("CareerIntel ingestion complete")
    print("=" * 60)
    print(f"Rows saved: {len(df):,}")
    print(f"Columns saved: {len(df.columns)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()