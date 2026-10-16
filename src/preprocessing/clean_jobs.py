from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "jobs"
    / "jobs_sample_10000.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "jobs"
OUTPUT_PATH = OUTPUT_DIR / "jobs_cleaned.csv"


def clean_jobs() -> None:
    print("Loading CareerIntel job dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Original shape: {df.shape}")

    # --------------------------------------------------
    # 1. Remove completely empty rows
    # --------------------------------------------------
    df = df.dropna(how="all")

    # --------------------------------------------------
    # 2. Remove exact duplicate rows
    # --------------------------------------------------
    df = df.drop_duplicates()

    # --------------------------------------------------
    # 3. Normalize text columns
    # --------------------------------------------------
    text_columns = [
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
        "education_level",
        "skills_required",
        "minimum_qualifications",
        "preferred_qualifications",
        "responsibilities",
        "certifications",
        "languages_required",
        "city",
        "country",
        "location_resolved",
        "locations",
        "remote_eligibility_regions",
        "job_description",
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            )

    # --------------------------------------------------
    # 4. Convert numeric columns
    # --------------------------------------------------
    numeric_columns = [
        "years_experience_numeric",
        "salary_min",
        "salary_max",
        "latitude",
        "longitude",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # --------------------------------------------------
    # 5. Create combined text for NLP
    # --------------------------------------------------
    text_parts = [
        "title",
        "skills_required",
        "minimum_qualifications",
        "preferred_qualifications",
        "responsibilities",
        "job_description",
    ]

    existing_text_parts = [
        column for column in text_parts if column in df.columns
    ]

    df["job_text"] = (
        df[existing_text_parts]
        .fillna("")
        .astype("string")
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # --------------------------------------------------
    # 6. Remove jobs without usable text
    # --------------------------------------------------
    before_text_filter = len(df)

    df = df[df["job_text"].str.len() > 20].copy()

    removed = before_text_filter - len(df)

    print(f"Removed jobs with insufficient text: {removed}")

    # --------------------------------------------------
    # 7. Create output directory
    # --------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 8. Save cleaned dataset
    # --------------------------------------------------
    df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("=" * 60)
    print("CareerIntel preprocessing complete")
    print("=" * 60)
    print(f"Final shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    clean_jobs()