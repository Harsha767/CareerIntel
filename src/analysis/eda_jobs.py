from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jobs"
    / "jobs_cleaned.csv"
)


def main():
    print("Loading cleaned CareerIntel dataset...")

    df = pd.read_csv(INPUT_PATH)

    print()
    print("=" * 60)
    print("CAREERINTEL — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(f"\nSHAPE: {df.shape}")

    print("\nDATA TYPES:")
    print(df.dtypes.to_string())

    print("\nMISSING VALUES (%):")
    missing = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    print(missing.to_string())

    print("\nDUPLICATE ROWS:")
    print(df.duplicated().sum())

    print("\nTOP JOB TITLES:")
    print(df["title"].value_counts().head(20).to_string())

    print("\nEXPERIENCE LEVELS:")
    print(df["experience_level"].value_counts(dropna=False).to_string())

    print("\nWORK MODELS:")
    print(df["work_model"].value_counts(dropna=False).to_string())

    print("\nEMPLOYMENT TYPES:")
    print(df["employment_type"].value_counts(dropna=False).to_string())

    print("\nTOP COUNTRIES:")
    print(df["country"].value_counts(dropna=False).head(20).to_string())

    print("\nTOP INDUSTRIES:")
    print(df["industry"].value_counts(dropna=False).head(20).to_string())

    print("\nEDUCATION LEVELS:")
    print(df["education_level"].value_counts(dropna=False).to_string())

    print("\nNUMERIC SUMMARY:")
    numeric_columns = [
        "years_experience_numeric",
        "salary_min",
        "salary_max",
        "latitude",
        "longitude",
    ]

    available_numeric = [
        column for column in numeric_columns
        if column in df.columns
    ]

    print(df[available_numeric].describe().to_string())

    print("\nJOB TEXT LENGTH:")
    text_lengths = df["job_text"].fillna("").str.len()

    print(text_lengths.describe().to_string())

    print("\nTOP JOB TITLES AFTER NORMALIZATION:")
    print(
        df["normalized_title"]
        .value_counts()
        .head(20)
        .to_string()
    )

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()