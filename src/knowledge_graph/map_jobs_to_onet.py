from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

JOBS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jobs"
    / "jobs_with_normalized_skills.csv"
)

ONET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "onet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "jobs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(value):
    """Normalize text for title matching."""
    if pd.isna(value):
        return ""

    return (
        str(value)
        .lower()
        .strip()
        .replace("-", " ")
        .replace("/", " ")
    )


def main():
    print("=" * 60)
    print("CareerIntel - Job to O*NET Mapping")
    print("=" * 60)

    print("\nLoading job dataset...")
    jobs = pd.read_csv(JOBS_FILE)

    print(f"Jobs loaded: {len(jobs):,}")

    print("\nLoading O*NET title mapping...")
    onet = pd.read_csv(
        ONET_DIR / "job_title_occupation_mapping.csv"
    )

    print(f"O*NET title mappings: {len(onet):,}")

    # ---------------------------------------------------------
    # Normalize titles
    # ---------------------------------------------------------

    jobs["title_normalized_match"] = jobs["title"].apply(normalize_text)

    onet["job_title_normalized"] = onet["job_title"].apply(
        normalize_text
    )

    onet["onet_title_normalized"] = onet["occupation_title"].apply(
        normalize_text
    )

    # ---------------------------------------------------------
    # Exact job-title match
    # ---------------------------------------------------------

    title_lookup = (
        onet[
            [
                "job_title_normalized",
                "onet_soc_code",
                "occupation_title",
            ]
        ]
        .drop_duplicates("job_title_normalized")
        .set_index("job_title_normalized")
    )

    jobs["onet_soc_code"] = jobs["title_normalized_match"].map(
        title_lookup["onet_soc_code"]
    )

    jobs["onet_occupation_title"] = jobs[
        "title_normalized_match"
    ].map(
        title_lookup["occupation_title"]
    )

    jobs["onet_match_method"] = "exact_title"

    jobs.loc[
        jobs["onet_soc_code"].isna(),
        "onet_match_method"
    ] = "unmatched"

    # ---------------------------------------------------------
    # Remove temporary matching column
    # ---------------------------------------------------------

    jobs.drop(
        columns=["title_normalized_match"],
        inplace=True,
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    matched = jobs["onet_soc_code"].notna().sum()
    unmatched = jobs["onet_soc_code"].isna().sum()

    match_rate = matched / len(jobs) * 100

    print()
    print("=" * 60)
    print("MAPPING RESULTS")
    print("=" * 60)

    print(f"Total jobs:       {len(jobs):,}")
    print(f"Matched jobs:     {matched:,}")
    print(f"Unmatched jobs:   {unmatched:,}")
    print(f"Exact match rate: {match_rate:.2f}%")

    # ---------------------------------------------------------
    # Save complete mapping
    # ---------------------------------------------------------

    output_file = OUTPUT_DIR / "jobs_mapped_to_onet.csv"

    jobs.to_csv(
        output_file,
        index=False,
    )

    # ---------------------------------------------------------
    # Save unmatched jobs for later NLP matching
    # ---------------------------------------------------------

    unmatched_jobs = jobs[
        jobs["onet_soc_code"].isna()
    ].copy()

    unmatched_file = OUTPUT_DIR / "jobs_unmatched_to_onet.csv"

    unmatched_jobs.to_csv(
        unmatched_file,
        index=False,
    )

    # ---------------------------------------------------------
    # Show sample mappings
    # ---------------------------------------------------------

    print()
    print("SAMPLE MATCHES")
    print("-" * 60)

    sample = jobs[
        jobs["onet_soc_code"].notna()
    ][
        [
            "title",
            "onet_soc_code",
            "onet_occupation_title",
        ]
    ].head(15)

    print(sample.to_string(index=False))

    print()
    print("=" * 60)
    print("JOB → O*NET MAPPING COMPLETE")
    print("=" * 60)

    print(f"Output: {output_file}")
    print(f"Unmatched: {unmatched_file}")


if __name__ == "__main__":
    main()
    