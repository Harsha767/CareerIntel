from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

PROJECT_ROOT = Path(__file__).resolve().parents[2]

JOBS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jobs"
    / "jobs_mapped_to_onet.csv"
)

ONET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "onet"
    / "occupation_master.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "jobs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCH_THRESHOLD = 70


def normalize_text(value):
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
    print("CareerIntel - Fuzzy Job Title Matching")
    print("=" * 60)

    jobs = pd.read_csv(JOBS_FILE)
    occupations = pd.read_csv(ONET_FILE)

    print(f"Jobs loaded: {len(jobs):,}")
    print(f"O*NET occupations: {len(occupations):,}")

    occupations["title_normalized"] = occupations[
        "occupation_title"
    ].apply(normalize_text)

    occupation_titles = occupations[
        "title_normalized"
    ].drop_duplicates().tolist()

    occupation_lookup = occupations.set_index(
        "title_normalized"
    ).to_dict("index")

    unmatched_mask = jobs["onet_soc_code"].isna()

    unmatched_jobs = jobs[unmatched_mask].copy()

    print(f"Jobs requiring fuzzy matching: {len(unmatched_jobs):,}")

    fuzzy_codes = []
    fuzzy_titles = []
    fuzzy_scores = []

    for index, row in unmatched_jobs.iterrows():

        job_title = normalize_text(row["title"])

        if not job_title:
            fuzzy_codes.append(None)
            fuzzy_titles.append(None)
            fuzzy_scores.append(0)
            continue

        result = process.extractOne(
            job_title,
            occupation_titles,
            scorer=fuzz.token_set_ratio,
        )

        if result is None:
            fuzzy_codes.append(None)
            fuzzy_titles.append(None)
            fuzzy_scores.append(0)
            continue

        matched_title, score, _ = result

        if score >= MATCH_THRESHOLD:
            matched_record = occupation_lookup[matched_title]

            fuzzy_codes.append(
                matched_record["onet_soc_code"]
            )

            fuzzy_titles.append(
                matched_record["occupation_title"]
            )

            fuzzy_scores.append(score)

        else:
            fuzzy_codes.append(None)
            fuzzy_titles.append(None)
            fuzzy_scores.append(score)

    unmatched_jobs["fuzzy_onet_soc_code"] = fuzzy_codes
    unmatched_jobs["fuzzy_onet_occupation_title"] = fuzzy_titles
    unmatched_jobs["fuzzy_match_score"] = fuzzy_scores

    # ---------------------------------------------------------
    # Combine exact + fuzzy results
    # ---------------------------------------------------------

    jobs["match_score"] = 100.0
    jobs["match_method"] = jobs["onet_match_method"]

    fuzzy_columns = unmatched_jobs[
        [
            "fuzzy_onet_soc_code",
            "fuzzy_onet_occupation_title",
            "fuzzy_match_score",
        ]
    ]

    jobs.loc[
        unmatched_mask,
        "onet_soc_code"
    ] = fuzzy_columns["fuzzy_onet_soc_code"].values

    jobs.loc[
        unmatched_mask,
        "onet_occupation_title"
    ] = fuzzy_columns["fuzzy_onet_occupation_title"].values

    jobs.loc[
        unmatched_mask,
        "match_score"
    ] = fuzzy_columns["fuzzy_match_score"].values

    jobs.loc[
        unmatched_mask & jobs["onet_soc_code"].notna(),
        "match_method"
    ] = "fuzzy_title"

    jobs.loc[
        jobs["onet_soc_code"].isna(),
        "match_method"
    ] = "unmatched"

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    total = len(jobs)

    exact_matches = (
        jobs["match_method"] == "exact_title"
    ).sum()

    fuzzy_matches = (
        jobs["match_method"] == "fuzzy_title"
    ).sum()

    unmatched = (
        jobs["match_method"] == "unmatched"
    ).sum()

    matched = exact_matches + fuzzy_matches

    print()
    print("=" * 60)
    print("FUZZY MATCHING RESULTS")
    print("=" * 60)

    print(f"Total jobs:       {total:,}")
    print(f"Exact matches:    {exact_matches:,}")
    print(f"Fuzzy matches:    {fuzzy_matches:,}")
    print(f"Still unmatched:  {unmatched:,}")
    print(f"Overall matched:  {matched:,}")
    print(f"Overall rate:     {matched / total * 100:.2f}%")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_file = (
        OUTPUT_DIR / "jobs_mapped_to_onet_fuzzy.csv"
    )

    jobs.to_csv(
        output_file,
        index=False,
    )

    unmatched_file = (
        OUTPUT_DIR / "jobs_still_unmatched.csv"
    )

    jobs[
        jobs["match_method"] == "unmatched"
    ].to_csv(
        unmatched_file,
        index=False,
    )

    # ---------------------------------------------------------
    # Show fuzzy examples
    # ---------------------------------------------------------

    print()
    print("FUZZY MATCH EXAMPLES")
    print("-" * 60)

    examples = jobs[
        jobs["match_method"] == "fuzzy_title"
    ][
        [
            "title",
            "onet_occupation_title",
            "match_score",
        ]
    ].head(20)

    print(
        examples.to_string(index=False)
    )

    print()
    print("=" * 60)
    print("FUZZY MATCHING COMPLETE")
    print("=" * 60)

    print(f"Output: {output_file}")
    print(f"Unmatched: {unmatched_file}")


if __name__ == "__main__":
    main()