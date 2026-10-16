from pathlib import Path

import pandas as pd


# ============================================================
# CAREERINTEL — TOP-10 RANKING INTEGRITY AUDIT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REVIEW_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "independent_relevance_review_completed_preliminary.csv"
)

MODEL_FILES = {
    "TF-IDF": (
        PROJECT_ROOT
        / "results"
        / "baseline_evaluation"
        / "tfidf_ranked_recommendations.csv"
    ),
    "Semantic": (
        PROJECT_ROOT
        / "results"
        / "semantic_evaluation"
        / "semantic_ranked_recommendations.csv"
    ),
    "Hybrid": (
        PROJECT_ROOT
        / "results"
        / "hybrid_evaluation"
        / "hybrid_ranked_recommendations.csv"
    ),
}

OUTPUT_DIR = PROJECT_ROOT / "results" / "top10_audit"

K = 10
LABEL_COLUMN = "reviewer_relevant"


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in ["user_id", "job_id"]:
        df[column] = df[column].astype("string").str.strip()

    return df


def require_columns(
    df: pd.DataFrame,
    columns: set[str],
    description: str,
) -> None:
    missing = columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{description} is missing columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )


# ============================================================
# LOAD REVIEW LABELS
# ============================================================

def load_review() -> pd.DataFrame:
    if not REVIEW_FILE.exists():
        raise FileNotFoundError(
            f"Review file not found:\n{REVIEW_FILE}"
        )

    review = pd.read_csv(REVIEW_FILE)

    require_columns(
        review,
        {"user_id", "job_id", LABEL_COLUMN},
        "Review dataset",
    )

    review = normalize_ids(review)

    review = review.dropna(
        subset=["user_id", "job_id", LABEL_COLUMN]
    ).copy()

    labels = pd.to_numeric(
        review[LABEL_COLUMN],
        errors="coerce",
    )

    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(
            "Review labels must contain only 0 or 1."
        )

    review[LABEL_COLUMN] = labels.astype(int)

    if review.duplicated(["user_id", "job_id"]).any():
        raise ValueError(
            "Duplicate user/job pairs found in the review dataset."
        )

    if "reviewer_confidence" not in review.columns:
        review["reviewer_confidence"] = "unknown"

    if "reviewer_notes" not in review.columns:
        review["reviewer_notes"] = ""

    review["reviewer_confidence"] = (
        review["reviewer_confidence"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return review[
        [
            "user_id",
            "job_id",
            LABEL_COLUMN,
            "reviewer_confidence",
            "reviewer_notes",
        ]
    ]


# ============================================================
# LOAD MODEL RANKINGS
# ============================================================

def load_rankings(
    model_name: str,
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} ranking file not found:\n{path}"
        )

    rankings = pd.read_csv(path)

    require_columns(
        rankings,
        {"user_id", "job_id", "rank"},
        f"{model_name} rankings",
    )

    rankings = normalize_ids(rankings)

    rankings["rank"] = pd.to_numeric(
        rankings["rank"],
        errors="coerce",
    )

    rankings = rankings.dropna(
        subset=["user_id", "job_id", "rank"]
    ).copy()

    if (rankings["rank"] < 1).any():
        raise ValueError(
            f"{model_name} contains ranks below 1."
        )

    if rankings.duplicated(["user_id", "job_id"]).any():
        raise ValueError(
            f"{model_name} contains duplicate user/job pairs."
        )

    if rankings.duplicated(["user_id", "rank"]).any():
        raise ValueError(
            f"{model_name} contains duplicate ranks for a user."
        )

    rankings["rank"] = rankings["rank"].astype(int)

    return rankings.sort_values(
        ["user_id", "rank"],
        kind="stable",
    ).reset_index(drop=True)


# ============================================================
# AUDIT THE ACTUAL TOP K
# ============================================================

def audit_model(
    model_name: str,
    rankings: pd.DataFrame,
    review: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Preserve every candidate's original position.
    top_k = rankings.loc[rankings["rank"] <= K].copy()

    top_k = top_k.merge(
        review,
        on=["user_id", "job_id"],
        how="left",
        validate="one_to_one",
    )

    # Missing labels remain unknown, not non-relevant.
    top_k["label_status"] = "unreviewed"

    top_k.loc[
        top_k[LABEL_COLUMN] == 0,
        "label_status",
    ] = "reviewed_non_relevant"

    top_k.loc[
        top_k[LABEL_COLUMN] == 1,
        "label_status",
    ] = "reviewed_relevant"

    top_k["model"] = model_name

    summaries = []

    section(f"MODEL: {model_name}")

    print("Total ranking rows:", len(rankings))
    print("Actual top-K rows:", len(top_k))

    for user_id, group in top_k.groupby(
        "user_id",
        sort=True,
    ):
        group = group.sort_values("rank")

        relevant = int(
            (group["label_status"] == "reviewed_relevant").sum()
        )

        non_relevant = int(
            (group["label_status"] == "reviewed_non_relevant").sum()
        )

        unreviewed = int(
            (group["label_status"] == "unreviewed").sum()
        )

        reviewed_count = relevant + non_relevant

        # Conditional precision among reviewed top-K jobs only.
        # This is not full Precision@K when any job is unreviewed.
        reviewed_precision = (
            relevant / reviewed_count
            if reviewed_count > 0
            else float("nan")
        )

        summaries.append(
            {
                "model": model_name,
                "user_id": user_id,
                "top_k": K,
                "actual_top_k_rows": len(group),
                "reviewed_relevant": relevant,
                "reviewed_non_relevant": non_relevant,
                "unreviewed": unreviewed,
                "reviewed_candidates": reviewed_count,
                "precision_among_reviewed_only": reviewed_precision,
            }
        )

        print(f"\nProfile: {user_id}")
        print(
            group[
                [
                    "rank",
                    "job_id",
                    "label_status",
                    "reviewer_confidence",
                ]
            ].to_string(index=False)
        )

        print("\nSummary:")
        print("  Relevant:", relevant)
        print("  Non-relevant:", non_relevant)
        print("  Unreviewed:", unreviewed)
        print("  Reviewed candidates:", reviewed_count)

        if reviewed_count:
            print(
                "  Precision among reviewed candidates only:",
                f"{reviewed_precision:.4f}",
            )

        if unreviewed:
            print(
                "  NOTE: Full Precision@10 cannot be established "
                "from these labels."
            )

    return top_k, pd.DataFrame(summaries)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    section("CAREERINTEL — TOP-10 RANKING INTEGRITY AUDIT")

    print("Project root:", PROJECT_ROOT)
    print("Review file:", REVIEW_FILE)
    print("Evaluation cutoff:", K)

    review = load_review()

    section("REVIEW DATASET")

    print("Reviewed pairs:", len(review))
    print("Profiles:", review["user_id"].nunique())

    print("\nRelevance distribution:")
    print(review[LABEL_COLUMN].value_counts().sort_index())

    print("\nConfidence distribution:")
    print(review["reviewer_confidence"].value_counts())

    all_top_k = []
    all_summaries = []

    for model_name, model_path in MODEL_FILES.items():
        rankings = load_rankings(
            model_name,
            model_path,
        )

        top_k, summary = audit_model(
            model_name,
            rankings,
            review,
        )

        all_top_k.append(top_k)
        all_summaries.append(summary)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    top_k_output = OUTPUT_DIR / "top10_candidates_audited.csv"
    summary_output = OUTPUT_DIR / "top10_summary_by_profile.csv"

    pd.concat(
        all_top_k,
        ignore_index=True,
    ).to_csv(top_k_output, index=False)

    pd.concat(
        all_summaries,
        ignore_index=True,
    ).to_csv(summary_output, index=False)

    section("AUDIT SUMMARY")

    summary_df = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    print(
        summary_df.to_string(index=False)
    )

    print("\nSaved candidate-level audit:")
    print(top_k_output)

    print("\nSaved profile-level summary:")
    print(summary_output)

    section("INTERPRETATION")

    print(
        "1. Original ranking positions are preserved."
    )
    print(
        "2. Unreviewed candidates remain unknown, not non-relevant."
    )
    print(
        "3. Precision among reviewed candidates is conditional "
        "and must not be presented as full Precision@K."
    )
    print(
        "4. These labels are preliminary; their independence from "
        "the recommendation models has not been established."
    )
    print(
        "5. Independent relevance assessment is needed for "
        "stronger research conclusions."
    )

    print("\nAudit completed. Existing results were not overwritten.")


if __name__ == "__main__":
    main()