from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CAREERINTEL — RANKING DIAGNOSTICS
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

OUTPUT_DIR = PROJECT_ROOT / "results" / "ranking_diagnostics"

K = 10

LABEL_COLUMN = "reviewer_relevant"

LABEL_TYPE = "preliminary_ai_assisted_review"


# ============================================================
# GENERAL HELPERS
# ============================================================

def print_section(title: str) -> None:
    """Print a formatted section heading."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    description: str,
) -> None:
    """Validate that the required columns exist."""
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{description} is missing required columns: "
            f"{sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )


def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize identifiers for reliable matching."""
    df = df.copy()

    for column in ("user_id", "job_id"):
        df[column] = df[column].astype("string").str.strip()

    return df


# ============================================================
# METRIC CALCULATIONS
# ============================================================

def calculate_metrics(
    relevance,
    total_relevant: int,
    k: int = K,
) -> dict:
    """
    Calculate Precision@K, Recall@K, NDCG@K, and MRR@K.

    Parameters
    ----------
    relevance:
        Binary relevance labels ordered by predicted rank.
        Relevant = 1; non-relevant = 0.

        The first K elements are evaluated.

    total_relevant:
        Number of relevant jobs in the complete reviewed
        candidate set for this profile.

        This denominator comes from the review dataset,
        not just the candidates returned by the model.

    k:
        Evaluation cutoff.

    Notes
    -----
    Precision@K:
        Relevant jobs in the top K divided by K.

    Recall@K:
        Relevant jobs in the top K divided by all relevant
        jobs in the reviewed candidate set.

    NDCG@K:
        Discounted cumulative gain divided by ideal DCG.
        Binary relevance is used.

    MRR@K:
        Reciprocal rank of the first relevant job in the
        top K. Returns zero if none is found in the top K.
    """

    relevance = np.asarray(relevance, dtype=float).reshape(-1)

    if k < 1:
        raise ValueError("k must be at least 1.")

    if total_relevant < 0:
        raise ValueError("total_relevant cannot be negative.")

    if np.isnan(relevance).any():
        raise ValueError(
            "Relevance contains missing labels. "
            "Do not silently treat unreviewed jobs as irrelevant."
        )

    if not np.isin(relevance, [0, 1]).all():
        raise ValueError(
            "Relevance labels must contain only 0 and 1."
        )

    top_k = relevance[:k]

    relevant_top_k = int(top_k.sum())

    # Precision@K uses K as the denominator.
    # If fewer than K recommendations exist, this measures
    # precision against the requested K recommendation slots.
    precision = relevant_top_k / k

    # Recall@K uses the complete reviewed relevance set.
    recall = (
        relevant_top_k / total_relevant
        if total_relevant > 0
        else np.nan
    )

    # NDCG@K: actual discounted cumulative gain.
    if len(top_k) > 0:
        discounts = np.log2(
            np.arange(2, len(top_k) + 2)
        )

        dcg = float(
            np.sum(top_k / discounts)
        )
    else:
        dcg = 0.0

    # The ideal ranking places relevant candidates first.
    ideal_relevant_count = min(total_relevant, k)

    if ideal_relevant_count > 0:
        ideal_discounts = np.log2(
            np.arange(2, ideal_relevant_count + 2)
        )

        ideal_dcg = float(
            np.sum(1.0 / ideal_discounts)
        )

        ndcg = dcg / ideal_dcg
    else:
        ndcg = np.nan

    # MRR@K: reciprocal rank of the first relevant result.
    relevant_positions = np.flatnonzero(top_k == 1)

    if len(relevant_positions) > 0:
        mrr = 1.0 / (int(relevant_positions[0]) + 1)
    else:
        mrr = 0.0

    return {
        f"precision_at_{k}": float(precision),
        f"recall_at_{k}": float(recall),
        f"ndcg_at_{k}": float(ndcg),
        f"mrr_at_{k}": float(mrr),
        "candidate_count": int(len(relevance)),
        "evaluated_at_k": int(len(top_k)),
        "relevant_in_top_k": relevant_top_k,
        "total_relevant_reviewed": int(total_relevant),
    }


# ============================================================
# LOAD AND VALIDATE REVIEW DATA
# ============================================================

def load_review_data() -> pd.DataFrame:
    """Load and validate the relevance review dataset."""

    if not REVIEW_FILE.exists():
        raise FileNotFoundError(
            f"Review dataset not found:\n{REVIEW_FILE}"
        )

    review = pd.read_csv(REVIEW_FILE)

    required = {
        "user_id",
        "job_id",
        LABEL_COLUMN,
    }

    require_columns(
        review,
        required,
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

    invalid = labels.isna() | ~labels.isin([0, 1])

    if invalid.any():
        bad_values = (
            review.loc[invalid, LABEL_COLUMN]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Invalid relevance labels found: {bad_values}. "
            "Expected binary labels 0 or 1."
        )

    review[LABEL_COLUMN] = labels.astype(int)

    # Each user/job pair must have one unambiguous label.
    duplicates = review.duplicated(
        ["user_id", "job_id"],
        keep=False,
    )

    if duplicates.any():
        example = review.loc[
            duplicates,
            ["user_id", "job_id", LABEL_COLUMN],
        ]

        raise ValueError(
            "Duplicate user/job pairs found in the review data:\n"
            f"{example.head(20).to_string(index=False)}"
        )

    return review


# ============================================================
# LOAD AND VALIDATE MODEL RANKINGS
# ============================================================

def load_model_rankings(
    model_name: str,
    model_file: Path,
) -> pd.DataFrame:
    """Load and validate one model's ranked recommendations."""

    if not model_file.exists():
        raise FileNotFoundError(
            f"{model_name} rankings file not found:\n"
            f"{model_file}"
        )

    df = pd.read_csv(model_file)

    required = {
        "user_id",
        "job_id",
        "rank",
    }

    require_columns(
        df,
        required,
        f"{model_name} rankings",
    )

    df = normalize_ids(df)

    df["rank"] = pd.to_numeric(
        df["rank"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["user_id", "job_id", "rank"]
    ).copy()

    if (df["rank"] < 1).any():
        raise ValueError(
            f"{model_name}: ranks must be at least 1."
        )

    if not np.equal(
        df["rank"].to_numpy(),
        np.floor(df["rank"].to_numpy()),
    ).all():
        raise ValueError(
            f"{model_name}: ranks must be integers."
        )

    df["rank"] = df["rank"].astype(int)

    duplicate_jobs = df.duplicated(
        ["user_id", "job_id"],
        keep=False,
    )

    if duplicate_jobs.any():
        example = df.loc[
            duplicate_jobs,
            ["user_id", "job_id", "rank"],
        ]

        raise ValueError(
            f"{model_name} contains duplicate user/job pairs:\n"
            f"{example.head(20).to_string(index=False)}"
        )

    duplicate_ranks = df.duplicated(
        ["user_id", "rank"],
        keep=False,
    )

    if duplicate_ranks.any():
        example = df.loc[
            duplicate_ranks,
            ["user_id", "job_id", "rank"],
        ]

        raise ValueError(
            f"{model_name} contains duplicate ranks per user:\n"
            f"{example.head(20).to_string(index=False)}"
        )

    df = df.sort_values(
        ["user_id", "rank"],
        kind="stable",
    ).reset_index(drop=True)

    return df


# ============================================================
# ALIGN REVIEW LABELS WITH RANKINGS
# ============================================================

def align_review_labels(
    rankings: pd.DataFrame,
    review: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match ranking candidates with review labels.

    Returns
    -------
    labelled_rankings:
        Ranking rows that have review labels.

    missing_reviewed_pairs:
        Reviewed user/job pairs missing from the model ranking.
    """

    review_keys = review[
        ["user_id", "job_id", LABEL_COLUMN]
    ].copy()

    labelled_rankings = rankings.merge(
        review_keys,
        on=["user_id", "job_id"],
        how="left",
        validate="one_to_one",
    )

    # Preserve ranking positions with missing labels.
    # Do not convert unknown labels to zero.
    ranking_keys = rankings[
        ["user_id", "job_id"]
    ].drop_duplicates()

    missing_reviewed_pairs = review.merge(
        ranking_keys,
        on=["user_id", "job_id"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    missing_reviewed_pairs = (
        missing_reviewed_pairs.loc[
            missing_reviewed_pairs["_merge"] == "left_only"
        ]
        .drop(columns="_merge")
        .copy()
    )

    return labelled_rankings, missing_reviewed_pairs


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    model_name: str,
    rankings: pd.DataFrame,
    review: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a model using the available review labels."""

    labelled, missing_pairs = align_review_labels(
        rankings,
        review,
    )

    print_section(f"MODEL: {model_name}")

    print(f"Ranking rows: {len(rankings)}")
    print(f"Ranking rows with review labels: {labelled[LABEL_COLUMN].notna().sum()}")
    print(f"Ranking rows without review labels: {labelled[LABEL_COLUMN].isna().sum()}")
    print(f"Reviewed pairs missing from ranking: {len(missing_pairs)}")

    # Full reviewed-set relevance counts, including reviewed
    # candidates that are absent from the model's ranking.
    total_relevant_by_user = (
        review.groupby("user_id")[LABEL_COLUMN]
        .sum()
        .astype(int)
        .to_dict()
    )

    profile_results = []

    for user_id, group in labelled.groupby(
        "user_id",
        sort=True,
    ):
        group = group.sort_values(
            "rank",
            kind="stable",
        ).copy()

        # Evaluation is restricted to the top K positions.
        top_k = group.head(K)

        # An unreviewed item in the top K prevents a reliable
        # calculation because its relevance is unknown.
        if top_k[LABEL_COLUMN].isna().any():
            missing_count = int(
                top_k[LABEL_COLUMN].isna().sum()
            )

            print(
                f"WARNING: {model_name}, {user_id}: "
                f"{missing_count} top-{K} candidates lack "
                "review labels. Metrics for this profile "
                "will be skipped."
            )

            continue

        relevance = top_k[LABEL_COLUMN].to_numpy(dtype=int)

        total_relevant = total_relevant_by_user.get(
            user_id,
            0,
        )

        metrics = calculate_metrics(
            relevance=relevance,
            total_relevant=total_relevant,
            k=K,
        )

        metrics["user_id"] = user_id
        metrics["model"] = model_name
        metrics["label_type"] = LABEL_TYPE

        profile_results.append(metrics)

        print(
            f"{user_id}: "
            f"P@{K}={metrics[f'precision_at_{K}']:.4f}, "
            f"R@{K}={metrics[f'recall_at_{K}']:.4f}, "
            f"NDCG@{K}={metrics[f'ndcg_at_{K}']:.4f}, "
            f"MRR@{K}={metrics[f'mrr_at_{K}']:.4f}, "
            f"Relevant in top {K}="
            f"{metrics['relevant_in_top_k']}, "
            f"Total relevant reviewed={total_relevant}"
        )

    profile_metrics = pd.DataFrame(profile_results)

    return profile_metrics, missing_pairs, labelled


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_section("CAREERINTEL — RANKING DIAGNOSTICS")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Review file: {REVIEW_FILE}")
    print(f"Evaluation cutoff: {K}")
    print(f"Label type: {LABEL_TYPE}")

    # --------------------------------------------------------
    # Load review dataset
    # --------------------------------------------------------

    review = load_review_data()

    print_section("REVIEW DATA SUMMARY")

    print(f"Review rows: {len(review)}")
    print(f"Unique profiles: {review['user_id'].nunique()}")
    print(
        "Unique reviewed pairs: "
        f"{review[['user_id', 'job_id']].drop_duplicates().shape[0]}"
    )

    print("\nRelevance label distribution:")

    print(
        review[LABEL_COLUMN]
        .value_counts()
        .sort_index()
        .rename(
            index={
                0: "Non-relevant (0)",
                1: "Relevant (1)",
            }
        )
        .to_string()
    )

    print("\nRelevance distribution by profile:")

    print(
        review.groupby("user_id")[LABEL_COLUMN]
        .agg(
            reviewed_candidates="count",
            total_relevant="sum",
            relevance_rate="mean",
        )
        .round(4)
        .to_string()
    )

    # --------------------------------------------------------
    # Evaluate all models
    # --------------------------------------------------------

    all_profile_metrics = []
    all_missing_pairs = []
    all_labelled_rankings = []

    for model_name, model_file in MODEL_FILES.items():
        rankings = load_model_rankings(
            model_name,
            model_file,
        )

        profile_metrics, missing_pairs, labelled = evaluate_model(
            model_name,
            rankings,
            review,
        )

        if not profile_metrics.empty:
            all_profile_metrics.append(profile_metrics)

        if not missing_pairs.empty:
            missing_pairs = missing_pairs.copy()
            missing_pairs["model"] = model_name
            all_missing_pairs.append(missing_pairs)

        if not labelled.empty:
            labelled = labelled.copy()
            labelled["model"] = model_name
            all_labelled_rankings.append(labelled)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if all_profile_metrics:
        profile_df = pd.concat(
            all_profile_metrics,
            ignore_index=True,
        )

        metric_columns = [
            f"precision_at_{K}",
            f"recall_at_{K}",
            f"ndcg_at_{K}",
            f"mrr_at_{K}",
        ]

        profile_output = (
            OUTPUT_DIR / "metrics_by_profile.csv"
        )

        profile_df.to_csv(
            profile_output,
            index=False,
        )

        # Macro-average across evaluated profiles.
        summary = (
            profile_df.groupby("model")[metric_columns]
            .mean()
            .reindex(["TF-IDF", "Semantic", "Hybrid"])
            .reset_index()
        )

        summary_output = (
            OUTPUT_DIR / "model_summary.csv"
        )

        summary.to_csv(
            summary_output,
            index=False,
        )

        print_section("EVALUATION SUMMARY")

        print(
            summary.round(4).to_string(index=False)
        )

        print(
            "\nPer-profile metrics saved to:\n"
            f"{profile_output}"
        )

        print(
            "\nModel summary saved to:\n"
            f"{summary_output}"
        )

    else:
        print_section("EVALUATION SUMMARY")
        print("No profiles could be evaluated.")

    if all_missing_pairs:
        missing_df = pd.concat(
            all_missing_pairs,
            ignore_index=True,
        )

        missing_output = (
            OUTPUT_DIR / "missing_reviewed_pairs.csv"
        )

        missing_df.to_csv(
            missing_output,
            index=False,
        )

        print(
            "\nMissing reviewed pairs saved to:\n"
            f"{missing_output}"
        )

    else:
        print(
            "\nNo reviewed pairs were missing from the "
            "model rankings."
        )

    if all_labelled_rankings:
        labelled_df = pd.concat(
            all_labelled_rankings,
            ignore_index=True,
        )

        labelled_output = (
            OUTPUT_DIR / "labelled_rankings.csv"
        )

        labelled_df.to_csv(
            labelled_output,
            index=False,
        )

        print(
            "\nLabelled rankings saved to:\n"
            f"{labelled_output}"
        )

    # --------------------------------------------------------
    # Interpretation and limitations
    # --------------------------------------------------------

    print_section("INTERPRETATION AND LIMITATIONS")

    print(
        "1. Metrics measure agreement with preliminary "
        "AI-assisted relevance labels."
    )

    print(
        "2. These labels are not independently verified "
        "ground truth."
    )

    print(
        "3. Missing reviewed pairs are reported separately "
        "and remain part of the reviewed-set recall denominator."
    )

    print(
        "4. Rankings with unreviewed items in their top-K "
        "positions are skipped for that profile rather than "
        "silently treating unknown relevance as zero."
    )

    print(
        "5. Recall uses all relevant candidates in the review "
        "dataset for that profile."
    )

    print(
        "6. These metrics do not establish real-world job "
        "suitability or prove that one model is universally best."
    )

    print(
        "7. Independent human assessment and a larger evaluation "
        "sample are needed for stronger research conclusions."
    )

    print("\nDiagnostic completed.")


if __name__ == "__main__":
    main()