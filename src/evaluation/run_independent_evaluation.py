"""CareerIntel — Preliminary Review-Label Evaluation.

Evaluates TF-IDF, Semantic, and Hybrid job rankings against
preliminary relevance-review labels.

Important:
- Candidates with missing relevance labels are excluded.
- All models must rank the same complete set of labeled candidates.
- Existing evaluation outputs are never overwritten.
- Preliminary review labels are not verified ground truth.
"""

import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

REVIEW_FILE = (
    ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "independent_relevance_review_completed_preliminary.csv"
)

MODEL_FILES = {
    "TF-IDF": (
        ROOT
        / "results"
        / "baseline_evaluation"
        / "tfidf_ranked_recommendations.csv"
    ),
    "Semantic": (
        ROOT
        / "results"
        / "semantic_evaluation"
        / "semantic_ranked_recommendations.csv"
    ),
    "Hybrid": (
        ROOT
        / "results"
        / "hybrid_evaluation"
        / "hybrid_ranked_recommendations.csv"
    ),
}

OUTPUT_ROOT = ROOT / "results" / "independent_evaluation"

K = 10


# ---------------------------------------------------------
# RANKING METRICS
# ---------------------------------------------------------

def precision_at_k(labels, k=10):
    """Calculate Precision@K."""

    if k < 1:
        raise ValueError("k must be positive.")

    top_k = labels[:k]

    return sum(label > 0 for label in top_k) / k


def recall_at_k(labels, total_relevant, k=10):
    """Calculate Recall@K."""

    if k < 1:
        raise ValueError("k must be positive.")

    if total_relevant < 0:
        raise ValueError("total_relevant cannot be negative.")

    if total_relevant == 0:
        return float("nan")

    relevant_in_top_k = sum(
        label > 0 for label in labels[:k]
    )

    return relevant_in_top_k / total_relevant


def dcg_at_k(labels, k=10):
    """Calculate Discounted Cumulative Gain@K."""

    if k < 1:
        raise ValueError("k must be positive.")

    return sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(
            labels[:k],
            start=1,
        )
    )


def ndcg_at_k(labels, k=10):
    """Calculate NDCG@K using the complete candidate set for IDCG."""

    if k < 1:
        raise ValueError("k must be positive.")

    actual_dcg = dcg_at_k(labels, k)

    # Ideal ranking is based on all labeled candidates.
    ideal_labels = sorted(labels, reverse=True)
    ideal_dcg = dcg_at_k(ideal_labels, k)

    if ideal_dcg == 0:
        return float("nan")

    return actual_dcg / ideal_dcg


def reciprocal_rank(labels):
    """Calculate reciprocal rank of the first relevant candidate."""

    for rank, relevance in enumerate(labels, start=1):
        if relevance > 0:
            return 1.0 / rank

    return 0.0


# ---------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------

def normalize_ids(df):
    """Normalize user and job IDs for consistent comparisons."""

    df = df.copy()

    for column in ["user_id", "job_id"]:
        df[column] = df[column].astype("string").str.strip()

        if df[column].isna().any() or df[column].eq("").any():
            raise ValueError(
                f"Missing or empty values found in {column}."
            )

    return df


def load_review():
    """Load review labels and exclude candidates without labels."""

    if not REVIEW_FILE.exists():
        raise FileNotFoundError(
            f"Review file not found: {REVIEW_FILE}"
        )

    review = pd.read_csv(REVIEW_FILE)

    required = {
        "user_id",
        "job_id",
        "reviewer_relevant",
    }

    missing = required - set(review.columns)

    if missing:
        raise ValueError(
            f"Review file is missing columns: {sorted(missing)}"
        )

    review = normalize_ids(review)

    if review.duplicated(["user_id", "job_id"]).any():
        raise ValueError(
            "Review dataset contains duplicate user/job pairs."
        )

    review["reviewer_relevant"] = pd.to_numeric(
        review["reviewer_relevant"],
        errors="coerce",
    )

    missing_labels = int(
        review["reviewer_relevant"].isna().sum()
    )

    # Exclude unlabeled pairs rather than assigning artificial labels.
    review = review.dropna(
        subset=["reviewer_relevant"]
    ).copy()

    if review.empty:
        raise ValueError(
            "No labeled candidates remain after excluding missing labels."
        )

    if not review["reviewer_relevant"].isin([0, 1]).all():
        raise ValueError(
            "Relevance labels must contain only 0 or 1."
        )

    review["reviewer_relevant"] = (
        review["reviewer_relevant"].astype(int)
    )

    print(f"Review rows originally loaded: {missing_labels + len(review)}")
    print(f"Candidates excluded for missing labels: {missing_labels}")
    print(f"Labeled candidates retained: {len(review)}")

    print(
        "Relevant labels:",
        int(review["reviewer_relevant"].sum()),
    )

    print(
        "Non-relevant labels:",
        int((review["reviewer_relevant"] == 0).sum()),
    )

    return review


def load_and_validate_rankings(model_name, model_path, review):
    """Load rankings and validate complete candidate coverage."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_name} results not found: {model_path}"
        )

    ranked = pd.read_csv(model_path)

    required = {
        "user_id",
        "job_id",
        "rank",
    }

    missing = required - set(ranked.columns)

    if missing:
        raise ValueError(
            f"{model_name} file is missing columns: "
            f"{sorted(missing)}"
        )

    ranked = normalize_ids(ranked)

    ranked["rank"] = pd.to_numeric(
        ranked["rank"],
        errors="coerce",
    )

    if ranked["rank"].isna().any():
        raise ValueError(
            f"{model_name} contains missing or invalid ranks."
        )

    if not ranked["rank"].map(math.isfinite).all():
        raise ValueError(
            f"{model_name} contains non-finite ranks."
        )

    if (ranked["rank"] < 1).any():
        raise ValueError(
            f"{model_name} contains ranks below 1."
        )

    if not ranked["rank"].map(
        lambda value: float(value).is_integer()
    ).all():
        raise ValueError(
            f"{model_name} contains non-integer ranks."
        )

    ranked["rank"] = ranked["rank"].astype(int)

    if ranked.duplicated(["user_id", "job_id"]).any():
        raise ValueError(
            f"{model_name} contains duplicate user/job pairs."
        )

    if ranked.duplicated(["user_id", "rank"]).any():
        raise ValueError(
            f"{model_name} assigns the same rank to multiple "
            "jobs for one user."
        )

    # Compare against the labeled review set only.
    review_pairs = set(
        zip(review["user_id"], review["job_id"])
    )

    ranking_pairs = set(
        zip(ranked["user_id"], ranked["job_id"])
    )

    missing_pairs = review_pairs - ranking_pairs
    extra_pairs = ranking_pairs - review_pairs

    if missing_pairs or extra_pairs:
        details = [
            f"{model_name} candidate coverage mismatch:",
            f"  Labeled pairs missing from rankings: {len(missing_pairs)}",
            f"  Unlabeled or unreviewed pairs in rankings: {len(extra_pairs)}",
        ]

        if missing_pairs:
            details.append(
                f"Example missing pairs: {sorted(missing_pairs)[:10]}"
            )

        if extra_pairs:
            details.append(
                f"Example extra pairs: {sorted(extra_pairs)[:10]}"
            )

        raise ValueError("\n".join(details))

    review_users = set(review["user_id"])
    ranking_users = set(ranked["user_id"])

    if review_users != ranking_users:
        raise ValueError(
            f"{model_name} user coverage differs from the review."
        )

    # Ranks must be consecutive, beginning at 1, for each user.
    for user_id, group in ranked.groupby("user_id"):
        actual_ranks = sorted(group["rank"].tolist())
        expected_ranks = list(range(1, len(group) + 1))

        if actual_ranks != expected_ranks:
            raise ValueError(
                f"{model_name} has missing or non-consecutive "
                f"ranks for user {user_id}."
            )

    # Attach labels only after candidate coverage has been validated.
    label_lookup = review.set_index(
        ["user_id", "job_id"]
    )["reviewer_relevant"]

    ranked["reviewer_relevant"] = pd.MultiIndex.from_frame(
        ranked[["user_id", "job_id"]]
    ).map(label_lookup)

    if ranked["reviewer_relevant"].isna().any():
        raise ValueError(
            f"{model_name} contains candidates without relevance labels."
        )

    ranked["reviewer_relevant"] = (
        ranked["reviewer_relevant"].astype(int)
    )

    return ranked.sort_values(
        ["user_id", "rank"]
    ).copy()


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate_model(model_name, ranked, review):
    """Calculate ranking metrics for each user profile."""

    profile_results = []

    for user_id, group in ranked.groupby("user_id", sort=True):
        labels = (
            group.sort_values("rank")["reviewer_relevant"].tolist()
        )

        # The complete labeled review set defines the denominator.
        total_relevant = int(
            review.loc[
                review["user_id"] == user_id,
                "reviewer_relevant",
            ].sum()
        )

        if total_relevant == 0:
            raise ValueError(
                f"No relevant reviewed jobs for profile {user_id}."
            )

        profile_results.append(
            {
                "user_id": user_id,
                "model": model_name,
                "candidate_count": len(labels),
                "relevant_candidates": total_relevant,
                "precision_at_10": precision_at_k(labels, K),
                "recall_at_10": recall_at_k(
                    labels,
                    total_relevant,
                    K,
                ),
                "ndcg_at_10": ndcg_at_k(labels, K),
                "mrr": reciprocal_rank(labels),
                "label_source": "preliminary_review_labels",
            }
        )

    return profile_results


# ---------------------------------------------------------
# OUTPUT HANDLING
# ---------------------------------------------------------

def create_output_directory():
    """Create a new output directory without overwriting previous results."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    output_dir = OUTPUT_ROOT / f"run_{timestamp}"

    # Avoid accidental collisions if called more than once per second.
    counter = 1

    while output_dir.exists():
        output_dir = OUTPUT_ROOT / f"run_{timestamp}_{counter}"
        counter += 1

    output_dir.mkdir(parents=True, exist_ok=False)

    return output_dir


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    review = load_review()

    print("\n" + "=" * 70)
    print("CAREERINTEL — PRELIMINARY REVIEW-LABEL EVALUATION")
    print("=" * 70)

    print(f"Review file: {REVIEW_FILE.relative_to(ROOT)}")
    print(f"Models to evaluate: {len(MODEL_FILES)}")

    # Validate every model before calculating or writing results.
    validated_rankings = {}

    for model_name, model_path in MODEL_FILES.items():
        ranked = load_and_validate_rankings(
            model_name,
            model_path,
            review,
        )

        validated_rankings[model_name] = ranked

        print(
            f"\n{model_name}: candidate coverage verified "
            f"({len(ranked)} rows)."
        )

    all_profile_results = []

    for model_name, ranked in validated_rankings.items():
        print(f"\nEvaluating: {model_name}")

        results = evaluate_model(
            model_name,
            ranked,
            review,
        )

        all_profile_results.extend(results)

        for result in results:
            print(
                f"  {result['user_id']}: "
                f"P@10={result['precision_at_10']:.4f}, "
                f"R@10={result['recall_at_10']:.4f}, "
                f"NDCG@10={result['ndcg_at_10']:.4f}, "
                f"MRR={result['mrr']:.4f}"
            )

    per_profile = pd.DataFrame(all_profile_results)

    summary = (
        per_profile.groupby("model")
        .agg(
            profiles_evaluated=("user_id", "nunique"),
            mean_precision_at_10=("precision_at_10", "mean"),
            mean_recall_at_10=("recall_at_10", "mean"),
            mean_ndcg_at_10=("ndcg_at_10", "mean"),
            mean_mrr=("mrr", "mean"),
        )
        .reindex(["TF-IDF", "Semantic", "Hybrid"])
        .reset_index()
    )

    output_dir = create_output_directory()

    per_profile_file = output_dir / "metrics_by_profile.csv"
    summary_file = output_dir / "evaluation_summary.csv"

    per_profile.to_csv(
        per_profile_file,
        index=False,
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(summary.round(4).to_string(index=False))

    print(f"\nResults saved to: {output_dir.relative_to(ROOT)}")

    print(
        "\nCaution: These metrics measure agreement with preliminary "
        "review labels. The labels are not verified human ground truth "
        "and do not establish real-world job suitability."
    )


if __name__ == "__main__":
    main()