"""CareerIntel: Recommendation Evaluation Module.

Implements ranking metrics for evaluating job recommendations.

Metrics:
    Precision@K
    Recall@K
    NDCG@K
    Mean Reciprocal Rank (MRR)

Important:
    Metrics require relevance labels. Do not treat semantic
    similarity scores as ground-truth relevance labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "evaluation_report.csv"


# ---------------------------------------------------------
# RANKING METRICS
# ---------------------------------------------------------

def precision_at_k(relevance, k=10):
    """Calculate Precision@K using binary relevance labels."""

    if k < 1:
        raise ValueError("k must be positive.")

    relevance = np.asarray(relevance, dtype=float)[:k]

    if np.isnan(relevance).any():
        raise ValueError("Relevance labels cannot contain NaN.")

    if len(relevance) == 0:
        return 0.0

    return float(np.sum(relevance > 0) / k)


def recall_at_k(relevance, total_relevant, k=10):
    """Calculate Recall@K."""

    if k < 1:
        raise ValueError("k must be positive.")

    if total_relevant < 0:
        raise ValueError("total_relevant cannot be negative.")

    if total_relevant == 0:
        return np.nan

    relevance = np.asarray(relevance, dtype=float)[:k]

    if np.isnan(relevance).any():
        raise ValueError("Relevance labels cannot contain NaN.")

    return float(
        np.sum(relevance > 0) / total_relevant
    )


def dcg_at_k(relevance, k=10):
    """Calculate Discounted Cumulative Gain@K."""

    if k < 1:
        raise ValueError("k must be positive.")

    relevance = np.asarray(relevance, dtype=float)[:k]

    if np.isnan(relevance).any():
        raise ValueError("Relevance labels cannot contain NaN.")

    if len(relevance) == 0:
        return 0.0

    discounts = np.log2(
        np.arange(2, len(relevance) + 2)
    )

    gains = (2 ** relevance - 1) / discounts

    return float(np.sum(gains))


def ndcg_at_k(relevance, k=10):
    """Calculate NDCG@K using all candidates to determine IDCG.

    Actual DCG is calculated from the top-K ranked candidates.

    Ideal DCG is calculated from the best possible ordering
    of the full candidate set, truncated to K.

    The relevance array must contain the complete candidate
    set for the user, ordered by the model's predicted rank.
    """

    if k < 1:
        raise ValueError("k must be positive.")

    relevance = np.asarray(relevance, dtype=float)

    if np.isnan(relevance).any():
        raise ValueError("Relevance labels cannot contain NaN.")

    # Actual DCG: use the model's top-K recommendations.
    actual_dcg = dcg_at_k(relevance, k)

    # Ideal DCG: sort the FULL candidate set before truncating.
    ideal_relevance = np.sort(relevance)[::-1]
    ideal_dcg = dcg_at_k(ideal_relevance, k)

    if ideal_dcg == 0:
        return np.nan

    return float(actual_dcg / ideal_dcg)


def reciprocal_rank(relevance):
    """Return reciprocal rank of the first relevant item."""

    relevance = np.asarray(relevance, dtype=float)

    if np.isnan(relevance).any():
        raise ValueError("Relevance labels cannot contain NaN.")

    relevant_positions = np.flatnonzero(
        relevance > 0
    )

    if len(relevant_positions) == 0:
        return 0.0

    return float(1 / (relevant_positions[0] + 1))


# ---------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------

def validate_relevance_labels(labels):
    """Validate a binary relevance-label column."""

    if labels.isna().any():
        raise ValueError(
            "Relevance labels contain missing values."
        )

    values = set(labels.unique())

    if not values.issubset({0, 1}):
        raise ValueError(
            "Relevance labels must be binary: 0 or 1."
        )


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate_recommendations(
    labels_file,
    k=10,
):
    """
    Evaluate ranked recommendations against relevance labels.

    Required CSV columns:
        user_id
        job_id
        rank
        relevant

    Each row represents a candidate job for a user.

    relevant:
        1 = relevant
        0 = not relevant

    Important:
        Each user's rows must contain the complete candidate
        set being evaluated, with valid relevance labels.
    """

    if k < 1:
        raise ValueError("k must be positive.")

    labels_path = Path(labels_file)

    if not labels_path.is_absolute():
        labels_path = ROOT / labels_path

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Evaluation labels not found: {labels_path}\n"
            "Create a labeled evaluation CSV before evaluating."
        )

    df = pd.read_csv(labels_path)

    required_columns = {
        "user_id",
        "job_id",
        "rank",
        "relevant",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    validate_relevance_labels(df["relevant"])

    if df[
        ["user_id", "job_id", "rank"]
    ].isna().any().any():
        raise ValueError(
            "User IDs, job IDs, and ranks cannot be missing."
        )

    df["rank"] = pd.to_numeric(
        df["rank"],
        errors="raise",
    )

    if not np.isfinite(df["rank"]).all():
        raise ValueError("Ranks must be finite numbers.")

    if (df["rank"] < 1).any():
        raise ValueError(
            "Ranks must start at 1."
        )

    if df.duplicated(["user_id", "job_id"]).any():
        raise ValueError(
            "Duplicate user/job pairs found in evaluation data."
        )

    if df.duplicated(["user_id", "rank"]).any():
        raise ValueError(
            "A user cannot have multiple jobs at the same rank."
        )

    df = df.sort_values(
        ["user_id", "rank"]
    )

    # Calculate metrics separately for each user.
    user_results = []

    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("rank")

        relevance = group["relevant"].to_numpy(dtype=int)

        # Valid only if this group contains the complete
        # labeled candidate set for this user.
        total_relevant = int(
            np.sum(relevance == 1)
        )

        user_results.append(
            {
                "user_id": user_id,
                f"precision_at_{k}": precision_at_k(
                    relevance,
                    k,
                ),
                f"recall_at_{k}": recall_at_k(
                    relevance,
                    total_relevant,
                    k,
                ),
                f"ndcg_at_{k}": ndcg_at_k(
                    relevance,
                    k,
                ),
                "reciprocal_rank": reciprocal_rank(
                    relevance
                ),
            }
        )

    per_user = pd.DataFrame(user_results)

    report = pd.DataFrame(
        [
            {
                "users_evaluated": len(per_user),
                f"precision_at_{k}": per_user[
                    f"precision_at_{k}"
                ].mean(),
                f"recall_at_{k}": per_user[
                    f"recall_at_{k}"
                ].mean(),
                f"ndcg_at_{k}": per_user[
                    f"ndcg_at_{k}"
                ].mean(),
                "mrr": per_user[
                    "reciprocal_rank"
                ].mean(),
            }
        ]
    )

    # Preserve the existing output filenames.
    report.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    per_user.to_csv(
        RESULTS_DIR / "evaluation_per_user.csv",
        index=False,
    )

    return report


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 65)
    print("CAREERINTEL | RECOMMENDATION EVALUATION")
    print("=" * 65)

    labels_file = (
        RESULTS_DIR / "evaluation_labels.csv"
    )

    if not labels_file.exists():
        print(
            "\nEvaluation labels have not been created yet."
        )

        print(
            "\nThe evaluation dataset must contain these columns:"
        )

        print(
            "user_id, job_id, rank, relevant"
        )

        print(
            "\nDo not generate relevance labels from model "
            "similarity scores."
        )

        print(
            "\nNext step: construct a valid labeled evaluation "
            "dataset from historical interactions or manually "
            "reviewed job relevance."
        )

    else:
        report = evaluate_recommendations(
            labels_file=labels_file,
            k=10,
        )

        print("\n=== EVALUATION RESULTS ===")

        print(
            report.to_string(index=False)
        )

        print(
            f"\nSaved evaluation report to: {OUTPUT_FILE}"
        )