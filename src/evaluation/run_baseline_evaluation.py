"""CareerIntel — TF-IDF baseline evaluation.

Uses provisional assistant labels, not independently verified ground truth.
Existing results are never overwritten.
"""

from pathlib import Path

import pandas as pd

from careerintel.models.baselines import TfidfRanker
from evaluation.evaluate_recommendations import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    reciprocal_rank,
)

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "careerintel_assistant_provisional_review.csv"
)

OUTPUT_DIR = ROOT / "results" / "baseline_evaluation"
K = 10


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {
        "user_id",
        "profile_text",
        "job_id",
        "job_title",
        "job_text",
        "assistant_provisional_relevant",
        "assistant_label_source",
        "assistant_label_verified",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Evaluation dataset is empty.")

    if df.duplicated(["user_id", "job_id"]).any():
        raise ValueError("Duplicate user/job pairs found.")

    if not (
        df["assistant_label_source"]
        .eq("assistant_provisional_review")
        .all()
    ):
        raise ValueError("Unexpected label provenance.")

    if df["assistant_label_verified"].fillna(True).astype(bool).any():
        raise ValueError(
            "This runner expects unverified provisional labels."
        )

    # Exclude rows with uncertain or missing relevance labels.
    df = df[
        pd.to_numeric(
            df["assistant_provisional_relevant"],
            errors="coerce",
        ).isin([0, 1])
    ].copy()

    if df.empty:
        raise ValueError("No binary provisional labels are available.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    recommendations_file = OUTPUT_DIR / "tfidf_ranked_recommendations.csv"
    metrics_file = OUTPUT_DIR / "tfidf_evaluation_metrics.csv"
    per_profile_file = OUTPUT_DIR / "tfidf_metrics_by_profile.csv"

    outputs = [
        recommendations_file,
        metrics_file,
        per_profile_file,
    ]

    existing = [str(p) for p in outputs if p.exists()]
    if existing:
        raise FileExistsError(
            "Existing results will not be overwritten:\n"
            + "\n".join(existing)
        )

    ranked_records = []
    metric_records = []

    for user_id, group in df.groupby("user_id"):
        group = group.copy()

        query = str(group["profile_text"].iloc[0])

        skills = (
            group["job_skills"].fillna("").astype(str)
            if "job_skills" in group.columns
            else pd.Series("", index=group.index)
        )

        texts = (
            group["job_title"].fillna("").astype(str)
            + " "
            + group["job_text"].fillna("").astype(str)
            + " "
            + skills
        ).tolist()

        ranker = TfidfRanker()
        ranker.fit(group["job_id"].tolist(), texts)

        ranked = ranker.rank(query, top_k=len(group))
        scores = dict(ranked)

        group["score"] = group["job_id"].map(scores)
        group = group.sort_values(
            ["score", "job_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

        group["rank"] = range(1, len(group) + 1)
        relevance = group["assistant_provisional_relevant"].to_numpy()
        total_relevant = int((relevance == 1).sum())

        ranked_records.extend(
            group[
                [
                    "user_id",
                    "job_id",
                    "job_title",
                    "score",
                    "rank",
                    "assistant_provisional_relevant",
                    "assistant_label_source",
                ]
            ].to_dict("records")
        )

        metric_records.append({
            "user_id": user_id,
            "candidate_count": len(group),
            "relevant_candidates": total_relevant,
            f"precision_at_{K}": precision_at_k(relevance, K),
            f"recall_at_{K}": recall_at_k(
                relevance, total_relevant, K
            ),
            f"ndcg_at_{K}": ndcg_at_k(relevance, K),
            "mrr": reciprocal_rank(relevance),
            "label_type": "provisional_assistant",
        })

    per_profile = pd.DataFrame(metric_records)

    summary = pd.DataFrame([{
        "profiles_evaluated": len(per_profile),
        "candidate_count": int(per_profile["candidate_count"].sum()),
        f"mean_precision_at_{K}": per_profile[
            f"precision_at_{K}"
        ].mean(),
        f"mean_recall_at_{K}": per_profile[
            f"recall_at_{K}"
        ].mean(),
        f"mean_ndcg_at_{K}": per_profile[
            f"ndcg_at_{K}"
        ].mean(),
        "mean_mrr": per_profile["mrr"].mean(),
        "label_type": "provisional_assistant",
    }])

    pd.DataFrame(ranked_records).to_csv(
        recommendations_file, index=False
    )
    summary.to_csv(metrics_file, index=False)
    per_profile.to_csv(per_profile_file, index=False)

    print("=" * 60)
    print("CAREERINTEL — TF-IDF BASELINE")
    print("=" * 60)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nPer-profile results:")
    print(per_profile.to_string(index=False))
    print(f"\nResults saved in: {OUTPUT_DIR.relative_to(ROOT)}")
    print(
        "\nIMPORTANT: These metrics measure agreement with provisional "
        "assistant labels, not independently verified ground truth."
    )


if __name__ == "__main__":
    main()