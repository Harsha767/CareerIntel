"""CareerIntel — Sentence Transformer evaluation.

Evaluates semantic ranking against provisional assistant labels.
These labels are not independently verified ground truth.

Existing results are never overwritten.
"""

from pathlib import Path

import pandas as pd

from careerintel.models.semantic_ranker import SemanticRanker
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

OUTPUT_DIR = ROOT / "results" / "semantic_evaluation"
K = 10


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {INPUT_FILE}"
        )

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
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.duplicated(["user_id", "job_id"]).any():
        raise ValueError("Duplicate user/job pairs found.")

    if not df["assistant_label_source"].eq(
        "assistant_provisional_review"
    ).all():
        raise ValueError("Unexpected label provenance.")

    verified = (
        df["assistant_label_verified"]
        .fillna(True)
        .astype(bool)
    )

    if verified.any():
        raise ValueError(
            "This runner expects unverified provisional labels."
        )

    # Exclude missing or non-binary labels.
    df["assistant_provisional_relevant"] = pd.to_numeric(
        df["assistant_provisional_relevant"],
        errors="coerce",
    )

    df = df[
        df["assistant_provisional_relevant"].isin([0, 1])
    ].copy()

    if df.empty:
        raise ValueError("No usable relevance labels found.")

    # Refuse to overwrite previous results.
    output_files = [
        OUTPUT_DIR / "semantic_ranked_recommendations.csv",
        OUTPUT_DIR / "semantic_evaluation_metrics.csv",
        OUTPUT_DIR / "semantic_metrics_by_profile.csv",
    ]

    existing = [str(path) for path in output_files if path.exists()]

    if existing:
        raise FileExistsError(
            "Existing results will not be overwritten:\n"
            + "\n".join(existing)
        )

    ranked_records = []
    metric_records = []

    # One ranking per candidate profile.
    for user_id, group in df.groupby("user_id", sort=True):
        group = group.copy()

        profile = str(group["profile_text"].iloc[0])

        if group["profile_text"].nunique() != 1:
            raise ValueError(
                f"Inconsistent profile text for user {user_id}."
            )

        job_skills = (
            group["job_skills"].fillna("").astype(str)
            if "job_skills" in group.columns
            else pd.Series("", index=group.index)
        )

        # Combine title, description, and available skills.
        job_texts = (
            group["job_title"].fillna("").astype(str)
            + ". "
            + group["job_text"].fillna("").astype(str)
            + ". Skills: "
            + job_skills
        ).tolist()

        ranker = SemanticRanker()

        ranker.fit(
            ids=group["job_id"].tolist(),
            texts=job_texts,
        )

        ranking = ranker.rank(
            query=profile,
            top_k=len(group),
        )

        scores = dict(ranking)

        group["score"] = group["job_id"].map(scores)

        group = group.sort_values(
            ["score", "job_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

        group["rank"] = range(1, len(group) + 1)

        relevance = group[
            "assistant_provisional_relevant"
        ].to_numpy()

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
            f"precision_at_{K}": precision_at_k(
                relevance, K
            ),
            f"recall_at_{K}": recall_at_k(
                relevance, total_relevant, K
            ),
            f"ndcg_at_{K}": ndcg_at_k(
                relevance, K
            ),
            "mrr": reciprocal_rank(relevance),
            "label_type": "provisional_assistant",
        })

        print(
            f"Completed semantic ranking for {user_id}: "
            f"{len(group)} candidates"
        )

    per_profile = pd.DataFrame(metric_records)

    summary = pd.DataFrame([{
        "profiles_evaluated": len(per_profile),
        "candidate_count": int(
            per_profile["candidate_count"].sum()
        ),
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(ranked_records).to_csv(
        output_files[0], index=False
    )

    summary.to_csv(output_files[1], index=False)

    per_profile.to_csv(output_files[2], index=False)

    print("\n" + "=" * 65)
    print("CAREERINTEL — SEMANTIC EVALUATION")
    print("=" * 65)

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nPer-profile metrics:")
    print(per_profile.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_DIR.relative_to(ROOT)}")

    print(
        "\nLIMITATION: Results measure agreement with provisional "
        "assistant labels, not independently verified ground truth."
    )


if __name__ == "__main__":
    main()