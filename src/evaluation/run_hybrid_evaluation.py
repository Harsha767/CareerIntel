"""CareerIntel — Hybrid recommendation evaluation.

Evaluates hybrid job ranking against provisional relevance labels.

IMPORTANT:
These labels are not independently verified ground truth.
Existing results are never overwritten.
"""

from pathlib import Path

import pandas as pd

from careerintel.models.hybrid_ranker import HybridRanker
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

OUTPUT_DIR = ROOT / "results" / "hybrid_evaluation"

K = 10


def main():
    # --------------------------------------------------
    # 1. Validate the input dataset
    # --------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "user_id",
        "profile_text",
        "profile_skills",
        "job_id",
        "job_title",
        "job_text",
        "job_skills",
        "assistant_provisional_relevant",
        "assistant_label_source",
        "assistant_label_verified",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Evaluation dataset is empty.")

    if df.duplicated(["user_id", "job_id"]).any():
        raise ValueError("Duplicate user/job pairs found.")

    if not df["assistant_label_source"].eq(
        "assistant_provisional_review"
    ).all():
        raise ValueError("Unexpected relevance-label source.")

    verified = (
        df["assistant_label_verified"]
        .fillna(True)
        .astype(bool)
    )

    if verified.any():
        raise ValueError(
            "This evaluation expects unverified provisional labels."
        )

    # --------------------------------------------------
    # 2. Clean relevance labels
    # --------------------------------------------------

    df["assistant_provisional_relevant"] = pd.to_numeric(
        df["assistant_provisional_relevant"],
        errors="coerce",
    )

    df = df[
        df["assistant_provisional_relevant"].isin([0, 1])
    ].copy()

    if df.empty:
        raise ValueError("No valid binary relevance labels found.")

    # --------------------------------------------------
    # 3. Protect existing results
    # --------------------------------------------------

    output_files = [
        OUTPUT_DIR / "hybrid_ranked_recommendations.csv",
        OUTPUT_DIR / "hybrid_evaluation_metrics.csv",
        OUTPUT_DIR / "hybrid_metrics_by_profile.csv",
    ]

    existing = [
        str(path)
        for path in output_files
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Existing results will not be overwritten:\n"
            + "\n".join(existing)
        )

    # --------------------------------------------------
    # 4. Evaluate each profile
    # --------------------------------------------------

    ranked_records = []
    metric_records = []

    print("=" * 65)
    print("CAREERINTEL — HYBRID RECOMMENDATION EVALUATION")
    print("=" * 65)

    print(f"Valid labeled rows: {len(df)}")
    print(f"Profiles: {df['user_id'].nunique()}")
    print(f"Evaluation cutoff: {K}")
    print("\nStarting evaluation...")

    for user_id, group in df.groupby("user_id", sort=True):
        group = group.copy()

        # Ensure one consistent profile per user.
        if group["profile_text"].nunique() != 1:
            raise ValueError(
                f"Inconsistent profile text for {user_id}."
            )

        if group["profile_skills"].nunique() != 1:
            raise ValueError(
                f"Inconsistent profile skills for {user_id}."
            )

        profile_text = str(group["profile_text"].iloc[0])

        profile_skills = group["profile_skills"].iloc[0]

        # Combine job title, description, and available skills.
        job_text = (
            group["job_title"].fillna("").astype(str)
            + ". "
            + group["job_text"].fillna("").astype(str)
            + ". Skills: "
            + group["job_skills"].fillna("").astype(str)
        )

        jobs = []

        for (_, row), text in zip(
            group.iterrows(),
            job_text.tolist(),
        ):
            jobs.append({
                "job_id": str(row["job_id"]),
                "title": str(row["job_title"]),
                "job_text": text,
                "job_description": text,
                "normalized_skills": row["job_skills"],
            })

        # Create and fit the hybrid ranker.
        ranker = HybridRanker()
        ranker.fit(jobs)

        # Rank all labeled candidates for this profile.
        recommendations = ranker.rank(
            profile_text=profile_text,
            profile_skills=profile_skills,
            top_k=len(jobs),
        )

        score_lookup = {
            result["job_id"]: result
            for result in recommendations
        }

        group["job_id"] = group["job_id"].astype(str)

        group["hybrid_score"] = group["job_id"].map(
            lambda job_id: score_lookup[job_id]["hybrid_score"]
        )

        group["semantic_score"] = group["job_id"].map(
            lambda job_id: score_lookup[job_id]["semantic_score"]
        )

        group["skill_score"] = group["job_id"].map(
            lambda job_id: score_lookup[job_id]["skill_score"]
        )

        group["title_score"] = group["job_id"].map(
            lambda job_id: score_lookup[job_id]["title_score"]
        )

        group["matched_skills"] = group["job_id"].map(
            lambda job_id: ", ".join(
                score_lookup[job_id]["matched_skills"]
            )
        )

        # Establish deterministic ranking.
        group = group.sort_values(
            ["hybrid_score", "job_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

        group["rank"] = range(1, len(group) + 1)

        relevance = group[
            "assistant_provisional_relevant"
        ].to_numpy()

        total_relevant = int((relevance == 1).sum())

        # Store ranked candidate records.
        ranked_records.extend(
            group[
                [
                    "user_id",
                    "job_id",
                    "job_title",
                    "hybrid_score",
                    "semantic_score",
                    "skill_score",
                    "title_score",
                    "matched_skills",
                    "rank",
                    "assistant_provisional_relevant",
                    "assistant_label_source",
                ]
            ].to_dict("records")
        )

        # Calculate ranking metrics.
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
            f"Evaluated {user_id}: "
            f"{len(group)} candidates, "
            f"{total_relevant} relevant"
        )

    # --------------------------------------------------
    # 5. Summarize results
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 6. Save results without overwriting existing files
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents overwriting existing outputs.
    for output_file, records in [
        (output_files[0], pd.DataFrame(ranked_records)),
        (output_files[1], summary),
        (output_files[2], per_profile),
    ]:
        with output_file.open("x", encoding="utf-8", newline="") as file:
            records.to_csv(file, index=False)

    # --------------------------------------------------
    # 7. Display results
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("HYBRID EVALUATION SUMMARY")
    print("=" * 65)

    print(summary.to_string(index=False))

    print("\nPer-profile metrics:")
    print(per_profile.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_DIR.relative_to(ROOT)}")

    print(
        "\nLIMITATION: These metrics measure agreement with "
        "provisional assistant labels, not independently verified "
        "ground truth or real-world employment outcomes."
    )


if __name__ == "__main__":
    main()