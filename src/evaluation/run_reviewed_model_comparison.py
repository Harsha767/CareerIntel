"""CareerIntel: Compare recommendation models using reviewed labels.

Models:
1. TF-IDF baseline
2. Transformer semantic ranker
3. Hybrid ranker

Important:
The relevance labels are preliminary AI-assisted annotations.
Results are agreement with those labels, not verified real-world accuracy.
Existing results are never overwritten.
"""

from pathlib import Path

import pandas as pd

from careerintel.models.baselines import TfidfRanker
from careerintel.models.semantic_ranker import SemanticRanker
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
    / "independent_relevance_review_completed_preliminary.csv"
)

OUTPUT_DIR = ROOT / "results" / "reviewed_model_comparison"
K = 10


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {
        "user_id",
        "profile_text",
        "profile_skills",
        "job_id",
        "job_title",
        "job_text",
        "job_skills",
        "reviewer_relevant",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Evaluation dataset is empty.")

    if df.duplicated(["user_id", "job_id"]).any():
        raise ValueError("Duplicate user/job pairs found.")

    df["reviewer_relevant"] = pd.to_numeric(
        df["reviewer_relevant"], errors="coerce"
    )

    # Exclude unreviewed or invalid labels.
    df = df[df["reviewer_relevant"].isin([0, 1])].copy()

    if df.empty:
        raise ValueError("No valid relevance labels found.")

    df["job_id"] = df["job_id"].astype(str)
    df["user_id"] = df["user_id"].astype(str)

    outputs = [
        OUTPUT_DIR / "model_comparison_metrics.csv",
        OUTPUT_DIR / "metrics_by_profile.csv",
        OUTPUT_DIR / "ranked_recommendations.csv",
    ]

    existing = [str(path) for path in outputs if path.exists()]

    if existing:
        raise FileExistsError(
            "Existing results will not be overwritten:\n"
            + "\n".join(existing)
        )

    ranked_records = []
    profile_metrics = []

    for user_id, group in df.groupby("user_id", sort=True):
        group = group.copy()

        profile_texts = group["profile_text"].map(clean_text).unique()

        if len(profile_texts) != 1:
            raise ValueError(
                f"Inconsistent profile text for user {user_id}"
            )

        profile_text = profile_texts[0]

        if not profile_text:
            raise ValueError(f"Empty profile text for user {user_id}")

        profile_skills = clean_text(
            group["profile_skills"].iloc[0]
        )

        job_texts = (
            group["job_title"].map(clean_text)
            + ". "
            + group["job_text"].map(clean_text)
            + ". Skills: "
            + group["job_skills"].map(clean_text)
        ).tolist()

        job_ids = group["job_id"].tolist()

        models = {
            "TF-IDF": TfidfRanker(),
            "Semantic": SemanticRanker(),
            "Hybrid": HybridRanker(),
        }

        model_results = {}

        for model_name, ranker in models.items():
            if model_name == "Hybrid":
                jobs = []

                for _, row in group.iterrows():
                    jobs.append({
                        "job_id": str(row["job_id"]),
                        "title": clean_text(row["job_title"]),
                        "job_text": clean_text(row["job_text"]),
                        "normalized_skills": clean_text(
                            row["job_skills"]
                        ),
                    })

                ranker.fit(jobs)

                ranking = ranker.rank(
                    profile_text=profile_text,
                    profile_skills=profile_skills.split(","),
                    top_k=len(group),
                )

                scores = {
                    str(item["job_id"]): item["hybrid_score"]
                    for item in ranking
                }

            else:
                ranker.fit(job_ids, job_texts)

                ranking = ranker.rank(
                    profile_text if model_name == "Semantic"
                    else profile_text,
                    top_k=len(group),
                )

                scores = {
                    str(job_id): score
                    for job_id, score in ranking
                }

            model_results[model_name] = scores

        relevance_by_job = dict(
            zip(group["job_id"], group["reviewer_relevant"])
        )

        for model_name, scores in model_results.items():
            ranked_jobs = sorted(
                scores.items(),
                key=lambda item: (-item[1], item[0]),
            )

            ranked_relevance = [
                int(relevance_by_job[job_id])
                for job_id, _ in ranked_jobs
            ]

            total_relevant = int(
                group["reviewer_relevant"].sum()
            )

            profile_metrics.append({
                "user_id": user_id,
                "model": model_name,
                "candidate_count": len(group),
                "relevant_candidates": total_relevant,
                f"precision_at_{K}": precision_at_k(
                    ranked_relevance, K
                ),
                f"recall_at_{K}": recall_at_k(
                    ranked_relevance, total_relevant, K
                ),
                f"ndcg_at_{K}": ndcg_at_k(
                    ranked_relevance, K
                ),
                "mrr": reciprocal_rank(ranked_relevance),
                "label_type": "preliminary_ai_assisted_review",
            })

            for rank, (job_id, score) in enumerate(
                ranked_jobs, start=1
            ):
                row = group[group["job_id"] == job_id].iloc[0]

                ranked_records.append({
                    "user_id": user_id,
                    "model": model_name,
                    "rank": rank,
                    "job_id": job_id,
                    "job_title": row["job_title"],
                    "score": score,
                    "reviewer_relevant": int(
                        relevance_by_job[job_id]
                    ),
                    "label_type": "preliminary_ai_assisted_review",
                })

        print(
            f"Evaluated profile {user_id}: "
            f"{len(group)} candidates across 3 models"
        )

    per_profile = pd.DataFrame(profile_metrics)

    metric_columns = [
        f"precision_at_{K}",
        f"recall_at_{K}",
        f"ndcg_at_{K}",
        "mrr",
    ]

    summary = (
        per_profile.groupby("model")[metric_columns]
        .mean()
        .reset_index()
    )

    summary.insert(
        1, "profiles_evaluated",
        per_profile.groupby("model")["user_id"].nunique().values
    )

    summary["candidate_count"] = len(df)
    summary["label_type"] = "preliminary_ai_assisted_review"

    # Do not overwrite existing files.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if any(path.exists() for path in outputs):
        raise FileExistsError(
            "An output file already exists. Nothing was overwritten."
        )

    summary.to_csv(outputs[0], index=False)
    per_profile.to_csv(outputs[1], index=False)
    pd.DataFrame(ranked_records).to_csv(outputs[2], index=False)

    print("\n" + "=" * 65)
    print("CAREERINTEL — REVIEWED MODEL COMPARISON")
    print("=" * 65)

    print(summary.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_DIR.relative_to(ROOT)}")
    print(
        "\nLIMITATION: These metrics measure agreement with preliminary "
        "AI-assisted labels, not independently verified ground truth."
    )


if __name__ == "__main__":
    main()