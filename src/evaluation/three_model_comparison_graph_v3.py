
"""Compare semantic, traditional hybrid, and graph hybrid recommenders."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.evaluation.evaluate_recommendations import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.knowledge_graph.build_graph import clean_code
from src.knowledge_graph.semantic_match_jobs import (
    CONFIDENCE_THRESHOLD,
    prepare_titles,
    semantic_match,
)
from src.recommendation.semantic_recommender import (
    GRAPH_WEIGHT,
    MODEL,
    SEMANTIC_WEIGHT,
    SKILL_WEIGHT,
    calculate_graph_skill_matches,
    calculate_skill_matches,
    load_jobs,
    normalize_skill_set,
)

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data/processed/evaluation/"
    / "careerintel_assistant_provisional_review.csv"
)

ONET_FILE = (
    ROOT / "data/processed/onet/"
    / "job_title_occupation_mapping.csv"
)

GRAPH_SKILL_FILE = (
    ROOT / "data/processed/knowledge_graph/"
    / "occupation_skill_edges.csv"
)

# New folder: preserves both baseline and previous experimental results.
OUTPUT = ROOT / "results/three_model_comparison_graph_v3"

K = 10


def make_key(title, company):
    return (
        str(title).strip().casefold()
        + "|"
        + str(company).strip().casefold()
    )


def build_evaluation_occupation_mapping(candidate_jobs):
    """Match evaluation jobs to existing graph occupations."""

    if not ONET_FILE.exists():
        raise FileNotFoundError(ONET_FILE)

    if not GRAPH_SKILL_FILE.exists():
        raise FileNotFoundError(GRAPH_SKILL_FILE)

    matching_jobs = candidate_jobs[
      ["_key", "normalized_title"]
    ].copy()

    matching_jobs = matching_jobs.rename(
     columns={
        "_key": "evaluation_job_key",
        "normalized_title": "title",
    }
)

    matching_jobs = matching_jobs.drop_duplicates(
      subset=["evaluation_job_key"]
    ).reset_index(drop=True)

    onet = pd.read_csv(
        ONET_FILE,
        dtype={"onet_soc_code": str},
    )

    matching_jobs, onet = prepare_titles(
        matching_jobs,
        onet,
    )

    matches = semantic_match(
        matching_jobs,
        onet,
        MODEL,
    )

    matches["occupation_id"] = (
        matches["onet_soc_code"].apply(clean_code)
    )

    graph_occupations = set(
        pd.read_csv(
            GRAPH_SKILL_FILE,
            dtype={"occupation_id": str},
            usecols=["occupation_id"],
        )["occupation_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    matches["occupation_in_graph"] = (
        matches["occupation_id"].isin(graph_occupations)
    )

    accepted = matches[
        matches["semantic_match"].astype(bool)
        & matches["occupation_in_graph"]
    ].copy()

    mapping = dict(
        zip(
            accepted["evaluation_job_key"],
            accepted["occupation_id"],
        )
    )

    matches.to_csv(
        OUTPUT / "evaluation_job_occupation_matches.csv",
        index=False,
    )

    print("\nEVALUATION OCCUPATION MATCHING")
    print(f"Candidates processed: {len(matches)}")
    print(
        f"Semantic matches (threshold {CONFIDENCE_THRESHOLD}): "
        f"{int(matches['semantic_match'].sum())}"
    )
    print(
        "Matched occupations present in graph: "
        f"{int(matches['occupation_in_graph'].sum())}"
    )
    print(f"Accepted evaluation mappings: {len(mapping)}")

    return mapping


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    if not ONET_FILE.exists():
        raise FileNotFoundError(ONET_FILE)

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {OUTPUT}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    evaluation = pd.read_csv(INPUT)

    required = {
        "user_id",
        "profile_text",
        "profile_skills",
        "job_id",
        "job_title",
        "company_name",
        "assistant_provisional_relevant",
        "assistant_label_source",
        "assistant_label_verified",
    }

    missing = required - set(evaluation.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if evaluation.duplicated(["user_id", "job_id"]).any():
        raise ValueError("Duplicate user/job pairs found.")

    if not evaluation["assistant_label_source"].eq(
        "assistant_provisional_review"
    ).all():
        raise ValueError("Unexpected relevance-label source.")

    verified = evaluation[
        "assistant_label_verified"
    ].fillna(True).astype(bool)

    if verified.any():
        raise ValueError("Expected unverified provisional labels.")

    evaluation["assistant_provisional_relevant"] = pd.to_numeric(
        evaluation["assistant_provisional_relevant"],
        errors="coerce",
    )

    evaluation = evaluation[
        evaluation["assistant_provisional_relevant"].isin([0, 1])
    ].copy()

    jobs = load_jobs().reset_index(drop=True)

    # Match evaluation titles to production jobs.
    jobs["_match_key"] = [
        make_key(title, company)
        for title, company in zip(
            jobs["title"],
            jobs["company_name"],
        )
    ]

    evaluation["_match_key"] = [
        make_key(title, company)
        for title, company in zip(
            evaluation["job_title"],
            evaluation["company_name"],
        )
    ]

    counts = jobs["_match_key"].value_counts()

    evaluation["_match_count"] = (
        evaluation["_match_key"].map(counts).fillna(0)
    )

    evaluation = evaluation[
        evaluation["_match_count"] == 1
    ].copy()

    if evaluation.empty:
        raise ValueError("No uniquely matched evaluation candidates.")

    # Use the same key as the graph scorer:
    # normalized_title + company_name.
    jobs["_key"] = [
        make_key(title, company)
        for title, company in zip(
            jobs["normalized_title"],
            jobs["company_name"],
        )
    ]

    job_lookup = jobs.drop_duplicates(
        "_match_key"
    ).set_index("_match_key")

    evaluation = evaluation.join(
        job_lookup,
        on="_match_key",
        rsuffix="_production",
    )

    

    print(
        f"Valid, uniquely matched candidates: {len(evaluation)}"
    )
    print(
        f"Profiles: {evaluation['user_id'].nunique()}"
    )

    candidate_jobs = jobs[
        jobs["_key"].isin(evaluation["_key"])
    ].drop_duplicates("_key").copy()

    if candidate_jobs["_key"].duplicated().any():
        raise ValueError(
            "Duplicate normalized candidate keys found."
        )

    # Build the evaluation-specific mapping once.
    evaluation_job_occupations = (
        build_evaluation_occupation_mapping(candidate_jobs)
    )

    ranked_records = []
    metric_records = []

    for user_id, group in evaluation.groupby(
        "user_id",
        sort=True,
    ):
        group = group.copy()

        if group["profile_text"].nunique() != 1:
            raise ValueError(
                f"Inconsistent profile text: {user_id}"
            )

        if group["profile_skills"].nunique() != 1:
            raise ValueError(
                f"Inconsistent profile skills: {user_id}"
            )

        profile = str(group["profile_text"].iloc[0])

        profile_skills = normalize_skill_set(
            str(group["profile_skills"].iloc[0]).split(",")
        )

        profile_jobs = jobs.loc[
            jobs["_key"].isin(group["_key"])
        ].drop_duplicates("_key").copy()

        candidate_jobs = profile_jobs.set_index("_key").loc[
            group["_key"]
        ].reset_index()

        embeddings = MODEL.encode(
            candidate_jobs["search_text"].tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        profile_embedding = MODEL.encode(
            [profile],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        semantic = cosine_similarity(
            profile_embedding,
            embeddings,
        ).ravel()

        direct, direct_matches = calculate_skill_matches(
            candidate_jobs,
            profile_skills,
        )

        graph, graph_matches = calculate_graph_skill_matches(
            candidate_jobs,
            profile_skills,
            evaluation_job_occupations=evaluation_job_occupations,
        )

        scores = {
            "semantic_only": semantic,
            "traditional_hybrid": (
                0.80 * semantic + 0.20 * direct
            ),
            "graph_hybrid": (
                SEMANTIC_WEIGHT * semantic
                + SKILL_WEIGHT * direct
                + GRAPH_WEIGHT * graph
            ),
        }

        labels = group[
            "assistant_provisional_relevant"
        ].to_numpy(dtype=float)

        total_relevant = int(np.sum(labels == 1))

        for model_name, model_scores in scores.items():
            ranked = group.copy()

            ranked["semantic_score"] = semantic
            ranked["direct_skill_coverage"] = direct
            ranked["graph_skill_coverage"] = graph
            ranked["direct_matching_skills"] = direct_matches
            ranked["graph_matching_skills"] = graph_matches
            ranked["model_score"] = model_scores

            ranked = ranked.sort_values(
                ["model_score", "job_id"],
                ascending=[False, True],
            ).reset_index(drop=True)

            ranked["rank"] = np.arange(
                1,
                len(ranked) + 1,
            )

            relevance = ranked[
                "assistant_provisional_relevant"
            ].to_numpy(dtype=float)

            for _, row in ranked.iterrows():
                ranked_records.append({
                    "model": model_name,
                    "user_id": user_id,
                    "job_id": row["job_id"],
                    "job_title": row["job_title"],
                    "rank": row["rank"],
                    "model_score": row["model_score"],
                    "semantic_score": row["semantic_score"],
                    "direct_skill_coverage": (
                        row["direct_skill_coverage"]
                    ),
                    "graph_skill_coverage": (
                        row["graph_skill_coverage"]
                    ),
                    "direct_matching_skills": (
                        row["direct_matching_skills"]
                    ),
                    "graph_matching_skills": (
                        row["graph_matching_skills"]
                    ),
                    "relevance": (
                        row["assistant_provisional_relevant"]
                    ),
                })

            metric_records.append({
                "model": model_name,
                "user_id": user_id,
                "candidate_count": len(ranked),
                "relevant_candidates": total_relevant,
                "precision_at_10": precision_at_k(
                    relevance, K
                ),
                "recall_at_10": recall_at_k(
                    relevance, total_relevant, K
                ),
                "ndcg_at_10": ndcg_at_k(
                    relevance, K
                ),
                "mrr": reciprocal_rank(relevance),
                "label_type": "provisional_assistant",
            })

        print(f"Evaluated profile: {user_id}")

    per_profile = pd.DataFrame(metric_records)

    summary = (
        per_profile.groupby("model", sort=False)
        .agg(
            profiles_evaluated=("user_id", "nunique"),
            candidate_count=("candidate_count", "sum"),
            mean_precision_at_10=("precision_at_10", "mean"),
            mean_recall_at_10=("recall_at_10", "mean"),
            mean_ndcg_at_10=("ndcg_at_10", "mean"),
            mean_mrr=("mrr", "mean"),
        )
        .reset_index()
    )

    summary["label_type"] = "provisional_assistant"

    pd.DataFrame(ranked_records).to_csv(
        OUTPUT / "ranked_candidates.csv",
        index=False,
    )

    per_profile.to_csv(
        OUTPUT / "metrics_by_profile.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT / "model_comparison_summary.csv",
        index=False,
    )

    print("\nTHREE-MODEL COMPARISON")
    print(summary.to_string(index=False))

    print(
        "\nPositive graph coverage:",
        int(
            (
                pd.DataFrame(ranked_records)
                .query("model == 'graph_hybrid'")
                ["graph_skill_coverage"] > 0
            ).sum()
        ),
    )

    print(
        f"\nResults saved to: {OUTPUT.relative_to(ROOT)}"
    )

    print(
        "\nLIMITATION: Labels are provisional assistant-generated "
        "judgments, not independently verified ground truth."
    )


if __name__ == "__main__":
    main()
