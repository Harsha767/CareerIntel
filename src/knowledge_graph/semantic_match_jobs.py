from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CAREERINTEL - Semantic Job to O*NET Matching
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

JOBS_DIR = PROJECT_ROOT / "data" / "processed" / "jobs"
ONET_DIR = PROJECT_ROOT / "data" / "processed" / "onet"

UNMATCHED_FILE = JOBS_DIR / "jobs_still_unmatched.csv"
ONET_MAPPING_FILE = ONET_DIR / "job_title_occupation_mapping.csv"

SEMANTIC_OUTPUT = JOBS_DIR / "jobs_semantic_matches.csv"
FINAL_UNMATCHED_OUTPUT = (
    JOBS_DIR / "jobs_still_unmatched_after_semantic.csv"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Minimum similarity score required for a semantic match.
CONFIDENCE_THRESHOLD = 0.70

# Batch sizes keep memory usage reasonable.
JOB_BATCH_SIZE = 128
ONET_BATCH_SIZE = 256


def load_data():
    """Load unmatched jobs and O*NET occupation mappings."""

    print("=" * 70)
    print("CareerIntel - Semantic Job to O*NET Matching")
    print("=" * 70)

    print("\nLoading unmatched jobs...")

    if not UNMATCHED_FILE.exists():
        raise FileNotFoundError(
            f"Unmatched jobs file not found:\n{UNMATCHED_FILE}"
        )

    jobs = pd.read_csv(UNMATCHED_FILE)

    print(f"Unmatched jobs loaded: {len(jobs):,}")

    print("\nLoading O*NET title mapping...")

    if not ONET_MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"O*NET mapping file not found:\n{ONET_MAPPING_FILE}"
        )

    onet = pd.read_csv(ONET_MAPPING_FILE)

    print(f"O*NET title records loaded: {len(onet):,}")

    return jobs, onet


def prepare_titles(jobs, onet):
    """Validate and clean the title columns."""

    # --------------------------------------------------------
    # Job dataset
    # --------------------------------------------------------

    if "title" not in jobs.columns:
        raise ValueError(
            "'title' column not found in jobs dataset.\n"
            f"Available columns: {list(jobs.columns)}"
        )

    # --------------------------------------------------------
    # O*NET mapping
    #
    # Actual columns in our generated mapping file:
    #
    # onet_soc_code
    # occupation_title
    # job_title
    # short_title
    # --------------------------------------------------------

    required_onet_columns = {
        "onet_soc_code",
        "occupation_title",
    }

    missing = required_onet_columns - set(onet.columns)

    if missing:
        raise ValueError(
            f"Missing O*NET columns: {sorted(missing)}\n"
            f"Available columns: {list(onet.columns)}"
        )

    jobs = jobs.copy()
    onet = onet.copy()

    # Clean job titles.
    jobs["title"] = (
        jobs["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Clean O*NET occupation titles.
    onet["occupation_title"] = (
        onet["occupation_title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove empty titles.
    jobs = jobs[
        jobs["title"] != ""
    ].copy()

    onet = onet[
        onet["occupation_title"] != ""
    ].copy()

    # Remove duplicate occupation records.
    onet = onet.drop_duplicates(
        subset=[
            "onet_soc_code",
            "occupation_title",
        ]
    ).reset_index(drop=True)

    jobs = jobs.reset_index(drop=True)

    return jobs, onet


def encode_titles(model, titles, batch_size):
    """Generate normalized Transformer embeddings."""

    print(f"Encoding {len(titles):,} titles...")

    embeddings = model.encode(
        titles,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embeddings.astype(np.float32)


def semantic_match(jobs, onet, model):
    """Find the best semantic O*NET occupation for each job."""

    job_titles = jobs["title"].tolist()

    onet_titles = (
        onet["occupation_title"]
        .tolist()
    )

    print("\n" + "=" * 70)
    print("GENERATING EMBEDDINGS")
    print("=" * 70)

    # --------------------------------------------------------
    # Encode O*NET occupation titles once.
    # --------------------------------------------------------

    onet_embeddings = encode_titles(
        model,
        onet_titles,
        ONET_BATCH_SIZE,
    )

    print(
        f"O*NET embedding matrix shape: "
        f"{onet_embeddings.shape}"
    )

    # --------------------------------------------------------
    # Encode unmatched job titles.
    # --------------------------------------------------------

    job_embeddings = encode_titles(
        model,
        job_titles,
        JOB_BATCH_SIZE,
    )

    print(
        f"Job embedding matrix shape: "
        f"{job_embeddings.shape}"
    )

    print("\n" + "=" * 70)
    print("CALCULATING SEMANTIC MATCHES")
    print("=" * 70)

    best_indices = []
    best_scores = []

    # --------------------------------------------------------
    # Compare jobs against O*NET occupations in batches.
    # --------------------------------------------------------

    for start in range(
        0,
        len(job_embeddings),
        JOB_BATCH_SIZE,
    ):

        end = min(
            start + JOB_BATCH_SIZE,
            len(job_embeddings),
        )

        batch_embeddings = (
            job_embeddings[start:end]
        )

        similarities = cosine_similarity(
            batch_embeddings,
            onet_embeddings,
        )

        indices = np.argmax(
            similarities,
            axis=1,
        )

        scores = similarities[
            np.arange(len(indices)),
            indices,
        ]

        best_indices.extend(indices)
        best_scores.extend(scores)

        print(
            f"Processed {end:,} / "
            f"{len(job_embeddings):,} jobs"
        )

    best_indices = np.array(best_indices)
    best_scores = np.array(best_scores)

    # --------------------------------------------------------
    # Build result dataframe.
    # --------------------------------------------------------

    matches = jobs.copy()

    matches["onet_soc_code"] = [
        onet.iloc[index]["onet_soc_code"]
        for index in best_indices
    ]

    matches["onet_occupation_title"] = [
        onet.iloc[index]["occupation_title"]
        for index in best_indices
    ]

    matches["semantic_score"] = best_scores

    matches["semantic_match"] = (
        matches["semantic_score"]
        >= CONFIDENCE_THRESHOLD
    )

    return matches


def save_results(matches):
    """Save semantic matches and remaining unmatched jobs."""

    semantic_matches = matches[
        matches["semantic_match"]
    ].copy()

    still_unmatched = matches[
        ~matches["semantic_match"]
    ].copy()

    semantic_matches.to_csv(
        SEMANTIC_OUTPUT,
        index=False,
    )

    still_unmatched.to_csv(
        FINAL_UNMATCHED_OUTPUT,
        index=False,
    )

    return semantic_matches, still_unmatched


def print_results(
    matches,
    semantic_matches,
    still_unmatched,
):
    """Print semantic matching statistics."""

    print("\n" + "=" * 70)
    print("SEMANTIC MATCHING RESULTS")
    print("=" * 70)

    total = len(matches)
    matched = len(semantic_matches)
    unmatched = len(still_unmatched)

    print(
        f"Jobs processed:          {total:,}"
    )

    print(
        f"Semantic matches:        {matched:,}"
    )

    print(
        f"Still unmatched:         {unmatched:,}"
    )

    if total > 0:
        match_rate = (
            matched / total * 100
        )

        print(
            f"Semantic match rate:     "
            f"{match_rate:.2f}%"
        )

    # --------------------------------------------------------
    # Score statistics
    # --------------------------------------------------------

    if matched > 0:

        print("\nSCORE STATISTICS")
        print("-" * 70)

        print(
            semantic_matches[
                "semantic_score"
            ].describe()
        )

        # ----------------------------------------------------
        # Highest-confidence matches
        # ----------------------------------------------------

        print("\nTOP SEMANTIC MATCHES")
        print("-" * 70)

        display_columns = [
            "title",
            "onet_occupation_title",
            "semantic_score",
        ]

        print(
            semantic_matches[
                display_columns
            ]
            .sort_values(
                "semantic_score",
                ascending=False,
            )
            .head(20)
            .to_string(index=False)
        )

        # ----------------------------------------------------
        # Lowest accepted matches
        # ----------------------------------------------------

        print("\nLOWEST ACCEPTED MATCHES")
        print("-" * 70)

        print(
            semantic_matches[
                display_columns
            ]
            .sort_values(
                "semantic_score",
                ascending=True,
            )
            .head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Final output locations
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SEMANTIC MATCHING COMPLETE")
    print("=" * 70)

    print(
        "\nMatched output:"
    )

    print(
        SEMANTIC_OUTPUT
    )

    print(
        "\nRemaining unmatched:"
    )

    print(
        FINAL_UNMATCHED_OUTPUT
    )


def main():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    jobs, onet = load_data()

    # --------------------------------------------------------
    # Prepare titles
    # --------------------------------------------------------

    jobs, onet = prepare_titles(
        jobs,
        onet,
    )

    print(
        f"\nUnique O*NET occupations available: "
        f"{len(onet):,}"
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD}"
    )

    # --------------------------------------------------------
    # Load Transformer
    # --------------------------------------------------------

    print(
        "\nLoading Transformer model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Transformer model loaded successfully."
    )

    # --------------------------------------------------------
    # Semantic matching
    # --------------------------------------------------------

    matches = semantic_match(
        jobs,
        onet,
        model,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    semantic_matches, still_unmatched = (
        save_results(matches)
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_results(
        matches,
        semantic_matches,
        still_unmatched,
    )


if __name__ == "__main__":
    main()