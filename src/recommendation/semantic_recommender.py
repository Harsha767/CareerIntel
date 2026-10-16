"""
CareerIntel: Hybrid semantic job recommendation engine.

Features:
1. Transformer-based semantic similarity.
2. Robust skill parsing and normalization.
3. Explicit profile-to-job skill matching.
4. Hybrid recommendation scoring.
5. Cached job embeddings.
6. Duplicate recommendation removal.
7. Recommendation validation and CSV export.
"""

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "jobs"
    / "jobs_with_normalized_skills.csv"
)

OUTPUT_DIR = ROOT / "results"

CACHE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "embeddings"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Hybrid recommendation weights.
SEMANTIC_WEIGHT = 0.80
SKILL_WEIGHT = 0.20

# Initialize the transformer model.
MODEL = SentenceTransformer(MODEL_NAME)


# ============================================================
# 2. TEXT CLEANING
# ============================================================

def clean(value):
    """Convert missing or structured values into clean text."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    # Convert Python-style lists into readable text.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, (list, tuple, set)):
                text = " ".join(str(item) for item in parsed)

        except (ValueError, SyntaxError):
            pass

    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# 3. SKILL NORMALIZATION
# ============================================================

SKILL_ALIASES = {
    "py": "python",
    "structured query language": "sql",
    "scikit learn": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "ml": "machine learning",
    "data visualisation": "data visualization",
    "ms excel": "excel",
    "power bi": "power bi",
}


def normalize_skill(skill):
    """Normalize skill capitalization and common aliases."""

    text = clean(skill).casefold()

    text = re.sub(r"\s+", " ", text).strip()

    return SKILL_ALIASES.get(text, text)


def normalize_skill_set(skills):
    """Normalize an iterable of individual skills."""

    return {
        normalize_skill(skill)
        for skill in skills
        if normalize_skill(skill)
    }


# ============================================================
# 4. ROBUST SKILL PARSING
# ============================================================

def parse_skills(value, candidate_skills=None):
    """
    Parse skills from different dataset formats.

    Supported formats:
    - Python-style lists stored in CSV cells.
    - Comma-separated skills.
    - Semicolon-separated skills.
    - Pipe-separated skills.
    - Space-separated normalized skill strings.

    For space-separated text, candidate_skills allows matching
    complete skill phrases without splitting multiword skills.
    """

    text = clean(value)

    if not text:
        return set()

    # --------------------------------------------------------
    # Case 1: Skills stored as a Python-style list.
    # Example:
    # "['Python', 'Machine Learning', 'SQL']"
    # --------------------------------------------------------

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        return normalize_skill_set(parsed)

    # --------------------------------------------------------
    # Case 2: Delimited skill strings.
    # Example:
    # "Python, SQL, Machine Learning"
    # --------------------------------------------------------

    if re.search(r"[,;|]", text):
        items = re.split(r"[,;|]", text)
        return normalize_skill_set(items)

    # --------------------------------------------------------
    # Case 3: Space-separated normalized skill strings.
    #
    # Example:
    # "data visualization pandas python scikit-learn sql"
    #
    # Match complete candidate skill phrases against the text.
    # This preserves phrases such as "data visualization".
    # --------------------------------------------------------

    normalized_text = normalize_skill(text)

    if candidate_skills:
        matched = set()

        normalized_candidates = normalize_skill_set(
            candidate_skills
        )

        for skill in normalized_candidates:
            # Match complete phrases, not partial words.
            pattern = (
                r"(?<!\w)"
                + re.escape(skill)
                + r"(?!\w)"
            )

            if re.search(pattern, normalized_text):
                matched.add(skill)

        return matched

    # Without a candidate vocabulary, preserve the text as
    # one phrase rather than pretending each word is a skill.
    return {normalized_text}


# ============================================================
# 5. LOAD AND PREPARE JOB DATA
# ============================================================

def load_jobs():
    """Load, clean, and prepare job records."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"\nJob dataset not found:\n{DATA_FILE}\n\n"
            "Make sure jobs_with_normalized_skills.csv "
            "exists inside data/processed/jobs/."
        )

    print(f"Loading jobs from: {DATA_FILE}")

    jobs = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )

    required_columns = {"title", "job_text"}

    missing_columns = required_columns - set(jobs.columns)

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    jobs = jobs.copy()

    # --------------------------------------------------------
    # Clean relevant columns.
    # --------------------------------------------------------

    text_columns = [
        "title",
        "job_text",
        "normalized_skills",
        "parsed_skills",
        "skills_required",
        "industry",
        "function",
        "experience_level",
        "employment_type",
        "work_model",
        "education_level",
    ]

    for column in text_columns:
        if column not in jobs.columns:
            jobs[column] = ""

        jobs[column] = jobs[column].map(clean)

    # --------------------------------------------------------
    # Select the best available skill source.
    # --------------------------------------------------------

    jobs["normalized_skills"] = jobs.apply(
        lambda row: (
            row["normalized_skills"]
            or row["parsed_skills"]
            or row["skills_required"]
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Build searchable job text.
    # --------------------------------------------------------

    searchable_columns = [
        "title",
        "job_text",
        "normalized_skills",
        "industry",
        "function",
        "experience_level",
        "employment_type",
        "work_model",
        "education_level",
    ]

    jobs["search_text"] = (
        jobs[searchable_columns]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Remove records without searchable text.
    jobs = jobs[
        jobs["search_text"].ne("")
    ].reset_index(drop=True)

    if jobs.empty:
        raise ValueError(
            "No searchable jobs remain after preprocessing."
        )

    print(f"Searchable jobs loaded: {len(jobs):,}")

    return jobs


# ============================================================
# 6. JOB EMBEDDINGS AND CACHE
# ============================================================

def get_job_embeddings(jobs):
    """
    Load cached transformer embeddings when possible.

    Recompute embeddings if the job text has changed.
    """

    embedding_file = CACHE_DIR / "job_embeddings.npy"

    text_file = CACHE_DIR / "job_embedding_texts.csv"

    current_texts = jobs["search_text"].tolist()

    # --------------------------------------------------------
    # Check whether the cache is valid.
    # --------------------------------------------------------

    if embedding_file.exists() and text_file.exists():

        try:
            cached_texts = (
                pd.read_csv(text_file)["search_text"]
                .fillna("")
                .tolist()
            )

            if cached_texts == current_texts:

                embeddings = np.load(embedding_file)

                if embeddings.shape[0] == len(jobs):

                    print("Using cached job embeddings.")

                    return embeddings

        except (
            OSError,
            ValueError,
            KeyError,
        ):
            print("Existing embedding cache is invalid.")

    # --------------------------------------------------------
    # Generate fresh embeddings.
    # --------------------------------------------------------

    print(
        f"Generating embeddings for {len(jobs):,} jobs..."
    )

    print("The first run may take several minutes.")

    embeddings = MODEL.encode(
        current_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Save embeddings and their corresponding source text.
    np.save(
        embedding_file,
        embeddings,
    )

    pd.DataFrame(
        {"search_text": current_texts}
    ).to_csv(
        text_file,
        index=False,
    )

    print(f"Embeddings cached at: {embedding_file}")

    return embeddings


# ============================================================
# 7. SKILL MATCHING
# ============================================================

def calculate_skill_matches(
    jobs,
    user_skills,
):
    """
    Calculate profile skill coverage for every job.

    Coverage =
        Number of matching profile skills
        ---------------------------------
        Total number of profile skills

    Example:
        Profile skills: Python, SQL, Pandas, Statistics

        Job skills: Python, SQL

        Coverage = 2 / 4 = 0.50
    """

    coverage_scores = []
    matching_skills_list = []

    normalized_user_skills = normalize_skill_set(
        user_skills
    )

    for _, row in jobs.iterrows():

        skill_text = row["normalized_skills"]

        # Pass the profile vocabulary so the parser can
        # detect complete phrases in space-separated text.
        job_skills = parse_skills(
            skill_text,
            candidate_skills=normalized_user_skills,
        )

        matching_skills = (
            normalized_user_skills.intersection(
                job_skills
            )
        )

        if normalized_user_skills:
            coverage = (
                len(matching_skills)
                / len(normalized_user_skills)
            )
        else:
            coverage = 0.0

        coverage_scores.append(coverage)

        matching_skills_list.append(
            ", ".join(sorted(matching_skills))
        )

    return (
        np.asarray(coverage_scores, dtype=float),
        matching_skills_list,
    )


# ============================================================
# 8. HYBRID JOB RECOMMENDATION
# ============================================================

def recommend_jobs(
    profile,
    preferred_skills=None,
    top_k=10,
    min_similarity=0.0,
):
    """
    Recommend jobs using transformer similarity and skill coverage.

    Hybrid score:
        80% semantic similarity
        20% profile skill coverage

    If no preferred skills are supplied, ranking uses
    semantic similarity alone.
    """

    # --------------------------------------------------------
    # Validate inputs.
    # --------------------------------------------------------

    if not isinstance(profile, str) or not profile.strip():
        raise ValueError(
            "Provide a non-empty career profile."
        )

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
    ):
        raise ValueError(
            "top_k must be a positive integer."
        )

    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError(
            "min_similarity must be between 0 and 1."
        )

    # --------------------------------------------------------
    # Load jobs.
    # --------------------------------------------------------

    jobs = load_jobs()

    # Normalize the profile skills.
    user_skills = normalize_skill_set(
        preferred_skills or []
    )

    print("\nProfile skills:")
    print(sorted(user_skills))

    # --------------------------------------------------------
    # Generate/load job embeddings.
    # --------------------------------------------------------

    job_embeddings = get_job_embeddings(jobs)

    # --------------------------------------------------------
    # Encode the user profile.
    # --------------------------------------------------------

    profile_embedding = MODEL.encode(
        [profile.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # --------------------------------------------------------
    # Calculate semantic similarity.
    # --------------------------------------------------------

    semantic_scores = cosine_similarity(
        profile_embedding,
        job_embeddings,
    ).ravel()

    # --------------------------------------------------------
    # Calculate skill coverage and matching skill names.
    # --------------------------------------------------------

    (
        skill_scores,
        matching_skills_list,
    ) = calculate_skill_matches(
        jobs,
        user_skills,
    )

    # --------------------------------------------------------
    # Calculate final hybrid scores.
    # --------------------------------------------------------

    if user_skills:

        recommendation_scores = (
            SEMANTIC_WEIGHT * semantic_scores
            + SKILL_WEIGHT * skill_scores
        )

    else:

        recommendation_scores = semantic_scores

    # --------------------------------------------------------
    # Construct results.
    # --------------------------------------------------------

    results = jobs.copy()

    results["semantic_score"] = semantic_scores

    results["skill_coverage"] = skill_scores

    results["matching_skills"] = matching_skills_list

    results["recommendation_score"] = (
        recommendation_scores
    )

    # Filter by semantic similarity.
    results = results[
        results["semantic_score"] >= min_similarity
    ].copy()

    # Rank the strongest recommendations first.
    results = results.sort_values(
        [
            "recommendation_score",
            "semantic_score",
        ],
        ascending=False,
    )

    # --------------------------------------------------------
    # Remove duplicate title/company combinations.
    # --------------------------------------------------------

    results["_title_key"] = (
        results["title"]
        .map(clean)
        .str.casefold()
        .str.strip()
    )

    if "company_name" in results.columns:

        results["_company_key"] = (
            results["company_name"]
            .map(clean)
            .str.casefold()
            .str.strip()
        )

        results = results.drop_duplicates(
            subset=[
                "_title_key",
                "_company_key",
            ],
            keep="first",
        )

    else:

        results = results.drop_duplicates(
            subset=["_title_key"],
            keep="first",
        )

    # Select the requested number of recommendations.
    results = results.head(top_k)

    # --------------------------------------------------------
    # Select output columns.
    # --------------------------------------------------------

    output_columns = [
        "title",
        "company_name",
        "industry",
        "function",
        "experience_level",
        "work_model",
        "employment_type",
        "normalized_skills",
        "matching_skills",
        "semantic_score",
        "skill_coverage",
        "recommendation_score",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in results.columns
    ]

    return results[output_columns].reset_index(drop=True)


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("CAREERINTEL | HYBRID SEMANTIC JOB RECOMMENDATION ENGINE")
    print("=" * 75)

    # Example candidate profile.
    profile = (
        "Aspiring Data Scientist with knowledge of Python, SQL, "
        "machine learning, statistics, data analysis, pandas, "
        "and data visualization. Interested in predictive "
        "modeling and business intelligence."
    )

    skills = [
        "Python",
        "SQL",
        "Machine Learning",
        "Statistics",
        "Data Analysis",
        "Pandas",
        "Data Visualization",
    ]

    # --------------------------------------------------------
    # Run recommendations.
    # --------------------------------------------------------

    recommendations = recommend_jobs(
        profile=profile,
        preferred_skills=skills,
        top_k=10,
        min_similarity=0.0,
    )

    # --------------------------------------------------------
    # Save recommendations.
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR / "semantic_recommendations.csv"
    )

    recommendations.to_csv(
        output_file,
        index=False,
    )

    print("\n" + "=" * 75)
    print("RECOMMENDATION RESULTS")
    print("=" * 75)

    print(
        f"Recommendations generated: {len(recommendations)}"
    )

    print(
        f"Saved to: {output_file.relative_to(ROOT)}"
    )

    # --------------------------------------------------------
    # Display top recommendations.
    # --------------------------------------------------------

    if not recommendations.empty:

        print("\n=== TOP SEMANTIC RECOMMENDATIONS ===")

        display_columns = [
            "title",
            "company_name",
            "matching_skills",
            "semantic_score",
            "skill_coverage",
            "recommendation_score",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in recommendations.columns
        ]

        print(
            recommendations[display_columns]
            .to_string(index=False)
        )

        # ----------------------------------------------------
        # Validation metrics.
        # ----------------------------------------------------

        print("\n" + "=" * 75)
        print("SKILL MATCH VALIDATION")
        print("=" * 75)

        positive_coverage = int(
            (
                recommendations["skill_coverage"] > 0
            ).sum()
        )

        print(
            "Recommendations with positive skill coverage:",
            f"{positive_coverage}/{len(recommendations)}",
        )

        print(
            "Mean semantic similarity:",
            round(
                recommendations["semantic_score"].mean(),
                4,
            ),
        )

        print(
            "Mean skill coverage:",
            round(
                recommendations["skill_coverage"].mean(),
                4,
            ),
        )

        print(
            "Mean recommendation score:",
            round(
                recommendations[
                    "recommendation_score"
                ].mean(),
                4,
            ),
        )

        # ----------------------------------------------------
        # Show jobs with actual profile skill matches.
        # ----------------------------------------------------

        matched_jobs = recommendations[
            recommendations["skill_coverage"] > 0
        ]

        print("\n=== JOBS WITH PROFILE SKILL MATCHES ===")

        if matched_jobs.empty:

            print(
                "No recommended jobs matched the supplied "
                "profile skills. Inspect the dataset's skill "
                "values and matching logic."
            )

        else:

            for _, row in matched_jobs.iterrows():

                print(f"\nJob: {row['title']}")

                if "company_name" in row.index:
                    print(
                        f"Company: {row['company_name']}"
                    )

                print(
                    f"Matching skills: {row['matching_skills']}"
                )

                print(
                    f"Skill coverage: "
                    f"{row['skill_coverage']:.2%}"
                )

    else:

        print(
            "No recommendations were returned. "
            "Check the input dataset and similarity threshold."
        )

    print("\nExecution completed.")