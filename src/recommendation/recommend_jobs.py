"""
CareerIntel - Hybrid Job Recommendation Engine

Data Scientist recommendations:
- Data Scientist
- Data Analyst
- Machine Learning Engineer
- AI Engineer
- Business Intelligence Analyst
- Relevant Analytics and Research roles

Includes experience filtering, skill matching, TF-IDF similarity,
role relevance, duplicate removal and hybrid ranking.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "data" / "processed" / "jobs"
    / "jobs_with_normalized_skills.csv"
)

OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MINIMUM_ROLE_MATCH = 0.55
REQUIRE_SKILL_MATCH = True

CONTENT_WEIGHT = 0.20
SKILL_WEIGHT = 0.30
REQUIREMENT_WEIGHT = 0.15
ROLE_WEIGHT = 0.30
EXPERIENCE_WEIGHT = 0.05


# =========================================================
# 2. TEXT AND SKILL UTILITIES
# =========================================================

def clean(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_skill(skill: Any) -> str:
    value = clean(skill).casefold()

    aliases = {
        "scikit learn": "scikit-learn",
        "sklearn": "scikit-learn",
        "py": "python",
        "structured query language": "sql",
        "ml": "machine learning",
        "data visualisation": "data visualization",
        "powerbi": "power bi",
        "ms excel": "excel",
    }

    return aliases.get(value, value)


def parse_skills(value: Any) -> set[str]:
    if value is None:
        return set()

    try:
        if pd.isna(value):
            return set()
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text:
        return set()

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        items = parsed
    elif isinstance(parsed, str):
        items = [parsed]
    else:
        items = re.split(r"[,;|]", text)

    return {
        normalize_skill(item)
        for item in items
        if normalize_skill(item)
    }


def extract_skills_from_text(
    value: Any,
    vocabulary: set[str],
) -> set[str]:

    text = clean(value).casefold()
    found = set()

    for skill in sorted(vocabulary, key=len, reverse=True):
        skill = normalize_skill(skill)

        if skill and re.search(
            r"(?<![a-z0-9])"
            + re.escape(skill)
            + r"(?![a-z0-9])",
            text,
        ):
            found.add(skill)

    return found


def contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def count_evidence(text: str, phrases: list[str]) -> int:
    normalized = clean(text).casefold()

    return sum(
        phrase in normalized
        for phrase in phrases
    )


# =========================================================
# 3. EXPERIENCE MATCHING
# =========================================================

def classify_experience(value: Any) -> str:
    text = clean(value).casefold()

    if not text:
        return "unknown"

    if any(
        word in text
        for word in (
            "fresher", "entry", "intern", "junior",
            "graduate", "trainee", "co-op", "coop",
            "working student",
        )
    ):
        return "entry"

    if any(
        word in text
        for word in ("mid", "intermediate", "associate")
    ):
        return "mid"

    if any(
        word in text
        for word in (
            "senior", "lead", "manager", "director",
            "executive", "principal", "head", "chief",
            "vice president",
        )
    ):
        return "senior"

    return "unknown"


def calculate_experience_match(
    candidate_level: str,
    job_level: Any,
) -> float:

    candidate = classify_experience(candidate_level)
    job = classify_experience(job_level)

    if "unknown" in (candidate, job):
        return 0.5

    if candidate == job:
        return 1.0

    compatibility = {
        "entry": {"mid": 0.25, "senior": 0.0},
        "mid": {"entry": 0.5, "senior": 0.5},
        "senior": {"mid": 0.5, "entry": 0.0},
    }

    return compatibility.get(candidate, {}).get(job, 0.5)


def filter_by_experience(
    jobs: pd.DataFrame,
    candidate_experience: str,
) -> pd.DataFrame:

    if classify_experience(candidate_experience) != "entry":
        return jobs.copy()

    levels = jobs["experience_level"].fillna("").map(
        classify_experience
    )

    titles = jobs["title"].fillna("").astype(str)

    entry_title = titles.str.contains(
        r"\b(?:fresher|entry[\s-]?level|junior|jr\.?|"
        r"intern|internship|graduate|trainee|co[ -]?op|"
        r"working student)\b",
        case=False,
        regex=True,
    )

    senior_title = titles.str.contains(
        r"\b(?:senior|sr\.?|lead|manager|director|"
        r"executive|principal|head|chief|vp|"
        r"vice president)\b",
        case=False,
        regex=True,
    )

    eligible = (
        (levels == "entry")
        | ((levels == "unknown") & entry_title)
    ) & ~senior_title

    return jobs.loc[eligible].copy()


# =========================================================
# 4. ROLE RELEVANCE
# =========================================================

ROLE_FAMILIES = {
    "data_scientist": {
        "direct": [
            "data scientist",
            "data science intern",
            "data science trainee",
            "junior data scientist",
            "associate data scientist",
            "applied data scientist",
            "data science researcher",
        ],
        "machine_learning": [
            "machine learning engineer",
            "junior machine learning engineer",
            "associate machine learning engineer",
            "machine learning scientist",
            "ml engineer",
            "deep learning engineer",
            "applied machine learning engineer",
        ],
        "artificial_intelligence": [
            "ai engineer",
            "junior ai engineer",
            "associate ai engineer",
            "artificial intelligence engineer",
            "applied ai engineer",
        ],
        "data_analyst": [
            "data analyst",
            "junior data analyst",
            "business data analyst",
            "business intelligence analyst",
            "bi analyst",
            "analytics analyst",
            "reporting analyst",
            "insights analyst",
        ],
        "other_related": [
            "analytics engineer",
            "statistical analyst",
            "research analyst",
            "data engineer",
            "applied scientist",
            "decision scientist",
            "quantitative analyst",
            "quant analyst",
            "research scientist",
            "analytics consultant",
            "product analyst",
            "business analyst",
            "market analyst",
        ],
    },

    "data_analyst": {
        "direct": [
            "data analyst",
            "junior data analyst",
            "business data analyst",
            "associate data analyst",
        ],
        "machine_learning": [
            "machine learning engineer",
            "ml engineer",
        ],
        "artificial_intelligence": [
            "ai engineer",
            "artificial intelligence engineer",
        ],
        "data_analyst": [
            "business intelligence analyst",
            "reporting analyst",
            "analytics analyst",
            "insights analyst",
            "product analyst",
            "business analyst",
            "market analyst",
        ],
        "other_related": [
            "data scientist",
            "analytics engineer",
            "quantitative analyst",
            "research analyst",
        ],
    },
}


DS_STRONG_EVIDENCE = [
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "ai models",
    "ml models",
    "predictive modeling",
    "predictive modelling",
    "predictive analytics",
    "data science",
    "computer vision",
    "natural language processing",
    "nlp",
    "statistical modeling",
    "statistical modelling",
    "statistical analysis",
    "optimization algorithm",
    "feature engineering",
    "model training",
    "model deployment",
    "data mining",
    "experimental design",
    "research analysis",
]


DS_TECHNICAL_EVIDENCE = [
    "python",
    "sql",
    "r programming",
    "statistics",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "regression",
    "classification",
    "clustering",
    "forecasting",
    "data cleaning",
    "data processing",
    "data modeling",
    "data modelling",
    "data pipeline",
    "data visualization",
    "tableau",
    "power bi",
    "snowflake",
    "spark",
    "computer vision",
    "nlp",
    "llm",
    "generative ai",
]


DS_EXCLUDED_TITLE_PHRASES = [
    "talent management",
    "human resources",
    "hr operations",
    "recruitment",
    "talent acquisition",
    "procurement",
    "purchasing",
    "sourcing",
    "supply chain",
    "vendor management",
    "merchandising",
    "category management",
    "payroll",
    "accounts payable",
    "accounts receivable",
    "revenue management",
    "financial analyst",
    "content creator",
    "compliance analyst",
    "reservoir engineering",
    "field operations",
    "process engineering",
    "avionics",
    "operations analyst",
    "achats",
    "procurement analyst",
    "purchasing analyst",
    "procurement analytics",
    "purchasing analytics",
    "process automation",
    "workflow automation",
]


UNRELATED_ENGINEERING_TITLE = re.compile(
    r"\b(?:software|electrical|mechanical|civil|"
    r"computer|industrial)\s+"
    r"(?:engineering|engineer|development|developer)\b",
    re.IGNORECASE,
)


def get_role_family(target_role: str) -> str | None:
    role = clean(target_role).casefold()

    if "data scientist" in role or "data science" in role:
        return "data_scientist"

    if "data analyst" in role:
        return "data_analyst"

    return None


def calculate_role_relevance(
    target_role: str,
    job_title: str,
    normalized_title: str = "",
    job_context: str = "",
) -> float:
    """
    Score how closely a job fits the candidate's target role.

    Title matches receive the strongest scores. Broader role titles are
    accepted only when their descriptions contain meaningful data/technical
    evidence, reducing false positives from generic job titles.
    """

    target = clean(target_role).casefold()

    raw_title = (
        f"{clean(job_title)} {clean(normalized_title)}"
    ).casefold()

    title = re.sub(
        r"[^a-z0-9]+",
        " ",
        raw_title,
    ).strip()

    context = clean(job_context).casefold()

    if not target:
        return 0.0

    family = get_role_family(target)

    if family is None:
        target_tokens = set(
            re.findall(r"[a-z0-9]+", target)
        )

        title_tokens = set(
            re.findall(r"[a-z0-9]+", title)
        )

        if not target_tokens:
            return 0.0

        return (
            len(target_tokens & title_tokens)
            / len(target_tokens)
        )

    categories = ROLE_FAMILIES[family]

    # Hard exclusions are limited to clearly unrelated job functions.
    if (
        family == "data_scientist"
        and contains_any(title, DS_EXCLUDED_TITLE_PHRASES)
    ):
        return 0.0

    if UNRELATED_ENGINEERING_TITLE.search(title):
        return 0.0

    # Direct role matches
    if contains_any(title, categories["direct"]):
        return 1.0

    # Machine Learning roles
    if contains_any(title, categories["machine_learning"]):
        return 0.88

    # Artificial Intelligence roles
    if contains_any(
        title,
        categories["artificial_intelligence"],
    ):
        return 0.84

    # Data Analyst / BI roles
    if contains_any(title, categories["data_analyst"]):
        return (
            0.80
            if family == "data_scientist"
            else 0.85
        )

    strong_hits = count_evidence(
        context,
        DS_STRONG_EVIDENCE,
    )

    technical_hits = count_evidence(
        context,
        DS_TECHNICAL_EVIDENCE,
    )

    # Related science, research, analytics and engineering titles
    # must demonstrate relevant technical work.
    if contains_any(title, categories["other_related"]):

        if strong_hits >= 1 and technical_hits >= 2:
            return 0.72

        if technical_hits >= 4:
            return 0.64

        return 0.0

    # AI/ML-adjacent titles need evidence.
    ai_ml_title = re.search(
        r"\b(ai|artificial intelligence|ml|machine learning|"
        r"deep learning|applied scientist)\b",
        title,
    )

    if ai_ml_title:

        valid_ai_title = re.search(
            r"\b(ai|artificial intelligence|machine learning|ml|"
            r"deep learning)\b.*\b(engineer|scientist|researcher|"
            r"intern|trainee|graduate)\b",
            title,
        )

        if (
            valid_ai_title
            and strong_hits >= 1
            and technical_hits >= 2
        ):
            return 0.82

        return 0.0

    # Generic early-career titles need explicit data-related evidence.
    early_career = bool(
        re.search(
            r"\b(intern|internship|co[ -]?op|working student|"
            r"trainee|research|graduate)\b",
            title,
        )
    )

    if early_career:

        explicit_data_evidence = [
            "data science",
            "data scientist",
            "data analytics",
            "data analyst",
            "business intelligence",
            "machine learning",
            "artificial intelligence",
            "predictive modeling",
            "predictive modelling",
            "statistical analysis",
            "data mining",
            "deep learning",
            "data modeling",
            "data modelling",
            "data analysis",
            "analytics",
        ]

        explicit_hits = count_evidence(
            context,
            explicit_data_evidence,
        )

        if explicit_hits >= 2 and technical_hits >= 3:
            return 0.68

        if (
            explicit_hits >= 1
            and strong_hits >= 2
            and technical_hits >= 4
        ):
            return 0.62

    return 0.0


def filter_by_target_role(
    jobs: pd.DataFrame,
    target_role: str,
    minimum_role_match: float = MINIMUM_ROLE_MATCH,
) -> pd.DataFrame:

    if not target_role:
        return jobs.copy()

    context_columns = [
        "job_description",
        "description",
        "responsibilities",
        "minimum_qualifications",
        "preferred_qualifications",
        "normalized_skills",
    ]

    def build_context(row: pd.Series) -> str:
        return " ".join(
            clean(row.get(column, ""))
            for column in context_columns
        )

    filtered = jobs.copy()

    filtered["role_match"] = filtered.apply(
        lambda row: calculate_role_relevance(
            target_role,
            row.get("title", ""),
            row.get("normalized_title", ""),
            build_context(row),
        ),
        axis=1,
    )

    return filtered.loc[
        filtered["role_match"] >= minimum_role_match
    ].reset_index(drop=True)


# =========================================================
# 5. LOAD AND PREPARE JOB DATA
# =========================================================

def load_jobs() -> pd.DataFrame:

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Job dataset not found:\n{DATA_FILE}"
        )

    jobs = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    ).copy()

    if "title" not in jobs.columns:

        if "normalized_title" in jobs.columns:
            jobs["title"] = jobs["normalized_title"]

        else:
            raise ValueError(
                "Dataset must contain title or normalized_title."
            )

    optional_columns = [
        "normalized_title",
        "company_name",
        "industry",
        "function",
        "experience_level",
        "employment_type",
        "work_model",
        "education_level",
        "normalized_skills",
        "parsed_skills",
        "skills_required",
        "job_description",
        "description",
        "responsibilities",
        "minimum_qualifications",
        "preferred_qualifications",
    ]

    for column in optional_columns:

        if column not in jobs.columns:
            jobs[column] = ""

    text_columns = [
        "title",
        "normalized_title",
        "company_name",
        "industry",
        "function",
        "experience_level",
        "employment_type",
        "work_model",
        "education_level",
        "job_description",
        "description",
        "responsibilities",
        "minimum_qualifications",
        "preferred_qualifications",
    ]

    for column in text_columns:
        jobs[column] = jobs[column].map(clean)

    skill_columns = [
        "normalized_skills",
        "parsed_skills",
        "skills_required",
    ]

    jobs["skill_set"] = [
        set().union(
            *(
                parse_skills(row[column])
                for column in skill_columns
            )
        )
        for _, row in jobs[skill_columns].iterrows()
    ]

    jobs["normalized_skills"] = jobs["skill_set"].map(
        lambda values: ", ".join(sorted(values))
    )

    jobs["search_text"] = (
        jobs["title"] + " "
        + jobs["normalized_title"] + " "
        + jobs["job_description"] + " "
        + jobs["description"] + " "
        + jobs["responsibilities"] + " "
        + jobs["minimum_qualifications"] + " "
        + jobs["preferred_qualifications"] + " "
        + jobs["normalized_skills"]
    ).str.strip()

    jobs = jobs.loc[
        jobs["search_text"].ne("")
    ].reset_index(drop=True)

    if jobs.empty:
        raise ValueError(
            "No searchable job records remain."
        )

    return jobs


# =========================================================
# 6. MAIN RECOMMENDATION ENGINE
# =========================================================

def recommend_jobs(
    profile: str,
    preferred_skills: list[str] | None = None,
    top_k: int = 10,
    min_similarity: float = 0.0,
    target_role: str | None = None,
    experience_level: str | None = None,
) -> pd.DataFrame:
    """
    Hybrid recommendation scoring:

    Content relevance:         20%
    Candidate skill coverage:  30%
    Job requirement coverage:  15%
    Target-role relevance:     30%
    Experience compatibility:  5%

    The function signature and output columns are preserved
    for compatibility with the existing CareerIntel backend.
    """

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

    if (
        not isinstance(min_similarity, (int, float))
        or not 0 <= min_similarity <= 1
    ):
        raise ValueError(
            "min_similarity must be between 0 and 1."
        )

    target_role = clean(target_role)
    experience_level = clean(experience_level)

    # Load dataset
    jobs = load_jobs()
    original_count = len(jobs)

    print(f"Total searchable jobs: {original_count}")

    # Experience filter
    if experience_level:

        jobs = filter_by_experience(
            jobs,
            experience_level,
        )

    print(
        f"Jobs after experience filtering: "
        f"{len(jobs)} / {original_count}"
    )

    if jobs.empty:
        print("No jobs match the selected experience level.")
        return pd.DataFrame()

    # Target-role filter
    if target_role:

        before = len(jobs)

        jobs = filter_by_target_role(
            jobs,
            target_role,
            MINIMUM_ROLE_MATCH,
        )

        print(
            f"Jobs after target-role filtering: "
            f"{len(jobs)} / {before}"
        )

    if jobs.empty:
        print("No jobs match the selected target role.")
        return pd.DataFrame()

    # Candidate skills
    user_skills: set[str] = set()

    for skill in preferred_skills or []:
        user_skills.update(parse_skills(skill))

    skill_vocabulary = set(user_skills)

    for job_skill_set in jobs["skill_set"]:
        skill_vocabulary.update(job_skill_set)

    # TF-IDF content similarity
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100_000,
        sublinear_tf=True,
        min_df=1,
    )

    try:

        job_matrix = vectorizer.fit_transform(
            jobs["search_text"]
        )

        profile_vector = vectorizer.transform(
            [profile.strip()]
        )

    except ValueError as exc:

        raise ValueError(
            "Unable to build TF-IDF vocabulary. "
            "Check the job dataset."
        ) from exc

    content_scores = cosine_similarity(
        profile_vector,
        job_matrix,
    ).ravel()

    # Skill coverage and requirement coverage
    skill_scores = []
    requirement_scores = []
    matched_skill_details = []

    for _, row in jobs.iterrows():

        job_skills = row["skill_set"]

        matching = user_skills.intersection(job_skills)

        matched_skill_details.append(
            ", ".join(sorted(matching))
        )

        skill_score = (
            len(matching) / len(user_skills)
            if user_skills else 0.0
        )

        skill_scores.append(skill_score)

        required = (
            parse_skills(row["skills_required"])
            | parse_skills(row["parsed_skills"])
        )

        required |= extract_skills_from_text(
            row["minimum_qualifications"],
            skill_vocabulary,
        )

        required |= extract_skills_from_text(
            row["preferred_qualifications"],
            skill_vocabulary,
        )

        if required and user_skills:

            requirement_score = (
                len(user_skills.intersection(required))
                / len(required)
            )

        else:
            requirement_score = float("nan")

        requirement_scores.append(requirement_score)

    skill_scores = pd.Series(
        skill_scores,
        dtype=float,
    ).to_numpy()

    requirement_scores = pd.Series(
        requirement_scores,
        dtype=float,
    ).to_numpy()

    # Role relevance
    if target_role:
        role_scores = jobs["role_match"].tolist()
    else:
        role_scores = [None] * len(jobs)

    # Experience compatibility
    if experience_level:

        experience_scores = [
            calculate_experience_match(
                experience_level,
                level,
            )
            for level in jobs["experience_level"]
        ]

    else:
        experience_scores = [None] * len(jobs)

    # Hybrid scoring
    final_scores = []

    for index in range(len(jobs)):

        components = [
            (content_scores[index], CONTENT_WEIGHT),
            (skill_scores[index], SKILL_WEIGHT),
        ]

        if not pd.isna(requirement_scores[index]):

            components.append(
                (
                    requirement_scores[index],
                    REQUIREMENT_WEIGHT,
                )
            )

        if role_scores[index] is not None:

            components.append(
                (role_scores[index], ROLE_WEIGHT)
            )

        if experience_scores[index] is not None:

            components.append(
                (
                    experience_scores[index],
                    EXPERIENCE_WEIGHT,
                )
            )

        total_weight = sum(
            weight
            for _, weight in components
        )

        final_score = sum(
            value * weight
            for value, weight in components
        ) / total_weight

        final_scores.append(final_score)

    # Build results
    results = jobs.copy()

    results["content_score"] = content_scores
    results["skill_coverage"] = skill_scores
    results["requirement_coverage"] = requirement_scores
    results["role_match"] = role_scores
    results["experience_match"] = experience_scores
    results["recommendation_score"] = final_scores
    results["matching_skills"] = matched_skill_details

    # Quality filters
    results = results.loc[
        results["content_score"] >= min_similarity
    ].copy()

    if REQUIRE_SKILL_MATCH and user_skills:

        results = results.loc[
            results["skill_coverage"] > 0
        ].copy()

    if results.empty:

        print(
            "No jobs passed the recommendation quality filters."
        )

        return pd.DataFrame()

    # Sort recommendations
    results = results.sort_values(
        by=[
            "recommendation_score",
            "role_match",
            "skill_coverage",
            "content_score",
        ],
        ascending=[False, False, False, False],
    )

    # Remove duplicate title/company combinations
    results["_title_key"] = (
        results["title"]
        .str.casefold()
        .str.strip()
    )

    results["_company_key"] = (
        results["company_name"]
        .map(clean)
        .str.casefold()
    )

    results = results.drop_duplicates(
        subset=["_title_key", "_company_key"],
        keep="first",
    )

    results = results.head(top_k).copy()

    # Preserve backend output columns
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
        "content_score",
        "skill_coverage",
        "requirement_coverage",
        "role_match",
        "experience_match",
        "recommendation_score",
    ]

    return results[output_columns].reset_index(drop=True)


# =========================================================
# 7. COMMAND-LINE TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 65)
    print("CAREERINTEL | JOB RECOMMENDATION ENGINE")
    print("=" * 65)

    profile = (
        "Aspiring Data Scientist with Python, SQL, "
        "machine learning, statistics, data analysis, "
        "pandas, and data visualization skills."
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

    print("\n=== DATASET VALIDATION ===")

    jobs = load_jobs()

    print(f"Total searchable jobs: {len(jobs)}")

    user_skill_set: set[str] = set()

    for skill in skills:
        user_skill_set.update(parse_skills(skill))

    print(
        "Jobs containing at least one parsed skill: "
        f"{sum(bool(x) for x in jobs['skill_set'])}"
    )

    print(
        "Jobs matching profile skills: "
        f"{sum(bool(user_skill_set.intersection(x)) for x in jobs['skill_set'])}"
    )

    print("\n=== RECOMMENDATION RESULTS ===")

    recommendations = recommend_jobs(
        profile=profile,
        preferred_skills=skills,
        target_role="Data Scientist",
        experience_level="Fresher",
        top_k=10,
        min_similarity=0.0,
    )

    print(
        f"\nRecommendations generated: "
        f"{len(recommendations)}"
    )

    output_file = OUTPUT_DIR / "sample_recommendations.csv"

    recommendations.to_csv(
        output_file,
        index=False,
    )

    print(f"Saved to: {output_file}")

    print("\n=== TOP RECOMMENDATIONS ===")

    if not recommendations.empty:

        columns = [
            "title",
            "company_name",
            "experience_level",
            "matching_skills",
            "content_score",
            "skill_coverage",
            "requirement_coverage",
            "role_match",
            "experience_match",
            "recommendation_score",
        ]

        print(
            recommendations[columns].to_string(index=False)
        )

        print("\n=== RECOMMENDATION VALIDATION ===")

        print(
            f"Mean skill coverage: "
            f"{recommendations['skill_coverage'].mean():.4f}"
        )

        print(
            f"Mean content similarity: "
            f"{recommendations['content_score'].mean():.4f}"
        )

        print(
            f"Mean recommendation score: "
            f"{recommendations['recommendation_score'].mean():.4f}"
        )

    else:
        print("No recommendations found.")

    print("\nExecution completed.")