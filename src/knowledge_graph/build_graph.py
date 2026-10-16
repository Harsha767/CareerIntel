from pathlib import Path

import pandas as pd

# ============================================================
# CAREERINTEL - KNOWLEDGE GRAPH DATASET BUILDER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ONET_DIR = PROCESSED_DIR / "onet"
GRAPH_DIR = PROCESSED_DIR / "knowledge_graph"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

SEMANTIC_MATCHES_FILE = (
    PROCESSED_DIR / "jobs" / "jobs_semantic_matches.csv"
)

OCCUPATION_MASTER_FILE = (
    ONET_DIR / "occupation_master.csv"
)

OCCUPATION_SKILL_FILE = (
    GRAPH_DIR / "occupation_skill_edges.csv"
)

RELATED_OCCUPATION_FILE = (
    ONET_DIR / "related_occupations_mapping.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

JOB_NODES_FILE = GRAPH_DIR / "job_nodes.csv"

OCCUPATION_NODES_FILE = GRAPH_DIR / "occupation_nodes.csv"

SKILL_NODES_FILE = GRAPH_DIR / "skill_nodes.csv"

JOB_OCCUPATION_EDGES_FILE = (
    GRAPH_DIR / "job_occupation_edges.csv"
)

OCCUPATION_SKILL_EDGES_FILE = (
    GRAPH_DIR / "occupation_skill_edges.csv"
)

OCCUPATION_OCCUPATION_EDGES_FILE = (
    GRAPH_DIR / "occupation_occupation_edges.csv"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 65)
    print(title)
    print("=" * 65)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


def require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    filename: str,
) -> None:
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"\n{filename} is missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    for column in candidates:
        if column in df.columns:
            return column

    return None


def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def clean_code(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    # Avoid values such as 11-1011.0
    # becoming inconsistent with 11-1011.00
    try:
        if "-" in text and "." in text:
            left, right = text.split(".", 1)

            if right.isdigit():
                text = f"{left}.{right.zfill(2)}"
    except (ValueError, TypeError, AttributeError):
      return text

    return text


# ============================================================
# LOAD SEMANTIC JOB MATCHES
# ============================================================

def load_semantic_matches() -> pd.DataFrame:

    require_file(SEMANTIC_MATCHES_FILE)

    print("Loading semantic job matches...")

    df = pd.read_csv(SEMANTIC_MATCHES_FILE)

    print(f"Jobs loaded: {len(df):,}")
    print(f"Columns detected: {list(df.columns)}")

    if df.empty:
        raise ValueError(
            "Semantic job matches file is empty."
        )

    return df


# ============================================================
# BUILD JOB NODES
# ============================================================

def build_job_nodes(
    jobs: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("Building JOB nodes...")

    title_column = first_existing_column(
        jobs,
        [
            "title",
            "job_title",
            "normalized_title",
        ],
    )

    if title_column is None:
        raise ValueError(
            "Could not find a job title column.\n"
            f"Available columns: {list(jobs.columns)}"
        )

    job_id_column = first_existing_column(
        jobs,
        [
            "job_id",
            "id",
            "job_idx",
            "job_index",
        ],
    )

    output = pd.DataFrame()

    # --------------------------------------------------------
    # Stable job IDs
    # --------------------------------------------------------

    if job_id_column is not None:

        output["job_id"] = (
            jobs[job_id_column]
            .astype(str)
            .str.strip()
        )

    else:

        output["job_id"] = [
            f"job_{index:06d}"
            for index in range(1, len(jobs) + 1)
        ]

    # --------------------------------------------------------
    # Job title
    # --------------------------------------------------------

    output["job_title"] = (
        jobs[title_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Optional job attributes
    # --------------------------------------------------------

    optional_columns = [
        "company_name",
        "industry",
        "function",
        "country",
        "city",
        "work_model",
        "employment_type",
        "experience_level",
        "education_level",
    ]

    for column in optional_columns:

        if column in jobs.columns:

            output[column] = (
                jobs[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        else:

            output[column] = ""

    # --------------------------------------------------------
    # Remove invalid records
    # --------------------------------------------------------

    output = output[
        output["job_title"].str.len() > 0
    ].copy()

    output = output.drop_duplicates(
        subset=["job_id"],
        keep="first",
    )

    output.to_csv(
        JOB_NODES_FILE,
        index=False,
    )

    print(
        f"Job nodes saved: {len(output):,}"
    )

    print(
        f"Saved: {JOB_NODES_FILE}"
    )

    return output


# ============================================================
# LOAD O*NET OCCUPATIONS
# ============================================================

def load_occupations() -> pd.DataFrame:

    require_file(OCCUPATION_MASTER_FILE)

    print()
    print("Loading O*NET occupations...")

    df = pd.read_csv(
        OCCUPATION_MASTER_FILE
    )

    print(
        f"Occupations loaded: {len(df):,}"
    )

    print(
        f"Columns detected: {list(df.columns)}"
    )

    code_column = first_existing_column(
        df,
        [
            "O*NET-SOC Code",
            "onet_soc_code",
            "soc_code",
            "occupation_code",
        ],
    )

    title_column = first_existing_column(
        df,
        [
            "Title",
            "occupation_title",
            "title",
        ],
    )

    if code_column is None:
        raise ValueError(
            "Could not identify O*NET occupation code column."
        )

    if title_column is None:
        raise ValueError(
            "Could not identify occupation title column."
        )

    output = pd.DataFrame()

    output["occupation_id"] = (
        df[code_column]
        .apply(clean_code)
    )

    output["occupation_title"] = (
        df[title_column]
        .apply(clean_text)
    )

    description_column = first_existing_column(
        df,
        [
            "Description",
            "description",
        ],
    )

    if description_column is not None:

        output["description"] = (
            df[description_column]
            .apply(clean_text)
        )

    else:

        output["description"] = ""

    output = output[
        output["occupation_id"].str.len() > 0
    ]

    output = output[
        output["occupation_title"].str.len() > 0
    ]

    output = output.drop_duplicates(
        subset=["occupation_id"],
        keep="first",
    )

    return output


# ============================================================
# BUILD OCCUPATION NODES
# ============================================================

def build_occupation_nodes() -> pd.DataFrame:

    print()
    print("Building OCCUPATION nodes...")

    occupations = load_occupations()

    occupations.to_csv(
        OCCUPATION_NODES_FILE,
        index=False,
    )

    print(
        f"Occupation nodes saved: "
        f"{len(occupations):,}"
    )

    print(
        f"Saved: {OCCUPATION_NODES_FILE}"
    )

    return occupations


# ============================================================
# BUILD JOB -> OCCUPATION EDGES
# ============================================================

def build_job_occupation_edges(
    jobs: pd.DataFrame,
    job_nodes: pd.DataFrame,
    occupations: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("Building JOB -> OCCUPATION edges...")

    title_column = first_existing_column(
        jobs,
        [
            "title",
            "job_title",
            "normalized_title",
        ],
    )

    occupation_column = first_existing_column(
        jobs,
        [
            "onet_occupation_title",
            "occupation_title",
        ],
    )

    score_column = first_existing_column(
        jobs,
        [
            "semantic_score",
            "similarity_score",
            "score",
        ],
    )

    if title_column is None:
        raise ValueError(
            "Job title column not found."
        )

    if occupation_column is None:
        raise ValueError(
            "O*NET occupation title column not found."
        )

    if score_column is None:
        raise ValueError(
            "Semantic score column not found."
        )

    # --------------------------------------------------------
    # Create row-based temporary ID so it matches job_nodes
    # when original dataset has no explicit job ID.
    # --------------------------------------------------------

    temp = jobs.copy()

    temp["_row_number"] = range(
        1,
        len(temp) + 1,
    )

    if first_existing_column(
        jobs,
        [
            "job_id",
            "id",
            "job_idx",
            "job_index",
        ],
    ) is None:

        temp["job_id"] = [
            f"job_{index:06d}"
            for index in range(1, len(temp) + 1)
        ]

    else:

        id_column = first_existing_column(
            jobs,
            [
                "job_id",
                "id",
                "job_idx",
                "job_index",
            ],
        )

        temp["job_id"] = (
            temp[id_column]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Match occupation title -> occupation ID
    # --------------------------------------------------------

    occupation_lookup = (
        occupations[
            [
                "occupation_id",
                "occupation_title",
            ]
        ]
        .copy()
    )

    occupation_lookup["occupation_title_key"] = (
        occupation_lookup["occupation_title"]
        .str.lower()
        .str.strip()
    )

    temp["occupation_title_key"] = (
        temp[occupation_column]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    merged = temp.merge(
        occupation_lookup[
            [
                "occupation_id",
                "occupation_title",
                "occupation_title_key",
            ]
        ],
        on="occupation_title_key",
        how="inner",
        suffixes=(
            "_job",
            "_onet",
        ),
    )

    # --------------------------------------------------------
    # Build final edge table
    # --------------------------------------------------------

    edges = pd.DataFrame()

    edges["job_id"] = (
        merged["job_id"]
        .astype(str)
    )

    edges["occupation_id"] = (
        merged["occupation_id"]
        .astype(str)
    )

    edges["occupation_title"] = (
        merged["occupation_title_onet"]
        .astype(str)
    )

    edges["semantic_score"] = pd.to_numeric(
        merged[score_column],
        errors="coerce",
    )

    edges = edges.dropna(
        subset=["semantic_score"]
    )

    edges["semantic_score"] = (
        edges["semantic_score"]
        .clip(0.0, 1.0)
    )

    edges = edges.drop_duplicates(
        subset=[
            "job_id",
            "occupation_id",
        ],
        keep="first",
    )

    edges.to_csv(
        JOB_OCCUPATION_EDGES_FILE,
        index=False,
    )

    print(
        f"Job -> occupation edges: "
        f"{len(edges):,}"
    )

    print(
        f"Saved: {JOB_OCCUPATION_EDGES_FILE}"
    )

    return edges


# ============================================================
# LOAD EXISTING OCCUPATION -> SKILL EDGES
# ============================================================

def load_occupation_skill_edges() -> pd.DataFrame:

    require_file(OCCUPATION_SKILL_FILE)

    print()
    print("Loading OCCUPATION -> SKILL edges...")

    df = pd.read_csv(
        OCCUPATION_SKILL_FILE
    )

    print(
        f"Raw occupation-skill edges: "
        f"{len(df):,}"
    )

    print(
        f"Columns detected: {list(df.columns)}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Your actual file has:
    #
    # source
    # target
    # skill_name
    #
    # source = O*NET-SOC code
    # target = O*NET Element ID
    # skill_name = human-readable skill
    # --------------------------------------------------------

    require_columns(
        df,
        [
            "source",
            "target",
            "skill_name",
        ],
        "occupation_skill_edges.csv",
    )

    output = pd.DataFrame()

    output["occupation_id"] = (
        df["source"]
        .apply(clean_code)
    )

    output["skill_source_id"] = (
        df["target"]
        .apply(clean_text)
    )

    output["skill_name"] = (
        df["skill_name"]
        .apply(clean_text)
    )

    output["relationship"] = (
        "REQUIRES"
    )

    output = output[
        (output["occupation_id"] != "")
        & (output["skill_source_id"] != "")
        & (output["skill_name"] != "")
    ].copy()

    output = output.drop_duplicates(
        subset=[
            "occupation_id",
            "skill_source_id",
        ],
        keep="first",
    )

    print(
        f"Valid occupation-skill edges: "
        f"{len(output):,}"
    )

    return output


# ============================================================
# BUILD SKILL NODES
# ============================================================

def build_skill_nodes(
    occupation_skill_edges: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("Building SKILL nodes...")

    skills = (
        occupation_skill_edges[
            [
                "skill_source_id",
                "skill_name",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    skills = skills[
        skills["skill_name"].str.len() > 0
    ]

    # --------------------------------------------------------
    # Create our own clean integer skill ID
    # --------------------------------------------------------

    skills = skills.reset_index(drop=True)

    skills["skill_id"] = [
        f"skill_{index:04d}"
        for index in range(1, len(skills) + 1)
    ]

    skills = skills[
        [
            "skill_id",
            "skill_source_id",
            "skill_name",
        ]
    ]

    skills.to_csv(
        SKILL_NODES_FILE,
        index=False,
    )

    print(
        f"Skill nodes saved: "
        f"{len(skills):,}"
    )

    print(
        f"Saved: {SKILL_NODES_FILE}"
    )

    return skills


# ============================================================
# SAVE NORMALIZED OCCUPATION -> SKILL EDGES
# ============================================================

def save_occupation_skill_edges(
    occupation_skill_edges: pd.DataFrame,
    skill_nodes: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("Normalizing OCCUPATION -> SKILL edges...")

    edges = occupation_skill_edges.merge(
        skill_nodes[
            [
                "skill_id",
                "skill_source_id",
            ]
        ],
        on="skill_source_id",
        how="inner",
    )

    edges = edges[
        [
            "occupation_id",
            "skill_id",
            "skill_name",
            "relationship",
        ]
    ]

    edges = edges.drop_duplicates(
        subset=[
            "occupation_id",
            "skill_id",
        ],
        keep="first",
    )

    edges.to_csv(
        OCCUPATION_SKILL_EDGES_FILE,
        index=False,
    )

    print(
        f"Occupation -> skill edges saved: "
        f"{len(edges):,}"
    )

    print(
        f"Saved: {OCCUPATION_SKILL_EDGES_FILE}"
    )

    return edges


# ============================================================
# BUILD OCCUPATION -> OCCUPATION EDGES
# ============================================================

def build_occupation_occupation_edges(
    occupations: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("Building OCCUPATION -> OCCUPATION edges...")

    if not RELATED_OCCUPATION_FILE.exists():

        print(
            "Related occupation file not found."
        )

        print(
            "Creating empty occupation-transition table."
        )

        empty = pd.DataFrame(
            columns=[
                "source_occupation_id",
                "target_occupation_id",
                "source_title",
                "target_title",
                "relationship_tier",
                "relationship_index",
            ]
        )

        empty.to_csv(
            OCCUPATION_OCCUPATION_EDGES_FILE,
            index=False,
        )

        return empty

    related = pd.read_csv(
        RELATED_OCCUPATION_FILE
    )

    print(
        f"Related occupation records: "
        f"{len(related):,}"
    )

    print(
        f"Columns detected: {list(related.columns)}"
    )

    source_code_column = first_existing_column(
        related,
        [
            "O*NET-SOC Code",
            "onet_soc_code",
            "source_occupation_id",
            "source",
        ],
    )

    target_code_column = first_existing_column(
        related,
        [
            "Related O*NET-SOC Code",
            "related_onet_soc_code",
            "target_occupation_id",
            "target",
        ],
    )

    source_title_column = first_existing_column(
        related,
        [
            "Title",
            "source_title",
        ],
    )

    target_title_column = first_existing_column(
        related,
        [
            "Related Title",
            "related_title",
            "target_title",
        ],
    )

    tier_column = first_existing_column(
        related,
        [
            "Relatedness Tier",
            "relatedness_tier",
            "relationship_tier",
        ],
    )

    index_column = first_existing_column(
        related,
        [
            "Index",
            "index",
            "relationship_index",
        ],
    )

    if source_code_column is None:
        raise ValueError(
            "Could not identify source occupation code "
            "in related occupations file."
        )

    if target_code_column is None:
        raise ValueError(
            "Could not identify target occupation code "
            "in related occupations file."
        )

    output = pd.DataFrame()

    output["source_occupation_id"] = (
        related[source_code_column]
        .apply(clean_code)
    )

    output["target_occupation_id"] = (
        related[target_code_column]
        .apply(clean_code)
    )

    if source_title_column is not None:

        output["source_title"] = (
            related[source_title_column]
            .apply(clean_text)
        )

    else:

        output["source_title"] = ""

    if target_title_column is not None:

        output["target_title"] = (
            related[target_title_column]
            .apply(clean_text)
        )

    else:

        output["target_title"] = ""

    if tier_column is not None:

        output["relationship_tier"] = (
            related[tier_column]
            .apply(clean_text)
        )

    else:

        output["relationship_tier"] = ""

    if index_column is not None:

        output["relationship_index"] = (
            related[index_column]
        )

    else:

        output["relationship_index"] = ""

    output = output[
        (output["source_occupation_id"] != "")
        & (output["target_occupation_id"] != "")
    ]

    # Keep only occupations that exist in our occupation nodes.
    valid_ids = set(
        occupations["occupation_id"]
    )

    output = output[
        output["source_occupation_id"].isin(valid_ids)
        & output["target_occupation_id"].isin(valid_ids)
    ].copy()

    output = output.drop_duplicates(
        subset=[
            "source_occupation_id",
            "target_occupation_id",
        ],
        keep="first",
    )

    output.to_csv(
        OCCUPATION_OCCUPATION_EDGES_FILE,
        index=False,
    )

    print(
        f"Occupation -> occupation edges: "
        f"{len(output):,}"
    )

    print(
        f"Saved: {OCCUPATION_OCCUPATION_EDGES_FILE}"
    )

    return output


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

def run_quality_checks(
    job_nodes: pd.DataFrame,
    occupation_nodes: pd.DataFrame,
    skill_nodes: pd.DataFrame,
    job_occupation_edges: pd.DataFrame,
    occupation_skill_edges: pd.DataFrame,
    occupation_occupation_edges: pd.DataFrame,
) -> None:

    print_header("KNOWLEDGE GRAPH QUALITY CHECK")

    # --------------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------------

    print("DUPLICATE CHECK")
    print("-" * 40)

    print(
        "Duplicate job nodes:",
        job_nodes["job_id"].duplicated().sum(),
    )

    print(
        "Duplicate occupation nodes:",
        occupation_nodes["occupation_id"].duplicated().sum(),
    )

    print(
        "Duplicate skill nodes:",
        skill_nodes["skill_id"].duplicated().sum(),
    )

    print(
        "Duplicate job-occupation edges:",
        job_occupation_edges[
            [
                "job_id",
                "occupation_id",
            ]
        ].duplicated().sum(),
    )

    print(
        "Duplicate occupation-skill edges:",
        occupation_skill_edges[
            [
                "occupation_id",
                "skill_id",
            ]
        ].duplicated().sum(),
    )

    print(
        "Duplicate occupation-occupation edges:",
        occupation_occupation_edges[
            [
                "source_occupation_id",
                "target_occupation_id",
            ]
        ].duplicated().sum(),
    )

    # --------------------------------------------------------
    # Referential integrity
    # --------------------------------------------------------

    print()
    print("REFERENTIAL INTEGRITY")
    print("-" * 40)

    valid_jobs = set(
        job_nodes["job_id"]
    )

    valid_occupations = set(
        occupation_nodes["occupation_id"]
    )

    valid_skills = set(
        skill_nodes["skill_id"]
    )

    invalid_job_edges = (
        ~job_occupation_edges["job_id"].isin(valid_jobs)
    ).sum()

    invalid_job_occupation_edges = (
        ~job_occupation_edges["occupation_id"]
        .isin(valid_occupations)
    ).sum()

    invalid_skill_edges = (
        ~occupation_skill_edges["skill_id"]
        .isin(valid_skills)
    ).sum()

    invalid_occupation_edges = (
        ~occupation_skill_edges["occupation_id"]
        .isin(valid_occupations)
    ).sum()

    invalid_transition_source = (
        ~occupation_occupation_edges[
            "source_occupation_id"
        ].isin(valid_occupations)
    ).sum()

    invalid_transition_target = (
        ~occupation_occupation_edges[
            "target_occupation_id"
        ].isin(valid_occupations)
    ).sum()

    print(
        "Invalid job references:",
        invalid_job_edges,
    )

    print(
        "Invalid job -> occupation references:",
        invalid_job_occupation_edges,
    )

    print(
        "Invalid occupation -> skill references:",
        invalid_occupation_edges,
    )

    print(
        "Invalid skill references:",
        invalid_skill_edges,
    )

    print(
        "Invalid occupation transition sources:",
        invalid_transition_source,
    )

    print(
        "Invalid occupation transition targets:",
        invalid_transition_target,
    )


# ============================================================
# SAMPLE OUTPUT
# ============================================================

def print_samples(
    job_occupation_edges: pd.DataFrame,
    occupation_skill_edges: pd.DataFrame,
) -> None:

    print()
    print("SAMPLE JOB -> OCCUPATION EDGES")
    print("-" * 50)

    if not job_occupation_edges.empty:

        print(
            job_occupation_edges[
                [
                    "job_id",
                    "occupation_id",
                    "occupation_title",
                    "semantic_score",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print()
    print("SAMPLE OCCUPATION -> SKILL EDGES")
    print("-" * 50)

    if not occupation_skill_edges.empty:

        print(
            occupation_skill_edges[
                [
                    "occupation_id",
                    "skill_id",
                    "skill_name",
                    "relationship",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )


# ============================================================
# GRAPH SUMMARY
# ============================================================

def print_summary(
    job_nodes: pd.DataFrame,
    occupation_nodes: pd.DataFrame,
    skill_nodes: pd.DataFrame,
    job_occupation_edges: pd.DataFrame,
    occupation_skill_edges: pd.DataFrame,
    occupation_occupation_edges: pd.DataFrame,
) -> None:

    print_header("KNOWLEDGE GRAPH SUMMARY")

    print(
        f"Job nodes                  : "
        f"{len(job_nodes):,}"
    )

    print(
        f"Occupation nodes           : "
        f"{len(occupation_nodes):,}"
    )

    print(
        f"Skill nodes                : "
        f"{len(skill_nodes):,}"
    )

    print(
        f"Job -> Occupation edges    : "
        f"{len(job_occupation_edges):,}"
    )

    print(
        f"Occupation -> Skill edges  : "
        f"{len(occupation_skill_edges):,}"
    )

    print(
        f"Occupation -> Occupation   : "
        f"{len(occupation_occupation_edges):,}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_header(
        "CareerIntel - Knowledge Graph Dataset Builder"
    )

    # --------------------------------------------------------
    # 1. Load semantic job matches
    # --------------------------------------------------------

    jobs = load_semantic_matches()

    # --------------------------------------------------------
    # 2. Build job nodes
    # --------------------------------------------------------

    job_nodes = build_job_nodes(
        jobs
    )

    # --------------------------------------------------------
    # 3. Build occupation nodes
    # --------------------------------------------------------

    occupation_nodes = (
        build_occupation_nodes()
    )

    # --------------------------------------------------------
    # 4. Build job -> occupation edges
    # --------------------------------------------------------

    job_occupation_edges = (
        build_job_occupation_edges(
            jobs,
            job_nodes,
            occupation_nodes,
        )
    )

    # --------------------------------------------------------
    # 5. Load occupation -> skill relationships
    # --------------------------------------------------------

    raw_occupation_skill_edges = (
        load_occupation_skill_edges()
    )

    # --------------------------------------------------------
    # 6. Build skill nodes
    # --------------------------------------------------------

    skill_nodes = build_skill_nodes(
        raw_occupation_skill_edges
    )

    # --------------------------------------------------------
    # 7. Save normalized occupation -> skill edges
    # --------------------------------------------------------

    occupation_skill_edges = (
        save_occupation_skill_edges(
            raw_occupation_skill_edges,
            skill_nodes,
        )
    )

    # --------------------------------------------------------
    # 8. Build occupation -> occupation edges
    # --------------------------------------------------------

    occupation_occupation_edges = (
        build_occupation_occupation_edges(
            occupation_nodes
        )
    )

    # --------------------------------------------------------
    # 9. Quality checks
    # --------------------------------------------------------

    run_quality_checks(
        job_nodes,
        occupation_nodes,
        skill_nodes,
        job_occupation_edges,
        occupation_skill_edges,
        occupation_occupation_edges,
    )

    # --------------------------------------------------------
    # 10. Samples
    # --------------------------------------------------------

    print_samples(
        job_occupation_edges,
        occupation_skill_edges,
    )

    # --------------------------------------------------------
    # 11. Summary
    # --------------------------------------------------------

    print_summary(
        job_nodes,
        occupation_nodes,
        skill_nodes,
        job_occupation_edges,
        occupation_skill_edges,
        occupation_occupation_edges,
    )

    # --------------------------------------------------------
    # 12. Completion
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("KNOWLEDGE GRAPH DATASET COMPLETE")
    print("=" * 65)

    print()
    print(
        f"Output directory: {GRAPH_DIR}"
    )

    print()
    print("Generated files:")

    print(
        f"  - {JOB_NODES_FILE.name}"
    )

    print(
        f"  - {OCCUPATION_NODES_FILE.name}"
    )

    print(
        f"  - {SKILL_NODES_FILE.name}"
    )

    print(
        f"  - {JOB_OCCUPATION_EDGES_FILE.name}"
    )

    print(
        f"  - {OCCUPATION_SKILL_EDGES_FILE.name}"
    )

    print(
        f"  - {OCCUPATION_OCCUPATION_EDGES_FILE.name}"
    )

    print()
    print("CareerIntel knowledge graph dataset is ready.")


if __name__ == "__main__":
    main()