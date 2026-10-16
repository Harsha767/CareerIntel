from pathlib import Path

import pandas as pd

# ============================================================
# CAREERINTEL — FIX O*NET SKILL NODES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ONET_DIR = PROJECT_ROOT / "data" / "external" / "onet" / "db_31_0_csv"
PROCESSED_ONET_DIR = PROJECT_ROOT / "data" / "processed" / "onet"
GRAPH_DIR = PROJECT_ROOT / "data" / "processed" / "knowledge_graph"

ESSENTIAL_SKILLS_FILE = ONET_DIR / "essential_skills.csv"
OCCUPATION_SKILLS_FILE = PROCESSED_ONET_DIR / "occupation_essential_skills.csv"

SKILL_NODES_OUTPUT = GRAPH_DIR / "skill_nodes.csv"
OCCUPATION_SKILL_EDGES_OUTPUT = GRAPH_DIR / "occupation_skill_edges.csv"


def normalize_text(value):
    """Normalize text while preserving meaningful skill names."""

    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )


def find_column(df, candidates):
    """Find the first matching column from a list of candidates."""

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def main():

    print("=" * 60)
    print("CareerIntel — Fix O*NET Skill Nodes")
    print("=" * 60)

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load O*NET essential skills
    # --------------------------------------------------------

    print("\nLoading O*NET essential skills...")

    if not ESSENTIAL_SKILLS_FILE.exists():
        raise FileNotFoundError(
            f"O*NET file not found:\n{ESSENTIAL_SKILLS_FILE}"
        )

    essential = pd.read_csv(
        ESSENTIAL_SKILLS_FILE,
        low_memory=False
    )

    print(f"Raw O*NET skill records: {len(essential):,}")

    # --------------------------------------------------------
    # 2. Detect required columns
    # --------------------------------------------------------

    code_col = find_column(
        essential,
        [
            "O*NET-SOC Code",
            "O*NET SOC Code",
            "onet_soc_code",
        ],
    )

    title_col = find_column(
        essential,
        [
            "Title",
            "occupation_title",
        ],
    )

    element_id_col = find_column(
        essential,
        [
            "Element ID",
            "element_id",
        ],
    )

    element_name_col = find_column(
        essential,
        [
            "Element Name",
            "element_name",
        ],
    )

    scale_id_col = find_column(
        essential,
        [
            "Scale ID",
            "scale_id",
        ],
    )

    find_column(
        essential,
        [
            "Scale Name",
            "scale_name",
        ],
    )

    required = {
        "occupation code": code_col,
        "occupation title": title_col,
        "element id": element_id_col,
        "element name": element_name_col,
    }

    missing = [
        name
        for name, column in required.items()
        if column is None
    ]

    if missing:
        raise ValueError(
            "Missing required O*NET columns: "
            + ", ".join(missing)
        )

    print("\nDetected columns:")

    print(f"  Occupation code : {code_col}")
    print(f"  Occupation title: {title_col}")
    print(f"  Element ID      : {element_id_col}")
    print(f"  Element Name    : {element_name_col}")

    if scale_id_col:
        print(f"  Scale ID        : {scale_id_col}")

    # --------------------------------------------------------
    # 3. Keep only importance records
    #
    # O*NET contains multiple measurements for the same
    # skill, commonly Importance (IM) and Level (LV).
    #
    # For the CareerIntel graph we need one occupation → skill
    # relationship, so use Importance records when available.
    # --------------------------------------------------------

    df = essential.copy()

    if scale_id_col is not None:

        scale_values = (
            df[scale_id_col]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        importance_mask = scale_values == "IM"

        if importance_mask.any():

            df = df.loc[importance_mask].copy()

            print(
                f"\nImportance records retained: {len(df):,}"
            )

        else:

            print(
                "\nWARNING: No 'IM' scale records found."
            )

    # --------------------------------------------------------
    # 4. Clean occupation and skill fields
    # --------------------------------------------------------

    df["onet_soc_code"] = (
        df[code_col]
        .apply(normalize_text)
    )

    df["occupation_title"] = (
        df[title_col]
        .apply(normalize_text)
    )

    df["element_id"] = (
        df[element_id_col]
        .apply(normalize_text)
    )

    df["skill_name"] = (
        df[element_name_col]
        .apply(normalize_text)
    )

    # Remove invalid records

    df = df[
        (df["onet_soc_code"] != "")
        & (df["element_id"] != "")
        & (df["skill_name"] != "")
    ].copy()

    print(
        f"Valid occupation-skill records: {len(df):,}"
    )

    # --------------------------------------------------------
    # 5. Remove duplicate occupation → skill relationships
    # --------------------------------------------------------

    edges = df[
        [
            "onet_soc_code",
            "occupation_title",
            "element_id",
            "skill_name",
        ]
    ].drop_duplicates(
        subset=[
            "onet_soc_code",
            "element_id",
        ]
    )

    # --------------------------------------------------------
    # 6. Create human-readable skill nodes
    #
    # IMPORTANT:
    # The previous graph incorrectly used Element IDs such as:
    #
    #     2.A.1.a
    #     2.A.1.b
    #
    # as skill names.
    #
    # Now we use:
    #
    #     Reading Comprehension
    #     Active Listening
    #     Social Perceptiveness
    #
    # while keeping the Element ID as the stable identifier.
    # --------------------------------------------------------

    skill_nodes = (
        edges[
            [
                "element_id",
                "skill_name",
            ]
        ]
        .drop_duplicates(
            subset=["element_id"]
        )
        .sort_values(
            "skill_name"
        )
        .reset_index(drop=True)
    )

    skill_nodes.insert(
        0,
        "skill_id",
        range(1, len(skill_nodes) + 1)
    )

    # --------------------------------------------------------
    # 7. Create final occupation → skill edges
    # --------------------------------------------------------

    occupation_skill_edges = edges[
        [
            "onet_soc_code",
            "element_id",
            "skill_name",
        ]
    ].copy()

    occupation_skill_edges = (
        occupation_skill_edges
        .sort_values(
            [
                "onet_soc_code",
                "skill_name",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 8. Save files
    # --------------------------------------------------------

    skill_nodes.to_csv(
        SKILL_NODES_OUTPUT,
        index=False,
    )

    occupation_skill_edges.to_csv(
        OCCUPATION_SKILL_EDGES_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # 9. Validation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SKILL NODE FIX COMPLETE")
    print("=" * 60)

    print(
        f"Unique skills          : "
        f"{len(skill_nodes):,}"
    )

    print(
        f"Occupation → skill edges: "
        f"{len(occupation_skill_edges):,}"
    )

    print(
        f"Occupations represented : "
        f"{occupation_skill_edges['onet_soc_code'].nunique():,}"
    )

    print(
        f"Unique O*NET elements   : "
        f"{occupation_skill_edges['element_id'].nunique():,}"
    )

    # --------------------------------------------------------
    # 10. Show sample skill nodes
    # --------------------------------------------------------

    print("\nSAMPLE SKILL NODES")
    print("-" * 60)

    print(
        skill_nodes.head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 11. Show most common skills
    # --------------------------------------------------------

    top_skills = (
        occupation_skill_edges
        .groupby(
            "skill_name"
        )
        .size()
        .sort_values(
            ascending=False
        )
        .head(30)
    )

    print("\nTOP 30 O*NET SKILLS")
    print("-" * 60)

    for skill, count in top_skills.items():
        print(
            f"{skill:<45} {count}"
        )

    # --------------------------------------------------------
    # 12. Verify that IDs are NOT being used as names
    # --------------------------------------------------------

    id_like_names = skill_nodes[
        skill_nodes["skill_name"]
        .str.match(
            r"^\d+\.[A-Z]\.\d+\.[a-z0-9]+$",
            case=False,
            na=False,
        )
    ]

    print("\nQUALITY CHECK")
    print("-" * 60)

    print(
        f"Skill nodes with ID-like names: "
        f"{len(id_like_names)}"
    )

    if len(id_like_names) == 0:

        print(
            "PASS: Skill nodes now contain "
            "human-readable O*NET skill names."
        )

    else:

        print(
            "WARNING: Some skill names still "
            "look like O*NET element IDs."
        )

    # --------------------------------------------------------
    # 13. Output paths
    # --------------------------------------------------------

    print("\nOUTPUT FILES")
    print("-" * 60)

    print(
        f"Skill nodes:\n"
        f"{SKILL_NODES_OUTPUT}"
    )

    print(
        f"\nOccupation-skill edges:\n"
        f"{OCCUPATION_SKILL_EDGES_OUTPUT}"
    )

    print("\n" + "=" * 60)
    print("READY FOR GRAPH REBUILD")
    print("=" * 60)


if __name__ == "__main__":
    main()