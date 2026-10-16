from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONET_DIR = PROJECT_ROOT / "data" / "external" / "onet" / "db_31_0_csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "onet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_onet_file(filename: str) -> pd.DataFrame:
    """Load an O*NET CSV file."""
    path = ONET_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"O*NET file not found: {path}")

    return pd.read_csv(path)


def main():
    print("=" * 60)
    print("CareerIntel - O*NET Knowledge Mapping")
    print("=" * 60)

    # Load core O*NET tables
    job_titles = load_onet_file("job_titles.csv")
    occupations = load_onet_file("occupation_data.csv")
    essential_skills = load_onet_file("essential_skills.csv")
    transferable_skills = load_onet_file("transferable_skills.csv")
    related_occupations = load_onet_file("related_occupations.csv")
    software_skills = load_onet_file("software_skills.csv")

    print(f"Job titles loaded: {len(job_titles):,}")
    print(f"Occupations loaded: {len(occupations):,}")
    print(f"Essential skill records: {len(essential_skills):,}")
    print(f"Transferable skill records: {len(transferable_skills):,}")
    print(f"Related occupation records: {len(related_occupations):,}")
    print(f"Software skill records: {len(software_skills):,}")

    # ---------------------------------------------------------
    # 1. Occupation master table
    # ---------------------------------------------------------

    occupation_master = occupations[
        ["O*NET-SOC Code", "Title", "Description"]
    ].copy()

    occupation_master.columns = [
        "onet_soc_code",
        "occupation_title",
        "occupation_description",
    ]

    occupation_master.to_csv(
        OUTPUT_DIR / "occupation_master.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 2. Job title -> occupation mapping
    # ---------------------------------------------------------

    title_mapping = job_titles[
        ["O*NET-SOC Code", "Title", "Job Title", "Short Title"]
    ].copy()

    title_mapping.columns = [
        "onet_soc_code",
        "occupation_title",
        "job_title",
        "short_title",
    ]

    title_mapping.to_csv(
        OUTPUT_DIR / "job_title_occupation_mapping.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 3. Essential skills
    # ---------------------------------------------------------

    essential = essential_skills[
        [
            "O*NET-SOC Code",
            "Element ID",
            "Element Name",
            "Scale ID",
            "Scale Name",
            "Data Value",
        ]
    ].copy()

    essential.columns = [
        "onet_soc_code",
        "element_id",
        "skill_name",
        "scale_id",
        "scale_name",
        "data_value",
    ]

    essential.to_csv(
        OUTPUT_DIR / "occupation_essential_skills.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 4. Transferable skills
    # ---------------------------------------------------------

    transferable = transferable_skills[
        [
            "O*NET-SOC Code",
            "Element ID",
            "Element Name",
            "Scale ID",
            "Scale Name",
            "Data Value",
        ]
    ].copy()

    transferable.columns = [
        "onet_soc_code",
        "element_id",
        "skill_name",
        "scale_id",
        "scale_name",
        "data_value",
    ]

    transferable.to_csv(
        OUTPUT_DIR / "occupation_transferable_skills.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 5. Related occupations
    # ---------------------------------------------------------

    related = related_occupations[
        [
            "O*NET-SOC Code",
            "Related O*NET-SOC Code",
            "Related Title",
            "Relatedness Tier",
            "Index",
        ]
    ].copy()

    related.columns = [
        "onet_soc_code",
        "related_onet_soc_code",
        "related_title",
        "relatedness_tier",
        "relationship_index",
    ]

    related.to_csv(
        OUTPUT_DIR / "related_occupations_mapping.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 6. Software / technology skills
    # ---------------------------------------------------------

    software = software_skills[
        [
            "O*NET-SOC Code",
            "Title",
            "Workplace Example",
            "Element ID",
            "Element Name",
            "Hot Technology",
            "In Demand",
        ]
    ].copy()

    software.columns = [
        "onet_soc_code",
        "occupation_title",
        "workplace_example",
        "software_element_id",
        "software_name",
        "hot_technology",
        "in_demand",
    ]

    software.to_csv(
        OUTPUT_DIR / "occupation_software_skills.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("O*NET MAPPING COMPLETE")
    print("=" * 60)

    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Generated files:")

    for file in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  - {file.name}")

    print()
    print("This data will form the foundation of the")
    print("CareerIntel career knowledge graph.")


if __name__ == "__main__":
    main()