from pathlib import Path
import ast
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jobs"
    / "jobs_cleaned.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "jobs"
OUTPUT_PATH = OUTPUT_DIR / "jobs_with_normalized_skills.csv"


def parse_skills(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [
                str(skill).strip()
                for skill in parsed
                if str(skill).strip()
            ]

    except (ValueError, SyntaxError):
        pass

    return [
        skill.strip()
        for skill in value.split(",")
        if skill.strip()
    ]


def normalize_skill(skill):
    skill = str(skill).strip().lower()

    # Normalize whitespace
    skill = re.sub(r"\s+", " ", skill)

    # Remove surrounding punctuation
    skill = skill.strip(".,;:!?()[]{}\"'")

    return skill


def main():
    print("Loading cleaned dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Rows loaded: {len(df):,}")

    # Parse original skill lists
    df["parsed_skills"] = df["skills_required"].apply(parse_skills)

    # Normalize skill names
    df["normalized_skills"] = df["parsed_skills"].apply(
        lambda skills: sorted(
            set(
                normalize_skill(skill)
                for skill in skills
                if normalize_skill(skill)
            )
        )
    )

    # Number of skills per job
    df["skill_count"] = df["normalized_skills"].str.len()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    # Statistics
    all_skills = []

    for skills in df["normalized_skills"]:
        all_skills.extend(skills)

    skill_counts = (
        pd.Series(all_skills)
        .value_counts()
    )

    print()
    print("=" * 60)
    print("SKILL NORMALIZATION COMPLETE")
    print("=" * 60)

    print(f"Jobs: {len(df):,}")
    print(f"Unique normalized skills: {len(skill_counts):,}")

    print("\nSKILLS PER JOB:")
    print(df["skill_count"].describe().to_string())

    print("\nTOP 30 NORMALIZED SKILLS:")
    print(skill_counts.head(30).to_string())

    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()