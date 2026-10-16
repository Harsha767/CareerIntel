import ast
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jobs"
    / "jobs_with_normalized_skills.csv"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    print("=" * 60)
    print("CAREERINTEL — SKILL AUDIT")
    print("=" * 60)

    # Collect all normalized skills
    skills = []

    for value in df["normalized_skills"].dropna():
        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                skills.extend(parsed)

        except (ValueError, SyntaxError):
            continue

    skill_counts = pd.Series(skills).value_counts()

    print(f"\nTotal unique skills: {len(skill_counts):,}")
    print(f"Total skill occurrences: {len(skills):,}")

    print("\nTOP 50 SKILLS:")
    print(skill_counts.head(50).to_string())

    print("\nRARE SKILLS — 1 occurrence:")
    rare = skill_counts[skill_counts == 1]

    print(f"Count: {len(rare):,}")
    print(rare.head(100).to_string())

    print("\nSKILLS — 2 TO 5 OCCURRENCES:")
    uncommon = skill_counts[
        (skill_counts >= 2) & (skill_counts <= 5)
    ]

    print(f"Count: {len(uncommon):,}")
    print(uncommon.head(100).to_string())

    print("\n" + "=" * 60)
    print("SKILL AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()