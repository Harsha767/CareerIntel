from pathlib import Path
import ast
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jobs"
    / "jobs_cleaned.csv"
)


def parse_skills(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]

    except (ValueError, SyntaxError):
        pass

    return [x.strip() for x in value.split(",") if x.strip()]


def main():
    df = pd.read_csv(INPUT_PATH)

    print("=" * 60)
    print("CAREERINTEL — DATA QUALITY PROFILE")
    print("=" * 60)

    # --------------------------------------------------
    # Missingness
    # --------------------------------------------------
    print("\nMISSING VALUES:")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing):
        print(missing.to_string())
    else:
        print("No missing values.")

    # --------------------------------------------------
    # Title consistency
    # --------------------------------------------------
    print("\nTITLE CHECK:")
    print("Unique title values:", df["title"].nunique())

    print("\nTop raw titles:")
    print(df["title"].value_counts().head(20).to_string())

    # --------------------------------------------------
    # Normalized title consistency
    # --------------------------------------------------
    print("\nNORMALIZED TITLE CHECK:")
    print("Unique normalized titles:", df["normalized_title"].nunique())

    print("\nTop normalized titles:")
    print(
        df["normalized_title"]
        .value_counts()
        .head(20)
        .to_string()
    )

    # --------------------------------------------------
    # Skills
    # --------------------------------------------------
    print("\nSKILLS REQUIRED:")

    df["parsed_skills"] = df["skills_required"].apply(parse_skills)

    skill_counts = {}

    for skills in df["parsed_skills"]:
        for skill in skills:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1

    skill_series = (
        pd.Series(skill_counts)
        .sort_values(ascending=False)
    )

    print("Jobs containing skills:",
          (df["parsed_skills"].str.len() > 0).sum())

    print("Unique skills:", len(skill_series))

    print("\nTOP 30 SKILLS:")

    if len(skill_series):
        print(skill_series.head(30).to_string())
    else:
        print("No structured skills detected.")

    # --------------------------------------------------
    # Job text
    # --------------------------------------------------
    print("\nJOB TEXT QUALITY:")

    text_length = df["job_text"].fillna("").str.len()

    print("Minimum:", text_length.min())
    print("Median:", text_length.median())
    print("Mean:", round(text_length.mean(), 2))
    print("Maximum:", text_length.max())

    # --------------------------------------------------
    # High-value modeling fields
    # --------------------------------------------------
    print("\nFIELD AVAILABILITY:")

    important_fields = [
        "title",
        "company_name",
        "industry",
        "function",
        "experience_level",
        "education_level",
        "skills_required",
        "responsibilities",
        "job_description",
        "job_text",
        "country",
        "work_model",
        "salary_min",
        "salary_max",
    ]

    for column in important_fields:
        if column in df.columns:
            available = df[column].notna().mean() * 100
            print(f"{column:30} {available:6.2f}%")

    print("\n" + "=" * 60)
    print("QUALITY PROFILE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()