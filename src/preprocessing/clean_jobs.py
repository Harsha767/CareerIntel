
from pathlib import Path
import re
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "jobs"
    / "jobs_sample_10000.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jobs"
    / "jobs_cleaned.csv"
)


# ============================================================
# INVALID JOB TITLES
# ============================================================

INVALID_TITLES = {
    "",
    "job",
    "jobs",
    "career",
    "careers",
    "opportunity",
    "opportunities",
    "employment",
    "position",
    "positions",
    "full time",
    "full-time",
    "part time",
    "part-time",
    "contract",
    "temporary",
    "internship",
    "freelance",
    "seasonal",
    "casual",
    "per diem",
    "permanent",
    "temps partiel",
    "temps plein",
}


# ============================================================
# EMPLOYMENT AND SCHEDULE PATTERNS
# ============================================================

EMPLOYMENT_LABELS = (
    r"part[\s-]*time"
    r"|full[\s-]*time"
    r"|temps\s+partiel"
    r"|temps\s+plein"
)

OTHER_EMPLOYMENT_LABELS = (
    r"contract"
    r"|temporary"
    r"|internship"
    r"|freelance"
    r"|seasonal"
    r"|casual"
    r"|per\s+diem"
    r"|permanent"
)

ALL_EMPLOYMENT_LABELS = (
    rf"(?:{EMPLOYMENT_LABELS}|{OTHER_EMPLOYMENT_LABELS})"
)

# Employment labels can appear at the beginning or end,
# including inside parentheses.
EMPLOYMENT_PREFIX_PATTERN = re.compile(
    rf"""
    ^\s*
    (?:
        \( \s* {ALL_EMPLOYMENT_LABELS} \s* \)
        |
        {ALL_EMPLOYMENT_LABELS}
    )
    \s*[-:|–—,]?\s*
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

EMPLOYMENT_QUALIFIER_PATTERN = re.compile(
    rf"""
    \s*
    (?:
        \( \s* {ALL_EMPLOYMENT_LABELS} \s* \)
        |
        \b{ALL_EMPLOYMENT_LABELS}\b
    )
    \s*
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

EMPLOYMENT_SUFFIX_PATTERN = re.compile(
    rf"""
    \s*[-:|–—,]\s*
    (?:{ALL_EMPLOYMENT_LABELS})
    \s*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

# Remove explicit shift descriptions.
# Do NOT remove standalone words such as Night or Evening.
SHIFT_DESCRIPTION_PATTERN = re.compile(
    r"""
    \s*
    (?:
        [-:|–—,]\s*
    )?
    (?:
        (?:day|evening|night|afternoon|morning|overnight)
        \s+shift
        |
        shift\s+(?:day|evening|night)
        |
        (?:jour|soir|nuit)
        \s+(?:de\s+travail|shift)
    )
    \b.*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

# French schedule combinations such as:
# "temps partiel jour"
# "temps partiel soir"
FRENCH_SCHEDULE_PATTERN = re.compile(
    r"""
    \s*
    temps\s+
    (?:partiel|plein)
    (?:\s+(?:jour|soir|nuit|après-midi))?
    \s*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

# Extract schedule details from original titles.
SCHEDULE_EXTRACTION_PATTERN = re.compile(
    rf"""
    (?:
        {EMPLOYMENT_LABELS}
        (?:\s+(?:day|evening|night|afternoon|morning|overnight|
        jour|soir|nuit|après-midi))?
        |
        (?:day|evening|night|afternoon|morning|overnight)
        \s+shift
        |
        shift\s+(?:day|evening|night)
        |
        temps\s+(?:partiel|plein)
        (?:\s+(?:jour|soir|nuit|après-midi))?
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def is_missing(value):
    if value is None:
        return True

    try:
        return bool(pd.isna(value)) or not str(value).strip()
    except (TypeError, ValueError):
        return not str(value).strip()


def normalize_spaces(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def is_invalid_title(value):
    if is_missing(value):
        return True

    title = normalize_spaces(value).lower()
    title = title.strip(" .,:;|-–—()")

    return title in INVALID_TITLES


# ============================================================
# SCHEDULE EXTRACTION
# ============================================================

def extract_schedule(value):
    """
    Extract explicit employment and shift information
    from the original job title.
    """

    if is_missing(value):
        return None

    title = normalize_spaces(value)

    matches = SCHEDULE_EXTRACTION_PATTERN.findall(title)

    if not matches:
        return None

    cleaned_matches = []

    for match in matches:
        item = normalize_spaces(match).strip(" -:|,()")

        if item:
            cleaned_matches.append(item.lower())

    if not cleaned_matches:
        return None

    return " ".join(dict.fromkeys(cleaned_matches))


# ============================================================
# JOB TITLE CLEANING
# ============================================================

def clean_title(value):
    """
    Clean employment and explicit shift wording while
    preserving legitimate job-title words.
    """

    if is_missing(value):
        return None

    title = normalize_spaces(value)

    # Remove explicit shift descriptions.
    title = SHIFT_DESCRIPTION_PATTERN.sub("", title)

    # Remove French schedule suffixes.
    title = FRENCH_SCHEDULE_PATTERN.sub("", title)

    # Remove employment labels in parentheses or as qualifiers.
    title = EMPLOYMENT_QUALIFIER_PATTERN.sub(" ", title)

    # Remove employment prefixes repeatedly.
    previous = None

    while previous != title:
        previous = title
        title = EMPLOYMENT_PREFIX_PATTERN.sub("", title)

    # Remove punctuation-separated employment suffixes.
    previous = None

    while previous != title:
        previous = title
        title = EMPLOYMENT_SUFFIX_PATTERN.sub("", title)

    # Clean remaining punctuation and whitespace.
    title = normalize_spaces(title)
    title = title.strip(" -:|–—,()")

    # Remove empty parentheses.
    title = re.sub(r"\(\s*\)", "", title)

    title = normalize_spaces(title)

    if is_invalid_title(title):
        return None

    return title if title else None


# ============================================================
# TITLE EXTRACTION FROM DESCRIPTION
# ============================================================

def extract_title_from_description(description):
    """
    Extract a title only when the description explicitly
    identifies a position or role.
    """

    if is_missing(description):
        return None

    text = str(description)

    patterns = [
        r"(?:job title|position title|position|role)"
        r"\s*[:\-–—]\s*([^\n.;|]{3,80})",

        r"\bAs an?\s+"
        r"([A-Z][A-Za-z0-9 .&()/+-]{2,65}?)"
        r"(?=[,.;\n])",

        r"\b(?:hiring|seeking|looking for)\s+"
        r"(?:an?\s+)?"
        r"([A-Z][A-Za-z0-9 .&()/+-]{2,65}?)"
        r"(?=[,.;\n])",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            candidate = clean_title(match.group(1))

            if candidate:
                return candidate

    return None


# ============================================================
# WORK MODEL NORMALIZATION
# ============================================================

def normalize_work_model(value):
    if is_missing(value):
        return pd.NA

    value = str(value).strip()
    key = re.sub(r"[-_\s]", "", value.lower())

    mapping = {
        "onsite": "On-site",
        "hybrid": "Hybrid",
        "remote": "Remote",
    }

    return mapping.get(key, value)


# ============================================================
# EMPLOYMENT TYPE NORMALIZATION
# ============================================================

def normalize_employment_type(value):
    if is_missing(value):
        return pd.NA

    value = str(value).strip()
    key = re.sub(r"[-_\s]", "", value.lower())

    mapping = {
        "fulltime": "Full-time",
        "parttime": "Part-time",
        "contract": "Contract",
        "temporary": "Temporary",
        "internship": "Internship",
        "freelance": "Freelance",
        "seasonal": "Seasonal",
        "perdiem": "Per diem",
        "permanent": "Permanent",
    }

    return mapping.get(key, value)


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def clean_jobs() -> None:

    print("Loading CareerIntel job dataset...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"Original shape: {df.shape}")

    # --------------------------------------------------------
    # REMOVE EXACT DUPLICATES
    # --------------------------------------------------------

    original_count = len(df)

    df = df.drop_duplicates().copy()

    duplicates_removed = original_count - len(df)

    print(
        f"Exact duplicate rows removed: {duplicates_removed}"
    )

    # --------------------------------------------------------
    # ENSURE REQUIRED COLUMNS EXIST
    # --------------------------------------------------------

    for column in [
        "title",
        "normalized_title",
        "description",
        "work_model",
        "employment_type",
    ]:
        if column not in df.columns:
            df[column] = pd.NA

    # --------------------------------------------------------
    # EXTRACT SCHEDULE BEFORE CLEANING TITLE
    # --------------------------------------------------------

    df["shift_schedule"] = df["title"].apply(
        extract_schedule
    )

    # --------------------------------------------------------
    # CLEAN JOB TITLES
    # --------------------------------------------------------

    original_titles = df["title"].copy()

    df["title"] = df["title"].apply(clean_title)

    # Use normalized_title as fallback.
    missing_title_mask = df["title"].isna()

    df.loc[missing_title_mask, "title"] = (
        df.loc[missing_title_mask, "normalized_title"]
        .apply(clean_title)
    )

    # Use description only as a final fallback.
    missing_title_mask = df["title"].isna()

    df.loc[missing_title_mask, "title"] = (
        df.loc[missing_title_mask, "description"]
        .apply(extract_title_from_description)
    )

    # --------------------------------------------------------
    # COUNT CHANGED TITLES
    # --------------------------------------------------------

    changed_mask = (
        original_titles.fillna("").astype(str).str.strip()
        != df["title"].fillna("").astype(str).str.strip()
    )

    titles_changed = int(changed_mask.sum())

    print(
        f"Titles changed or repaired: {titles_changed}"
    )

    # --------------------------------------------------------
    # REMOVE INVALID TITLES
    # --------------------------------------------------------

    invalid_mask = df["title"].apply(is_invalid_title)

    invalid_count = int(invalid_mask.sum())

    df = df.loc[~invalid_mask].copy()

    print(
        f"Jobs removed due to invalid titles: {invalid_count}"
    )

    print(
        f"Jobs remaining after title validation: {len(df)}"
    )

    # --------------------------------------------------------
    # NORMALIZE WORK MODEL
    # --------------------------------------------------------

    df["work_model"] = df["work_model"].apply(
        normalize_work_model
    )

    # --------------------------------------------------------
    # NORMALIZE EMPLOYMENT TYPE
    # --------------------------------------------------------

    df["employment_type"] = df["employment_type"].apply(
        normalize_employment_type
    )

    # --------------------------------------------------------
    # NUMERIC COLUMN CONVERSION
    # --------------------------------------------------------

    numeric_columns = [
        "salary",
        "salary_min",
        "salary_max",
        "min_salary",
        "max_salary",
        "company_size",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # BUILD JOB TEXT
    # --------------------------------------------------------

    text_columns = [
        "title",
        "description",
        "company_name",
        "industry",
        "function",
        "experience_level",
        "work_model",
        "employment_type",
    ]

    available_text_columns = [
        column
        for column in text_columns
        if column in df.columns
    ]

    df["job_text"] = (
        df[available_text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    df["job_text"] = (
        df["job_text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # --------------------------------------------------------
    # REMOVE JOBS WITH INSUFFICIENT TEXT
    # --------------------------------------------------------

    before_text_filter = len(df)

    df = df[
        df["job_text"].str.len() > 20
    ].copy()

    removed_for_text = before_text_filter - len(df)

    print(
        "Jobs removed with insufficient text: "
        f"{removed_for_text}"
    )

    # --------------------------------------------------------
    # SAVE CLEANED DATASET
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CareerIntel preprocessing complete")
    print("=" * 60)

    print(f"Final shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Output: {OUTPUT_PATH}")

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    clean_jobs()
