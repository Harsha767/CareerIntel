"""CareerIntel: browser-based human relevance review."""

import ast
from pathlib import Path

import pandas as pd
import streamlit as st

# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "evaluation_candidates.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "evaluation_candidates_reviewed.csv"
)


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CareerIntel | Relevance Review",
    page_icon="🎯",
    layout="wide",
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_text(value):
    """Convert missing values and list-like strings into readable text."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return ""

    # Convert strings representing Python lists into readable text.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, (list, tuple, set)):
                return ", ".join(
                    str(item).strip()
                    for item in parsed
                    if str(item).strip()
                )

        except (ValueError, SyntaxError):
            pass

    return text


@st.cache_data
def load_candidates():
    """Load the original evaluation dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"job_id": str},
        low_memory=False,
    )

    required_columns = {
        "user_id",
        "profile_text",
        "job_id",
        "job_title",
        "relevant",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df


def load_existing_reviews():
    """Load saved reviews or initialize from the original dataset."""

    if OUTPUT_FILE.exists():
        df = pd.read_csv(
            OUTPUT_FILE,
            dtype={"job_id": str},
            low_memory=False,
        )
    else:
        df = load_candidates().copy()

    # Ensure required columns exist.
    if "relevant" not in df.columns:
        df["relevant"] = pd.NA

    if "review_notes" not in df.columns:
        df["review_notes"] = ""

    # Normalize relevance labels.
    labels = (
        df["relevant"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    labels = labels.where(
        labels.isin(["", "0", "1"]),
        "",
    )

    df["relevant"] = labels.astype(object)

    # Keep notes as text to prevent float64 assignment errors.
    df["review_notes"] = (
        df["review_notes"]
        .fillna("")
        .astype(str)
        .astype(object)
    )

    return df


def save_review(df, row_index, selected_label, notes):
    """Save one human relevance judgment safely."""

    label_to_value = {
        "Unreviewed": pd.NA,
        "1 — Relevant": 1,
        "0 — Not relevant": 0,
    }

    # Use nullable integer dtype for relevance labels.
    df["relevant"] = pd.to_numeric(
        df["relevant"],
        errors="coerce",
    ).astype("Int64")

    # Keep review notes in a text-compatible column.
    df["review_notes"] = (
        df["review_notes"]
        .fillna("")
        .astype(str)
        .astype(object)
    )

    df.at[row_index, "relevant"] = label_to_value[selected_label]
    df.at[row_index, "review_notes"] = str(notes).strip()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


def make_job_label(df, row_index, position):
    """Create a readable label for the job selector."""

    row = df.loc[row_index]

    title = clean_text(
        row.get("job_title", "Untitled job")
    )

    company = clean_text(
        row.get("company_name", "")
    )

    if company:
        title = f"{title} — {company}"

    relevance = clean_text(
        row.get("relevant", "")
    )

    if relevance in {"0", "1"}:
        status = "✓ Reviewed"
    else:
        status = "○ Not reviewed"

    return f"{position + 1}. {title} [{status}]"


def navigate_job(profile, position):
    """
    Update both the stored position and the selectbox state.

    Updating both is essential because Streamlit preserves widget
    state across reruns.
    """

    selector_key = f"job_selector_{profile}"

    # Clamp the position to valid bounds.
    profile_rows = st.session_state.get(
        f"profile_rows_{profile}",
        [],
    )

    if profile_rows:
        position = max(
            0,
            min(position, len(profile_rows) - 1),
        )

    st.session_state["job_positions"][profile] = position

    # Update the actual selectbox value.
    st.session_state[selector_key] = position


# =========================================================
# APPLICATION HEADER
# =========================================================

st.title("🎯 CareerIntel")
st.subheader("Human Relevance Review")

st.write(
    "Review how relevant each job is to a candidate profile. "
    "Your judgments will be saved locally to a CSV file."
)


# =========================================================
# LOAD DATA
# =========================================================

try:
    df = load_existing_reviews()

except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
    st.error(f"Unable to load evaluation data: {exc}")
    st.stop()

# Normalize relevance labels for display.
df["relevant"] = (
    df["relevant"]
    .astype("string")
    .fillna("")
    .astype(object)
)

df["review_notes"] = (
    df["review_notes"]
    .fillna("")
    .astype(str)
    .astype(object)
)


# =========================================================
# PROGRESS METRICS
# =========================================================

reviewed_mask = df["relevant"].isin(["0", "1"])

reviewed_count = int(reviewed_mask.sum())
total_count = len(df)
remaining_count = total_count - reviewed_count

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Total pairs",
    total_count,
)

metric2.metric(
    "Reviewed",
    reviewed_count,
)

metric3.metric(
    "Remaining",
    remaining_count,
)

st.progress(
    reviewed_count / total_count if total_count else 0
)

st.divider()


# =========================================================
# CANDIDATE PROFILE SELECTION
# =========================================================

profiles = sorted(
    df["user_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

if not profiles:
    st.error("No candidate profiles are available.")
    st.stop()


selected_profile = st.selectbox(
    "Choose candidate profile",
    options=profiles,
    key="selected_profile",
)


# Find all jobs belonging to this candidate.
profile_rows = df.index[
    df["user_id"].astype(str) == selected_profile
].tolist()

if not profile_rows:
    st.warning("No jobs found for this profile.")
    st.stop()


# Store the profile's row indexes for navigation.
st.session_state[
    f"profile_rows_{selected_profile}"
] = profile_rows


# =========================================================
# CANDIDATE PROFILE DETAILS
# =========================================================

profile_text = clean_text(
    df.loc[profile_rows[0], "profile_text"]
)

st.markdown("### Candidate profile")

st.info(
    profile_text
    if profile_text
    else "No profile description available."
)


# =========================================================
# JOB SELECTOR
# =========================================================

# Initialize navigation state for each profile.
if "job_positions" not in st.session_state:
    st.session_state["job_positions"] = {}


if selected_profile not in st.session_state["job_positions"]:
    st.session_state["job_positions"][selected_profile] = 0


selector_key = f"job_selector_{selected_profile}"

current_position = st.session_state["job_positions"][selected_profile]

# Keep the position inside the valid range.
current_position = max(
    0,
    min(current_position, len(profile_rows) - 1),
)

st.session_state["job_positions"][selected_profile] = current_position


# Initialize the selector's state if necessary.
if selector_key not in st.session_state:
    st.session_state[selector_key] = current_position


# Correct any stale selector value.
if st.session_state[selector_key] >= len(profile_rows):
    st.session_state[selector_key] = len(profile_rows) - 1


def sync_job_selector():
    """Keep navigation state synchronized with the dropdown."""

    st.session_state["job_positions"][selected_profile] = (
        st.session_state[selector_key]
    )


selected_position = st.selectbox(
    "Choose a job to review",
    options=list(range(len(profile_rows))),
    format_func=lambda position: make_job_label(
        df,
        profile_rows[position],
        position,
    ),
    key=selector_key,
    on_change=sync_job_selector,
)

row_index = profile_rows[selected_position]
row = df.loc[row_index]


# Keep navigation state synchronized after selection.
st.session_state["job_positions"][selected_profile] = (
    selected_position
)


# Get the selected row.
row_index = profile_rows[selected_position]
row = df.loc[row_index]


# =========================================================
# JOB DETAILS
# =========================================================

st.markdown("### Job details")

job_title = clean_text(
    row.get("job_title", "")
)

st.write(
    f"**Job title:** {job_title or 'Not available'}"
)


company = clean_text(
    row.get("company_name", "")
)

if company:
    st.write(
        f"**Company:** {company}"
    )


st.write("**Required skills:**")

job_skills = clean_text(
    row.get("job_skills", "")
)

st.write(
    job_skills
    if job_skills
    else "No required skills listed."
)


st.write("**Job description:**")

job_text = clean_text(
    row.get("job_text", "")
)

if job_text:
    st.write(job_text)
else:
    st.warning(
        "No job description is available for this record."
    )


st.divider()


# =========================================================
# HUMAN RELEVANCE JUDGMENT
# =========================================================

st.markdown("### Your relevance judgment")

st.caption(
    "Choose 1 if the job is relevant to the candidate, "
    "0 if it is not relevant, or Unreviewed if you are uncertain."
)


label_options = [
    "Unreviewed",
    "1 — Relevant",
    "0 — Not relevant",
]

value_to_label = {
    "": "Unreviewed",
    "0": "0 — Not relevant",
    "1": "1 — Relevant",
}

current_label = clean_text(
    row.get("relevant", "")
)

default_label = value_to_label.get(
    current_label,
    "Unreviewed",
)


selected_label = st.radio(
    "Relevance",
    options=label_options,
    index=label_options.index(default_label),
    key=f"relevance_{selected_profile}_{row_index}",
)


existing_note = clean_text(
    row.get("review_notes", "")
)

notes = st.text_area(
    "Review notes (optional)",
    value=existing_note,
    placeholder="Briefly explain your judgment...",
    key=f"notes_{selected_profile}_{row_index}",
)


# =========================================================
# SAVE REVIEW
# =========================================================

if st.button(
    "Save review",
    type="primary",
    key=f"save_{selected_profile}_{row_index}",
):

    try:
        save_review(
            df=df,
            row_index=row_index,
            selected_label=selected_label,
            notes=notes,
        )

        # Clear cached data so the next rerun reflects saved labels.
        st.cache_data.clear()

        st.success(
            "Review saved successfully!"
        )

        st.rerun()

    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(
           f"Unable to save review: {exc}"
    )


# =========================================================
# PREVIOUS / NEXT NAVIGATION
# =========================================================

st.divider()

col1, col2, col3 = st.columns(3)


with col1:

    st.button(
        "⬅ Previous job",
        disabled=(selected_position == 0),
        use_container_width=True,
        key=f"previous_{selected_profile}",
        on_click=navigate_job,
        args=(
            selected_profile,
            selected_position - 1,
        ),
    )


with col2:

    st.markdown(
        f"<div style='text-align:center; padding-top:10px;'>"
        f"Job <b>{selected_position + 1}</b> "
        f"of <b>{len(profile_rows)}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )


with col3:

    st.button(
        "Next job ➡",
        disabled=(
            selected_position >= len(profile_rows) - 1
        ),
        use_container_width=True,
        key=f"next_{selected_profile}",
        on_click=navigate_job,
        args=(
            selected_profile,
            selected_position + 1,
        ),
    )


# =========================================================
# OUTPUT LOCATIONS
# =========================================================

st.divider()

st.caption(
    f"Original dataset: {INPUT_FILE}"
)

st.caption(
    f"Reviewed dataset: {OUTPUT_FILE}"
)