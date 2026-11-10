import streamlit as st
import pandas as pd

from src.recommendation.recommend_jobs import recommend_jobs
from src.recommendation.skill_gap_analyzer import analyze_skill_gap

st.set_page_config(
    page_title="CareerIntel",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 CareerIntel")
st.subheader("AI-Enhanced Personalized Job Recommendation Platform")
st.write(
    "Discover relevant job opportunities and identify "
    "the skills you need to develop."
)

st.sidebar.header("Candidate Profile")

profile = st.sidebar.text_area(
    "Career Profile",
    placeholder="Describe your education, experience, interests and career goals...",
    height=150
)

skills_input = st.sidebar.text_area(
    "Your Skills",
    placeholder="Python, SQL, Machine Learning, Excel"
)

top_k = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

generate = st.sidebar.button(
    "Find Recommended Jobs",
    type="primary",
    use_container_width=True
)

skills = [
    s.strip()
    for s in skills_input.replace("\n", ",").split(",")
    if s.strip()
]

if generate:
    if not profile.strip():
        st.error("Please enter your career profile.")
    else:
        try:
            with st.spinner("Analyzing your profile and matching jobs..."):
                results = recommend_jobs(
                    profile=profile,
                    preferred_skills=skills,
                    top_k=top_k
                )

            st.session_state["results"] = results
            st.session_state["skills"] = skills

        except Exception as e:
            st.error(f"Recommendation error: {e}")

if "results" in st.session_state:

    results = st.session_state["results"]

    st.divider()
    st.header("Recommended Jobs")

    if results.empty:
        st.warning("No matching jobs found.")
    else:
        col1, col2, col3 = st.columns(3)

        col1.metric("Jobs Found", len(results))

        col2.metric(
            "Average Match",
            f"{results['recommendation_score'].mean() * 100:.1f}%"
        )

        col3.metric(
            "Your Skills",
            len(st.session_state["skills"])
        )

        for i, (_, job) in enumerate(results.iterrows()):

            title = str(job.get("title", "Job"))
            company = str(job.get("company_name", "Unknown Company"))

            with st.expander(
                f"{i + 1}. {title} — {company}",
                expanded=(i == 0)
            ):

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Recommendation Score",
                    f"{job['recommendation_score'] * 100:.1f}%"
                )

                c2.metric(
                    "Content Similarity",
                    f"{job['content_score'] * 100:.1f}%"
                )

                c3.metric(
                    "Skill Coverage",
                    f"{job['skill_coverage'] * 100:.1f}%"
                )

                st.write("**Industry:**", job.get("industry", "N/A"))
                st.write("**Function:**", job.get("function", "N/A"))
                st.write("**Experience:**", job.get("experience_level", "N/A"))
                st.write("**Work Model:**", job.get("work_model", "N/A"))

                st.write(
                    "**Matching Skills:**",
                    job.get("matching_skills") or "None identified"
                )

                st.write(
                    "**Job Skills:**",
                    job.get("normalized_skills", "N/A")
                )

        st.divider()
        st.header("Skill Gap Analysis")

        job_options = [
            f"{row['title']} — {row['company_name']}"
            for _, row in results.iterrows()
        ]

        selected = st.selectbox(
            "Choose a job to analyze",
            range(len(job_options)),
            format_func=lambda x: job_options[x]
        )

        if st.button("Analyze Skill Gap", type="primary"):

            job = results.iloc[selected]

            try:
                with st.spinner("Analyzing skill requirements..."):
                    gap = analyze_skill_gap(
                        candidate_skills=st.session_state["skills"],
                        job_title=str(job["title"]),
                        company_name=str(job["company_name"])
                    )

                left, right = st.columns(2)

                left.metric(
                    "Skill Coverage",
                    f"{gap['skill_coverage'] * 100:.1f}%"
                )

                right.metric(
                    "Skill Gap",
                    f"{gap['skill_gap_percentage']:.1f}%"
                )

                st.progress(gap["skill_coverage"])

                st.subheader("Skills You Already Have")
                st.write(
                    ", ".join(gap["possessed_skills"]) or "None identified"
                )

                st.subheader("Skills You Need to Develop")
                st.write(
                    ", ".join(gap["missing_skills"]) or "None identified"
                )

                st.caption(
                    f"Matched {gap['matched_skill_count']} of "
                    f"{gap['required_skill_count']} identified required skills."
                )

            except Exception as e:
                st.error(f"Skill-gap analysis unavailable: {e}")

st.divider()

st.caption(
    "CareerIntel | Personalized Job Recommendations and Skill Gap Analysis"
)
