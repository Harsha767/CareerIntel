from src.recommendation.skill_gap_analyzer import (
    normalize_skill,
    parse_skills,
)


def test_normalize_skill_aliases():
    assert normalize_skill("Py") == "python"
    assert normalize_skill("ML") == "machine learning"
    assert normalize_skill("SQL") == "sql"


def test_normalize_skill_handles_empty_values():
    assert normalize_skill(None) == ""
    assert normalize_skill("   ") == ""


def test_parse_skills_from_list_string():
    skills = parse_skills(
        "['Python', 'SQL', 'Machine Learning']"
    )

    assert skills == {
        "python",
        "sql",
        "machine learning",
    }


def test_parse_skills_from_delimited_string():
    skills = parse_skills("Python, SQL; Pandas")

    assert skills == {"python", "sql", "pandas"}


def test_parse_skills_handles_empty_values():
    assert parse_skills(None) == set()
    assert parse_skills("") == set()