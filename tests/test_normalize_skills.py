from src.nlp.normalize_skills import normalize_skill


def test_normalize_skill_preserves_meaningful_punctuation():
    assert normalize_skill("CRM (Salesforce)") == "crm (salesforce)"
    assert normalize_skill("EMR (EPIC)") == "emr (epic)"
    assert normalize_skill("Quality standards (AS9100)") == (
        "quality standards (as9100)"
    )
    assert normalize_skill(".NET") == ".net"
    assert normalize_skill(".NET C#") == ".net c#"
    assert normalize_skill("C#") == "c#"
    assert normalize_skill("CI/CD") == "ci/cd"


def test_normalize_skill_preserves_parentheses():
    assert normalize_skill("CRM (Salesforce)") == "crm (salesforce)"


def test_normalize_skill_preserves_leading_dot():
    assert normalize_skill(".NET") == ".net"
    assert normalize_skill(".NET C#") == ".net c#"