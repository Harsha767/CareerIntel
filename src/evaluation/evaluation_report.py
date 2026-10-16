"""CareerIntel: descriptive report for provisional relevance suggestions.

IMPORTANT:
These statistics describe heuristic suggestions.
They do not measure recommendation accuracy or model performance.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "evaluation_candidates_ai_suggested.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "evaluation"
    / "reports"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Suggestions file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required = {
        "user_id",
        "job_id",
        "suggested_relevant",
        "suggestion_confidence",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    labels = pd.to_numeric(
        df["suggested_relevant"],
        errors="coerce",
    )

    if not labels.isin([-1, 0, 1]).all():
        raise ValueError(
            "Unexpected labels found. Expected -1, 0, or 1."
        )

    df["suggested_relevant"] = labels.astype(int)

    label_names = {
        -1: "Needs review",
        0: "Likely irrelevant",
        1: "Likely relevant",
    }

    counts = (
        df["suggested_relevant"]
        .map(label_names)
        .value_counts()
        .reindex(
            [
                "Likely relevant",
                "Likely irrelevant",
                "Needs review",
            ],
            fill_value=0,
        )
    )

    summary = pd.DataFrame(
        {
            "suggestion_category": counts.index,
            "count": counts.values,
            "percentage": (
                counts.values / len(df) * 100
                if len(df)
                else 0
            ),
        }
    )

    summary.to_csv(
        OUTPUT_DIR / "label_distribution.csv",
        index=False,
    )

    candidate_summary = (
        df.groupby("user_id")
        .agg(
            total_pairs=("job_id", "size"),
            likely_relevant=(
                "suggested_relevant",
                lambda values: (values == 1).sum(),
            ),
            likely_irrelevant=(
                "suggested_relevant",
                lambda values: (values == 0).sum(),
            ),
            needs_review=(
                "suggested_relevant",
                lambda values: (values == -1).sum(),
            ),
        )
        .reset_index()
    )

    candidate_summary.to_csv(
        OUTPUT_DIR / "candidate_summary.csv",
        index=False,
    )

    confidence_summary = (
        df["suggestion_confidence"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("confidence")
        .reset_index(name="count")
    )

    confidence_summary.to_csv(
        OUTPUT_DIR / "confidence_distribution.csv",
        index=False,
    )

    # Chart 1: Overall label distribution.
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        summary["suggestion_category"],
        summary["count"],
    )

    ax.set_title(
        "CareerIntel: Provisional Relevance Suggestions"
    )
    ax.set_xlabel("Suggestion category")
    ax.set_ylabel("Number of candidate–job pairs")

    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "label_distribution.png",
        dpi=200,
    )

    plt.close(fig)

    # Chart 2: Suggestions by candidate.
    chart_data = candidate_summary.set_index("user_id")[
        [
            "likely_relevant",
            "likely_irrelevant",
            "needs_review",
        ]
    ]

    ax = chart_data.plot(
        kind="bar",
        figsize=(9, 5),
    )

    ax.set_title(
        "Provisional Suggestions by Candidate"
    )
    ax.set_xlabel("Candidate")
    ax.set_ylabel("Number of job pairs")
    ax.legend(
        [
            "Likely relevant",
            "Likely irrelevant",
            "Needs review",
        ]
    )

    fig = ax.get_figure()
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "candidate_suggestions.png",
        dpi=200,
    )

    plt.close(fig)

    report = f"""# CareerIntel Evaluation Summary

## Purpose

This report describes provisional relevance suggestions generated
using heuristic rules based on skill overlap and role-family evidence.

These suggestions are not independently verified ground-truth labels.

## Dataset

- Candidate–job pairs: {len(df)}
- Candidates: {df["user_id"].nunique()}
- Unique job IDs: {df["job_id"].nunique()}

## Suggestion distribution

{summary.to_string(index=False)}

## Candidate summary

{candidate_summary.to_string(index=False)}

## Interpretation

The distribution describes the behavior of the heuristic labeling
procedure on this dataset. It does not establish recommendation
accuracy, precision, recall, F1-score, or generalization performance.

Reliable model evaluation requires an independently validated
relevance-label dataset and a defined evaluation protocol.
"""

    (OUTPUT_DIR / "evaluation_summary.md").write_text(
        report,
        encoding="utf-8",
    )

    print("=" * 55)
    print("CAREERINTEL | EVALUATION SUMMARY")
    print("=" * 55)

    print(f"\nTotal candidate–job pairs: {len(df)}")
    print("\nSuggestion distribution:")
    print(summary.to_string(index=False))

    print(f"\nReports saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()