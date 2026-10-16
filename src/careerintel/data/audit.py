"""Data audit: row counts, columns, dtypes, missing values."""
import sys
import pandas as pd


def audit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column summary of dtype, missing %, and unique count."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(dropna=True),
        }
    ).sort_values("missing_pct", ascending=False)


def audit_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"{path}: {len(df):,} rows x {df.shape[1]} columns")
    summary = audit_frame(df)
    print(summary.to_string())
    return summary


if __name__ == "__main__":
    for p in sys.argv[1:]:
        audit_csv(p)
