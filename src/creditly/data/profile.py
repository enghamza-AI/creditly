# profiling
import pandas as pd


def profile_dataframe(df: pd.DataFrame, name: str = "dataframe") -> pd.DataFrame:
    print(f"---profiling: {name} ---")
    print(f"shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
    df.info()
    null_report = (
        df.isna()
        .sum()
        .to_frame("null_count")
        .assign(null_pct=lambda x: (x["null_count"] / len(df) * 100).round(2))
        .sort_values("null_pct", ascending=False)
    )

    print("\nnull report (worst first):")
    print(null_report[null_report["null_count"] > 0])

    return null_report

def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    return numeric_df.describe().T

