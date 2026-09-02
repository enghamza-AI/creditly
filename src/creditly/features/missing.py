
import pandas as pd


MISSINGNESS_DIAGNOSIS = {
    # column_name: (mechanism, one-line reasoning)
    "emp_length": ("MAR", "missing more often for recently issued loans"),
    "dti": ("MCAR", "small, scattered null rate with no obvious pattern"),
    "mths_since_last_delinq": ("MNAR", "likely missing because borrower never delinquent — meaningful, not random"),
}


def summarize_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    
    null_counts = df.isna().sum()
    rows = []
    for col, count in null_counts[null_counts > 0].items():
        mechanism, reason = MISSINGNESS_DIAGNOSIS.get(col, ("UNDIAGNOSED", "not yet reasoned about"))
        rows.append({"column": col, "null_count": count, "mechanism": mechanism, "reasoning": reason})
    return pd.DataFrame(rows).sort_values("null_count", ascending=False)


def drop_mcar_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
   
    return df.dropna(subset=columns).reset_index(drop=True)


def impute_mar_by_group(df: pd.DataFrame, target_col: str, group_col: str) -> pd.DataFrame:
   
    df = df.copy()
    group_medians = df.groupby(group_col)[target_col].transform("median")
    df[target_col] = df[target_col].fillna(group_medians)


    df[target_col] = df[target_col].fillna(df[target_col].median())
    return df


def flag_and_fill_mnar(df: pd.DataFrame, column: str, fill_value=0) -> pd.DataFrame:
   
    df = df.copy()
    df[f"{column}_was_missing"] = df[column].isna()
    df[column] = df[column].fillna(fill_value)
    return df
