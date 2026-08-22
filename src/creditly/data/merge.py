# Merging both datasets by aggregate_census_to_zip3
import pandas as pd
 
def extract_zip3(zip_series: pd.Series) -> pd.Series:
    return (
        zip_series
        .astype(str)
        .str.strip()

        .str[:3]
    )
def aggregate_census_to_zip3(census_df: pd.DataFrame, zcta_col: str, income_col: str) -> pd.DataFrame:
    df = census_df.copy()
    df["zip3"] = extract_zip3(df[zcta_col])

    zip3_income = (
        df.groupby("zip3")[income_col]
        .mean()
        .reset_index()
        .rename(columns={income_col: "median_income_zip3"})
    )
    return zip3_income

def merge_loans_with_income(loans_df: pd.DataFrame, zip3_income_df: pd.DataFrame) -> pd.DataFrame:
    loans_df = loans_df.copy()
    loans_df["zip3"] = extract_zip3(loans_df["zip_code"])

    merged = loans_df.merge(
        zip3_income_df,
        on="zip3",
        how="left",
    )
    return merged
